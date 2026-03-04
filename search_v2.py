from agent import OpenAIAgent
import pdb
import os 
from get_parser import get_parser
import numpy as np
import pandas as pd
import random
from utils import write_to_csv_file, bootstrap_confidence_interval
import re 
from sys_prompt import build_prompt_search, Strategy_proposer
from chat import safe_execution_once
import json
import copy 
import openai
from utils import read_data, store_data, extract_content
from cli2 import handle_input_file_format, handle_target_and_output_file
from chat import format_user_query, evaluating_output
# from search import seed_everything, evaluate, generate_and_evaluate_without_error

global_dict, local_dict = globals(), locals()

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)

def evaluate(args, code_snippet, is_testing=False, verbose=False):
	output = code_snippet
	error_list = []
	if is_testing:
		default_dir = './benchmark_full/' 
		start, end =1, 4
	else:
		default_dir = './benchmark_full/'
		start, end = 4, 20
	for i in range(start, end):
		args.index = str(i)
		handle_input_file_format(default_dir, args)
		handle_target_and_output_file(args)
		try:
			input_data, sampling_rate = read_data(args.input_file)
		except:
			continue

		# global_dict, local_dict = globals(), locals()
		# local_dict = locals()
		local_dict.update(
			{'sensor_readings': input_data, 'sampling_rate':sampling_rate}
		)
		# global_dict = local_dict = {"__builtins__": __builtins__}
		code_output, returned_code = safe_execution_once(args, output, global_dict, local_dict, verbose=verbose)

		if "An error occurred" in code_output:
			print(code_output)
			return [np.nan]*20, returned_code, code_output
		# evluation
		try:
			distance, metric = evaluating_output(args, write_result=False, return_mse=True, verbose=args.verbose)
		except:
			distance, metric = np.nan, np.nan
		
		error_list.append(metric)
	
	return error_list, returned_code, code_output

def generate_and_evaluate_without_error(args, Agent):
	try:
		output = Agent.step()
	except Exception as error:
		output = ""
		
	n = 10
	for i in range(n):
		error_list, returned_code, code_output = evaluate(args, output, verbose=args.verbose)

		if "An error occurred:" in code_output:
			message = f"Your previous attempt contains the following error, please try again: {code_output}"
			Agent.update(message, role='user')
			try:
				output = Agent.step()
			except Exception as error:
				output = ""
		else:
			break 
	return error_list, returned_code, code_output, output

def main(args):
	pool_limit = 5
	output_json = f"results/{args.openai}/search_v2-{args.query}-{args.index}.json"
	print(f'Saving results to {output_json}')
	seed_everything(seed=0)
	if 'gpt' in args.openai or 'o1' in args.openai or 'o3' in args.openai or 'o4' in args.openai:
		openai_key = open("key.txt").read().strip()
	else:
		openai_key = open("together_key.txt").read().strip()
	os.environ["OPENAI_API_KEY"] = openai_key
	args.index = str(1)
	if args.full_benchmark:
		dataset_dir = './benchmark_full/'
	else:
		dataset_dir = './benchmark/'
	
	handle_input_file_format(dataset_dir, args)
	handle_target_and_output_file(args)
	
	input_data, sampling_rate = read_data(args.input_file)
	query_str = format_user_query(args)
	
	# Propose args.num_island diverse strategies
	StrategyAgent = OpenAIAgent(args, args.openai, system_prompt=
						Strategy_proposer.format(query=query_str, n=args.num_islands),
						temperature=1, top_p=1)
	strategies = StrategyAgent.step()
	try:
		strategies = json.loads(strategies)
	except json.JSONDecodeError as e:
		print("Invalid JSON:", e)
		strategies = [""]

	mean_hist = []
	code_error_pair = []
	for _i, strategy_i in enumerate(strategies):
		print(f"{_i} | Now implementing the solver function following the strategy: {strategy_i}...")
		system_prompt = build_prompt_search(args, is_initial=True, example=query_str, sampling_rate=sampling_rate)
		Agent = OpenAIAgent(args, args.openai, system_prompt=
							system_prompt, temperature=1, top_p=1)
		Agent.update(
			"Now please start implementing the solver function for my query: QUERY[{}]. " \
			"[IMPORTANT] Please follow the strategy for implementation: {}".format(query_str, strategy_i),
			role="user"
		)

		error_list, returned_code, code_output, raw_output = generate_and_evaluate_without_error(args, Agent)
		
		code_idea = extract_content(raw_output)

		error_str, mean, metric = bootstrap_confidence_interval(args, error_list)
		print(f">>>>> Initial result: {error_str} \n")

		eval_error_list, _, _ = evaluate(args, "```Python\n"+returned_code+"\n```", is_testing=True, verbose=args.verbose)
		eval_error_str, mean_eval, metric = bootstrap_confidence_interval(args, eval_error_list)
		print(f">>>>> Initial eval result: {eval_error_str} \n")
		mean_hist.append(mean)
		# best_mean = mean_eval

		if metric == 'MSE':
			if mean <= min(mean_hist):
				best_mean = mean_eval 
		else:
			if mean >= max(mean_hist):
				best_mean = mean_eval 

		code_error_pair_i = [
			{'code':returned_code, 'error':error_str, 'val':mean, 'iteration': 'initial', 'eval_error': eval_error_str, 'best': best_mean, 'idea': code_idea, 'metric': metric}
		]
		code_error_pair.extend(code_error_pair_i)
	

	AgentDesigner = Agent
	AgentDesigner.clear()
	system_prompt = build_prompt_search(args, is_initial=False, example=query_str, sampling_rate=sampling_rate)
	
	AgentDesigner.update(system_prompt, role='system')
	for i in range(args.num_trial):
		chat_checkpoint = copy.deepcopy(AgentDesigner.chat)
		code_error_pair_to_llm = []
		# only present val error to llm
		# show only the top pool_limit pair to llm
		
		#TODO: Here I use the greedy approach. But can be randomized.
		if code_error_pair[0]['metric'] == 'MSE':
			is_reverse = False 
			default_val = 1e6
		else:
			is_reverse = True
			default_val = 0

		top_pairs = sorted(
			code_error_pair,
			key=lambda x: x.get("val", float(default_val)),
			reverse=is_reverse
		)[:pool_limit]

		for pair in top_pairs:
			code_error_pair_to_llm.append(
				{'code': pair['code'], 'val_str': pair['error'], 'val':pair['val'], 'iteration': pair['iteration'], 'idea': pair['idea'], 'metric': pair['metric']}
			)
		instruction = f"""
Problem:
{query_str}
Based on the following code and error pair, proposed a new solution that can improve the performance:
{code_error_pair_to_llm}
"""
		# pdb.set_trace()
		AgentDesigner.update(instruction, role='user')
		# try:
		# 	output_i = AgentDesigner.step()
		# except Exception as error:
		# 	output_i = ""
		# error_list_i, returned_code_i, code_output = evaluate(args, output_i)
		error_list_i, returned_code_i, code_output, raw_output = generate_and_evaluate_without_error(args, AgentDesigner)

		code_idea = extract_content(raw_output)

		error_str_i, mean_i, metric = bootstrap_confidence_interval(args, error_list_i)

		eval_error_list_i, _, _ = evaluate(args, "```Python\n"+returned_code_i+"\n```", is_testing=True, verbose=args.verbose)
		eval_error_str_i, mean_eval_i, metric = bootstrap_confidence_interval(args, eval_error_list_i)

		mean_hist.append(mean_i)

		if metric == 'MSE':
			if mean_i <= min(mean_hist):
				best_mean = mean_eval_i 
		else:
			if mean_i >= max(mean_hist):
				best_mean = mean_eval_i 

		code_error_pair.append({'code':returned_code_i, 'error':error_str_i, 'val':mean_i, 'iteration': i+1, 'eval_error': eval_error_str_i, 'best': best_mean, 'idea': code_idea, 'metric': metric})
		AgentDesigner.chat = chat_checkpoint
		print(f"\n>>>>> Iteration {i+1} produce result: {error_str_i} \n\n")
		print(f"\n>>>>> Iteration {i+1} eval result: {eval_error_str_i} best: {best_mean} \n\n")
		# pdb.set_trace()

	# Save to JSON file
	if not os.path.exists(f"results/{args.openai}"):
		os.makedirs(f"results/{args.openai}", exist_ok=True)
	
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(code_error_pair, f, ensure_ascii=False, indent=4)

	

if __name__ == '__main__':
	args = get_parser()
	main(args)