import pdb 
import openai 
import re 
import csv
import subprocess
from tqdm import tqdm
import os
import sys
import io
import numpy as np
from time import sleep
from datetime import datetime
import pandas as pd
from mse_distance import compute_mse_from_target, compute_mse
from caption import inspect_spectrogram
from constr_system_prompt import append_signal_knowledge, append_task_knowledge
from agent import OpenAIAgent, ReflectOpenAIAgent, EvalOpenAIAgent
from utils import extract_code, write_to_csv_file, redirect_stdout, \
					extract_array_from_str, convert_to_message, \
					read_data, store_data, add_execution_string, \
					add_gaurdrail_for_no_api
from prompt import no_code_feedback, reflect_prompt, eval_prompt, eval_prompt_coding, verifier_prompt

def iteration_program_output(output):
	if "An error occurred:" in output:
		# Print the captured output
		program_output = "The above program printed errors. Please fix it:\n" + output
	elif len(output) == 0:
		# program_output = "The above prorgam printed nothing. Please continue if it is meant to be the case."
		program_output = "The above code completed successfully or no code is written. If this is meant to be the case, state the keyword [SUCCESS] and the iteration will stop."
	elif len(output) >= 2048:
		program_output = "The above program printed too lengthy output. I've cropped it to 4096 characters for you.\n" + output[:4096] 
	else:
		# Print the captured output
		program_output = "The above program printed:\n" + output
	program_output = ">>>>>>" + program_output
	print(program_output)
	return program_output

def format_user_query(args):
	if args.query is not None:
		with open('query'+'/'+args.query+'.txt') as file:
			user_message = file.readline()
	else:
		user_message = input(f"user: ")

	if args.knowledge_signal:
		user_message += append_signal_knowledge(args)
	if args.knowledge_task:
		user_message += append_task_knowledge(args)

	if args.mode != 'text':
		user_message = "\\QUERY[{}]".format(user_message)
	
	return user_message

def evaluating_output(args, input_array=None, write_result=False):

	if args.target_file is not None:
		
		if args.mode == 'text':
			mse = compute_mse_from_target(args, input_array)
		else:
			mse = compute_mse(args.output_file, args.target_file, args)
		
		if 'speech' in args.target_file:
			target_metric = 'SDR (speech to noise ratio)'
		elif 'synthesis' in args.target_file:
			target_metric = 'F1 score'
		else:
			target_metric = 'MSE (mean square error)'

		m_mse = "The {} is: {:.4f}".format(target_metric, mse)
		print(m_mse)
		
		if args.write_to_csv and write_result:
			write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
		return m_mse
	else:
		return "The groundtruth is not provided."

def Agent_with_reflection(openai_key: str, system_prompt: str, global_dict, local_dict, model="gpt-3.5-turbo-0613", temperature=0.2, top_p=0.1, args=None):
	openai.api_key = openai_key
	if 'Llama' in args.openai or 'Qwen' in args.openai:
		openai.api_base = args.base_url
	
	n = args.num_trial
	reflection_piece = None
	reflect_llm = ReflectOpenAIAgent(args, model=model, system_prompt=reflect_prompt, temperature=1, top_p=1)
	
	# define challenger
	if args.eval == 'self_vis':
		eval_llm = EvalOpenAIAgent(args, model=model, system_prompt=eval_prompt, temperature=1, top_p=1)
	elif args.eval == 'self_coding':
		eval_llm = EvalOpenAIAgent(args, model=model, system_prompt=eval_prompt_coding, temperature=1, top_p=1)
	elif args.eval == 'self_verifier':
		eval_llm = EvalOpenAIAgent(args, model=model, system_prompt=verifier_prompt, temperature=1, top_p=1)

	succeed = False
	performance_list = []
	for _trial in range(n):
		
		print(f'========> Round {_trial+1} starts...')

		reply, chat, user_message, m_mse = Agent_with_API(openai_key, system_prompt, global_dict, local_dict, _trial, model, temperature, top_p, args, reflection_piece=reflection_piece, write_result=False)

		performance_list.append(m_mse)

		if n >= 2 and _trial <= n - 1:
			
			if args.eval in ('self_vis', 'self_verifier', 'self_coding'):
				eval_result = eval_llm.eval(context=chat, question=user_message, global_dict=global_dict, local_dict=local_dict, trial=_trial)
				reflect_llm.update(context=chat, question=user_message, performance=eval_result)

				if "The test passed." in eval_result:
					succeed = True 
				else:
					reflection_piece = reflect_llm.step(trial=_trial)
					reflect_llm.reset()

			elif args.eval == 'env':
				feedback = convert_to_message(m_mse)
				print(f"******** Feedback: {feedback}")
				reflect_llm.update(context=chat, question=user_message, performance=feedback)

				reflection_piece = reflect_llm.step(trial=_trial)
				reflect_llm.reset()

				if "[SUCCESS]" in reflection_piece:
					succeed = True
		
		if _trial == n - 1 or succeed:
			# if it is successful, write the results.
			# else write the first attempt
			
			mse = m_mse.split("is: ")[-1]
			if not succeed:
				mse = performance_list[0].split("is: ")[-1]
			if args.write_to_csv:
			# if True:
				write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
			return reply


def Agent_with_API(openai_key: str, system_prompt: str, global_dict, local_dict, trial, model="gpt-3.5-turbo-0613", temperature=0.2, top_p=0.1, args=None, reflection_piece=None, write_result=True):
	print(f"temperature: {temperature}, top_p: {top_p}")
	openai.api_key = openai_key
	if 'Llama' in args.openai:
		openai.api_base = args.base_url
	
	agent = OpenAIAgent(args, model=model, system_prompt=system_prompt, temperature=temperature, top_p=top_p)

	reply = agent.step()

	user_message = format_user_query(args)
	agent.update(content=user_message, role="user")

	if reflection_piece is not None:
		reflecting_message = f"""
			You've previously attempted this. Try to improve the performance based on the following reflection. {reflection_piece}
		"""
		agent.update(content=reflecting_message, role="user")

	got_result = False
	
	num_iter = 0
	max_iter = 10
	failed = 0

	while (not got_result) and num_iter <= max_iter:
		num_iter += 1
		
		reply = agent.step(stop=agent.stop)
		
		# reply = """```python inspect('./benchmark/speech-TelephoneRing2/1.wav', "Can you describe the spectrogram and identify if there are any noise artifacts, especially phone ringing? What are the frequency ranges of the speech and noise?")```"""

		returned_code = extract_code(reply)

		violation = add_gaurdrail_for_no_api(args, returned_code)
		if violation is not None:
			print('The AI attempted to use non permited API.')
			agent.update(role = "user", content=violation)
			continue

		# raise error if no code is returned
		new_text = re.sub('\n', '', returned_code, flags=re.IGNORECASE)
		if len(new_text) == 0:
			print('No code is detected in the current round!')
			agent.update(role = "user", content=no_code_feedback)
			output = ""
		else:
			code_to_execute = add_execution_string(args, returned_code)
			# input_data, sampling_rate = read_data(args.input_file)
			output = redirect_stdout(code_to_execute, global_dict, local_dict)

			program_output = iteration_program_output(output)
			agent.update(role = "user", content = program_output)

		if "An error occurred:" in output:
			if failed >= 5:
				# too many faults occur in this implementation. skip it
				return reply, agent.chat, user_message, "The result is: nan"
			failed += 1
			continue

		# check if the result is obtained. Weird keyword by GPT-4...
		if "[SUCCESS]" in reply or "SUCCESS" in reply or "SUCCEESS" in reply or num_iter == max_iter \
				or ("Llama-3-70b" in args.openai and "def solver(" in reply):
			got_result = True
			print("The result has been obtained or the max iter has been achieve.")

			# agent.update(role="assistant", content=reply)
			m_mse = evaluating_output(args, write_result=write_result)
			agent.save_chat(result=m_mse, trial=trial)
			
			return reply, agent.chat, user_message, m_mse
	return reply, agent.chat, user_message, "The result is: nan"

def Agent_based_on_text(openai_key: str, system_prompt: str, global_dict, local_dict, model="gpt-3.5-turbo-0613", temperature=0.2, top_p=0.1, args=None, write_result=True):
	print(temperature, top_p)
	openai.api_key = openai_key
	if 'Llama' in args.openai:
		openai.api_base = args.base_url

	agent = OpenAIAgent(args, model=model, system_prompt=system_prompt[0], temperature=temperature, top_p=top_p)

	# Check if input is out of context length
	if 'gpt-3.5' in args.openai and len(system_prompt[1]) > 12000:
		mse = np.nan
		if args.write_to_csv:
			write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
		return 
	elif 'Llama' in args.openai and len(system_prompt[1]) > 6000:
		mse = np.nan
		if args.write_to_csv:
			write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
		return 
	elif 'gpt-4' in args.openai and len(system_prompt[1]) > 217600:
		mse = np.nan
		if args.write_to_csv:
			write_to_csv_file(args.mode, args.query, args.index, args.log_name, mse)
		return 

	user_message = format_user_query(args)
	agent.update(role="user", content = system_prompt[1] + " " + user_message)
	
	got_result = False

	iter_num = 0
	while not got_result and iter_num <= 5:

		iter_num += 1

		reply = agent.step()
		print("assistant: " + reply)
		# agent.update(content="Tell me how you obtained the number.", role="user")
		# pdb.set_trace()
		print("The result has been obtained.")
		
		input_array = extract_array_from_str(reply)
		
		m_mse = evaluating_output(args, input_array=input_array, write_result=write_result)
		agent.save_chat(result=m_mse)
		return reply
		