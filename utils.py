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
from caption import inspect_spectrogram, inspect_fft, inspect_ts
# from constr_system_prompt import append_signal_knowledge, append_task_knowledge
from scipy.io import wavfile
from openai import OpenAI

def safe_execute(code_string: str, global_dict, local_dict, keys=None):
	ans = None
	# print(global_dict, local_dict)
	try:
		exec(code_string, global_dict, local_dict)
	except Exception as e:
		print(f"An error occurred: {e}")

	return ans


def read_data(inputs):
	if isinstance(inputs, list):
		input_file = inputs[0]
	else:
		input_file = inputs
	if 'wav' in input_file:
		sampling_rate, data = wavfile.read(input_file)
	elif 'npy' in input_file:
		
		if isinstance(inputs, list) and len(inputs) > 1:
			data = [np.load(f) for f in inputs]
			min_len = min([data[i].shape[0] for i in range(len(data))])
			data = [d[:min_len] for d in data]
			# pdb.set_trace()
			data = np.stack(data)
		else:
			data = np.load(input_file)
		if 'extrapolation' in input_file or 'imputation' in input_file:
			sampling_rate = 50
		elif 'ecg' in input_file:
			sampling_rate = 500
		elif 'resampling' in input_file:
			sampling_rate = 100
		elif 'gait' in input_file:
			sampling_rate = 300 
		elif 'synthesis_7' in input_file:
			sampling_rate = 500
		elif 'recover_respiration' in input_file:
			sampling_rate = 50
		else:
			sampling_rate = None
	return data, sampling_rate

def store_data(args, data, fs):
	if 'wav' in args.output_file and 'VoiceDetector' not in args.output_file:
		wavfile.write(args.output_file, fs, data.astype(np.int16))
	else:
		np.save(args.output_file, data)
	# print("The processed data has been successfully stored.")

def convert_to_message(m_mse):
	message = 'None'
	if ':' not in m_mse:
		return message
	if "MSE" in m_mse:
		start =  m_mse.find(':')
		value = float(m_mse[start+1:])
		if value > 0.003:
			message = "The signal still looks noisy. Noise does not seem to be reduced."
		else:
			message = "The signal looks good now."
	elif "SDR" in m_mse:
		start =  m_mse.find(':')
		value = float(m_mse[start+1:])
		if value < 5:
			message = "The quality of the audio sounds bad."
		elif value < 10:
			message = "The quality of the audio sounds poor."
		elif value < 15:
			message = "The quality of the audio sounds fair now."
		elif value < 20:
			message = "The quality of the audio sounds good."
		else:
			message = "The quality of the audio is excellent"
	return message

def challenge_feedback(pre_result, inspection=False):
	message = None
	if inspection and pre_result:
		print("The inspection passed. Continue...")
		return True
	print('The challenge/verification result is: ', pre_result)
	message = "A challenger/verifier evaluated the result." 
	message = message + " The test passed." if pre_result else message + " The test failed."
	print(message)
	return message

def openai_api(messages, model, temperature=0.2, top_p=0.1, stop=None):

	got_result = False
	using_together = 'Llama' in model or 'Qwen' in model
	if using_together:
		client = openai.OpenAI(
			api_key=open("together_key.txt").read().strip(),
			base_url="https://api.together.xyz/v1",
			)
	else:
		client = OpenAI()
	trial = 0
	while not got_result and trial <= 10:
		try:
			if model in ('o1', 'o3-mini'):
				stream = client.chat.completions.create(
					model=model,
					messages=messages,
					stream=True,
					stop=stop)
			else:
				stream = client.chat.completions.create(
					model=model,
					messages=messages,
					stream=True,
					max_tokens=2048, 
					temperature=temperature, top_p=top_p, stop=stop)
			
			message = ""
			for chunk in stream:
				# print(chunk.choices[0].delta.content or "", end="", flush=True)
				if chunk.choices[0].delta.content is not None:
					message += chunk.choices[0].delta.content
			got_result = True
			trial += 1
			
		except Exception:
			sleep(3)

	return message

def extract_code(response):

	code = ''

	# index = re.search('```', response)
	# index = index.span()

	index = [(match.start(), match.end()) for match in re.finditer('```', response, flags=re.IGNORECASE)]
	index = [i[0] for i in index]
	
	if len(index) % 2 != 0:
		# raise ValueError('Incorrect format of python code detected! Please check the reply from the model.')
		# drop the last one
		index = index[:-1]

	# it is possible that the same reply contains multiple code snippet
	for i in range(0, len(index), 2):
		if '[SUCCESS]' in response[index[i]:index[i]+10]:
			continue
		if 'python' in response[index[i]:index[i]+10] or \
			'Python' in response[index[i]:index[i]+10]:
			start = index[i]+10
		else:
			continue
			# return ""

		end = index[i+1]
		# if response[index[i]+3:index[i]+9] != 'python':
		# 	raise ValueError('The model should use python code')
		code += '\n' + response[start:end] + '\n'
	
	return code 

def add_gaurdrail_for_no_api(args, code, list_of_apis=['scipy']):
	
	if args.mode != 'no_api':
		return None 

	reply = "You should only use numpy for implementation. You should not use the following libraries: "
	violate = False
	for api in list_of_apis:
		if api in code:
			reply += " " + api + " "
			violate = True

	return reply if violate else None
	

def write_to_csv_file(mode, query, index, log_name, mse):

	# Data to append
	data = [mode, query, index, mse]
	csv_file_path = './conv_history/' + log_name + '.csv'
	# Try to append to the file if it exists, otherwise create it
	if os.path.exists(csv_file_path):
		# Open the file in append mode
		with open(csv_file_path, 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(data)
	else:
		# If the file does not exist, open it in write mode to create it and write the data
		with open(csv_file_path, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['mode', 'task', 'index', 'score'])
			writer.writerow(data)

def redirect_stdout(code_to_execute, global_dict, local_dict):
	# Create a string buffer to capture the output
	buffer = io.StringIO()

	# Redirect the standard output to the buffer

	sys.stdout = buffer

	# Execute the code
	safe_execute(code_to_execute, global_dict, local_dict)

	# Reset the standard output to its original value
	sys.stdout = sys.__stdout__

	# Get the captured output from the buffer
	output = buffer.getvalue()

	# Close the buffer
	buffer.close()

	return output

def extract_array_from_str(string):
	def is_number(string):
		try:
			float(string)
			return True
		except ValueError:
			return False
	last_index_1 = string.rfind('[')
	last_index_2 = string.rfind(']')
	num_string = string[last_index_1+1:last_index_2]
	num_string = ''.join(char for char in num_string if not char.isalpha())
	if len(num_string) == 0:
		return np.array([])
	num_string = num_string.split(', ')

	result = []
	for n in num_string:
		if is_number(n):
			result.append(float(n))
	return np.array(result)


def add_execution_string(args, returned_code):
	
	code_to_execute = """
from utils import read_data, store_data
input_data, sampling_rate = read_data(args.input_file)
"""
	code_to_execute += returned_code
	code_to_execute += """
input_data, sampling_rate = read_data(args.input_file)
"""
	if "def inspection(" in code_to_execute:
		code_to_execute += "\ninspection(input_data, sampling_rate)\n"
	if "def inspection_freq(" in code_to_execute:
		code_to_execute += "\ninspection_freq(input_data, sampling_rate)\n"
	if "def preprocessing(" in code_to_execute:
		code_to_execute += "\ninput_data = preprocessing(input_data, sampling_rate)\n"
	if "def solver(" in code_to_execute:
		code_to_execute +="output_data = solver(input_data, sampling_rate)\n"
		code_to_execute +="store_data(args, output_data, sampling_rate)\n"
		code_to_execute +="print('The solver runs successfully.')"
		# code_to_execute += "print(output_data)"
	
	return code_to_execute


def bootstrap_confidence_interval(args, data, num_bootstrap_samples=100000, confidence_level=0.95):
	"""
	Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
	Also returns the median of the bootstrap means.
	
	Args:
	- data (list or array of float): 1D list or array of data points.
	- num_bootstrap_samples (int): Number of bootstrap samples.
	- confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
	
	Returns:
	- str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
	"""
	# Convert data to a numpy array for easier manipulation
	data = np.array(data)

	# List to store the means of bootstrap samples
	bootstrap_means = []

	# Generate bootstrap samples and compute the mean for each sample
	for _ in range(num_bootstrap_samples):
		# Resample with replacement
		bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
		# Compute the mean of the bootstrap sample
		bootstrap_mean = np.mean(bootstrap_sample)
		bootstrap_means.append(bootstrap_mean)

	# Convert bootstrap_means to a numpy array for percentile calculation
	bootstrap_means = np.array(bootstrap_means)

	# Compute the lower and upper percentiles for the confidence interval
	lower_percentile = (1.0 - confidence_level) / 2.0
	upper_percentile = 1.0 - lower_percentile
	ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
	ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

	# Compute the median of the bootstrap means
	median = np.median(bootstrap_means)

	# Convert to percentages and format to one decimal place
	ci_lower_percent = ci_lower
	ci_upper_percent = ci_upper
	median_percent = median

	if 'speech' in args.query:
		metric = 'SDR'
		if np.isnan(median_percent):
			median_percent = -1e6
	elif 'change_point_detect' in args.query or 'outlier_detect' in args.query:
		metric = 'F1'
		if np.isnan(median_percent):
			median_percent = 0
	else:
		metric = 'MSE'
		if np.isnan(median_percent):
			median_percent = 1e6

	# Return the formatted string with confidence interval and median
	return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.3f}, {ci_upper_percent:.3f}), {metric}: {median_percent:.3f}", median_percent, metric

def extract_content(s, k1="<idea>", k2="</idea>"):
	match = re.search(rf"{k1}(.*?){k2}", s, re.DOTALL)
	return match.group(1) if match else ""