import pdb 
import numpy as np 
import pandas as pd 
from sys_prompt import api, no_api, text, react, base, CoT
from utils import read_data

def encode_number(args, ECG_str, ecg_data):

	# normalize the PPG
	if args.file in ('ppg', 'general'):
		ecg_data = ecg_data / max(np.abs(ecg_data))
		ecg_data = np.round(ecg_data, decimals=2)
		ecg_data = np.round(100 * ecg_data, decimals=0)
	else:
		ecg_data = np.round(1000 * ecg_data, decimals=0)

	for n, val in enumerate(ecg_data):
		if np.isnan(val):
			# handle NaN differetly
			ECG_str.append(str(val))
		else:
			ECG_str.append(str(val)[:-2])

	return ECG_str


def append_api_info(args):
	content = "\n"
	api_content = ""
	if args.mode in ('api', 'CoT', 'react', 'base'):
		with open('API/general.txt', 'r') as file:
			# Step 2: Read the file's content into a string variable

			api_content += file.read()
	
	return (content + api_content + "\n")

def append_signal_knowledge(args):
	content = "\n"
	api_content = ""
	with open('knowledge_signal/'+args.target_file.split('/')[2]+'.txt', 'r') as file:
		# Step 2: Read the file's content into a string variable
		api_content += file.read()
	return (content + api_content + "\n")

def append_task_knowledge(args):
	content = "\n"
	api_content = ""
	if args.mode == 'text':
		if args.query in ('extrapolation'):
			with open('knowledge_task/'+args.query+'.txt', 'r') as file:
				# Step 2: Read the file's content into a string variable
				api_content += file.read()
	return (content + api_content + "\n")

class SystemPrompt(object):
	"""docstring for SystemPrompt"""
	def __init__(self, 
		  imu_file = "./data/sample.csv",
		  geo_file = "./data/geo.csv",
		  input_file = "./data/heart_rate.csv",
		  system_prompt_file = './system_prompt.txt',
		  args = None
		):
		super(SystemPrompt, self).__init__()
		self.imu_file = imu_file
		self.geo_file = geo_file
		self.input_file = input_file

		file_content = globals()[args.mode]
		
		self.system_prompt = file_content

		self.system_prompt += append_api_info(args)

		self.system_prompt += """
			Now I am going to provide the query to you, and you need to start answering the query using Python. Say "I am ready" if you understand the problem.
		"""

def num2str(args, input_file, length):
	# ecg_data = np.load(input_file)
	ecg_data, _ = read_data(input_file)
	ECG_str = encode_number(args, [], ecg_data)
	if length is not None:
		ECG_str = ECG_str[:length]
	ECG_str = ', '.join(ECG_str)
	return ECG_str

class SystemPromptECGPPG(object):
	"""docstring for SystemPrompt"""
	def __init__(self, 
		  imu_file = "./data/sample.csv",
		  geo_file = "./data/geo.csv",
		  input_file = "./data/heart_rate.csv",
		  system_prompt_file = './system_prompt_ecg.txt',
		  length = 200,
		  freq = 50,
		  ecg_idx = 14,
		  mode = 'text',
		  args = None
		):
		super(SystemPromptECGPPG, self).__init__()
		self.imu_file = imu_file
		self.geo_file = geo_file
		self.input_file = input_file

		file_content = globals()[args.mode]

		# Step 3: Replace the desired string with a specific phrase
		modified_content = file_content.replace("imu_file", imu_file)
		modified_content = modified_content.replace("geo_file", geo_file)
		
		self.system_prompt = modified_content

		self.system_prompt += """
		Put the needed values inside the bracket Predicted_results=[] only. Do not combine it with the original data and and return them together. Do NOT write code.
		"""

		if 'gait-delay_detection' in args.query:
			ECG_str_1 = num2str(args, args.input_file[0], length)
			ECG_str_2 = num2str(args, args.input_file[1], length)
		else:
			ECG_str = num2str(args, args.input_file[0], length)

		if mode == 'text':
			self.system_prompt = [self.system_prompt, ""]
			if args.file == 'ppg':
				self.system_prompt[1] += "\nI have a PPG sequence: \n PPG=[" + ECG_str +"]"
			elif args.file == 'ecg':
				self.system_prompt[1] += "\nI have an ECG sequence (unit in Microvolts): \n ECG=[" + ECG_str +"]"
			elif 'gait-delay_detection' in args.query:
				self.system_prompt[1] += "\n Seq_1=[{}]. Seq_2=[{}]".format(ECG_str_1, ECG_str_2)
			else:
				self.system_prompt[1] += "\n Seq=[" + ECG_str +"]"

				

if __name__ == '__main__':
	# prompt = SystemPrompt()
	prompt = SystemPromptECGPPG()
