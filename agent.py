import pdb 
from utils import openai_api, extract_code, redirect_stdout
import os 
from caption import inspect_spectrogram, inspect_fft, inspect_ts 
import re 
from datetime import datetime

class OpenAIAgent:
    def __init__(self, args, model, system_prompt, temperature, top_p):
        self.args = args 
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p 
        self.model = model 
        self.chat = [ {"role": "system", "content": self.system_prompt
				}]
        
        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        conv_dir = './conv_history/{}/'.format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name 
        
        if args.mode == 'base':
            self.stop = None
        else:
            self.stop = ["```\n", "```\n\n", "</s>"]

    def update(self, content, role):
        self.chat.append(
            {"role": role, "content": content}
        )

    def step(self, stop=None):
        # chat = openai_api(self.chat, self.model, temperature=self.temperature, top_p=self.top_p, stop="```\n")
        message = openai_api(self.chat, self.model, temperature=self.temperature, top_p=self.top_p, stop=stop)
        # message = chat.choices[0].message.content
        if "```Python" in message or "```python" in message:
            message += "```\n"
        self.update(message, "assistant")
        print("assistant: " + message)
        return message

    def reset(self):
        # only the system prompt is kept
        self.chat = self.chat[0]
    
    def save_chat(self, trial=1, result="None"):
        # save conversation history
        with open(self.file_name + f'_trial_{trial}.txt', 'w') as file:
            # Iterate over each item in the list
            for item in self.chat:
                # Write each item to the file followed by a newline character
                file.write(str(item) + '\n')
            if self.args.target_file is not None:
                file.write(result)

class ReflectOpenAIAgent(OpenAIAgent):
    def __init__(self, args, model, system_prompt, temperature, top_p):
        super(OpenAIAgent).__init__()

        self.args = args 
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p 
        self.model = model 
        self.chat = [ {"role": "system", "content": self.system_prompt
				}]
        self.performance_hist = []

        # self.context, self.question, self.performance = context, question, performance
        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"Reflector_{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        conv_dir = './conv_history/{}/'.format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name 
    
    def reset(self):
        self.chat = [ {"role": "system", "content": self.system_prompt
				}]
    
    def update(self, context, question, performance):
        self.performance_hist.append(performance)

        perf_hist = [f"In trial #{i+1}, the performance is - " + self.performance_hist[i] for i in range(len(self.performance_hist))]
        perf_hist = "An external source perform evalution on your output signal w.r.t. the ground truth signal. " \
            + ". ".join(perf_hist)
        self.chat[0]["content"] = self.chat[0]["content"].format(context=context[2:], question=question, performance=performance, performance_hist=perf_hist)

    def step(self, trial=0):
        
        message = openai_api(self.chat, self.model, temperature=self.temperature, top_p=self.top_p)
        # message = chat.choices[0].message.content
        print(f"""Reflecting...
            {message}
            """)
        self.chat.append(
            {"role": 'assistant', "content": message}
        )
        self.save_chat(trial=trial)
        
        return message

class EvalOpenAIAgent(OpenAIAgent):
    def __init__(self, args, model, system_prompt, temperature, top_p):
        super(OpenAIAgent).__init__()

        self.args = args 
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p 
        self.model = model 
        self.chat = [ {"role": "system", "content": self.system_prompt
				}]
        self.performance_hist = []
        self.memory = []
        self.memory_str = ""

        # self.context, self.question, self.performance = context, question, performance
        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"{args.eval}_{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        conv_dir = './conv_history/{}/'.format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name 
    
    def extract_result(self, result: str):
        # start = result.find('EVAL[')
        # end = result[start:].find(']')
        # return result[start:start+end]
        start = result.find('EVALUATION')
        end = -1
        return result[start:end]

    def update_memory(self, result):
        self.memory.append(result)
        self.memory_str = [f"In trial #{i+1}, your evaluation is - " + self.memory[i] for i in range(len(self.memory))]
        self.memory_str = " ".join(self.memory_str)

    def reset(self):
        self.chat = [ {"role": "system", "content": self.system_prompt
				}]
    
    def init(self, context, question, vis_result=None):
        self.chat[0]["content"] = self.chat[0]["content"].format(context=context[2:], question=question, memory=self.memory_str, vis_result=vis_result)
        # self.chat[0]["content"] = self.chat[0]["content"].format(context="None", question=question, memory=self.memory_str)

    def update(self, content, role):
        self.chat.append(
            {"role": role, "content": content}
        )

    def step(self, stop=None):
        message = openai_api(self.chat, self.model, 
                temperature=self.temperature, top_p=self.top_p,
                stop=stop)
        # message = chat.choices[0].message.content

        if "```Python" in message or "```python" in message:
            message += "```\n"

        print(message)
        self.update(message, "assistant")
        return message

    def eval(self, context, question, global_dict, local_dict, trial=0):

        vis_output_str = """
from utils import read_data, store_data
output_data, sampling_rate = read_data(args.output_file)
input_data, sampling_rate = read_data(args.input_file)
print(f"The produced output_data is: ", output_data)
"""
        vis_result = redirect_stdout(vis_output_str, global_dict, local_dict)

        self.init(context, question, vis_result)
        reply = "" 
        i = 0
        succeeded = False
        failed = 0
        while i < 5 and not succeeded:

            reply = self.step(stop=["```\n", "```\n\n", "</s>"])
            code = extract_code(reply)
            result = ""
            if len(code) == 0:
                if '[EVALUATION]' in reply:
                    succeeded = True
                self.update(
                    content="Please go ahead. Remember to put your final evaluation after [EVALUATION] and the iteration will stop.", role="user"
                )
            else:

                code_to_execute = "\n" + code
                code_to_execute += """
from utils import read_data, store_data
output_data, sampling_rate = read_data(args.output_file)
input_data, sampling_rate = read_data(args.input_file)
"""

                if 'def inspection(' in code_to_execute:
                    code_to_execute += "inspect_result = inspection(input_data, output_data, sampling_rate)\n"
                    code_to_execute += "challenge_feedback(inspect_result, inspection=True)\n"
                elif 'def challenger(' in code_to_execute or 'def verifier(' in code_to_execute:
                    if self.args.eval == 'self_coding':
                        code_to_execute += "result = challenger(input_data, output_data, sampling_rate)\n"
                    else:
                        code_to_execute += "result = verifier()\n"
                    code_to_execute += "challenge_feedback(result)\n"
                
                # redirect_stdout("print(output_data)", global_dict, local_dict)
                # redirect_stdout("print(input_data)", global_dict, local_dict)
                result = redirect_stdout(code_to_execute, global_dict, local_dict)

                if len(result) == 0:
                    result += "The above program prints nothing. If it is not intended, remember to use print() function. Remember to put your final evaluation after [EVALUATION] and the iteration will stop."
                else:
                    result = "The program output: " + result
                print(result)
                
                self.update(
                    content=result, role = "user"
                )
                
                if "An error occurred:" in result:
                    if failed >= 3:
                        return 'The challenge/verification result is: False'
                    failed += 1
                    continue
                
                if '[EVALUATION]' in reply or 'The challenge/verification result is: ' in result:
                    succeeded = True
            i += 1
        # update memory
        # evaluation = self.extract_result(reply)
        
        evaluation = reply + '\n' +result
        self.update_memory(evaluation)
        self.save_chat(trial=trial)
        self.reset()
        return evaluation



