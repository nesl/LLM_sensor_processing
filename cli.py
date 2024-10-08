import argparse
import os
import re
import sys

# import torch

from utils import challenge_feedback
from constr_system_prompt import  SystemPrompt, SystemPromptECGPPG
import pdb 
from caption import inspect_spectrogram, inspect_fft, inspect_ts 
import numpy as np
from chat import Agent_with_API, Agent_based_on_text, Agent_with_reflection

global_dict, local_dict = globals(), locals()

def main(args):
    
    if 'speech' in args.query:
        args.input_file = ['./benchmark/' + args.query + '/' + args.index + '.wav']
    elif 'imputation' in args.query or 'extrapolation' in args.query:
        args.input_file = ['./benchmark/' + args.query + '/' + args.index + '_50.npy']
    elif 'gait-delay_detection' in args.query:
        args.input_file = ['./benchmark/' + args.query + '/' + args.index + '_1.npy', \
                           './benchmark/' + args.query + '/' + args.index + '_2.npy']
    elif 'gait-period_detection' in args.query:
        args.input_file = ['./benchmark/' + args.query + '/' + args.index + '_1.npy']
    else:
        args.input_file = ['./benchmark/' + args.query + '/' + args.index + '.npy']


    assert args.openai in ('gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-4-0125-preview', \
        'gpt-4-turbo', 
        'Llama-2-70b', 'Llama-2-13b', 'Llama-2-7b', 'Llama-3-8b', 'Llama-3-70b', \
        'Qwen1.5-110B', 'Qwen2-72B')
    
    prefix = ''
    if 'Llama' in args.openai:
        prefix = 'meta-llama/'
        args.openai = prefix + args.openai + '-chat-hf'
    elif 'Qwen1.5' in args.openai:
        prefix = 'Qwen/'
        args.openai = prefix + args.openai + '-Chat'
    elif 'Qwen2' in args.openai:
        prefix = 'Qwen/'
        args.openai = prefix + args.openai + '-Instruct'
    elif 'Mixtral' in args.openai:
        prefix = 'mistralai/'
        args.openai = prefix + args.openai + '-Instruct-v0.1'

    if 'ecg' in args.input_file[0]:
        args.file = 'ecg_data'
    elif 'ppg' in args.input_file[0]:
        args.file = 'ppg'
    else:
        args.file = 'general'

    # handle target file formating
    target_file = args.input_file[0].split('/')
    filename = target_file[-1].split('.')
    filename[0] = filename[0] + '_gt'
    filename = '.'.join(filename)
    target_file[-1] = filename
    if 'VoiceDetector' in args.input_file[0]:
        target_file[-1] = filename.replace('wav', 'npy')

    args.target_file = '/'.join(target_file)

    # handle output file formating
    output_dir = './llm_response/'
    args.output_file = output_dir + f'{args.openai}_{args.query}_{args.index}_{args.num_trial}.' + filename.split('.')[-1]
    if not os.path.exists(output_dir + prefix):
        os.makedirs(output_dir + prefix)

    if 'VoiceDetector' in args.input_file[0]:
        args.output_file = output_dir + f'{args.openai}_{args.query}_{args.index}_{args.num_trial}.' + 'npy'
    # # file index
    # args.index = args.input_file[0].split('/')[-1].split('.')[0].split('_')[0]
    # log name
    args.log_name = f"{args.openai}_{args.mode}_{args.eval}_#trial_{args.num_trial}"

    # update reflection number
    if args.adaptive_reflect and \
        'extrapolation' in args.input_file[0] or \
        'imputation' in args.input_file[0]:
        print('Disable reflection since models are incapable of doing so.')
        args.num_trial = 1
        # pdb.set_trace()

    try:
        if 'gpt-3' in args.openai or 'gpt-4' in args.openai\
            or 'Llama' in args.openai or 'Qwen' in args.openai \
                or 'Mixtral' in args.openai:
            
            # model = 'gpt-3.5-turbo' if 'gpt-3' in args.openai else 'gpt-4'
            model = args.openai

            if 'Llama' in args.openai or 'Qwen' in args.openai or \
                'Mixtral' in args.openai:
                openai_key = open("together_key.txt").read().strip()
            else:
                openai_key = open("key.txt").read().strip()

            # construct system prompt
            args.system_prompt_file = './sys/system_prompt_signal_processing_'+args.mode+'.txt'
            if args.mode == 'text':
                system_prompt = SystemPromptECGPPG(system_prompt_file=args.system_prompt_file,
                                    length=args.ts_len, mode=args.mode, args=args)
                # chat_openai_io_text(openai_key, system_prompt.system_prompt, global_dict, local_dict, model=model, temperature=0.8, top_p=1, args=args)
                Agent_based_on_text(openai_key, system_prompt.system_prompt, global_dict, local_dict, model=model, temperature=0.8, top_p=1, args=args)
            elif args.mode in ('api', 'no_api', 'CoT', 'react', 'base'):
                system_prompt = SystemPrompt(imu_file = args.imu_file, 
                                            geo_file=args.geo_file, input_file=args.input_file, system_prompt_file=args.system_prompt_file, args=args)
                Agent_with_reflection(openai_key, system_prompt.system_prompt, global_dict, local_dict, model=model, temperature=args.temperature, top_p=args.top_p, args=args)
                # Agent_with_API(openai_key, system_prompt.system_prompt, global_dict, local_dict, model=model, temperature=args.temperature, top_p=args.top_p, args=args)
                # chat_openai_io(openai_key, system_prompt.system_prompt, global_dict, local_dict, model=model, temperature=args.temperature, top_p=args.top_p, args=args)
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default it to use gpt-4o
    parser.add_argument(
        "--openai", type=str, default='gpt-4o', help="default use gpt-4 model"
    )
    parser.add_argument(
        "--imu_file", type=str, default='./data/sample.csv', help="default imu data"
    )
    parser.add_argument(
        "--geo_file", type=str, default='./data/geo.csv', help="default geolocation data"
    )
    parser.add_argument(
        "--input_file", type=str, default=None, nargs='+', help="default heart rate data"
    )
    parser.add_argument(
        "--target_file", type=str, default=None, help="default heart rate data"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="The file that you want the model to produce."
    )
    parser.add_argument(
        "--system_prompt_file", type=str, default='./system_prompt.txt', help="default system prompt"
    )
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument(
        "--mode", type=str, default='code', help="Conversational AI mode"
    )
    parser.add_argument(
        "--index", type=str, default=None, help="file index"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="-"
    )
    parser.add_argument(
        "--ts_len", type=int, default=None, help="Time series sequence length"
    )
    parser.add_argument(
        "--CoT",
        action="store_true",
        help="Use chain of thought",
    )
    parser.add_argument(
        "--knowledge_signal",
        action="store_true",
        help="Whether to inject signal knowledge",
    )
    parser.add_argument(
        "--adaptive_reflect",
        action="store_true",
        help="Adaptively reflect on its solution. When it is True, \
            number of reflection will be set to 1 for imputation and extrapolation.",
    )
    parser.add_argument(
        "--knowledge_task",
        action="store_true",
    )
    parser.add_argument(
        "--write_to_csv",
        action="store_true",
        help="write results to csv file",
    )
    parser.add_argument(
        "--query", type=str, default=None, help="user's query for testing"
    )
    parser.add_argument(
        "--encode",
        type=str, default='env', help="""
        Ways to present sequences to the modes. They include: number, space, and alpabet.
        """
    )
    parser.add_argument(
        "--eval",
        type=str, default='self_coding', help="""
        Feedback from the environment or self-generated. (env | self_vis | self_coding | self_verifier)
        """
    )
    parser.add_argument(
        "--bw_pred",
        type=int, default=0, help="""
        Whether we want the model to do backward extrapolation (if bw_pred >= 1)
        """
    )
    parser.add_argument(
        "--num_trial",
        type=int, default=1, help="""
        How many times can the model reflect and retry
        """
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.together.xyz/v1", help="together.ai interface"
    )
    parser.add_argument(
        "--log_name", type=str, default="test", help="The type of task we are testing."
    )
    args = parser.parse_args()
    main(args)
