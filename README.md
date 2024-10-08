# SensorBench: Benchmarking LLMs in Coding-Based Sensor Processing
Code to reproduce results in our paper: SensorBench: Benchmarking LLMs in Coding-Based Sensor Processing

![Benchmark composition](images/pie_chart.png)

**Benchmark composition**

![Benchmark composition](images/algo_bd-Overview.drawio.png)

**We envision an intelligent assistant to support users, making advanced sensor data analysis accessible to a much broader audience.**

## Setup
1. Install required packages by:
```
pip install -r requirements.txt
```

### Alternatively, install the environment by docker
i) Build the docker image:
```
docker build -t my-python-app .
```
ii) Build the docker container:
```
docker run -p 4000:80 -v ./:/usr/src/myapp --name my-container  my-python-app /bin/bash
```
iii) Start the container:
```
docker start my-container
```
iv)Execute code on my container:
```
docker exec -it my-container /bin/bash
```
or 
```
docker exec -it my-container2 python cli.py --mode api \
        --query ecg_data-powerline_2 --openai gpt-4 \
        --index 1 --num_trial 1
```
### Set up OpenAI key
1.3 Put your openai token into key.txt
```
echo "YOUR_OPENAI_TOKEN" >> key.txt
```
### (Optional) Set up together.ai key
1.4 Put your together.ai token into together_key.txt
```
echo "YOUR_TOGETHER_AI_TOKEN" >> together_key.txt
```

### Prepare Benchmark
Download the benchmark used in the paper and unzip it in the main folder from [here](https://drive.google.com/file/d/1a6M2MOHWu1JQL5cb9BMicT-J7vbqWnRg/view?usp=sharing).

(Optional) To access the full benchmark, you can view it from [here](https://drive.google.com/file/d/1gVE8_MEk0ZU9ZcspZPhLlO5_L_NQy2Mf/view?usp=sharing).

### Run the code:
Try out denoising the powerline noise from the ECG data sample #1:
```
python cli.py --mode api \
        --query ecg_data-powerline_2 --openai gpt-4o \
        --index 1 --num_trial 1
```

## Explanations of arguments:
1. --mode: Choose between text, no_api, and api. It allows users to specify how LLMs interact with data directly through text, writing their own code, or calling upon APIs.

*mode* \in {'text', 'api', 'no_api', 'CoT', 'react', 'base'}

```
--mode api
```

*--mode* options:

        1.1 text: Feeding LLMs with signals in numerical form

        1.2 api: Python conding environment + API access + inspection + ReACT prompting 

        1.3 no_api: Python conding environment + inspection + ReACT prompting 

        1.4 CoT:  Python conding environment + API access + Chain of Thought prompting

        1.5 ReAct: Python conding environment + API access + ReAct prompting

        1.6 Base: Python conding environment + API access


2. --model: Choose between ('gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-4-0125-preview', \
        'gpt-4-turbo', 
        'Llama-2-70b', 'Llama-2-13b', 'Llama-2-7b', 'Llama-3-8b', 'Llama-3-70b', \
        'Qwen1.5-110B', 'Qwen2-72B). To use models from [together.ai](https://https://www.together.ai/pricing)(not gpt-X models from OpenAI), you will need to specify your Together.ai key in together_key.txt.
```
--model gpt-4
```
3. --query: The type of signal processing problem you want to solve. These includes: 
ecg_data-extrapolation, ecg_data-gaussian, ecg_data-heartrate, ecg_data-imputation, ecg_data-motion, ecg_data-powerline_1, ecg_data-powerline_2, ecg_data-powerline_3,  gait-delay_detection, gait-period_detection, ppg-extrapolation, 
ppg-imputation, resampling, speech-echo, speech-Siren, speech-TelephoneRing1, 
speech-TelephoneRing2, speech-TelephoneRing3, change_point_detect_1, change_point_detect_2, change_point_detect_3, change_point_detect_4, outlier_detect_1, outlier_detect_2, outlier_detect_3, outlier_detect_4
```
--query filtering_ring
``` 
4. --index: The index of data sample provided for LLMs. *index* \in {1, 2, 3}
```
--index 1
```
5. --num_trial: The number of self-verification round. Setting *num_trial* to 1 disable self-verification. In the paper, we select *num_trial* \in {1, 3, 4, 5}
```
--num_trial 5
```

## Plug in your sensor data and query
TODOs

## Modify the prompts
In *sys_prompt.py*, we define the prompting strategies. We suggest your add or modify prompts in the file to build your own agents.

### Examples of prompts
1. base prompt:
'''
```
You are an expert in signal processing. Your role is to analyze and process various types of signals (such as audio, electromagnetic, or physiological signals) using your Python coding. You are expected to process signal directly without user interference.

Instructions:

1. Python Coding: Use Python codinng for signal processing tasks. Implement your functions inside ```Python ``` code block. Do not write code outside the functions. The function prototypes are as follows:

You just need to implement the function the solver (mandatory):

 ```Python 
def solver(input_data, sampling_rate=None):
    # HERE is where you put your solution
    # Args:
    #   input_data: The data type is numpy.ndarray. This is the data provided by the user to perform DSP. 
    #   sampling_rate: The sampling rate of the data. sampling_rate is mandatory for speech, ecg, ppg, and gait data. It could be optional for others.
    # Output:
    #   return: return the processed data in numpy.ndarray
 ```

Please note that variables input_data and sampling_rate are provided through the function API. Do not simulate them or write code outside the designated function.

2. [IMPORTANT] Specific Interactive Format: State all your output directly. DO NOT put it inside code or with ```. Users will put their queries into the format \\QUERY[text]. For example, \\QUERY[Can you denoise my ECG signal that's corrupted by powerline noise?]. When you finished, state the keyword [SUCCEESS], and the iteration will stop. Output [SUCCEESS] in the chat directly. 

```
'''