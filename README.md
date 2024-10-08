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
docker exec -it my-container2 python cli2.py --mode api \
        --query TelephoneRing2 --openai gpt-4 \
        --input_file ./benchmark/speech-TelephoneRing2/1.wav
```
### Set up OpenAI key
1.3 Put your openai token into key.txt
```
echo "YOUR_OPENAI_TOKEN" >> key.txt
```

### Run the code:
Try out denoising the powerline noise from the ECG:
```
python cli2.py --mode api \
        --query powerline_2 --openai gpt-4 \
        --input_file ./benchmark/ecg_data-powerline_2/1.npy
```

## Explanations of arguments:
1. --mode: Choose between text, no_api, and api. It allows users to specify how LLMs interact with data directly through text, writing their own code, or calling upon APIs.

*mode* \in {text, api, no_api, CoT, api_react, base}

```
--mode api
```
*--mode* options:
        1.1 text: Feeding LLMs with signals in numerical form
        1.2 api: Python conding environment + API access + inspection + ReACT prompting 
        1.3 no_api: Python conding environment + inspection + ReACT prompting 
        1.4 CoT:  Python conding environment + Chain of Thought prompting
        1.5 ReAct: Python conding environment + ReAct prompting
        1.6 Base: Provide LLMs with Python conding environment only


2. --model: Choose between (gpt-3.5-turbo, gpt-4, llama-2-70b, llama-2-13b, llama-2-7b). To use Llama-2, you will need to specify your Together.ai key in together_key.txt.
```
--model gpt-4
```
3. --query: The type of signal processing problem you want to solve. These includes: extrapolation, imputation, filtering_motion, filtering_gaussian, filtering_powerline, outlier_detection, filtering_echo, filtering_ring, filtering_siren, change_point_detection, delay_detection, period_detection
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

