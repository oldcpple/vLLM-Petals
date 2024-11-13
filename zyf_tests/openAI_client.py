import requests
import time
import multiprocessing
import random
import os
import numpy as np
import json

os.chdir(os.path.dirname(__file__))

prompts_file = 'prompts.txt'
prompts = []
with open(prompts_file, 'r') as f:
    for line in f:
        line = line[:-1]
        prompts.append(line)

#print(prompts)

timestamps_file = 'timestamps.txt'
timestamps = []
with open(timestamps_file, 'r') as f:
    for line in f:
        line = int(line)
        timestamps.append(line)

base = timestamps[0]
timestamps = [timestamps[i] - base for i in range(len(timestamps))]

def vllm_client_func(prompt):
    url = "http://127.0.0.1:8001/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0
    }
    start = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(data))
    end = time.time()
    
    if response.status_code == 200:
        response_json = response.json()
        output = response_json["choices"][0]["text"]
        span = end - start
        num_tokens = len(output.split(' '))
        speed = num_tokens / span
        return [num_tokens, span, speed]
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return [0, 0, 0]

results = []
num_client = 100
max_span = timestamps[num_client - 1]
random_list = [random.randint(0, 150) for _ in range(num_client)]

# submit prompts with time distribution
start = time.time()
index = 0
output = []
with multiprocessing.Pool(processes=num_client) as pool:
    results = []
    while True:
        if time.time() - start >= max_span:
            break
        if index < num_client and time.time() - start >= timestamps[index]:
            print('prompt: {}, timestamp: {}'.format(prompts[random_list[index]], time.time()))
            results.append(pool.apply_async(vllm_client_func, args=(prompts[random_list[index]],)))
            index += 1
    output = [result.get() for result in results]

for result in output:
    print('num tokens: {}, time span: {}, speed: {}'.format(result[0], result[1], result[2]))

print()
avg = np.mean(output, axis=0)
print('avg num tokens: {}, avg time span: {}, avg speed: {}'.format(avg[0], avg[1], avg[2]))
