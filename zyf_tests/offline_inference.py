import sys
import time
import random
from vllm import LLM, SamplingParams

# Read parameters from command line
batch_size = int(sys.argv[1])
tensor_parallel = int(sys.argv[2])

# Read prompts from file
with open("prompts.txt", "r") as file:
    prompts = file.readlines()

# Strip newline characters from prompts
prompts = [prompt.strip() for prompt in prompts]

# Randomly select the appropriate number of prompts for the current batch size
batch_prompts = random.sample(prompts, batch_size)

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=150, max_tokens=155)

# Initialize the LLM with the specified tensor parallel configuration
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=tensor_parallel)

# Measure latency
start_time = time.time()
outputs = llm.generate(batch_prompts, sampling_params)
end_time = time.time()
latency = (end_time - start_time) * 1000  # in milliseconds

# Calculate throughput
total_tokens = sum(len(output.outputs[0].text.split()) for output in outputs)
throughput = total_tokens / (end_time - start_time)  # tokens per second

# Print the results in the format: batch_size, tensor_parallel, latency, throughput
print(f"{batch_size} × 32\t{tensor_parallel} × RTX 3090 Ti\t{latency:.2f} ms\t{throughput:.2f} tokens/s")
