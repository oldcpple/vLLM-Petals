import subprocess

# Define batch sizes and tensor parallel configurations to test
batch_sizes = [1, 8, 16,32, 64, 128]
tensor_parallel_configs = [1, 2]

# Store results to generate the final table
results = []

# Run the test for each combination of batch size and tensor parallel configuration
for tensor_parallel in tensor_parallel_configs:
    for batch_size in batch_sizes:
        # Use subprocess to call offline_inference.py with the parameters
        process = subprocess.Popen(
            ["python", "offline_inference.py", str(batch_size), str(tensor_parallel)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Real-time monitoring of the output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Real-time print for monitoring
                if "ms" in output and "tokens/s" in output:
                    # Parse the output line for batch size, tensor parallel, latency, and throughput
                    parsed_line = output.split("\t")
                    batch_size_value = parsed_line[0].split(" × ")[0]
                    tensor_parallel_value = parsed_line[1].split(" × ")[0]
                    latency_value = parsed_line[2].replace(" ms", "")
                    throughput_value = parsed_line[3].replace(" tokens/s", "")
                    results.append((batch_size_value, tensor_parallel_value, latency_value, throughput_value))

# Generate the final table
print("\nBatch Size\tTensor Parallel\tLatency(ms)\tThroughput(tokens/s)")
for result in results:
    print(f"{result[0]}\t\t{result[1]}\t\t{result[2]}\t\t{result[3]}")
