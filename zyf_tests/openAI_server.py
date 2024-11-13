import subprocess

# 启动服务的命令
command = ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Llama-2-7b-chat-hf", "--port", "8001"]

# 运行命令启动服务
subprocess.Popen(command)
print("Server started with model meta-llama/Llama-2-7b-hf.")
