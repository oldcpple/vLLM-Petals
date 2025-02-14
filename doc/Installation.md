# Installation (Temporary)

1. 创建虚拟环境

   ```bash
   conda create -n MoLink python=3.12
   conda activate MoLink
   ```

2. 安装vllm==0.6.3post1，务必指定本版本

   ```bash
   pip install vllm==0.6.3post1
   ```

3. 安装其他依赖包

   ```bash
   pip install hivemind==1.1.10.post2
   pip install pydantic==2.9.2
   pip install vllm-flash-attn
   ```

4. 克隆MoLink项目

   ```bash
   git clone https://github.com/oldcpple/MoLink.git
   ```

5. 使用MoLink项目下的相关文件替换vllm安装路径下的对应文件

   * 进入MoLink项目

     ```bash
     cd MoLink
     ```

   * 指定源路径和目标路径，执行`copy_vllm_files.py`脚本进行替换

     ```bash
     python copy_vllm_files.py --src-dir src_dir --dest-dir dest_dir
     ```

6. 执行`conflicts.py`脚本，解决hivemind冲突

   * 修改`conflicts.py`脚本，把hivemind的路径修改为当前虚拟环境中hivemind的安装路径，如`/home/username/miniconda3/envs/MoLink/lib/python3.12/site-packages/hivemind`

   * 运行脚本

     ```bash
     python conflicts.py
     ```

7. 复制MoLink根路径下的dht文件夹到pip安装路径下

   ```bash
   cp -r ./dht dest_dir
   ```

8. 安装完毕，进行测试

   * 节点1

     ```bash
     export CUDA_VISIBLE_DEVICES=0
     vllm serve meta-llama/Llama-2-7b-chat-hf --port 8080 --serving-blocks 0,15
     ```

   * 节点2

     ```bash
     export CUDA_VISIBLE_DEVICES=1
     python -m vllm.entrypoints.api_server_subsequent \
     --model meta-llama/Llama-2-7b-chat-hf \
     --port 8081 \
     --serving-blocks 16,31 \
     --initial-peer /ip4/127.0.0.1/tcp/44649/p2p/12D3KooWPf2rtWhSLRrVgVzuX6wGqzcKkDL7DqKDdBK857WoVzo8
     ```

     * 注意：把`--initial-peer`替换为节点1启动后的实际地址

   * 测试服务

     ```bash
     curl http://localhost:8080/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
             "model": "meta-llama/Llama-2-7b-chat-hf",
             "prompt": "介绍一下你自己。",
             "max_tokens": 1024,
             "temperature": 0
         }'
     ```

     