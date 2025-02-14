import os
import shutil
import argparse

def copy_vllm_files(vllm_src_dir: str, vllm_dest_dir: str):
    """
    Copy VLLM files from source directory to destination directory.
    
    Args:
        vllm_src_dir (str): Source directory containing VLLM files
        vllm_dest_dir (str): Destination directory where files will be copied
    """    
    # Define the files to copy with their relative paths
    files_to_copy = [
        "vllm/sampling_params.py",
        "vllm/model_executor/layers/sampler.py",
        "vllm/sequence.py",
        "vllm/utils.py",
        "vllm/worker/model_runner_base.py",
        "vllm/engine/llm_engine.py",
        "vllm/engine/async_llm_engine.py",
        "vllm/entrypoints/api_server.py",
        "vllm/entrypoints/api_server_subsequent.py",
        "vllm/entrypoints/openai/api_server.py",
        "vllm/executor/executor_base.py",
        "vllm/executor/gpu_executor.py",
        "vllm/model_executor/model_loader/loader.py",
        "vllm/model_executor/model_loader/utils.py",
        "vllm/model_executor/model_loader/__init__.py",
        "vllm/model_executor/models/llama.py",
        "vllm/model_executor/models/pixtral.py",
        "vllm/model_executor/models/llava_next.py",
        "vllm/model_executor/models/utils.py",
        "vllm/model_executor/utils.py",
        "vllm/worker/utils.py",
        "vllm/worker/worker.py",
        "vllm/worker/worker_base.py",
        "vllm/config.py",
        "vllm/engine/arg_utils.py",
        "vllm/worker/model_runner.py"
    ]

    for file_path in files_to_copy:
        # Construct full source and destination paths
        src_file = os.path.join(vllm_src_dir, file_path)
        dest_file = os.path.join(vllm_dest_dir, file_path)
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        
        try:
            # Copy the file
            shutil.copy2(src_file, dest_file)
            print(f"Successfully copied: {file_path}")
        except FileNotFoundError:
            print(f"Error: Source file not found: {src_file}")
        except PermissionError:
            print(f"Error: Permission denied when copying: {file_path}")
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Copy VLLM files from source to destination directory')
    parser.add_argument('--src-dir', dest='vllm_src_dir', default="/home/gpu2/zk/project/LLM/MoLink", help='Source directory containing VLLM files')
    parser.add_argument('--dest-dir', dest='vllm_dest_dir', default="/home/gpu2/miniconda3/envs/zk_MoLink_manual/lib/python3.12/site-packages", help='Destination directory where files will be copied')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the copy operation
    copy_vllm_files(args.vllm_src_dir, args.vllm_dest_dir)

if __name__ == "__main__":
    main()
