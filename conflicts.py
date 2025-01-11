import os
import re
from pathlib import Path
import importlib.util

# 通过 importlib.util 导入包并获取文件路径
package_name = 'hivemind'
module_name = 'dht.schema'

# 获取模块的路径
spec = importlib.util.find_spec(package_name)
if spec is None:
    raise ImportError(f"包 {package_name} 未找到")

# 获取模块文件的路径
module_path = os.path.join(Path(spec.origin).parent, "dht", "schema.py")

# 读取文件内容
with open(module_path, 'r') as file:
    lines = file.readlines()

# 修改特定行，添加注释符号
with open(module_path, 'w') as file:
    for line in lines:
        # 在 yield from super().__get_validators__() 行前加注释符号
        if 'yield from super().__get_validators__()' in line:
            file.write('# ' + line)
        else:
            file.write(line)

print("注释添加成功！")

def modify_hivemind_files_target():
    spec = importlib.util.find_spec("hivemind")
    if not spec or not spec.origin:
        raise RuntimeError("no package named hivemind found")

    hivemind_path = Path(spec.origin).parent
    print(f"hivemind installation path：{hivemind_path}")
    modifications = [
        {
            "file": hivemind_path / "optim" / "grad_scaler.py",
            "replacements": [
                {
                    "target_line": "from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state",
                    "replacement": "from torch.cuda.amp.grad_scaler import OptState",
                },
            ],
        },
    ]

    for mod in modifications:
        file_path = mod["file"]
        if not file_path.exists():
            print(f"file {file_path} does not exist, skip this one...")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for replacement in mod["replacements"]:
            content = re.sub(replacement["target_line"], replacement["replacement"], content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"File fixed：{file_path}")

def modify_hivemind_files_pattern():
    spec = importlib.util.find_spec("hivemind")
    if not spec or not spec.origin:
        raise RuntimeError("no package named hivemind found")

    hivemind_path = Path(spec.origin).parent
    print(f"hivemind installation path：{hivemind_path}")
    modifications = [
        {
            "file": hivemind_path / "optim" / "grad_scaler.py",
            "replacements": [
                {
                    "pattern": r"self\._per_optimizer_states\[opt_id\] = .*",
                    "replacement": "self._per_optimizer_states[opt_id] = 0",
                },
            ],
        },
        {
            "file": hivemind_path / "p2p" / "p2p_daemon_bindings" / "control.py",
            "replacements": [
                {
                    "pattern": r"DEFAULT_MAX_MSG_SIZE = .*",
                    "replacement": "DEFAULT_MAX_MSG_SIZE = 30000 * 4 * 1024**2",
                },
            ],
        },
        {
            "file": hivemind_path / "dht" / "schema.py",
            "replacements": [
                {
                    "pattern": r"^\s+yield\s+from\s+super\(\)\s*\.__get_validators__\(\)",
                    "replacement": "#",
                },
            ],
        },
    ]

    for mod in modifications:
        file_path = mod["file"]
        if not file_path.exists():
            print(f"file {file_path} does not exist, skip this one...")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for replacement in mod["replacements"]:
            content = re.sub(replacement["pattern"], replacement["replacement"], content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"File fixed：{file_path}")

if __name__ == "__main__":
    modify_hivemind_files_target()
    modify_hivemind_files_pattern()
    
