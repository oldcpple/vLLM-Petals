import os
import re
from pathlib import Path
import importlib.util

def modify_hivemind_files():
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
                    "pattern": r"^from torch\.cuda\.amp\.grad_scaler import \w+",
                    "replacement": "from torch.cuda.amp.grad_scaler import OptState",
                },
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
                    "pattern": r"^yield from super",
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
    modify_hivemind_files()
