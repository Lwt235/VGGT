#!/usr/bin/env python3
"""Read-only asset/dependency probe, optional tiny host GPU operation.

This is infrastructure validation, not the three-scene overfit experiment.
"""

import argparse
import importlib
import json
from pathlib import Path
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", default="results/infrastructure/runtime_probe.json")
    args = parser.parse_args()
    configure_local_storage()
    config = load_config()
    result = {"python": sys.executable, "dependencies": {}, "weights": {}, "gpu": {"tested": False}}
    for module in ["torch", "numpy", "pytest", "yaml", "PIL", "matplotlib", "gsplat",
                   "einops", "safetensors", "huggingface_hub"]:
        try:
            loaded = importlib.import_module(module)
            result["dependencies"][module] = {"imported": True, "version": getattr(loaded, "__version__", None)}
        except Exception as exc:
            result["dependencies"][module] = {"imported": False, "error": str(exc)}
    for name in ["vggt_weights", "dinov2_weights"]:
        path = config["assets"][name]["path"]
        result["weights"][name] = {"path": path, "found": bool(path and Path(path).is_file()), "load_tested": False}
    if args.gpu:
        # Check live resources/processes immediately before creating a CUDA context.
        command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.used", "--format=csv,noheader,nounits"]
        snapshot = subprocess.check_output(command, text=True)
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
        result["gpu"]["resource_snapshot"] = snapshot.strip().splitlines()
        result["gpu"]["existing_processes"] = processes.strip().splitlines()
        candidates = [line.split(", ") for line in snapshot.strip().splitlines()]
        chosen = max(candidates, key=lambda row: int(row[3]))
        if int(chosen[3]) < 8192:
            result["gpu"]["blocked"] = "Less than 8 GiB free; no GPU work launched"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = chosen[1]
            import torch
            torch.set_num_threads(2)
            torch.cuda.set_per_process_memory_fraction(.08, 0)
            a = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256
            expected = a @ a.T
            actual = (a.cuda() @ a.cuda().T).cpu()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            result["gpu"].update(tested=True, matmul_verified=True, physical_index=chosen[0],
                                  name=torch.cuda.get_device_name(), cuda_build=torch.version.cuda,
                                  memory_cap_fraction=.08)
    target = (PROJECT_ROOT / args.output).resolve()
    if not target.is_relative_to(PROJECT_ROOT / "results"):
        raise ValueError("Probe output must be under project/results")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "gpu"}, indent=2))
    print("GPU:", {k: v for k, v in result["gpu"].items() if k not in ("resource_snapshot", "existing_processes")})


if __name__ == "__main__":
    main()
