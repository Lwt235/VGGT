#!/usr/bin/env python3
"""Bounded GPU launcher: inspect live resources, never stop another process."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-mib", type=int, default=12288)
    parser.add_argument("--memory-fraction", type=float, default=.08)
    parser.add_argument("--gpu", type=int, help="Optional physical GPU index; all live resource checks still apply")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.memory_fraction <= .8:
        parser.error("A command and a memory fraction in (0,.8] are required")
    configure_local_storage()
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(", ") for line in snapshot.strip().splitlines()]
    candidates = [row for row in rows if (args.gpu is None or int(row[0]) == args.gpu)
                  and int(row[2]) >= max(args.min_free_mib, int(row[4])*args.memory_fraction + 2048)]
    if not candidates:
        raise SystemExit("Insufficient free memory for the configured cap and 2 GiB reserve; no work launched")
    selected = max(candidates, key=lambda row: int(row[2]))
    print(json.dumps({"physical_gpu": selected[0], "free_mib": int(selected[2]),
                      "memory_fraction_cap": args.memory_fraction,
                      "existing_compute_processes": len(processes.strip().splitlines())}), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected[1]
    os.environ["VGGT_GPU_MEMORY_FRACTION"] = str(args.memory_fraction)
    os.environ["MAX_JOBS"] = "2"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["PATH"] = str(Path(__file__).resolve().parents[1] / ".deps/bin") + os.pathsep + os.environ["PATH"]
    # Child gets only this project context and the selected GPU visibility.
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
