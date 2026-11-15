#!/usr/bin/env bash
# Source from any working directory. Never repurpose HOME or CODEX_HOME.
vggt_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$vggt_project_root/.cache"
export TORCH_HOME="$vggt_project_root/.cache/torch"
export HF_HOME="$vggt_project_root/.cache/huggingface"
export TORCH_EXTENSIONS_DIR="$vggt_project_root/.cache/torch_extensions"
export TRITON_CACHE_DIR="$vggt_project_root/.cache/triton"
export CUDA_CACHE_PATH="$vggt_project_root/.cache/cuda"
export MPLCONFIGDIR="$vggt_project_root/.cache/matplotlib"
export TMPDIR="$vggt_project_root/.cache/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$vggt_project_root${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$vggt_project_root/.deps/bin:$PATH"
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$TMPDIR"
