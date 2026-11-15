"""Runtime storage policy: shared assets are opened only for reading."""

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_local_storage():
    sys.dont_write_bytecode = True
    dependencies = PROJECT_ROOT / ".deps"
    if dependencies.is_dir() and str(dependencies) not in sys.path:
        sys.path.append(str(dependencies))  # prefer existing environment packages
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable, relative in {
        "XDG_CACHE_HOME": ".cache",
        "TORCH_HOME": ".cache/torch",
        "HF_HOME": ".cache/huggingface",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions",
        "TRITON_CACHE_DIR": ".cache/triton",
        "CUDA_CACHE_PATH": ".cache/cuda",
        "TMPDIR": ".cache/tmp",
    }.items():
        path = PROJECT_ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_config():
    import yaml
    with (PROJECT_ROOT / "configs/runtime_paths.yaml").open("r") as handle:
        return yaml.safe_load(handle)


def load_checkpoint_readonly(path):
    """Never torch.hub/from_pretrained/download or write next to shared files."""
    import torch
    if path is None:
        raise FileNotFoundError("Weight path is missing in configs/runtime_paths.yaml")
    asset = Path(path).resolve(strict=True)
    if asset.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(asset), device="cpu")
    with asset.open("rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=True)
