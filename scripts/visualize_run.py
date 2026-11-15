#!/usr/bin/env python3
"""Regenerate V1–V5 from this project's trained checkpoint, without retraining."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_checkpoint_readonly, load_config
configure_local_storage()

import torch
from scripts.train_smoke import load_scene_samples
from src.models.vggt_memory import VGGTSpatialMemory
from src.vis.smoke import deliver_visualizations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="Existing run name under this project's results")
    args = parser.parse_args()
    run = (PROJECT_ROOT/"results"/args.run).resolve()
    if not run.is_relative_to(PROJECT_ROOT/"results"):
        parser.error("Run must be in this project")
    if not os.environ.get("VGGT_GPU_MEMORY_FRACTION"):
        parser.error("Use run_on_available_gpu.py for a live resource check")
    torch.cuda.set_per_process_memory_fraction(float(os.environ["VGGT_GPU_MEMORY_FRACTION"]), 0)
    torch.set_num_threads(2)
    checkpoint = load_checkpoint_readonly(run/"trainable.pt")
    runtime = load_config()
    model = VGGTSpatialMemory(runtime["assets"]["official_vggt_source"]["path"],
                               checkpoint["weight_source"], receive_layer=checkpoint["config"]["receive_layer"],
                               revision=checkpoint['config'].get('architecture_revision', 'v2')).cuda().eval()
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    if expected != set(checkpoint["trainable"]):
        raise RuntimeError("Checkpoint does not match the model's trainable components")
    with torch.no_grad():
        for name, value in checkpoint["trainable"].items():
            parameters[name].copy_(value)
    records = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    samples = load_scene_samples(checkpoint["config"], "cuda")
    deliver_visualizations(model, samples, run, records)


if __name__ == "__main__":
    main()
