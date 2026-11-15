#!/usr/bin/env python3
"""Real three-scene VGGT overfit; no synthetic/untrained-backbone fallback."""

import argparse
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config
configure_local_storage()

import numpy as np
import torch
import yaml
from src.data import QueryInput, Sun3DScene, Supervision
from src.augmentation import augment_training_world
from src.losses.world import world_loss
from src.models.vggt_memory import VGGTSpatialMemory


def move(value, device):
    return type(value)(**{field.name: getattr(value, field.name).to(device) for field in fields(value)})


def load_scene_samples(config, device):
    paths = load_config()["datasets"]["sun3d"]["train_scenes"]
    result = []
    for path in paths:
        scene = Sun3DScene(path)
        mapping = [tuple(move(x, device) for x in scene.load(i, resize_hw=config["image_hw"]))
                   for i in config["mapping_indices"]]
        queries = [tuple(move(x, device) for x in scene.load(i, resize_hw=config["image_hw"]))
                   for i in config["query_indices"]]
        heldout = [tuple(move(x, device) for x in scene.load(i, resize_hw=config["image_hw"]))
                   for i in config["heldout_query_indices"]]
        result.append({"path": path, "name": scene.path.parent.name, "mapping": mapping,
                       "queries": queries, "heldout": heldout,
                       "mapping_frame_ids": [scene.frame_ids[i] for i in config["mapping_indices"]],
                       "query_frame_ids": [scene.frame_ids[i] for i in config["query_indices"]]})
    return result


def gradient_norm(module):
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="smoke_seed17")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--diagnostic", action="store_true", help="Short integration check, never a completed smoke")
    parser.add_argument("--config", default="configs/smoke_v2.yaml")
    args = parser.parse_args()
    with (PROJECT_ROOT/args.config).open() as handle:
        config = yaml.safe_load(handle)
    steps = args.steps if args.steps is not None else config["steps"]
    if not args.diagnostic and not 800 <= steps <= 1500:
        parser.error("A formal smoke requires 800–1500 steps")
    output_dir = (PROJECT_ROOT/"results"/args.run).resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT/"results") or output_dir.exists():
        parser.error("Use a new run directory under results; existing work is never overwritten")
    runtime = load_config()
    weights = runtime["assets"]["vggt_weights"]["path"]
    if not weights or not Path(weights).is_file():
        raise SystemExit("Missing official VGGT weights; no training launched")
    if not os.environ.get("VGGT_GPU_MEMORY_FRACTION"):
        raise SystemExit("Launch through scripts/run_on_available_gpu.py for a live resource check")
    # GPU was selected by the parent launcher. Strict real geometry precedes all training.
    env = dict(os.environ, REQUIRE_GSPLAT="1")
    subprocess.run([sys.executable, "-B", "-m", "pytest", "tests", "-q"],
                   cwd=PROJECT_ROOT, env=env, check=True)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.set_per_process_memory_fraction(float(os.environ["VGGT_GPU_MEMORY_FRACTION"]), 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    output_dir.mkdir(parents=True)
    (output_dir/"config.yaml").write_text(yaml.safe_dump(dict(config, steps=steps, diagnostic=args.diagnostic), sort_keys=False))
    print("Loading complete official VGGT/DINOv2/DPT checkpoint (read-only)", flush=True)
    model = VGGTSpatialMemory(runtime["assets"]["official_vggt_source"]["path"], weights,
                               receive_layer=config["receive_layer"]).cuda().train()
    samples = load_scene_samples(config, "cuda")
    pose_params = list(model.initial_pose.parameters())+list(model.refine_pose.parameters())
    native_params = [p for p in model.native.parameters() if p.requires_grad]
    gate_params = [model.render_reader.gate, model.pe_reader.gate]
    other_params = [p for name, p in model.named_parameters() if p.requires_grad and
                    not name.startswith(("native.", "initial_pose.", "refine_pose.")) and
                    id(p) not in {id(g) for g in gate_params}]
    optim_config = config["optimizer"]
    groups = [{"params": other_params, "lr": optim_config["heads_lr"]},
              {"params": pose_params, "lr": optim_config["pose_lr"]},
              {"params": gate_params, "lr": optim_config["gate_lr"], "weight_decay": 0.},
              {"params": native_params, "lr": optim_config["native_lr"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/optim_config["warmup_steps"], 1.))
    modules = {"writer": model.writer, "o5": model.render_reader, "o6": model.pe_reader,
               "dpt": model.native.point_head, "pose0": model.initial_pose,
               "pose1": model.refine_pose, "reception": model.native.aggregator.global_blocks[config["receive_layer"]]}
    representatives = {name: max(module.parameters(), key=lambda p: p.numel()) for name, module in modules.items()}
    print("Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    started = time.monotonic()
    records = []
    with (output_dir/"metrics.jsonl").open("w", buffering=1) as logfile:
        for step in range(steps):
            scene_idx = step % 3
            scene = samples[scene_idx]
            query_idx = (step//3) % len(scene["queries"])
            query, target = scene["queries"][query_idx]
            optimizer.zero_grad(set_to_none=True)
            mapping, target, gauge = augment_training_world(scene["mapping"], target)
            memory = model.build_memory(mapping)
            memory.retain_grad()
            prediction = model(query, memory)
            loss, metrics = world_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}; training stopped")
            loss.backward()
            norms = {name: gradient_norm(module) for name, module in modules.items()}
            state_norms = {name: float(value.grad.norm()) if value.grad is not None else 0.
                           for name, value in zip(("mu", "Sigma", "alpha", "z"),
                                                  (memory.means, memory.covars, memory.alpha, memory.features))}
            if not all(math.isfinite(x) for x in [*norms.values(), *state_norms.values()]):
                raise RuntimeError(f"Nonfinite gradients at step {step}")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], optim_config["grad_clip"], error_if_nonfinite=True)
            before = {name: p.detach().flatten()[:128].clone() for name, p in representatives.items()}
            optimizer.step()
            updates = {name: float((p.detach().flatten()[:128]-before[name]).norm()) for name, p in representatives.items()}
            scheduler.step()
            metrics.update(step=step+1, scene=scene["name"], query_index=query_idx,
                           public_gaussians=len(memory), active_gaussians=len(prediction["attention"]["indices"]),
                           o5_gate=float(model.render_reader.gate.detach().tanh().abs().mean()), o6_gate=float(model.pe_reader.gate.detach().tanh().abs().mean()),
                           o5_gate_max_abs=float(model.render_reader.gate.detach().tanh().abs().max()),
                           o6_gate_max_abs=float(model.pe_reader.gate.detach().tanh().abs().max()),
                           world_gauge=gauge.detach().cpu().tolist(),
                           module_grad_norms=norms, memory_grad_norms=state_norms, sampled_parameter_update_norms=updates,
                           min_covariance_eigenvalue=float(torch.linalg.eigvalsh(memory.covars.detach()).min()),
                           alpha_min=float(memory.alpha.detach().min()), alpha_max=float(memory.alpha.detach().max()))
            records.append(metrics)
            logfile.write(json.dumps(metrics)+"\n")
            if step < 3 or (step+1) % config["log_every"] == 0:
                print(json.dumps({key: metrics[key] for key in ("step", "scene", "loss", "point_epe_m", "rotation0_deg", "rotation1_deg", "translation1_m", "o5_gate", "o6_gate")}), flush=True)
            if (step+1) % config["checkpoint_every"] == 0 or step+1 == steps:
                # Save only trainable state; frozen official backbone is loaded
                # from its read-only path on restoration, without copying 5 GB.
                trainable = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
                torch.save({"trainable": trainable, "step": step+1, "config": config,
                            "weight_source": weights}, output_dir/"trainable.pt")
            if step >= 30 and (step+1) % 50 == 0:
                recent = records[-20:]
                for branch in ("o5", "o6", "writer"):
                    if not any(record["module_grad_norms"][branch] > 0 for record in recent):
                        raise RuntimeError(f"Persistent disconnected {branch} after gate opening")
            del loss, prediction, memory
    summary = {"completed_steps": steps, "diagnostic_only": args.diagnostic,
               "initial_loss_mean": float(np.mean([r["loss"] for r in records[:min(30, steps)]])),
               "final_loss_mean": float(np.mean([r["loss"] for r in records[-min(30, steps):]])),
               "elapsed_minutes": round((time.monotonic()-started)/60, 1)}
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    if not args.diagnostic:
        from src.vis.smoke import deliver_visualizations
        model.eval()
        deliver_visualizations(model, samples, output_dir, records)


if __name__ == "__main__":
    main()
