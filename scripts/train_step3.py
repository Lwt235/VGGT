#!/usr/bin/env python3
"""Train differentiable predicted-geometry streaming without GT write-back.

The immutable Full18k checkpoint supplies all adapted native parameters. Step3
checkpoints contain only spatial-module updates and optimizer/RNG state. This
keeps two simultaneous atomic-save inodes within the available root disk space.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_checkpoint_readonly, load_config
configure_local_storage()
import numpy as np
import torch
import yaml
from src.streaming_data import StreamingDataset
from src.losses.world import world_loss


def load_engine(config, checkpoint=None, device='cuda'):
    """Restore immutable Full18k base first, then optional Step3 delta."""
    from src.models.vggt_memory import VGGTSpatialMemory
    from src.streaming import StreamingEngine
    runtime = load_config()
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'],
        runtime['assets']['vggt_weights']['path'], receive_layer=config['receive_layer'],
        revision=config['architecture_revision'], pose_jitter_probability=0.)
    base_path = (PROJECT_ROOT/config['base_checkpoint']).resolve(strict=True)
    base = load_checkpoint_readonly(base_path)
    if base['step'] != 18000:
        raise ValueError('Step3 must warm-start the completed Full18k model')
    incompatible = model.load_state_dict(base['trainable'], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError('Full18k parameter mismatch: '+str(incompatible.unexpected_keys))
    del base
    # Frozen native layers remain in the autograd input path: spatial readers
    # and writer receive end-to-end DPT geometry gradients through all layers.
    model.native.requires_grad_(False)
    # StreamingMomentFusion is a separate stateful moment update module; the
    # offline mapping fusion is only its immutable Full18k initialization.
    model.fusion.requires_grad_(False)
    engine = StreamingEngine(model)
    engine.to(device)
    if checkpoint is not None:
        state = load_checkpoint_readonly(checkpoint) if not isinstance(checkpoint, dict) else checkpoint
        if state['config'] != config:
            raise ValueError('Step3 configuration changed at checkpoint restore')
        load_spatial_delta(engine, state['trainable'])
    return engine


def load_spatial_delta(engine, delta):
    parameters = {name: p for name, p in engine.named_parameters() if p.requires_grad}
    if parameters.keys() != delta.keys():
        raise ValueError('Step3 trainable parameter names changed')
    with torch.no_grad():
        for name, value in delta.items():
            if value.shape != parameters[name].shape:
                raise ValueError('Step3 trainable parameter shape changed: '+name)
            parameters[name].copy_(value)


def build_optimizer(engine, config):
    opt = config['optimizer']
    grouped = {}
    for name, parameter in engine.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'gate' in name:
            key = 'gate_lr'
        elif 'initial_pose' in name:
            key = 'bootstrap_lr'
        elif 'refine_pose' in name:
            key = 'pose_lr'
        else:
            key = 'heads_lr'
        grouped.setdefault(key, []).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=parameters, lr=opt[key],
        weight_decay=0. if key == 'gate_lr' else .01) for key, parameters in grouped.items()])

    def schedule(step):
        warmup = opt['warmup_steps']
        if step < warmup:
            return (step+1)/warmup
        ratio = min(1., (step-warmup)/max(1, config['steps']-warmup))
        return opt['end_lr_ratio']+(1-opt['end_lr_ratio'])*.5*(1+math.cos(math.pi*ratio))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def criterion(output, target, config):
    return world_loss(output, target,
        camera_weight=config['camera_alignment_weight'],
        refinement_safety_weight=config['refinement_safety_weight'],
        anchor_weight=config['anchor_alignment_weight'], matching_weight=config['matching_weight'],
        depth_weight=config['patch_depth_weight'], relative_depth_weight=config['relative_depth_weight'],
        anchored_camera_weight=config['anchored_camera_weight'],
        anchored_relative_depth_weight=config['anchored_relative_depth_weight'])


def checkpoint_bytes(engine):
    # Parameters + Adam's two FP32 moments, a small tensor metadata/RNG margin.
    return int(sum(p.numel()*p.element_size() for p in engine.parameters() if p.requires_grad)*3.1)+16*2**20


def require_checkpoint_space(folder, engine, *, initial=False):
    required = checkpoint_bytes(engine)*(2 if initial else 1)+64*2**20
    free = shutil.disk_usage(folder).free
    if free < required:
        raise RuntimeError(f'Disk guard: {free/2**20:.0f} MiB free, require {required/2**20:.0f} MiB for atomic spatial checkpoint')


def save_checkpoint(engine, optimizer, scheduler, config, folder, step):
    require_checkpoint_space(folder, engine)
    checkpoint = dict(step=step, config=config,
        initialization='step2_full_18000_spatial_adaptation', base_checkpoint=config['base_checkpoint'],
        trainable={n: p.detach().cpu() for n, p in engine.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    temporary, latest = folder/'training_state.tmp', folder/'training_state.pt'
    torch.save(checkpoint, temporary)
    temporary.replace(latest)
    if step == config['steps']:
        final = folder/f'training_state_{step:05d}.pt'
        if not final.exists():
            final.hardlink_to(latest)


def module_gradient_norm(engine, prefix):
    values = [p.grad.detach().float().square().sum() for name, p in engine.named_parameters()
              if name.startswith(prefix) and p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def restore_optimizer_rng(engine, optimizer, scheduler, checkpoint, config):
    if checkpoint['config'] != config:
        raise ValueError('Step3 resume requires identical configuration')
    load_spatial_delta(engine, checkpoint['trainable'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    random.setstate(checkpoint['python_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint['step']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/step3_streaming.yaml')
    parser.add_argument('--run', default='step3_streaming_seed17')
    parser.add_argument('--until', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT_ROOT/args.config).read_text())
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results') or (folder.exists() and not args.resume):
        parser.error('Use a new Step3 project directory, or resume its own state')
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):
        parser.error('Use run_on_available_gpu.py after live GPU/process checks')
    if not config['native_frozen']:
        parser.error('Step3 adaptation keeps native weights frozen for disk/activation budget')
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    folder.mkdir(exist_ok=args.resume)
    dataset = StreamingDataset(PROJECT_ROOT/config['split'], config['image_hw'],
        config['minimum_frames'], config['maximum_frames'])
    engine = load_engine(config).train()
    require_checkpoint_space(folder, engine, initial=not args.resume)
    optimizer, scheduler = build_optimizer(engine, config)
    start = 0
    if args.resume:
        checkpoint = load_checkpoint_readonly(folder/'training_state.pt')
        start = restore_optimizer_rng(engine, optimizer, scheduler, checkpoint, config)
        del checkpoint
    (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print('Step3 Full18k spatial adaptation; train clips:', {sid:len(w) for sid,w in dataset.training_windows.items()}, flush=True)
    print('Trainable parameters:', sum(p.numel() for p in engine.parameters() if p.requires_grad),
          'estimated checkpoint MiB:', round(checkpoint_bytes(engine)/2**20), flush=True)
    prefixes = dict(writer='model.writer', o5='model.render_reader', o6='model.pe_reader',
                    pose0='model.initial_pose', pose1='model.refine_pose',
                    write_gate='write_gate', fusion='fusion')
    recent_gradients = []
    until = min(config['steps'], start+args.profile if args.profile else (args.until or config['steps']))
    started = time.monotonic()
    with (folder/'metrics.jsonl').open('a' if args.resume else 'w', buffering=1) as handle:
        for step in range(start, until):
            tic = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            stream = dataset.sample(device='cuda')
            observations = iter(stream['observations'])
            seed_query, seed_truth = next(observations)
            optimizer.zero_grad(set_to_none=True)
            state = engine.seed(seed_query, seed_truth.depth, seed_truth.valid)
            del seed_query, seed_truth
            losses, measurements, sizes, writes, active_counts = [], [], [], [], []
            for query, target in observations:
                # This is the only inference call after the seed. Ground truth
                # remains exclusively on the following criterion line.
                output, state = engine(query, state)
                loss, metrics = criterion(output, target, config)
                losses.append(loss)
                measurements.append(metrics)
                sizes.append(len(state.memory))
                writes.append(output['write'])
                active_counts.append(output['active_count'])
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite streaming training loss')
            loss.backward()
            norms = {name:module_gradient_norm(engine, prefix) for name,prefix in prefixes.items()}
            if not all(math.isfinite(value) for value in norms.values()):
                bad = [n for n,p in engine.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                raise RuntimeError('Nonfinite streaming gradients: '+str(bad))
            torch.nn.utils.clip_grad_norm_([p for p in engine.parameters() if p.requires_grad],
                config['optimizer']['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); torch.cuda.synchronize()
            keys = ('point_epe_m','point_l1_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m',
                    'dustbin_mean','surface_support_rate','anchored_camera_epe_m')
            record = dict(step=step+1, scene=stream['scene'], mode=stream['mode'], frame_ids=stream['frame_ids'],
                frames=len(stream['frame_ids']), loss=float(loss.detach()),
                **{key:float(np.mean([r[key] for r in measurements])) for key in keys},
                public_gaussians=sizes[-1], public_gaussians_per_frame=sizes, module_grad_norms=norms,
                active_gaussians_per_frame=active_counts,
                mean_write_confidence=float(np.mean([r['mean_confidence'] for r in writes])),
                mean_frame_write_quality=float(np.mean([r['frame_quality'] for r in writes])),
                mean_written_candidates=float(np.mean([r['candidates'] for r in writes])),
                mean_observation_support=float(state.support.detach().mean()) if len(state.support) else 0.,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                step_seconds=time.monotonic()-tic)
            handle.write(json.dumps(record, allow_nan=False)+'\n')
            recent_gradients.append(norms); recent_gradients = recent_gradients[-50:]
            if step < 3 or (step+1) % config['log_every'] == 0 or args.profile:
                print(json.dumps(record), flush=True)
            del state, output, loss, losses, measurements, observations, stream, query, target
            if not args.profile and ((step+1) % config['checkpoint_every'] == 0 or step+1 == until):
                save_checkpoint(engine, optimizer, scheduler, config, folder, step+1)
            if len(recent_gradients) == 50 and (step+1) % 100 == 0:
                for name in ('writer', 'o5', 'o6', 'write_gate'):
                    if not any(row[name] > 0 for row in recent_gradients):
                        raise RuntimeError('Persistent disconnected streaming module: '+name)
    summary = dict(completed_steps=until, profile=bool(args.profile), elapsed_minutes=round((time.monotonic()-started)/60, 1),
        train_only=True, no_post_alignment=True, checkpoint_policy=config['checkpoint_policy'])
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
