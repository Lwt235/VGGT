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
import contextlib
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_checkpoint_readonly, load_config
configure_local_storage()
import numpy as np
import torch
import yaml
from src.streaming_data import StreamingDataset
from src.losses.world import world_loss
from src.streaming_distributed import canonical_delta, scientific_config, trajectory_rng, trajectory_seed, TrajectorySegment


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
    if config.get('streaming_revision') == 'v2':
        from src.streaming_v2 import StreamingEngineV2
        engine = StreamingEngineV2(model)
    else:
        engine = StreamingEngine(model)
    engine.to(device)
    if checkpoint is not None:
        state = load_checkpoint_readonly(checkpoint) if not isinstance(checkpoint, dict) else checkpoint
        if scientific_config(state['config']) != scientific_config(config):
            raise ValueError('Step3 configuration changed at checkpoint restore')
        load_spatial_delta(engine, state['trainable'])
    return engine


def load_spatial_delta(engine, delta):
    delta = canonical_delta(delta)
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


def save_checkpoint(engine, optimizer, scheduler, config, folder, step, rank_rng=None):
    require_checkpoint_space(folder, engine)
    checkpoint = dict(step=step, config=config,
        initialization='step2_full_18000_spatial_adaptation', base_checkpoint=config['base_checkpoint'],
        trainable={n: p.detach().cpu() for n, p in engine.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=[torch.cuda.get_rng_state()], rank_rng=rank_rng,
        distributed_runtime=dict(world_size=int(os.environ.get('WORLD_SIZE','1')),
            trajectory_seed='seed + rank * 10007 + zero_based_step', find_unused_parameters=True))
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
    if scientific_config(checkpoint['config']) != scientific_config(config):
        raise ValueError('Step3 resume requires identical configuration')
    load_spatial_delta(engine, checkpoint['trainable'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    rank = int(os.environ.get('RANK','0'))
    per_rank = checkpoint.get('rank_rng')
    if per_rank and len(per_rank) == int(os.environ.get('WORLD_SIZE','1')):
        rng = per_rank[rank]
        random.setstate(rng['python_rng'])
        torch.set_rng_state(rng['torch_rng'])
        torch.cuda.set_rng_state(rng['cuda_rng'])
    else:
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        saved = checkpoint['cuda_rng']
        torch.cuda.set_rng_state(saved[min(torch.cuda.current_device(), len(saved)-1)])
    return checkpoint['step']


def reconcile_metrics_for_resume(folder, completed_step):
    """Replay only updates after the checkpoint; keep their old logs separate.

    A crash may leave flushed metrics ahead of atomic latest. Those updates
    will be recomputed from restored optimizer/RNG state and must not appear
    twice in the canonical convergence curve. Interrupted metrics remain as
    ordinary diagnostic logs, never as checkpoints or model inputs.
    """
    path = folder/'metrics.jsonl'
    if not path.exists():
        if completed_step:
            raise ValueError('Saved streaming state has no corresponding metric prefix')
        return
    prefix, interrupted = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            interrupted.append(line)
            continue
        if record['step'] <= completed_step:
            prefix.append((record['step'], line))
        else:
            interrupted.append(line)
    if [step for step, _ in prefix] != list(range(1, completed_step+1)):
        raise ValueError('Checkpoint and streaming metric prefix do not agree')
    if interrupted:
        with (folder/'metrics_interrupted.jsonl').open('a') as handle:
            handle.writelines(line if line.endswith('\n') else line+'\n' for line in interrupted)
        temporary = folder/'metrics.jsonl.tmp'
        temporary.write_text(''.join(line if line.endswith('\n') else line+'\n' for _,line in prefix))
        temporary.replace(path)


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
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank, local_rank = int(os.environ.get('RANK', '0')), int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed and config.get('streaming_revision') != 'v2':
        parser.error('DDP requires the detachable v2 3D streaming state')
    torch.cuda.set_device(local_rank)
    if distributed:
        import torch.distributed as dist
        from datetime import timedelta
        dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=torch.device(f'cuda:{local_rank}'))
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.set_per_process_memory_fraction(min(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 20*2**30/torch.cuda.get_device_properties(local_rank).total_memory), local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    if rank == 0:
        folder.mkdir(exist_ok=args.resume)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    dataset_type = StreamingDataset
    if config.get('streaming_revision') == 'v2':
        from src.streaming_data_v2 import StridedStreamingDataset
        dataset_type = StridedStreamingDataset
    dataset = dataset_type(PROJECT_ROOT/config['split'], config['image_hw'],
        config['minimum_frames'], config['maximum_frames'])
    engine = load_engine(config, device=f'cuda:{local_rank}').train()
    require_checkpoint_space(folder, engine, initial=not args.resume)
    optimizer, scheduler = build_optimizer(engine, config)
    start = 0
    if args.resume:
        checkpoint = load_checkpoint_readonly(folder/'training_state.pt')
        start = restore_optimizer_rng(engine, optimizer, scheduler, checkpoint, config)
        del checkpoint
        if rank == 0:
            reconcile_metrics_for_resume(folder, start)
    if rank == 0:
        (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    segment_runner = None
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        segment_runner = DistributedDataParallel(TrajectorySegment(engine, criterion, config),
            device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        dist.barrier(device_ids=[local_rank])
    from collections import Counter
    print('Step3 Full18k spatial adaptation; train clip starts:', dict(Counter(sid for sid,_ in dataset.training_examples)), flush=True)
    print('Trainable parameters:', sum(p.numel() for p in engine.parameters() if p.requires_grad),
          'estimated checkpoint MiB:', round(checkpoint_bytes(engine)/2**20), flush=True)
    prefixes = dict(writer='model.writer', o5='model.render_reader', o6='model.pe_reader',
                    pose0='model.initial_pose', pose1='model.refine_pose',
                    write_gate='write_gate', fusion='fusion')
    recent_gradients = []
    until = min(config['steps'], start+args.profile if args.profile else (args.until or config['steps']))
    started = time.monotonic()
    log_context = (folder/'metrics.jsonl').open('a' if args.resume else 'w', buffering=1) if rank == 0 else contextlib.nullcontext(None)
    with log_context as handle:
        for step in range(start, until):
            tic = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            if distributed:
                # New trajectory-level batch parallelism, reproducible across
                # resumes and independent between ranks; restored Adam/LR intact.
                torch.manual_seed(trajectory_seed(config['seed'], rank, step))
                torch.cuda.manual_seed(trajectory_seed(config['seed'], rank, step))
                stream = dataset.sample(trajectory_rng(config['seed'], rank, step), device=f'cuda:{local_rank}')
            else:
                stream = dataset.sample(device='cuda')
            observations = iter(stream['observations'])
            seed_query, seed_truth = next(observations)
            optimizer.zero_grad(set_to_none=True)
            if distributed:
                state = None
                seed = (seed_query, seed_truth)
                measurements, sizes, writes, active_counts = [], [], [], []
                total_queries = len(stream['frame_ids'])-1
                tbptt = config.get('tbptt_frames', total_queries)
                detached_losses = []
                # Same number of synchronized backward calls on every rank,
                # despite independent 12--16 frame trajectories and unused gates.
                segment_count = math.ceil((config['maximum_frames']-1)/tbptt)
                for segment_index in range(segment_count):
                    frames = list(itertools.islice(observations, tbptt))
                    segment, state, info = segment_runner(frames, state, seed, total_queries)
                    seed = None
                    if not torch.isfinite(segment):
                        raise RuntimeError('Nonfinite distributed streaming loss')
                    segment.backward()
                    detached_losses.append(segment.detach())
                    for destination, values in zip((measurements,sizes,writes,active_counts), info):
                        destination.extend(values)
                    del frames, segment, info
                loss = torch.stack(detached_losses).sum()
                del seed_query, seed_truth
            else:
                state = engine.seed(seed_query, seed_truth.depth, seed_truth.valid)
                del seed_query, seed_truth
                losses, measurements, sizes, writes, active_counts = [], [], [], [], []
                detached_losses = []
                total_queries = len(stream['frame_ids'])-1
                tbptt = config.get('tbptt_frames', total_queries)
                for frame_index, (query, target) in enumerate(observations, 1):
                    # This is the only inference call after the seed. Ground truth
                    # remains exclusively on the following criterion line.
                    output, state = engine(query, state)
                    loss, metrics = criterion(output, target, config)
                    losses.append(loss)
                    measurements.append(metrics)
                    sizes.append(len(state.memory))
                    writes.append(output['write'])
                    active_counts.append(output['active_count'])
                    if frame_index % tbptt == 0 or frame_index == total_queries:
                        segment = torch.stack(losses).sum()/total_queries
                        if not torch.isfinite(segment):
                            raise RuntimeError('Nonfinite streaming training loss')
                        segment.backward()
                        detached_losses.extend(x.detach() for x in losses)
                        losses.clear()
                        if config.get('streaming_revision') == 'v2':
                            from src.streaming_v2 import detach_state
                            state = detach_state(state)
                        del segment
                loss = torch.stack(detached_losses).mean()
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
                tbptt_frames=tbptt, source_indices=stream.get('source_indices'),
                strides=stream.get('strides'), seed_surface_overlap=stream.get('seed_surface_overlap'),
                **{key:float(np.mean([r[key] for r in measurements])) for key in keys},
                public_gaussians=sizes[-1], public_gaussians_per_frame=sizes, module_grad_norms=norms,
                active_gaussians_per_frame=active_counts,
                mean_write_confidence=float(np.mean([r['mean_confidence'] for r in writes])),
                mean_frame_write_quality=float(np.mean([r['frame_quality'] for r in writes])),
                mean_written_candidates=float(np.mean([r['candidates'] for r in writes])),
                write_checks=writes if config.get('streaming_revision') == 'v2' else None,
                mean_observation_support=float(state.support.detach().mean()) if len(state.support) else 0.,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                step_seconds=time.monotonic()-tic)
            if distributed:
                local_summary = {key:record[key] for key in ('scene','mode','frame_ids','frames','source_indices','strides','seed_surface_overlap',
                    'loss',*keys,'peak_allocated_gib','peak_reserved_gib')}
                gathered = [None]*world_size
                dist.all_gather_object(gathered, local_summary)
                record['rank_trajectories'] = gathered
                record['distributed_world_size'] = world_size
                for key in ('loss', *keys):
                    record[key] = float(np.mean([row[key] for row in gathered]))
                for key in ('peak_allocated_gib','peak_reserved_gib'):
                    record[key] = max(row[key] for row in gathered)
            if handle is not None:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
            recent_gradients.append(norms); recent_gradients = recent_gradients[-50:]
            if rank == 0 and (step < 3 or (step+1) % config['log_every'] == 0 or args.profile):
                print(json.dumps(record), flush=True)
            del state, loss, measurements, observations, stream
            if not args.profile and ((step+1) % config['checkpoint_every'] == 0 or step+1 == until):
                rank_rng = None
                if distributed:
                    rank_rng = [None]*world_size
                    dist.all_gather_object(rank_rng, dict(python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
                if rank == 0:
                    save_checkpoint(engine, optimizer, scheduler, config, folder, step+1, rank_rng=rank_rng)
                if distributed:
                    dist.barrier(device_ids=[local_rank])
            if len(recent_gradients) == 50 and (step+1) % 100 == 0:
                for name in ('writer', 'o5', 'o6', 'write_gate'):
                    if not any(row[name] > 0 for row in recent_gradients):
                        raise RuntimeError('Persistent disconnected streaming module: '+name)
    summary = dict(completed_steps=until, profile=bool(args.profile), elapsed_minutes=round((time.monotonic()-started)/60, 1),
        train_only=True, no_post_alignment=True, checkpoint_policy=config['checkpoint_policy'])
    summary['distributed_world_size'] = world_size
    if rank == 0:
        (folder/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
