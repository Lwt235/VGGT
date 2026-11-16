#!/usr/bin/env python3
"""Fresh Step2b variants, paired to Full's episodes and SE(3) augmentation.

Only official weights or this run's exact checkpoint may initialize a model.
EPE is measured without a stopping threshold: branch removal may worsen EPE.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config, load_checkpoint_readonly
configure_local_storage()
import numpy as np
import torch
import yaml
from src.augmentation import transform_supervision
from src.evaluation import benchmark_observable
from src.step2_data import EpisodeDataset
from src.training_state import restore_training_state
from src.models.vggt_memory import VGGTSpatialMemory
from src.losses.mapping import supervise_mapping_observation
from src.losses.pose_consistency import pose_consistency_loss
from src.vis.ablations import ablation_visualizations
from src.vis.step2 import dump_json, curves
from src.vis.smoke import gate_statistics
from scripts.train_smoke import gradient_norm
from scripts.train_step2 import build_optimizer, criterion, save_checkpoint as common_save_checkpoint

VARIANTS = ('render_only', 'pe_only', 'isotropic_point')
SUMMARY_KEYS = ('point_epe_m', 'coarse_point_epe_m', 'rotation0_deg', 'rotation1_deg',
                'translation0_m', 'translation1_m', 'camera_epe_m', 'surface_support_rate')


def validate_protocol(config, source_config):
    if config.get('ablation') not in VARIANTS:
        raise ValueError('Unknown Step2b ablation')
    if config.get('initialize_from') or config['initialization'] != 'official_vggt_only_no_smoke_checkpoint':
        raise ValueError('Official fresh initialization required')
    for key in ('seed', 'split', 'image_hw', 'steps', 'architecture_revision', 'optimizer',
                'receive_layer', 'downstream_trainable_global_block',
                'camera_alignment_weight', 'patch_depth_weight', 'relative_depth_weight',
                'anchor_alignment_weight', 'matching_weight', 'refinement_safety_weight',
                'read_jitter_probability', 'paired_pose_consistency', 'paired_erasure_radius_m'):
        if config[key] != source_config[key]:
            raise ValueError('Full/ablation protocol mismatch: '+key)
    expected = 0. if config['ablation'] == 'pe_only' else None
    for key in ('anchored_camera_weight', 'anchored_relative_depth_weight'):
        if config[key] != (source_config[key] if expected is None else expected):
            raise ValueError('Invalid ablation loss: '+key)
    if config['mapping_self_observation'] != (config['ablation'] != 'pe_only'):
        raise ValueError('PE-only mapping supervision must be disabled')
    if config.get('stop_on_epe') is not False or config.get('systemic_epe_limit_m') is not None:
        raise ValueError('Ablations must not stop on EPE degradation')
    if config.get('full_eval_steps') != [0, 1500, 3000, 6000, 12000, 18000] or config.get('checkpoint_milestones') != [18000]:
        raise ValueError('Fixed full evaluation schedule and final-only checkpoint milestone required')


def validate_episode(spec, scenes):
    item = scenes[spec['scene']][1]
    allowed = set(item['train_frame_ids'])
    if (item['partition'] != 'train' or spec['query_frame_id'] not in allowed
            or not spec['mapping_frame_ids'] or not set(spec['mapping_frame_ids']) <= allowed
            or spec['query_frame_id'] in spec['mapping_frame_ids']):
        raise ValueError('Invalid Full training schedule entry')


def validate_replay(replay, count):
    for key in ('world_gauges', 'read_jitters'):
        matrix = replay[key]
        if not isinstance(matrix, torch.Tensor) or matrix.shape != (count, 4, 4) or not torch.isfinite(matrix).all():
            raise ValueError('Invalid augmentation replay: '+key)
        bottom = matrix.new_tensor([0., 0., 0., 1.]).expand(count, 4)
        rotation = matrix[:, :3, :3]
        identity = torch.eye(3, dtype=matrix.dtype).expand(count, 3, 3)
        if (not torch.allclose(matrix[:, 3], bottom, atol=2e-5, rtol=0)
                or not torch.allclose(rotation.transpose(-1, -2)@rotation, identity, atol=2e-5, rtol=0)
                or not torch.allclose(torch.linalg.det(rotation), torch.ones(count, dtype=matrix.dtype), atol=2e-5, rtol=0)):
            raise ValueError('Replay is not SE(3): '+key)
    flag = replay['read_jitter_applied']
    if not isinstance(flag, torch.Tensor) or flag.shape != (count,) or flag.dtype != torch.bool:
        raise ValueError('Invalid replay jitter flags')
    paired = replay['paired_requested']
    if not isinstance(paired, torch.Tensor) or paired.shape != (count,) or paired.dtype != torch.bool:
        raise ValueError('Invalid replay paired-request flags')
    radius = replay['paired_radius_m']
    if (radius.shape != (count,) or not torch.isfinite(radius).all() or (radius[~paired] != 0).any()
            or (radius[paired] < .6-1e-7).any() or (radius[paired] > .75+1e-7).any()
            or (paired & flag).any()):
        raise ValueError('Invalid paired erasure radius replay')


def replay_world(mapping, target, matrix):
    gauge = matrix.to(device=target.c2w.device, dtype=target.c2w.dtype)
    return [(query, transform_supervision(truth, gauge)) for query, truth in mapping], transform_supervision(target, gauge)


def replay_read_jitter(replay, step, target):
    # Full's recorded application flag already includes its >=5% support rule.
    # Rechecking learned ablation support here would break paired augmentation.
    return replay['read_jitters'][step].to(target.c2w) if bool(replay['read_jitter_applied'][step]) else None


def reader_statistics(reader):
    return 'disabled' if reader is None else gate_statistics(reader.gate)


def module_gradient_norms(modules):
    return {name: 'disabled' if module is None else gradient_norm(module) for name, module in modules.items()}


def check_connected(records, modules, *, window=50):
    for name, module in modules.items():
        if module is not None and not any(row['module_grad_norms'].get(name, 0.) > 0 for row in records[-window:]):
            raise RuntimeError('Persistent disconnected active module '+name)


def active_count(output):
    attention = output.get('attention')
    return len(attention['indices']) if attention is not None and 'indices' in attention else 0


def save_checkpoint(model, optimizer, scheduler, config, folder, step, weights):
    # All three variants share disk; serialize only checkpoint I/O, never updates.
    lock = PROJECT_ROOT/'results/step2b_protocol/checkpoint.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        common_save_checkpoint(model, optimizer, scheduler, config, folder, step, weights)
        fcntl.flock(handle, fcntl.LOCK_UN)


def resume_logs(folder, step):
    """Keep interrupted observations while replaying from the last exact state."""
    def read(name):
        path = folder/name
        original = path.read_text() if path.exists() else ''
        return path, original, [json.loads(line) for line in original.splitlines()]
    metrics_path, metrics_text, records = read('metrics.jsonl')
    validation_path, validation_text, validation = read('validation.jsonl')
    if len(records) < step or any(row['step'] != i+1 for i, row in enumerate(records)):
        raise ValueError('Metrics precede own checkpoint or contain an invalid step sequence')
    def trim(path, original, rows, retained):
        if len(rows) == len(retained):
            return
        laststep = max(row['step'] for row in rows)
        backup = path.with_name(path.stem+f'_interrupted_{laststep:05d}.jsonl')
        serial = 1
        while backup.exists():
            backup = path.with_name(path.stem+f'_interrupted_{laststep:05d}_{serial}.jsonl')
            serial += 1
        # Exclusive create preserves every previous interruption's observations.
        with backup.open('x') as handle:
            handle.write(original)
        path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in retained))
    retained_records = records[:step]
    retained_validation = [row for row in validation if row['step'] <= step]
    trim(metrics_path, metrics_text, records, retained_records)
    trim(validation_path, validation_text, validation, retained_validation)
    return retained_records, retained_validation


@torch.no_grad()
def evaluate(model, dataset, config, folder, step, *, quick=False, visualize=False, device='cuda'):
    model.eval()
    records, shown = [], set()
    scenes = {sid: item for sid, (_, item) in dataset.scenes.items()}
    for spec in dataset.evaluation_episodes(max_per_scene=2 if quick else None):
        episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device=device)
        query, target = episode['query']
        memory = model.build_memory(episode['mapping'])
        output = model(query, memory)
        _, metrics = criterion(output, target, config)
        metrics['depth_confidence_mean'] = float(output['local_geometry_confidence'].mean())
        item = dict(**spec, **metrics, benchmark_observable=benchmark_observable(spec, scenes),
                    public_gaussians=len(memory), active_gaussians=active_count(output))
        family = spec['scene'].split('/')[0]
        if visualize and family in ('harvard_c5', 'harvard_c6', 'harvard_c11', 'harvard_robotics_lab') and family not in shown:
            path = folder/'vis'/f'step_{step:05d}'/spec['scene'].replace('/', '__')
            item['wall'] = ablation_visualizations(episode, memory, output, path, isotropic=config['ablation']=='isotropic_point')
            shown.add(family)
            print('Ablation V1/V2:', path, flush=True)
        records.append(item)
        if len(records) % 25 == 0:
            print('Ablation evaluation', step, len(records), flush=True)
        del output, memory, episode
    aggregates = []
    for group in ('cross_scene_val', 'same_scene_heldout'):
        rows = [row for row in records if row['split']==group and row['benchmark_observable']]
        blind = [row for row in records if row['split']==group and not row['benchmark_observable']]
        if not rows:
            raise RuntimeError('No covisible validation episodes')
        aggregates.append(dict(step=step, scope='quick' if quick else 'full', group=group,
            queries=len(rows), blind_queries=len(blind), **{key:float(np.mean([row[key] for row in rows])) for key in SUMMARY_KEYS}))
    if not quick and [(row['queries'], row['blind_queries']) for row in aggregates] != [(45, 0), (79, 2)]:
        raise ValueError('Fixed 45+79 observable and 2 blind evaluation protocol changed')
    dump_json(folder/f'evaluation_{step:05d}_{"quick" if quick else "full"}.json', dict(
        step=step, ablation=config['ablation'], evaluation_gauge='first mapping camera, G=I',
        query_inputs=['rgb', 'K'], aggregates=aggregates, episodes=records))
    print('Ablation validation:', json.dumps(aggregates), flush=True)
    model.train()
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--until', type=int, default=18000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', type=int, default=0, help='Disposable run; no checkpoints/evaluations or warm start')
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT_ROOT/args.config).read_text())
    source = (PROJECT_ROOT/'results'/config['source_full_run']).resolve()
    if not source.is_relative_to(PROJECT_ROOT/'results'):
        parser.error('Full replay source must be inside project results')
    validate_protocol(config, yaml.safe_load((source/'config.yaml').read_text()))
    schedule = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    if len(schedule) != config['steps'] or any(row['step'] != i+1 for i, row in enumerate(schedule)):
        parser.error('Exactly the completed 18k Full episode sequence is required')
    replay = load_checkpoint_readonly(PROJECT_ROOT/config['augmentation_replay'])
    validate_replay(replay, config['steps'])
    if (replay['source_full_run'] != config['source_full_run'] or replay['seed'] != config['seed']
            or replay['steps'] != config['steps']):
        raise ValueError('Augmentation replay must belong to the exact completed Full protocol')
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results') or folder==source or (folder.exists() and not args.resume):
        parser.error('Use a new project results directory, or resume only its own ablation state')
    if args.profile and args.resume:
        parser.error('Profile must use a fresh disposable directory')
    if args.until > config['steps'] or args.until < 1 or args.profile < 0:
        parser.error('Invalid training budget')
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):
        parser.error('Use run_on_available_gpu.py for live resource selection')
    subprocess.run([sys.executable, '-B', '-m', 'pytest', 'tests', '-q'], cwd=PROJECT_ROOT,
                   env=dict(os.environ, REQUIRE_GSPLAT='1'), check=True)
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dataset = EpisodeDataset(PROJECT_ROOT/config['split'], config['image_hw'])
    for spec in schedule:
        validate_episode(spec, dataset.scenes)
    runtime = load_config(); weights = runtime['assets']['vggt_weights']['path']
    print('FRESH official VGGT only; ablation:', config['ablation'], 'weights:', weights, flush=True)
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], weights,
        revision=config['architecture_revision'], ablation=config['ablation'], receive_layer=config['receive_layer'],
        pose_jitter_probability=0.).cuda().train()
    optimizer, scheduler = build_optimizer(model, config)
    start = 0
    if args.resume:
        previous = load_checkpoint_readonly(folder/'training_state.pt')
        # Unlike the Full safety-threshold exception, ablation config is exact.
        if previous['config'] != config:
            raise ValueError('Resume requires its own unchanged ablation configuration')
        start = restore_training_state(model, optimizer, scheduler, previous, config)
        del previous
    until = start+args.profile if args.profile else args.until
    if until <= start or until > config['steps']:
        parser.error('Requested endpoint must follow own checkpoint and fit 18k replay')
    folder.mkdir(exist_ok=args.resume)
    (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    records, validation = resume_logs(folder, start) if args.resume else ([], [])
    if not args.profile and (not args.resume or (start == 0 and not validation)):
        save_checkpoint(model, optimizer, scheduler, config, folder, 0, weights)
        validation += evaluate(model, dataset, config, folder, 0)
        (folder/'validation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in validation))
    modules = dict(writer=model.writer, o5=model.render_reader, o6=model.pe_reader, dpt=model.native.point_head,
        pose0=model.initial_pose, pose1=model.refine_pose, reception=model.native.aggregator.global_blocks[4],
        downstream=model.native.aggregator.global_blocks[11])
    started = time.monotonic(); timings = []
    with (folder/'metrics.jsonl').open('a' if args.resume else 'w', buffering=1) as log:
        for step in range(start, until):
            tic = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            spec = schedule[step]
            episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device='cuda')
            query, target = episode['query']
            mapping, target = replay_world(episode['mapping'], target, replay['world_gauges'][step])
            read_jitter = replay_read_jitter(replay, step, target)
            optimizer.zero_grad(set_to_none=True)
            memory = model.build_memory(mapping); memory.retain_grad()
            prediction = model(query, memory, read_jitter=read_jitter)
            loss, metrics = criterion(prediction, target, config)
            metrics['depth_confidence_mean'] = float(prediction['local_geometry_confidence'].detach().mean())
            for prefix, key in (('lm', 'lm_energy_history'), ('refinement_lm', 'refinement_lm_energy_history')):
                if key in prediction:
                    history = prediction[key]
                    metrics.update({prefix+'_energy_initial':float(history[0]), prefix+'_energy_final':float(history[-1]),
                                    prefix+'_energy_nonincreasing':bool((torch.diff(history)<=1e-6).all())})
            if config['mapping_self_observation']:
                extra_loss, extra = supervise_mapping_observation(model, memory, mapping[step % len(mapping)])
                loss = loss+extra_loss; metrics.update(extra); del extra_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite ablation training loss')
            metrics['loss'] = float(loss.detach()); loss.backward()
            if config['paired_pose_consistency'] and bool(replay['paired_requested'][step]) and metrics['world_supervision_enabled']:
                teacher = {key:prediction[key].detach() for key in ('T0', 'T1')}
                index = torch.multinomial(prediction['bootstrap_attention'].detach().mean(0)+.001, 1)[0]
                center = memory.means[index].detach()
                radius = float(replay['paired_radius_m'][step])
                if not config['paired_erasure_radius_m'][0]-1e-7 <= radius <= config['paired_erasure_radius_m'][1]+1e-7:
                    raise ValueError('Paired replay radius outside Full protocol')
                partial = model.build_memory(mapping); partial, erased = partial.erase_box(center-radius, center+radius)
                if len(partial) >= .5*len(memory) and erased.any():
                    after = model(query, partial)
                    consistency, extra = pose_consistency_loss(after, teacher, target,
                        erased_reference=(memory.means, memory.covars, erased), matching_weight=config['matching_weight'])
                    if not torch.isfinite(consistency):
                        raise RuntimeError('Nonfinite ablation paired consistency')
                    consistency.backward(); metrics.update(extra); metrics['loss'] += float(consistency.detach())
                    del after, consistency
                del partial
            norms = module_gradient_norms(modules)
            state_norms = {name:float(value.grad.norm()) if value.grad is not None else 0.
                for name, value in zip(('mu', 'Sigma', 'alpha', 'z'), (memory.means, memory.covars, memory.alpha, memory.features))}
            if not all(math.isfinite(value) for value in [*[v for v in norms.values() if isinstance(v, (int, float))], *state_norms.values()]):
                dump_json(folder/f'nonfinite_gradient_{step+1:05d}.json', dict(
                    step=step+1, scene=spec['scene'], query_frame_id=spec['query_frame_id'],
                    mapping_frame_ids=spec['mapping_frame_ids'], loss=metrics['loss'],
                    module_norms={key:(str(value) if isinstance(value, (int, float)) and not math.isfinite(value) else value)
                                  for key, value in norms.items()},
                    memory_norms={key:(value if math.isfinite(value) else str(value)) for key, value in state_norms.items()},
                    nonfinite_parameters=[name for name, p in model.named_parameters()
                        if p.grad is not None and not torch.isfinite(p.grad).all()]))
                raise RuntimeError('Nonfinite ablation gradients')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['optimizer']['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); torch.cuda.synchronize(); seconds = time.monotonic()-tic
            metrics.update(step=step+1, ablation=config['ablation'], scene=spec['scene'], query_frame_id=spec['query_frame_id'],
                mapping_frame_ids=spec['mapping_frame_ids'], public_gaussians=len(memory), active_gaussians=active_count(prediction),
                read_jitter_applied=read_jitter is not None, o5_gate=reader_statistics(model.render_reader), o6_gate=reader_statistics(model.pe_reader),
                module_grad_norms=norms, memory_grad_norms=state_norms, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, step_seconds=seconds)
            log.write(json.dumps(metrics, allow_nan=False)+'\n'); records.append(metrics); timings.append(seconds)
            if step < 3 or (step+1) % config['log_every'] == 0 or args.profile:
                print(json.dumps({key:metrics[key] for key in ('step', 'scene', 'loss', 'point_epe_m', 'rotation1_deg', 'translation1_m', 'peak_reserved_gib', 'step_seconds')}), flush=True)
            del loss, prediction, memory, episode, mapping, target
            if step > 50 and (step+1) % 100 == 0:
                check_connected(records, modules)
            if not args.profile and ((step+1) % config['checkpoint_every']==0 or step+1 == until):
                save_checkpoint(model, optimizer, scheduler, config, folder, step+1, weights)
            if not args.profile and ((step+1) % config['quick_eval_every']==0 or step+1 == until):
                full = step+1 in config['full_eval_steps'] or step+1 == until
                validation += evaluate(model, dataset, config, folder, step+1, quick=not full,
                    visualize=step+1==config['steps'])
                (folder/'validation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in validation))
                curves(folder/'vis', records, validation)
    summary = dict(completed_steps=until, ablation=config['ablation'], profile=bool(args.profile),
        initialization='fresh_official_vggt', image_hw=config['image_hw'], stop_on_epe=False,
        final_loss_mean=float(np.mean([row['loss'] for row in records[-100:]])),
        median_step_seconds=float(np.median(timings[-20:])), peak_reserved_gib=max(row['peak_reserved_gib'] for row in records),
        elapsed_minutes=round((time.monotonic()-started)/60, 1))
    dump_json(folder/'summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
