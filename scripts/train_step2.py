#!/usr/bin/env python3
"""Fresh Step2 training. Query input is always RGB/K; no smoke warm-start.

Older revisions support the 3000-step inspection stop. V4 runs continuously
through 18k under the configured numerical/gradient/systemic-EPE guard.
"""
import argparse
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
from src.step2_data import EpisodeDataset
from src.augmentation import augment_training_world
from src.losses.world import world_loss, surface_support_rate
from src.losses.mapping import supervise_mapping_observation
from src.losses.pose_consistency import pose_consistency_loss
from src.models.ellipsoid_pose import sample_read_jitter
from src.models.vggt_memory import VGGTSpatialMemory
from src.vis.step2 import dump_json, curves, unseen_visualizations
from src.vis.smoke import gate_statistics
from scripts.train_smoke import gradient_norm


def build_optimizer(model, config):
    opt = config['optimizer']
    groups, used = [], set()
    def add(parameters, lr, decay=.01):
        values = [p for p in parameters if p.requires_grad and id(p) not in used]
        used.update(id(p) for p in values)
        if values:
            groups.append(dict(params=values, lr=lr, weight_decay=decay))
    add([model.render_reader.gate, model.pe_reader.gate], opt['gate_lr'], 0)
    add([model.initial_pose.dustbin_logit], opt['dustbin_lr'], 0)
    add(model.initial_pose.dustbin_head.parameters(), opt['dustbin_head_lr'])
    add(model.initial_pose.parameters(), opt['bootstrap_lr'])
    add(model.refine_pose.residual_pose.parameters(), opt['refinement_geometry_lr'])
    add(model.refine_pose.parameters(), opt['pose_lr'])
    add(model.native.parameters(), opt['native_lr'])
    add(model.parameters(), opt['heads_lr'])
    optimizer = torch.optim.AdamW(groups)
    def schedule(step):
        warmup = opt['warmup_steps']
        if step < warmup:
            return (step+1)/warmup
        progress = min(1., (step-warmup)/max(1, config['steps']-warmup))
        return opt['end_lr_ratio']+(1-opt['end_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def criterion(output, target, config):
    return world_loss(output, target, camera_weight=config['camera_alignment_weight'],
        refinement_safety_weight=config['refinement_safety_weight'], anchor_weight=config['anchor_alignment_weight'],
        matching_weight=config['matching_weight'], depth_weight=config['patch_depth_weight'], relative_depth_weight=config.get('relative_depth_weight', 0.),
        anchored_camera_weight=config.get('anchored_camera_weight', 0.), anchored_relative_depth_weight=config.get('anchored_relative_depth_weight', 0.))


def intervention_box(memory, target):
    h, w = target.depth.shape
    valid = target.valid.nonzero()
    desired = valid.new_tensor([h//2, 3*w//4])
    pixel = valid[(valid-desired).float().square().sum(-1).argmin()]
    point = target.world_points[pixel[0], pixel[1]]
    nearest = memory.means[(memory.means-point).norm(dim=-1).argmin()].detach()
    center = .5*(nearest+point)
    radius = torch.maximum(point.new_tensor([.4, .5, .4]), (nearest-point).abs()*.5+.1)
    return center-radius, center+radius


def erasure_metrics(full, after, target, box):
    local = ((target.world_points >= box[0]) & (target.world_points <= box[1])).all(-1) & target.valid
    outside = ~local & target.valid
    before = (full['world_points']-target.world_points).norm(dim=-1)
    delta = (after['world_points']-target.world_points).norm(dim=-1)-before
    return dict(local_pixels=int(local.sum()), outside_pixels=int(outside.sum()),
        epe_increase_inside_m=float(delta[local].mean()) if local.any() else None,
        epe_increase_outside_m=float(delta[outside].mean()) if outside.any() else None,
        mean_abs_epe_change_inside_m=float(delta[local].abs().mean()) if local.any() else None,
        mean_abs_epe_change_outside_m=float(delta[outside].abs().mean()) if outside.any() else None,
        t0_change_m=float((full['T0'][:3, 3]-after['T0'][:3, 3]).norm()),
        box_lower=box[0].cpu().tolist(), box_upper=box[1].cpu().tolist())


@torch.no_grad()
def evaluate(model, dataset, config, folder, step, *, quick=False, interventions=False, visualize=False):
    model.eval()
    episodes = dataset.evaluation_episodes(max_per_scene=2 if quick else None)
    records, visualized = [], set()
    for i, spec in enumerate(episodes):
        episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device='cuda')
        query, target = episode['query']
        memory = model.build_memory(episode['mapping'])
        wanted_vis = visualize and spec['scene'].split('/')[0] in ('harvard_c5', 'harvard_c6', 'harvard_c11', 'harvard_robotics_lab') and spec['scene'] not in visualized
        full = model(query, memory, return_attention=wanted_vis)
        _, metrics = criterion(full, target, config)
        metrics['depth_confidence_mean'] = float(full['local_geometry_confidence'].mean())
        if 'camera_scale' in full:
            metrics.update(camera_scale=float(full['camera_scale']), depth_confidence_mean=float(full['local_geometry_confidence'].mean()))
        item = dict(**spec, **metrics, public_gaussians=len(memory), active_gaussians=len(full['attention']['indices']))
        if interventions:
            empty = model(query, memory.subset(slice(0, 0)))
            _, item['empty_same_weights'] = criterion(empty, target, config)
            other = next(s for s in episodes if s['scene'] != spec['scene'])
            mismatch_episode = dataset.episode(other['scene'], other['mapping_frame_ids'], other['query_frame_id'], device='cuda')
            mismatch_memory = model.build_memory(mismatch_episode['mapping'])
            mismatch = model(query, mismatch_memory)
            _, item['mismatched_same_weights'] = criterion(mismatch, target, config)
            box = intervention_box(memory, target)
            partial, mask = memory.erase_box(*box)
            erased = model(query, partial)
            _, item['erased_same_weights'] = criterion(erased, target, config)
            item['erasure'] = erasure_metrics(full, erased, target, box)
            item['erasure']['erased_count'] = int(mask.sum())
            if wanted_vis:
                vis_folder = folder/'vis'/f'step_{step:05d}'/spec['scene'].replace('/', '__')
                item['wall'], item['erasure_visualization'] = unseen_visualizations(
                    model, episode, memory, full, empty, erased, box, mask, vis_folder)
                visualized.add(spec['scene'])
                print('Unseen V1–V5:', vis_folder, flush=True)
            del empty, mismatch, mismatch_memory, mismatch_episode, erased, partial
        records.append(item)
        if (i+1) % 25 == 0:
            print('Evaluation', step, i+1, '/', len(episodes), flush=True)
        del full, memory, episode
    aggregates = []
    for split in ('cross_scene_val', 'same_scene_heldout'):
        rows = [r for r in records if r['split']==split and r['surface_support_rate'] >= .05]
        blind = [r for r in records if r['split']==split and r['surface_support_rate'] < .05]
        if not rows:
            raise RuntimeError('No covisible validation episodes')
        keys = ('point_epe_m', 'coarse_point_epe_m', 'rotation0_deg', 'rotation1_deg', 'translation0_m', 'translation1_m', 'camera_epe_m', 'surface_support_rate')
        aggregate = dict(step=step, scope='quick' if quick else 'full', group=split, queries=len(rows), blind_queries=len(blind),
                         **{k:float(np.mean([r[k] for r in rows])) for k in keys})
        if interventions:
            aggregate.update(empty_epe_m=float(np.mean([r['empty_same_weights']['point_epe_m'] for r in rows])),
                mismatched_epe_m=float(np.mean([r['mismatched_same_weights']['point_epe_m'] for r in rows])))
        aggregates.append(aggregate)
    result = dict(step=step, evaluation_gauge='first mapping camera, G=I', query_inputs=['rgb', 'K'],
                  aggregates=aggregates, episodes=records)
    dump_json(folder/f'evaluation_{step:05d}_{"quick" if quick else "full"}.json', result)
    print('Validation:', json.dumps(aggregates), flush=True)
    model.train()
    return aggregates


def save_checkpoint(model, optimizer, scheduler, config, folder, step, weights):
    checkpoint = dict(step=step, config=config, weight_source=weights, initialization='fresh_official_vggt',
        trainable={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    path = folder/'training_state.pt'
    torch.save(checkpoint, folder/'training_state.tmp')
    (folder/'training_state.tmp').replace(path)
    if step in config.get('checkpoint_milestones', (config['midpoint_steps'], config['steps'])):
        snapshot = folder/f'training_state_{step:05d}.pt'
        if not snapshot.exists():
            # Zero-copy milestone: later atomic latest-checkpoint replacement
            # preserves this inode, including model/optimizer/RNG at the gate.
            snapshot.hardlink_to(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/step2.yaml')
    parser.add_argument('--run', required=True)
    parser.add_argument('--until', type=int, default=3000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', type=int, default=0)
    args = parser.parse_args()
    with (PROJECT_ROOT/args.config).open() as f:
        config = yaml.safe_load(f)
    if config.get('initialize_from') or config['initialization'] != 'official_vggt_only_no_smoke_checkpoint':
        parser.error('Step2 must initialize from official VGGT, never smoke checkpoints')
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results') or (folder.exists() and not args.resume):
        parser.error('Use a new project results directory, or explicitly resume its own Step2 state')
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):
        parser.error('Use run_on_available_gpu.py for live resource selection')
    subprocess.run([sys.executable, '-B', '-m', 'pytest', 'tests', '-q'], cwd=PROJECT_ROOT,
                   env=dict(os.environ, REQUIRE_GSPLAT='1'), check=True)
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    folder.mkdir(exist_ok=args.resume)
    (folder/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    dataset = EpisodeDataset(PROJECT_ROOT/config['split'], config['image_hw'])
    runtime = load_config(); weights = runtime['assets']['vggt_weights']['path']
    print('FRESH official VGGT/DINOv2/DPT only:', weights, flush=True)
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], weights, revision=config['architecture_revision'],
                              receive_layer=config['receive_layer'], pose_jitter_probability=0.).cuda().train()
    optimizer, scheduler = build_optimizer(model, config)
    start = 0
    if args.resume:
        previous = load_checkpoint_readonly(folder/'training_state.pt')
        if previous['config'] != config or previous.get('initialization') != 'fresh_official_vggt':
            raise ValueError('Resume is restricted to this exact fresh Step2 run/config')
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in previous['trainable'].items(): parameters[name].copy_(value)
        optimizer.load_state_dict(previous['optimizer']); scheduler.load_state_dict(previous['scheduler'])
        random.setstate(previous['python_rng']); torch.set_rng_state(previous['torch_rng']); torch.cuda.set_rng_state_all(previous['cuda_rng'])
        start = previous['step']; del previous
    print('Eligible training queries:', len(dataset.training_queries), 'trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    records = [json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()] if args.resume else []
    validation = [json.loads(l) for l in (folder/'validation.jsonl').read_text().splitlines()] if args.resume else []
    if not args.profile and not args.resume:
        save_checkpoint(model, optimizer, scheduler, config, folder, 0, weights) if config.get('continuous_training') else None
        validation += evaluate(model, dataset, config, folder, 0)
        (folder/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in validation))
    modules = dict(writer=model.writer, o5=model.render_reader, o6=model.pe_reader, dpt=model.native.point_head,
                   pose0=model.initial_pose, pose1=model.refine_pose, reception=model.native.aggregator.global_blocks[4],
                   downstream=model.native.aggregator.global_blocks[11])
    started = time.monotonic(); timings = []
    until = start+args.profile if args.profile else args.until
    with (folder/'metrics.jsonl').open('a' if args.resume else 'w', buffering=1) as log:
        for step in range(start, until):
            tic = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            episode = dataset.sample(device='cuda'); query, target = episode['query']
            mapping, target, _ = augment_training_world(episode['mapping'], target)
            optimizer.zero_grad(set_to_none=True)
            memory = model.build_memory(mapping); memory.retain_grad()
            read_jitter = sample_read_jitter('cuda', include_zero=config.get('architecture_revision')=='step2_v4') if random.random() < config['read_jitter_probability'] and surface_support_rate(memory.means, memory.covars, target) >= .05 else None
            prediction = model(query, memory, read_jitter=read_jitter)
            loss, metrics = criterion(prediction, target, config)
            metrics['depth_confidence_mean'] = float(prediction['local_geometry_confidence'].detach().mean())
            if 'camera_scale' in prediction:
                metrics.update(camera_scale=float(prediction['camera_scale'].detach()), depth_confidence_mean=float(prediction['local_geometry_confidence'].detach().mean()))
            lm = prediction['lm_energy_history']
            metrics.update(lm_energy_initial=float(lm[0]), lm_energy_final=float(lm[-1]), lm_energy_nonincreasing=bool((torch.diff(lm) <= 1e-6).all()))
            if config['mapping_self_observation']:
                map_loss, map_metrics = supervise_mapping_observation(model, memory, mapping[step % len(mapping)])
                loss = loss+map_loss; metrics.update(map_metrics); del map_loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            metrics['loss'] = float(loss.detach()); loss.backward()
            if config['paired_pose_consistency'] and step % 2 == 0 and read_jitter is None and metrics['world_supervision_enabled']:
                teacher = {k:prediction[k].detach() for k in ('T0', 'T1')}
                index = torch.multinomial(prediction['bootstrap_attention'].detach().mean(0)+.001, 1)[0]
                center = memory.means[index].detach(); radius = random.uniform(*config['paired_erasure_radius_m'])
                partial = model.build_memory(mapping); partial, erased = partial.erase_box(center-radius, center+radius)
                if len(partial) >= .5*len(memory) and erased.any():
                    after = model(query, partial)
                    consistency, extra = pose_consistency_loss(after, teacher, target, erased_reference=(memory.means, memory.covars, erased), matching_weight=config['matching_weight'])
                    if not torch.isfinite(consistency): raise RuntimeError('Nonfinite paired consistency')
                    consistency.backward(); metrics.update(extra); metrics['loss'] += float(consistency.detach())
                    del after, consistency
                del partial
            norms = {name:gradient_norm(module) for name,module in modules.items()}
            state_norms = {n:float(v.grad.norm()) if v.grad is not None else 0. for n,v in zip(('mu', 'Sigma', 'alpha', 'z'), (memory.means,memory.covars,memory.alpha,memory.features))}
            if not all(math.isfinite(x) for x in [*norms.values(), *state_norms.values()]): raise RuntimeError('Nonfinite gradients')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['optimizer']['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); torch.cuda.synchronize(); seconds = time.monotonic()-tic
            metrics.update(step=step+1, scene=episode['name'], query_frame_id=episode['query_frame_id'], mapping_frame_ids=episode['mapping_frame_ids'],
                public_gaussians=len(memory), active_gaussians=len(prediction['attention']['indices']),
                o5_gate=gate_statistics(model.render_reader.gate), o6_gate=gate_statistics(model.pe_reader.gate),
                module_grad_norms=norms, memory_grad_norms=state_norms,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                step_seconds=seconds)
            log.write(json.dumps(metrics, allow_nan=False)+'\n'); records.append(metrics); timings.append(seconds)
            if step < 3 or (step+1) % config['log_every'] == 0 or args.profile:
                print(json.dumps({k:metrics[k] for k in ('step','scene','loss','point_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m','peak_reserved_gib','step_seconds')}), flush=True)
            del loss, prediction, memory, episode, mapping, target
            if not args.profile and ((step+1) % config['checkpoint_every']==0 or step+1 == until):
                save_checkpoint(model,optimizer,scheduler,config,folder,step+1,weights)
            if not args.profile and ((step+1) % config['quick_eval_every']==0 or step+1 == until):
                full = step+1 in config['full_eval_steps'] or step+1 == until
                validation += evaluate(model,dataset,config,folder,step+1,quick=not full,
                    interventions=full and step+1 >= config['midpoint_steps'], visualize=full and (config.get('continuous_training', False) or step+1 == until))
                (folder/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in validation))
                curves(folder/'vis', records, validation)
                if config.get('continuous_training') and step+1 >= config['midpoint_steps'] and full:
                    current = [r for r in validation if r['step']==step+1 and r['group']=='cross_scene_val'][-1]
                    if current['point_epe_m'] > config['systemic_epe_limit_m']:
                        raise RuntimeError('Systemic cross-scene EPE collapse above configured limit: '+str(current['point_epe_m']))
            if step > 50 and (step+1) % 100 == 0:
                for name in ('writer','o5','o6','downstream'):
                    if not any(r['module_grad_norms'][name] > 0 for r in records[-50:]): raise RuntimeError('Persistent disconnected '+name)
    summary = dict(completed_steps=until, profile=bool(args.profile), initialization='fresh_official_vggt', image_hw=config['image_hw'],
        final_loss_mean=float(np.mean([r['loss'] for r in records[-min(100,len(records)):]])),
        median_step_seconds=float(np.median(timings[-min(20,len(timings)):])),
        peak_reserved_gib=max(r['peak_reserved_gib'] for r in records), elapsed_minutes=round((time.monotonic()-started)/60,1))
    dump_json(folder/'summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
