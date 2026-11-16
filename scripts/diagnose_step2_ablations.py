#!/usr/bin/env python3
"""Live-launcher six-way diagnostics; all 18k source checkpoints stay read-only.

Produces 11-robotics-query pixel depth bins and Full-weight, unfused Single V2.
All outputs live under results/step2b_matrix_seed17. No query GT enters forward.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_checkpoint_readonly, load_config
configure_local_storage()
import numpy as np
import torch
import yaml
from scripts.compare_step2_ablations import (OUTPUT, TAGS, VARIANTS, SCENES, make_parser,
    roots_from_args, read_json, write_json, digest, identities, validate_episodes, validate_depth_bins)
from scripts.diagnose_step2_final import single_frame_unfused
from src.models.vggt_memory import VGGTSpatialMemory
from src.models.nomemory_vggt import VGGTNoMemory
from src.step2_data import EpisodeDataset

DEPTH_BINS = {'near_0.1_3m': (.1, 3.), 'mid_3_5m': (3., 5.), 'far_5_10m': (5., 10.)}
CACHE_NAMES = {'Full': 'full', 'No-Memory': 'no_memory', 'Single-map': 'single_map_depth', **VARIANTS}
DEPTH_TASKS = ('Full', 'No-Memory', 'Single-map-depth', *VARIANTS)


def accumulate_bins(output, target, bins):
    delta = output['world_points']-target.world_points
    world_epe = delta.norm(dim=-1)
    world_l1 = delta.abs().mean(dim=-1)
    raw = output['raw_camera_points'][..., 2]-target.depth
    final = output['camera_points'][..., 2]-target.depth
    counts = {}
    for label, (lo, hi) in DEPTH_BINS.items():
        mask = target.valid & (target.depth >= lo) & (target.depth < hi)
        counts[label] = int(mask.sum())
        if counts[label]:
            values = torch.stack((world_epe[mask], world_l1[mask], raw[mask], final[mask]), -1)
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite pixel diagnostics in '+label)
            bins[label].append(values.detach().cpu().numpy())
    return counts


def summarize_bins(bins):
    result = {}
    for label, arrays in bins.items():
        if not arrays:
            result[label] = dict(valid_pixels=0, status='N/A: no valid pixels')
            continue
        values = np.concatenate(arrays).astype(np.float64)
        row = dict(valid_pixels=len(values))
        for name, column in (('world_epe', values[:, 0]), ('world_l1', values[:, 1]),
                             ('raw_depth_l1', abs(values[:, 2])), ('final_depth_l1', abs(values[:, 3]))):
            row[name+'_m'] = float(column.mean())
            row[name+'_median_m'] = float(np.median(column))
            row[name+'_p90_m'] = float(np.quantile(column, .9))
            row[name+'_max_m'] = float(column.max())
        row['raw_depth_bias_m'] = float(values[:, 2].mean())
        row['final_depth_bias_m'] = float(values[:, 3].mean())
        result[label] = row
    return result


def checkpoint_metadata(root, config, specs, *, tag=None):
    checkpoint = root/'training_state_18000.pt'
    stat = checkpoint.stat()
    import hashlib, json
    metadata = dict(run_name=root.name, step=18000, checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns, config_sha256=digest(root/'config.yaml'),
        split_sha256=digest(PROJECT_ROOT/config['split']), image_hw=config['image_hw'],
        architecture_revision=config['architecture_revision'], ablation=config.get('ablation'),
        episode_specs_sha256=hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest())
    if tag is not None:
        metadata.update(diagnostic_schema_version=2, method=tag,
            mapping_mode=('none' if tag == 'No-Memory' else
                          'first_frame_unfused' if tag == 'Single-map' else 'all_frames_fused'))
    return metadata


@torch.no_grad()
def load_memory_model(root, config, assets, *, no_memory=False):
    state = load_checkpoint_readonly(root/'training_state_18000.pt')
    if state['step'] != 18000 or state['config'] != config:
        raise ValueError('Final checkpoint/config mismatch: '+root.name)
    kwargs = dict(source=assets['official_vggt_source']['path'],
                  weights=assets['vggt_weights']['path'], receive_layer=config['receive_layer'])
    model = (VGGTNoMemory(**kwargs) if no_memory else VGGTSpatialMemory(**kwargs,
        revision=config['architecture_revision'], pose_jitter_probability=0.,
        ablation=config.get('ablation'))).cuda().eval()
    params = dict(model.named_parameters())
    expected = {name for name, value in params.items() if value.requires_grad}
    if set(state['trainable']) != expected:
        raise ValueError('Final checkpoint trainable parameter names differ from model: '+root.name)
    for name, value in state['trainable'].items():
        if params[name].shape != value.shape:
            raise ValueError('Checkpoint parameter shape mismatch: '+name)
        params[name].copy_(value)
    return model


def predict_episode(model, episode, tag):
    """Only the branch's permitted query/map inputs enter its prediction."""
    query = episode['query'][0]
    if tag == 'No-Memory':
        return model(query)
    if tag == 'Single-map':
        memory = single_frame_unfused(model, episode['mapping'][0])
    elif tag in TAGS:
        memory = model.build_memory(episode['mapping'])
    else:
        raise ValueError('Unknown diagnostic method: '+tag)
    return model(query, memory)


def cache_path(tag):
    return OUTPUT/'diagnostics'/(CACHE_NAMES[tag]+'.json')


def read_matching_cache(path, metadata, tag, specs):
    """Reuse only a complete same-method checkpoint/protocol measurement."""
    if not path.exists():
        return None
    try:
        cache = read_json(path)
        robotics = [s for s in specs if s['scene'].startswith('harvard_robotics_lab/')]
        if cache['metadata'] != metadata or identities(cache['episodes']) != identities(robotics):
            return None
        validate_depth_bins({tag:cache['robotics_depth_bins']}, required_tags=(tag,))
        for label in DEPTH_BINS:
            if sum(r['valid_pixels_by_bin'][label] for r in cache['episodes']) != cache['robotics_depth_bins'][label]['valid_pixels']:
                return None
        return cache
    except (OSError, ValueError, TypeError, KeyError):
        return None


@torch.no_grad()
def evaluate_variant(tag, root, dataset, specs, assets):
    """Measure one of all six methods, including read-only legacy weights."""
    config = yaml.safe_load((root/'config.yaml').read_text())
    if tag in VARIANTS and config.get('ablation') != VARIANTS[tag]:
        raise ValueError('Unexpected ablation model: '+tag)
    if tag in ('Full', 'Single-map') and (config.get('ablation') is not None or config['architecture_revision'] != 'step2_v5'):
        raise ValueError('Full/Single-map requires the original Full v5 weights')
    if tag == 'No-Memory' and (config.get('ablation') is not None or config['architecture_revision'] != 'single_frame_nomemory'):
        raise ValueError('No-Memory requires its own single-frame weights')
    metadata = checkpoint_metadata(root, config, specs, tag=tag)
    destination = cache_path(tag)
    cache = read_matching_cache(destination, metadata, tag, specs)
    if cache is not None:
        print('Reuse matching measured diagnostic:', tag, flush=True)
        return cache
    robotics = [s for s in specs if s['scene'].startswith('harvard_robotics_lab/')]
    if len(robotics) != 11:
        raise ValueError('Expected the original 11 robotics query specifications')
    model = load_memory_model(root, config, assets, no_memory=tag == 'No-Memory')
    bins = {label:[] for label in DEPTH_BINS}
    records = []
    for index, spec in enumerate(robotics):
        episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device='cuda')
        output = predict_episode(model, episode, tag)
        # Query GT is read only after the forward, for masks and errors.
        target = episode['query'][1]
        counts = accumulate_bins(output, target, bins)
        records.append(dict(**spec, valid_pixels_by_bin=counts))
        del output, episode
        print(tag, 'robotics', index+1, '/', len(robotics), flush=True)
    del model
    torch.cuda.empty_cache()
    result = dict(metadata=metadata, pixel_aggregation='All valid pixels pooled over the fixed 11 queries',
        depth_bin_reference='GT camera z for evaluation masks only; never forwarded',
        alignment='none', query_inputs=['rgb', 'K'], episodes=records, robotics_depth_bins=summarize_bins(bins))
    validate_depth_bins({tag:result['robotics_depth_bins']}, required_tags=(tag,))
    # Atomic files also permit independent bounded GPU jobs to merge caches.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    write_json(temporary, result)
    os.replace(temporary, destination)
    return result


@torch.no_grad()
def single_map_v2(root, dataset, specs, assets):
    config = yaml.safe_load((root/'config.yaml').read_text())
    selected = [next(s for s in specs if s['scene'] == scene) for scene in SCENES]
    metadata = checkpoint_metadata(root, config, selected)
    provenance = OUTPUT/'single_map_frame_vis/provenance.json'
    paths = [OUTPUT/'single_map_frame_vis'/s.replace('/', '__')/'V2_ellipsoid_shells.png' for s in SCENES]
    if provenance.exists() and read_json(provenance)['metadata'] == metadata and all(p.is_file() for p in paths):
        print('Reuse matching Single-map V2', flush=True)
        return
    from src.vis.smoke import v2
    model = load_memory_model(root, config, assets)
    records = []
    for spec, path in zip(selected, paths):
        episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device='cuda')
        memory = single_frame_unfused(model, episode['mapping'][0])
        path.parent.mkdir(parents=True, exist_ok=True)
        v2(path.parent, dict(name=episode['name']+' / first mapping view only, no fusion',
                            mapping=episode['mapping'][:1]), memory)
        records.append(dict(**spec, model_mapping_frame_ids=spec['mapping_frame_ids'][:1],
                            fusion=False, gaussians=len(memory), source=str(path), sha256=digest(path)))
        del memory, episode
        print('Single-map V2:', path, flush=True)
    del model
    torch.cuda.empty_cache()
    write_json(provenance, dict(metadata=metadata, official_plus_full_18k_trainable=True,
        description='First permitted map RGB-D only, shared writer/context, no fusion; no query forward or alignment', episodes=records))


def collect_cached_diagnostics(roots, specs):
    """Merge only new complete same-method measurements; never old partial bins."""
    result, sources = {}, {}
    for tag in TAGS:
        root = roots[tag]
        path = cache_path(tag)
        if not path.is_file() or not (root/'config.yaml').is_file() or not (root/'training_state_18000.pt').is_file():
            continue
        config = yaml.safe_load((root/'config.yaml').read_text())
        metadata = checkpoint_metadata(root, config, specs, tag=tag)
        cache = read_matching_cache(path, metadata, tag, specs)
        if cache is not None:
            result[tag] = cache['robotics_depth_bins']
            sources[tag] = dict(**metadata, cache=str(path), cache_sha256=digest(path))
    if len(result) == len(TAGS):
        validate_depth_bins(result)
    return result, sources


def main():
    parser = make_parser(comparison=False)
    parser.description = __doc__
    parser.add_argument('--only', nargs='+', choices=[*DEPTH_TASKS, 'Single-map'],
                        default=[*DEPTH_TASKS, 'Single-map'],
                        help='Bounded jobs: Single-map generates V2 only; Single-map-depth measures bins; matching caches reused')
    args = parser.parse_args()
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):
        parser.error('Use scripts/run_on_available_gpu.py for live GPU selection')
    fraction = float(os.environ['VGGT_GPU_MEMORY_FRACTION'])
    if not 0 < fraction <= 1:
        parser.error('Invalid live-launcher memory fraction')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    roots = roots_from_args(args)
    config = yaml.safe_load((roots['Full']/'config.yaml').read_text())
    sequences = {r['id']:r for r in yaml.safe_load((PROJECT_ROOT/config['split']).read_text())['sequences']}
    specs = [{k:r[k] for k in ('scene', 'mapping_frame_ids', 'query_frame_id', 'split')}
             for r in validate_episodes(read_json(roots['Full']/'evaluation_18000_full.json'), sequences)]
    dataset = EpisodeDataset(PROJECT_ROOT/config['split'], config['image_hw'])
    if identities(dataset.evaluation_episodes()) != identities(specs):
        raise ValueError('Current dataset fixed evaluation differs from original Full final protocol')
    assets = load_config()['assets']
    for task in args.only:
        if task == 'Single-map':
            single_map_v2(roots['Full'], dataset, specs, assets)
            continue
        tag = 'Single-map' if task == 'Single-map-depth' else task
        variant_config = yaml.safe_load((roots[tag]/'config.yaml').read_text())
        if variant_config['image_hw'] != config['image_hw'] or digest(PROJECT_ROOT/variant_config['split']) != digest(PROJECT_ROOT/config['split']):
            raise ValueError('Diagnostic split/resolution changed: '+tag)
        validate_episodes(read_json(roots[tag]/'evaluation_18000_full.json'), sequences, reference=specs)
        evaluate_variant(tag, roots[tag], dataset, specs, assets)
    result, sources = collect_cached_diagnostics(roots, specs)
    merged = OUTPUT/'robotics_depth_bins.json'
    temporary = merged.with_suffix(f'.{os.getpid()}.tmp')
    write_json(temporary, dict(protocol=dict(robotics_queries=11, depth_bins=DEPTH_BINS,
        alignment='none', source='Read-only 18k checkpoint forwards for all six methods',
        missing_metrics='Permitted only in explicitly incomplete previews'), sources=sources,
        status='complete' if len(result) == len(TAGS) else 'incomplete', robotics_depth_bins=result))
    os.replace(temporary, merged)
    print('Diagnostics saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
