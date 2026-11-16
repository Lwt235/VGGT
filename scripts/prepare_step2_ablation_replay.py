#!/usr/bin/env python3
"""Recover Full's augmentation draws without rerunning or loading its model.

Only RNG metadata and episode logs are read from Full; trainable checkpoint
tensors never initialize an ablation. Frozen native v5 execution has no random
layers. CUDA draws are world gauge, optional read jitter and paired sampling.
"""
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_checkpoint_readonly
configure_local_storage()
import torch
from src.augmentation import sample_world_gauge
from src.models.ellipsoid_pose import sample_read_jitter
from src.step2_data import EpisodeDataset


def main():
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):
        raise RuntimeError('Use the live GPU resource launcher')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    source = PROJECT_ROOT/'results'/'step2_v5_full_seed17'
    output = PROJECT_ROOT/'results'/'step2b_protocol'
    output.mkdir(exist_ok=True)
    destination = output/'augmentation_replay.pt'
    if destination.exists():
        raise FileExistsError('Existing augmentation replay is preserved')
    rows = [json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1, 18001)):
        raise ValueError('Full episode sequence must be complete and unique')
    first = load_checkpoint_readonly(source/'training_state_00000.pt')
    config = first['config']
    cpu_rng = random.Random()
    cpu_rng.setstate(first['python_rng'])
    cuda_initial = first['cuda_rng'][0].clone()
    del first
    dataset = EpisodeDataset(PROJECT_ROOT/config['split'], config['image_hw'])
    # Replay only metadata sampling; no RGB-D pixels are loaded or copied.
    dataset.episode = lambda sid, maps, query, **kw: dict(scene=sid, mapping_frame_ids=maps, query_frame_id=query)
    radii, paired = [], []
    checks = {}
    for i, row in enumerate(rows):
        spec = dataset.sample(rng=cpu_rng)
        if any(spec[k] != row[k] for k in spec):
            raise RuntimeError(f'Full Python episode RNG mismatch at {i+1}')
        coin = cpu_rng.random()
        if row['read_jitter_applied'] and coin >= config['read_jitter_probability']:
            raise RuntimeError(f'Full jitter coin mismatch at {i+1}')
        eligible = i % 2 == 0 and not row['read_jitter_applied'] and bool(row['world_supervision_enabled'])
        paired.append(eligible)
        radii.append(cpu_rng.uniform(*config['paired_erasure_radius_m']) if eligible else 0.)
        if i+1 in (1500, 3000, 6000, 12000, 13000, 18000):
            state = load_checkpoint_readonly(source/f'training_state_{i+1:05d}.pt')
            if cpu_rng.getstate() != state['python_rng']:
                raise RuntimeError(f'Full Python RNG checkpoint mismatch at {i+1}')
            checks[i+1] = state['cuda_rng'][0].clone()
            del state
    world, jitters, flags = [], [], []
    with torch.random.fork_rng(devices=[0]):
        torch.cuda.set_rng_state(cuda_initial)
        for i, row in enumerate(rows):
            world.append(sample_world_gauge('cuda'))
            applied = bool(row['read_jitter_applied'])
            flags.append(applied)
            jitters.append(sample_read_jitter('cuda', include_zero=True) if applied else torch.eye(4, device='cuda'))
            if paired[i]:
                # CUDA exponential/multinomial counter advance depends on
                # cardinality, not learned probabilities or sampled identity.
                torch.multinomial(torch.ones(row['public_gaussians'], device='cuda'), 1)
            if i+1 in checks:
                if not torch.equal(torch.cuda.get_rng_state(), checks[i+1]):
                    raise RuntimeError(f'Full CUDA RNG checkpoint mismatch at {i+1}')
                print('Full Python/CUDA augmentation RNG matched:', i+1, flush=True)
        payload = dict(source_full_run=source.name, seed=config['seed'], steps=len(rows),
            world_gauges=torch.stack(world).cpu(), read_jitters=torch.stack(jitters).cpu(),
            read_jitter_applied=torch.tensor(flags), paired_radius_m=torch.tensor(radii),
            paired_requested=torch.tensor(paired), rng_verified_steps=sorted(checks),
            description='Full RNG metadata only; no model parameters used for initialization')
    torch.save(payload, destination)
    print('Saved augmentation-only replay:', destination, flush=True)


if __name__ == '__main__':
    main()
