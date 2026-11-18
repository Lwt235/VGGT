"""Trajectory-segment DDP, with references to spatial parameters only.

The frozen VGGT is owned by the engine, outside DDP's registered module tree:
no second copy or broadcast of the billion-parameter native backbone is needed.
Each forward returns one TBPTT loss and a detached 3D state. All ranks execute
the same number of segments, including graph-connected zero padding if needed.
"""
import random
import torch
from torch import nn


PARALLEL_METADATA = frozenset({'distributed_runtime', 'world_size', 'ddp_backend', 'rank_seed_offset'})


def scientific_config(config):
    return {k: v for k, v in config.items() if k not in PARALLEL_METADATA}


def canonical_delta(delta):
    result = {}
    for name, value in delta.items():
        while name.startswith('module.'):
            name = name[len('module.'):]
        if name in result:
            raise ValueError('Duplicate canonical checkpoint parameter: '+name)
        result[name] = value
    return result


def trajectory_seed(seed, rank, step):
    return int(seed)+int(rank)*10007+int(step)


def trajectory_rng(seed, rank, step):
    return random.Random(trajectory_seed(seed, rank, step))


class TrajectorySegment(nn.Module):
    def __init__(self, engine, criterion, config):
        super().__init__()
        # Bypass Module registration only for the engine; parameters below are
        # the exact same objects used by engine and its existing Adam optimizer.
        object.__setattr__(self, '_engine', engine)
        self.spatial_parameters = nn.ParameterList(p for p in engine.parameters() if p.requires_grad)
        self.criterion = criterion
        self.config = config

    def forward(self, frames, state, seed, total_queries, seed_overlaps=None):
        from src.streaming_v2 import detach_state
        engine = self._engine
        if seed is not None:
            query, truth = seed
            state = engine.seed(query, truth.depth, truth.valid)
        losses, measurements, sizes, writes, active = [], [], [], [], []
        for i,(query,target) in enumerate(frames):
            output, state = engine(query, state)
            if seed_overlaps is None:
                loss,metrics = self.criterion(output,target,self.config)
            else:
                loss,metrics = self.criterion(output,target,self.config,seed_surface_overlap=seed_overlaps[i])
            losses.append(loss)
            measurements.append(metrics)
            sizes.append(len(state.memory))
            writes.append(output['write'])
            active.append(output['active_count'])
        if losses:
            loss = torch.stack(losses).sum()/total_queries
        else:
            # Keep the collective count equal for arbitrary configured lengths.
            loss = self.spatial_parameters[0].sum()*0
        return loss, detach_state(state), (measurements, sizes, writes, active)
