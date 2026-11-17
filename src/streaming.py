"""Read-then-Write state: only spatial Gaussians and scalar 3D statistics.

Only seed() accepts RGB-D. forward() accepts RGB/K and the previous spatial
state, with no path for subsequent ground-truth geometry or pose inputs.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from src.data import QueryInput
from src.geometry import patch_intrinsics
from src.memory.gaussians import GaussianMemory
from src.memory.streaming_fusion import StreamingMomentFusion
from src.models.stream_pose import StreamPoseGate


def empty_like(memory):
    return memory.subset(slice(0, 0))


@dataclass(frozen=True)
class SpatialWrite:
    memory: GaussianMemory
    support: torch.Tensor
    step: int


@dataclass(frozen=True)
class StreamingState:
    memory: GaussianMemory
    support: torch.Tensor
    step: int
    # Local-W alone keeps recent 3D candidate writes, never RGB or 2D KV.
    window_entries: tuple = ()


class WriteConfidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.modulation = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.constant_(self.modulation[-1].bias, 1.)

    def forward(self, depth, output):
        p = 1-output['dustbin_probability'].reshape_as(depth)
        c = output['c_depth'].reshape_as(depth)
        observed = output.get('rendered_T1', output.get('rendered'))
        if observed is None:
            opacity = torch.zeros_like(depth)
            variance, residual = opacity, opacity
        else:
            opacity = observed.opacity[..., 0]
            variance = observed.depth_variance[..., 0]/observed.depth[..., 0].square().clamp_min(.04)
            residual = ((depth-observed.depth[..., 0])/observed.depth[..., 0].clamp_min(.2)).abs()
        attrs = torch.stack((p, c, depth.clamp_min(0).log1p(), opacity,
                             variance.clamp_max(10), residual.clamp_max(10)), -1)
        learned = self.modulation(attrs).sigmoid()[..., 0]
        # Supported patches establish the global pose. Once pose is supported,
        # unseen valid patches CAN expand the map; they need not already match.
        frame_quality = torch.sigmoid((p.mean()-.15)*25)
        consistency = 1/(1+opacity*(residual/.30).square())
        weights = learned*frame_quality*(.4+.6*p)*(.5+.5*c)*consistency/(1+variance)
        valid = torch.isfinite(depth) & (depth > .15) & (depth < 12.)
        return torch.where(valid, weights, 0.), frame_quality


class StreamingEngine(nn.Module):
    def __init__(self, model, window=None, *, min_write_confidence=.10):
        super().__init__()
        if window is not None and window < 1:
            raise ValueError('Local-W must retain at least one 3D write')
        self.model = model
        self.window = window
        self.min_write_confidence = min_write_confidence
        self.fusion = StreamingMomentFusion()
        # Start the learned feature/opacity update gates at the trained values.
        with torch.no_grad():
            for name in ('feature_gate', 'opacity_gate'):
                if hasattr(self.fusion, name) and hasattr(model.fusion, name):
                    getattr(self.fusion, name).copy_(getattr(model.fusion, name))
        self.write_gate = WriteConfidence()
        if not hasattr(self.model, 'stream_pose_gate'):
            self.model.stream_pose_gate = StreamPoseGate()

    def read(self, query: QueryInput, memory: GaussianMemory, *, return_attention=False):
        """Same conservative read for fixed-state counterfactuals; no write."""
        output = self.model(query, memory, return_attention=return_attention, streaming=True)
        output.pop('stream_writer_tokens', None)
        return output

    def seed(self, query: QueryInput, depth, valid=None):
        if type(query) is not QueryInput:
            raise TypeError('Seed takes QueryInput plus explicit initial RGB-D depth')
        from src.models.vggt_memory import frozen_patch_tokens, aggregate_single
        rgb = query.rgb
        if tuple(depth.shape) != tuple(rgb.shape[-2:]):
            raise ValueError('Seed depth and RGB must share calibrated pixels')
        valid = torch.isfinite(depth) & (depth > 0) if valid is None else valid & torch.isfinite(depth) & (depth > 0)
        with torch.no_grad(), torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=rgb.is_cuda):
            shallow = frozen_patch_tokens(self.model.native.aggregator, rgb).float()
            context = aggregate_single(self.model.native.aggregator, rgb, shallow,
                receive_layer=self.model.receive_layer, stop_before_injection=True).float()
        tokens = torch.cat((shallow[0], context), -1)
        fraction = F.avg_pool2d(valid.float()[None, None], 14, 14)[0, 0]
        patch = F.avg_pool2d(torch.where(valid, depth, 0.)[None, None], 14, 14)[0, 0]/fraction.clamp_min(1e-6)
        identity = torch.eye(4, device=depth.device, dtype=depth.dtype)
        candidates = self.model.writer(tokens, patch, patch_intrinsics(query.K), identity, valid_mask=fraction > .25)
        support = fraction[fraction > .25]*candidates.alpha
        memory, support = self.fusion(empty_like(candidates), support[:0], candidates, support)
        entries = (SpatialWrite(candidates, fraction[fraction > .25]*candidates.alpha, 0),) if self.window else ()
        return StreamingState(memory, support, 0, entries)

    def forward(self, query: QueryInput, state: StreamingState, *, return_attention=False):
        if type(query) is not QueryInput or type(state) is not StreamingState:
            raise TypeError('Streaming step accepts only QueryInput(rgb,K), StreamingState')
        output = self.model(query, state.memory, return_attention=return_attention, streaming=True)
        tokens = output.pop('stream_writer_tokens')
        points = output['camera_points']
        hp, wp = output['patch_hw']
        # Positive predicted CAMERA-Z only. No supervision exists in this API.
        positive = torch.isfinite(points).all(-1) & (points[..., 2] > .15) & (points[..., 2] < 12.)
        fraction = F.adaptive_avg_pool2d(positive.float()[None, None], (hp, wp))[0, 0]
        depth = F.adaptive_avg_pool2d(torch.where(positive, points[..., 2], 0.)[None, None], (hp, wp))[0, 0]/fraction.clamp_min(1e-6)
        confidence, quality = self.write_gate(depth, output)
        keep = (confidence.detach() >= self.min_write_confidence) & (fraction > .5)
        candidates = self.model.writer(tokens, depth, patch_intrinsics(query.K), output['T1'], valid_mask=keep)
        candidate_support = confidence[keep]*fraction[keep]*candidates.alpha
        step = state.step+1
        if self.window:
            entries = tuple(x for x in state.window_entries if x.step > step-self.window)
            entries = entries+(SpatialWrite(candidates, candidate_support, step),)
            # Discard expired candidate contributions exactly. Re-fusion uses
            # only the retained 3D writes, not past frames or their GT inputs.
            memory, support = empty_like(state.memory), state.support[:0]
            for entry in entries:
                memory, support = self.fusion(memory, support, entry.memory, entry.support)
        else:
            entries = ()
            memory, support = self.fusion(state.memory, state.support, candidates, candidate_support)
        output.update(write_confidence=confidence, frame_write_quality=quality,
            active_count=len(output['attention']['indices']),
            write=dict(candidates=len(candidates), mean_confidence=float(confidence.detach().mean()),
                       frame_quality=float(quality.detach()), total_gaussians=len(memory)))
        return output, StreamingState(memory, support, step, entries)
