"""Seed-anchored, self-checked streaming; all persistent state is spatial.

The additional state is one scalar support and lock flag per 3D primitive and
the last accepted write's predicted pose. There is no image, token grid or KV.
The original StreamingEngine remains the immutable v1 execution path.
"""
from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F
from src.data import QueryInput
from src.geometry import patch_intrinsics
from src.memory.gaussians import GaussianMemory
from src.models.stream_landmarks import ray_consensus, appearance_agreement
from src.streaming import StreamingEngine, SpatialWrite, empty_like


@dataclass(frozen=True)
class StreamingStateV2:
    memory: GaussianMemory
    support: torch.Tensor
    step: int
    window_entries: tuple = ()
    seed_lock: torch.Tensor | None = None
    last_write_pose: torch.Tensor | None = None


def detached_memory(memory):
    return GaussianMemory(*(getattr(memory, key).detach() for key in ('means', 'covars', 'alpha', 'features')))


def detach_state(state):
    """TBPTT cuts the graph, never the public spatial memory or write history."""
    tracking = {'prev_pose': state.prev_pose.detach()} if hasattr(state, 'prev_pose') else {}
    return replace(state, memory=detached_memory(state.memory), support=state.support.detach(),
        window_entries=tuple(SpatialWrite(detached_memory(e.memory), e.support.detach(), e.step) for e in state.window_entries),
        last_write_pose=None if state.last_write_pose is None else state.last_write_pose.detach(), **tracking)


def locked_fusion(fusion, memory, support, candidates, candidate_support, seed_lock):
    updated, mass = fusion(memory, support, candidates, candidate_support)
    if seed_lock is None:
        seed_lock = torch.zeros(len(memory), device=mass.device, dtype=torch.bool)
    lock = torch.cat((seed_lock, seed_lock.new_zeros(len(updated)-len(memory))))
    if len(memory):
        # Seed RGB-D defines the metric gauge. Its means/covariance are immutable;
        # visual descriptors and opacity still accumulate learned observations.
        old_means = torch.cat((memory.means, updated.means[len(memory):]))
        old_covars = torch.cat((memory.covars, updated.covars[len(memory):]))
        updated = GaussianMemory(torch.where(lock[:, None], old_means, updated.means),
            torch.where(lock[:, None, None], old_covars, updated.covars), updated.alpha, updated.features)
    return updated, mass, lock


class ConsistentWriteConfidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.modulation = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.constant_(self.modulation[-1].bias, 1.)

    def forward(self, depth, output, query_features):
        p = 1-output.get('refined_dustbin_probability', output['dustbin_probability'])
        covariance = output.get('refined_posterior_covars', output.get('posterior_covars'))
        anchors = output.get('refined_anchors', output['anchors'])
        if covariance is None:
            m2, consensus = p*0.+1e6, p.sum()*0.
        else:
            m2, consensus = ray_consensus(output['T1'], anchors, covariance, output['camera_rays'], p)
        rendered = output['rendered_T1']
        raw_depth = F.adaptive_avg_pool2d(output['raw_camera_points'][..., 2][None, None], depth.shape)[0, 0]
        similarity, relative_depth_error, coverage = appearance_agreement(rendered, query_features, raw_depth)
        hard_consensus = ((p > .4) & (m2 <= 4)).float().mean()
        # Hard safety rejection cannot be bypassed by learning a high scalar
        # write score. The smooth factors preserve trainable gating gradients.
        safe = (hard_consensus >= .08) & (coverage >= .04) & (similarity >= .35) & (relative_depth_error <= .85)
        quality = (torch.sigmoid((consensus-.10)*20)*torch.sigmoid((similarity-.40)*12)
                   *torch.sigmoid((.65-relative_depth_error)*8))*safe
        variance = (rendered.depth_variance/rendered.depth.square().clamp_min(.04))[..., 0]
        opacity = rendered.opacity[..., 0]
        residual = ((depth-rendered.depth[..., 0])/rendered.depth[..., 0].clamp_min(.2)).abs()
        patch_p = p.reshape_as(depth)
        attrs = torch.stack((patch_p, output['c_depth'].reshape_as(depth), depth.clamp_min(0).log1p(), opacity,
                             variance.clamp_max(10), residual.clamp_max(10)), -1)
        learned = self.modulation(attrs).sigmoid()[..., 0]
        weights = learned*quality*(.65+.35*patch_p)/(1+variance)
        valid = torch.isfinite(depth) & (depth > .15) & (depth < 12.)
        diagnostics = dict(rigid_consensus=float(hard_consensus.detach()), soft_consensus=float(consensus.detach()),
            feature_cosine=float(similarity.detach()), relative_depth_error=float(relative_depth_error.detach()),
            visible_coverage=float(coverage.detach()), safety_accepted=bool(safe))
        return torch.where(valid, weights, 0.), quality, diagnostics


class StreamingEngineV2(StreamingEngine):
    def __init__(self, model, window=None, *, min_write_confidence=.10):
        super().__init__(model, window, min_write_confidence=min_write_confidence)
        self.write_gate = ConsistentWriteConfidence()

    def seed(self, query, depth, valid=None):
        state = super().seed(query, depth, valid)
        support = torch.full_like(state.support, self.fusion.support_cap)
        lock = torch.ones(len(state.memory), device=depth.device, dtype=torch.bool)
        return StreamingStateV2(state.memory, support, 0, state.window_entries, lock, torch.eye(4, device=depth.device))

    def read(self, query, memory, *, return_attention=False, support=None):
        # Counterfactuals without a support argument have no stable-landmark
        # prior, but use identical frontier transport and model parameters.
        if support is None:
            support = memory.alpha.new_ones(len(memory))
        output = self.model(query, memory, streaming=True, streaming_support=support, return_attention=return_attention)
        output.pop('stream_writer_tokens', None)
        return output

    def fuse_write(self, state, candidates, candidate_support):
        step = state.step+1
        if not self.window:
            memory, support, lock = locked_fusion(
                self.fusion, state.memory, state.support, candidates, candidate_support, state.seed_lock)
            return memory, support, lock, ()
        entries = tuple(e for e in state.window_entries if e.step > step-self.window)
        entries += (SpatialWrite(candidates, candidate_support, step),)
        memory, support = empty_like(state.memory), state.support[:0]
        lock = torch.zeros(0, device=support.device, dtype=torch.bool)
        for entry in entries:
            memory, support, lock = locked_fusion(self.fusion, memory, support, entry.memory, entry.support, lock)
            if entry.step == 0:
                support = torch.full_like(support, self.fusion.support_cap)
                lock = torch.ones(len(memory), device=support.device, dtype=torch.bool)
        return memory, support, lock, entries

    def forward(self, query: QueryInput, state: StreamingStateV2, *, return_attention=False):
        if type(query) is not QueryInput or type(state) is not StreamingStateV2:
            raise TypeError('Streaming v2 accepts only RGB/K and spatial state')
        output = self.model(query, state.memory, streaming=True, streaming_support=state.support, return_attention=return_attention)
        tokens = output.pop('stream_writer_tokens')
        hp, wp = output['patch_hw']
        points = output['camera_points']
        positive = torch.isfinite(points).all(-1) & (points[..., 2] > .15) & (points[..., 2] < 12.)
        fraction = F.adaptive_avg_pool2d(positive.float()[None, None], (hp, wp))[0, 0]
        depth = F.adaptive_avg_pool2d(torch.where(positive, points[..., 2], 0.)[None, None], (hp, wp))[0, 0]/fraction.clamp_min(1e-6)
        confidence, quality, checks = self.write_gate(depth, output, self.model.writer.latent(tokens))
        with torch.no_grad():
            pose = output['T1'].detach()
            last = state.last_write_pose
            displacement = (pose[:3, 3]-last[:3, 3]).norm() if last is not None else pose.new_tensor(1.)
            rotation = torch.acos(((torch.trace(last[:3, :3].T @ pose[:3, :3])-1)/2).clamp(-1, 1)) if last is not None else pose.new_tensor(1.)
            frontier = (output['rendered_T1'].opacity < .05).float().mean()
            keyframe = bool((displacement > .04) | (rotation > .035) | (frontier > .18))
        keep = (confidence.detach() >= self.min_write_confidence) & (fraction > .5) & keyframe
        candidates = self.model.writer(tokens, depth, patch_intrinsics(query.K), output['T1'], valid_mask=keep)
        candidate_support = confidence[keep]*fraction[keep]*candidates.alpha
        step = state.step+1
        memory, support, lock, entries = self.fuse_write(state, candidates, candidate_support)
        last_pose = pose if len(candidates) else state.last_write_pose
        output.update(write_confidence=confidence, frame_write_quality=quality,
            active_count=len(output['attention']['indices']), write=dict(candidates=len(candidates),
                mean_confidence=float(confidence.detach().mean()), frame_quality=float(quality.detach()),
                total_gaussians=len(memory), keyframe=keyframe, frontier_fraction=float(frontier),
                landmark_selected=bool(output.get('landmark_selected', False)), **checks))
        return output, StreamingStateV2(memory, support, step, entries, lock, last_pose)
