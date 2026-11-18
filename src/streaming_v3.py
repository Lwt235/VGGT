"""Read with a prior predicted pose, then write through v1's smooth gate."""
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from src.data import QueryInput
from src.geometry import patch_intrinsics
from src.models.stream_landmarks import ray_consensus
from src.streaming import StreamingEngine
from src.streaming_v2 import StreamingEngineV2, StreamingStateV2


@dataclass(frozen=True)
class StreamingStateV3(StreamingStateV2):
    prev_pose: torch.Tensor | None = None


class StreamingEngineV3(StreamingEngineV2):
    def __init__(self, model, window=None, *, min_write_confidence=.10):
        StreamingEngine.__init__(self, model, window, min_write_confidence=min_write_confidence)

    def seed(self, query, depth, valid=None):
        state = super().seed(query, depth, valid)
        return StreamingStateV3(state.memory, state.support, state.step, state.window_entries,
                                state.seed_lock, prev_pose=state.last_write_pose)

    def read(self, query, memory, *, return_attention=False, support=None, previous_pose=None, seed_lock=None):
        output = self.model(query, memory, streaming=True, streaming_previous_pose=previous_pose,
            streaming_seed_lock=seed_lock, return_attention=return_attention)
        output.pop('stream_writer_tokens', None)
        return output

    def forward(self, query: QueryInput, state: StreamingStateV3, *, return_attention=False):
        if type(query) is not QueryInput or type(state) is not StreamingStateV3 or state.prev_pose is None:
            raise TypeError('Streaming v3 requires RGB/K and a spatial state with a predicted previous pose')
        output = self.model(query, state.memory, streaming=True, streaming_previous_pose=state.prev_pose,
            streaming_seed_lock=state.seed_lock, return_attention=return_attention)
        tokens = output.pop('stream_writer_tokens')
        points = output['camera_points']
        patch_hw = output['patch_hw']
        positive = torch.isfinite(points).all(-1) & (points[..., 2] > .15) & (points[..., 2] < 12.)
        fraction = F.adaptive_avg_pool2d(positive.float()[None, None], patch_hw)[0, 0]
        depth = F.adaptive_avg_pool2d(torch.where(positive, points[..., 2], 0.)[None, None], patch_hw)[0, 0]
        depth = depth/fraction.clamp_min(1e-6)
        confidence, quality = self.write_gate(depth, output)
        with torch.no_grad():
            if len(state.memory):
                match = 1-output['refined_dustbin_probability']
                m2, _ = ray_consensus(output['T1'], output['refined_anchors'], output['refined_posterior_covars'],
                                      output['camera_rays'], match)
                consensus = ((match >= .5) & (m2 <= 4.)).float().mean()
            else:
                consensus = confidence.new_zeros(())
            safe = bool(consensus >= .05)
        keep = (confidence.detach() >= self.min_write_confidence) & (fraction > .5) & safe
        candidates = self.model.writer(tokens, depth, patch_intrinsics(query.K), output['T1'], valid_mask=keep)
        mass = confidence[keep]*fraction[keep]*candidates.alpha
        memory, support, lock, entries = self.fuse_write(state, candidates, mass)
        output.update(write_confidence=confidence, frame_write_quality=quality,
            active_count=len(output['attention']['indices']), write=dict(candidates=len(candidates),
                mean_confidence=float(confidence.detach().mean()), frame_quality=float(quality.detach()),
                total_gaussians=len(memory), safety_accepted=safe, rigid_consensus=float(consensus),
                pose_source=output['stream_pose_source'], pose_evidence=output['stream_pose_evidence'],
                landmark_selected=output['stream_pose_source'] == 'seed'))
        return output, replace(state, memory=memory, support=support, step=state.step+1,
            window_entries=entries, seed_lock=lock, prev_pose=output['T1'].detach())
