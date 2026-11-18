"""Prediction-only v4 streaming with immutable seeds and soft write support."""
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from src.data import QueryInput
from src.geometry import patch_intrinsics
from src.memory.gaussians import GaussianMemory
from src.models.stream_landmarks import ray_consensus
from src.streaming_v3 import StreamingEngineV3, StreamingStateV3


@dataclass(frozen=True)
class StreamingStateV4(StreamingStateV3):
    older_pose: torch.Tensor | None = None


def feature_reliability(rendered, query_features):
    cosine = F.cosine_similarity(rendered.features.reshape_as(query_features),query_features,dim=-1)
    cosine = cosine.reshape_as(rendered.opacity)
    return cosine, ((cosine-.50)/.30).clamp(0.,1.)


def soft_write_support(rendered, similarity, raw_depth, source):
    """Shared per-frame support attenuation with a positive prior for exploration."""
    weight = rendered.opacity[...,0].clamp(0.,1.)
    weight = torch.where(weight >= .20,weight,0.)
    depth = rendered.depth[...,0]
    residual = (raw_depth-depth).abs()/depth.clamp_min(.20)
    denom = weight.sum()
    cosine = (weight*similarity[...,0]).sum()/denom.clamp_min(1e-6)
    relative = (weight*residual.clamp(max=4.)).sum()/denom.clamp_min(1e-6)
    appearance = torch.sigmoid((cosine-.72)/.10)
    agreement = 1/(1+(relative/.65).square())
    quality = torch.where(denom>0,appearance*agreement,torch.full_like(denom,.50))
    source_weight = .65 if source == 'global' else 1.
    return .10+.90*source_weight*quality, cosine, relative


def freeze_seed_fields(memory, support, lock, previous):
    if not lock.any():
        return memory,support
    # Fusion preserves seed identity order. W4 can reconstruct the seed
    # geometry with a fresh covariance epsilon, so restore all four fields.
    source = previous.memory.subset(previous.seed_lock)
    values = []
    for key in ('means','covars','alpha','features'):
        value = getattr(memory,key).clone()
        value[lock] = getattr(source,key)
        values.append(value)
    mass = support.clone()
    mass[lock] = previous.support[previous.seed_lock]
    return GaussianMemory(*values),mass


class StreamingEngineV4(StreamingEngineV3):
    def seed(self, query, depth, valid=None):
        state = super().seed(query,depth,valid)
        return StreamingStateV4(state.memory,state.support,state.step,state.window_entries,state.seed_lock,
                                prev_pose=state.prev_pose)

    def fuse_write(self, state, candidates, candidate_support):
        memory,support,lock,entries = super().fuse_write(state,candidates,candidate_support)
        memory,support = freeze_seed_fields(memory,support,lock,state)
        return memory,support,lock,entries

    def predict(self, query, memory, previous, older, seed_lock, *, return_attention):
        return self.model(query,memory,streaming=True,streaming_v4=True,
                          streaming_previous_pose=previous,streaming_older_pose=older,
                          streaming_seed_lock=seed_lock,return_attention=return_attention)

    def read(self, query, memory, *, return_attention=False, support=None, previous_pose=None, seed_lock=None, older_pose=None):
        if previous_pose is None:
            previous_pose = torch.eye(4,device=query.rgb.device)
        output = self.predict(query,memory,previous_pose,older_pose,seed_lock,return_attention=return_attention)
        output.pop('stream_writer_tokens',None)
        output.pop('stream_query_features',None)
        return output

    def forward(self, query: QueryInput, state: StreamingStateV4, *, return_attention=False):
        if type(query) is not QueryInput or type(state) is not StreamingStateV4 or state.prev_pose is None:
            raise TypeError('Streaming v4 requires RGB/K and a spatial state with predicted motion')
        output = self.predict(query,state.memory,state.prev_pose,state.older_pose,state.seed_lock,
                              return_attention=return_attention)
        tokens = output.pop('stream_writer_tokens')
        output.pop('stream_query_features')
        points, shape = output['camera_points'], output['patch_hw']
        z = points[...,2]
        positive = torch.isfinite(points).all(-1) & (z > .15) & (z < 12.)

        def patch_average(grid):
            return F.adaptive_avg_pool2d(grid[None,None],shape)[0,0]

        fraction = patch_average(positive.float())
        depth = patch_average(torch.where(positive,z,0.))/fraction.clamp_min(1e-6)
        raw_depth = patch_average(output['raw_camera_points'][...,2])
        confidence, frame_quality = self.write_gate(depth,output,frontier_threshold=.05)
        source = output['stream_pose_source']
        multiplier, cosine, residual = soft_write_support(output['rendered_T1'], output['stream_feature_cosine'],
                                                          raw_depth,source)
        with torch.no_grad():
            consensus = confidence.new_zeros(())
            if len(state.memory):
                p = 1-output['refined_dustbin_probability']
                m2,_ = ray_consensus(output['T1'],output['refined_anchors'],output['refined_posterior_covars'],
                                     output['camera_rays'],p)
                consensus = ((p >= .5) & (m2 <= 4.)).float().mean()
            safe = bool(consensus >= .05)
        keep = (confidence.detach() >= self.min_write_confidence) & (fraction > .5) & safe
        candidates = self.model.writer(tokens,depth,patch_intrinsics(query.K),output['T1'],valid_mask=keep)
        mass = multiplier*confidence[keep]*fraction[keep]*candidates.alpha
        memory,support,lock,entries = self.fuse_write(state,candidates,mass)
        output.update(write_confidence=confidence,frame_write_quality=frame_quality,
            active_count=len(output['attention']['indices']),write=dict(candidates=len(candidates),
                mean_confidence=float(confidence.detach().mean()),frame_quality=float(frame_quality.detach()),
                effective_confidence=float((multiplier*confidence).detach().mean()),
                support_multiplier=float(multiplier.detach()),candidate_support=float(mass.detach().sum()),
                feature_cosine=float(cosine.detach()),raw_depth_relative_residual=float(residual.detach()),
                total_gaussians=len(memory),safety_accepted=safe,rigid_consensus=float(consensus),
                pose_source=source,pose_evidence=output['stream_pose_evidence'],landmark_selected=source=='seed'))
        return output,replace(state,memory=memory,support=support,step=state.step+1,seed_lock=lock,
                              window_entries=entries,prev_pose=output['T1'].detach(),older_pose=state.prev_pose.detach())
