"""Camera-local tracking and overlap-normalized seed relocalization."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.geometry import world_rays
from src.models.stream_landmarks import ray_consensus


@dataclass(frozen=True)
class PoseEvidence:
    inliers: int
    visible: int
    inlier_fraction: float
    cosine: float

    def supports(self, minimum_inliers, minimum_fraction, minimum_cosine):
        return (self.inliers >= minimum_inliers and self.inlier_fraction >= minimum_fraction
                and self.cosine >= minimum_cosine)


@torch.no_grad()
def tracking_logit_prior(means, rays, previous_pose):
    """Angular locality in the previous camera gauge with soft frustum edges."""
    local = (means-previous_pose[:3, 3]) @ previous_pose[:3, :3]
    directions = F.normalize(local, dim=-1)
    cosine = (F.normalize(rays, dim=-1) @ directions.T).clamp(-1, 1)
    # (1-cos)/sigma^2 approximates angular distance squared / (2*sigma^2).
    # Long-range lookalikes and points behind the camera receive a soft penalty.
    penalty = (1-cosine)/.15**2 + F.softplus(-local[:, 2]/.15)[None]
    return -penalty.clamp_max(24.)


@torch.no_grad()
def overlap_evidence(pose, match, rendered, query_features):
    visible = (rendered.opacity.flatten() > .20) & (rendered.depth.flatten() > 0)
    p_match = 1-match['dustbin_probability']
    m2, _ = ray_consensus(pose, match['anchors'], match['posterior_covars'], match['camera_rays'], p_match)
    inliers = visible & (p_match >= .5) & (m2 <= 4.)
    count, coverage = int(inliers.sum()), int(visible.sum())
    cosine = F.cosine_similarity(rendered.features.reshape_as(query_features), query_features, dim=-1)
    return PoseEvidence(count, coverage, count/max(coverage, 1),
                        float((cosine*visible).sum()/max(coverage, 1)))


@torch.no_grad()
def local_motion(pose, previous_pose):
    distance = (pose[:3, 3]-previous_pose[:3, 3]).norm()
    relative = previous_pose[:3, :3].T @ pose[:3, :3]
    angle = torch.rad2deg(torch.acos(((relative.trace()-1)*.5).clamp(-1., 1.)))
    return float(distance), float(angle)


def select_stream_pose(model, tokens, shallow, features, memory, Kp, hp, wp, previous_pose, seed_lock):
    """Select from stable seed, continuous track and unconstrained global poses.

    Seed and tracking have independent acceptance criteria. The global solve
    is lazy because it cannot override a verified seed or continuous track.
    """
    solve = model.initial_pose
    evidence = {}

    def estimate(source, render_memory, **kwargs):
        pose, match = solve(tokens, features, memory, Kp, hp, wp, shallow=shallow, **kwargs)
        with torch.no_grad():
            rendered = model.render_reader.render(render_memory, pose, Kp, hp, wp)
            support = overlap_evidence(pose, match, rendered, features)
            motion = local_motion(pose, previous_pose)
        evidence[source] = dict(inliers=support.inliers, visible=support.visible,
            inlier_fraction=support.inlier_fraction, cosine=support.cosine,
            translation_m=motion[0], rotation_deg=motion[1])
        return pose, match, support, motion

    def selected(source, pose, match):
        match.update(stream_pose_source=source, stream_pose_evidence=evidence,
                     landmark_selected=source == 'seed')
        return pose, match

    if seed_lock is not None and int(seed_lock.sum()) >= 18:
        pose, match, support, _ = estimate('seed', memory.subset(seed_lock), match_mask=seed_lock)
        if support.supports(18, .30, .75):
            return selected('seed', pose, match)
        del pose, match

    if len(memory):
        with torch.no_grad():
            _, rays, _ = world_rays(Kp, torch.eye(4, device=Kp.device, dtype=Kp.dtype), hp, wp)
            prior = tracking_logit_prior(memory.means, rays, previous_pose)
        pose, match, support, motion = estimate('track', memory, matching_prior=prior, init_pose=previous_pose)
        if support.supports(16, .20, .68) and motion[0] <= .50 and motion[1] <= 25.:
            return selected('track', pose, match)
        del pose, match

    pose, match = solve(tokens, features, memory, Kp, hp, wp, shallow=shallow)
    return selected('global', pose, match)
