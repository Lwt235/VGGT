"""GT-free, gauge-invariant checks of alternative spatial-memory poses."""
import torch
from torch.nn import functional as F


def ray_consensus(pose, anchors, covars, rays, p_match):
    """Ellipsoid distance to the positive calibrated ray (no monocular depth)."""
    direction = rays @ pose[:3, :3].T
    delta = anchors-pose[:3, 3]
    precision = torch.linalg.inv(covars+1e-6*torch.eye(3, device=covars.device, dtype=covars.dtype))
    Ad = (precision @ direction[..., None])[..., 0]
    distance = ((Ad*delta).sum(-1)/(Ad*direction).sum(-1).clamp_min(1e-8)).clamp_min(.05)
    residual = distance[:, None]*direction-delta
    m2 = (residual*(precision @ residual[..., None])[..., 0]).sum(-1).clamp_min(0)
    consensus = (p_match*((1-m2/4).clamp(0, 1))).mean()
    return m2, consensus


def appearance_agreement(rendered, query_features, raw_depth):
    cosine = F.cosine_similarity(rendered.features.reshape_as(query_features), query_features, dim=-1)
    opacity = rendered.opacity.flatten()
    rel_var = (rendered.depth_variance/rendered.depth.square().clamp_min(.04)).flatten()
    weight = (opacity/.25).clamp(0, 1)/(1+rel_var)
    denominator = weight.sum().clamp_min(1e-6)
    similarity = (weight*cosine).sum()/denominator
    depth_error = (weight*((rendered.depth.flatten()-raw_depth.flatten()).abs()/rendered.depth.flatten().clamp_min(.2)).clamp_max(3)).sum()/denominator
    return similarity, depth_error, weight.mean()


def candidate_score(pose, bootstrap, rendered, query_features):
    m2, consensus = ray_consensus(pose, bootstrap['anchors'], bootstrap['posterior_covars'],
                                  bootstrap['camera_rays'], 1-bootstrap['dustbin_probability'])
    similarity, depth_error, coverage = appearance_agreement(rendered, query_features, bootstrap['z_cam'])
    # Normalized robust geometric evidence plus independently rendered visual
    # evidence. Penalize unsupported candidates instead of rewarding tiny sums.
    p = 1-bootstrap['dustbin_probability']
    evidence = (p/(1+m2/9)).mean()
    return evidence+coverage*(similarity-.15*depth_error)+consensus


def select_landmark_pose(model, tokens, shallow, query_features, memory, support, Kp, hp, wp, full_pose, full_bootstrap):
    stable = support >= 2.
    if len(memory) < 384 or int(stable.sum()) < 16 or bool(stable.all()):
        return full_pose, full_bootstrap
    landmark_pose, landmark_bootstrap = model.initial_pose(tokens, query_features, memory, Kp, hp, wp,
        shallow=shallow, match_mask=stable)
    with torch.no_grad():
        full_render = model.render_reader.render(memory, full_pose, Kp, hp, wp)
        stable_render = model.render_reader.render(memory.subset(stable), landmark_pose, Kp, hp, wp)
        a = candidate_score(full_pose, full_bootstrap, full_render, query_features)
        b = candidate_score(landmark_pose, landmark_bootstrap, stable_render, query_features)
        chosen = bool(b > a+.02)
    bootstrap = landmark_bootstrap if chosen else full_bootstrap
    bootstrap.update(landmark_selected=chosen, landmark_candidate_scores=torch.stack((a, b)),
                     stable_landmark_count=int(stable.sum()))
    return (landmark_pose if chosen else full_pose), bootstrap
