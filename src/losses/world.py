"""World metric supervision only: no fitting, centering, relative-depth loss."""

import math
import torch
from torch.nn import functional as F
from src.geometry import invert_se3, transform_points


def rotation_geodesic(prediction, target):
    relative = prediction.transpose(-1, -2) @ target
    skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2], relative[..., 0, 2]-relative[..., 2, 0],
                        relative[..., 1, 0]-relative[..., 0, 1]), -1)
    sine = .5*torch.linalg.vector_norm(skew, dim=-1)
    cosine = .5*(relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)
    return torch.atan2(sine, cosine.clamp(-1, 1))


def pose_errors(prediction, target):
    degrees = torch.rad2deg(rotation_geodesic(prediction[:3, :3], target[:3, :3]))
    translation = (prediction[:3, 3]-target[:3, 3]).norm()
    return degrees, translation


def world_loss(output, target, *, camera_weight=.5, refinement_safety_weight=0., anchor_weight=.2, matching_weight=.2):
    valid = target.valid
    if not valid.any():
        raise ValueError("No valid metric supervision pixels")
    delta = output["world_points"][valid]-target.world_points[valid]
    point = delta.abs().mean()
    coarse = output["T0"]
    refined = output["T1"]
    pose = sum(F.smooth_l1_loss((T[:3, 3]-target.c2w[:3, 3]).norm(), point.new_zeros(()), beta=.03) +
               F.smooth_l1_loss(rotation_geodesic(T[:3, :3], target.c2w[:3, :3]), point.new_zeros(()), beta=.03)
               for T in (coarse, refined))
    coarse_point = (output['coarse_world_points'][valid]-target.world_points[valid]).abs().mean()
    # Explicit patch-anchor supervision, evaluator/criterion only. It teaches
    # correspondences rather than feeding any query geometry to the model.
    hp, wp = output['rendered'].opacity.shape[:2]
    valid_patch = F.adaptive_avg_pool2d(valid.float()[None, None], (hp, wp))[0, 0]
    world_patch = F.adaptive_avg_pool2d((target.world_points*valid[..., None]).permute(2, 0, 1)[None], (hp, wp))[0].permute(1, 2, 0)/valid_patch.clamp_min(1e-6)[..., None]
    anchor_mask = valid_patch.flatten() > .25
    match = point*0.
    if len(output['anchor_support_means']):
        with torch.no_grad():
            delta_anchor = world_patch.reshape(-1, 3)[:, None]-output['anchor_support_means'].detach()[None]
            precision = torch.linalg.inv(output['anchor_support_covars'].detach())
            rho2 = torch.einsum('qmi,mij,qmj->qm', delta_anchor, precision, delta_anchor)
            rho2 = rho2.masked_fill(~output['bootstrap_support'][None], float('inf'))
            anchor_mask = anchor_mask & (rho2.min(-1).values <= 9.)
            teacher = (-rho2/2.).softmax(-1)
        if anchor_mask.any():
            match = -(teacher[anchor_mask]*output['bootstrap_attention'][anchor_mask].clamp_min(1e-9).log()).sum(-1).mean()/max(1., math.log(len(output['anchor_support_means'])))
    anchor = (output['anchors'][anchor_mask]-world_patch.reshape(-1, 3)[anchor_mask]).abs().mean() if anchor_mask.any() else point*0.
    camera_truth = transform_points(target.world_points, invert_se3(target.c2w))
    camera_alignment = (output['camera_points'][valid]-camera_truth[valid]).abs().mean()
    # Training supervision only: the reference is detached, so this cannot
    # reward making T0 worse. Inference always returns the predicted T1.
    r0_angle = rotation_geodesic(coarse[:3, :3], target.c2w[:3, :3])
    r1_angle = rotation_geodesic(refined[:3, :3], target.c2w[:3, :3])
    t0_error = (coarse[:3, 3]-target.c2w[:3, 3]).norm()
    t1_error = (refined[:3, 3]-target.c2w[:3, 3]).norm()
    safety = F.relu(r1_angle-r0_angle.detach())+F.relu(t1_error-t0_error.detach())
    loss = point + .25*coarse_point + pose + anchor_weight*anchor + matching_weight*match + camera_weight*camera_alignment
    loss = loss+refinement_safety_weight*safety
    with torch.no_grad():
        r0, t0 = pose_errors(coarse, target.c2w)
        r1, t1 = pose_errors(refined, target.c2w)
    return loss, {"loss": loss.detach().item(), "point_l1_m": point.detach().item(),
                  "point_epe_m": delta.detach().norm(dim=-1).mean().item(),
                  "coarse_point_l1_m": coarse_point.detach().item(), "anchor_l1_m": anchor.detach().item(),
                  "camera_alignment_l1_m": camera_alignment.detach().item(), "correspondence_loss": match.detach().item(),
                  "refinement_regression_penalty": safety.detach().item(),
                  "coarse_point_epe_m": (output['coarse_world_points'][valid]-target.world_points[valid]).detach().norm(dim=-1).mean().item(),
                  "rotation0_deg": r0.item(), "rotation1_deg": r1.item(),
                  "translation0_m": t0.item(), "translation1_m": t1.item()}
