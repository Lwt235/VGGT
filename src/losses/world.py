"""World metric supervision only: no fitting, centering, relative-depth loss."""

import torch
from torch.nn import functional as F


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


def world_loss(output, target):
    valid = target.valid
    if not valid.any():
        raise ValueError("No valid metric supervision pixels")
    delta = output["world_points"][valid]-target.world_points[valid]
    point = delta.abs().mean()
    coarse = output["T0"]
    refined = output["T1"]
    pose = sum((T[:3, 3]-target.c2w[:3, 3]).abs().mean() +
               rotation_geodesic(T[:3, :3], target.c2w[:3, :3]) for T in (coarse, refined))
    coarse_point = (output['coarse_world_points'][valid]-target.world_points[valid]).abs().mean()
    # Explicit patch-anchor supervision, evaluator/criterion only. It teaches
    # correspondences rather than feeding any query geometry to the model.
    hp, wp = output['rendered'].opacity.shape[:2]
    valid_patch = F.adaptive_avg_pool2d(valid.float()[None, None], (hp, wp))[0, 0]
    world_patch = F.adaptive_avg_pool2d((target.world_points*valid[..., None]).permute(2, 0, 1)[None], (hp, wp))[0].permute(1, 2, 0)/valid_patch.clamp_min(1e-6)[..., None]
    anchor_mask = valid_patch.flatten() > .25
    anchor = (output['anchors'][anchor_mask]-world_patch.reshape(-1, 3)[anchor_mask]).abs().mean()
    loss = point + .25*coarse_point + .5*pose + .2*anchor
    with torch.no_grad():
        r0, t0 = pose_errors(coarse, target.c2w)
        r1, t1 = pose_errors(refined, target.c2w)
    return loss, {"loss": loss.detach().item(), "point_l1_m": point.detach().item(),
                  "point_epe_m": delta.detach().norm(dim=-1).mean().item(),
                  "coarse_point_l1_m": coarse_point.detach().item(), "anchor_l1_m": anchor.detach().item(),
                  "rotation0_deg": r0.item(), "rotation1_deg": r1.item(),
                  "translation0_m": t0.item(), "translation1_m": t1.item()}
