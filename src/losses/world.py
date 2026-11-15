"""World metric supervision only: no fitting, centering, relative-depth loss."""

import torch


def pose_errors(prediction, target):
    relative = prediction[:3, :3].T @ target[:3, :3]
    skew = torch.stack((relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]))
    sin = .5*skew.norm()
    cos = .5*(relative.trace()-1)
    degrees = torch.rad2deg(torch.atan2(sin, cos.clamp(-1, 1)))
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
               (T[:3, :3]-target.c2w[:3, :3]).square().mean() for T in (coarse, refined))
    loss = point + .5*pose
    with torch.no_grad():
        r0, t0 = pose_errors(coarse, target.c2w)
        r1, t1 = pose_errors(refined, target.c2w)
    return loss, {"loss": loss.detach().item(), "point_l1_m": point.detach().item(),
                  "point_epe_m": delta.detach().norm(dim=-1).mean().item(),
                  "rotation0_deg": r0.item(), "rotation1_deg": r1.item(),
                  "translation0_m": t0.item(), "translation1_m": t1.item()}
