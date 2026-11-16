"""Unaligned world metric supervision with optional relative depth auxiliary."""

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


def _legacy_world_loss(output, target, *, camera_weight=.5, refinement_safety_weight=0., anchor_weight=.2, matching_weight=.2):
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
    camera_alignment = (output.get('raw_camera_points', output['camera_points'])[valid]-camera_truth[valid]).abs().mean()
    # Training supervision only: the reference is detached, so this cannot
    # reward making T0 worse. Inference always returns the predicted T1.
    safety_reference = output.get('T0_read', coarse)
    r0_angle = rotation_geodesic(safety_reference[:3, :3], target.c2w[:3, :3])
    r1_angle = rotation_geodesic(refined[:3, :3], target.c2w[:3, :3])
    t0_error = (safety_reference[:3, 3]-target.c2w[:3, 3]).norm()
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


@torch.no_grad()
def surface_support_rate(means, covars, target):
    """Criterion-only label; never supplied to model inference."""
    if not len(means):
        return target.world_points.new_zeros(())
    means, covars = means.detach(), covars.detach()
    precision = torch.linalg.inv(covars)
    points = target.world_points[target.valid]
    covered = points.new_zeros(())
    for start in range(0, len(points), 2048):
        delta = points[start:start+2048, None]-means[None]
        distance = torch.einsum('pmi,mij,pmj->pm', delta, precision, delta)
        covered += (distance <= 9).any(-1).float().sum()
    return covered/max(1, len(points))


def open_set_matching_loss(output, target, erased_reference=None):
    """GT surface support and positive/negative labels live ONLY in criterion."""
    h, w = output['rendered'].opacity.shape[:2]
    valid = target.valid
    fraction = F.adaptive_avg_pool2d(valid.float()[None, None], (h, w))[0, 0]
    patch_valid = fraction.flatten() > .25
    world = F.adaptive_avg_pool2d((target.world_points*valid[..., None]).permute(2, 0, 1)[None], (h, w))[0].permute(1, 2, 0)/fraction.clamp_min(1e-6)[..., None]
    means, covars = output['anchor_support_means'].detach(), output['anchor_support_covars'].detach()
    with torch.no_grad():
        if len(means):
            precision = torch.linalg.inv(covars)
            diff = world.reshape(-1, 3)[:, None]-means[None]
            rho2 = torch.einsum('qmi,mij,qmj->qm', diff, precision, diff)
            minimum = rho2.min(-1).values
            teacher = (-rho2/2).softmax(-1)
            support_rate = surface_support_rate(means, covars, target)
        else:
            minimum = world.new_full((h*w,), float('inf'))
            teacher = world.new_zeros(h*w, 0)
            support_rate = world.new_zeros(())
        observable = bool(support_rate >= .05)
        positive = patch_valid & (minimum <= 9)
        negative = patch_valid & (minimum > 16)
        if erased_reference is not None:
            original_means, original_covars, erased = erased_reference
            delta = world.reshape(-1, 3)[:, None]-original_means.detach()[None]
            original_rho2 = torch.einsum('qmi,mij,qmj->qm', delta, torch.linalg.inv(original_covars.detach()), delta)
            nearest_rho, nearest = original_rho2.min(-1)
            removed = patch_valid & (nearest_rho <= 9) & erased[nearest] & (minimum > 9)
            positive &= ~removed
            negative |= removed
        if not observable:
            positive.zero_()
            negative = patch_valid
    W = output['matching_attention'].clamp_min(1e-9)
    terms = W.new_zeros(h*w)
    if positive.any():
        terms[positive] = -(teacher[positive]*W[positive, :-1].log()).sum(-1)
    if negative.any():
        terms[negative] = -W[negative, -1].log()
    # Equal class contributions: many supported patches cannot drown the
    # sparse true-missing negatives. Single-class blind frames remain valid.
    classes = [terms[mask].mean() for mask in (positive, negative) if mask.any()]
    match = torch.stack(classes).sum() if classes else W.sum()*0
    anchor = (output['anchors'][positive]-world.reshape(-1, 3)[positive]).abs().mean() if positive.any() else output['anchors'].sum()*0
    return match, anchor, support_rate, observable


def world_loss(output, target, *, camera_weight=.5, refinement_safety_weight=0., anchor_weight=.2, matching_weight=.2, depth_weight=1., relative_depth_weight=0.):
    if 'matching_attention' not in output:
        return _legacy_world_loss(output, target, camera_weight=camera_weight, refinement_safety_weight=refinement_safety_weight,
                                  anchor_weight=anchor_weight, matching_weight=matching_weight)
    base, metrics = _legacy_world_loss(output, target, camera_weight=camera_weight,
                                       refinement_safety_weight=refinement_safety_weight, anchor_weight=0., matching_weight=0.)
    match, anchor, support_rate, observable = open_set_matching_loss(output, target)
    hp, wp = output['rendered'].opacity.shape[:2]
    valid = target.valid.float()
    fraction = F.adaptive_avg_pool2d(valid[None, None], (hp, wp))[0, 0].flatten()
    depth = (F.adaptive_avg_pool2d((target.depth*valid)[None, None], (hp, wp))[0, 0].flatten()/fraction.clamp_min(1e-6))
    mask = fraction > .25
    depth_error = (output['z_cam'][mask]-depth[mask]).abs()
    z_loss = depth_error.mean()
    relative_z = (depth_error/depth[mask].clamp_min(.2)).mean()
    confidence_loss = F.binary_cross_entropy(output['c_depth'][mask].clamp(1e-6, 1-1e-6), torch.exp(-depth_error.detach()/.5))
    camera_truth = transform_points(target.world_points, invert_se3(target.c2w))
    camera_delta = output.get('raw_camera_points', output['camera_points'])[target.valid]-camera_truth[target.valid]
    # No random-world pose/point/anchor loss on the unobservable frame.
    loss = base+anchor_weight*anchor if observable else camera_weight*camera_delta.abs().mean()
    loss = loss+matching_weight*match+depth_weight*(z_loss+.05*confidence_loss)
    relative_camera = (camera_delta[:, 2].abs()/camera_truth[target.valid][:, 2].clamp_min(.2)).mean()
    loss = loss+relative_depth_weight*(relative_z+relative_camera)
    metrics.update(relative_depth_l1=float(relative_z.detach()), relative_camera_depth_l1=float(relative_camera.detach()), loss=float(loss.detach()), anchor_l1_m=float(anchor.detach()), correspondence_loss=float(match.detach()),
                   surface_support_rate=float(support_rate), world_supervision_enabled=float(observable),
                   z_cam_l1_m=float(z_loss.detach()), camera_epe_m=float(camera_delta.detach().norm(dim=-1).mean()),
                   dustbin_mean=float(output['dustbin_probability'].detach().mean()),
                   dustbin_rejection_rate=float((output['dustbin_probability'].detach() > .5).float().mean()),
                   match_mass=float(output['match_mass'].detach()))
    if 'raw_camera_points' in output:
        metrics['anchored_camera_epe_m'] = float((output['camera_points'][target.valid]-camera_truth[target.valid]).detach().norm(dim=-1).mean())
    if 'T0_read' in output:
        rr, tr = pose_errors(output['T0_read'].detach(), target.c2w)
        metrics.update(rotation_read_deg=float(rr), translation_read_m=float(tr),
                       read_jitter_applied=float(output['read_jitter_applied']))
    return loss, metrics
