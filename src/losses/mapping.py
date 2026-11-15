"""Self-observation of permitted RGB-D mapping inputs, never query poses."""
import torch
from torch.nn import functional as F
from src.geometry import patch_intrinsics


def mapping_observation_loss(rendered, mapping_truth):
    hp, wp = rendered.depth.shape[:2]
    valid = mapping_truth.valid.float()
    fraction = F.adaptive_avg_pool2d(valid[None, None], (hp, wp))[0, 0]
    mask = fraction > .25
    depth = F.adaptive_avg_pool2d((mapping_truth.depth*valid)[None, None], (hp, wp))[0, 0]/fraction.clamp_min(1e-6)
    world = F.adaptive_avg_pool2d((mapping_truth.world_points*valid[..., None]).permute(2, 0, 1)[None], (hp, wp))[0].permute(1, 2, 0)/fraction.clamp_min(1e-6)[..., None]
    depth_error = (rendered.depth[..., 0][mask]-depth[mask]).abs().mean()
    world_error = (rendered.world[mask]-world[mask]).norm(dim=-1).mean()
    coverage = -rendered.opacity[..., 0][mask].clamp_min(1e-6).log().mean()
    loss = .2*depth_error+.2*world_error+.05*coverage
    return loss, dict(mapping_depth_l1_m=float(depth_error.detach()), mapping_world_epe_m=float(world_error.detach()),
                      mapping_coverage_loss=float(coverage.detach()), mapping_mean_opacity=float(rendered.opacity[..., 0][mask].detach().mean()),
                      mapping_loss=float(loss.detach()))


def supervise_mapping_observation(model, memory, mapping_observation):
    rgb_k, mapping_truth = mapping_observation
    h, w = rgb_k.rgb.shape[-2:]
    # Explicitly allowed mapping pose, separate from model(query,memory).
    rendered = model.render_reader.render(memory, mapping_truth.c2w, patch_intrinsics(rgb_k.K), h//14, w//14)
    return mapping_observation_loss(rendered, mapping_truth)
