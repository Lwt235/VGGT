"""Local, finite-radius metric scale transport for predicted streaming writes.

This deliberately differs from the Step2 fixed-map intervention path. A stream
must carry a known surface's metric scale into the adjacent unobserved frontier.
No spatial mean, historical image, GT depth or global scale alignment is used.
"""
import torch
from torch.nn import functional as F
from src.geometry import pixel_grid


def propagate_scale(ratio, reliability, iterations=4):
    """Normalized local diffusion; zero support is exactly the identity.

    Known high-confidence cells are clamped boundary conditions. Confidence
    decays with graph distance, so distant unsupported regions return to one.
    """
    shape = ratio.shape
    source = reliability.reshape(1, 1, *shape[:2]).clamp(0, 1)
    values = ratio.clamp(.25, 4.).log().reshape_as(source)
    mass, moment = source, source*values
    kernel = ratio.new_tensor([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]])[None, None]/6
    for _ in range(iterations):
        spread = F.conv2d(mass, kernel, padding=1)*.85
        spread_value = F.conv2d(moment, kernel, padding=1)*.85
        mass = source+(1-source)*spread
        moment = source*values+(1-source)*spread_value
    local = moment/mass.clamp_min(1e-8)
    # Saturate confidence for immediate frontiers without forcing a global
    # correction through isolated weak observations or large disconnected holes.
    gain = (mass/.03).clamp(0, 1)
    return (gain*local).exp().reshape(shape), mass.reshape(shape)


def anchor_streaming_geometry(reader, camera_points, rendered, K_cv):
    hp, wp = rendered.depth.shape[:2]
    h, w = camera_points.shape[:2]
    raw = F.adaptive_avg_pool2d(camera_points[..., 2][None, None], (hp, wp))[0, 0, ..., None]
    variance = rendered.depth_variance/rendered.depth.square().clamp_min(.04)
    attrs = torch.cat((rendered.features, rendered.opacity, variance.log1p(), rendered.depth.log1p(),
                       (rendered.depth-raw)/rendered.depth.clamp_min(.2)), -1)
    confidence = reader.depth_confidence(attrs).sigmoid()
    physical = (rendered.opacity/.25).clamp(0, 1)/(1+variance)
    reliable = physical*(.9+.1*confidence)*(raw > .05)*(rendered.depth > .05)
    ratio = (rendered.depth/raw.clamp_min(.05)).clamp(.25, 4.)
    scale, transported = propagate_scale(ratio, reliable)
    correction = raw*(scale-1)
    correction = F.interpolate(correction.permute(2, 0, 1)[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    rays = pixel_grid(h, w, device=K_cv.device, dtype=K_cv.dtype, center=0.) @ torch.linalg.inv(K_cv).T
    return camera_points+rays*correction[..., None], confidence, dict(
        frontier_scale=scale, frontier_reliability=transported, physical_reliability=reliable)
