"""Metric-scale recovery, gauge-invariant reads and final-pose raster gradients."""
import os
from unittest.mock import patch
import pytest
import torch
from src.augmentation import sample_world_gauge
from src.geometry import world_rays, transform_points
from src.losses.world import pose_errors
from src.models.ellipsoid_pose import anisotropic_ray_pose
from src.memory.gaussians import GaussianMemory
from src.readout.render import RenderReader, RenderedMemory
from src.readout.parameter_pe import ParameterPEReader


@pytest.mark.parametrize('underestimate', [2., 3.])
def test_far_metric_scale_with_biased_monocular_depth(underestimate):
    torch.manual_seed(17)
    K = torch.tensor([[10., 0., 6.], [0., 10., 4.], [0., 0., 1.]])
    _, rays, dz = world_rays(K, torch.eye(4), 8, 12)
    camera = rays/dz[:, None]*(6+2*torch.rand(96, 1))
    G = sample_world_gauge('cpu')
    anchors = transform_points(camera, G)
    covariance = torch.diag(torch.tensor([.2**2, .15**2, .03**2])).repeat(96, 1, 1)
    covariance = G[:3, :3]@covariance@G[:3, :3].T
    points = (camera/underestimate).requires_grad_()
    T, _, info = anisotropic_ray_pose(points, rays, anchors, covariance, torch.ones(96), metric_scale=True, return_diagnostic=True)
    rotation, translation = pose_errors(T, G)
    assert rotation < .05 and translation < .01
    torch.testing.assert_close(info['scatter_scale'], torch.tensor(underestimate, dtype=torch.float64), atol=1e-5, rtol=1e-5)
    assert info['radial_precision'].max() < .0015
    assert (info['energy_history'][1:] <= info['energy_history'][:-1]+1e-8).all()
    H = sample_world_gauge('cpu')
    moved, _ = anisotropic_ray_pose(points, rays, transform_points(anchors, H), H[:3, :3]@covariance@H[:3, :3].T, torch.ones(96), metric_scale=True)
    torch.testing.assert_close(moved, H@T, atol=1e-4, rtol=1e-4)
    T[:3, 3].square().sum().backward()
    assert torch.isfinite(points.grad).all() and points.grad.abs().sum() > 0


def read_case(device='cpu', dtype=torch.float64):
    torch.manual_seed(17)
    K = torch.tensor([[6., 0., 2.], [0., 6., 1.5], [0., 0., 1.]], device=device, dtype=dtype)
    _, rays, dz = world_rays(K, torch.eye(4, device=device, dtype=dtype), 3, 4)
    means = rays/dz[:, None]*(5+torch.rand(12, 1, device=device, dtype=dtype))
    covariance = torch.diag(means.new_tensor([.09, .04, .0009])).repeat(12, 1, 1)
    features = torch.randn(12, 4, device=device, dtype=dtype)
    memory = GaussianMemory(means, covariance, means.new_full((12,), .8), features)
    normal = means.new_tensor([0., 0., -1.]).expand(3, 4, 3)
    rendered = RenderedMemory(means.reshape(3, 4, 3), features.reshape(3, 4, 4), means[:, 2].reshape(3, 4, 1),
                              means.new_full((3, 4, 1), .2), means.new_full((3, 4, 1), .8), normal)
    return K, memory, rendered


def test_o5_o6_camera_frame_world_gauge_invariance_with_open_gates():
    # Float64 isolates algebraic invariance from float32 raster roundoff and
    # high-frequency phase amplification. Real raster path tested below.
    K, memory, rendered = read_case()
    tokens = torch.randn(12, 16, dtype=torch.float64)
    pose = torch.eye(4, dtype=torch.float64)
    G = sample_world_gauge('cpu').double()
    # Re-orthogonalize the float32 sampled rotation in double via Rodrigues.
    from src.models.ellipsoid_pose import local_se3_exp
    G[:3, :3] = local_se3_exp(torch.tensor([.7, -.4, 1.2], dtype=torch.float64), torch.zeros(3, dtype=torch.float64))[0]
    moved = GaussianMemory(transform_points(memory.means, G), G[:3, :3]@memory.covars@G[:3, :3].T, memory.alpha, memory.features)
    render_g = RenderedMemory(transform_points(rendered.world, G), rendered.features, rendered.depth, rendered.depth_variance,
                              rendered.opacity, rendered.normal@G[:3, :3].T)
    o5 = RenderReader(4, 16, surface_normals=True, camera_frame=True).double()
    o6 = ParameterPEReader(16, 4, heads=4, camera_frame=True).double()
    with torch.no_grad():
        o5.gate.fill_(.2); o6.gate.fill_(.2)
    with patch.object(o5, 'render', return_value=rendered):
        a, _ = o5(tokens, memory, pose, K, 3, 4)
    with patch.object(o5, 'render', return_value=render_g):
        b, _ = o5(tokens, moved, G, K, 3, 4)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    a, da = o6(tokens, memory, pose, K, 3, 4, rendered.depth, rendered=rendered, return_attention=True)
    b, db = o6(tokens, moved, G, K, 3, 4, rendered.depth, rendered=render_g, return_attention=True)
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(da['attention'], db['attention'], atol=1e-9, rtol=1e-9)


@pytest.mark.gpu
def test_final_t1_rerender_anchor_has_pose_gradient_and_real_read_gauge_invariance():
    if not torch.cuda.is_available():
        if os.environ.get('REQUIRE_GSPLAT') == '1':
            pytest.fail('CUDA required')
        pytest.skip('CUDA unavailable')
    K, memory, _ = read_case('cuda', torch.float32)
    reader = RenderReader(4, 16, surface_normals=True, camera_frame=True).cuda()
    with torch.no_grad(): reader.gate.fill_(.2)
    pose = torch.eye(4, device='cuda', requires_grad=True)
    tokens = torch.randn(12, 16, device='cuda')
    a, rendered = reader(tokens, memory, pose, K, 3, 4)
    raw = memory.means.reshape(3, 4, 3)*.7
    anchored, _ = reader.anchor_camera_geometry(raw, rendered, K)
    # Isolate the raster/anchoring pose path: do not transform output by T1.
    anchored[..., 2].mean().backward()
    assert torch.isfinite(pose.grad).all() and pose.grad[:3].abs().sum() > 1e-4
    G = sample_world_gauge('cuda')
    moved = GaussianMemory(transform_points(memory.means, G), G[:3, :3]@memory.covars@G[:3, :3].T, memory.alpha, memory.features)
    b, render_g = reader(tokens, moved, G, K, 3, 4)
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    pe = ParameterPEReader(16, 4, heads=4, camera_frame=True).cuda()
    with torch.no_grad(): pe.gate.fill_(.2)
    pa, _ = pe(tokens, memory, pose.detach(), K, 3, 4, rendered.depth, rendered=rendered)
    pb, _ = pe(tokens, moved, G, K, 3, 4, render_g.depth, rendered=render_g)
    torch.testing.assert_close(pa, pb, atol=1e-3, rtol=1e-3)
