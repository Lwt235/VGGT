"""Independent geometric checks, without importing old experiment code.

Real gsplat checks skip explicitly when absent. REQUIRE_GSPLAT=1 makes missing
CUDA/dependency an error for the final smoke gate. No CPU raster substitute.
"""

import importlib.util
import math
import os

import pytest
import torch

from src.geometry import (covariance_from_quat_scale, intrinsics_cv_to_edge,
                          invert_se3, patch_intrinsics, pixel_grid, project,
                          quat_wxyz_to_matrix, ray_ellipsoid_distance,
                          transform_points, unproject, world_rays)


def camera(dtype=torch.float64):
    K = torch.tensor([[180., 0., 51.3], [0., 190., 36.7], [0., 0., 1.]], dtype=dtype)
    c2w = torch.eye(4, dtype=dtype)
    c2w[:3, :3] = quat_wxyz_to_matrix(torch.tensor([.8, .2, -.3, .1], dtype=dtype))
    c2w[:3, 3] = torch.tensor([.7, -1.3, 2.1], dtype=dtype)
    return K, c2w


def test_wxyz_rotation_and_anisotropic_covariance():
    q = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)], dtype=torch.float64)
    R = quat_wxyz_to_matrix(q)
    torch.testing.assert_close(R @ q.new_tensor([1., 0., 0.]), q.new_tensor([0., 1., 0.]))
    covariance = covariance_from_quat_scale(q, q.new_tensor([.1, .3, .05]))
    torch.testing.assert_close(covariance.diag(), q.new_tensor([.09, .01, .0025]))
    torch.testing.assert_close(quat_wxyz_to_matrix(-q), R)


def test_vggt_w2c_and_gsplat_viewmat_are_same_rigid_transform():
    _, c2w = camera()
    w2c = invert_se3(c2w)
    torch.testing.assert_close(w2c @ c2w, torch.eye(4, dtype=c2w.dtype))
    torch.testing.assert_close(invert_se3(w2c[:3]), c2w)
    p = torch.tensor([[1., 2., 3.], [-.3, .7, 4.]], dtype=c2w.dtype)
    torch.testing.assert_close(transform_points(transform_points(p, c2w), w2c), p)


@pytest.mark.parametrize("batch", [False, True])
def test_world_unprojection_reprojection(batch):
    K, c2w = camera()
    depth = torch.linspace(.4, 3., 7*9, dtype=K.dtype).reshape(7, 9)
    if batch:
        K, c2w, depth = K.repeat(2, 1, 1), c2w.repeat(2, 1, 1), depth.repeat(2, 1, 1)
    points = unproject(depth, K, c2w)
    uv, z = project(points, K, invert_se3(c2w))
    expected = pixel_grid(7, 9, dtype=K.dtype)[..., :2]
    if batch:
        expected = expected.repeat(2, 1, 1, 1)
    torch.testing.assert_close(uv, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(z, depth)


def test_official_vggt_world_unprojection_matches_independent_geometry():
    import sys
    import numpy as np
    from src.runtime import load_config
    source = load_config()["assets"]["official_vggt_source"]["path"]
    # Read-only official third-party dependency, never old experiment modules.
    sys.path.insert(0, source)
    from vggt.utils.geometry import depth_to_world_coords_points
    K, c2w = camera(torch.float32)
    depth = torch.linspace(.5, 3., 63).reshape(7, 9)
    actual, _, _ = depth_to_world_coords_points(depth.numpy(), invert_se3(c2w)[:3].numpy(), K.numpy())
    expected = unproject(depth, K, c2w)
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-6)


def test_axes_camera_z_not_ray_length_and_image_coordinates():
    K = torch.tensor([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
    p = unproject(torch.full((3, 3), 2.), K, torch.eye(4))
    assert p[1, 2, 0] > 0 and p[2, 1, 1] > 0  # right/down
    assert p[0, 0].norm() > p[0, 0, 2]  # length != z
    torch.testing.assert_close(p[1, 1], torch.tensor([0., 0., 2.]))


def test_patch_intrinsics_preserve_physical_patch_center_ray():
    K, c2w = camera()
    Kp_edge = patch_intrinsics(K)
    Kp_cv = patch_intrinsics(K, output_center=0.)
    grid = pixel_grid(6, 8, dtype=K.dtype, center=.5)
    original_cv = grid.clone()
    original_cv[..., :2] = original_cv[..., :2] * 14 - .5
    image_ray = torch.einsum("ij,hwj->hwi", torch.linalg.inv(K), original_cv)
    patch_ray = torch.einsum("ij,hwj->hwi", torch.linalg.inv(Kp_edge), grid)
    torch.testing.assert_close(image_ray, patch_ray)
    torch.testing.assert_close(intrinsics_cv_to_edge(Kp_cv), Kp_edge)
    # A naive principal_point / 14 loses the original half-pixel convention.
    assert abs(Kp_edge[0, 2] - K[0, 2] / 14) > .03
    points = unproject(torch.ones(6, 8, dtype=K.dtype), Kp_edge, c2w, center=.5)
    uv, _ = project(points, K, invert_se3(c2w))
    torch.testing.assert_close(uv, original_cv[..., :2])


def test_ray_ellipsoid_half_ray_minimum_and_z_conversion():
    means = torch.tensor([[1., 0., 3.], [0., 0., -2.]])
    covars = torch.diag(torch.tensor([.25, 1., 4.])).repeat(2, 1, 1)
    rho, t = ray_ellipsoid_distance(torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]), means, covars)
    torch.testing.assert_close(t, torch.tensor([[3., 0.]]))
    torch.testing.assert_close(rho, torch.tensor([[4., 1.]]))
    K, c2w = camera()
    _, directions, dz = world_rays(K, c2w, 2, 3)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones_like(dz))
    assert (dz > 0).all() and (dz <= 1).all()


@pytest.fixture
def cuda_gsplat():
    missing = importlib.util.find_spec("gsplat") is None
    if missing or not torch.cuda.is_available():
        reason = "Real gsplat/CUDA missing; raster convention NOT verified"
        if os.getenv("REQUIRE_GSPLAT") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    # Host GPU selection/check must have occurred before invoking GPU tests.
    from src.runtime import configure_local_storage
    configure_local_storage()
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("VGGT_GPU_MEMORY_FRACTION", ".08")), 0)
    from gsplat import rasterization
    return rasterization


@pytest.mark.gpu
@pytest.mark.parametrize("opacity", [.25, .85])
def test_real_gsplat_expected_world_coordinates_depth_and_coverage(cuda_gsplat, opacity):
    raster = cuda_gsplat
    device = "cuda"
    K, c2w = camera(torch.float32)
    K = torch.tensor([[60., 0., 16.], [0., 60., 16.], [0., 0., 1.]])
    K, c2w = K.to(device), c2w.to(device)
    means = transform_points(K.new_tensor([[0., 0., 2.]]), c2w).requires_grad_()
    quat = K.new_tensor([[1., 0., 0., 0.]])
    scales = K.new_tensor([[.08, .09, .03]])
    colors = torch.cat((means, means.new_tensor([[2., 4.]])), -1)
    image, alpha, _ = raster(means=means, quats=quat, scales=scales,
                             opacities=K.new_tensor([opacity]), colors=colors,
                             viewmats=invert_se3(c2w)[None],
                             Ks=intrinsics_cv_to_edge(K)[None], width=33, height=33,
                             render_mode="RGB", packed=False)
    covered = alpha[0, ..., 0] > .05
    expected = image[0][covered] / alpha[0][covered].clamp_min(1e-8)
    target = colors[0].expand_as(expected)
    torch.testing.assert_close(expected, target, atol=2e-4, rtol=2e-4)
    # Half-integer sampling + corrected K puts the peak at cv pixel (16,16).
    assert alpha[0, ..., 0].argmax().item() == 16*33+16
    assert not torch.allclose(image[0][covered], target)  # accumulated != expectation
    variance = expected[:, 4] - expected[:, 3].square()
    assert variance.abs().max() < 2e-4
    image.sum().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()


@pytest.mark.gpu
def test_real_gsplat_occlusion_depth_variance_and_wxyz(cuda_gsplat):
    raster = cuda_gsplat
    device = "cuda"
    means = torch.tensor([[0., 0., 2.], [0., 0., 3.]], device=device)
    # Foreground elongated horizontally: a WXYZ z rotation makes it vertical.
    q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)], [1., 0., 0., 0.]], device=device)
    scales = means.new_tensor([[.2, .04, .01], [.3, .3, .01]])
    attributes = torch.cat((means, means[:, 2:3].square()), -1)
    K_edge = means.new_tensor([[50., 0., 16.5], [0., 50., 16.5], [0., 0., 1.]])
    image, alpha, _ = raster(means=means, quats=q, scales=scales,
                             opacities=means.new_tensor([.8, .6]), colors=attributes,
                             viewmats=torch.eye(4, device=device)[None], Ks=K_edge[None],
                             width=33, height=33, render_mode="RGB", packed=False)
    a = alpha[0, 16, 16, 0]
    expectation = image[0, 16, 16] / a
    # Central weights: .8 and (1-.8)*.6; boundaries are intentionally excluded.
    torch.testing.assert_close(a, a.new_tensor(.92), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(expectation[2], a.new_tensor((.8*2+.12*3)/.92), atol=2e-3, rtol=2e-3)
    assert (expectation[3] - expectation[2].square()) > .05
    # Compare only foreground morphology, using its own rendered scalar channel.
    fg, _, _ = raster(means=means[:1], quats=q[:1], scales=scales[:1],
                       opacities=means.new_tensor([.8]), colors=means.new_ones(1, 1),
                       viewmats=torch.eye(4, device=device)[None], Ks=K_edge[None],
                       width=33, height=33, packed=False)
    assert fg[0, 20, 16, 0] > fg[0, 16, 20, 0]
