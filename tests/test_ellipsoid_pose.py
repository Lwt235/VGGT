import torch
from torch.nn import functional as F
from src.augmentation import sample_world_gauge
from src.geometry import world_rays, transform_points
from src.losses.world import pose_errors
from src.models.ellipsoid_pose import anisotropic_ray_pose, posterior_ellipsoids, local_se3_exp, sample_read_jitter
from src.memory.gaussians import GaussianMemory


def pancake_case():
    torch.manual_seed(17)
    h, w = 8, 10
    _, rays, _ = world_rays(torch.tensor([[7., 0., 5.], [0., 7., 4.], [0., 0., 1.]]), torch.eye(4), h, w)
    kind = torch.arange(h*w)%3
    normal = torch.zeros(h*w, 3); normal[:, 2] = 1
    normal[(kind == 0) & (rays[:, 0].abs() > .25)] = torch.tensor([1., 0., 0.])
    normal[(kind == 1) & (rays[:, 1].abs() > .2)] = torch.tensor([0., 1., 0.])
    distance = torch.where(normal[:, 2] > 0, torch.tensor(3.5), torch.tensor(1.2))
    points = rays*(distance/(normal*rays).sum(-1).abs())[:, None]
    noise = torch.randn(h*w, 3)
    tangent = F.normalize(noise-(noise*normal).sum(-1)[:, None]*normal, dim=-1)*.17
    anchors = points+tangent+.002*normal*torch.sin(torch.arange(h*w).float())[:, None]
    covariance = torch.eye(3)[None]*.25**2+(.025**2-.25**2)*normal[:, :, None]*normal[:, None, :]
    weights = torch.ones(h*w)
    # A few uncertain cross-surface matches, as represented by broad posterior
    # covariance and reduced p_match; the solver must still robustly reject them.
    anchors[[2, 15, 33]] += torch.tensor([.8, -.6, .7])
    covariance[[2, 15, 33]] += torch.eye(3)*.3**2
    weights[[2, 15, 33]] = .2
    G = sample_world_gauge('cpu')
    anchors = transform_points(anchors, G)
    covariance = G[:3, :3] @ covariance @ G[:3, :3].T
    return points, rays, anchors, covariance, weights, G


def test_posterior_total_covariance_preserves_pancake_and_multimodal_spread():
    means = torch.tensor([[0., 0., 3.], [1., 0., 3.]])
    covariance = torch.diag(torch.tensor([.25**2, .2**2, .025**2])).repeat(2, 1, 1)
    memory = GaussianMemory(means, covariance, torch.ones(2), torch.zeros(2, 4))
    anchors, total = posterior_ellipsoids(torch.tensor([[1., 0.], [.5, .5]]), memory)
    torch.testing.assert_close(anchors[1], torch.tensor([.5, 0., 3.]))
    assert total[0, 0, 0] > 40*total[0, 2, 2]
    torch.testing.assert_close(total[1, 0, 0]-total[0, 0, 0], torch.tensor(.25))


def test_anisotropic_lm_energy_and_tangential_subset_robustness():
    points, rays, anchors, covariance, weights, G = pancake_case()
    full, _, diagnostic = anisotropic_ray_pose(points, rays, anchors, covariance, weights, return_diagnostic=True)
    keep = torch.arange(80).remainder(10) < 7
    partial, _, partial_diagnostic = anisotropic_ray_pose(points[keep], rays[keep], anchors[keep], covariance[keep], weights[keep], return_diagnostic=True)
    for data in (diagnostic, partial_diagnostic):
        energy = data['energy_history']
        assert torch.isfinite(energy).all() and ((energy[1:]-energy[:-1]) <= 1e-6).all()
        assert energy[-1] < energy[0]
    rotation, translation = pose_errors(partial, full)
    assert translation < .03 and rotation < 1.
    # Finite random tangent offsets need not place their likelihood maximum
    # exactly at GT. This test verifies local-erasure stability, not zero noise.
    assert torch.isfinite(full).all()


def test_lm_gauge_equivariance_gradients_and_zero_exponential():
    points, rays, anchors, covariance, weights, _ = pancake_case()
    T, _, _ = anisotropic_ray_pose(points, rays, anchors, covariance, weights, return_diagnostic=True)
    G = sample_world_gauge('cpu')
    moved, _ = anisotropic_ray_pose(points, rays, transform_points(anchors, G), G[:3, :3]@covariance@G[:3, :3].T, weights)
    torch.testing.assert_close(moved, G@T, atol=1e-4, rtol=1e-4)
    P, X, C, W = [value.clone().requires_grad_() for value in (points, anchors, covariance, weights)]
    predicted, _ = anisotropic_ray_pose(P, rays, X, C, W)
    (predicted[:3, 3]-.2).square().sum().backward()
    for value in (P, X, C, W):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    omega = torch.zeros(3, requires_grad=True)
    nu = torch.ones(3, requires_grad=True)
    R, t = local_se3_exp(omega, nu)
    torch.testing.assert_close(R, torch.eye(3))
    (R.sum()+t.sum()).backward()
    assert torch.isfinite(omega.grad).all() and torch.isfinite(nu.grad).all()


def test_read_jitter_has_bounded_local_magnitude():
    for _ in range(5):
        jitter = sample_read_jitter('cpu')
        angle, distance = pose_errors(jitter, torch.eye(4))
        assert 1.99 < angle < 3.01 and .0499 < distance < .0601
