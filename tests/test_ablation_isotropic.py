"""Spherical ablation: scalar geometry throughout writing, fusion and pose."""
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.geometry import transform_points
from src.memory.gaussians import GaussianMemory, SpatialMomentFusion, TokenGaussianWriter
from src.models.deep_refinement import DeepEllipsoidRefinement
from src.models.ellipsoid_pose import local_se3_exp, posterior_ellipsoids
from src.models.spatial_pose import SpatialBootstrap


def assert_spherical(covars):
    expected = covars[..., 0, 0][..., None, None]*torch.eye(
        3, dtype=covars.dtype, device=covars.device)
    torch.testing.assert_close(covars, expected, atol=0, rtol=0)


def no_eigen():
    return patch.multiple(torch.linalg, eigh=forbidden_eigen, eigvalsh=forbidden_eigen)


def forbidden_eigen(*args, **kwargs):
    raise AssertionError('Spherical memory must not extract eigenvalues/normals')


def test_scalar_writer_has_no_orientation_and_visual_latent_only():
    torch.manual_seed(17)
    writer = TokenGaussianWriter(8, 6, isotropic=True).double()
    assert writer.geometry.out_features == 5
    assert not hasattr(writer, 'base_shape')
    tokens = torch.randn(6, 8, dtype=torch.double, requires_grad=True)
    depth = torch.linspace(1., 3., 6, dtype=torch.double).reshape(2, 3).requires_grad_()
    K = torch.tensor([[4., 0., 1.5], [0., 4., 1.], [0., 0., 1.]], dtype=torch.double)
    pose = torch.eye(4, dtype=torch.double)
    pose[:3, :3], _ = local_se3_exp(torch.tensor([.3, -.2, .1], dtype=torch.double), torch.zeros(3, dtype=torch.double))
    with no_eigen():
        memory = writer(tokens, depth, K, pose)
        other = writer(tokens, depth*2, K*torch.tensor([[2., 1., 1.], [1., 2., 1.], [1., 1., 1.]]), torch.eye(4, dtype=torch.double))
    assert_spherical(memory.covars)
    raw = writer.geometry(tokens)
    expected = (depth.flatten()/8*torch.exp(.7*raw[:, 3].tanh())).square()
    torch.testing.assert_close(memory.covars[:, 0, 0], expected)
    torch.testing.assert_close(memory.features, writer.latent(tokens), atol=0, rtol=0)
    torch.testing.assert_close(memory.features, other.features, atol=0, rtol=0)
    (memory.means.square().mean()+memory.covars.sum()+memory.features.square().mean()+memory.alpha.sum()).backward()
    for value in (tokens, depth, writer.geometry.weight, writer.latent.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert (writer.geometry.weight.grad.abs().sum(-1) > 0).all()


def test_scalar_fusion_projects_complete_moment_and_uses_first_camera_grid():
    means = torch.tensor([[.12, .11, .13], [.31, .13, .11], [1.2, .1, .1]], dtype=torch.double, requires_grad=True)
    covariance = torch.diag_embed(torch.tensor([[.01, .04, .04], [.04, .01, .04], [.04, .04, .01]], dtype=torch.double)).requires_grad_()
    alpha = torch.tensor([.3, .7, .6], dtype=torch.double, requires_grad=True)
    features = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    memory = GaussianMemory(means, covariance, alpha, features)
    fusion = SpatialMomentFusion(min_cell=1., isotropic=True).double()
    with no_eigen():
        fused = fusion(memory, torch.eye(4, dtype=torch.double))
    # Perpendicular thin axes share one voxel: no normal filter remains.
    assert len(fused) == 2
    assert_spherical(fused.covars)
    weights = alpha[:2]/alpha[:2].sum()
    center = (weights[:, None]*means[:2]).sum(0)
    trace = (weights*(covariance[:2].diagonal(dim1=-2, dim2=-1).sum(-1)+(means[:2]-center).square().sum(-1))).sum()
    torch.testing.assert_close(fused.covars[0], torch.eye(3, dtype=torch.double)*(trace/3+1e-8))
    G = torch.eye(4, dtype=torch.double)
    G[:3, :3], _ = local_se3_exp(torch.tensor([.4, -.7, .2], dtype=torch.double), torch.zeros(3, dtype=torch.double))
    G[:3, 3] = torch.tensor([.6, -.8, .3])
    moved = GaussianMemory(transform_points(means, G), G[:3, :3]@covariance@G[:3, :3].T, alpha, features)
    with no_eigen():
        transformed = fusion(moved, G)
    torch.testing.assert_close(transformed.means, transform_points(fused.means, G))
    torch.testing.assert_close(transformed.covars, fused.covars)
    torch.testing.assert_close(transformed.features, fused.features)
    (fused.means.square().sum()+fused.covars.sum()+fused.features.square().sum()+fused.alpha.sum()).backward()
    for value in (means, covariance, alpha, features, fusion.feature_gate, fusion.opacity_gate):
        assert value.grad is not None and torch.isfinite(value.grad).all()


def test_spherical_posterior_preserves_between_center_trace_and_gradients():
    means = torch.tensor([[0., 0., 3.], [1., 0., 3.]], requires_grad=True)
    covars = torch.diag(torch.tensor([.04, .01, .0025])).repeat(2, 1, 1).requires_grad_()
    logits = torch.tensor([[2., -1.], [.3, .3]], requires_grad=True)
    memory = GaussianMemory(means, covars, torch.ones(2), torch.zeros(2, 3))
    weights = logits.softmax(-1)
    with no_eigen():
        anchors, spherical = posterior_ellipsoids(weights, memory, isotropic=True)
    _, full = posterior_ellipsoids(weights, memory)
    assert_spherical(spherical)
    torch.testing.assert_close(spherical.diagonal(dim1=-2, dim2=-1).sum(-1), full.diagonal(dim1=-2, dim2=-1).sum(-1))
    torch.testing.assert_close(spherical[1, 0, 0], torch.tensor((.0525+.25)/3+.03**2))
    (anchors.square().sum()+spherical.sum()).backward()
    for value in (means, covars, logits):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_t0_t1_spherical_posterior_and_scalar_matching_keys_without_eigen():
    torch.manual_seed(17)
    h, w, dim, latent = 3, 4, 8, 12
    model = SpatialBootstrap(dim, latent, ray_bundle=True, anisotropic=True,
                             metric_scale=True, planar_safe=True, isotropic=True)
    deep = DeepEllipsoidRefinement(dim, latent, isotropic=True)
    tokens, shallow = torch.randn(h*w, dim), torch.randn(h*w, dim)
    K = torch.tensor([[5., 0., 2.], [0., 5., 1.5], [0., 0., 1.]])
    points = model.camera_observation(tokens, shallow, K, h, w)[0].detach()
    visual = torch.eye(latent)
    covars = torch.diag_embed(torch.tensor([1/256, 4/256, 7/256]).repeat(h*w, 1)).requires_grad_()
    memory = GaussianMemory(points, covars, torch.ones(h*w), visual)
    captured = []
    hook = model.key_residual.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        model.key_residual.weight.normal_(std=.02)
        model.dustbin_logit.fill_(-3.)
    with no_eigen():
        T0, bootstrap = model(tokens, visual, memory, K, h, w, shallow=shallow)
        alternate = GaussianMemory(points, torch.diag_embed(torch.tensor([4/256]*3).repeat(h*w, 1)), memory.alpha, visual)
        _, other = model(tokens, visual, alternate, K, h, w, shallow=shallow)
        T1, _, _, refined = deep(tokens, memory, bootstrap, T0)
    hook.remove()
    # Identical trace, different axes/eigenvalues: appearance keys stay identical.
    torch.testing.assert_close(captured[0], captured[1], atol=0, rtol=0)
    torch.testing.assert_close(captured[0][:, -4:-1], torch.full((h*w, 3), .5*torch.tensor(4/256).log()))
    torch.testing.assert_close(captured[0][:, :latent], visual, atol=0, rtol=0)
    torch.testing.assert_close(bootstrap['matching_keys'], other['matching_keys'], atol=0, rtol=0)
    assert_spherical(bootstrap['posterior_covars'])
    assert_spherical(refined['refined_posterior_covars'])
    assert torch.isfinite(T0).all() and torch.isfinite(T1).all()
    assert (torch.diff(refined['refinement_lm_energy_history']) <= 1e-8).all()
    loss = T1[:3, 3].square().sum()+bootstrap['posterior_covars'].sum()+refined['refined_posterior_covars'].sum()+refined['refined_matching_attention'].square().mean()
    loss.backward()
    for value in (covars, model.key_residual.weight, model.camera_geometry[-1].weight, deep.query_delta[-1].weight):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    # Covariance reaches neural keys only through trace: all diagonal derivatives
    # agree, and no off-diagonal derivative can carry an orientation.
    torch.testing.assert_close(covars.grad, covars.grad.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]*torch.eye(3), atol=2e-6, rtol=2e-5)


def test_explicit_false_preserves_full_initialization_and_outputs():
    tokens = torch.randn(4, 8)
    depth = torch.ones(2, 2)
    K = torch.tensor([[4., 0., 1.], [0., 4., 1.], [0., 0., 1.]])
    torch.manual_seed(23)
    default = TokenGaussianWriter(8, 6)
    torch.manual_seed(23)
    explicit = TokenGaussianWriter(8, 6, isotropic=False)
    a, b = default(tokens, depth, K, torch.eye(4)), explicit(tokens, depth, K, torch.eye(4))
    for field in ('means', 'covars', 'alpha', 'features'):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), atol=0, rtol=0)
    assert (a.covars[:, 0, 0] > a.covars[:, 2, 2]).all()
    x, y = SpatialMomentFusion()(a), SpatialMomentFusion(isotropic=False)(a)
    torch.testing.assert_close(x.covars, y.covars, atol=0, rtol=0)
    weights = F.softmax(torch.randn(2, 4), -1)
    _, x = posterior_ellipsoids(weights, a)
    _, y = posterior_ellipsoids(weights, a, isotropic=False)
    torch.testing.assert_close(x, y, atol=0, rtol=0)
