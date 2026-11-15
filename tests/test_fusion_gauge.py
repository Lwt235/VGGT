import torch
from src.geometry import quat_wxyz_to_matrix, transform_points
from src.memory.gaussians import GaussianMemory, SpatialMomentFusion


def test_fusion_grouping_and_moments_transport_with_mapping_reference():
    torch.manual_seed(17)
    means = torch.tensor([[.021, .03, 2.025], [.028, .035, 2.03],
                          [.181, .03, 2.025], [.188, .035, 2.03],
                          [.341, .03, 2.025], [.348, .035, 2.03]], dtype=torch.float64)
    means.requires_grad_()
    covars = torch.diag(torch.tensor([.0001, .0025, .0036], dtype=means.dtype)).repeat(6, 1, 1).requires_grad_()
    alpha = torch.full((6,), .4, dtype=means.dtype, requires_grad=True)
    features = torch.randn(6, 8, dtype=means.dtype, requires_grad=True)
    memory = GaussianMemory(means, covars, alpha, features)
    fusion = SpatialMomentFusion().double()
    reference = torch.eye(4, dtype=means.dtype)
    base = fusion(memory, reference_c2w=reference)
    assert len(base) == 3  # Real fusion, not a test with all singleton groups.
    G = torch.eye(4, dtype=means.dtype)
    G[:3, :3] = quat_wxyz_to_matrix(torch.randn(4, dtype=means.dtype))
    G[:3, 3] = torch.tensor([1.3, -2.2, .7], dtype=means.dtype)
    moved = GaussianMemory(transform_points(means, G), G[:3, :3] @ covars @ G[:3, :3].T, alpha, features)
    rebuilt = fusion(moved, reference_c2w=G @ reference)
    assert len(rebuilt) == len(base)
    torch.testing.assert_close(rebuilt.means, transform_points(base.means, G), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(rebuilt.covars, G[:3, :3] @ base.covars @ G[:3, :3].T, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(rebuilt.features, base.features)
    torch.testing.assert_close(rebuilt.alpha, base.alpha)
    (rebuilt.means.square().sum()+rebuilt.covars.sum()+rebuilt.features.sum()+rebuilt.alpha.sum()).backward()
    for value in (means, covars, features, alpha):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
