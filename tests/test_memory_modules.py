"""Necessary moment, gradient, visibility and intervention checks.

These are component tests, not a substitute for real VGGT overfit training.
"""

import torch

from src.geometry import patch_intrinsics
from src.memory.gaussians import GaussianMemory, SpatialMomentFusion, TokenGaussianWriter
from src.readout.parameter_pe import ParameterPEReader, active_memory, center_pe
from src.readout.render import normalize_rendered_moments


def test_closed_form_total_covariance_includes_center_spread_and_is_spd():
    means = torch.tensor([[.02, .02, 2.02], [.04, .02, 2.02]], requires_grad=True)
    covars = torch.diag(torch.tensor([.01, .02, .001])).repeat(2, 1, 1).requires_grad_()
    memory = GaussianMemory(means, covars, torch.tensor([.5, .5], requires_grad=True),
                            torch.tensor([[1., 2.], [3., 4.]], requires_grad=True))
    fused = SpatialMomentFusion(min_cell=1.)(memory)
    assert len(fused) == 1
    torch.testing.assert_close(fused.means, torch.tensor([[.03, .02, 2.02]]))
    torch.testing.assert_close(fused.covars[0, 0, 0], torch.tensor(.0101), atol=2e-8, rtol=1e-5)
    assert torch.linalg.eigvalsh(fused.covars).min() > 0
    (fused.means.square().sum()+fused.covars.sum()+fused.features.sum()+fused.alpha.sum()).backward()
    for value in (means, covars, memory.features, memory.alpha):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_render_moment_normalization_transparency_and_empty_coverage():
    # Two front-to-back splats, colors are geometry/latent attributes, not RGB.
    first = torch.tensor([0., 0., 2., 7., 2., 4.])
    back = torch.tensor([1., 0., 4., 3., 4., 16.])
    accumulated = (.5*first+.5*.4*back).reshape(1, 1, -1).requires_grad_()
    opacity = torch.tensor([[[.7]]], requires_grad=True)
    rendered = normalize_rendered_moments(accumulated, opacity, latent_dim=1)
    torch.testing.assert_close(rendered.depth.squeeze(), torch.tensor(1.8/.7))
    assert rendered.depth_variance.item() > 0
    (rendered.world.sum()+rendered.features.sum()+rendered.depth_variance.sum()).backward()
    assert torch.isfinite(accumulated.grad).all() and torch.isfinite(opacity.grad).all()
    empty = normalize_rendered_moments(torch.zeros_like(accumulated), torch.zeros_like(opacity), 1)
    assert empty.world.count_nonzero() == 0


def test_center_pe_independent_of_covariance_and_has_high_frequency():
    p = torch.tensor([[.13, .27, .8]], requires_grad=True)
    encoding = center_pe(p, bands=8)
    assert encoding.shape == (1, 48)
    # The highest band's sine is exactly its unattenuated phase.
    torch.testing.assert_close(encoding[0, 7], torch.sin(p[0, 0]*torch.pi*128))
    encoding.sum().backward()
    assert p.grad.abs().sum() > 0


def test_zero_gate_opens_then_writer_and_o6_receive_nonzero_gradients():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    writer = TokenGaussianWriter(16, latent_dim=8)
    reader = ParameterPEReader(16, latent_dim=8, heads=2, bands=4)
    optimizer = torch.optim.Adam(list(writer.parameters())+list(reader.parameters()), lr=.02)
    K = patch_intrinsics(torch.tensor([[70., 0., 27.5], [0., 70., 20.5], [0., 0., 1.]]))
    frozen_tokens = torch.randn(12, 16)
    query_tokens = torch.randn(12, 16)
    target = torch.randn(12, 16)
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        memory = writer(frozen_tokens, torch.full((3, 4), 2.), K, torch.eye(4))
        memory.retain_grad()
        output, info = reader(query_tokens, memory, torch.eye(4), K, 3, 4,
                              torch.full((3, 4, 1), 2.), return_attention=True)
        loss = (output-target).square().mean()
        loss.backward()
        if step == 0:
            torch.testing.assert_close(output, query_tokens)
            assert reader.gate.grad.abs() > 0
        else:
            for value in (memory.means, memory.covars, memory.alpha, memory.features):
                assert value.grad is not None and torch.isfinite(value.grad).all()
                assert value.grad.abs().sum() > 0
            assert writer.geometry.weight.grad.abs().sum() > 0
            assert writer.latent.weight.grad.abs().sum() > 0
            assert reader.bias[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert reader.gate.detach().abs() > 0
    assert info["attention"].shape == (2, 12, len(info["indices"]))


def test_active_read_budget_and_box_erasure_do_not_mutate_public_memory():
    writer = TokenGaussianWriter(8, latent_dim=4)
    K = torch.tensor([[4., 0., 2.], [0., 4., 1.5], [0., 0., 1.]])
    memory = writer(torch.randn(12, 8), torch.full((3, 4), 2.), K, torch.eye(4))
    original = memory.means.detach().clone()
    active, indices = active_memory(memory, torch.eye(4), K, 3, 4, budget=3)
    assert len(active) == 3 and len(indices) == 3 and len(memory) == 12
    erased, mask = memory.erase_box(torch.tensor([-.3, -3., 1.]), torch.tensor([.3, 3., 3.]))
    assert len(erased) == 12-mask.sum().item()
    torch.testing.assert_close(memory.means.detach(), original)
