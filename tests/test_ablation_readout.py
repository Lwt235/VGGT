"""CPU checks of ablation contracts; rasterization is explicitly mocked here."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from src.geometry import world_rays
from src.memory.gaussians import GaussianMemory
from src.readout.parameter_pe import ParameterPEReader, active_memory, center_pe
from src.readout.render import RenderReader


def read_case():
    torch.manual_seed(17)
    K = torch.tensor([[3., 0., 1.], [0., 3., 1.], [0., 0., 1.]])
    _, rays, dz = world_rays(K, torch.eye(4), 2, 2)
    means = (rays/dz[:, None]*torch.tensor([[2.], [2.4], [2.8], [3.1]])).requires_grad_()
    covars = torch.diag(torch.tensor([.04, .02, .005])).repeat(4, 1, 1).requires_grad_()
    memory = GaussianMemory(means, covars, torch.full((4,), .7, requires_grad=True),
                            torch.randn(4, 4, requires_grad=True))
    return K, memory, torch.randn(4, 16, requires_grad=True)


@pytest.mark.parametrize('isotropic', [False, True])
@pytest.mark.parametrize('camera_frame', [False, True])
def test_pe_only_none_render_has_no_o5_geometry_and_finite_gradients(isotropic, camera_frame):
    K, memory, tokens = read_case()
    reader = ParameterPEReader(16, 4, heads=2, bands=3, isotropic=isotropic,
                               camera_frame=camera_frame)
    with torch.no_grad():
        reader.gate.fill_(.3)
    query_geometry, geometry = [], []
    query_hook = reader.query_geometry.register_forward_pre_hook(lambda _, args: query_geometry.append(args[0]))
    bias_hook = reader.bias.register_forward_pre_hook(lambda _, args: geometry.append(args[0]))
    output, info = reader(tokens, memory, torch.eye(4), K, 2, 2,
                          render_depth=None, rendered=None, return_attention=True, query_chunk=2)
    query_hook.remove()
    bias_hook.remove()
    assert query_geometry[0][:, :18].count_nonzero() == 0
    assert all(g[..., 2].count_nonzero() == 0 for g in geometry)
    assert info['attention'].shape == (2, 4, 4)
    assert not torch.equal(output, tokens)
    output.square().mean().backward()
    for value in (tokens, memory.means, memory.covars, memory.alpha, memory.features):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_isotropic_pe_never_uses_eigensystem_shape_or_normal_and_keeps_center_phase():
    K, memory, tokens = read_case()
    reader = ParameterPEReader(16, 4, heads=2, bands=8, isotropic=True)
    with torch.no_grad():
        reader.gate.fill_(.3)
    # Equal trace with a different anisotropic orientation must be indistinguishable.
    alternative = memory.covars.detach().clone()[:, [2, 0, 1]][:, :, [2, 0, 1]]
    alternative[:, 0, 1] = alternative[:, 1, 0] = .001
    other = GaussianMemory(memory.means, alternative, memory.alpha, memory.features)
    geometry, keys = [], []
    bias_hook = reader.bias.register_forward_pre_hook(lambda _, args: geometry.append(args[0]))
    key_hook = reader.key.register_forward_pre_hook(lambda _, args: keys.append(args[0]))
    with patch('torch.linalg.eigh', side_effect=AssertionError('point sphere cannot use eigh')), \
         patch('torch.linalg.eigvalsh', side_effect=AssertionError('point sphere cannot use eigvalsh')):
        selected, indices = active_memory(memory, torch.eye(4), K, 2, 2, isotropic=True)
        a, da = reader(tokens, memory, torch.eye(4), K, 2, 2, return_attention=True)
        b, db = reader(tokens, other, torch.eye(4), K, 2, 2, return_attention=True)
    bias_hook.remove()
    key_hook.remove()
    assert len(selected) == 4 and len(indices) == 4
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(da['attention'], db['attention'])
    for g in geometry:
        assert g[..., 3].count_nonzero() == 0
        torch.testing.assert_close(g[..., 4], g[..., 5], atol=0, rtol=0)
        torch.testing.assert_close(g[..., 5], g[..., 6], atol=0, rtol=0)
        expected = memory.covars.diagonal(dim1=-2, dim2=-1).mean(-1).sqrt().log()
        torch.testing.assert_close(g[0, :, 4], expected)
    torch.testing.assert_close(keys[0][:, 4:], center_pe(memory.means, bands=8), atol=0, rtol=0)
    torch.testing.assert_close(keys[0][:, 4+7], torch.sin(memory.means[:, 0]*torch.pi*128))


def test_isotropic_render_sends_only_scalar_covariance_and_zero_normal_to_rasterizer():
    K, memory, tokens = read_case()
    reader = RenderReader(4, 16, surface_normals=True, camera_frame=True, isotropic=True)
    with torch.no_grad():
        reader.gate.fill_(.3)
    captured = {}

    def mock_rasterization(**kwargs):
        # Differentiable interface stand-in, not a physical or GPU raster test.
        captured.update(kwargs)
        weights = kwargs['opacities']/(1+kwargs['covars'].diagonal(dim1=-2, dim2=-1).sum(-1))
        opacity = weights.sum()/(1+weights.sum())
        attributes = (weights[:, None]*kwargs['colors']).sum(0)/weights.sum()
        accumulated = (attributes*opacity).expand(1, kwargs['height'], kwargs['width'], -1)
        return accumulated, opacity.expand(1, kwargs['height'], kwargs['width'], 1), {}

    with patch.dict('sys.modules', {'gsplat': SimpleNamespace(rasterization=mock_rasterization)}), \
         patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
         patch('torch.linalg.eigh', side_effect=AssertionError('sphere normals forbidden')), \
         patch('torch.linalg.eigvalsh', side_effect=AssertionError('sphere eigensystem forbidden')), \
         patch('torch.linalg.solve', side_effect=AssertionError('sphere inverse iteration forbidden')):
        output, rendered = reader(tokens, memory, torch.eye(4), K, 2, 2)
    expected = memory.covars.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]*torch.eye(3)
    torch.testing.assert_close(captured['covars'], expected)
    assert captured['colors'][:, -3:].count_nonzero() == 0
    assert rendered.normal.shape == (2, 2, 3) and rendered.normal.count_nonzero() == 0
    assert captured['quats'] is None and captured['scales'] is None
    (output.square().mean()+rendered.depth.mean()).backward()
    for value in (tokens, memory.means, memory.covars, memory.alpha, memory.features):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize('isotropic', [False, True])
def test_empty_memory_pe_is_identity_with_open_gate_and_render_retains_zero_gate_identity(isotropic):
    K, memory, tokens = read_case()
    empty = memory.subset(torch.zeros(len(memory), dtype=torch.bool))
    pe = ParameterPEReader(16, 4, heads=2, isotropic=isotropic)
    render = RenderReader(4, 16, surface_normals=True, camera_frame=True, isotropic=isotropic)
    with torch.no_grad():
        pe.gate.fill_(.7)
        render.unseen.fill_(.8)
        render.project[-1].bias.fill_(.4)
    a, info = pe(tokens, empty, torch.eye(4), K, 2, 2)
    b, rendered = render(tokens, empty, torch.eye(4), K, 2, 2)
    assert a is tokens
    torch.testing.assert_close(b, tokens, atol=0, rtol=0)
    assert info['indices'].numel() == 0 and info['attention'] is None
    for value in (rendered.world, rendered.features, rendered.depth,
                  rendered.depth_variance, rendered.opacity, rendered.normal):
        assert value.count_nonzero() == 0
    (a+b).square().mean().backward()
    assert torch.isfinite(tokens.grad).all()
