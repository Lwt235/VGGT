"""Spatial identity and cumulative sufficient-statistic tests; no RGB/KV state."""

import torch

from src.memory.gaussians import GaussianMemory
from src.memory.streaming_fusion import StreamingMomentFusion


def memory(points, scale=.08, features=None):
    points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
    covariance = torch.diag(torch.tensor([scale**2, scale**2, .015**2])).expand(len(points), 3, 3).clone()
    if features is None:
        features = torch.zeros(len(points), 4)
    return GaussianMemory(points, covariance, torch.full((len(points),), .5), features)


def empty():
    return memory(torch.empty(0, 3))


def test_loiter_and_backtrack_saturate_across_scale_band_boundary():
    fusion = StreamingMomentFusion()
    base = torch.tensor([[0., 0., 3.], [.5, 0., 3.], [1., 0., 3.], [1.5, 0., 3.]])
    state, support = fusion(empty(), torch.empty(0), memory(base), torch.ones(4))
    for step in range(48):
        # 2*scale crosses the .02*2**3 physical hash band every step.
        scale = .0799 if step % 2 else .0801
        displacement = .003 if step % 2 else -.003
        candidate = memory((base+torch.tensor([displacement, 0., 0.])).flip(0), scale)
        state, support = fusion(state, support, candidate, torch.ones(4))
        assert len(state) == 4
    assert support.max() == fusion.support_cap
    assert torch.allclose(state.means, base, atol=.004)
    assert torch.linalg.eigvalsh(state.covars).min() > 0


def test_new_space_grows_and_untouched_old_surface_is_exactly_retained():
    fusion = StreamingMomentFusion()
    old = memory([[0., 0., 3.], [.5, 0., 3.]])
    new = memory([[10., 0., 3.], [10.5, 0., 3.]])
    result, support = fusion(old, torch.ones(2), new, torch.ones(2))
    assert len(result) == 4
    for name in ('means', 'covars', 'alpha', 'features'):
        assert torch.equal(getattr(result, name)[:2], getattr(old, name))
    assert torch.equal(support, torch.ones(4))


def test_accumulated_support_rejects_single_frame_surface_noise():
    fusion = StreamingMomentFusion()
    old = memory([[0., 0., 3.]])
    new = memory([[.05, 0., 3.]])
    young, _ = fusion(old, torch.ones(1), new, torch.ones(1))
    mature, support = fusion(old, torch.full((1,), 32.), new, torch.ones(1))
    assert mature.means[0, 0] < young.means[0, 0]/10
    assert support.item() == 32
    # Support caps still permit adaptation, rather than freezing the old map.
    assert mature.means[0, 0] > 0


def test_many_duplicate_patches_do_not_outvote_previous_frames():
    fusion = StreamingMomentFusion()
    old = memory([[0., 0., 3.]])
    new = memory(torch.tensor([[.05, 0., 3.]]).expand(50, 3))
    result, support = fusion(old, torch.full((1,), 10.), new, torch.ones(50))
    assert len(result) == 1
    assert torch.allclose(support, torch.tensor([11.]))
    assert torch.allclose(result.means[0, 0], torch.tensor(.05/11), atol=1e-6)


def test_different_surface_normals_do_not_merge():
    fusion = StreamingMomentFusion()
    old = memory([[0., 0., 3.]])
    new = memory([[0., 0., 3.]])
    new.covars = new.covars[:, [2, 1, 0]][:, :, [2, 1, 0]]
    result, _ = fusion(old, torch.ones(1), new, torch.ones(1))
    assert len(result) == 2


def test_geometry_features_support_and_update_gates_have_finite_gradients():
    fusion = StreamingMomentFusion()
    old = memory([[0., 0., 3.]], features=torch.zeros(1, 4))
    new = memory([[.03, 0., 3.]], features=torch.ones(1, 4))
    for state in (old, new):
        for value in (state.means, state.covars, state.alpha, state.features):
            value.requires_grad_()
    old_support = torch.tensor([2.], requires_grad=True)
    new_support = torch.tensor([.6], requires_grad=True)
    result, support = fusion(old, old_support, new, new_support)
    loss = sum(getattr(result, name).square().sum() for name in ('means', 'covars', 'alpha', 'features'))+support.sum()
    loss.backward()
    for state in (old, new):
        for value in (state.means, state.covars, state.alpha, state.features):
            assert value.grad is not None and torch.isfinite(value.grad).all()
    for value in (old_support, new_support, fusion.feature_gate, fusion.opacity_gate):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_no_public_memory_capacity_limit_or_image_history():
    fusion = StreamingMomentFusion()
    old = memory(torch.stack((torch.arange(2050.)*.5, torch.zeros(2050), torch.full((2050,), 3.)), -1))
    result, support = fusion(old, torch.ones(len(old)), memory([[2000., 0., 3.]]), torch.ones(1))
    assert len(result) == 2051
    assert len(support) == len(result)
    assert set(vars(result)) == {'means', 'covars', 'alpha', 'features'}
    assert not list(fusion.buffers())


def test_empty_or_zero_confidence_write_is_exact_identity():
    fusion = StreamingMomentFusion()
    old = memory([[0., 0., 3.]])
    support = torch.ones(1)
    result, result_support = fusion(old, support, empty(), torch.empty(0))
    assert result is old and result_support is support
    result, result_support = fusion(old, support, memory([[0., 0., 3.]]), torch.zeros(1))
    assert result is old and result_support is support
