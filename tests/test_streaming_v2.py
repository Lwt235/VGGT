from dataclasses import fields
import inspect
from pathlib import Path
import random
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from src.geometry import transform_points
from src.memory.gaussians import GaussianMemory
from src.memory.streaming_fusion import StreamingMomentFusion
from src.models.ellipsoid_pose import local_se3_exp
from src.models.stream_landmarks import ray_consensus, select_landmark_pose
from src.readout.render import RenderReader, RenderedMemory
from src.readout.streaming_depth import propagate_scale, anchor_streaming_geometry
from src.streaming import SpatialWrite
from src.streaming_v2 import StreamingEngineV2, StreamingStateV2, locked_fusion, detach_state, ConsistentWriteConfidence
from src.streaming_data_v2 import StridedStreamingDataset


def memory(means, latent=4):
    return GaussianMemory(means, torch.eye(3).expand(len(means), 3, 3).clone()*.02,
                          torch.ones(len(means))*.7, torch.ones(len(means), latent))


def test_seed_geometry_cannot_be_overwritten_by_repeated_noisy_observations():
    original = memory(torch.tensor([[0., 0., 3.], [1., 0., 3.]]))
    current, support, lock = original, torch.ones(2)*32, torch.ones(2, dtype=torch.bool)
    fusion = StreamingMomentFusion()
    incoming = memory(original.means+torch.tensor([.05, .01, -.04]))
    incoming.features = incoming.features*2
    for _ in range(40):
        current, support, lock = locked_fusion(fusion, current, support, incoming, torch.ones(2), lock)
    torch.testing.assert_close(current.means, original.means, atol=0, rtol=0)
    torch.testing.assert_close(current.covars, original.covars, atol=0, rtol=0)
    assert (current.features > original.features).all() and support.max() <= 32


def test_frontier_scale_transport_continuous_and_finite_radius_identity():
    ratio = torch.ones(4, 14, 1)*2
    reliability = torch.zeros_like(ratio); reliability[:, :4] = 1
    scale, _ = propagate_scale(ratio, reliability)
    assert scale[:, 3:6].min() > 1.9
    torch.testing.assert_close(scale[:, 10:], torch.ones_like(scale[:, 10:]), atol=0, rtol=0)
    empty, _ = propagate_scale(ratio, reliability*0)
    torch.testing.assert_close(empty, torch.ones_like(empty), atol=0, rtol=0)


def test_stream_depth_anchor_full_metric_rays_empty_identity_and_gradients():
    reader = RenderReader(4, 8, camera_frame=True, physical_local=True)
    K = torch.tensor([[100., 0., 55.], [0., 100., 27.], [0., 0., 1.]])
    from src.geometry import pixel_grid
    rays = pixel_grid(56, 112, center=0.) @ torch.linalg.inv(K).T
    raw = (rays*3).requires_grad_()
    z = torch.ones(4, 8, 1)*6
    opacity = torch.zeros_like(z); opacity[:, :4] = .65
    rendered = RenderedMemory(torch.zeros(4, 8, 3), torch.ones(4, 8, 4), z, z*0, opacity)
    corrected, confidence, info = anchor_streaming_geometry(reader, raw, rendered, K)
    assert corrected[..., 2][:, :65].mean() > 5.8
    corrected.square().mean().backward()
    assert torch.isfinite(raw.grad).all() and torch.isfinite(reader.depth_confidence[-1].bias.grad).all()
    rendered.opacity.zero_()
    unchanged, _, _ = anchor_streaming_geometry(reader, raw, rendered, K)
    torch.testing.assert_close(unchanged, raw, atol=0, rtol=0)


def gate_input(bad_geometry=False, bad_features=False):
    rays = F.normalize(torch.tensor([[x, y, 2.] for y in (-.4, .4) for x in (-.5, 0., .5)]), dim=-1)
    anchors = rays*3
    if bad_geometry:
        anchors = anchors+torch.tensor([2., 0., 0.])
    observed = RenderedMemory(anchors.reshape(2, 3, 3), torch.ones(2, 3, 4)*(-1 if bad_features else 1),
                              torch.ones(2, 3, 1)*3, torch.zeros(2, 3, 1), torch.ones(2, 3, 1)*.7)
    return dict(dustbin_probability=torch.ones(6)*.05, c_depth=torch.ones(6), anchors=anchors,
        posterior_covars=torch.eye(3).expand(6, 3, 3)*.01, camera_rays=rays,
        T1=torch.eye(4), rendered_T1=observed, raw_camera_points=torch.ones(28, 42, 3)*3)


def test_double_write_gate_rejects_confident_geometric_or_visual_mismatch():
    gate = ConsistentWriteConfidence()
    depth = torch.ones(2, 3)*3
    good, _, checks = gate(depth, gate_input(), torch.ones(6, 4))
    assert good.mean() > .1 and checks['safety_accepted']
    for bad in (gate_input(bad_geometry=True), gate_input(bad_features=True)):
        weight, _, checks = gate(depth, bad, torch.ones(6, 4))
        assert weight.max() == 0 and not checks['safety_accepted']
    good.mean().backward()
    assert gate.modulation[-1].bias.grad.abs().sum() > 0


def test_consensus_and_high_support_candidate_are_gauge_equivariant():
    data = gate_input()
    rays, anchors, covars = data['camera_rays'], data['anchors'], data['posterior_covars']
    spatial = memory(anchors.repeat(64, 1))
    support = torch.ones(384); support[:64] = 32
    good = torch.eye(4); bad = good.clone(); bad[0, 3] = 1.
    qfeat = torch.ones(6, 4)
    boot = dict(anchors=anchors, posterior_covars=covars, camera_rays=rays,
                dustbin_probability=torch.zeros(6), z_cam=torch.ones(6)*3)
    def run(G):
        mapped = memory(transform_points(spatial.means, G))
        covariance = G[:3, :3] @ covars @ G[:3, :3].T
        transformed = dict(boot, anchors=transform_points(anchors, G), posterior_covars=covariance)
        model = SimpleNamespace(initial_pose=lambda *a, **k: (G@good, dict(transformed, matching_mask=k['match_mask'])),
            render_reader=SimpleNamespace(render=lambda *a: data['rendered_T1']))
        chosen, diag = select_landmark_pose(model, torch.ones(6, 8), torch.ones(6, 8), qfeat,
            mapped, support, torch.eye(3), 2, 3, G@bad, dict(transformed))
        assert diag['landmark_selected'] and diag['matching_mask'].sum() == 64
        return chosen, diag['landmark_candidate_scores']
    base, score = run(torch.eye(4))
    G = torch.eye(4); G[:3, :3], _ = local_se3_exp(torch.tensor([.3, -.8, .4]), torch.zeros(3)); G[:3, 3] = torch.tensor([3., -2., 1.])
    moved, moved_score = run(G)
    torch.testing.assert_close(moved, G@base, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(score, moved_score, atol=2e-6, rtol=2e-6)


def test_long_sampling_is_strided_train_only_and_reaches_beyond_seed():
    root = Path(__file__).resolve().parents[1]
    dataset = StridedStreamingDataset(root/'configs/step2_splits.yaml', (168, 224))
    rng = random.Random(17)
    modes, max_stride, low_seed = set(), 0, 0
    for _ in range(250):
        clip = dataset.sample(rng)
        item = dataset.sequences[clip['scene']]
        assert item['partition'] == 'train' and 12 <= len(clip['frame_ids']) <= 16
        assert set(clip['frame_ids']) <= set(item['train_frame_ids'])
        assert not set(clip['frame_ids']) & set(item['heldout_frame_ids'])
        assert max(clip['strides']) <= 4 and min(clip['strides']) >= 1
        for a, b in zip(clip['source_indices'], clip['source_indices'][1:]):
            assert max(item['surface_overlap'][a][b], item['surface_overlap'][b][a]) >= .10
        modes.add(clip['mode']); max_stride = max(max_stride, max(clip['strides']))
        low_seed += min(clip['seed_surface_overlap'][1:]) < .05
    assert max_stride == 4 and low_seed > 0 and len(modes) == 3
    assert not dataset.scenes


def test_tbptt_detaches_all_spatial_history_but_preserves_finite_future_gradients():
    parameter = torch.nn.Parameter(torch.tensor(.04))
    state = StreamingStateV2(memory(torch.tensor([[0., 0., 3.]])), torch.ones(1), 0,
        seed_lock=torch.ones(1, dtype=torch.bool), last_write_pose=torch.eye(4))
    for step in range(1, 16):
        m = GaussianMemory(state.memory.means+parameter, state.memory.covars, state.memory.alpha, state.memory.features+parameter)
        state = StreamingStateV2(m, state.support+parameter, step, (SpatialWrite(m, state.support+parameter, step),), state.seed_lock, state.last_write_pose)
        if step % 5 == 0:
            (m.means.square().mean()+state.support.mean()).backward()
            state = detach_state(state)
            assert state.memory.means.grad_fn is None and state.window_entries[0].support.grad_fn is None
    assert torch.isfinite(parameter.grad) and parameter.grad > 0
    assert [f.name for f in fields(state)] == ['memory', 'support', 'step', 'window_entries', 'seed_lock', 'last_write_pose']
    assert list(inspect.signature(StreamingEngineV2.forward).parameters) == ['self', 'query', 'state', 'return_attention']


def test_v2_engine_keyframe_gate_and_window_expire_seed_without_history():
    from src.memory.gaussians import TokenGaussianWriter, SpatialMomentFusion
    from src.data import QueryInput
    import pytest
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.writer = TokenGaussianWriter(8, 4)
            self.fusion = SpatialMomentFusion()
            self.offset = torch.nn.Parameter(torch.tensor(.1))
            with torch.no_grad():
                self.writer.latent.weight.zero_(); self.writer.latent.bias.fill_(1.)
        def forward(self, query, spatial, **kwargs):
            assert kwargs['streaming'] and len(kwargs['streaming_support']) == len(spatial)
            out = gate_input()
            T = torch.eye(4); T[0, 3] = self.offset
            out.update(T1=T, anchors=out['anchors']+T[:3, 3], camera_points=out['raw_camera_points'],
                patch_hw=(2, 3), stream_writer_tokens=torch.ones(6, 8), attention={'indices': torch.arange(len(spatial))})
            return out
    engine = StreamingEngineV2(Model(), window=2)
    spatial = memory(gate_input()['anchors'])
    state = StreamingStateV2(spatial, torch.ones(6)*32, 0,
        (SpatialWrite(spatial, torch.ones(6), 0),), torch.ones(6, dtype=torch.bool), torch.eye(4))
    query = QueryInput(torch.zeros(3, 28, 42), torch.tensor([[40., 0., 20.], [0., 40., 14.], [0., 0., 1.]]))
    first, state = engine(query, state)
    assert first['write']['candidates'] > 0
    redundant, state = engine(query, state)
    assert redundant['write']['candidates'] == 0 and not state.seed_lock.any()
    assert [e.step for e in state.window_entries] == [1, 2]
    _, state = engine(query, state)
    assert len(state.memory) == 0
    assert 'stream_writer_tokens' not in first
    with pytest.raises(TypeError):
        engine(query, state, depth=torch.ones(28, 42))


def test_selected_landmark_supervision_does_not_force_match_to_excluded_noise():
    from src.losses.world import open_set_matching_loss
    from src.data import Supervision
    spatial = memory(torch.tensor([[10., 0., 3.], [0., 0., 3.]]))
    points = torch.tensor([0., 0., 3.]).expand(14, 14, 3)
    target = Supervision(torch.ones(14, 14)*3, torch.eye(4), points, torch.ones(14, 14, dtype=torch.bool))
    W = torch.tensor([[.1, 0., .9]], requires_grad=True)
    out = dict(patch_hw=(1, 1), anchors=torch.tensor([[10., 0., 3.]]),
        anchor_support_means=spatial.means, anchor_support_covars=spatial.covars,
        matching_mask=torch.tensor([True, False]), matching_attention=W)
    loss, _, support, observable = open_set_matching_loss(out, target)
    assert support == 0 and not observable
    torch.testing.assert_close(loss, -W[0, -1].log())
    loss.backward()
    assert W.grad[0, -1] < 0 and W.grad[0, 1] == 0
