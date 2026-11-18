from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.diagnose_step3_v3 import (
    candidate_outcome, depth_regions, depth_statistics, extrapolate_pose, final_pose_diagnostics,
    motion_prior, move_candidates, seed_overlap_grid, seed_subset_weights, sync_seed, write_group_summary,
)
from src.data import Supervision
from src.geometry import unproject
from src.memory.gaussians import GaussianMemory, TokenGaussianWriter
from src.models.stream_pose import StreamPoseGate
from src.models.stream_tracking import tracking_logit_prior
from src.streaming_v3 import StreamingStateV3


def test_depth_regions_and_signed_bias_keep_empty_regions_explicit():
    opacity = torch.tensor([[[.2], [.05], [.01]]])
    depth = torch.full((2, 6), 4.)
    regions = depth_regions(opacity, depth.shape)
    assert torch.stack(list(regions.values())).sum(dim=0).eq(1).all()
    assert [int(v.sum()) for v in regions.values()] == [4, 4, 4]
    xyz = torch.zeros(2, 6, 3)
    xyz[..., 2] = 2.
    anchored = xyz.clone()
    anchored[:, :2, 2] = 4.
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[:, 2:4] = False
    out = dict(rendered_T1=SimpleNamespace(opacity=opacity), patch_hw=(1, 3), z_cam=torch.full((3,), 3.),
               raw_camera_points=xyz, camera_points=anchored)
    stats = {(r['region'], r['head']):r for r in depth_statistics(out, SimpleNamespace(depth=depth, valid=valid))}
    assert stats['frontier','raw_dpt']['median_ratio'] == .5
    assert stats['covered','o5_anchored']['bias_m'] == 0
    assert stats['frontier','z_cam']['l1_m'] == 1.
    assert stats['transition','raw_dpt']['bias_m'] is None


def test_seed_overlap_requires_same_observed_surface():
    K, pose = torch.eye(3), torch.eye(4)
    depth = torch.ones(2, 4)
    valid = torch.ones_like(depth, dtype=torch.bool)
    seed = Supervision(depth, pose, unproject(depth, K, pose), valid)
    back = depth.clone()
    back[:, 2:] = 2.
    query = Supervision(back, pose, unproject(back, K, pose), valid)
    overlap, supported = seed_overlap_grid(K, seed, query, (1, 2))
    assert torch.equal(overlap, torch.tensor([1., 0.]))
    assert supported.eq(1).all()


def test_freezing_seed_appearance_keeps_updated_nonseed_and_support():
    n = 3
    m = GaussianMemory(torch.arange(3*n).reshape(n, 3).float(), torch.eye(3).repeat(n, 1, 1),
                       torch.tensor([.2, .5, .7]), torch.ones(n, 2))
    lock = torch.tensor([True, False, True])
    seed = m.subset(lock)
    changed = GaussianMemory(m.means, m.covars, m.alpha+.1, m.features*2)
    state = StreamingStateV3(changed, torch.tensor([32., 2., 32.]), 3, seed_lock=lock, prev_pose=torch.eye(4))
    frozen = sync_seed(state, seed)
    assert torch.equal(frozen.memory.alpha[lock], seed.alpha)
    assert torch.equal(frozen.memory.features[lock], seed.features)
    assert torch.equal(frozen.memory.features[~lock], changed.features[~lock])
    assert torch.equal(frozen.support, state.support)
    assert torch.equal(state.memory.alpha, m.alpha+.1)
    with pytest.raises(ValueError, match='identity'):
        sync_seed(replace(state, memory=replace(changed, means=changed.means+1)), seed)


def test_seed_fit_weights_ignore_ground_truth_and_cap_background_count():
    p = torch.linspace(.02, .99, 192)
    match = dict(dustbin_probability=1-p, c_depth=torch.full_like(p, .8))
    weights = seed_subset_weights(match)
    assert (weights > 0).sum() == 48
    assert torch.allclose(weights[-48:], p[-48:]*.8)
    match['dustbin_probability'] = torch.full_like(p, .98)
    assert (seed_subset_weights(match) > 0).sum() == 16


def test_offline_gaussian_transform_matches_writer_without_passing_oracle_inputs_to_replay():
    torch.manual_seed(4)
    writer = TokenGaussianWriter(5, latent_dim=2)
    tokens = torch.randn(6, 5)
    depth = torch.tensor([[1., 2., 3.], [2., 4., 1.]])
    scale = torch.tensor([[.6, 1.2, .9], [1., 1.4, 1.1]])
    K = torch.tensor([[8., 0, 1.5], [0., 7., 1.], [0., 0., 1.]])
    pred, oracle = torch.eye(4), torch.eye(4)
    pred[:3, 3] = torch.tensor([1., .2, 2.])
    oracle[:3,:3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    oracle[:3, 3] = torch.tensor([0., 1., -2.])
    moved = move_candidates(writer(tokens, depth, K, pred), pred, oracle, scale.flatten())
    expected = writer(tokens, scale*depth, K, oracle)
    for key in ('means', 'covars', 'alpha', 'features'):
        assert torch.allclose(getattr(moved, key), getattr(expected, key), atol=2e-6, rtol=1e-6)


def test_tracking_sweep_prior_and_prediction_only_velocity_equivariance():
    old, previous, gauge = torch.eye(4), torch.eye(4), torch.eye(4)
    previous[:3,3] = torch.tensor([.15, -.10, .30])
    gauge[:3,:3] = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    gauge[:3,3] = torch.tensor([2., -.1, 3.])
    future = extrapolate_pose(old, previous)
    assert torch.allclose(future[:3,3], 2*previous[:3,3])
    assert torch.allclose(extrapolate_pose(gauge @ old, gauge @ previous), gauge @ future, atol=1e-6)
    means = torch.tensor([[1., 0., 2.], [0., 0., 2.]])
    rays = torch.tensor([[0., 0., 1.]])
    assert torch.equal(motion_prior(means, rays, previous, .15), tracking_logit_prior(means, rays, previous))
    assert (motion_prior(means, rays, previous, .35) > motion_prior(means, rays, previous, .15)).all()


def test_candidate_failure_and_better_alternative_are_distinct():
    bad = dict(t0_m=2., R0_deg=40.)
    good = dict(t0_m=.2, R0_deg=8.)
    row = dict(pose_source='global', d3={'global':bad, 'track':bad, 'seed':good})
    result = candidate_outcome(row)
    assert result['rejected_adequate'] and not result['all_mismatch']
    assert result['materially_better_dominated'] == ['seed']
    row['d3']['seed'] = dict(t0_m=.7, R0_deg=16.)
    assert candidate_outcome(row)['all_mismatch']


def test_gate_diagnostic_separates_raw_motion_limiter_and_learned_value():
    pose = torch.eye(4)
    raw = pose.clone()
    raw[0, 3] = .12
    diagnostic = dict(dustbin_probability=torch.zeros(3),
                      matching_attention=torch.tensor([[1., 0.]]).repeat(3,1),
                      refinement_lm_energy_history=torch.tensor([1., .9, .8]))
    gated, gate = StreamPoseGate()(pose, raw, diagnostic)
    result = final_pose_diagnostics(dict(T0=pose, T1=gated, stream_pose_gate=gate, **diagnostic), raw,
                                    float(torch.sigmoid(torch.tensor(-2.))), SimpleNamespace(c2w=raw))
    assert result['limiter'] == pytest.approx(.5)
    assert result['reliability'] == pytest.approx(1.)
    assert result['T1_raw']['translation_m'] == 0
    assert 0 < result['T1_gated']['translation_m'] < result['T0']['translation_m']


def test_write_signal_proxy_is_positive_and_no_gt_in_formula():
    def make(trans, similarity, fraction, pose_source):
        signals = {k:1. for k in ('inliers', 'visible', 'raw_depth_relative_residual',
                   'anchored_depth_relative_residual', 'z_cam_relative_residual', 'motion_m', 'motion_deg')}
        signals.update(pose_source=pose_source, feature_cosine=similarity, inlier_fraction=fraction,
                       candidates=10, mean_confidence=.5)
        return dict(translation1_m=trans, write={'candidates':10}, write_signals=signals)
    summary = write_group_summary([make(.3, .8, .8, 'global'), make(2., .45, .1, 'global'),
                                   make(3., .5, .2, 'track')])
    proxy = summary['fixed_soft_global_reweighting_proxy']
    assert summary['in_sample_auc']['feature_cosine'] == 1.
    assert proxy['erroneous']['kept_positive'] == 2
    assert np.isclose(proxy['erroneous']['proxy_weight_retained'], .625)
    assert proxy['healthy']['proxy_weight_retained'] == 1.
