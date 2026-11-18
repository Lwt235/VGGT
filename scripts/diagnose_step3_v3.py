#!/usr/bin/env python3
"""Frozen Step 3 v3 replay. GT substitutions are offline write attribution only."""
import argparse
from contextlib import contextmanager
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage

configure_local_storage()
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from torch.nn import functional as F
import yaml

from scripts.evaluate_step3 import prediction_metrics, seed_surface_overlap
from src.geometry import invert_se3, patch_intrinsics, project, transform_points, world_rays
from src.losses.world import pose_errors
from src.memory.gaussians import GaussianMemory
from src.models.ellipsoid_pose import anisotropic_ray_pose, planar_safe_initial
from src.models.spatial_pose import rotation_6d
from src.models.stream_tracking import local_motion, overlap_evidence, tracking_logit_prior
from src.vis.step2 import dump_json

RUN = PROJECT_ROOT / 'results/step3_v3_seed17'
OUT = PROJECT_ROOT / 'results/step3_v3_diagnostics'
CONFIG = PROJECT_ROOT / 'configs/step3_v3.yaml'
C11 = 'harvard_c11__hv_c11_2__natural'
ROBOTICS = 'harvard_robotics_lab__hv_s1_1__natural'
SEGMENTED = 'harvard_c11__hv_c11_2__segmented'
ORIGINAL, FROZEN, FILTERED = 'v3', 'frozen_seed', 'frozen_seed_top48'
FRONTIER, ORACLE_POSE, ORACLE_DEPTH = 'frontier_z_cam', 'oracle_write_pose', 'oracle_write_depth'
BRANCHES = (ORIGINAL, FRONTIER, ORACLE_POSE, ORACLE_DEPTH)
ORACLE_LABEL = 'Offline attribution (not model evaluation)'
FOCUS = {C11: (138, 430, 669, 781, 797), ROBOTICS: (68, 362)}


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=float(values.max())) if len(values) else dict(
                    n=0, mean=None, median=None, p90=None, max=None)


def grid_resize(values, shape, mode='nearest'):
    return F.interpolate(values[None, None], size=shape, mode=mode,
                         align_corners=False if mode == 'bilinear' else None)[0, 0]


def patch_depth(z, valid, shape):
    def pool(grid):
        return F.adaptive_avg_pool2d(grid[None, None], shape)[0, 0]
    fraction = pool(valid.float())
    return pool(torch.where(valid, z, 0.))/fraction.clamp_min(1e-6), fraction


def depth_regions(opacity, shape):
    opacity = grid_resize(opacity.squeeze(-1), shape)
    return {'covered': opacity >= .20, 'transition': (opacity >= .05) & (opacity < .20),
            'frontier': opacity < .05}


def depth_statistics(output, target):
    regions = depth_regions(output['rendered_T1'].opacity, target.depth.shape)
    z_cam = output['z_cam'].reshape(output['patch_hw'])
    depths = {'raw_dpt': output['raw_camera_points'][..., 2], 'o5_anchored': output['camera_points'][..., 2],
              'z_cam': grid_resize(z_cam, target.depth.shape, 'bilinear')}
    rows = []
    for region, region_mask in regions.items():
        for name, z in depths.items():
            mask = target.valid & region_mask & torch.isfinite(z)
            error = (z-target.depth)[mask]
            ratio = z[mask]/target.depth[mask]
            rows.append(dict(region=region, head=name, pixels=len(error),
                             bias_m=float(error.mean()) if len(error) else None,
                             l1_m=float(error.abs().mean()) if len(error) else None,
                             median_ratio=float(ratio.quantile(.5)) if len(ratio) else None))
    return rows


def seed_overlap_grid(seed_K, seed_target, target, shape):
    """Evaluator-only visibility at native replay resolution using benchmark z tolerance."""
    uv, z = project(target.world_points, seed_K, invert_se3(seed_target.c2w))
    x, y = uv.round().long().unbind(-1)
    h, w = seed_target.depth.shape
    fov = (z > .1) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = x.clamp(0, w-1), y.clamp(0, h-1)
    support = target.valid & fov & seed_target.valid[y, x] & (
        (z-seed_target.depth[y, x]).abs() < .10+.02*z)
    fraction, valid = patch_depth(support.float(), target.valid, shape)
    return fraction.flatten(), valid.flatten()


def sync_seed(state, seed):
    """Stable Full-fusion identity order; restores only opacity and descriptors."""
    lock = state.seed_lock
    if not torch.equal(state.memory.means[lock], seed.means) or not torch.equal(state.memory.covars[lock], seed.covars):
        raise ValueError('Seed identity/geometry changed during replay')
    alpha, features = state.memory.alpha.clone(), state.memory.features.clone()
    alpha[lock], features[lock] = seed.alpha, seed.features
    return replace(state, memory=GaussianMemory(state.memory.means, state.memory.covars, alpha, features))


def seed_drift(state, seed):
    current = state.memory.subset(state.seed_lock)
    feature = (1-F.cosine_similarity(current.features, seed.features)).clamp_min(0)
    alpha = (current.alpha-seed.alpha).abs()
    return {'feature': describe(feature.cpu()), 'alpha': describe(alpha.cpu()),
            'mean_displacement_max_m': float((current.means-seed.means).norm(dim=-1).max()),
            'covariance_delta_max': float((current.covars-seed.covars).abs().max())}


def seed_subset_weights(match, limit=48):
    p = 1-match['dustbin_probability']
    # Exclude dustbin-dominated patches first. A sparse fallback keeps enough
    # spatial samples for a solve, but does not lower the pose acceptance bar.
    eligible = int((p >= .50).sum())
    count = min(len(p), limit, max(eligible, 16))
    chosen = p.topk(count).indices
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[chosen] = True
    return p*match['c_depth']*keep


def filtered_seed_pose(pose, match, bootstrap):
    weights = seed_subset_weights(match)
    if float(weights.sum()) < .25:
        return pose, match
    updated, votes, lm = anisotropic_ray_pose(
        match['camera_patch_points'], match['camera_rays'], match['anchors'],
        match['posterior_covars'], weights, metric_scale=bootstrap.metric_scale,
        planar_safe=bootstrap.planar_safe, return_diagnostic=True)
    # Same observable-mass blend as SpatialBootstrap. W itself stays fixed
    # so overlap evidence and the subsequent refinement see the same logits.
    mass = weights.sum()
    quality = ((mass-.25)/.025).sigmoid()*mass/(mass+1e-8)
    eye = torch.eye(4, device=updated.device, dtype=updated.dtype)
    mixed = quality*updated[:3, :2]+(1-quality)*eye[:3, :2]
    rotation = rotation_6d(mixed.T.flatten())
    top = torch.cat((rotation, (quality*updated[:3, 3])[:, None]), dim=-1)
    return torch.cat((top, eye[3:])), dict(match, anchor_votes=votes, lm_energy_history=lm['energy_history'],
                                         lm_accepted_scales=lm['accepted_scales'], diagnosis_weights=weights)


@contextmanager
def observe_forward(engine, *, filtered=False):
    """Read-only taps. The optional solve override uses predictions only."""
    captured = {}

    def match_tap(module, args, kwargs, result):
        captured.setdefault('observation', (args[:2], kwargs['shallow']))
        if filtered and kwargs.get('match_mask') is not None:
            return filtered_seed_pose(*result, module)

    def writer_tap(module, args, kwargs, result):
        captured['write'] = (args[1], kwargs['valid_mask'], result)

    def pose_tap(module, args):
        captured['raw_pose'] = args[1]

    def confidence_tap(module, args, result):
        captured['learned_gate'] = float(result.sigmoid().squeeze())

    handles = [
        engine.model.initial_pose.register_forward_hook(match_tap, with_kwargs=True),
        engine.model.writer.register_forward_hook(writer_tap, with_kwargs=True),
        engine.model.stream_pose_gate.register_forward_pre_hook(pose_tap),
        engine.model.stream_pose_gate.confidence.register_forward_hook(confidence_tap),
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def motion_prior(means, rays, pose, sigma):
    if sigma == .15:
        return tracking_logit_prior(means, rays, pose)
    local = (means-pose[:3, 3]) @ pose[:3, :3]
    cosine = (F.normalize(rays, dim=-1) @ F.normalize(local, dim=-1).T).clamp(-1, 1)
    return -((1-cosine)/sigma**2+F.softplus(-local[:, 2]/.15)[None]).clamp_max(24.)


def extrapolate_pose(older, previous):
    return previous @ (invert_se3(older) @ previous)


def solve_candidate(model, state, query, observation, source, *, sigma=.15, init=None, filtered=False):
    (tokens, features), shallow = observation
    h, w = query.rgb.shape[-2:]
    hp, wp = h//14, w//14
    Kp = patch_intrinsics(query.K)
    memory = state.memory
    kwargs = {}
    render_memory = memory
    if source == 'seed':
        kwargs['match_mask'] = state.seed_lock
        render_memory = memory.subset(state.seed_lock)
    elif source == 'track':
        init = state.prev_pose if init is None else init
        _, rays, _ = world_rays(Kp, torch.eye(4, device=Kp.device), hp, wp)
        kwargs.update(init_pose=init, matching_prior=motion_prior(memory.means, rays, init, sigma))
    pose, match = model.initial_pose(tokens, features, memory, Kp, hp, wp, shallow=shallow, **kwargs)
    if filtered and source == 'seed':
        pose, match = filtered_seed_pose(pose, match, model.initial_pose)
    rendered = model.render_reader.render(render_memory, pose, Kp, hp, wp)
    return pose, match, overlap_evidence(pose, match, rendered, features)


def seed_weight_evidence(pose, match, overlap, valid, seed):
    p = 1-match['dustbin_probability']
    weights = match.get('diagnosis_weights', p*match['c_depth'])
    total = weights.sum().clamp_min(1e-8)
    info = {}
    groups = {'covisible': (overlap >= .5) & (valid > .5),
              'noncovisible': (overlap < .5) & (valid > .5), 'insufficient_depth': valid <= .5}
    seed_center = seed.means.mean(0)
    anchors = match['anchors']
    for name, keep in groups.items():
        mass = weights[keep].sum()
        center = (weights[keep, None]*anchors[keep]).sum(0)/mass.clamp_min(1e-8)
        info[name] = dict(patches=int(keep.sum()), positive_match_count=int((p[keep] >= .5).sum()),
                          match_probability=describe(p[keep].cpu()), weight_sum=float(mass),
                          weight_share=float(mass/total), anchor_center_xyz=center.cpu().tolist() if mass > 0 else None,
                          anchor_center_to_seed_center_m=float((center-seed_center).norm()) if mass > 0 else None)
    normalized = weights/total
    center = normalized @ anchors
    points = match['camera_patch_points']
    rotation, initial, scatter, _ = planar_safe_initial(points.double(), anchors.double(), weights.double())

    def facing(origin, forward):
        radius = seed_center-origin
        return float(F.cosine_similarity(radius, forward, dim=0))
    return dict(groups=info, soft_nonoverlap_weight_share=float((weights*(1-overlap)*(valid > .5)).sum()/total),
                weighted_anchor_center_xyz=center.cpu().tolist(), planar_initial_center_xyz=initial.cpu().tolist(),
                planar_initial_forward_xyz=rotation[:, 2].cpu().tolist(), planar_scatter_scale=float(scatter),
                planar_facing_seed_cosine=facing(initial, rotation[:, 2]),
                final_facing_seed_cosine=facing(pose[:3, 3], pose[:3, 2]),
                patches=[dict(index=i, overlap=float(o), valid_fraction=float(v), p_match=float(a),
                              weight=float(b), soft_anchor_xyz=c.tolist())
                         for i, (o, v, a, b, c) in enumerate(zip(overlap.cpu(), valid.cpu(), p.cpu(), weights.cpu(), anchors.cpu()))])


def candidate_row(pose, match, evidence, previous, target, camera_points=None):
    rotation, translation = pose_errors(pose, target.c2w)
    motion, angle = local_motion(pose, previous)
    row = dict(inliers=evidence.inliers, visible=evidence.visible,
               inlier_fraction=evidence.inlier_fraction, cosine=evidence.cosine,
               match_mass=float(match['match_mass']), lm_final_energy=float(match['lm_energy_history'][-1]),
               motion_m=motion, motion_deg=angle, R0_deg=float(rotation), t0_m=float(translation),
               pose=pose.cpu().tolist())
    if camera_points is not None:
        # Fixed geometry transported at the candidate pose for attribution.
        error = transform_points(camera_points[target.valid], pose)-target.world_points[target.valid]
        row['fixed_camera_points_EPE_m'] = float(error.norm(dim=-1).mean())
    return row


def final_pose_diagnostics(output, raw, learned_gate, target):
    base = output['T0']
    gated = output['T1']
    movement, angle = local_motion(raw, base)
    gated_move, gated_angle = local_motion(gated, base)
    gate = float(output['stream_pose_gate'])
    limit = min(1., .06/max(movement, 1e-6), 3./max(angle, 1e-6))
    rows = {}
    for name, pose in [('T0', base), ('T1_raw', raw), ('T1_gated', gated)]:
        rotation, translation = pose_errors(pose, target.c2w)
        rows[name] = dict(rotation_deg=float(rotation), translation_m=float(translation))
    energy = output['refinement_lm_energy_history']
    return dict(**rows, raw_motion_m=movement, raw_motion_deg=angle, gated_motion_m=gated_move,
                gated_motion_deg=gated_angle, gate=gate, learned_gate=learned_gate,
                limiter=limit, reliability=gate/max(learned_gate*limit, 1e-8),
                lm_initial_energy=float(energy[0]), lm_final_energy=float(energy[-1]))


def write_features(output, query_features, previous):
    rendered = output['rendered_T1']
    visible = (rendered.opacity[..., 0] >= .20) & (rendered.depth[..., 0] > 0)
    def residual(z):
        z, valid = patch_depth(z, torch.isfinite(z) & (z > 0), output['patch_hw'])
        mask = visible & (valid > .5)
        error = ((z-rendered.depth[..., 0]).abs()/rendered.depth[..., 0].clamp_min(.20))[mask]
        return float(error.median()) if len(error) else None
    cosine = F.cosine_similarity(rendered.features.reshape_as(query_features), query_features, dim=-1)
    p = 1-output['refined_dustbin_probability']
    from src.models.stream_landmarks import ray_consensus
    m2, _ = ray_consensus(output['T1'], output['refined_anchors'], output['refined_posterior_covars'],
                          output['camera_rays'], p)
    motion, angle = local_motion(output['T1'], previous)
    count = int((visible.flatten() & (p >= .5) & (m2 <= 4)).sum())
    return dict(pose_source=output['write']['pose_source'], inliers=count, visible=int(visible.sum()),
                inlier_fraction=count/max(int(visible.sum()), 1),
                feature_cosine=float(cosine[visible.flatten()].mean()) if visible.any() else None,
                raw_depth_relative_residual=residual(output['raw_camera_points'][..., 2]),
                anchored_depth_relative_residual=residual(output['camera_points'][..., 2]),
                z_cam_relative_residual=residual(grid_resize(output['z_cam'].reshape(output['patch_hw']),
                                                            output['camera_points'].shape[:2], 'bilinear')),
                motion_m=motion, motion_deg=angle, candidates=output['write']['candidates'],
                mean_confidence=output['write']['mean_confidence'])


def move_candidates(candidates, predicted_pose, write_pose, ratio):
    """Offline similarity about the predicted camera, then an optional rigid move."""
    local_shift = candidates.means-predicted_pose[:3, 3]
    rotation = write_pose[:3, :3] @ predicted_pose[:3, :3].T
    return GaussianMemory((ratio[:, None]*local_shift) @ rotation.T+write_pose[:3, 3],
                          rotation @ (ratio[:, None, None].square()*candidates.covars) @ rotation.T,
                          candidates.alpha, candidates.features)


def attribute_write(engine, mode, state, new, output, write, target):
    """Post-forward diagnostic write; even the Writer receives no oracle input."""
    depth, keep, candidates = write
    if not len(candidates):
        return new, dict(candidates=0, changed_candidates=0)
    pose = output['T1']
    if mode == ORACLE_POSE:
        alternative_pose = target.c2w
        ratio = torch.ones_like(depth[keep])
        valid = torch.ones_like(ratio, dtype=torch.bool)
        changed = len(ratio)
    else:
        alternative_pose = pose
        z = output['camera_points'][..., 2]
        if mode == ORACLE_DEPTH:
            adjusted = target.depth
            valid_pixels = target.valid
        else:
            frontier = depth_regions(output['rendered_T1'].opacity, z.shape)['frontier']
            # Replace only z on the calibrated ray. The Writer is linear in
            # patch z, including radius and the learned camera-frame offset.
            adjusted = torch.where(frontier, grid_resize(output['z_cam'].reshape_as(depth), z.shape, 'bilinear'), z)
            valid_pixels = torch.isfinite(adjusted) & (adjusted > .15) & (adjusted < 12.)
        new_depth, fraction = patch_depth(adjusted, valid_pixels, depth.shape)
        valid = fraction[keep] > .50
        ratio = new_depth[keep][valid]/depth[keep][valid]
        changed = int(((ratio-1).abs() > 1e-5).sum())
    modified = move_candidates(candidates.subset(valid), pose, alternative_pose, ratio)
    original_points = output['camera_points']
    original_z = original_points[..., 2]
    original_valid = torch.isfinite(original_points).all(-1) & (original_z > .15) & (original_z < 12.)
    _, original_fraction = patch_depth(original_z, original_valid, depth.shape)
    fresh_support = (output['write_confidence'][keep]*original_fraction[keep])[valid]*modified.alpha
    memory, support, lock, entries = engine.fuse_write(state, modified, fresh_support)
    return replace(new, memory=memory, support=support, seed_lock=lock, window_entries=entries), dict(
        candidates=len(modified), changed_candidates=changed)


def compare_drift(plans, output_dir):
    for plan in plans:
        data = json.loads((output_dir/plan['name']/'details.json').read_text())
        old = json.loads((RUN/'evaluation_final'/plan['name']/'persistent_metrics.json').read_text())['rows']
        new = data['replays'][ORIGINAL]['rows']
        if [(r['t'], r['frame_id']) for r in old] != [(r['t'], r['frame_id']) for r in new]:
            raise ValueError('The saved frame order changed')
        keys = ('point_epe_m', 'rotation0_deg', 'rotation1_deg', 'translation0_m', 'translation1_m')
        delta = {k: max(abs(a[k]-b[k]) for a, b in zip(old, new)) for k in keys}
        source = sum(a['write_checks']['pose_source'] != b['pose_source'] for a, b in zip(old, new))
        counts = sum(a['memory_after'] != b['memory_after'] for a, b in zip(old, new))
        changed = [a['frame_id'] for a,b in zip(old,new)
                   if any(abs(a[k]-b[k]) > 1e-6 for k in keys)]
        data['baseline_replay_check'] = dict(max_absolute_error_delta=delta, changed_source_steps=source,
                                             changed_memory_count_steps=counts, changed_error_frame_ids=changed)
        if plan['name'] == C11:
            # Signal values come from the instrumented replay in both cohorts.
            # This variant fixes frame membership to the previously archived run.
            archived = [dict(b, translation1_m=a['translation1_m'], write=a['write_checks'])
                        for a,b in zip(old,new)]
            data['archived_write_cohorts'] = write_group_summary(archived)
            for row, archive in zip(new, archived):
                if archive['write']['candidates'] and archive['translation1_m'] <= .50:
                    row['archived_write_group'] = 'healthy'
                elif archive['write']['candidates'] and archive['translation1_m'] > 1.:
                    row['archived_write_group'] = 'erroneous'
                else:
                    row['archived_write_group'] = None
        dump_json(output_dir/plan['name']/'details.json', data)


def reference_map_metrics(reference, centers):
    _, index = np.unique(np.floor(reference/.03).astype(np.int64), axis=0, return_index=True)
    reference = reference[np.sort(index)]
    return dict(gaussians=len(centers), reference_points=len(reference),
                map_center_to_surface_m=describe(cKDTree(reference).query(centers)[0]),
                surface_to_map_center_m=describe(cKDTree(centers).query(reference)[0]))


@torch.inference_mode()
def replay(engine, dataset, plan, mode):
    name = plan['name']
    original = mode == ORIGINAL
    d1 = name in (C11, SEGMENTED) and mode in (ORIGINAL, FROZEN, FILTERED)
    natural = name != SEGMENTED
    stream = iter(dataset.load_stream(plan['scene'], plan['frame_ids'], device='cuda')['observations'])
    q0, seed_target = next(stream)
    state = engine.seed(q0, seed_target.depth, seed_target.valid)
    seed = state.memory
    seed_K = q0.K
    del q0
    older_pose = state.prev_pose
    rows, reference = [], []
    if natural:
        reference.append(seed_target.world_points[seed_target.valid][::4].cpu().numpy())
    for t, (query, target) in enumerate(stream, 1):
        frame_id, phase = plan['frame_ids'][t], plan['phases'][t]
        with observe_forward(engine, filtered=mode == FILTERED) as captured:
            output, new = engine(query, state)
        row = dict(t=t, frame_id=frame_id, phase=phase, **prediction_metrics(output, target),
                   predicted_xyz=output['T1'][:3, 3].cpu().tolist(), reference_xyz=target.c2w[:3, 3].cpu().tolist(),
                   pose_source=output['write']['pose_source'], write=output['write'])
        if mode in BRANCHES[1:]:
            new, changed = attribute_write(engine, mode, state, new, output, captured['write'], target)
            row['write_attribution'] = changed
        if mode in (FROZEN, FILTERED):
            new = sync_seed(new, seed)
        if original:
            row['d4'] = final_pose_diagnostics(output, captured['raw_pose'], captured['learned_gate'], target)
        if original and natural:
            reference.append(target.world_points[target.valid][::4].cpu().numpy())
            row['d2'] = depth_statistics(output, target)
            row['d3'] = {}
            row['write_signals'] = write_features(output, captured['observation'][0][1], state.prev_pose)
            for source in ('seed', 'track', 'global'):
                pose, match, evidence = solve_candidate(engine.model, state, query, captured['observation'], source)
                row['d3'][source] = candidate_row(pose, match, evidence, state.prev_pose, target)
                if source == 'seed' and name == C11 and phase == 'Revisit':
                    seed_candidate = (pose, match, evidence)
            if frame_id in FOCUS[name]:
                for sigma, init_name in [(s, n) for s in (.15, .25, .35) for n in ('prev', 'velocity')]:
                    is_velocity = init_name == 'velocity'
                    init = extrapolate_pose(older_pose, state.prev_pose) if is_velocity else state.prev_pose
                    pose, match, evidence = solve_candidate(engine.model, state, query, captured['observation'],
                                                            'track', sigma=sigma, init=init)
                    row.setdefault('track_sweep', []).append(dict(sigma=sigma, initialization=init_name,
                        **candidate_row(pose, match, evidence, state.prev_pose, target)))
        if d1 and (phase == 'Revisit' or name == SEGMENTED):
            if original and name == C11:
                pose, match, evidence = seed_candidate
            else:
                pose, match, evidence = solve_candidate(engine.model, state, query, captured['observation'],
                                                       'seed', filtered=mode == FILTERED)
            overlap, valid = seed_overlap_grid(seed_K, seed_target, target, output['patch_hw'])
            row['d1'] = dict(metadata_seed_overlap=seed_surface_overlap(dataset, plan, frame_id),
                             **candidate_row(pose, match, evidence, state.prev_pose, target, output['camera_points']))
            if name == C11:
                row['d1']['weights'] = seed_weight_evidence(pose, match, overlap, valid, seed)
                if original:
                    # Same-history paired check isolates the direct changes in
                    # the seed solve from earlier differences in trajectory.
                    immutable = sync_seed(state, seed)
                    for alt in (FROZEN, FILTERED):
                        alt_pose, alt_match, alt_evidence = solve_candidate(engine.model, immutable, query,
                            captured['observation'], 'seed', filtered=alt == FILTERED)
                        row['d1'].setdefault('same_history', {})[alt] = dict(
                            **candidate_row(alt_pose, alt_match, alt_evidence, state.prev_pose, target, output['camera_points']),
                            weights=seed_weight_evidence(alt_pose, alt_match, overlap, valid, seed))
        if original and name == C11:
            row['seed_drift'] = seed_drift(new, seed)
        row['memory_after'] = len(new.memory)
        rows.append(row)
        older_pose, state = state.prev_pose, new
    result = dict(rows=rows)
    if natural and mode in BRANCHES:
        centers = state.memory.means.cpu().numpy()
        result['map_centers_xz'] = centers[:, [0, 2]].tolist()
        if original:
            # Same evaluator RGB-D surfaces for all write-only counterfactuals.
            result['reference'] = np.concatenate(reference)
        result['map_centers_xyz'] = centers
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_outcome(row):
    candidates = row['d3']
    selected = candidates[row['pose_source']]
    adequate = [key for key, value in candidates.items() if value['t0_m'] <= .50 and value['R0_deg'] <= 15.]
    dominates = [key for key, value in candidates.items() if key != row['pose_source']
                 and value['t0_m'] <= selected['t0_m']+.01 and value['R0_deg'] <= selected['R0_deg']+.5
                 and (value['t0_m'] < selected['t0_m']-.10 or value['R0_deg'] < selected['R0_deg']-3.)]
    return dict(adequate=adequate, selected_adequate=row['pose_source'] in adequate,
                all_mismatch=not adequate, rejected_adequate=bool(adequate) and row['pose_source'] not in adequate,
                materially_better_dominated=dominates)


def write_group_summary(rows):
    fields = ('inliers', 'visible', 'inlier_fraction', 'feature_cosine', 'raw_depth_relative_residual',
              'anchored_depth_relative_residual', 'z_cam_relative_residual', 'motion_m', 'motion_deg')
    good = [r['write_signals'] for r in rows if r['translation1_m'] <= .5 and r['write']['candidates']]
    bad = [r['write_signals'] for r in rows if r['translation1_m'] > 1. and r['write']['candidates']]

    def values(group, key):
        return [g[key] for g in group if g[key] is not None]

    def summarize(group):
        return dict(n=len(group), pose_sources={k:sum(g['pose_source'] == k for g in group) for k in ('seed','track','global')},
                    signals={k:describe(values(group, k)) for k in fields})

    def auc(key):
        # AUROC for healthy writes, oriented by magnitude only for residual/motion.
        a, b = values(good, key), values(bad, key)
        if not a or not b:
            return None
        oriented = np.asarray(a+b) * (-1 if 'residual' in key or 'motion' in key else 1)
        return float((rankdata(oriented)[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b)))

    def multiplier(g):
        if g['pose_source'] != 'global':
            return 1.
        overlap = np.clip(g['inlier_fraction']/.30, 0, 1)
        appearance = np.clip(((g['feature_cosine'] or 0)-.50)/.25, 0, 1)
        return float(.25+.75*overlap*appearance)

    def proxy(group):
        baseline = [g['candidates']*g['mean_confidence'] for g in group]
        adjusted = [w*multiplier(g) for w,g in zip(baseline,group)]
        global_ids = [i for i,g in enumerate(group) if g['pose_source'] == 'global']
        global_mass = sum(baseline[i] for i in global_ids)
        return dict(kept_positive=sum(multiplier(g) > 0 for g in group),
                    proxy_weight_retained=sum(adjusted)/sum(baseline) if sum(baseline) else None,
                    global_count=len(global_ids), global_weight_retained=(
                        sum(adjusted[i] for i in global_ids)/global_mass if global_mass else None))

    return dict(healthy=summarize(good), erroneous=summarize(bad),
                in_sample_auc={k:auc(k) for k in fields},
                fixed_soft_global_reweighting_proxy={'healthy':proxy(good), 'erroneous':proxy(bad)})


def summarize(data):
    old = data['replays'][ORIGINAL]['rows']
    result = {}
    if 'd3' in old[0]:
        outcomes = [candidate_outcome(r) for r in old]
        result['d3'] = dict(total=len(outcomes), all_mismatch=sum(x['all_mismatch'] for x in outcomes),
                            rejected_adequate=sum(x['rejected_adequate'] for x in outcomes),
                            selected_adequate=sum(x['selected_adequate'] for x in outcomes),
                            materially_better_alternative=sum(bool(x['materially_better_dominated']) for x in outcomes),
                            frame_outcomes=[dict(t=r['t'], frame_id=r['frame_id'], **o) for r,o in zip(old,outcomes)])
        result['write_signals'] = write_group_summary(old)
        if 'archived_write_cohorts' in data:
            result['archived_write_cohorts'] = data['archived_write_cohorts']
        depth = [d for r in old for d in r['d2']]
        result['d2_by_region'] = {
            region: {head: {
                'pixel_count': sum(d['pixels'] for d in depth if d['region'] == region and d['head'] == head),
                **{key: describe([d[key] for d in depth if d['region'] == region and d['head'] == head and d[key] is not None])
                   for key in ('bias_m','l1_m','median_ratio')}}
                for head in ('raw_dpt','o5_anchored','z_cam')}
            for region in ('covered','transition','frontier')}
    d4 = [r['d4'] for r in old]
    result['d4'] = {stage:{key:describe([r[stage][key] for r in d4]) for key in ('rotation_deg','translation_m')}
                    for stage in ('T0','T1_raw','T1_gated')}
    result['d4'].update({key:describe([r[key] for r in d4]) for key in (
        'raw_motion_m','raw_motion_deg','gated_motion_m','gated_motion_deg','gate','learned_gate','limiter','reliability')})
    def gains(stage, key):
        diffs = np.array([r['T0'][key]-r[stage][key] for r in d4])
        threshold = .001 if key == 'translation_m' else .05
        return dict(mean_gain=float(diffs.mean()), positive=int((diffs > threshold).sum()),
                    negative=int((diffs < -threshold).sum()))
    for stage in ('T1_raw', 'T1_gated'):
        result['d4'][stage]['gains'] = {key:gains(stage,key) for key in ('rotation_deg','translation_m')}
    result['d4']['raw_limiter_active_frames'] = sum(r['limiter'] < .999 for r in d4)
    result['d4']['lm_energy_decreased_frames'] = sum(r['lm_initial_energy'] > r['lm_final_energy']+1e-8 for r in d4)
    result['replay_metrics'] = {label: dict(
        all_EPE_m=describe([r['point_epe_m'] for r in replay['rows']]),
        final_translation_m=replay['rows'][-1]['translation1_m'],
        final_rotation_deg=replay['rows'][-1]['rotation1_deg'],
        final_map=replay.get('map_metrics'),
        revisit_seed_selections=sum(r['pose_source'] == 'seed' and r['phase']=='Revisit' for r in replay['rows']),
        changed_candidates=sum(r.get('write_attribution', {}).get('changed_candidates',0) for r in replay['rows']))
        for label, replay in data['replays'].items()}
    return result


def export_tables(output_dir, datasets):
    d1, d1_weights, d2, drift, candidates, sweep, writes, d4, attribution = ([] for _ in range(9))
    for name, data in datasets.items():
        for mode, run in data['replays'].items():
            for row in run['rows']:
                common = {key:row[key] for key in ('t','frame_id','phase')} | {'stream':name}
                if 'd1' in row:
                    variants = {mode:row['d1']} | row['d1'].get('same_history', {})
                    for variant, record in variants.items():
                        history = 'same_v3_state' if variant != mode else 'replayed'
                        identity = dict(common, variant=variant, history=history)
                        d1.append(dict(identity, **{k:v for k,v in record.items()
                            if k not in ('weights','same_history','pose')},
                            replay_pose_source=row['pose_source'] if history == 'replayed' else None,
                            replay_T0_m=row['translation0_m'] if history == 'replayed' else None,
                            replay_R0_deg=row['rotation0_deg'] if history == 'replayed' else None,
                            replay_EPE_m=row['point_epe_m'] if history == 'replayed' else None))
                        d1_weights.extend(dict(identity, **patch) for patch in record.get('weights', {}).get('patches', []))
                if 'seed_drift' in row:
                    drift.append(dict(common, **{f'{k}_{a}':b for k,v in row['seed_drift'].items()
                        for a,b in (v.items() if isinstance(v,dict) else [('value',v)])}))
                if 'd2' in row:
                    d2.extend(dict(common, **d) for d in row['d2'])
                    candidates.extend(dict(common, source=k, selected=k==row['pose_source'],
                        **{a:b for a,b in d.items() if a != 'pose'}) for k,d in row['d3'].items())
                    sweep.extend(dict(common, **{a:b for a,b in d.items() if a != 'pose'}) for d in row.get('track_sweep', []))
                    writes.append(dict(common, **row['write_signals'], evaluator_translation_m=row['translation1_m'],
                                       archived_write_group=row.get('archived_write_group')))
                if 'd4' in row:
                    d4.append(dict(common, **{f'{k}_{a}':b for k,v in row['d4'].items()
                        for a,b in (v.items() if isinstance(v,dict) else [('value',v)])}))
                if mode in BRANCHES and name != SEGMENTED:
                    attribution.append(dict(common, replay=mode,
                        offline_attribution=mode in (ORACLE_POSE,ORACLE_DEPTH),
                        **{k:row[k] for k in ('point_epe_m','translation1_m','rotation1_deg','memory_after')},
                        **row.get('write_attribution', {})))
    for name, rows in zip(('d1_seed_candidates','d1_patch_weights','d2_depth_regions','d1_seed_drift',
                           'd3_candidates','d3_tracking_sweep','d3_write_signals','d4_refinement',
                           'd2_attribution'), (d1,d1_weights,d2,drift,candidates,sweep,writes,d4,attribution)):
        write_csv(output_dir/(name+'.csv'), rows)


def draw_figures(datasets, output_dir):
    import matplotlib.pyplot as plt
    colors = dict(seed='#2a9d76', track='#2872bc', global_='#c36d36')
    short = {ORIGINAL:'v3', FROZEN:'Frozen', FILTERED:'Frozen + top 48',
             FRONTIER:'Frontier z_cam', ORACLE_POSE:'Oracle write pose', ORACLE_DEPTH:'Oracle write depth'}
    vis = output_dir/'vis'
    vis.mkdir(exist_ok=True)
    data = datasets[C11]
    rows = data['replays'][ORIGINAL]['rows']
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for j,(key,label) in enumerate((('feature','1 − cosine'),('alpha','Absolute opacity drift'))):
        for stat in ('mean','p90','max'):
            ax[0,j].plot([0]+[r['t'] for r in rows],[0]+[r['seed_drift'][key][stat] for r in rows],label=stat)
        ax[0,j].axvspan(42.5,51.4,alpha=.12,color='#8163a1')
        ax[0,j].set(xlabel='Time step',ylabel=label)
        ax[0,j].legend()
    for j, frame in enumerate((781,797)):
        r = next(r for r in rows if r['frame_id']==frame)
        same = [r['d1']]+[r['d1']['same_history'][k] for k in (FROZEN,FILTERED)]
        x = np.arange(3)
        ax[1,j].bar(x,[w['weights']['groups']['noncovisible']['weight_share'] for w in same],color='#d59354')
        ax[1,j].set(xticks=x,xticklabels=[short[k] for k in (ORIGINAL,FROZEN,FILTERED)],ylim=(0,1),
                    title=f'Frame {frame} · same v3 history',ylabel='Noncovisible weight share')
        for i,w in enumerate(same):
            g = w['weights']['groups']['noncovisible']['weight_share']
            ax[1,j].text(i,max(.06,g-.08),f'{w["inliers"]}/{w["visible"]}',ha='center',fontsize=9)
    for mode in (ORIGINAL,FROZEN,FILTERED):
        rr = [r for r in data['replays'][mode]['rows'] if 'd1' in r]
        ax[0,2].plot([r['frame_id'] for r in rr],[r['d1']['t0_m'] for r in rr],marker='.',label=short[mode])
        ax[1,2].plot([r['frame_id'] for r in rr],[r['d1']['R0_deg'] for r in rr],marker='.',label=short[mode])
    ax[0,2].set(ylabel='Seed candidate translation error [m]',title='Full replay · C11 return')
    ax[1,2].set(xlabel='Frame ID',ylabel='Seed candidate rotation error [°]')
    ax[0,2].legend(fontsize=8)
    fig.suptitle('D1 · C11 seed appearance and partial-overlap relocalization\nBar labels show rigid inliers / predicted visible patches')
    fig.savefig(vis/'d1_seed_lock_and_reloc.png',dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(4, 4, figsize=(16, 12), layout='constrained')
    methods = ('raw_dpt','o5_anchored','z_cam')
    for i,name in enumerate((C11,ROBOTICS)):
        data = datasets[name]
        rows = data['replays'][ORIGINAL]['rows']
        name_label = ('C11','robotics')[i]
        region_colors = ('#689bbd','#c9ad6e','#d47b5b')
        for j,region in enumerate(('covered','transition','frontier')):
            vals = [[d['bias_m'] for r in rows for d in r['d2']
                     if d['region']==region and d['head']==method and d['bias_m'] is not None] for method in methods]
            box = ax[0,i].boxplot(vals,positions=np.arange(3)*1.3+(j-1)*.28,widths=.24,patch_artist=True,
                                 boxprops=dict(facecolor=region_colors[j]), showfliers=False)
            box['boxes'][0].set_label(region)
        ax[0,i].set(xticks=np.arange(3)*1.3,xticklabels=['Raw DPT','O5 anchored','z_cam'],
                    ylabel='Per-frame signed depth bias [m]',title=name_label)
        ax[0,i].axhline(0,ls='--',c='.55',lw=.8)
        ax[0,i].legend(fontsize=7)
        for mode in BRANCHES:
            r = data['replays'][mode]['rows']
            style = '--' if mode in (ORACLE_POSE,ORACLE_DEPTH) else '-'
            ax[0,i+2].plot([x['t'] for x in r],[x['translation1_m'] for x in r],style,label=short[mode])
            ax[1,i].plot([x['t'] for x in r],[x['rotation1_deg'] for x in r],style,label=short[mode])
            ax[1,i+2].plot([x['t'] for x in r],[x['point_epe_m'] for x in r],style,label=short[mode])
        ax[0,i+2].set(title=name_label+' · translation',ylabel='Error [m]',xlabel='Time step')
        ax[1,i].set(title=name_label+' · rotation',ylabel='Error [°]',xlabel='Time step')
        ax[1,i+2].set(title=name_label+' · world EPE',ylabel='Error [m]',xlabel='Time step')
        ax[0,i+2].legend(fontsize=7)
        ref = np.asarray(data['reference_xz'])
        bounds = np.concatenate([ref]+[np.asarray(data['replays'][m]['map_centers_xz']) for m in BRANCHES])
        low, high = bounds.min(0)-.1, bounds.max(0)+.1
        trace = np.asarray([r['reference_xyz'] for r in rows])[:,[0,2]]
        for k,mode in enumerate(BRANCHES):
            axis = ax[2+i,k]
            run = data['replays'][mode]
            cloud = np.asarray(run['map_centers_xz'])
            path = np.asarray([r['predicted_xyz'] for r in run['rows']])[:,[0,2]]
            axis.scatter(*ref.T,s=.5,color='.55',alpha=.27)
            axis.scatter(*cloud.T,s=.65,color='#df8d55',alpha=.55)
            axis.plot(*trace.T,c='.25',lw=1)
            axis.plot(*path.T,c='#266bad',lw=1)
            axis.set(title=f'{name_label} · {short[mode]}',xlim=(low[0],high[0]),ylim=(low[1],high[1]),
                     xlabel='World x [m]',ylabel='World z [m]',aspect='equal')
    fig.suptitle(f'D2 · {ORACLE_LABEL}\nOracle branches replace only a write component. Gray = observed reference; orange = final map; blue = predicted trajectory.')
    fig.savefig(vis/'d2_frontier_curl_attribution.png',dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(4, 4, figsize=(16, 11), layout='constrained')
    for i,name in enumerate((C11,ROBOTICS)):
        rows = datasets[name]['replays'][ORIGINAL]['rows']
        name_label = ('C11','robotics')[i]
        for j,(key,label) in enumerate((('t0_m','Translation [m]'),('R0_deg','Rotation [°]'))):
            for source,c in zip(('seed','track','global'),colors.values()):
                ax[i,j].plot([r['t'] for r in rows],[r['d3'][source][key] for r in rows],label=source,color=c,alpha=.85)
            ax[i,j].scatter([r['t'] for r in rows],[r['d3'][r['pose_source']][key] for r in rows],
                            s=10,c='#252525',label='Selected')
            ax[i,j].set(title=f'{name_label} · T0 candidates',ylabel=label,xlabel='Time step')
            if i==0:
                ax[i,j].legend(fontsize=7,ncol=2)
        for j,(key,label) in enumerate((('translation_m','Translation [m]'),('rotation_deg','Rotation [°]'))):
            for stage, c in (('T1_raw','#c36d36'),('T1_gated','#2872bc')):
                ax[i,j+2].plot([r['t'] for r in rows],[r['d4'][stage][key]-r['d4']['T0'][key] for r in rows],
                               '.-',label=stage,color=c)
            ax[i,j+2].axhline(0, color='.6',lw=.8)
            ax[i,j+2].set(title=f'{name_label} · T1 minus T0 error',ylabel=label,xlabel='Time step')
            ax[i,j+2].legend(fontsize=7)
    rows=datasets[C11]['replays'][ORIGINAL]['rows']
    good = [r['write_signals'] for r in rows if r['archived_write_group'] == 'healthy']
    bad = [r['write_signals'] for r in rows if r['archived_write_group'] == 'erroneous']
    for j,(key,label) in enumerate((('feature_cosine','Rendered feature cosine'),('inlier_fraction','Visible inlier fraction'),
                                   ('raw_depth_relative_residual','Raw depth relative residual'))):
        vals=[[r[key] for r in group if r[key] is not None] for group in (good,bad)]
        ax[2,j].boxplot(vals)
        ax[2,j].set(title='C11 · archived write cohorts',ylabel=label,xticks=(1,2),
                    xticklabels=[f'Healthy ({len(good)})',f'Erroneous ({len(bad)})'])
    for name,label in ((C11,'C11 natural'),(ROBOTICS,'robotics'),(SEGMENTED,'C11 segmented')):
        rows=datasets[name]['replays'][ORIGINAL]['rows']
        ax[2,3].plot([r['t'] for r in rows],[r['d4']['gate'] for r in rows],label=label)
    ax[2,3].set(title='Effective gate',xlabel='Time step',ylabel='Gate multiplier')
    ax[2,3].legend(fontsize=7)
    segmented = datasets[SEGMENTED]['replays'][ORIGINAL]['rows']
    for j,(key,label) in enumerate((('translation_m','Translation [m]'),('rotation_deg','Rotation [°]'))):
        for stage,color in (('T1_raw','#c36d36'),('T1_gated','#2872bc')):
            errors = [r['d4'][stage][key]-r['d4']['T0'][key] for r in segmented]
            ax[3,j].plot([r['t'] for r in segmented],errors,'.-',label=stage,color=color)
        ax[3,j].axhline(0,color='.6',lw=.8)
        ax[3,j].set(title='C11 segmented · T1 minus T0 error',xlabel='Time step',ylabel=label)
        ax[3,j].legend(fontsize=7)
        for stage,offset,color in (('T1_raw',-.18,'#c36d36'),('T1_gated',.18,'#2872bc')):
            means = [np.mean([r['d4'][stage][key]-r['d4']['T0'][key]
                         for r in datasets[name]['replays'][ORIGINAL]['rows']])
                     for name in (C11,ROBOTICS,SEGMENTED)]
            ax[3,j+2].bar(np.arange(3)+offset,means,width=.36,label=stage,color=color)
        ax[3,j+2].axhline(0,color='.6',lw=.8)
        ax[3,j+2].set(title='Mean change in error',ylabel=label,xticks=(0,1,2),
                      xticklabels=('C11 natural','robotics','C11 segmented'))
        ax[3,j+2].legend(fontsize=7)
    fig.suptitle('D3–D4 · all candidates, observed write signals, and refinement\nT1 curves show error change relative to the T0 zero line')
    fig.savefig(vis/'d3_d4_candidates_and_t1_gate.png',dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only',action='store_true',help='Run GPU replays and save frame records')
    parser.add_argument('--plot-only',action='store_true',help='Rebuild tables and figures from saved frame records')
    parser.add_argument('--stream',choices=(C11,ROBOTICS,SEGMENTED),help='Collect one independent fixed plan')
    args = parser.parse_args()
    if args.stream and not args.collect_only or args.collect_only and args.plot_only:
        parser.error('Use --stream only with --collect-only, or run --plot-only after collection')
    plans = [p for p in json.loads((RUN/'evaluation_final/stream_plans.json').read_text())
             if p['name'] in (C11,ROBOTICS,SEGMENTED) and (not args.stream or p['name']==args.stream)]
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        fraction = os.environ.get('VGGT_GPU_MEMORY_FRACTION')
        if not fraction or not torch.cuda.is_available():
            raise RuntimeError('Use scripts/run_on_available_gpu.py for GPU replays')
        if float(fraction)*torch.cuda.get_device_properties(0).total_memory > 20*2**30:
            raise ValueError('GPU cap exceeds 20 GiB')
        torch.cuda.set_per_process_memory_fraction(float(fraction),0)
        torch.set_num_threads(2)
        from scripts.train_step3 import load_engine
        from src.streaming_data import StreamingDataset
        config = yaml.safe_load(CONFIG.read_text())
        engine = load_engine(config, checkpoint=RUN/'training_state.pt', device='cuda').eval().requires_grad_(False)
        dataset = StreamingDataset(PROJECT_ROOT/config['split'],config['image_hw'])
        for plan in plans:
            name = plan['name']
            variants = list(BRANCHES) if name != SEGMENTED else [ORIGINAL]
            if name != ROBOTICS:
                variants += [FROZEN,FILTERED]
            runs = {}
            for mode in variants:
                print('Replaying',name,mode,flush=True)
                runs[mode] = replay(engine,dataset,plan,mode)
            data = dict(plan=plan,replays=runs)
            if name != SEGMENTED:
                ref = runs[ORIGINAL].pop('reference')
                data['reference_xz'] = ref[::32,[0,2]].tolist()
                for mode in BRANCHES:
                    result = runs[mode]
                    result['map_metrics'] = reference_map_metrics(ref,result.pop('map_centers_xyz'))
            output = OUT/name
            output.mkdir(exist_ok=True)
            data['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            dump_json(output/'details.json',data)
            print('Finished',name,flush=True)
    compare_drift(plans,OUT)
    if args.collect_only:
        return
    data = {p['name']:json.loads((OUT/p['name']/'details.json').read_text()) for p in plans}
    summary = {name:summarize(record) | dict(baseline_replay_check=record['baseline_replay_check'],
                        peak_allocated_gib=record['peak_allocated_gib']) for name,record in data.items()}
    dump_json(OUT/'summary.json',dict(oracle_label=ORACLE_LABEL, checkpoint=str(RUN/'training_state.pt'),
                                     streams=summary))
    export_tables(OUT,data)
    draw_figures(data,OUT)


if __name__ == '__main__':
    main()
