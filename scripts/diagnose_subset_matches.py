"""Fixed-weight evaluator: separate anchor changes from voting changes.

Counterfactual moment solves below only diagnose the learned correspondences;
they never replace the normal prediction or its V5 score. GT labels are used
after inference solely to identify which original supported patches lost keys.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config, load_checkpoint_readonly
configure_local_storage()
import argparse
import json
import math
import os
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.train_smoke import load_scene_samples
from src.models.vggt_memory import VGGTSpatialMemory
from src.models.spatial_pose import anchor_vote_weights, whitened_irls_pose
from src.geometry import transform_points
from src.losses.world import pose_errors


def initial_votes(output, memory):
    weights = output['bootstrap_attention']
    confidence = 1+(weights*weights.clamp_min(1e-9).log()).sum(-1)/math.log(max(2, len(memory)))
    return (1-output['dustbin_probability'])*output['c_depth']*anchor_vote_weights(
        weights, memory, output['anchors'], confidence)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results'):
        parser.error('Run must belong to this project')
    output_path = folder/'subset_match_diagnostic.json'
    if output_path.exists():
        parser.error('Existing diagnostic is preserved')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    checkpoint = load_checkpoint_readonly(folder/'trainable.pt')
    config, runtime = checkpoint['config'], load_config()
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], checkpoint['weight_source'],
                              revision=config['architecture_revision']).cuda().eval()
    parameters = dict(model.named_parameters())
    for name, value in checkpoint['trainable'].items():
        parameters[name].copy_(value)
    evaluation = json.loads((folder/'evaluation.json').read_text())
    report = {}
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), layout='constrained')
    for row, scene in enumerate(load_scene_samples(config, 'cuda')):
        query, truth = scene['queries'][1]
        memory = model.build_memory(scene['mapping'])
        full = model(query, memory)
        box = evaluation[scene['name']]['erasure']
        partial, erased = memory.erase_box(*[memory.means.new_tensor(box[key]) for key in ('box_lower', 'box_upper')])
        after = model(query, partial)
        hp, wp = query.rgb.shape[-2]//14, query.rgb.shape[-1]//14
        valid_fraction = F.adaptive_avg_pool2d(truth.valid.float()[None, None], (hp, wp))[0, 0].flatten()
        world_patch = F.adaptive_avg_pool2d((truth.world_points*truth.valid[..., None]).permute(2, 0, 1)[None], (hp, wp))[0].permute(1, 2, 0).reshape(-1, 3)/valid_fraction.clamp_min(1e-6)[:, None]
        difference = world_patch[:, None]-memory.means[None]
        rho2 = torch.einsum('qmi,mij,qmj->qm', difference, torch.linalg.inv(memory.covars), difference)
        rho_min, nearest = rho2.min(-1)
        lost = (valid_fraction > .25) & (rho_min <= 9.) & erased[nearest]
        kept = (valid_fraction > .25) & (rho_min <= 9.) & ~erased[nearest]
        camera = full['camera_patch_points']
        anchors = full['anchors']
        votes, after_votes = initial_votes(full, memory), initial_votes(after, partial)
        baseline, _ = whitened_irls_pose(camera, anchors, votes)
        moments = {}
        for label, values, weights in (
            ('votes_changed_only', anchors, after_votes),
            ('anchors_changed_only', after['anchors'], votes),
            ('both_changed', after['anchors'], after_votes)):
            predicted, _ = whitened_irls_pose(camera, values, weights)
            rot, trans = pose_errors(predicted, baseline)
            moments[label] = dict(rotation_change_deg=float(rot), translation_change_m=float(trans))
        # Exact-correspondence control uses the model's own full pose and its
        # own camera points. It has no GT transform, no effect on normal V5.
        exact = transform_points(camera, baseline)
        Tfull, _ = whitened_irls_pose(camera, exact, votes)
        Tpartial, _ = whitened_irls_pose(camera, exact, after_votes*(~lost))
        rot, trans = pose_errors(Tpartial, Tfull)
        residual = (anchors-transform_points(camera, baseline)).norm(dim=-1)
        anchor_change = (after['anchors']-anchors).norm(dim=-1)
        def masked_mean(value, mask):
            return float(value[mask].mean()) if mask.any() else None
        report[scene['name']] = dict(
            lost_supported_patches=int(lost.sum()), kept_supported_patches=int(kept.sum()),
            lost_patch_dustbin_before=masked_mean(full['dustbin_probability'], lost),
            lost_patch_dustbin_after=masked_mean(after['dustbin_probability'], lost),
            lost_patch_rejection_after=masked_mean((after['dustbin_probability']>.5).float(), lost),
            kept_patch_anchor_change_m=masked_mean(anchor_change, kept),
            lost_patch_anchor_change_m=masked_mean(anchor_change, lost),
            kept_patch_baseline_rigid_residual_m=masked_mean(residual, kept),
            lost_patch_baseline_rigid_residual_m=masked_mean(residual, lost),
            pre_injection_camera_point_max_change_m=float((camera-after['camera_patch_points']).abs().max()),
            moment_pose_changes=moments,
            exact_correspondence_control=dict(rotation_change_deg=float(rot), translation_change_m=float(trans)))
        for column, (title, value, upper) in enumerate((
            ('Dustbin before', full['dustbin_probability'], 1.),
            ('Dustbin after', after['dustbin_probability'], 1.),
            ('Anchor change [m]', anchor_change, .3),
            ('Full rigid residual [m]', residual, .5))):
            ax = axes[row, column]
            heat = ax.imshow(value.reshape(hp, wp).cpu(), cmap='magma', vmin=0, vmax=upper)
            if lost.any():
                ax.contour(lost.reshape(hp, wp).float().cpu(), levels=[.5], colors=['cyan'], linewidths=.7)
            ax.set_title(scene['name']+' / '+title, fontsize=9)
            ax.axis('off')
            fig.colorbar(heat, ax=ax, shrink=.7)
    fig.suptitle('Evaluator only; cyan = originally supported patches whose nearest key was erased; no GT enters model')
    fig.savefig(folder/'vis/subset_match_diagnostic.png', dpi=150)
    plt.close(fig)
    output_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
