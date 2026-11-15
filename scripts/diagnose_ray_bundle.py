"""Fixed-weight iteration diagnosis, never a replacement inference result.

No query GT enters this diagnostic. The bbox is the existing normal V5 bbox.
Iteration counts only inspect solver behavior; the model remains at three.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config, load_checkpoint_readonly
configure_local_storage()
import argparse
import json
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.train_smoke import load_scene_samples
from scripts.diagnose_subset_matches import initial_votes
from src.geometry import patch_intrinsics, world_rays
from src.losses.world import pose_errors
from src.models.vggt_memory import VGGTSpatialMemory
from src.models.spatial_pose import whitened_irls_pose
from src.models.ellipsoid_pose import anisotropic_ray_pose


def plot_diagnostic(folder, report, iterations=3):
    has_energy = 'full_energy' in next(iter(report.values()))[0]
    fig, axes = plt.subplots(3, 4 if has_energy else 3, figsize=(16 if has_energy else 12, 9), layout='constrained')
    for row, (scene, records) in enumerate(report.items()):
        steps = [r['iterations'] for r in records]
        axes[row, 0].plot(steps, [100*r['translation_change_m'] for r in records], 'o-')
        axes[row, 1].plot(steps, [r['rotation_change_deg'] for r in records], 'o-')
        for label in ('full', 'erased'):
            axes[row, 2].plot(steps, [r[label+'_effective_patches'] for r in records], 'o-', label=label)
            if has_energy:
                axes[row, 3].plot(steps, [r[label+'_energy'] for r in records], 'o-', label=label)
        titles = ['Translation difference [cm]', 'Rotation difference [deg]', 'Effective weighted patches']
        if has_energy:
            titles.append('Each map: robust energy')
        for column, title in enumerate(titles):
            axes[row, column].set_title(scene+'\n'+title, fontsize=10)
            axes[row, column].set_xlabel(f'Internal iterations (model uses {iterations})', fontsize=9)
            axes[row, column].axvline(iterations, color='grey', ls='--', lw=.8)
        axes[row, 2].legend()
    fig.suptitle('Fixed-weight full/erased solver diagnosis; normal V5 / model unchanged; no query GT', fontsize=11)
    fig.savefig(folder/'vis/ray_bundle_diagnostic.png', dpi=150)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results'):
        parser.error('Run must belong to this project')
    output_path = folder/'ray_bundle_diagnostic.json'
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
    for scene in load_scene_samples(config, 'cuda'):
        query, _ = scene['queries'][1]
        memory = model.build_memory(scene['mapping'])
        box = evaluation[scene['name']]['erasure']
        partial, _ = memory.erase_box(*[memory.means.new_tensor(box[key]) for key in ('box_lower', 'box_upper')])
        full, after = model(query, memory), model(query, partial)
        hp, wp = query.rgb.shape[-2]//14, query.rgb.shape[-1]//14
        _, rays, _ = world_rays(patch_intrinsics(query.K), torch.eye(4, device=query.rgb.device), hp, wp)
        votes, after_votes = initial_votes(full, memory), initial_votes(after, partial)
        records = []
        step_counts = (0, 1, 3, 4, 8, 12) if model.revision=='v2l' else (0, 1, 2, 3, 5, 8, 12)
        for steps in step_counts:
            if model.revision=='v2l':
                T, robust, df = anisotropic_ray_pose(full['camera_patch_points'], rays, full['anchors'], full['posterior_covars'], votes, iterations=steps, return_diagnostic=True)
                TE, robust_erased, de = anisotropic_ray_pose(after['camera_patch_points'], rays, after['anchors'], after['posterior_covars'], after_votes, iterations=steps, return_diagnostic=True)
            else:
                T, robust = whitened_irls_pose(full['camera_patch_points'], rays, full['anchors'], votes, iterations=steps)
                TE, robust_erased = whitened_irls_pose(after['camera_patch_points'], rays, after['anchors'], after_votes, iterations=steps)
            rotation, translation = pose_errors(TE, T)
            entry = dict(iterations=steps, rotation_change_deg=float(rotation), translation_change_m=float(translation))
            if model.revision=='v2l':
                entry.update(full_energy=float(df['energy_history'][-1]), erased_energy=float(de['energy_history'][-1]),
                             full_energy_nonincreasing=bool((torch.diff(df['energy_history']) <= 1e-6).all()),
                             erased_energy_nonincreasing=bool((torch.diff(de['energy_history']) <= 1e-6).all()))
            for label, pose, weights in (('full', T, robust), ('erased', TE, robust_erased)):
                weights = weights.double()
                direction = rays.double() @ pose[:3, :3].double().T
                Q = torch.eye(3, device=weights.device)-direction[:, :, None]*direction[:, None, :]
                if model.revision=='v2l':
                    precision = torch.linalg.inv((full if label=='full' else after)['posterior_covars'].double())
                    Ad = torch.einsum('qij,qj->qi', precision, direction)
                    Q = precision-Ad[:, :, None]*Ad[:, None, :]/((Ad*direction).sum(-1)+4.)[:, None, None]
                spectrum = torch.linalg.eigvalsh((weights[:, None, None]*Q).sum(0))
                entry[label+'_effective_patches'] = float(weights.sum().square()/weights.square().sum().clamp_min(1e-12))
                entry[label+'_vote_sum'] = float(weights.sum())
                entry[label+'_bundle_condition'] = float(spectrum[-1]/spectrum[0].clamp_min(1e-12))
            records.append(entry)
        report[scene['name']] = records
    plot_diagnostic(folder, report, iterations=4 if model.revision=='v2l' else 3)
    output_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
