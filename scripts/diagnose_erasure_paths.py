"""Fixed-weight causal path diagnosis using ONLY the model's own poses.

This is an evaluator intervention, not an inference mode or a trained ablation.
Normal V5 is unchanged. Neither GT poses nor fitted transforms enter a reader.
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.train_smoke import load_scene_samples
from src.models.vggt_memory import VGGTSpatialMemory
from src.losses.world import pose_errors
from torch.nn import functional as F


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results'):
        parser.error('Run must belong to this new project')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    checkpoint = load_checkpoint_readonly(folder/'trainable.pt')
    config, runtime = checkpoint['config'], load_config()
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], checkpoint['weight_source'],
                               revision=config.get('architecture_revision', 'v2')).cuda().eval()
    params = dict(model.named_parameters())
    for name, value in checkpoint['trainable'].items():
        params[name].copy_(value)
    evaluation = json.loads((folder/'evaluation.json').read_text())
    report = {}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout='constrained')
    for row, scene in enumerate(load_scene_samples(config, 'cuda')):
        query, target = scene['queries'][1]
        memory = model.build_memory(scene['mapping'])
        full = model(query, memory)
        bootstrap = full['bootstrap_attention']
        hp, wp = query.rgb.shape[-2]//14, query.rgb.shape[-1]//14
        fraction = F.adaptive_avg_pool2d(target.valid.float()[None, None], (hp, wp))[0, 0]
        world = F.adaptive_avg_pool2d((target.world_points*target.valid[..., None]).permute(2, 0, 1)[None], (hp, wp))[0].permute(1, 2, 0)/fraction.clamp_min(1e-6)[..., None]
        difference = world.reshape(-1, 3)[:, None]-memory.means[None]
        rho2 = torch.einsum('qmi,mij,qmj->qm', difference, torch.linalg.inv(memory.covars), difference)
        supported = (rho2.min(-1).values <= 9.) & (fraction.flatten() > .25)
        neighboring = rho2 <= 9.
        box = evaluation[scene['name']]['erasure']
        lower, upper = [memory.means.new_tensor(box[key]) for key in ('box_lower', 'box_upper')]
        erased, mask = memory.erase_box(lower, upper)
        local = ((target.world_points >= lower) & (target.world_points <= upper)).all(-1) & target.valid
        outside = target.valid & ~local
        before = (full['world_points']-target.world_points).norm(dim=-1)
        predictions = [('Normal erasure', model(query, erased))]
        # Hooks are used only in eval/no_grad, removed before each next case.
        # Both replacement poses came from model(rgb,K,full_memory), not GT.
        hook0 = model.initial_pose.register_forward_hook(lambda module, inputs, output: (full['T0'], output[1]))
        try:
            predictions.append(('Hold predicted T0', model(query, erased)))
            hook1 = model.refine_pose.register_forward_hook(lambda module, inputs, output: (full['T1'], output[1], output[2]))
            try:
                predictions.append(('Hold predicted T0 + T1', model(query, erased)))
            finally:
                hook1.remove()
        finally:
            hook0.remove()
        report[scene['name']] = {'bootstrap_diagnostic': dict(
            normalized_entropy=float((-(bootstrap*bootstrap.clamp_min(1e-9).log()).sum(-1)/np.log(len(memory))).mean()),
            supported_patch_fraction=float(supported.float().mean()),
            matching_mass_in_supported_3sigma=float((bootstrap*neighboring).sum(-1)[supported].mean()),
            uniform_mass_in_supported_3sigma=float(neighboring.float().mean(-1)[supported].mean()),
            anchor_error_supported_m=float((full['anchors']-world.reshape(-1, 3)).norm(dim=-1)[supported].mean()),
            erased_key_mass_mean=float(bootstrap[:, mask].sum(-1).mean()))}
        for column, (label, prediction) in enumerate(predictions):
            change = (prediction['world_points']-target.world_points).norm(dim=-1)-before
            inside_value, outside_value = float(change[local].mean()), float(change[outside].mean())
            report[scene['name']][label] = dict(epe_increase_inside_m=inside_value, epe_increase_outside_m=outside_value,
                                               translation_change_m=float((prediction['T1'][:3, 3]-full['T1'][:3, 3]).norm()),
                                               rotation0_change_deg=float(pose_errors(prediction['T0'], full['T0'])[0]),
                                               translation0_change_m=float(pose_errors(prediction['T0'], full['T0'])[1]),
                                               anchor_change_mean_m=float((prediction['anchors']-full['anchors']).norm(dim=-1).mean()))
            heat = change.cpu().numpy()
            valid = target.valid.cpu().numpy()
            ax = axes[row, column]
            im = ax.imshow(np.where(valid, heat, np.nan), cmap='coolwarm', vmin=-.15, vmax=.15)
            ax.contour(local.cpu().numpy().astype(float), levels=[.5], colors=['cyan'], linewidths=.7)
            ax.set_title(f"{scene['name']} · {label}\ninside/outside {inside_value:+.3f}/{outside_value:+.3f} m", fontsize=9)
            ax.axis('off')
    fig.suptitle('Diagnostic only: hold FULL-MEMORY model predictions, never GT / no alignment; normal V5 remains the first column', fontsize=10)
    fig.colorbar(im, ax=axes, shrink=.6, label='EPE change [m]; red=worse; cyan=original fixed bbox')
    fig.savefig(folder/'vis/erasure_path_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (folder/'erasure_path_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
