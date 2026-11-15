"""Read-only fixed-weight diagnosis of fusion grouping under world gauges.

Only mapping poses are transported. Query RGB/K stay fixed. Compare rebuilding
the same mapping observations with rigidly transporting an already-built map.
No query ground truth enters either pose prediction.
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
from scripts.train_smoke import load_scene_samples
from src.augmentation import sample_world_gauge, transform_supervision
from src.geometry import transform_points, patch_intrinsics
from src.losses.world import pose_errors
from src.memory.gaussians import GaussianMemory
from src.models.vggt_memory import VGGTSpatialMemory, frozen_patch_tokens


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    folder = (PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results'):
        parser.error('Run must belong to the new project')
    output = folder/'world_gauge_diagnostic.json'
    if output.exists():
        parser.error('Existing diagnostic is preserved')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    checkpoint = load_checkpoint_readonly(folder/'trainable.pt')
    config, runtime = checkpoint['config'], load_config()
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], checkpoint['weight_source'],
                             revision=config['architecture_revision']).cuda().eval()
    parameters = dict(model.named_parameters())
    for name, value in checkpoint['trainable'].items():
        parameters[name].copy_(value)
    report = {}
    for scene in load_scene_samples(config, 'cuda'):
        query, _ = scene['queries'][1]
        tokens = frozen_patch_tokens(model.native.aggregator, query.rgb)[0].float()
        features = model.writer.latent(tokens)
        h, w = query.rgb.shape[-2:]
        Kp = patch_intrinsics(query.K)
        def pose(memory):
            return model.initial_pose(tokens, features, memory, Kp, h//14, w//14)[0]
        base = model.build_memory(scene['mapping'])
        Tbase = pose(base)
        records = []
        for _ in range(12):
            G = sample_world_gauge('cuda')
            transported = GaussianMemory(transform_points(base.means, G),
                                         G[:3, :3] @ base.covars @ G[:3, :3].T,
                                         base.alpha, base.features)
            mapping = [(query, transform_supervision(truth, G)) for query, truth in scene['mapping']]
            rebuilt = model.build_memory(mapping)
            direct_error = pose_errors(pose(transported), G @ Tbase)
            rebuilt_error = pose_errors(pose(rebuilt), G @ Tbase)
            records.append(dict(base_count=len(base), rebuilt_count=len(rebuilt),
                                transported_rotation_difference_deg=float(direct_error[0]),
                                transported_translation_difference_m=float(direct_error[1]),
                                rebuilt_rotation_difference_deg=float(rebuilt_error[0]),
                                rebuilt_translation_difference_m=float(rebuilt_error[1])))
        report[scene['name']] = records
        print(scene['name'], {key: sum(r[key] for r in records)/len(records) for key in records[0]}, flush=True)
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
