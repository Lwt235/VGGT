"""Read-only heldout coverage diagnostic; GT never enters query inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage, load_config, load_checkpoint_readonly, PROJECT_ROOT
configure_local_storage()
import json
import os
import torch
from src.data import Sun3DScene
from src.geometry import project, invert_se3
from src.models.vggt_memory import VGGTSpatialMemory
from scripts.train_smoke import load_scene_samples


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']), 0)
    checkpoint = load_checkpoint_readonly(PROJECT_ROOT/'results/smoke_seed17_v1/trainable.pt')
    runtime = load_config()
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], checkpoint['weight_source']).cuda().eval()
    params = dict(model.named_parameters())
    for name, value in checkpoint['trainable'].items():
        # Coverage depends only on the v1 Writer/fusion, not on its obsolete
        # pose/reader architecture. Official frozen DINO comes from its source.
        if name.startswith(('writer.', 'fusion.')):
            params[name].copy_(value)
    result = {}
    for item in load_scene_samples(checkpoint['config'], 'cuda'):
        memory = model.build_memory(item['mapping'])
        means, precision = memory.means.cpu(), torch.linalg.inv(memory.covars.cpu())
        scene = Sun3DScene(item['path'])
        _, truth = scene.load(6)  # Native sensor resolution; evaluator only.
        points = truth.world_points[truth.valid]
        frustum = torch.zeros(len(points), dtype=torch.bool)
        visible = frustum.clone()
        for index in [0, 2, 4]:
            image, mapping = scene.load(index)
            uv, z = project(points, image.K, invert_se3(mapping.c2w))
            h, w = mapping.depth.shape
            inside = (z > .1) & (z < 10) & (uv[:, 0] >= -.5) & (uv[:, 0] < w-.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5)
            frustum |= inside
            xy = uv.round().long()
            x, y = xy[:, 0].clamp(0, w-1), xy[:, 1].clamp(0, h-1)
            # Fixed 10 cm RGB-D visibility diagnostic, no tuning from query data.
            visible |= inside & mapping.valid[y, x] & ((mapping.depth[y, x]-z).abs() <= .10)
        neighborhood = torch.zeros_like(frustum)
        for start in range(0, len(points), 4096):
            delta = points[start:start+4096, None]-means[None]
            rho2 = torch.einsum('pmi,mij,pmj->pm', delta, precision, delta)
            neighborhood[start:start+4096] = (rho2 <= 9.).any(-1)
        result[item['name']] = dict(frame_id=scene.frame_ids[6], valid_points=len(points), sensor_hw=list(truth.depth.shape),
            mapping_frame_ids=item['mapping_frame_ids'], public_gaussians=len(memory),
            frustum_fraction=frustum.float().mean().item(), depth_visible_10cm_fraction=visible.float().mean().item(),
            ellipsoid_3sigma_fraction=neighborhood.float().mean().item(),
            frustum_or_ellipsoid_fraction=(frustum | neighborhood).float().mean().item(),
            visible_and_ellipsoid_fraction=(visible & neighborhood).float().mean().item())
    out = PROJECT_ROOT/'results/covisibility_seed17'
    out.mkdir(exist_ok=False)
    report = dict(memory_source='smoke_seed17_v1 fixed checkpoint; mapping [0,2,4] at 112x140',
                  diagnostic='Native-resolution heldout index6 valid metric points; GT evaluator only; no query forward or parameter selection', scenes=result)
    (out/'coverage.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
