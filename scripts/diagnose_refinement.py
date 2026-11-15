"""Controlled module learnability diagnostic, NOT real-scene smoke evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage, PROJECT_ROOT
configure_local_storage()
import json
import torch
from src.models.spatial_pose import SpatialRefinement
from src.geometry import unproject, patch_intrinsics, quat_wxyz_to_matrix
from src.readout.render import RenderedMemory
from src.losses.world import rotation_geodesic


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    head = SpatialRefinement(token_dim=32)
    K = torch.tensor([[110., 0., 69.5], [0., 110., 55.5], [0., 0., 1.]])
    points = unproject(torch.ones(112, 140)*2., K, torch.eye(4))
    patch = torch.nn.functional.avg_pool2d(points.permute(2, 0, 1)[None], 14, 14)[0].permute(1, 2, 0)
    rendered = RenderedMemory(patch, torch.zeros(8, 10, 8), torch.ones(8, 10, 1)*2., torch.zeros(8, 10, 1), torch.ones(8, 10, 1))
    tokens = torch.randn(80, 32)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    records = []
    for step in range(400):
        T0 = torch.eye(4)
        T0[:3, 3] = torch.randn(3)*.1
        T0[:3, :3] = quat_wxyz_to_matrix(torch.cat((torch.ones(1), torch.randn(3)*.015)))
        T1, _, _ = head(tokens, points, T0, rendered, patch_intrinsics(K), 8, 10)
        r = rotation_geodesic(T1[:3, :3], torch.eye(3))
        t = T1[:3, 3].norm()
        loss = torch.nn.functional.smooth_l1_loss(t, t.new_zeros(()), beta=.03)+torch.nn.functional.smooth_l1_loss(r, r.new_zeros(()), beta=.03)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        records.append([T0[:3, 3].norm().item(), t.item(), torch.rad2deg(rotation_geodesic(T0[:3, :3], torch.eye(3))).item(), torch.rad2deg(r).item()])
    report = dict(scope='Ideal correspondence module diagnostic only; no real-scene or gsplat claims',
                  columns=['t0_m', 't1_m', 'R0_deg', 'R1_deg'], last100_mean=torch.tensor(records[-100:]).mean(0).tolist())
    output = PROJECT_ROOT/'results/refinement_conditioning'
    output.mkdir(exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
