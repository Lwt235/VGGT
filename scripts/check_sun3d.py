#!/usr/bin/env python3
"""Actual read-only three-scene format/reprojection diagnostic; no predictions."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage, load_config
configure_local_storage()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data import Sun3DScene
from src.geometry import invert_se3, project


def main():
    torch.set_num_threads(2)
    config = load_config()
    output = PROJECT_ROOT / "results/infrastructure"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), layout="constrained")
    report = []
    for row, path in enumerate(config["datasets"]["sun3d"]["train_scenes"]):
        scene = Sun3DScene(path)
        source, source_gt = scene.load(0)
        query, query_gt = scene.load(min(3, len(scene)-1))
        points = source_gt.world_points[::8, ::8].reshape(-1, 3)
        valid = source_gt.valid[::8, ::8].reshape(-1)
        uv, z = project(points, query.K, invert_se3(query_gt.c2w))
        height, width = query_gt.depth.shape
        ix = uv[:, 0].round().long().clamp(0, width-1)
        iy = uv[:, 1].round().long().clamp(0, height-1)
        in_image = (valid & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < height))
        valid_pairs = in_image & query_gt.valid[iy, ix]
        residual = (z[valid_pairs] - query_gt.depth[iy[valid_pairs], ix[valid_pairs]]).abs()
        decoded = source_gt.depth[source_gt.valid]
        report.append({"scene": path, "matched_frames": len(scene), "pose_file": str(scene.pose_file),
                       "mapping_frame_id": scene.frame_ids[0], "query_frame_id": scene.frame_ids[min(3, len(scene)-1)],
                       "shape": list(source.rgb.shape[-2:]), "depth_p10_p50_p90_m": torch.quantile(decoded, torch.tensor([.1, .5, .9])).tolist(),
                       "valid_fraction": source_gt.valid.float().mean().item(),
                       "projected_valid_pairs": int(valid_pairs.sum()),
                       "cross_view_abs_z_residual_median_m": residual.median().item() if len(residual) else None,
                       "cross_view_fraction_below_10cm": (residual < .1).float().mean().item() if len(residual) else None})
        axes[row, 0].imshow(source.rgb.permute(1, 2, 0))
        axes[row, 0].set_title(f"{scene.path.parent.name}: mapping RGB")
        axes[row, 1].imshow(query.rgb.permute(1, 2, 0))
        axes[row, 1].set_title("Query RGB (data check)")
        axes[row, 2].imshow(source_gt.depth, cmap="viridis", vmin=.1, vmax=5.)
        axes[row, 2].set_title("Decoded camera-z depth [m]")
        source_points = source_gt.world_points[::8, ::8][source_gt.valid[::8, ::8]]
        query_points = query_gt.world_points[::8, ::8][query_gt.valid[::8, ::8]]
        ax = axes[row, 3]
        # x-z slice in FIRST mapping camera gauge, no estimated alignment.
        ax.scatter(source_points[:, 0], source_points[:, 2], s=.5, color="#89939b", alpha=.4)
        ax.scatter(query_points[:, 0], query_points[:, 2], s=.5, color="#ff8a18", alpha=.4)
        ax.set_title("GT/GT overlap, NOT model V1")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world z [m]")
        for col in range(3):
            axes[row, col].axis("off")
    fig.suptitle("Read-only SUN3D diagnostic — real inputs / depth / annotated poses; no model predictions")
    fig.savefig(output / "sun3d_data_check.png", dpi=150)
    plt.close(fig)
    (output / "sun3d_data_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
