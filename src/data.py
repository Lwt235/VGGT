"""Read-only SUN3D RGB-D assets; input and supervision are separate objects.

SUN3Dsfm's depth2XYZcamera.m uses MATLAB 1-based pixels, hence subtract one
from the stored principal point before using OpenCV integer-center geometry.
The original timestamped extrinsics are c2w, one block per one-based frame ID.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.geometry import invert_se3, unproject


def decode_sun3d_depth(encoded):
    value = np.asarray(encoded).astype(np.uint16)
    decoded = (value >> 3) | (value << 13)
    return decoded.astype(np.float32) / 1000.


def sun3d_K_to_cv(raw):
    K = torch.as_tensor(raw, dtype=torch.float32).clone()
    K[:2, 2] -= 1.
    return K


def resize_intrinsics(K, old_hw, new_hw):
    sy, sx = new_hw[0]/old_hw[0], new_hw[1]/old_hw[1]
    affine = K.new_tensor([[sx, 0., .5*sx-.5], [0., sy, .5*sy-.5], [0., 0., 1.]])
    return affine @ K


@dataclass(frozen=True)
class QueryInput:
    rgb: torch.Tensor
    K: torch.Tensor


@dataclass(frozen=True)
class Supervision:
    depth: torch.Tensor
    c2w: torch.Tensor
    world_points: torch.Tensor
    valid: torch.Tensor


class Sun3DScene:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        images = {int(p.name.split("-")[0]): p for p in (self.path/"image").glob("*.jpg")}
        depths = {int(p.name.split("-")[0]): p for p in (self.path/"depth").glob("*.png")}
        self.frame_ids = sorted(images.keys() & depths.keys())
        if not self.frame_ids:
            raise FileNotFoundError(f"No matched RGB/depth frame IDs in {self.path}")
        self.images, self.depths = images, depths
        extrinsics_files = sorted((self.path/"extrinsics").glob("*.txt"))
        if not extrinsics_files:
            raise FileNotFoundError(f"No c2w annotations in {self.path}")
        self.pose_file = extrinsics_files[-1]
        with self.pose_file.open("r") as handle:
            values = torch.from_numpy(np.loadtxt(handle).astype(np.float32))
        raw_c2w = values.reshape(-1, 3, 4)
        self.c2w = torch.eye(4).repeat(len(raw_c2w), 1, 1)
        self.c2w[:, :3] = raw_c2w
        if max(self.frame_ids) > len(self.c2w):
            raise ValueError("SUN3D frame number exceeds annotated pose count")
        with (self.path/"intrinsics.txt").open("r") as handle:
            self.K = sun3d_K_to_cv(np.loadtxt(handle))
        # Gauge fixed using the FIRST MAPPING frame before prediction, no fitting.
        self.world_from_native = invert_se3(self.c2w[self.frame_ids[0]-1])

    def __len__(self):
        return len(self.frame_ids)

    def load(self, index, *, resize_hw=None):
        frame_id = self.frame_ids[index]
        with self.images[frame_id].open("rb") as handle:
            with Image.open(handle) as image:
                rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float()/255.
        with self.depths[frame_id].open("rb") as handle:
            with Image.open(handle) as image:
                depth = torch.from_numpy(decode_sun3d_depth(image))
        if tuple(rgb.shape[-2:]) != tuple(depth.shape):
            raise ValueError("RGB/depth shape mismatch; calibrated registration is required")
        K = self.K.clone()
        if resize_hw:
            K = resize_intrinsics(K, depth.shape, resize_hw)
            rgb = F.interpolate(rgb[None], size=resize_hw, mode="bilinear", align_corners=False, antialias=True)[0]
            depth = F.interpolate(depth[None, None], size=resize_hw, mode="nearest-exact")[0, 0]
        c2w = self.world_from_native @ self.c2w[frame_id-1]
        points = unproject(depth, K, c2w)
        valid = (depth > .1) & (depth < 10.) & torch.isfinite(points).all(-1)
        return QueryInput(rgb, K), Supervision(depth, c2w, points, valid)
