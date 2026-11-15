"""Continuous 6D rotations and bootstrap/refinement camera heads."""

import torch
from torch import nn
from torch.nn import functional as F


def rotation_6d(value):
    first = F.normalize(value[..., :3], dim=-1, eps=1e-7)
    second = value[..., 3:]-(first*value[..., 3:]).sum(-1, keepdim=True)*first
    second = F.normalize(second, dim=-1, eps=1e-7)
    return torch.stack((first, second, torch.linalg.cross(first, second)), -1)


class PoseHead(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, 256), nn.GELU(), nn.Linear(256, 9))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:6] = torch.tensor([1., 0., 0., 0., 1., 0.])

    def forward(self, pooled):
        value = self.net(pooled)
        return pose_from_vector(value)


def pose_from_vector(value):
    rotation = rotation_6d(value[..., :6])
    top = torch.cat((rotation, value[..., 6:, None]), -1)
    bottom = value.new_tensor([0., 0., 0., 1.]).expand(*value.shape[:-1], 1, 4)
    return torch.cat((top, bottom), -2)
