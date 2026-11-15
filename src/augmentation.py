"""Training-only world gauge changes; query RGB/K are never transformed."""
import torch
from src.data import Supervision
from src.geometry import quat_wxyz_to_matrix, transform_points


def sample_world_gauge(device, translation_extent_m=3.):
    # Normalized isotropic Gaussian quaternion is uniform on SO(3).
    q = torch.randn(4, device=device)
    G = torch.eye(4, device=device)
    G[:3, :3] = quat_wxyz_to_matrix(q)
    G[:3, 3] = (torch.rand(3, device=device)*2-1)*translation_extent_m
    return G


def transform_supervision(target, G):
    return Supervision(target.depth, G @ target.c2w,
                       transform_points(target.world_points, G), target.valid)


def augment_training_world(mapping, target):
    G = sample_world_gauge(target.c2w.device)
    return [(rgb_k, transform_supervision(truth, G)) for rgb_k, truth in mapping], transform_supervision(target, G), G
