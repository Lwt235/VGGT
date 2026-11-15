import torch
from src.data import Supervision
from src.losses.pose_consistency import pose_consistency_loss


def test_partial_map_pose_robustness_neither_changes_teacher_nor_forces_geometry():
    teacher_t = torch.tensor(.05, requires_grad=True)
    partial_t = torch.tensor(.2, requires_grad=True)
    teacher_T, partial_T = torch.eye(4), torch.eye(4)
    teacher_T[0, 3], partial_T[0, 3] = teacher_t, partial_t
    geometry = torch.randn(3, 4, 3, requires_grad=True)
    partial = dict(T0=partial_T, T1=partial_T, world_points=geometry)
    teacher = dict(T0=teacher_T, T1=teacher_T)
    target = Supervision(torch.ones(3, 4), torch.eye(4), torch.zeros(3, 4, 3), torch.ones(3, 4, dtype=torch.bool))
    loss, metrics = pose_consistency_loss(partial, teacher, target)
    loss.backward()
    assert partial_t.grad > 0
    assert teacher_t.grad is None
    assert geometry.grad is None
    assert metrics['counterfactual_t1_shift_m'] > .1
