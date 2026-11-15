import torch
from src.data import Supervision
from src.geometry import transform_points
from src.losses.world import world_loss
from src.readout.render import RenderedMemory


def test_refinement_comparison_cannot_reward_making_t0_worse():
    t0 = torch.tensor(.1, requires_grad=True)
    t1 = torch.tensor(.2, requires_grad=True)
    T0, T1 = torch.eye(4), torch.eye(4)
    T0[0, 3], T1[0, 3] = t0, t1
    points = torch.zeros(2, 3, 3)
    points[..., 2] = 2.
    target = Supervision(torch.ones(2, 3)*2., torch.eye(4), points, torch.ones(2, 3, dtype=torch.bool))
    output = dict(world_points=transform_points(points, T1), coarse_world_points=transform_points(points, T0),
                  camera_points=points, T0=T0, T1=T1, anchors=torch.zeros(6, 3),
                  anchor_support_means=torch.zeros(0, 3), anchor_support_covars=torch.zeros(0, 3, 3),
                  rendered=RenderedMemory(points, torch.zeros(2, 3, 8), torch.ones(2, 3, 1)*2., torch.zeros(2, 3, 1), torch.ones(2, 3, 1)))
    base, _ = world_loss(output, target)
    supervised, _ = world_loss(output, target, refinement_safety_weight=2.)
    g0, g1 = torch.autograd.grad(supervised-base, (t0, t1))
    assert abs(g0.item()) < 1e-6
    assert g1.item() > 0
