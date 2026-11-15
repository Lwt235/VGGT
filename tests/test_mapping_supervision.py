import torch
from src.data import Supervision
from src.losses.mapping import mapping_observation_loss
from src.readout.render import RenderedMemory


def test_mapping_surface_coverage_prevents_opacity_escape_without_changing_truth():
    depth = torch.ones(28, 42)*2.
    world = torch.ones(28, 42, 3)
    truth = Supervision(depth, torch.eye(4), world, torch.ones(28, 42, dtype=torch.bool))
    opacity = torch.full((2, 3, 1), .1, requires_grad=True)
    rendered = RenderedMemory(torch.ones(2, 3, 3), torch.zeros(2, 3, 8), torch.ones(2, 3, 1)*2.,
                              torch.zeros(2, 3, 1), opacity)
    loss, metrics = mapping_observation_loss(rendered, truth)
    assert metrics['mapping_depth_l1_m'] == 0 and metrics['mapping_world_epe_m'] == 0
    assert metrics['mapping_coverage_loss'] > 2.
    loss.backward()
    assert torch.isfinite(opacity.grad).all() and (opacity.grad < 0).all()
    assert truth.world_points is world and truth.depth is depth
