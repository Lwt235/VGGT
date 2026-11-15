import torch

from src.geometry import transform_points
from src.models.pose import PoseHead
from src.models.vggt_memory import aggregate_single, frozen_patch_tokens, official_vggt_class
from src.runtime import load_config


def test_pose_identity_initialization_rotation_and_gradients():
    head = PoseHead(12)
    prediction = head(torch.randn(12))
    torch.testing.assert_close(prediction, torch.eye(4))
    points = transform_points(torch.tensor([[1., 2., 3.]]), prediction)
    (points-torch.tensor([[1.1, 1.8, 3.2]])).square().sum().backward()
    assert head.net[-1].weight.grad.abs().sum() > 0


def test_native_block_adapter_matches_official_one_frame_aggregation():
    official_vggt_class(load_config()["assets"]["official_vggt_source"]["path"])
    from vggt.models.aggregator import Aggregator
    torch.manual_seed(17)
    torch.set_num_threads(2)
    # Small actual official class for API/gradient testing, never an overfit result.
    aggregator = Aggregator(img_size=56, embed_dim=32, depth=2, num_heads=4, patch_embed="conv", cached_layer_indices=(0, 1)).eval()
    image = torch.rand(3, 56, 56)
    native, start = aggregator(image[None, None])
    shallow = frozen_patch_tokens(aggregator, image)
    adapted, adapted_start = aggregate_single(aggregator, image, shallow)
    assert start == adapted_start
    for actual, expected in zip(adapted, native):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # Injection is outside checkpointed native blocks and survives backward.
    aggregator.train()
    gate = torch.tensor(.1, requires_grad=True)
    adapted, _ = aggregate_single(aggregator, image, shallow, lambda x: x+gate*x.sin(), receive_layer=0)
    adapted[-1].square().mean().backward()
    assert gate.grad is not None and torch.isfinite(gate.grad) and gate.grad.abs() > 0
