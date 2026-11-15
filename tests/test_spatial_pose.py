import math
import torch
from src.augmentation import sample_world_gauge, transform_supervision
from src.data import QueryInput, Supervision
from src.geometry import transform_points
from src.losses.world import rotation_geodesic
from src.memory.gaussians import GaussianMemory
from src.models.spatial_pose import SpatialBootstrap, SpatialRefinement, anchor_vote_weights
from src.readout.render import RenderReader, RenderedMemory


def test_geodesic_rotation_has_large_angle_gradient():
    theta = torch.tensor(math.pi-1e-4, requires_grad=True)
    zero, one = theta*0, theta*0+1
    R = torch.stack((torch.stack((theta.cos(), -theta.sin(), zero)),
                     torch.stack((theta.sin(), theta.cos(), zero)), torch.stack((zero, zero, one))))
    angle = rotation_geodesic(R, torch.eye(3))
    angle.backward()
    assert abs(angle.item()-(math.pi-1e-4)) < 1e-5
    assert theta.grad > .99
    identity = torch.eye(3, requires_grad=True)
    rotation_geodesic(identity, torch.eye(3)).backward()
    assert torch.isfinite(identity.grad).all()


def test_random_world_gauge_changes_only_supervision_and_transports_bootstrap():
    torch.manual_seed(17)
    h, w, dim = 3, 4, 16
    K = torch.tensor([[4., 0., 2.], [0., 4., 1.5], [0., 0., 1.]])
    query = QueryInput(torch.rand(3, 42, 56), K)
    truth = Supervision(torch.ones(42, 56), torch.eye(4), torch.rand(42, 56, 3), torch.ones(42, 56, dtype=torch.bool))
    G = sample_world_gauge('cpu')
    changed = transform_supervision(truth, G)
    torch.testing.assert_close(changed.c2w, G)
    torch.testing.assert_close(changed.world_points, transform_points(truth.world_points, G))
    assert changed.depth is truth.depth and changed.valid is truth.valid
    assert set(query.__dataclass_fields__) == {'rgb', 'K'}
    memory = GaussianMemory(torch.randn(30, 3), torch.eye(3).repeat(30, 1, 1)*.01,
                            torch.full((30,), .5), torch.randn(30, 8))
    transported = GaussianMemory(transform_points(memory.means, G), G[:3, :3]@memory.covars@G[:3, :3].T,
                                 memory.alpha, memory.features)
    model = SpatialBootstrap(dim, 8, robust_voting=True).eval()
    # Exercise learned geometric/context corrections, not only their initial
    # zero/identity values, under an arbitrary world gauge.
    with torch.no_grad():
        model.geometry_context.weight.normal_(std=.002)
        model.local_rotation[-1].weight.normal_(std=.002)
    tokens, features = torch.randn(h*w, dim), torch.randn(h*w, 8)
    T, _ = model(tokens, features, memory, K, h, w)
    TG, _ = model(tokens, features, transported, K, h, w)
    torch.testing.assert_close(TG, G@T, atol=3e-4, rtol=3e-4)
    T.square().sum().backward()
    assert model.coefficients.weight.grad.abs().sum() > 0


def test_o5_channel_gate_zero_identity_then_opens():
    reader = RenderReader(latent_dim=8, token_dim=16)
    memory = GaussianMemory(torch.zeros(0, 3), torch.zeros(0, 3, 3), torch.zeros(0), torch.zeros(0, 8))
    tokens = torch.randn(6, 16)
    # Even unseen branch must open: use a nonconstant learned blind vector.
    with torch.no_grad():
        reader.unseen.copy_(torch.randn(16))
    output, _ = reader(tokens, memory, torch.eye(4), torch.eye(3), 2, 3)
    torch.testing.assert_close(output, tokens, atol=0, rtol=0)
    (output-torch.randn_like(output)).square().mean().backward()
    assert reader.gate.shape == (16,) and reader.gate.grad.abs().sum() > 0


def test_residual_refinement_receives_spatial_geometry_and_detaches_base_pose():
    model = SpatialRefinement(token_dim=16)
    T0 = torch.eye(4, requires_grad=True)
    points = torch.randn(28, 42, 3, requires_grad=True)
    rendered = RenderedMemory(torch.randn(2, 3, 3), torch.randn(2, 3, 8), torch.ones(2, 3, 1),
                              torch.ones(2, 3, 1)*.1, torch.ones(2, 3, 1)*.5)
    T1, world, residual = model(torch.randn(6, 16), points, T0, rendered, torch.eye(3), 2, 3)
    torch.testing.assert_close(T1, torch.eye(4))
    (T1[:3, 3]-torch.tensor([.2, -.1, .3])).square().sum().backward()
    assert T0.grad is None
    assert model.pose.net[-1].weight.grad.abs().sum() > 0
    assert model.residual_pose.weight.grad.abs().sum() > 0
    assert world.shape == residual.shape == (2, 3, 3)


def test_local_metric_readout_changes_covered_camera_rays_and_is_differentiable():
    reader = RenderReader(latent_dim=8, token_dim=16)
    points = torch.zeros(28, 42, 3, requires_grad=True)
    rendered = RenderedMemory(torch.zeros(2, 3, 3), torch.randn(2, 3, 8), torch.ones(2, 3, 1)*2.,
                              torch.zeros(2, 3, 1), torch.ones(2, 3, 1))
    K = torch.tensor([[40., 0., 20.5], [0., 40., 13.5], [0., 0., 1.]])
    output, confidence = reader.anchor_camera_geometry(points, rendered, K)
    torch.testing.assert_close(output[..., 2], torch.ones(28, 42))
    output.square().mean().backward()
    assert reader.depth_confidence[-1].weight.grad.abs().sum() > 0
    assert points.grad.abs().sum() > 0
    rendered.opacity.zero_()
    uncovered, _ = reader.anchor_camera_geometry(points, rendered, K)
    torch.testing.assert_close(uncovered, points, atol=0, rtol=0)


def test_ambiguous_spatial_matches_have_less_influence_on_world_pose():
    means = torch.tensor([[-1., 0., 2.], [1., 0., 2.]], requires_grad=True)
    covars = (torch.eye(3).repeat(2, 1, 1)*.01).requires_grad_()
    memory = GaussianMemory(means, covars, torch.ones(2)*.5, torch.zeros(2, 8))
    weights = torch.tensor([[.99, .01], [.5, .5]], requires_grad=True)
    anchors = weights @ means
    votes = anchor_vote_weights(weights, memory, anchors, torch.ones(2)*.5)
    assert votes[0] > 2*votes[1]
    votes.sum().backward()
    assert torch.isfinite(means.grad).all() and means.grad.abs().sum() > 0
    assert torch.isfinite(covars.grad).all() and covars.grad.abs().sum() > 0
