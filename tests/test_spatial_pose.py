import math
import torch
import pytest
from src.augmentation import sample_world_gauge, transform_supervision
from src.data import QueryInput, Supervision
from src.geometry import transform_points
from src.losses.world import rotation_geodesic
from src.memory.gaussians import GaussianMemory
from src.models.spatial_pose import LegacySpatialBootstrap, SpatialBootstrap, SpatialRefinement, anchor_vote_weights, point_irls_pose, whitened_irls_pose
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
    model = LegacySpatialBootstrap(dim, 8, robust_voting=True).eval()
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


def test_learned_refinement_cannot_guess_a_pose_without_alignment_residual():
    torch.manual_seed(17)
    model = SpatialRefinement(token_dim=16, residual_only=True)
    with torch.no_grad():
        model.residual_pose.weight.normal_(std=.02)
        model.residual_modulator.weight.normal_(std=.001)
    points = torch.randn(28, 42, 3, requires_grad=True)
    patch = torch.nn.functional.adaptive_avg_pool2d(points.permute(2, 0, 1)[None], (2, 3))[0].permute(1, 2, 0)
    rendered = RenderedMemory(patch.detach().clone(), torch.randn(2, 3, 8), torch.ones(2, 3, 1),
                              torch.ones(2, 3, 1)*.01, torch.ones(2, 3, 1))
    T0 = torch.eye(4, requires_grad=True)
    T1, _, _ = model(torch.randn(6, 16), points, T0, rendered, torch.eye(3), 2, 3)
    torch.testing.assert_close(T1, T0, atol=1e-6, rtol=1e-6)
    rendered.world[..., 0] += .05
    changed, _, _ = model(torch.randn(6, 16), points, T0, rendered, torch.eye(3), 2, 3)
    assert (changed-T0).abs().max() > 1e-4
    changed[:3, 3].square().sum().backward()
    assert T0.grad is None
    assert model.residual_pose.weight.grad.abs().sum() > 0
    assert model.residual_modulator.weight.grad.abs().sum() > 0
    rendered.opacity.zero_()
    unseen, _, _ = model(torch.randn(6, 16), points, T0, rendered, torch.eye(3), 2, 3)
    torch.testing.assert_close(unseen, T0, atol=1e-6, rtol=1e-6)


def test_whitened_two_sided_pose_is_subset_invariant_and_gauge_equivariant():
    torch.manual_seed(17)
    h, w = 8, 10
    _, rays, dz = __import__('src.geometry', fromlist=['world_rays']).world_rays(
        torch.tensor([[8., 0., 5.], [0., 8., 4.], [0., 0., 1.]]), torch.eye(4), h, w)
    P = rays/dz[:, None]*(2+torch.rand(h*w, 1)*2)
    G = sample_world_gauge('cpu')
    X = transform_points(P, G)
    vote = torch.rand(h*w)+.2
    full, _ = point_irls_pose(P, X, vote)
    # Right-side local removal changes both centroids and camera covariance.
    keep = torch.arange(h*w).reshape(h, w).remainder(w).flatten() < 3*w//4
    partial, _ = point_irls_pose(P[keep], X[keep], vote[keep])
    torch.testing.assert_close(full, G, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(partial, full, atol=2e-5, rtol=2e-5)
    H = sample_world_gauge('cpu')
    transported, _ = point_irls_pose(P[keep], transform_points(X[keep], H), vote[keep])
    torch.testing.assert_close(transported, H@partial, atol=3e-5, rtol=3e-5)
    # A few low-quality, grossly wrong correspondences must be suppressed.
    bad = X.clone(); bad[[2, 15, 33]] += torch.tensor([2., -2., 3.])
    vote[[2, 15, 33]] = .1
    robust, _ = point_irls_pose(P, bad, vote)
    robust_subset, _ = point_irls_pose(P[keep], bad[keep], vote[keep])
    from src.losses.world import pose_errors
    for T in (robust, robust_subset):
        r, t = pose_errors(T, G)
        assert r < .2 and t < .01


def test_dustbin_rejects_erased_right_side_and_pre_geometry_is_memory_independent():
    from types import MethodType
    torch.manual_seed(17)
    h, w, dim, latent = 6, 8, 16, 48
    P = torch.randn(h*w, 3); P[:, 2] = P[:, 2].abs()+2
    G = sample_world_gauge('cpu')
    features = torch.eye(latent)
    memory = GaussianMemory(transform_points(P, G), torch.eye(3).repeat(latent, 1, 1)*.01,
                            torch.ones(latent)*.5, features)
    model = SpatialBootstrap(dim, latent).eval()
    with torch.no_grad(): model.log_temperature.fill_(math.log(60))
    # Controlled synthetic camera observation; this is a unit-test fixture,
    # never a query interface accepting measured camera geometry.
    def observation(self, tokens, shallow, K, height, width):
        return P, P[:, 2], torch.ones(len(P)), torch.nn.functional.normalize(P, dim=-1)
    model.camera_observation = MethodType(observation, model)
    tokens, shallow = torch.randn(latent, dim), torch.randn(latent, dim)
    full, df = model(tokens, features, memory, torch.eye(3), h, w, shallow=shallow)
    keep = torch.arange(latent).remainder(w) < 3*w//4
    partial, dp = model(tokens, features, memory.subset(keep), torch.eye(3), h, w, shallow=shallow)
    torch.testing.assert_close(df['camera_patch_points'], dp['camera_patch_points'], atol=0, rtol=0)
    assert dp['dustbin_probability'][~keep].mean() > .99
    torch.testing.assert_close(partial, full, atol=.002, rtol=.002)
    # The actual geometry head also has no memory argument or state dependency.
    real = SpatialBootstrap(dim, latent)
    a = real.camera_observation(tokens, shallow, torch.eye(3), h, w)[0]
    memory.means.add_(10)
    b = real.camera_observation(tokens, shallow, torch.eye(3), h, w)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_whitened_pose_gradients_and_empty_low_match_safety():
    torch.manual_seed(17)
    P = torch.randn(40, 3, requires_grad=True)
    X = (P.detach()+torch.randn(40, 3)*.02).requires_grad_()
    weights = torch.ones(40, requires_grad=True)
    T, _ = point_irls_pose(P, X, weights)
    (T[:3, 3]-.1).square().sum().backward()
    for x in (P, X, weights):
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    degenerate, _ = point_irls_pose(P.detach(), torch.ones_like(X), weights.detach())
    torch.testing.assert_close(degenerate[:3, :3].T@degenerate[:3, :3], torch.eye(3), atol=1e-6, rtol=1e-6)
    model = SpatialBootstrap(16, 8)
    empty = GaussianMemory(torch.zeros(0, 3), torch.zeros(0, 3, 3), torch.zeros(0), torch.zeros(0, 8))
    tokens = torch.randn(6, 16)
    T, diag = model(tokens, torch.randn(6, 8), empty, torch.eye(3), 2, 3, shallow=tokens)
    torch.testing.assert_close(T, torch.eye(4))
    assert (diag['dustbin_probability'] == 1).all()
    memory = GaussianMemory(torch.randn(10, 3), torch.eye(3).repeat(10, 1, 1)*.01, torch.ones(10)*.5, torch.randn(10, 8))
    with torch.no_grad(): model.dustbin_logit.fill_(100)
    T, _ = model(tokens, torch.randn(6, 8), memory, torch.eye(3), 2, 3, shallow=tokens)
    assert torch.isfinite(T).all()
    torch.testing.assert_close(T, torch.eye(4), atol=1e-6, rtol=1e-6)


def test_ray_bundle_irls_biased_depth_outliers_subset_and_gauge():
    from src.geometry import world_rays
    from src.losses.world import pose_errors
    torch.manual_seed(17)
    h, w = 8, 10
    K = torch.tensor([[8., 0., 5.], [0., 8., 4.], [0., 0., 1.]])
    _, rays, dz = world_rays(K, torch.eye(4), h, w)
    camera = rays/dz[:, None]*(2+torch.rand(h*w, 1)*2)
    biased = camera+rays*(.3+.08*torch.sin(torch.arange(h*w).float()))[:, None]
    G = sample_world_gauge('cpu')
    anchors = transform_points(camera, G)
    anchors[[2, 15, 33]] += torch.tensor([.3, -.3, 0.]) @ G[:3, :3].T
    weights = torch.ones(h*w)
    full, _ = whitened_irls_pose(biased, rays, anchors, weights)
    keep = torch.arange(h*w).remainder(w) < 3*w//4
    partial, _ = whitened_irls_pose(biased[keep], rays[keep], anchors[keep], weights[keep])
    for T in (full, partial):
        rotation, translation = pose_errors(T, G)
        assert rotation < .5 and translation < .025
    rotation, translation = pose_errors(partial, full)
    assert rotation < .5 and translation < .025
    H = sample_world_gauge('cpu')
    moved, _ = whitened_irls_pose(biased[keep], rays[keep], transform_points(anchors[keep], H), weights[keep])
    torch.testing.assert_close(moved, H@partial, atol=3e-5, rtol=3e-5)
    # Gradients traverse the robust weights, depth projections and linear solves.
    P = biased.requires_grad_()
    X = anchors.requires_grad_()
    votes = weights.requires_grad_()
    T, _ = whitened_irls_pose(P, rays, X, votes)
    (T[:3, 3]-G[:3, 3]+.05).square().sum().backward()
    for tensor in (P, X, votes):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_v2k_spatial_geometry_adaptive_dustbin_and_gauge_independence():
    from src.geometry import world_rays
    torch.manual_seed(17)
    model = SpatialBootstrap(16, 8, ray_bundle=True).eval()
    tokens, shallow, features = torch.randn(20, 16), torch.randn(20, 16), torch.randn(20, 8)
    K = torch.tensor([[5., 0., 2.5], [0., 5., 2.], [0., 0., 1.]])
    # Zero-tail residual starts preserve the compatible per-patch depth head.
    assert (model.dustbin_head(torch.cat((tokens, shallow), -1)) == 0).all()
    with torch.no_grad():
        model.camera_context[-1].weight.normal_(std=.01)
        model.dustbin_head[-1].weight.normal_(std=.01)
    points = model.camera_observation(tokens, shallow, K, 4, 5)[0]
    changed_tokens = tokens.clone(); changed_tokens[7] += 1.
    changed = model.camera_observation(changed_tokens, shallow, K, 4, 5)[0]
    assert (changed[6]-points[6]).abs().sum() > 0  # Neighboring patch context.
    memory = GaussianMemory(torch.randn(30, 3), torch.eye(3).repeat(30, 1, 1)*.01,
                            torch.ones(30)*.5, torch.randn(30, 8))
    T, full = model(tokens, features, memory, K, 4, 5, shallow=shallow)
    G = sample_world_gauge('cpu')
    moved_memory = GaussianMemory(transform_points(memory.means, G), G[:3, :3]@memory.covars@G[:3, :3].T,
                                  memory.alpha, memory.features)
    TG, moved = model(tokens, features, moved_memory, K, 4, 5, shallow=shallow)
    torch.testing.assert_close(full['camera_patch_points'], moved['camera_patch_points'], atol=0, rtol=0)
    torch.testing.assert_close(full['dustbin_probability'], moved['dustbin_probability'], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(TG, G@T, atol=3e-4, rtol=3e-4)
    empty = memory.subset(slice(0, 0))
    TE, _ = model(tokens, features, empty, K, 4, 5, shallow=shallow)
    torch.testing.assert_close(TE, torch.eye(4))
    with torch.no_grad():
        model.dustbin_logit.fill_(100.)
    rejected, _ = model(tokens, features, memory, K, 4, 5, shallow=shallow)
    assert torch.isfinite(rejected).all()
    torch.testing.assert_close(rejected, torch.eye(4), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('revision', ['v2k', 'v2l', 'step2_v2', 'step2_v3'])
def test_refinement_receives_raw_native_points_before_local_depth_anchor(monkeypatch, revision):
    from types import SimpleNamespace
    from torch import nn
    import src.models.vggt_memory as adapter
    raw = torch.randn(28, 42, 3, requires_grad=True)
    rendered = RenderedMemory(torch.zeros(2, 3, 3), torch.zeros(2, 3, 8), torch.ones(2, 3, 1),
                              torch.zeros(2, 3, 1), torch.ones(2, 3, 1))
    class NativeHead(nn.Module):
        def forward(self, *args, **kwargs):
            return raw[None, None], None
    class Bootstrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.eye(4, requires_grad=True)
        def forward(self, *args, **kwargs):
            return self.base, {}
    class Render(nn.Module):
        def forward(self, tokens, *args):
            self.seen_pose = args[1]
            return tokens, rendered
        def calibrate_camera_scale(self, points, rendered):
            return points*2, points.new_tensor(2.)
        def render(self, memory, pose, *args):
            if revision == 'step2_v3':
                assert not pose.requires_grad
            self.final_pose = pose
            self.final_render = RenderedMemory(rendered.world, rendered.features, rendered.depth+1,
                                               rendered.depth_variance, rendered.opacity)
            return self.final_render
        def anchor_camera_geometry(self, points, *args):
            if revision in ('step2_v2', 'step2_v3'):
                assert hasattr(self, 'final_pose')
                assert args[0] is self.final_render
            return points+1, torch.ones(2, 3, 1)
    class PE(nn.Module):
        def forward(self, tokens, *args, **kwargs):
            return tokens, {'attention': None}
    class Refinement(nn.Module):
        def __init__(self):
            super().__init__()
            self.correction = nn.Parameter(torch.tensor(.2))
        def forward(self, tokens, points, *args):
            if revision == 'step2_v3':
                torch.testing.assert_close(points, raw*2)
            else:
                assert points.data_ptr() == raw.data_ptr()
                torch.testing.assert_close(points, raw, atol=0, rtol=0)
            self.seen_pose = args[0]
            refined = args[0]
            if revision in ('step2_v2', 'step2_v3'):
                delta = torch.eye(4); delta[2, 3] = self.correction
                refined = refined @ delta
            return refined, torch.zeros(2, 3, 3), torch.zeros(2, 3, 3)
    model = adapter.VGGTSpatialMemory.__new__(adapter.VGGTSpatialMemory)
    nn.Module.__init__(model)
    model.revision, model.receive_layer = revision, 4
    model.native = nn.Module()
    model.native.aggregator = SimpleNamespace(patch_embed=nn.Identity())
    model.native.point_head = NativeHead()
    model.writer = nn.Module(); model.writer.latent = nn.Identity()
    model.initial_pose, model.render_reader, model.pe_reader, model.refine_pose = Bootstrap(), Render(), PE(), Refinement()
    monkeypatch.setattr(adapter, 'frozen_patch_tokens', lambda *args: torch.zeros(1, 6, 16))
    def aggregate(aggregator, rgb, shallow, receive, layer):
        tokens = receive(shallow[0])
        return [torch.cat((tokens, tokens), -1)[None, None]], 0
    monkeypatch.setattr(adapter, 'aggregate_single', aggregate)
    query = QueryInput(torch.rand(3, 28, 42), torch.eye(3))
    memory = GaussianMemory(torch.zeros(0, 3), torch.zeros(0, 3, 3), torch.zeros(0), torch.zeros(0, 8))
    prediction = model(query, memory)
    assert prediction['raw_camera_points'].data_ptr() == raw.data_ptr()
    baseline = raw*2 if revision == 'step2_v3' else raw
    torch.testing.assert_close(prediction['camera_points'], baseline+1)
    expected_world = baseline+1+(raw.new_tensor([0., 0., .2]) if revision in ('step2_v2', 'step2_v3') else 0)
    torch.testing.assert_close(prediction['world_points'], expected_world)
    if revision in ('step2_v2', 'step2_v3'):
        torch.testing.assert_close(model.render_reader.final_pose, prediction['T1'])
        assert not torch.equal(model.render_reader.final_pose, prediction['T0_read'])
    if revision == 'step2_v3':
        prediction['T1'].retain_grad()
        (prediction['world_points'].square().mean()+prediction['camera_points'].square().mean()).backward(retain_graph=True)
        assert prediction['T1'].grad is None
        assert model.refine_pose.correction.grad is None
        assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0
        prediction['T1'][:3,3].square().sum().backward()
        assert model.refine_pose.correction.grad.abs()>0
    if revision in ('v2l', 'step2_v2', 'step2_v3'):
        from src.models.ellipsoid_pose import sample_read_jitter
        jitter = sample_read_jitter('cpu')
        noisy = model(query, memory, read_jitter=jitter)
        torch.testing.assert_close(noisy['T0'], prediction['T0'])
        torch.testing.assert_close(noisy['T0_read'], jitter)
        torch.testing.assert_close(model.render_reader.seen_pose, jitter)
        torch.testing.assert_close(model.refine_pose.seen_pose, jitter)
        torch.testing.assert_close(noisy['coarse_world_points'], raw)
        assert not noisy['T0_read'].requires_grad and noisy['T0'].requires_grad
        model.eval()
        with pytest.raises(ValueError, match='training augmentation'):
            model(query, memory, read_jitter=jitter)
