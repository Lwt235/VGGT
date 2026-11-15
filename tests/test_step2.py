import os
from pathlib import Path
import pytest
import torch
from src.geometry import unproject
from src.models.spatial_pose import SpatialRefinement
from src.readout.render import RenderedMemory, RenderReader, camera_facing_normals, normalize_rendered_moments
from src.memory.gaussians import GaussianMemory
from src.step2_data import EpisodeDataset, frame_partition


def test_normal_alpha_normalization_and_finite_covariance_gradient():
    covariance = torch.diag(torch.tensor([.04, .04, .0004]))[None].requires_grad_()
    means = torch.tensor([[0., 0., 2.]])
    normal = camera_facing_normals(covariance, means, torch.zeros(3))
    torch.testing.assert_close(normal, torch.tensor([[0., 0., -1.]]))
    normal.sum().backward()
    assert torch.isfinite(covariance.grad).all()
    # world3 + feature1 + depth2 + normal3, all one alpha-weighted pass.
    attributes = torch.cat((means, torch.ones(1, 1), torch.tensor([[2., 4.]]), normal.detach()), -1).reshape(1, 1, -1)
    for alpha in (.2, .8):
        rendered = normalize_rendered_moments(attributes*alpha, torch.full((1, 1, 1), alpha), 1, with_normals=True)
        torch.testing.assert_close(rendered.normal[0, 0], normal[0])
        torch.testing.assert_close(rendered.depth.squeeze(), torch.tensor(2.))


def test_point_to_plane_ignores_coarse_tangent_centers_but_retains_normal_signal():
    h, w = 8, 10
    K = torch.tensor([[8., 0., 5.], [0., 8., 4.], [0., 0., 1.]])
    points = unproject(torch.full((h, w), 3.), K, torch.eye(4), center=.5)
    offset = torch.randn_like(points)*.2
    offset[..., 2] = 0
    normal = torch.zeros_like(points); normal[..., 2] = -1
    rendered = RenderedMemory(points+offset, torch.zeros(h, w, 4), torch.full((h, w, 1), 3.),
                              torch.zeros(h, w, 1), torch.ones(h, w, 1), normal)
    refinement = SpatialRefinement(token_dim=16, residual_only=True, point_to_plane=True)
    # A nonzero learned readout must not convert tangent-only errors into pose.
    torch.nn.init.normal_(refinement.residual_pose.weight, std=.05)
    tokens = torch.randn(h*w, 16)
    T, _, residual = refinement(tokens, points, torch.eye(4), rendered, K, h, w)
    torch.testing.assert_close(T, torch.eye(4), atol=1e-6, rtol=1e-6)
    assert residual.abs().max() < 1e-6
    rendered.world = rendered.world+torch.tensor([0., 0., .05])
    T, _, residual = refinement(tokens, points, torch.eye(4), rendered, K, h, w)
    assert residual.norm() > .1 and (T-torch.eye(4)).abs().max() > 1e-4


def test_fixed_split_and_dynamic_sampling_do_not_train_on_heldout_or_val():
    assert frame_partition([1, 2, 3])[2] is not None
    dataset = EpisodeDataset(Path(__file__).resolve().parents[1]/'configs/step2_splits.yaml', (84, 112))
    import random
    rng = random.Random(17)
    val = {s for s, (_, item) in dataset.scenes.items() if item['partition']=='val'}
    assert any('harvard_c11/' in s for s in val)
    assert not val & {s for s, _ in dataset.training_queries}
    for _ in range(10):
        episode = dataset.sample(rng)
        item = dataset.scenes[episode['name']][1]
        used = [episode['query_frame_id'], *episode['mapping_frame_ids']]
        assert not set(used) & set(item['heldout_frame_ids'])
        torch.testing.assert_close(episode['mapping'][0][1].c2w, torch.eye(4), atol=1e-5, rtol=1e-5)
        assert episode['query_frame_id'] not in episode['mapping_frame_ids']


@pytest.mark.gpu
def test_real_o5_renders_camera_facing_normal_in_same_moment_pass():
    if not torch.cuda.is_available():
        if os.environ.get('REQUIRE_GSPLAT') == '1':
            pytest.fail('CUDA is required for actual O5 normal rendering')
        pytest.skip('CUDA unavailable')
    reader = RenderReader(latent_dim=4, token_dim=16, surface_normals=True).cuda()
    means = torch.tensor([[0., 0., 2.]], device='cuda', requires_grad=True)
    covariance = torch.diag(torch.tensor([.04, .04, .0004], device='cuda'))[None].requires_grad_()
    memory = GaussianMemory(means, covariance, means.new_tensor([.8]), means.new_zeros(1, 4))
    render = reader.render(memory, torch.eye(4, device='cuda'), means.new_tensor([[30., 0., 8.5], [0., 30., 8.5], [0., 0., 1.]]), 17, 17)
    visible = render.opacity[..., 0] > .1
    torch.testing.assert_close(render.normal[visible], means.new_tensor([0., 0., -1.]).expand(int(visible.sum()), 3))
    (render.world.sum()+render.normal.sum()).backward()
    assert torch.isfinite(covariance.grad).all() and torch.isfinite(means.grad).all()
