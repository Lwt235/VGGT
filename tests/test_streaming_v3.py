import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.data import QueryInput
from src.geometry import patch_intrinsics, transform_points, world_rays
from src.memory.gaussians import GaussianMemory, SpatialMomentFusion, TokenGaussianWriter
from src.models.ellipsoid_pose import anisotropic_ray_pose, local_se3_exp
from src.models.spatial_pose import SpatialBootstrap
from src.models.stream_tracking import select_stream_pose, tracking_logit_prior
from src.readout.render import RenderedMemory
from src.streaming_data_v2 import StridedStreamingDataset
from src.streaming_v2 import detach_state
from src.streaming_v3 import StreamingEngineV3, StreamingStateV3


def spatial(means, features=None):
    return GaussianMemory(means, torch.eye(3).expand(len(means), 3, 3).clone()*.0025,
                          torch.ones(len(means))*.8, torch.ones(len(means), 4) if features is None else features)


def render(anchors, mask, height, width, depth=3.):
    opacity = mask.reshape(height, width, 1)*.8
    return RenderedMemory(anchors.reshape(height, width, 3), torch.ones(height, width, 4),
                          torch.ones_like(opacity)*depth, torch.zeros_like(opacity), opacity)


def intrinsics():
    return torch.tensor([[14., 0., 8.], [0., 16., 6.], [0., 0., 1.]])


def fixture():
    height, width = 12, 16
    _, rays, _ = world_rays(intrinsics(), torch.eye(4), height, width)
    anchors = rays*torch.linspace(2.4, 4.0, len(rays))[:, None]
    return height, width, rays, anchors


def test_verified_partial_seed_overlap_has_priority_under_world_gauge_change():
    h, w, rays, anchors = fixture()
    visible = torch.arange(h*w) < 44
    query_features = torch.eye(h*w)
    seed_count = int(visible.sum())
    local_memory = torch.cat((anchors[visible], anchors+torch.tensor([2., 0., 0.])))
    features = torch.cat((query_features[visible], query_features))
    seed_mask = torch.arange(len(local_memory)) < seed_count
    solver = SpatialBootstrap(token_dim=8, latent_dim=h*w, ray_bundle=True,
                              anisotropic=True, metric_scale=True, planar_safe=True)
    solver.camera_observation = lambda *_: (anchors, anchors[:, 2], torch.ones(h*w), rays)
    tokens = torch.zeros(h*w, 8)
    world = torch.eye(4)
    world[:3, :3], _ = local_se3_exp(torch.tensor([.5, .1, -.3]), torch.zeros(3))
    world[:3, 3] = torch.tensor([1.4, -2., .3])
    poses = []
    for gauge in (torch.eye(4), world):
        full = spatial(transform_points(local_memory, gauge), features)
        previous = gauge.clone()
        previous[:3, 3] += gauge[:3, 0]*2
        calls = []

        def raster(memory, pose, *_):
            calls.append(len(memory))
            mask = visible if len(memory) == seed_count else torch.arange(h*w)<174
            rasterized = render(transform_points(anchors, pose), mask, h, w)
            return RenderedMemory(rasterized.world, query_features.reshape(h,w,-1), rasterized.depth,
                                  rasterized.depth_variance, rasterized.opacity)

        model = SimpleNamespace(initial_pose=solver, render_reader=SimpleNamespace(render=raster))
        pose, out = select_stream_pose(model, tokens, tokens, query_features, full, intrinsics(),
                                       h, w, previous, seed_mask)
        assert calls == [seed_count] and torch.equal(out['matching_mask'], seed_mask)
        assert out['stream_pose_source'] == 'seed'
        assert out['stream_pose_evidence']['seed']['inliers'] == seed_count
        torch.testing.assert_close(pose, gauge, atol=5e-3, rtol=0)
        poses.append(pose)
    torch.testing.assert_close(poses[1], world @ poses[0], atol=2e-5, rtol=0)


def test_tracking_prior_and_lm_keep_small_motion_instead_of_symmetric_far_wall():
    h, w, rays, anchors = fixture()
    prev = torch.eye(4)
    motion = torch.tensor([.04, 0., 0.])
    near = anchors+motion
    far = near.clone()
    far[:, [0, 2]] *= -1
    means = torch.cat((near,far))
    world = torch.eye(4)
    world[:3, :3], _ = local_se3_exp(torch.tensor([.25, -.3, .2]), torch.zeros(3))
    world[:3, 3] = torch.tensor([2.1, -1.3, .2])
    prior = tracking_logit_prior(means,rays,prev)
    torch.testing.assert_close(prior,tracking_logit_prior(transform_points(means,world),rays,world),atol=8e-6,rtol=1e-5)
    assert (prior.diag() > prior[:,len(rays):].amax(-1)+10).all()

    class Solver:
        def __call__(self,*_, init_pose=None, matching_prior=None, **kwargs):
            assert init_pose is prev and matching_prior is not None
            appearance = -30.*torch.ones_like(matching_prior)
            idx = torch.arange(len(rays))
            appearance[idx, idx], appearance[idx, idx+len(rays)] = 0., 1.
            weights = (appearance+matching_prior).softmax(-1)
            observed = weights @ means
            cov = spatial(observed).covars
            pose, _ = anisotropic_ray_pose(anchors, rays, observed, cov, torch.ones(len(rays)),
                                          init_pose=init_pose)
            return pose, dict(anchors=observed, posterior_covars=cov, camera_rays=rays,
                              dustbin_probability=torch.full((len(rays),),.02))

    model = SimpleNamespace(initial_pose=Solver(),render_reader=SimpleNamespace(
        render=lambda *_: render(near,torch.ones(len(rays),dtype=torch.bool),h,w)))
    pose,out = select_stream_pose(model,None,None,torch.ones(len(rays),4),spatial(means),intrinsics(),
                                  h,w,prev,torch.zeros(len(means),dtype=torch.bool))
    assert out['stream_pose_source']=='track'
    torch.testing.assert_close(pose[:3,3],motion,atol=.015,rtol=0)


def test_loiter_repeated_writes_fuse_and_prev_pose_advances_even_on_rejection():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.writer=TokenGaussianWriter(8,4)
            self.fusion=SpatialMomentFusion()
            self.blind=False

        def forward(self, query, memory, *, streaming, streaming_previous_pose, streaming_seed_lock, return_attention):
            assert streaming and streaming_seed_lock.shape == (len(memory),)
            _, r, _=world_rays(patch_intrinsics(query.K),torch.eye(4),2,3)
            r=r.reshape(-1,3)
            pose=streaming_previous_pose.clone()
            if self.blind:
                pose[0,3]+=.2
            p=torch.full((6,),.99 if self.blind else .02)
            return dict(T1=pose, stream_writer_tokens=torch.ones(6,8), patch_hw=(2,3),
                camera_points=torch.full((28,42,3),3.),raw_camera_points=torch.full((28,42,3),.6),
                camera_rays=r,refined_dustbin_probability=p,dustbin_probability=p,c_depth=torch.ones(6),
                refined_anchors=r/r[:,2:]*3, refined_posterior_covars=spatial(r).covars,
                rendered_T1=render(r,torch.ones(6,dtype=torch.bool),2,3),
                attention={'indices':torch.arange(len(memory))},stream_pose_source='track',stream_pose_evidence={})

    model=Model()
    engine=StreamingEngineV3(model)
    query=QueryInput(torch.zeros(3,28,42),torch.tensor([[55.,0.,20.],[0.,55.,13.],[0.,0.,1.]]))
    with torch.no_grad():
        seed=model.writer(torch.ones(6,8),torch.full((2,3),3.),patch_intrinsics(query.K),torch.eye(4))
    state=StreamingStateV3(seed,torch.full((len(seed),),32.),0,
        seed_lock=torch.ones(len(seed),dtype=torch.bool),prev_pose=torch.eye(4))
    for _ in range(6):
        output,state=engine(query,state)
        assert output['write']['candidates']==6 and len(state.memory)==len(seed)
        assert 'stream_writer_tokens' not in output
    torch.testing.assert_close(state.memory.means,seed.means,atol=0,rtol=0)
    torch.testing.assert_close(state.memory.covars,seed.covars,atol=0,rtol=0)
    model.blind=True
    out,state=engine(query,state)
    assert out['write']['candidates']==0 and state.prev_pose[0,3]==pytest.approx(.2)
    assert detach_state(state).prev_pose.grad_fn is None
    assert list(inspect.signature(engine.forward).parameters)==['query','state','return_attention']


def test_training_jump_returns_to_seed_neighborhood_and_uses_only_train_frames():
    dataset=StridedStreamingDataset(Path(__file__).resolve().parents[1]/'configs/step2_splits.yaml',
                                    (168,224),jump_revisits=True)
    rng=random.Random(17)
    jumps=0
    for _ in range(80):
        clip=dataset.sample(rng)
        item=dataset.sequences[clip['scene']]
        assert item['partition']=='train'
        assert set(clip['frame_ids']) <= set(item['train_frame_ids'])
        if clip['mode']=='jump_revisit':
            jumps+=1
            assert sum(stride>4 for stride in clip['strides'])==1
    assert jumps>0
