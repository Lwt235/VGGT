from dataclasses import fields
from types import SimpleNamespace

import torch
from torch import nn

from src.data import QueryInput, Supervision
from src.geometry import transform_points, unproject
from src.losses.world import pose_errors, seed_partial_matching_loss
from src.memory.gaussians import GaussianMemory
from src.models.deep_refinement import DeepEllipsoidRefinement
from src.models.stream_tracking_v4 import (
    bounded_motion, open_attention, reachable_penalty, render_evidence, seed_fit, track_membership,
)
from src.readout.render import RenderReader, RenderedMemory
from src.streaming import WriteConfidence
from src.streaming_distributed import TrajectorySegment
from src.streaming_v4 import StreamingEngineV4, StreamingStateV4, feature_reliability, soft_write_support


def make_pose(xyz=(0.,0.,0.), angle=0.):
    angle = torch.deg2rad(torch.tensor(angle))
    co, si = float(angle.cos()),float(angle.sin())
    pose = torch.eye(4)
    pose[:3,:3] = torch.tensor([[co,0,si],[0,1,0],[-si,0,co]])
    pose[:3,3] = torch.as_tensor(xyz)
    return pose


def test_mutual_columns_discount_repeated_background_and_keep_dustbin_gradients():
    logits = torch.full((100,4),-10.)
    logits[:75,0] = 6.
    logits[75:99,1] = 5.
    logits[99,2] = 7.
    logits.requires_grad_()
    dustbin = torch.full((100,1),5.,requires_grad=True)
    independent = open_attention(logits,dustbin)
    dual = open_attention(logits,dustbin,mutual=True)
    assert dual[0,-1] > .7 and independent[0,-1] < .3
    assert torch.allclose(dual[99],independent[99],atol=1e-6)
    loss = -dual[0,-1].log()-dual[99,2].log()
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(dustbin.grad).all()
    assert logits.grad[0,0] != 0 and dustbin.grad[0,0] != 0


def partial_overlap_fixture():
    gen = torch.Generator().manual_seed(91)
    count, supported = 192,46
    points = torch.rand((count,3),generator=gen)
    points[:,:2] = (points[:,:2]-.5)*2.
    points[:,2] = 1.8+3*points[:,2]
    rays = torch.nn.functional.normalize(points,dim=-1)
    truth = make_pose((1.3,-.6,.4),63.)
    exact = transform_points(points,truth)
    extra = torch.rand((34,3),generator=gen)*3+torch.tensor([-1.,-.7,.5])
    means = torch.cat((exact[:supported],extra))
    covars = torch.eye(3)[None].repeat(len(means),1,1)*.04**2
    seed = GaussianMemory(means,covars,torch.full((len(means),),.9),torch.ones(len(means),2))
    logits = torch.full((count,len(seed)),-12.)
    ix = torch.arange(supported)
    logits[ix,ix] = 13.
    # A high-scoring repeated wall claims 76% of patch rows.
    bg = torch.arange(supported,count)
    logits[bg,supported+bg.remainder(4)] = 11.5
    W = open_attention(logits,torch.full((count,1),10.),mutual=True)
    return points,rays,seed,W,truth


def test_partial_seed_pose_with_three_quarters_outliers_and_world_gauge():
    points,rays,seed,W,truth = partial_overlap_fixture()

    def appearance(_, probability, distances):
        # Geometry-only mock renderer for the small CPU fixture. A visible
        # ellipsoid contributes its matching feature without any GT input.
        visible = (distances < 4).float().reshape(-1,1)
        render = SimpleNamespace(opacity=visible,depth=visible,features=torch.ones(len(visible),2))
        return render_evidence(probability,distances,render,torch.ones_like(render.features))

    conf, mask = torch.ones(len(points)),torch.ones(len(seed),dtype=torch.bool)
    fit,_,_,_,lm,count = seed_fit(points*.8,rays,W,conf,seed,mask,appearance=appearance)
    R,t = pose_errors(fit,truth)
    assert count >= 16 and R < .4 and t < .01
    assert (lm['energy_history'][1:] <= lm['energy_history'][:-1]).all()
    gauge = make_pose((-.8,1.4,.3),-33.)
    rot = gauge[:3,:3]
    changed = GaussianMemory(transform_points(seed.means,gauge),rot @ seed.covars @ rot.T,seed.alpha,seed.features)
    again,*_ = seed_fit(points*.8,rays,W,conf,changed,mask,appearance=appearance)
    assert torch.allclose(again,gauge @ fit,atol=3e-4,rtol=3e-4)


def test_tracking_initialization_and_flat_reachability_have_no_old_pose_soft_bias():
    previous, older = make_pose((.2,0.,0.),10.),make_pose()
    predicted = bounded_motion(previous,older)
    assert pose_errors(predicted,previous)[1] > .09
    assert torch.equal(bounded_motion(make_pose(angle=75.),older),make_pose(angle=75.))
    local = torch.tensor([[0.,0.,2.],[.7,0.,2.],[.2,0.,-3.],[0.,0.,15.]])
    means = transform_points(local,predicted)
    rays = torch.tensor([[0.,0.,1.]])
    prior = reachable_penalty(means,rays,predicted)
    assert prior[0,0] == prior[0,1] == 0
    assert (prior[0,2:] < -6).all()
    logits = torch.tensor([[2.,2.,6.,7.]])
    pairs = track_membership(logits,means,rays,predicted)
    W = open_attention(logits,torch.zeros(1,1),pair_mask=pairs)
    assert torch.equal(pairs,torch.tensor([[True,True,False,False]]))
    assert W[0,0] == W[0,1] and W[0,2] < 1e-15
    gauge = make_pose((.5,1.,.4),42.)
    assert torch.allclose(bounded_motion(gauge @ previous,gauge @ older),gauge @ predicted,atol=1e-6)


def test_deep_local_refinement_does_not_reuse_additive_tracking_bias():
    _,rays,seed,_,_ = partial_overlap_fixture()
    n,d = 4,2
    memory = GaussianMemory(seed.means[:3],seed.covars[:3],seed.alpha[:3],seed.features[:3])
    q = torch.tensor([[1.,0.]]).expand(n,d)
    k = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    pair = torch.tensor([[True,True,False]]).expand(n,3)
    out = DeepEllipsoidRefinement(4,d)(torch.ones(n,4),memory,dict(
        matching_query=q,matching_keys=k,matching_temperature=torch.tensor(2.),
        matching_dustbin_logit=torch.zeros(n,1),matching_prior=torch.tensor([[10.,-10.,0.]]).expand(n,3),
        matching_column_competition=False,matching_pair_mask=pair,matching_mask=None,
        camera_rays=rays[:n],c_depth=torch.ones(n)),torch.eye(4))[-1]
    W = out['refined_matching_attention']
    assert torch.equal(W[:,0],W[:,1]) and W[:,2].max() < 1e-15


def test_appearance_protects_o5_correction_and_soft_write_never_hard_rejects_depth():
    opacity = torch.tensor([[[1.],[.01]]])
    features = torch.tensor([[[1.,0.],[1.,0.]]])
    depth = torch.full_like(opacity,3.)
    render = RenderedMemory(torch.zeros(1,2,3),features,depth,torch.zeros_like(depth),opacity)
    aligned, distant = torch.tensor([[1.,0.],[1.,0.]]),torch.tensor([[0.,1.],[0.,1.]])
    good_cosine, good_gate = feature_reliability(render,aligned)
    bad_cosine, bad_gate = feature_reliability(render,distant)
    reader = RenderReader(latent_dim=2,token_dim=2,physical_local=True,camera_frame=True)
    points = torch.zeros(14,28,3)
    points[...,2] = 5.
    untouched,_ = reader.anchor_camera_geometry(points,render,torch.eye(3),reliability=bad_gate)
    corrected,_ = reader.anchor_camera_geometry(points,render,torch.eye(3),reliability=good_gate)
    assert torch.equal(untouched,points) and corrected[6,6,2] < 3.6
    good = soft_write_support(render,good_cosine,torch.full((1,2),3.),'track')[0]
    weaker = soft_write_support(render,good_cosine,torch.full((1,2),3.),'global')[0]
    bad = soft_write_support(render,bad_cosine,torch.full((1,2),8.),'global')[0]
    assert .10 <= bad < weaker < good <= 1.
    gate = WriteConfidence()
    output = dict(rendered_T1=render,dustbin_probability=torch.zeros(2),c_depth=torch.ones(2))
    before,_ = gate(torch.tensor([[3.,1.]]),output,frontier_threshold=.05)
    after,_ = gate(torch.tensor([[3.,11.]]),output,frontier_threshold=.05)
    assert torch.equal(before,after)


def streaming_fixture(window=None):
    fake_model = nn.Module()
    fake_model.fusion = nn.Module()
    engine = StreamingEngineV4(fake_model,window=window)
    gaussian = GaussianMemory(torch.tensor([[0.,0.,2.]]),.015*torch.eye(3)[None],
                              torch.tensor([.5]),torch.ones(1,2))
    support = torch.full((1,),engine.fusion.support_cap)
    from src.streaming import SpatialWrite
    history = () if window is None else (SpatialWrite(gaussian,support,0),)
    state = StreamingStateV4(gaussian,support,0,history,torch.ones(1,dtype=torch.bool),
                            prev_pose=torch.eye(4))
    return engine,state


def test_seed_fields_and_support_are_locked_until_w4_expiration():
    for window in (None,4):
        engine,state = streaming_fixture(window)
        seed = state.memory
        from dataclasses import replace
        for i in range(1,6):
            candidate = GaussianMemory(seed.means+.005,.95*seed.covars,
                                       torch.tensor([.9]),-2*seed.features)
            memory,mass,lock,entries = engine.fuse_write(state,candidate,torch.ones(1))
            if window is None or i<4:
                for field in ('means','covars','alpha','features'):
                    assert torch.equal(getattr(memory,field)[lock],getattr(seed,field))
                assert mass[lock].eq(engine.fusion.support_cap).all()
            else:
                assert not lock.any()
            state = replace(state,memory=memory,support=mass,seed_lock=lock,step=i,window_entries=entries)
        assert not {'rgb','kv','depth'} & {f.name for f in fields(state)}


def test_seed_auxiliary_loss_is_criterion_only_with_positive_and_dustbin_labels():
    truth_depth = torch.full((4,4),2.)
    target = Supervision(truth_depth,torch.eye(4),unproject(truth_depth,torch.eye(3),torch.eye(4)),
                         torch.ones_like(truth_depth,dtype=torch.bool))
    scores = torch.zeros(4,2,requires_grad=True)
    W = scores.softmax(-1)
    auxiliary = dict(matching_attention=W,anchor_support_means=torch.tensor([[1.,1.,2.]]),
        anchor_support_covars=torch.eye(3)[None]*.30,anchors=torch.ones(4,3),patch_hw=(2,2))
    loss,enabled = seed_partial_matching_loss(auxiliary,target,.099)
    assert not enabled and loss == 0
    loss,enabled = seed_partial_matching_loss(auxiliary,target,.10)
    assert enabled and loss > 0
    loss.backward()
    assert scores.grad[0,0] < 0 and scores.grad[-1,-1] < 0


def test_ddp_segment_supplies_overlap_to_criterion_only():
    base,state = streaming_fixture()
    seen = []

    class Engine(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.base = base

        def forward(self, query, state):
            assert type(query) is QueryInput
            assert type(state) is StreamingStateV4
            return dict(loss=self.scale.square(),write={},active_count=0),state

    truth = object()
    query = QueryInput(torch.ones(3,14,14),torch.eye(3))

    def criterion(output,target,config,*,seed_surface_overlap):
        assert target is truth
        seen.append(seed_surface_overlap)
        return output['loss'],{}

    engine = Engine()
    segment = TrajectorySegment(engine,criterion,{})
    result,_,_ = segment([(query,truth),(query,truth)],state,None,2,(.08,.21))
    result.backward()
    assert seen == [.08,.21] and engine.scale.grad.item() == 2.
