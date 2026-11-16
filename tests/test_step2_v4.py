"""Local metric anchoring, context-only map state and near-zero denoising."""
from types import SimpleNamespace
from dataclasses import fields
import torch
from torch import nn
from src.readout.render import RenderReader, RenderedMemory
from src.models.spatial_pose import SpatialRefinement
from src.models.ellipsoid_pose import sample_read_jitter
from src.memory.gaussians import GaussianMemory, TokenGaussianWriter
from src.data import QueryInput, Supervision


def test_patch_local_anchor_has_no_cross_patch_scale_leak_and_empty_identity():
    torch.manual_seed(17)
    reader=RenderReader(8,16,camera_frame=True,local_multiplicative=True)
    raw=torch.randn(28,56,3);raw[...,2]=2.
    depth=torch.tensor([[[2.],[3.],[6.],[7.]],[[2.],[3.],[6.],[7.]]],requires_grad=True)
    render=RenderedMemory(torch.zeros(2,4,3),torch.randn(2,4,8),depth,torch.zeros(2,4,1),torch.ones(2,4,1))
    full,confidence=reader.anchor_camera_geometry(raw,render,torch.eye(3))
    opacity=render.opacity.clone();opacity[:,2:]=0
    partial=RenderedMemory(render.world,render.features,depth,render.depth_variance,opacity)
    after,_=reader.anchor_camera_geometry(raw,partial,torch.eye(3))
    torch.testing.assert_close(full[:,:28],after[:,:28],atol=0,rtol=0)
    torch.testing.assert_close(after[:,28:],raw[:,28:],atol=0,rtol=0)
    assert confidence.mean()>.7 and full[:,42:,2].mean()>5.
    # DPT pixel edges inside a patch retain their multiplicative contrast.
    raw[0,0,2]=1.;scaled,_=reader.anchor_camera_geometry(raw,render,torch.eye(3))
    torch.testing.assert_close(scaled[0,1,2]/scaled[0,0,2],torch.tensor(2.))
    scaled.sum().backward();assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum()>0
    partial.opacity.zero_();empty,_=reader.anchor_camera_geometry(raw,partial,torch.eye(3))
    torch.testing.assert_close(empty,raw,atol=0,rtol=0)


def test_symmetric_map_context_is_no_grad_and_only_gaussian_fields_escape(monkeypatch):
    import src.models.vggt_memory as adapter
    model=adapter.VGGTSpatialMemory.__new__(adapter.VGGTSpatialMemory);nn.Module.__init__(model)
    model.revision='step2_v4';model.receive_layer=4
    model.native=nn.Module();model.native.aggregator=nn.Identity()
    model.writer=TokenGaussianWriter(32,8);model.fusion=nn.Identity()
    shallow=torch.randn(1,6,16,requires_grad=True)
    context=torch.randn(6,16,requires_grad=True)
    monkeypatch.setattr(adapter,'frozen_patch_tokens',lambda *args:shallow.detach())
    def aggregate(*args,**kwargs):
        assert not torch.is_grad_enabled() and kwargs['stop_before_injection']
        return context.detach()
    monkeypatch.setattr(adapter,'aggregate_single',aggregate)
    # Identity wrapper accepts the reference camera but adds no stored state.
    class Fusion(nn.Module):
        def forward(self,memory,reference_c2w):return memory
    model.fusion=Fusion()
    q=QueryInput(torch.rand(3,28,42),torch.eye(3))
    t=Supervision(torch.ones(28,42)*2,torch.eye(4),torch.zeros(28,42,3),torch.ones(28,42,dtype=torch.bool))
    memory=model.build_memory([(q,t)])
    assert {f.name for f in fields(memory)}=={'means','covars','alpha','features'}
    assert set(memory.__dict__)=={'means','covars','alpha','features'}
    memory.features.sum().backward()
    assert context.grad is None and shallow.grad is None
    assert model.writer.latent.weight.grad.abs().sum()>0
    torch.testing.assert_close(model.writer.latent.weight.grad[0],torch.cat((shallow[0],context),-1).sum(0))


def test_refinement_gate_is_small_and_uncovered_or_zero_residual_is_identity():
    torch.manual_seed(17)
    h,w=4,5;model=SpatialRefinement(16,residual_only=True,point_to_plane=True,relative_geometry=True,confidence_gate=True)
    nn.init.normal_(model.residual_pose.weight,std=.02)
    points=torch.randn(h,w,3);points[...,2]=3
    normal=torch.zeros_like(points);normal[...,2]=1
    r=RenderedMemory(points.clone(),torch.zeros(h,w,8),torch.ones(h,w,1)*3,torch.zeros(h,w,1),torch.ones(h,w,1),normal)
    tokens=torch.randn(h*w,16);T=torch.eye(4)
    exact,_,_=model(tokens,points,T,r,torch.eye(3),h,w)
    torch.testing.assert_close(exact,T,atol=1e-6,rtol=1e-6)
    r.world=r.world+torch.tensor([0.,0.,.1])
    changed,_,_=model(tokens,points,T,r,torch.eye(3),h,w)
    assert (changed-T).abs().max()>1e-5
    changed[:3,3].sum().backward();assert model.confidence_gate.bias.grad.abs()>0
    assert model.confidence_gate.bias.sigmoid()<.15
    r.opacity.zero_();empty,_,_=model(tokens,points,T,r,torch.eye(3),h,w)
    torch.testing.assert_close(empty,T,atol=0,rtol=0)


def test_continuous_read_jitter_covers_near_zero_and_respects_bounds():
    from src.losses.world import pose_errors
    torch.manual_seed(17)
    values=[pose_errors(sample_read_jitter('cpu',include_zero=True),torch.eye(4)) for _ in range(100)]
    rotation=torch.stack([r for r,t in values]);translation=torch.stack([t for r,t in values])
    assert rotation.min()<.2 and rotation.max()<=3.001
    assert translation.min()<.003 and translation.max()<=.060001
