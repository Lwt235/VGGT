"""No-Memory cannot receive a map or query GT; depth loss cannot bias pose."""
from types import SimpleNamespace
import inspect
import torch
from torch import nn
import pytest
from src.data import QueryInput,Supervision
from src.models.nomemory_vggt import VGGTNoMemory
from src.models.pose import PoseHead


def test_single_frame_baseline_interface_and_pose_depth_gradient_separation(monkeypatch):
    import src.models.nomemory_vggt as adapter
    model=VGGTNoMemory.__new__(VGGTNoMemory);nn.Module.__init__(model)
    raw=nn.Parameter(torch.randn(28,42,3));raw.data[...,2]=2
    class Head(nn.Module):
        def forward(self,*args,**kwargs):return raw[None,None],None
    model.native=nn.Module();model.native.aggregator=SimpleNamespace(patch_embed=nn.Identity());model.native.point_head=Head()
    model.pose=PoseHead(4100)
    monkeypatch.setattr(adapter,'frozen_patch_tokens',lambda *args:torch.zeros(1,6,1024))
    monkeypatch.setattr(adapter,'aggregate_single',lambda *args:([torch.randn(1,1,7,2048)],1))
    q=QueryInput(torch.rand(3,28,42),torch.eye(3))
    out=model(q);out['T1'].retain_grad()
    out['world_points'].square().mean().backward(retain_graph=True)
    assert out['T1'].grad is None and raw.grad.abs().sum()>0
    assert all(p.grad is None for p in model.pose.parameters())
    (out['T1'][:3,3]-torch.tensor([.1,0.,0.])).square().mean().backward()
    assert model.pose.net[-1].weight.grad.abs().sum()>0
    assert list(inspect.signature(VGGTNoMemory.forward).parameters)==['self','query']
    assert set(model._modules)=={'native','pose'}
    with pytest.raises(TypeError):model(q,object())
    gt=Supervision(torch.ones(28,42),torch.eye(4),torch.zeros(28,42,3),torch.ones(28,42,dtype=torch.bool))
    with pytest.raises(TypeError,match='RGB|QueryInput'):model(gt)
