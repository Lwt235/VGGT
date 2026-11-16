"""The final one-frame control uses the same writer, with no fusion stage."""
from types import SimpleNamespace
from dataclasses import fields
import torch
from src.data import QueryInput,Supervision
from src.memory.gaussians import TokenGaussianWriter


def test_single_map_control_has_one_token_per_valid_patch_without_fusion(monkeypatch):
    import scripts.diagnose_step2_final as script
    tokens=torch.randn(1,2,16);context=torch.randn(2,16)
    monkeypatch.setattr(script,'frozen_patch_tokens',lambda *a:tokens)
    def aggregate(*a,**kw):
        assert not torch.is_grad_enabled() and kw['stop_before_injection']
        return context
    monkeypatch.setattr(script,'aggregate_single',aggregate)
    def forbidden(*a,**kw):raise AssertionError('One-map control must not fuse or read more mapping views')
    model=SimpleNamespace(native=SimpleNamespace(aggregator=object()),receive_layer=4,
        writer=TokenGaussianWriter(32,8),fusion=forbidden,build_memory=forbidden)
    q=QueryInput(torch.zeros(3,14,28),torch.tensor([[14.,0.,14.],[0.,14.,7.],[0.,0.,1.]]))
    t=Supervision(torch.ones(14,28)*2,torch.eye(4),torch.zeros(14,28,3),torch.ones(14,28,dtype=torch.bool))
    m=script.single_frame_unfused(model,(q,t))
    assert len(m)==2 and not m.features.requires_grad
    assert {f.name for f in fields(m)}=={'means','covars','alpha','features'}
    torch.testing.assert_close(m.features,model.writer.latent(torch.cat((tokens[0],context),-1)))
