import copy,random
import pytest
import torch
from src.training_state import restore_training_state,resume_config_compatible


def test_resume_only_safety_metadata_may_change():
    old=dict(systemic_epe_limit_m=.35,seed=17,steps=18000,split='fixed',optimizer=dict(lr=.001),loss=.2)
    new=dict(old,systemic_epe_limit_m=.50)
    assert resume_config_compatible(old,new)
    assert old['systemic_epe_limit_m']==.35
    for key,value in [('seed',18),('steps',20000),('split','other'),('optimizer',dict(lr=.002)),('loss',.3)]:
        assert not resume_config_compatible(old,dict(new,**{key:value}))


def test_resume_preserves_next_adam_update_and_python_torch_rng():
    random.seed(17);torch.manual_seed(17)
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.8)
    def update(m,o,s):
        x=torch.randn(4,3);y=torch.randn(4,2);r=random.random()
        o.zero_grad();((m(x)-y).square().mean()*r).backward();o.step();s.step()
        return x,y,r
    update(model,opt,sched)
    config=dict(systemic_epe_limit_m=.35,seed=17)
    state=dict(step=3000,config=config,initialization='fresh_official_vggt',trainable={n:p.detach().clone() for n,p in model.named_parameters()},
        optimizer=copy.deepcopy(opt.state_dict()),scheduler=copy.deepcopy(sched.state_dict()),
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=[])
    expected=update(model,opt,sched)
    clone=torch.nn.Linear(3,2);other=torch.optim.AdamW(clone.parameters(),lr=.9)
    other_sched=torch.optim.lr_scheduler.StepLR(other,step_size=2,gamma=.8)
    assert restore_training_state(clone,other,other_sched,state,dict(config,systemic_epe_limit_m=.50))==3000
    actual=update(clone,other,other_sched)
    for a,b in zip(actual[:2],expected[:2]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert actual[2]==expected[2]
    for a,b in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert opt.param_groups[0]['lr']==other.param_groups[0]['lr'] and sched.last_epoch==other_sched.last_epoch
