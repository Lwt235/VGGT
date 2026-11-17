"""DDP checkpoint compatibility, independent trajectories and unused ranks."""
from datetime import timedelta
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from src.streaming_distributed import canonical_delta, scientific_config, trajectory_rng, trajectory_seed


def test_canonical_ddp_delta_compatible_with_single_card():
    from scripts.train_step3 import load_spatial_delta
    model = nn.Linear(3, 2)
    delta = {'module.'+k: torch.ones_like(v) for k,v in model.named_parameters()}
    load_spatial_delta(model, delta)
    assert all(torch.equal(v, torch.ones_like(v)) for v in model.parameters())
    with pytest.raises(ValueError, match='Duplicate'):
        canonical_delta({'module.weight': 1, 'weight': 2})


def test_resume_exempts_only_parallel_runtime_fields():
    config = dict(steps=3000, optimizer={'lr': .01}, seed=17)
    assert scientific_config(config) == scientific_config(config | {'distributed_runtime': {'world_size': 3}})
    assert scientific_config(config) != scientific_config(config | {'steps': 4000})
    assert scientific_config(config) != scientific_config(config | {'optimizer': {'lr': .02}})


def test_ddp_registers_only_existing_trainable_parameter_references():
    from src.streaming_distributed import TrajectorySegment
    engine = nn.Sequential(nn.Linear(3,3),nn.Linear(3,1))
    engine[0].requires_grad_(False)
    wrapper = TrajectorySegment(engine,lambda *args: None,{})
    expected = [p for p in engine.parameters() if p.requires_grad]
    assert [id(p) for p in wrapper.parameters()] == [id(p) for p in expected]
    assert sum(p.numel() for p in wrapper.parameters()) < sum(p.numel() for p in engine.parameters())
    assert all(n.startswith('spatial_parameters.') for n,_ in wrapper.named_parameters())


def test_resume_preserves_adam_lr_and_per_rank_rng(monkeypatch):
    import copy
    from scripts.train_step3 import restore_optimizer_rng
    engine = nn.Linear(2,1)
    optimizer = torch.optim.AdamW(engine.parameters(),lr=.003)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: .9**step)
    engine(torch.ones(1,2)).square().sum().backward();optimizer.step();scheduler.step()
    rank_rng = [dict(python_rng=random.Random(17+r).getstate(),
        torch_rng=torch.Generator().manual_seed(17+r).get_state(),cuda_rng=torch.tensor([r],dtype=torch.uint8)) for r in range(2)]
    config = dict(seed=17,steps=3000)
    checkpoint = copy.deepcopy(dict(step=2500,config=config,trainable=engine.state_dict(),
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rank_rng=rank_rng))
    monkeypatch.setenv('RANK','1');monkeypatch.setenv('WORLD_SIZE','2')
    captured = []
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state:captured.append(state))
    fresh = nn.Linear(2,1)
    opt = torch.optim.AdamW(fresh.parameters(),lr=.99)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda step: .9**step)
    python_before,torch_before = random.getstate(),torch.get_rng_state()
    try:
        assert restore_optimizer_rng(fresh,opt,sched,checkpoint,config|{'world_size':2}) == 2500
        assert sched.state_dict() == checkpoint['scheduler']
        assert opt.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
        for key,old in optimizer.state_dict()['state'].items():
            for name,value in old.items():
                torch.testing.assert_close(opt.state_dict()['state'][key][name],value,rtol=0,atol=0)
        assert random.getstate() == rank_rng[1]['python_rng']
        assert torch.equal(torch.get_rng_state(),rank_rng[1]['torch_rng'])
        assert torch.equal(captured[0],rank_rng[1]['cuda_rng'])
    finally:
        random.setstate(python_before);torch.set_rng_state(torch_before)


def test_rank_trajectories_independent_reproducible():
    values = [[trajectory_rng(17, rank, step).sample(range(10000), 16) for step in range(2500,2504)] for rank in range(4)]
    assert len({tuple(x) for rows in values for x in rows}) == 16
    assert values[2][1] == random.Random(trajectory_seed(17,2,2501)).sample(range(10000),16)


def test_parallel_resource_selector_requires_two_live_free_snapshots():
    from scripts.run_step2_ablations import GPU, Snapshot
    from scripts.run_step3_parallel import select_gpus
    first = Snapshot(tuple(GPU(i,str(i),100,free,49140-free,49140)
        for i,free in enumerate((30000,28000,19000,35000))))
    second = Snapshot(tuple(GPU(i,str(i),100,free,49140-free,49140)
        for i,free in enumerate((30000,17000,32000,33000))))
    assert [g.index for g in select_gpus(first,second,4,.28)] == [3,0]


class _Conditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(2.))

    def forward(self, rank, segment):
        return self.a.square() + (self.b.square() if (rank+segment)%2 else 0)


def _rank_worker(rank, rendezvous, output):
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous,
        world_size=2, rank=rank, timeout=timedelta(seconds=45))
    model = DDP(_Conditional(), find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for segment in range(3):
            (model(rank,segment)/3).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        torch.testing.assert_close(model.module.a.grad, 2*model.module.a.detach())
        torch.testing.assert_close(model.module.b.grad, model.module.b.detach())
        optimizer.step()
    torch.save(canonical_delta(model.state_dict()), Path(output)/f'{rank}.pt')
    dist.destroy_process_group()


def test_two_rank_unused_branches_no_deadlock_and_equal_updates(tmp_path, monkeypatch):
    monkeypatch.setenv('MKL_THREADING_LAYER', 'GNU')
    mp.spawn(_rank_worker, args=(str(tmp_path/'rendezvous'),str(tmp_path)), nprocs=2, join=True)
    a,b = (torch.load(tmp_path/f'{rank}.pt',weights_only=True) for rank in range(2))
    assert a.keys() == {'a','b'}
    assert all(torch.equal(a[k],b[k]) and torch.isfinite(a[k]).all() for k in a)
