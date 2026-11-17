"""Leak-free streaming API and finite multi-step prediction-only writeback."""
from dataclasses import fields
import inspect
import torch
from torch import nn
import pytest
from src.data import QueryInput
from src.memory.gaussians import GaussianMemory, TokenGaussianWriter, SpatialMomentFusion
from src.streaming import StreamingEngine, StreamingState
from src.models.stream_pose import StreamPoseGate
from src.geometry import transform_points
from src.models.ellipsoid_pose import local_se3_exp


def empty():
    return GaussianMemory(torch.empty(0, 3), torch.empty(0, 3, 3), torch.empty(0), torch.empty(0, 4))


class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.writer = TokenGaussianWriter(8, 4)
        self.fusion = SpatialMomentFusion()
        self.predicted_depth = nn.Parameter(torch.tensor(2.))
        self.read_sizes = []

    def forward(self, query, memory, **kwargs):
        self.read_sizes.append(len(memory))
        camera = self.predicted_depth.expand(28, 42, 3).clone()
        camera = torch.cat((camera[..., :2]*0, camera[..., 2:]), -1)
        T = torch.eye(4)
        T[0, 3] = query.rgb.mean()*10
        return dict(camera_points=camera, T1=T, patch_hw=(2, 3),
            stream_writer_tokens=torch.ones(6, 8), dustbin_probability=torch.zeros(6),
            c_depth=torch.ones(6), attention={'indices': torch.arange(len(memory))})


def query(color=0):
    return QueryInput(torch.ones(3, 28, 42)*color,
                      torch.tensor([[40., 0., 20.], [0., 40., 14.], [0., 0., 1.]]))


def test_streaming_query_contract_read_before_write_no_history():
    assert list(inspect.signature(StreamingEngine.forward).parameters) == ['self', 'query', 'state', 'return_attention']
    model = PredictionModel()
    engine = StreamingEngine(model)
    state = StreamingState(empty(), torch.empty(0), 0)
    output, next_state = engine(query(), state)
    assert model.read_sizes == [0] and len(state.memory) == 0 and len(next_state.memory) > 0
    assert 'stream_writer_tokens' not in output
    assert [f.name for f in fields(next_state)] == ['memory', 'support', 'step', 'window_entries']
    assert next_state.window_entries == ()
    assert [f.name for f in fields(next_state.memory)] == ['means', 'covars', 'alpha', 'features']
    assert next_state.support.ndim == 1 and len(next_state.support) == len(next_state.memory)
    with pytest.raises(TypeError):
        engine(query(), state, target=torch.zeros(1))
    with pytest.raises(TypeError):
        engine((query(), torch.zeros(1)), state)


def test_multi_step_writer_geometry_gate_gradients_and_no_state_mutation():
    torch.manual_seed(17)
    engine = StreamingEngine(PredictionModel())
    state = StreamingState(empty(), torch.empty(0), 0)
    _, first = engine(query(), state)
    original = first.memory.means.clone()
    _, last = engine(query(), first)
    loss = last.memory.means.square().mean()+last.memory.features.square().mean()+last.support.sum()*.01
    loss.backward()
    for parameter in (engine.model.predicted_depth, engine.model.writer.geometry.weight,
                      engine.write_gate.modulation[-1].bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    torch.testing.assert_close(first.memory.means, original, atol=0, rtol=0)


def test_local_window_evicts_old_three_dimensional_writes():
    engine = StreamingEngine(PredictionModel(), window=2)
    state = StreamingState(empty(), torch.empty(0), 0)
    for color in (0., 1., 2., 3.):
        _, state = engine(query(color), state)
    assert [e.step for e in state.window_entries] == [3, 4]
    assert state.memory.means[:, 0].min() > 18
    assert all([f.name for f in fields(e)] == ['memory', 'support', 'step'] for e in state.window_entries)


def test_write_rejects_blind_frame_but_can_expand_from_supported_pose():
    engine = StreamingEngine(PredictionModel())
    depth = torch.ones(2, 3)*3
    data = dict(dustbin_probability=torch.ones(6)*.999, c_depth=torch.ones(6))
    blind, _ = engine.write_gate(depth, data)
    assert blind.max() < engine.min_write_confidence
    data['dustbin_probability'] = torch.tensor([0., 0., 0., 0., 1., 1.])
    expansion, _ = engine.write_gate(depth, data)
    assert expansion[1, 1:].min() > engine.min_write_confidence


def test_conservative_pose_gate_no_support_identity_and_world_equivariance():
    gate = StreamPoseGate()
    base = torch.eye(4)
    proposed = torch.eye(4)
    proposed[:3, :3], _ = local_se3_exp(torch.tensor([.03, .02, .01]), torch.zeros(3))
    proposed[:3, 3] = torch.tensor([.08, -.03, .02])
    diagnostic = dict(dustbin_probability=torch.ones(6), matching_attention=torch.ones(6, 1),
                      refinement_lm_energy_history=torch.tensor([3., 2., 1.]))
    result, gain = gate(base, proposed, diagnostic)
    torch.testing.assert_close(result, base, atol=0, rtol=0)
    assert gain == 0
    diagnostic['dustbin_probability'] = torch.zeros(6)
    diagnostic['matching_attention'] = torch.tensor([[1., 0.]]).expand(6, -1)
    result, gain = gate(base, proposed, diagnostic)
    assert 0 < gain < .2 and result[:3, 3].norm() < proposed[:3, 3].norm()
    G = torch.eye(4)
    G[:3, :3], _ = local_se3_exp(torch.tensor([.3, -.7, .1]), torch.zeros(3))
    G[:3, 3] = torch.tensor([1., -2., .5])
    shifted, _ = gate(G@base, G@proposed, diagnostic)
    torch.testing.assert_close(shifted, G@result, atol=2e-6, rtol=2e-6)
    (result[:3, 3].square().sum()).backward()
    assert gate.confidence.bias.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA mixed-device gate test')
def test_stream_gate_accepts_cpu_double_lm_diagnostic_on_cuda():
    gate = StreamPoseGate().cuda()
    base = torch.eye(4, device='cuda')
    proposed = base.clone(); proposed[0, 3] = .04
    diagnostic = dict(dustbin_probability=torch.zeros(6, device='cuda'),
        matching_attention=torch.tensor([[1., 0.]], device='cuda').expand(6, -1),
        refinement_lm_energy_history=torch.tensor([3., 2., 1.], dtype=torch.float64, device='cpu'))
    pose, gain = gate(base, proposed, diagnostic)
    assert pose.is_cuda and gain.is_cuda and 0 < gain < 1
    pose[:3, 3].square().sum().backward()
    assert torch.isfinite(gate.confidence.weight.grad).all()


def test_frozen_native_direct_execution_preserves_child_modes_rng_and_gradients():
    from torch.utils.checkpoint import checkpoint
    class Aggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.block=nn.Sequential(nn.Linear(8,8),nn.Dropout(.2),nn.GELU())
        def forward(self,value):
            return checkpoint(self.block,value,use_reentrant=False) if self.training else self.block(value)
    model=PredictionModel()
    model.native=nn.Module();model.native.aggregator=Aggregator()
    model.native.requires_grad_(False)
    engine=StreamingEngine(model)
    engine.train()
    aggregate=model.native.aggregator
    assert not aggregate.training and aggregate.block.training and aggregate.block[1].training
    data=torch.randn(6,8,requires_grad=True)
    torch.manual_seed(17)
    direct=aggregate(data);direct.square().sum().backward()
    direct_grad=data.grad.clone();direct_rng=torch.get_rng_state()
    aggregate.training=True
    duplicate=data.detach().clone().requires_grad_()
    torch.manual_seed(17)
    recomputed=aggregate(duplicate);recomputed.square().sum().backward()
    torch.testing.assert_close(direct,recomputed,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(direct_grad,duplicate.grad,atol=1e-6,rtol=1e-6)
    assert torch.equal(direct_rng,torch.get_rng_state())
    engine.eval()
    assert not aggregate.training and not aggregate.block.training


def test_trainable_native_keeps_original_checkpoint_training_mode():
    model=PredictionModel()
    model.native=nn.Module();model.native.aggregator=nn.Sequential(nn.Linear(2,2))
    engine=StreamingEngine(model).train()
    assert model.native.aggregator.training
