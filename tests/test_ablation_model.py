"""CPU adapter scheduling checks with a small, explicit official-VGGT double.

The actual adapter aggregation, Gaussian heads, readers and world criterion run.
Only official weights/DINO/DPT and pose solvers are replaced: these tests check
which read paths execute, while geometry/gsplat have separate strict tests.
"""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.data import QueryInput, Supervision
from src.geometry import unproject, world_rays
from src.losses.world import world_loss
from src.memory.gaussians import GaussianMemory
from src.readout.parameter_pe import ParameterPEReader
from src.readout.render import RenderReader
import src.models.vggt_memory as adapter


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.001))

    def forward(self, tokens):
        return tokens+self.offset


class TinyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.))

    def forward(self, images):
        patch = F.avg_pool2d(images.mean(1, keepdim=True), 14, 14).flatten(2).transpose(1, 2)
        channels = torch.linspace(-.2, .2, 1024, device=images.device)
        return {'x_norm_patchtokens': self.gain*(patch+channels)}


class TinyAggregator(nn.Module):
    """Matches the official alternating attention interfaces used by adapter."""
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self.global_blocks = nn.ModuleList([TinyBlock() for _ in range(12)])
        self.camera_token = nn.Parameter(torch.zeros(1, 1, 1, 1024))
        self.register_token = nn.Parameter(torch.zeros(1, 1, 1, 1024))
        self.register_buffer('_resnet_mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('_resnet_std', torch.ones(1, 3, 1, 1))
        self.aa_block_size, self.aa_order = 1, ['frame', 'global']
        self.patch_start_idx, self.depth, self.rope = 2, 12, None
        self.cached_layer_indices = {11}

    def _process_frame_attention(self, tokens, batch, frames, count, dim, layer, pos=None):
        return tokens, None, [tokens.reshape(batch, frames, count, dim)]

    def _process_global_attention(self, tokens, batch, frames, count, dim, layer, pos=None):
        return self.global_blocks[layer](tokens), None, None


class TinyPointHead(nn.Module):
    """DPT interface double; retains gradient through the received tokens."""
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([.02, -.03, 2.6]))

    def forward(self, outputs, *, images, patch_start_idx):
        height, width = images.shape[-2:]
        tokens = outputs[-1][0, 0, patch_start_idx:]
        grid = tokens[:, :3].T.reshape(1, 3, height//14, width//14)
        camera = F.interpolate(grid, size=(height, width), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)*.01+self.offset
        return camera[None, None], torch.ones(1, 1, height, width)


class TinyOfficialVGGT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.aggregator = TinyAggregator()
        self.point_head = TinyPointHead()


@pytest.fixture
def make_model(monkeypatch, tmp_path):
    checkpoint = tmp_path/'official-double.pt'
    checkpoint.write_bytes(b'Explicit official checkpoint test double')
    monkeypatch.setattr(adapter, 'official_vggt_class', lambda source: TinyOfficialVGGT)
    monkeypatch.setattr(adapter, 'load_checkpoint_readonly', lambda path: TinyOfficialVGGT().state_dict())

    def construct(ablation=None, *, explicit_full=False):
        kwargs = dict(revision='step2_v5', pose_jitter_probability=0.)
        if ablation is not None or explicit_full:
            kwargs['ablation'] = ablation
        model = adapter.VGGTSpatialMemory('explicit-tiny-official-double', checkpoint, **kwargs)
        model.eval()
        return model
    return construct


def inputs():
    rgb = torch.linspace(0., 1., 3*28*28).reshape(3, 28, 28)
    K = torch.tensor([[28., 0., 14.], [0., 28., 14.], [0., 0., 1.]])
    query = QueryInput(rgb, K)
    depth = torch.full((28, 28), 3.)
    target = Supervision(depth, torch.eye(4), unproject(depth, K, torch.eye(4), center=0.), torch.ones(28, 28, dtype=torch.bool))
    return query, target


def stub_pose_solvers(monkeypatch, model):
    """Keep matching and supervision interfaces while excluding LM numerics."""
    def bootstrap(tokens, query_features, memory, K, h, w, *, shallow):
        q = F.normalize(query_features, dim=-1)
        pose = torch.eye(4)+q.sum()*0
        _, rays, dz = world_rays(K, pose, h, w)
        logits = q @ F.normalize(memory.features, dim=-1).T/math.sqrt(q.shape[-1])
        W = torch.cat((logits, q[:, :1]*.01-2.), -1).softmax(-1)
        p = 1-W[:, -1]
        inlier = W[:, :-1]/p[:, None].clamp_min(1e-6)
        z = 2.5+.01*q[:, 0]
        confidence = q[:, 1].sigmoid()
        anchors = inlier @ memory.means
        points = rays/dz[:, None]*z[:, None]
        return pose, dict(anchors=anchors, confidence=p, matching_attention=W,
            bootstrap_attention=inlier, bootstrap_support=torch.ones(len(memory), dtype=torch.bool),
            dustbin_probability=W[:, -1], match_mass=p.sum(), z_cam=z, c_depth=confidence,
            camera_patch_points=points, camera_rays=rays)

    def refine(final, memory, diagnostic, init_pose):
        points = diagnostic['camera_patch_points']
        return init_pose.detach(), points, points*0, {}
    monkeypatch.setattr(model.initial_pose, 'forward', bootstrap)
    monkeypatch.setattr(model.refine_pose, 'forward', refine)


def empty_memory():
    return GaussianMemory(torch.empty(0, 3), torch.empty(0, 3, 3), torch.empty(0), torch.empty(0, 256))


def forbidden(*args, **kwargs):
    raise AssertionError('A closed reader path was called')


def test_render_only_never_invokes_parameter_pe_reader(make_model, monkeypatch):
    model = make_model('render_only')
    assert isinstance(model.render_reader, RenderReader) and model.pe_reader is None
    stub_pose_solvers(monkeypatch, model)
    monkeypatch.setattr(ParameterPEReader, 'forward', forbidden)
    calls = []
    original = RenderReader.render
    def render(self, *args, **kwargs):
        calls.append('o5')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(RenderReader, 'render', render)
    output = model(inputs()[0], empty_memory(), return_attention=True)
    assert calls == ['o5', 'o5']  # receive at T0, final anchoring at T1
    assert output['o5_enabled'] and not output['o6_enabled']
    assert output['attention']['disabled'] and output['attention']['attention'] is None
    assert output['rendered'] is not None and 'rendered_T1' in output


def test_pe_only_no_render_or_anchor_raw_camera_identity_and_finite_criterion(make_model, monkeypatch):
    model = make_model('pe_only')
    assert model.render_reader is None and isinstance(model.pe_reader, ParameterPEReader)
    stub_pose_solvers(monkeypatch, model)
    monkeypatch.setattr(RenderReader, 'forward', forbidden)
    monkeypatch.setattr(RenderReader, 'render', forbidden)
    monkeypatch.setattr(RenderReader, 'anchor_camera_geometry', forbidden)
    query, target = inputs()
    memory = model.build_memory([(query, target)])
    assert len(memory) > 0
    # Open the existing zero gate to verify the real PE branch has a gradient.
    with torch.no_grad():
        model.pe_reader.gate.fill_(.1)
    output = model(query, memory, return_attention=True)
    assert not output['o5_enabled'] and output['o6_enabled']
    assert output['rendered'] is None and 'rendered_T1' not in output
    assert output['patch_hw'] == (2, 2)
    assert output['camera_points'] is output['raw_camera_points']
    torch.testing.assert_close(output['camera_points'], output['raw_camera_points'], atol=0, rtol=0)
    assert output['local_geometry_confidence'].count_nonzero() == 0
    assert output['attention']['attention'] is not None
    assert output['attention']['indices'].numel() > 0
    loss, metrics = world_loss(output, target, relative_depth_weight=.1,
        anchored_camera_weight=.2, anchored_relative_depth_weight=.1)
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in metrics.values())
    loss.backward()
    for parameter in (model.native.point_head.offset, model.pe_reader.gate,
                      model.pe_reader.value.weight, model.writer.latent.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_isotropic_flags_reach_every_physical_module(make_model):
    model = make_model('isotropic_point')
    assert model.use_o5 and model.use_o6 and model.isotropic
    for module in (model.writer, model.fusion, model.initial_pose, model.refine_pose,
                   model.render_reader, model.pe_reader):
        assert module.isotropic is True
    assert model.writer.geometry.out_features == 5
    assert not hasattr(model.writer, 'base_shape')
    query, target = inputs()
    memory = model.build_memory([(query, target)])
    torch.testing.assert_close(memory.covars, memory.covars[:, 0, 0, None, None]*torch.eye(3), atol=0, rtol=0)


def test_default_full_matches_explicit_full_construction_and_cpu_forward(make_model, monkeypatch):
    torch.manual_seed(29)
    default = make_model()
    torch.manual_seed(29)
    explicit = make_model(explicit_full=True)
    assert default.use_o5 and default.use_o6 and default.ablation is None
    for module in (default.writer, default.fusion, default.initial_pose, default.refine_pose,
                   default.render_reader, default.pe_reader):
        assert module.isotropic is False
    assert default.writer.geometry.out_features == 11 and hasattr(default.writer, 'base_shape')
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
    stub_pose_solvers(monkeypatch, default)
    stub_pose_solvers(monkeypatch, explicit)
    query, _ = inputs()
    a, b = default(query, empty_memory()), explicit(query, empty_memory())
    assert a['o5_enabled'] and a['o6_enabled'] and a['rendered'] is not None
    for key in ('world_points', 'camera_points', 'raw_camera_points', 'T0', 'T1'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
