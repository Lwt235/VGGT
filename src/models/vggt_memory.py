"""Actual official VGGT/DINOv2 + DPT with token spatial memory.

Mapping uses explicitly permitted RGB-D and mapping poses. Query forward has
only QueryInput(rgb,K) and GaussianMemory; it cannot accept GT pose/depth.
Native frozen layers remain differentiable to the trainable reception layer.
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.data import QueryInput
from src.geometry import patch_intrinsics, transform_points
from src.memory.gaussians import GaussianMemory, SpatialMomentFusion, TokenGaussianWriter
from src.models.pose import PoseHead
from src.readout.parameter_pe import ParameterPEReader
from src.readout.render import RenderReader
from src.runtime import configure_local_storage, load_checkpoint_readonly


def official_vggt_class(source):
    configure_local_storage()
    if not Path(source, "vggt/models/vggt.py").is_file():
        raise FileNotFoundError("Official VGGT source is unavailable")
    if source not in sys.path:
        sys.path.insert(0, source)
    from vggt.models.vggt import VGGT
    return VGGT


def frozen_patch_tokens(aggregator, rgb):
    image = rgb[None] if rgb.ndim == 3 else rgb
    normalized = (image-aggregator._resnet_mean[0])/aggregator._resnet_std[0]
    with torch.no_grad(), torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=rgb.is_cuda):
        encoded = aggregator.patch_embed(normalized)
    return encoded["x_norm_patchtokens"] if isinstance(encoded, dict) else encoded


def aggregate_single(aggregator, rgb, patch_tokens, injection=None, receive_layer=4):
    """One-frame native block execution; injection outside checkpoint closures.

    Avoids temporary hooks, whose removal before checkpoint recomputation would
    silently change backward behavior. No frame/time KV is carried across calls.
    """
    height, width = rgb.shape[-2:]
    if aggregator.aa_block_size != 1 or aggregator.aa_order != ["frame", "global"]:
        raise ValueError("Adapter supports the registered official alternating-block layout")
    special = torch.cat((aggregator.camera_token[:, 0], aggregator.register_token[:, 0]), 1)
    tokens = torch.cat((special, patch_tokens), 1)
    start = aggregator.patch_start_idx
    count, dim = tokens.shape[1:]
    position = None
    if aggregator.rope is not None:
        patch_pos = aggregator.position_getter(1, height//14, width//14, device=rgb.device) + 1
        position = torch.cat((patch_pos.new_zeros(1, start, 2), patch_pos), 1)
    outputs = []
    cached = getattr(aggregator, "cached_layer_indices", set(range(aggregator.depth)))
    for layer in range(aggregator.depth):
        tokens, _, frame = aggregator._process_frame_attention(tokens, 1, 1, count, dim, layer, pos=position)
        tokens, _, _ = aggregator._process_global_attention(tokens, 1, 1, count, dim, layer, pos=position)
        if injection is not None and layer == receive_layer:
            patch = injection(tokens[0, start:].float()).to(tokens.dtype)
            tokens = torch.cat((tokens[:, :start], patch[None]), 1)
        outputs.append(torch.cat((frame[0], tokens.reshape(1, 1, count, dim)), -1) if layer in cached else None)
    return outputs, start


class VGGTSpatialMemory(nn.Module):
    def __init__(self, source, weights, *, receive_layer=4):
        super().__init__()
        # No random fallback or automatic download: missing weights stop here.
        if weights is None or not Path(weights).is_file():
            raise FileNotFoundError("Actual official VGGT checkpoint required")
        NativeVGGT = official_vggt_class(source)
        self.native = NativeVGGT(enable_camera=False, enable_depth=False, enable_track=False)
        checkpoint = load_checkpoint_readonly(weights)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        expected = self.native.state_dict()
        selected = {key: value for key, value in checkpoint.items() if key in expected}
        if not any(key.startswith("aggregator.patch_embed.") for key in selected):
            raise ValueError("Official DINOv2 parameters missing from complete checkpoint")
        self.native.load_state_dict(selected, strict=True)
        del checkpoint, selected, expected
        self.native.requires_grad_(False)
        self.native.point_head.requires_grad_(True)
        self.native.aggregator.global_blocks[receive_layer].requires_grad_(True)
        self.receive_layer = receive_layer
        self.writer = TokenGaussianWriter(1024)
        self.fusion = SpatialMomentFusion()
        self.render_reader = RenderReader()
        self.pe_reader = ParameterPEReader()
        self.initial_pose = PoseHead(1024+259)
        self.refine_pose = PoseHead(2048+259)

    def train(self, mode=True):
        super().train(mode)
        self.native.aggregator.patch_embed.eval()
        return self

    def build_memory(self, mapping_observations):
        memories = []
        batch_tokens = frozen_patch_tokens(self.native.aggregator, torch.stack([x.rgb for x, _ in mapping_observations])).float()
        for item_idx, (inputs, mapping_supervision) in enumerate(mapping_observations):
            rgb, depth = inputs.rgb, mapping_supervision.depth
            if rgb.shape[-2] % 14 or rgb.shape[-1] % 14:
                raise ValueError("Image size must be a multiple of 14")
            tokens = batch_tokens[item_idx]
            valid = mapping_supervision.valid.float()[None, None]
            fraction = F.avg_pool2d(valid, 14, 14)[0, 0]
            depth_patch = F.avg_pool2d((depth*valid[0, 0])[None, None], 14, 14)[0, 0]/fraction.clamp_min(1e-6)
            memories.append(self.writer(tokens, depth_patch, patch_intrinsics(inputs.K),
                                         mapping_supervision.c2w, valid_mask=fraction > .25))
        if not memories:
            raise ValueError("Mapping requires at least one historical observation")
        memory = GaussianMemory(*(torch.cat([getattr(m, key) for m in memories])
                                  for key in ("means", "covars", "alpha", "features")))
        # No image/token fields leave this function. Autograd retains only what
        # the current training step requires, never a subsequent query bypass.
        return self.fusion(memory)

    def forward(self, query: QueryInput, memory: GaussianMemory, *, return_attention=False):
        if type(query) is not QueryInput:
            raise TypeError("Query accepts only RGB/K, not supervision or camera pose")
        rgb, K = query.rgb, query.K
        height, width = rgb.shape[-2:]
        hp, wp = height//14, width//14
        shallow = frozen_patch_tokens(self.native.aggregator, rgb)
        if len(memory):
            weights = memory.alpha/memory.alpha.sum().clamp_min(1e-6)
            summary = torch.cat(((weights[:, None]*memory.means).sum(0),
                                 (weights[:, None]*memory.features).sum(0)))
        else:
            summary = rgb.new_zeros(259)
        T0 = self.initial_pose(torch.cat((shallow[0].float().mean(0), summary)))
        Kp = patch_intrinsics(K)
        diagnostic = {}
        def receive(tokens):
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                read, rendered = self.render_reader(tokens.float(), memory, T0, Kp, hp, wp)
                read, attention = self.pe_reader(read, memory, T0, Kp, hp, wp, rendered.depth,
                                                 return_attention=return_attention)
            diagnostic.update(rendered=rendered, attention=attention)
            return read
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=rgb.is_cuda):
            outputs, start = aggregate_single(self.native.aggregator, rgb, shallow,
                                               receive, self.receive_layer)
        final = outputs[-1][0, 0, start:].float().mean(0)
        T1 = T0 @ self.refine_pose(torch.cat((final, summary)))
        # The native single-frame DPT output is interpreted in query-camera
        # coordinates, transformed with the PREDICTED pose inside the model.
        # Loss and evaluation use the resulting world points directly.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            dpt_outputs = [None if value is None else value.float() for value in outputs]
            camera_points, _ = self.native.point_head(dpt_outputs, images=rgb[None, None], patch_start_idx=start)
            points = transform_points(camera_points[0, 0], T1)
        return {"world_points": points, "T0": T0, "T1": T1, **diagnostic}
