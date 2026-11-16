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
from src.models.spatial_pose import SpatialBootstrap, LegacySpatialBootstrap, SpatialRefinement
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
        if injection is not None and layer == receive_layer:
            patch = injection(tokens[0, start:].float()).to(tokens.dtype)
            tokens = torch.cat((tokens[:, :start], patch[None]), 1)
        tokens, _, _ = aggregator._process_global_attention(tokens, 1, 1, count, dim, layer, pos=position)
        outputs.append(torch.cat((frame[0], tokens.reshape(1, 1, count, dim)), -1) if layer in cached else None)
    return outputs, start


class VGGTSpatialMemory(nn.Module):
    def __init__(self, source, weights, *, receive_layer=4, revision='v2c', pose_jitter_probability=1.):
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
        if revision in ('step2', 'step2_v2', 'step2_v3'):
            self.native.aggregator.global_blocks[11].requires_grad_(True)
        self.receive_layer = receive_layer
        if revision not in ('v2', 'v2b', 'v2c', 'v2g', 'v2h', 'v2i', 'v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3'):
            raise ValueError('Unknown model revision')
        self.revision = revision
        self.pose_jitter_probability = pose_jitter_probability
        self.writer = TokenGaussianWriter(1024)
        self.fusion = SpatialMomentFusion()
        self.render_reader = RenderReader(metric_depth=revision!='v2', surface_normals=revision in ('step2', 'step2_v2', 'step2_v3'), camera_frame=revision in ('step2_v2', 'step2_v3'))
        self.pe_reader = ParameterPEReader(opacity_occlusion=revision!='v2', camera_frame=revision in ('step2_v2', 'step2_v3'))
        self.initial_pose = SpatialBootstrap(ray_bundle=revision in ('v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3'), anisotropic=revision in ('v2l', 'step2', 'step2_v2', 'step2_v3'), metric_scale=revision in ('step2_v2', 'step2_v3'), planar_safe=revision=='step2_v3') if revision in ('v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3') else LegacySpatialBootstrap(local_rotation=revision!='v2', robust_voting=revision in ('v2g', 'v2h', 'v2i'))
        self.refine_pose = SpatialRefinement(direct_residual=revision in ('v2c', 'v2g', 'v2h', 'v2i', 'v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3'), residual_only=revision in ('v2i', 'v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3'), point_to_plane=revision in ('step2', 'step2_v2', 'step2_v3'), relative_geometry=revision=='step2_v3')

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
        reference = mapping_observations[0][1].c2w if self.revision in ('v2h', 'v2i', 'v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3') else None
        return self.fusion(memory, reference_c2w=reference)

    def forward(self, query: QueryInput, memory: GaussianMemory, *, return_attention=False, read_jitter=None):
        if type(query) is not QueryInput:
            raise TypeError("Query accepts only RGB/K, not supervision or camera pose")
        if read_jitter is not None and (not self.training or self.revision not in ('v2l', 'step2', 'step2_v2', 'step2_v3')):
            raise ValueError('Read jitter is a v2l training augmentation only')
        rgb, K = query.rgb, query.K
        height, width = rgb.shape[-2:]
        hp, wp = height//14, width//14
        shallow = frozen_patch_tokens(self.native.aggregator, rgb)
        Kp = patch_intrinsics(K)
        if self.revision in ('v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3'):
            T0, diagnostic = None, {}
        else:
            T0, diagnostic = self.initial_pose(shallow[0].float(), self.writer.latent(shallow[0].float()), memory, Kp, hp, wp)
        if self.training and self.revision not in ('v2', 'v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3') and torch.rand((), device=rgb.device) < self.pose_jitter_probability:
            # Predicted-pose denoising: stochastic small *local* perturbations,
            # independent of GT, teach the residual head a stationary signal.
            from src.geometry import quat_wxyz_to_matrix
            jitter = torch.eye(4, device=rgb.device)
            jitter[:3, :3] = quat_wxyz_to_matrix(torch.cat((rgb.new_ones(1), torch.randn(3, device=rgb.device)*.015)))
            jitter[:3, 3] = torch.randn(3, device=rgb.device)*.08
            T0 = T0 @ jitter
        T0_read = None
        def receive(tokens):
            nonlocal T0, T0_read
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                if self.revision in ('v2j', 'v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3'):
                    T0, bootstrap = self.initial_pose(tokens.float(), self.writer.latent(shallow[0].float()),
                                                     memory, Kp, hp, wp, shallow=shallow[0].float())
                    diagnostic.update(bootstrap)
                T0_read = T0 if read_jitter is None else T0.detach() @ read_jitter
                read, rendered = self.render_reader(tokens.float(), memory, T0_read, Kp, hp, wp)
                read, attention = self.pe_reader(read, memory, T0_read, Kp, hp, wp, rendered.depth,
                                                 rendered=rendered, return_attention=return_attention)
            diagnostic.update(rendered=rendered, attention=attention)
            return read
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=rgb.is_cuda):
            outputs, start = aggregate_single(self.native.aggregator, rgb, shallow,
                                               receive, self.receive_layer)
        # The native single-frame DPT output is interpreted in query-camera
        # coordinates, transformed with the PREDICTED pose inside the model.
        # Loss and evaluation use the resulting world points directly.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            dpt_outputs = [None if value is None else value.float() for value in outputs]
            camera_points, _ = self.native.point_head(dpt_outputs, images=rgb[None, None], patch_start_idx=start)
            camera_points = camera_points[0, 0]
            raw_camera_points = camera_points
            local_confidence = camera_points.new_zeros(hp, wp, 1)
            if self.revision not in ('v2', 'step2_v2', 'step2_v3'):
                camera_points, local_confidence = self.render_reader.anchor_camera_geometry(camera_points, diagnostic['rendered'], K)
            refinement_points = raw_camera_points
            if self.revision == 'step2_v3':
                # Treat the read-pose scale calibration as a pose observation,
                # never let downstream geometry loss optimize its query pose.
                from src.readout.render import RenderedMemory
                observed = diagnostic['rendered']
                scale_render = RenderedMemory(*(None if getattr(observed, key) is None else getattr(observed, key).detach()
                    for key in ('world', 'features', 'depth', 'depth_variance', 'opacity', 'normal')))
                refinement_points, gamma = self.render_reader.calibrate_camera_scale(raw_camera_points, scale_render)
                diagnostic.update(camera_scale=gamma, scaled_camera_points=refinement_points)
            final = outputs[-1][0, 0, start:].float()
            T1, X_T0, residual = self.refine_pose(final, refinement_points if self.revision in ('v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3') else camera_points,
                                                T0_read, diagnostic['rendered'], Kp, hp, wp)
            if self.revision in ('step2_v2', 'step2_v3'):
                final_pose = T1.detach() if self.revision == 'step2_v3' else T1
                rendered_T1 = self.render_reader.render(memory, final_pose, Kp, hp, wp)
                camera_points, local_confidence = self.render_reader.anchor_camera_geometry(refinement_points, rendered_T1, K)
                diagnostic['rendered_T1'] = rendered_T1
            points = transform_points(camera_points, T1.detach() if self.revision == 'step2_v3' else T1)
            coarse_points = transform_points(raw_camera_points if self.revision in ('v2l', 'step2', 'step2_v2', 'step2_v3') else camera_points, T0)
        raw_geometry = {'raw_camera_points': raw_camera_points} if self.revision in ('v2k', 'v2l', 'step2', 'step2_v2', 'step2_v3') else {}
        if self.revision in ('v2l', 'step2', 'step2_v2', 'step2_v3'):
            raw_geometry.update(T0_read=T0_read, read_jitter_applied=read_jitter is not None)
        return {"world_points": points, "coarse_world_points": coarse_points, "camera_points": camera_points, **raw_geometry,
                "X_T0_patch": X_T0, "alignment_residual": residual, "T0": T0, "T1": T1,
                "anchor_support_means": memory.means, "anchor_support_covars": memory.covars,
                "local_geometry_confidence": local_confidence, **diagnostic}
