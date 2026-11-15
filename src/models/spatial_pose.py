"""Ray/anchor correspondences and spatial, camera-local residual refinement.

No query pose/depth is accepted. Rotation vectors are learned weighted anchor
differences, followed by continuous 6D Gram-Schmidt, not a rigid fit/SVD.
This construction transports the prediction with a changed memory gauge.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.geometry import invert_se3, transform_points, world_rays
from src.models.pose import PoseHead, rotation_6d, pose_from_vector


def anchor_vote_weights(weights, memory, anchors, confidence):
    # Total posterior spread relative to intrinsic token footprint: widely
    # separated candidate surfaces must not drag the entire camera frame.
    difference = memory.means[None]-anchors[:, None]
    between = (weights*difference.square().sum(-1)).sum(-1)
    within = weights @ memory.covars.diagonal(dim1=-2, dim2=-1).sum(-1)
    ambiguity = between/(within+.01)
    votes = (.05+confidence)/(1+ambiguity).sqrt()
    return votes.clamp_min(.005)


class SpatialBootstrap(nn.Module):
    def __init__(self, token_dim=1024, latent_dim=256, *, local_rotation=True, robust_voting=False):
        super().__init__()
        self.query_residual = nn.Linear(token_dim+3, latent_dim)
        self.key_residual = nn.Linear(latent_dim+4, latent_dim)
        for layer in (self.query_residual, self.key_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(20.)))
        self.key_dropout_mode = 'independent'
        self.robust_voting = robust_voting
        self.patch_features = nn.Sequential(nn.LayerNorm(token_dim+5), nn.Linear(token_dim+5, 64), nn.GELU())
        self.coefficients = nn.Linear(64, 3)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)
        self.translation = nn.Sequential(nn.Flatten(), nn.Linear(64*4*5, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.translation[-1].weight)
        nn.init.constant_(self.translation[-1].bias, 0.)
        with torch.no_grad():
            self.translation[-1].bias[2] = -2.
        self.local_rotation = None
        self.geometry_context = None
        if local_rotation:
            self.geometry_context = nn.Linear(3*4*5, 64*4*5, bias=False)
            nn.init.zeros_(self.geometry_context.weight)
            self.local_rotation = nn.Sequential(nn.Flatten(), nn.Linear(64*4*5, 128), nn.GELU(), nn.Linear(128, 6))
            nn.init.zeros_(self.local_rotation[-1].weight)
            with torch.no_grad():
                self.local_rotation[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, tokens, query_features, memory, K, height, width):
        eye = torch.eye(4, device=tokens.device, dtype=tokens.dtype)
        _, rays, _ = world_rays(K, eye, height, width)
        if not len(memory):
            return eye, dict(anchors=tokens.new_zeros(len(tokens), 3), confidence=tokens.new_zeros(len(tokens)))
        scales = torch.linalg.eigvalsh(memory.covars).clamp_min(1e-10).sqrt().log()
        q = F.normalize(query_features+self.query_residual(torch.cat((tokens, rays), -1)), dim=-1)
        k = F.normalize(memory.features+self.key_residual(torch.cat((memory.features, scales, memory.alpha[:, None]), -1)), dim=-1)
        logits = q @ k.T * self.log_temperature.exp().clamp(max=60.) + memory.alpha.clamp_min(1e-6).log()[None]
        support = torch.ones(len(memory), dtype=torch.bool, device=tokens.device)
        if self.training and self.local_rotation is not None and self.key_dropout_mode != 'none':
            # Global localization must tolerate small missing map subsets.
            # O5/O6 still read the full map; this is bootstrap key dropout.
            support = torch.rand(len(memory), device=tokens.device) > .1
            if self.key_dropout_mode == 'spatial':
                center_index = torch.randint(len(memory), (), device=tokens.device)
                spatial_support = ((memory.means.detach()-memory.means.detach()[center_index]).abs() > .5).any(-1)
                # A local dropout regularizer; never remove most of a map or
                # alter the stored memory / the O5 and O6 read paths.
                if spatial_support.sum() >= len(memory)*.5:
                    support = spatial_support
            support[0] = True
            logits = logits.masked_fill(~support[None], float('-inf'))
        weights = logits.softmax(-1)
        anchors = weights @ memory.means
        confidence = 1.+(weights*weights.clamp_min(1e-9).log()).sum(-1)/math.log(max(2, len(memory)))
        votes = anchor_vote_weights(weights, memory, anchors, confidence) if self.robust_voting else torch.ones_like(confidence)
        votes = votes/votes.sum().clamp_min(1e-7)
        center = (votes[:, None]*anchors).sum(0)
        radius = (anchors-center).norm(dim=-1, keepdim=True)
        features = self.patch_features(torch.cat((tokens, rays, confidence[:, None], radius), -1))
        coefficients = self.coefficients(features)
        # Preserve ray layout; uniform image pooling cannot recover these signed
        # world vectors. Learned corrections can handle oblique/reverse views.
        a = (votes[:, None]*(rays[:, 0]+coefficients[:, 0])[:, None]*(anchors-center)).sum(0)
        b = (votes[:, None]*(rays[:, 1]+coefficients[:, 1])[:, None]*(anchors-center)).sum(0)
        anchor_R = rotation_6d(torch.cat((a, b)))
        spatial = features.T.reshape(1, 64, height, width)
        pooled = F.adaptive_avg_pool2d(spatial, (4, 5))
        if self.geometry_context is not None:
            local_anchors = ((anchors-center) @ anchor_R).T.reshape(1, 3, height, width)
            local_layout = F.adaptive_avg_pool2d(local_anchors, (4, 5)).flatten(1)
            pooled = pooled+self.geometry_context(local_layout).reshape_as(pooled)
        local_translation = self.translation(pooled)[0]
        R = anchor_R if self.local_rotation is None else anchor_R @ rotation_6d(self.local_rotation(pooled)[0])
        centroid_weights = (coefficients[:, 2]+votes.clamp_min(1e-8).log()).softmax(0)
        translation = centroid_weights @ anchors + anchor_R @ local_translation
        T = torch.cat((torch.cat((R, translation[:, None]), -1), eye[3:]), 0)
        return T, dict(anchors=anchors, confidence=confidence, bootstrap_attention=weights, bootstrap_support=support, anchor_votes=votes)


class SpatialRefinement(nn.Module):
    def __init__(self, token_dim=2048, *, direct_residual=True, residual_only=False):
        super().__init__()
        self.token_project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 32), nn.GELU())
        # camera point, rendered point in T0 camera, metric residual, ray, O, var
        self.spatial = nn.Sequential(nn.Conv2d(46, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.pose = None if residual_only else PoseHead(64*4*5)
        self.residual_only = residual_only
        self.residual_modulator = None
        if residual_only:
            self.residual_modulator = nn.Linear(64*4*5, 3*4*5)
            nn.init.zeros_(self.residual_modulator.weight)
            nn.init.zeros_(self.residual_modulator.bias)
        self.residual_pose = None
        if direct_residual:
            # Preserve metric residual amplitude. A LayerNorm over the entire
            # appearance/geometry embedding can suppress a 5 cm displacement.
            # This is a learned spatial linear layer, not a pose fitting solver.
            self.residual_pose = nn.Linear(3*4*5, 9, bias=not residual_only)
            nn.init.zeros_(self.residual_pose.weight)
            if not residual_only:
                with torch.no_grad():
                    self.residual_pose.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.]))

    def forward(self, tokens, camera_points, T0, rendered, K, height, width):
        camera_patch = F.adaptive_avg_pool2d(camera_points.permute(2, 0, 1)[None], (height, width))[0].permute(1, 2, 0).reshape(-1, 3)
        fixed = T0.detach()
        X_T0 = transform_points(camera_patch, fixed)
        opacity = rendered.opacity.reshape(-1, 1)
        residual_world = opacity*(rendered.world.reshape(-1, 3)-X_T0)
        residual_local = residual_world @ fixed[:3, :3]
        rendered_local = transform_points(rendered.world.reshape(-1, 3), invert_se3(fixed))*opacity
        _, rays, _ = world_rays(K, torch.eye(4, device=K.device), height, width)
        geometry = torch.cat((camera_patch, rendered_local, residual_local, rays, opacity,
                              torch.log1p(rendered.depth_variance.reshape(-1, 1))), -1)
        inputs = torch.cat((self.token_project(tokens), geometry), -1).T.reshape(1, 46, height, width)
        spatial = F.adaptive_avg_pool2d(self.spatial(inputs), (4, 5)).flatten()
        delta = torch.eye(4, device=K.device, dtype=K.dtype) if self.pose is None else self.pose(spatial)
        if self.residual_pose is not None:
            pose_residual = residual_local
            if self.residual_only:
                # Mixed depth layers and large correspondence errors should not
                # dominate one global camera correction. Context may reweight
                # these metric residuals, but cannot invent a pose on its own.
                variance_weight = 1/(1+rendered.depth_variance.reshape(-1, 1)/.01)
                robust_weight = .2/torch.sqrt(residual_local.square().sum(-1, keepdim=True)+.04)
                pose_residual = pose_residual*variance_weight*robust_weight
            residual_grid = (pose_residual/.1).T.reshape(1, 3, height, width)
            residual_vector = F.adaptive_avg_pool2d(residual_grid, (4, 5)).flatten()
            if self.residual_modulator is not None:
                residual_vector = residual_vector*(1+.5*self.residual_modulator(spatial).tanh())
            value = self.residual_pose(residual_vector)
            if self.residual_only:
                value = value+value.new_tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.])
            delta = delta @ pose_from_vector(value)
        return fixed @ delta, X_T0.reshape(height, width, 3), residual_world.reshape(height, width, 3)
