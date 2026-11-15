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
from src.models.pose import PoseHead, rotation_6d


class SpatialBootstrap(nn.Module):
    def __init__(self, token_dim=1024, latent_dim=256):
        super().__init__()
        self.query_residual = nn.Linear(token_dim+3, latent_dim)
        self.key_residual = nn.Linear(latent_dim+4, latent_dim)
        for layer in (self.query_residual, self.key_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(20.)))
        self.patch_features = nn.Sequential(nn.LayerNorm(token_dim+5), nn.Linear(token_dim+5, 64), nn.GELU())
        self.coefficients = nn.Linear(64, 3)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)
        self.translation = nn.Sequential(nn.Flatten(), nn.Linear(64*4*5, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.translation[-1].weight)
        nn.init.constant_(self.translation[-1].bias, 0.)
        with torch.no_grad():
            self.translation[-1].bias[2] = -2.

    def forward(self, tokens, query_features, memory, K, height, width):
        eye = torch.eye(4, device=tokens.device, dtype=tokens.dtype)
        _, rays, _ = world_rays(K, eye, height, width)
        if not len(memory):
            return eye, dict(anchors=tokens.new_zeros(len(tokens), 3), confidence=tokens.new_zeros(len(tokens)))
        scales = torch.linalg.eigvalsh(memory.covars).clamp_min(1e-10).sqrt().log()
        q = F.normalize(query_features+self.query_residual(torch.cat((tokens, rays), -1)), dim=-1)
        k = F.normalize(memory.features+self.key_residual(torch.cat((memory.features, scales, memory.alpha[:, None]), -1)), dim=-1)
        weights = (q @ k.T * self.log_temperature.exp().clamp(max=60.) + memory.alpha.clamp_min(1e-6).log()[None]).softmax(-1)
        anchors = weights @ memory.means
        confidence = 1.+(weights*weights.clamp_min(1e-9).log()).sum(-1)/math.log(max(2, len(memory)))
        center = anchors.mean(0)
        radius = (anchors-center).norm(dim=-1, keepdim=True)
        features = self.patch_features(torch.cat((tokens, rays, confidence[:, None], radius), -1))
        coefficients = self.coefficients(features)
        # Preserve ray layout; uniform image pooling cannot recover these signed
        # world vectors. Learned corrections can handle oblique/reverse views.
        a = ((rays[:, 0]+coefficients[:, 0])[:, None]*(anchors-center)).sum(0)
        b = ((rays[:, 1]+coefficients[:, 1])[:, None]*(anchors-center)).sum(0)
        R = rotation_6d(torch.cat((a, b)))
        spatial = features.T.reshape(1, 64, height, width)
        local_translation = self.translation(F.adaptive_avg_pool2d(spatial, (4, 5)))[0]
        centroid_weights = coefficients[:, 2].softmax(0)
        translation = centroid_weights @ anchors + R @ local_translation
        T = torch.cat((torch.cat((R, translation[:, None]), -1), eye[3:]), 0)
        return T, dict(anchors=anchors, confidence=confidence, bootstrap_attention=weights)


class SpatialRefinement(nn.Module):
    def __init__(self, token_dim=2048):
        super().__init__()
        self.token_project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 32), nn.GELU())
        # camera point, rendered point in T0 camera, metric residual, ray, O, var
        self.spatial = nn.Sequential(nn.Conv2d(46, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.pose = PoseHead(64*4*5)

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
        delta = self.pose(spatial)
        return fixed @ delta, X_T0.reshape(height, width, 3), residual_world.reshape(height, width, 3)
