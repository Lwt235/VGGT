"""O5: gsplat attribute moments with alpha-normalized expectations."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from src.geometry import invert_se3, transform_points
from src.memory.gaussians import GaussianMemory


@dataclass
class RenderedMemory:
    world: Tensor
    features: Tensor
    depth: Tensor
    depth_variance: Tensor
    opacity: Tensor


def normalize_rendered_moments(accumulated: Tensor, opacity: Tensor,
                               latent_dim: int, eps=1e-6) -> RenderedMemory:
    valid = opacity > eps
    expectation = torch.where(valid, accumulated / opacity.clamp_min(eps), 0.)
    depth = expectation[..., -2:-1]
    return RenderedMemory(expectation[..., :3], expectation[..., 3:3+latent_dim], depth,
                          (expectation[..., -1:] - depth.square()).clamp_min(0), opacity)


class RenderReader(nn.Module):
    def __init__(self, latent_dim=256, token_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.project = nn.Sequential(nn.Linear(latent_dim+5, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.unseen = nn.Parameter(torch.zeros(token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.gate = nn.Parameter(torch.zeros(token_dim))

    def render(self, memory: GaussianMemory, query_c2w: Tensor, K_patch: Tensor,
               height: int, width: int) -> RenderedMemory:
        # query_c2w must be a MODEL prediction. No GT camera parameter exists here.
        if len(memory) == 0:
            accumulated = query_c2w.new_zeros(height, width, self.latent_dim+5)
            opacity = query_c2w.new_zeros(height, width, 1)
        else:
            if not memory.means.is_cuda:
                raise RuntimeError("O5 requires real CUDA gsplat; no silent CPU substitute")
            from gsplat import rasterization
            w2c = invert_se3(query_c2w)
            depth = transform_points(memory.means, w2c)[:, 2:3]
            attributes = torch.cat((memory.means, memory.features, depth, depth.square()), -1)
            accumulated, opacity, _ = rasterization(
                means=memory.means, quats=None, scales=None, covars=memory.covars,
                opacities=memory.alpha, colors=attributes, viewmats=w2c[None],
                Ks=K_patch[None], width=width, height=height, render_mode="RGB", packed=False)
            accumulated, opacity = accumulated[0], opacity[0]
        return normalize_rendered_moments(accumulated, opacity, self.latent_dim)

    def forward(self, tokens: Tensor, memory: GaussianMemory, query_c2w: Tensor,
                K_patch: Tensor, height: int, width: int):
        rendered = self.render(memory, query_c2w, K_patch, height, width)
        attributes = torch.cat((rendered.world, rendered.features,
                                rendered.depth, rendered.depth_variance), -1)
        visible = self.project(attributes.reshape(-1, self.latent_dim+5))
        opacity = rendered.opacity.reshape(-1, 1)
        received = opacity*visible + (1-opacity)*self.unseen
        return tokens + torch.tanh(self.gate)*self.norm(received), rendered
