"""O6: undamped center PE plus independent per-head ellipsoid/ray bias."""

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from src.geometry import ray_ellipsoid_distance, transform_points, world_rays
from src.memory.gaussians import GaussianMemory


def center_pe(means: Tensor, bands=8):
    frequency = (2. ** torch.arange(bands, device=means.device, dtype=means.dtype)) * math.pi
    phase = means[..., :, None] * frequency
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)


def active_memory(memory, query_c2w, K, height, width, *, budget=2048, margin=.25):
    """Read budget is separate from stored memory; functional view only.

    Conservative 3-sigma bounding sphere protects off-screen ellipsoid overlap.
    """
    from src.geometry import invert_se3
    if not len(memory):
        return memory, torch.empty(0, device=memory.means.device, dtype=torch.long)
    with torch.no_grad():
        camera = transform_points(memory.means, invert_se3(query_c2w))
        z = camera[:, 2]
        radius = 3*torch.linalg.eigvalsh(memory.covars)[:, -1].clamp_min(1e-10).sqrt()
        image = camera @ K.T
        xy = image[:, :2] / z[:, None].clamp_min(1e-5)
        radius_pixels = K[:2, :2].diag().max()*radius/(z-radius).clamp_min(1e-3)
        overlap = ((z+radius > .01) & (xy[:, 0]+radius_pixels >= -margin*width) &
                   (xy[:, 0]-radius_pixels <= (1+margin)*width) &
                   (xy[:, 1]+radius_pixels >= -margin*height) &
                   (xy[:, 1]-radius_pixels <= (1+margin)*height))
        indices = overlap.nonzero(as_tuple=True)[0]
        if budget is not None and len(indices) > budget:
            order = torch.argsort(camera[indices].norm(dim=-1))[:budget]
            indices = indices[order]
    return memory.subset(indices), indices


class ParameterPEReader(nn.Module):
    def __init__(self, token_dim=1024, latent_dim=256, heads=8, bands=8):
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must divide heads")
        self.heads, self.bands = heads, bands
        self.query = nn.Linear(token_dim, token_dim)
        self.query_geometry = nn.Linear(6*bands+6, token_dim, bias=False)
        self.key = nn.Linear(latent_dim+6*bands, token_dim)
        self.value = nn.Linear(latent_dim, token_dim)
        self.bias = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, heads))
        self.output = nn.Linear(token_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.gate = nn.Parameter(torch.zeros(token_dim))
        self.gamma = nn.Parameter(torch.full((heads,), 1.))
        self.beta = nn.Parameter(torch.full((heads,), 1.))

    def forward(self, tokens: Tensor, memory: GaussianMemory, query_c2w: Tensor,
                K_patch: Tensor, height: int, width: int, render_depth: Tensor,
                *, rendered=None, active_budget=2048, query_chunk=128, return_attention=False):
        selected, indices = active_memory(memory, query_c2w, K_patch, height, width, budget=active_budget)
        if len(selected) == 0:
            return tokens, {"indices": indices, "attention": None}
        nquery, channels = tokens.shape
        head_dim = channels//self.heads
        origins, directions, camera_dz = world_rays(K_patch, query_c2w, height, width)
        eigenvalues, eigenvectors = torch.linalg.eigh(selected.covars)
        log_scales = eigenvalues.clamp_min(1e-10).sqrt().log()
        normals = eigenvectors[..., 0]
        position = tokens.new_zeros(nquery, 6*self.bands)
        if rendered is not None:
            position = center_pe(rendered.world.reshape(-1, 3), self.bands)*rendered.opacity.reshape(-1, 1)
        ray_geometry = torch.cat((position, directions, origins/5.), -1)
        q = (self.query(tokens)+self.query_geometry(ray_geometry)).reshape(nquery, self.heads, head_dim)
        k = self.key(torch.cat((selected.features, center_pe(selected.means, self.bands)), -1))
        k = k.reshape(len(selected), self.heads, head_dim)
        v = self.value(selected.features).reshape(len(selected), self.heads, head_dim)
        chunks, attention = [], []
        for start in range(0, nquery, query_chunk):
            stop = min(start+query_chunk, nquery)
            rho2, t = ray_ellipsoid_distance(origins[start:stop], directions[start:stop],
                                             selected.means, selected.covars)
            closest_z = t*camera_dz[start:stop, None]
            relative_z = closest_z-render_depth.reshape(-1)[start:stop, None]
            normal_cos = torch.abs(directions[start:stop] @ normals.T)
            shape = log_scales[None].expand(stop-start, -1, -1)
            geometry = torch.cat((torch.log1p(rho2)[..., None], torch.log1p(closest_z)[..., None],
                                  torch.tanh(relative_z/2.)[..., None], normal_cos[..., None], shape,
                                  selected.alpha[None, :, None].expand(stop-start, -1, -1)), -1)
            bias = self.bias(geometry).permute(2, 0, 1)
            prior = (-F.softplus(self.gamma)[:, None, None]*torch.log1p(rho2)[None]
                     -F.softplus(self.beta)[:, None, None]*F.relu(relative_z)[None])
            logits = torch.einsum("qhd,mhd->hqm", q[start:stop], k)/math.sqrt(head_dim) + bias + prior
            weights = logits.softmax(-1)
            chunks.append(torch.einsum("hqm,mhd->qhd", weights, v).reshape(stop-start, channels))
            if return_attention:
                attention.append(weights.detach())
        output = tokens + self.gate.tanh()*self.norm(self.output(torch.cat(chunks)))
        return output, {"indices": indices, "attention": torch.cat(attention, 1) if attention else None}
