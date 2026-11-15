"""Differentiable Gaussian state, without historical images or frame KV."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.geometry import covariance_from_quat_scale, transform_points, unproject, invert_se3


@dataclass
class GaussianMemory:
    means: Tensor
    covars: Tensor
    alpha: Tensor
    features: Tensor

    def __len__(self):
        return len(self.means)

    def subset(self, indices):
        return GaussianMemory(self.means[indices], self.covars[indices],
                              self.alpha[indices], self.features[indices])

    def erase_box(self, lower: Tensor, upper: Tensor):
        """Functional intervention: full memory and model weights stay intact."""
        inside = ((self.means >= lower) & (self.means <= upper)).all(-1)
        return self.subset(~inside), inside

    def retain_grad(self):
        for value in (self.means, self.covars, self.alpha, self.features):
            if value.requires_grad:
                value.retain_grad()


class TokenGaussianWriter(nn.Module):
    """Mapping-only head. K is patch-edge K; c2w is permitted mapping pose.

    The interface cannot be used as a query reader; mapping RGB-D is explicit.
    Cell centers are lifted once per token, without pixel-level densification.
    """

    def __init__(self, token_dim, latent_dim=256, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.geometry = nn.Linear(token_dim, 11)  # delta xyz, delta log s, q, alpha
        self.latent = nn.Linear(token_dim, latent_dim)
        nn.init.normal_(self.geometry.weight, std=.001)
        nn.init.zeros_(self.geometry.bias)
        with torch.no_grad():
            self.geometry.bias[6] = 1.  # WXYZ identity
        # Non-degenerate local ellipsoids; z is thinner than x/y.
        self.register_buffer("base_shape", torch.tensor([1., .8, .25]))

    def forward(self, tokens: Tensor, mapping_depth: Tensor, K_patch: Tensor,
                mapping_c2w: Tensor, valid_mask=None) -> GaussianMemory:
        height, width = mapping_depth.shape
        if tokens.shape[0] != height*width:
            raise ValueError("One Gaussian per patch token required")
        finite_depth = torch.isfinite(mapping_depth) & (mapping_depth > 0)
        if valid_mask is None:
            if not finite_depth.all():
                raise ValueError("Mapping depth must be positive or explicitly masked")
            valid_mask = finite_depth
        else:
            valid_mask = valid_mask & finite_depth
            mapping_depth = torch.where(valid_mask, mapping_depth, 1.)
        raw = self.geometry(tokens)
        depth = mapping_depth.reshape(-1)
        # K_patch focal is image focal / patch_size, so r = depth / (2*f_patch).
        focal = (K_patch[0, 0] + K_patch[1, 1]) * .5
        radius = depth / (2*focal)
        means = unproject(mapping_depth, K_patch, mapping_c2w, center=.5).reshape(-1, 3)
        delta_cam = torch.tanh(raw[:, :3]) * radius[:, None]
        means = means + delta_cam @ mapping_c2w[:3, :3].T
        scale = radius[:, None] * self.base_shape * torch.exp(.7*torch.tanh(raw[:, 3:6]))
        cov_cam = covariance_from_quat_scale(raw[:, 6:10], scale.clamp_min(1e-4))
        R = mapping_c2w[:3, :3]
        covars = R @ cov_cam @ R.T
        return GaussianMemory(means, covars, torch.sigmoid(raw[:, 10]), self.latent(tokens)).subset(valid_mask.reshape(-1))


class SpatialMomentFusion(nn.Module):
    """Adaptive spatial hashing, orientation filtering, differentiable moments.

    Discrete cell/orientation membership is detached. Group arithmetic and the
    feature/opacity update gates remain differentiable. No total-capacity cap.
    Voxel size is derived from local physical scale; normals are unoriented.
    """

    def __init__(self, min_cell=.02, normal_cos=.8):
        super().__init__()
        self.min_cell = min_cell
        self.normal_cos = normal_cos
        self.feature_gate = nn.Parameter(torch.tensor(0.))
        self.opacity_gate = nn.Parameter(torch.tensor(0.))

    def forward(self, memory: GaussianMemory, reference_c2w=None) -> GaussianMemory:
        if len(memory) == 0:
            return memory
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(memory.covars.detach())
            # Quantized physical scale bands avoid a single global spatial cell.
            size = (2*eigenvalues[:, 1].clamp_min(1e-10).sqrt()).clamp_min(self.min_cell)
            band = torch.floor(torch.log2(size/self.min_cell))
            cell_size = self.min_cell * torch.pow(2., band)
            # Anchor only the discrete hash grid to the first mapping camera.
            # A random world gauge must transport the map, not change which
            # observations fuse. The stored/output moments remain WORLD values.
            hash_means = memory.means.detach()
            if reference_c2w is not None:
                hash_means = transform_points(hash_means, invert_se3(reference_c2w.detach()))
            cells = torch.floor(hash_means/cell_size[:, None]).long()
            keys = torch.cat((band.long()[:, None], cells), -1).cpu().tolist()
            normals = eigenvectors[..., 0].cpu().numpy()
            buckets = {}
            groups = []
            for i, key in enumerate(keys):
                candidate_groups = buckets.setdefault(tuple(key), [])
                for group_idx in candidate_groups:
                    first = groups[group_idx][0]
                    if abs(normals[first] @ normals[i]) >= self.normal_cos:
                        groups[group_idx].append(i)
                        break
                else:
                    candidate_groups.append(len(groups))
                    groups.append([i])
        group_ids = [0]*len(memory)
        for group_idx, group in enumerate(groups):
            for idx in group:
                group_ids[idx] = group_idx
        ids = torch.tensor(group_ids, device=memory.means.device)
        def total(values):
            out = values.new_zeros((len(groups), *values.shape[1:]))
            return out.index_add(0, ids, values)
        weights = memory.alpha.clamp_min(1e-6)
        weights = weights / total(weights)[ids]
        means = total(weights[:, None]*memory.means)
        delta = memory.means-means[ids]
        covars = total(weights[:, None, None]*(memory.covars+delta[:, :, None]*delta[:, None, :]))
        covars = (covars+covars.transpose(-1, -2))*.5
        covars = covars+torch.eye(3, device=covars.device, dtype=covars.dtype)*1e-8
        feature_mean = total(weights[:, None]*memory.features)
        first = torch.tensor([group[0] for group in groups], device=memory.means.device)
        features = torch.lerp(memory.features[first], feature_mean, self.feature_gate.sigmoid())
        alpha_mean = total(weights*memory.alpha)
        alpha_union = -torch.expm1(total(torch.log1p(-memory.alpha.clamp_max(1-1e-6))))
        alpha = torch.lerp(alpha_mean, alpha_union, self.opacity_gate.sigmoid()).clamp(1e-5, 1-1e-5)
        return GaussianMemory(means, covars, alpha, features)
