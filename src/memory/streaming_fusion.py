"""Persistent, differentiable moment updates with stable spatial identities.

Only Gaussian parameters and their scalar observation support are retained.  In
particular, association never re-hashes an already fused Gaussian using its
updated scale: crossing a scale-band boundary cannot create another copy of it.
The detached nearest-neighbour decisions do not truncate the public memory.
"""

import torch
from torch import Tensor, nn

from src.memory.gaussians import GaussianMemory


class StreamingMomentFusion(nn.Module):
    """Associate new observations to old spatial identities, then update moments.

    ``support`` is accumulated observation mass, with a finite memory horizon for
    *updates*, not a limit on Gaussian count.  Each incoming frame contributes at
    most ``frame_support_cap`` per identity, so several overlapping patches from
    one image do not overwhelm a surface observed in many previous frames.
    Geometry and all moment/update arithmetic remain differentiable; only group
    membership and the optional invalid-observation rejection are detached.
    """

    def __init__(self, *, support_cap=32., frame_support_cap=1.,
                 mahalanobis_threshold=1.5, normal_cos=.8,
                 scale_ratio=4., association_chunk=128):
        super().__init__()
        if support_cap <= 0 or frame_support_cap <= 0:
            raise ValueError("Observation support caps must be positive")
        self.support_cap = float(support_cap)
        self.frame_support_cap = float(frame_support_cap)
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.normal_cos = float(normal_cos)
        self.scale_ratio = float(scale_ratio)
        self.association_chunk = int(association_chunk)
        self.feature_gate = nn.Parameter(torch.tensor(0.))
        self.opacity_gate = nn.Parameter(torch.tensor(-2.))

    @torch.no_grad()
    def _cost(self, candidates, memory):
        """Scale-aware ellipsoid distance with unoriented surface agreement."""
        if not len(candidates) or not len(memory):
            return candidates.means.new_empty((len(candidates), len(memory)))
        cov_c = candidates.covars.detach().float()
        cov_m = memory.covars.detach().float()
        eig_c, axes_c = torch.linalg.eigh(cov_c)
        eig_m, axes_m = torch.linalg.eigh(cov_m)
        normal_agreement = (axes_c[..., 0] @ axes_m[..., 0].T).abs()
        # Spheres have no distinguished normal. This branch also makes the
        # association useful for analytical spherical fixtures without choosing
        # an arbitrary direction from a repeated eigenspace.
        sphere_c = eig_c[:, -1] <= eig_c[:, 0].clamp_min(1e-10)*1.05
        sphere_m = eig_m[:, -1] <= eig_m[:, 0].clamp_min(1e-10)*1.05
        compatible = ((normal_agreement >= self.normal_cos)
                      | sphere_c[:, None] | sphere_m[None, :])
        size_c = eig_c[:, 1].clamp_min(1e-10)
        size_m = eig_m[:, 1].clamp_min(1e-10)
        ratio = size_c[:, None]/size_m[None, :]
        compatible &= (ratio >= self.scale_ratio**-2) & (ratio <= self.scale_ratio**2)
        eye = torch.eye(3, device=cov_c.device, dtype=cov_c.dtype)
        costs = []
        for start in range(0, len(candidates), self.association_chunk):
            stop = start+self.association_chunk
            delta = (candidates.means[start:stop, None].detach().float()
                     - memory.means[None].detach().float())
            covariance = cov_c[start:stop, None]+cov_m[None]+1e-8*eye
            solved = torch.linalg.solve(covariance, delta[..., None]).squeeze(-1)
            costs.append((delta*solved).sum(-1).clamp_min(0))
        cost = torch.cat(costs)
        return cost.masked_fill(~compatible, torch.inf)

    @torch.no_grad()
    def _groups(self, memory, candidates):
        """Old identities stay fixed; only unmatched incoming patches group."""
        count = len(memory)
        if len(memory):
            cost = self._cost(candidates, memory)
            best_cost, ids = cost.min(-1)
            matched = best_cost <= self.mahalanobis_threshold**2
        else:
            ids = torch.zeros(len(candidates), device=candidates.means.device, dtype=torch.long)
            matched = torch.zeros(len(candidates), device=candidates.means.device, dtype=torch.bool)
        unmatched = (~matched).nonzero(as_tuple=False).flatten()
        if not len(unmatched):
            return ids, count
        fresh = candidates.subset(unmatched)
        fresh_cost = self._cost(fresh, fresh).cpu()
        # Representative neighbourhoods, rather than transitive connected
        # components, prevent a long planar chain from collapsing to one splat.
        representatives = []
        fresh_ids = []
        threshold = self.mahalanobis_threshold**2
        for i in range(len(fresh)):
            if representatives:
                neighbours = fresh_cost[i, representatives]
                value, chosen = neighbours.min(0)
                if value <= threshold:
                    fresh_ids.append(count+int(chosen))
                    continue
            fresh_ids.append(count+len(representatives))
            representatives.append(i)
        ids[unmatched] = torch.tensor(fresh_ids, device=ids.device)
        return ids, count+len(representatives)

    def forward(self, memory: GaussianMemory, support: Tensor,
                candidates: GaussianMemory, candidate_support: Tensor):
        if support.shape != (len(memory),) or candidate_support.shape != (len(candidates),):
            raise ValueError("One scalar observation support is required per Gaussian")
        if not len(candidates):
            return memory, support
        # The writer rejects invalid depth; this additional guard keeps a zero
        # confidence observation from creating a permanent empty identity.
        with torch.no_grad():
            keep = (torch.isfinite(candidate_support) & (candidate_support > 0)
                    & torch.isfinite(candidates.means).all(-1)
                    & torch.isfinite(candidates.covars).all((-1, -2)))
        candidates = candidates.subset(keep)
        candidate_support = candidate_support[keep]
        if not len(candidates):
            return memory, support
        ids, count = self._groups(memory, candidates)
        old_count = len(memory)

        def total(value):
            return value.new_zeros((count, *value.shape[1:])).index_add(0, ids, value)

        raw_mass = candidate_support.clamp_min(0)
        frame_mass = total(raw_mass)
        # A candidate frame contributes one independent observation per spatial
        # identity. Fractional write confidence is preserved below this ceiling.
        mass = raw_mass*(frame_mass.clamp_max(self.frame_support_cap)
                         / frame_mass.clamp_min(1e-8))[ids]
        incoming_mass = total(mass)
        old_mass = torch.cat((support.clamp(0, self.support_cap),
                              incoming_mass.new_zeros(count-old_count)))
        combined_mass = old_mass+incoming_mass
        denominator = combined_mass.clamp_min(1e-8)

        def extend(value):
            return torch.cat((value, value.new_zeros((count-old_count, *value.shape[1:]))))

        old_means = extend(memory.means)
        means = (old_mass[:, None]*old_means + total(mass[:, None]*candidates.means))/denominator[:, None]
        old_delta = old_means-means
        new_delta = candidates.means-means[ids]
        covariance = (
            old_mass[:, None, None]*(extend(memory.covars)+old_delta[..., :, None]*old_delta[..., None, :])
            + total(mass[:, None, None]*(candidates.covars+new_delta[..., :, None]*new_delta[..., None, :]))
        )/denominator[:, None, None]
        covariance = .5*(covariance+covariance.transpose(-1, -2))
        covariance = covariance+torch.eye(3, device=covariance.device, dtype=covariance.dtype)*1e-8
        old_features = extend(memory.features)
        feature_mean = (old_mass[:, None]*old_features+total(mass[:, None]*candidates.features))/denominator[:, None]
        features = torch.lerp(old_features, feature_mean, self.feature_gate.sigmoid())
        fresh = torch.arange(count, device=means.device) >= old_count
        features = torch.where(fresh[:, None], feature_mean, features)
        old_alpha = extend(memory.alpha)
        alpha_mean = (old_mass*old_alpha+total(mass*candidates.alpha))/denominator
        incoming_alpha = total(mass*candidates.alpha)/incoming_mass.clamp_min(1e-8)
        alpha_union = 1-(1-old_alpha)*(1-incoming_alpha)
        # Limit evidence union by the actual proportion of this observation;
        # repeated overlap does not immediately turn every surface opaque.
        alpha = torch.lerp(alpha_mean, alpha_union,
                           self.opacity_gate.sigmoid()*incoming_mass/denominator)
        alpha = torch.where(fresh, alpha_mean, alpha).clamp(1e-5, 1-1e-5)
        # Untouched identities are exactly retained, including covariance, rather
        # than accumulating a numerical floor at every unrelated frame.
        updated = incoming_mass > 0
        means = torch.where(updated[:, None], means, old_means)
        covariance = torch.where(updated[:, None, None], covariance, extend(memory.covars))
        features = torch.where(updated[:, None], features, old_features)
        alpha = torch.where(updated, alpha, old_alpha)
        return GaussianMemory(means, covariance, alpha, features), combined_mass.clamp_max(self.support_cap)
