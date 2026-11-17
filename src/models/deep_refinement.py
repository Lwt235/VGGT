"""Deep camera-frame matching residuals, followed by monotone forward LM.

No DPT points, query GT or free pose regression enters this module. A zero
matching residual preserves W0; an exact clean pose is a fixed point of LM.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.geometry import transform_points
from src.models.ellipsoid_pose import anisotropic_ray_pose, posterior_ellipsoids


class DeepEllipsoidRefinement(nn.Module):
    def __init__(self, token_dim=2048, latent_dim=256, *, isotropic=False):
        super().__init__()
        self.isotropic = isotropic
        self.query_delta = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, latent_dim))
        self.dustbin_delta = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        for head in (self.query_delta, self.dustbin_delta):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, final, memory, bootstrap, init_pose):
        base = init_pose.detach()
        count = len(final)
        if not len(memory):
            zero = final.new_zeros(count, 3)
            return base, zero, zero, dict(refined_matching_attention=final.new_ones(count, 1),
                refined_anchors=zero, refinement_lm_energy_history=final.new_zeros(3),
                refinement_lm_accepted_scales=final.new_zeros(2))
        query = F.normalize(bootstrap['matching_query']+self.query_delta(final), dim=-1)
        logits = query @ bootstrap['matching_keys'].T*bootstrap['matching_temperature']
        logits = logits+memory.alpha.clamp_min(1e-6).log()[None]
        if bootstrap.get('matching_mask') is not None:
            logits = logits.masked_fill(~bootstrap['matching_mask'][None], -80.)
        dustbin = bootstrap['matching_dustbin_logit']+self.dustbin_delta(final)
        W = torch.cat((logits, dustbin), -1).softmax(-1)
        p_match = 1-W[:, -1]
        inlier = W[:, :-1]/p_match[:, None].clamp_min(1e-6)
        anchors, covars = posterior_ellipsoids(inlier, memory, isotropic=self.isotropic)
        rays = bootstrap['camera_rays']
        # init_pose mode eliminates radial reference from MAP/pose/rays only;
        # these placeholder points supply shape, never any monocular depths.
        T, votes, lm = anisotropic_ray_pose(rays, rays, anchors, covars,
            p_match*bootstrap['c_depth'], init_pose=base, iterations=2, return_diagnostic=True)
        # Vanishing match mass safely keeps the supplied read pose. There is
        # no world-axis identity fallback and no GT-conditioned inference.
        if float(p_match.detach().sum()) < 1e-4:
            T = base+T.sum()*0
        world_rays = rays @ base[:3, :3].T
        distance = ((anchors-base[:3, 3])*world_rays).sum(-1).clamp_min(.05)
        X = transform_points(rays*distance[:, None], base)
        return T, X, anchors-X, dict(refined_matching_attention=W, refined_anchors=anchors,
            refined_posterior_covars=covars, refined_dustbin_probability=W[:, -1],
            refinement_lm_energy_history=lm['energy_history'],
            refinement_lm_accepted_scales=lm['accepted_scales'])
