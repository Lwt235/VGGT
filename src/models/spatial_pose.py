"""Open-set correspondences, invariant forward moments, spatial refinement.

No query pose/depth is accepted. v2j uses pre-injection camera geometry and
two-ended whitened IRLS moments; legacy heads restore earlier project runs.
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


class LegacySpatialBootstrap(nn.Module):
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


def point_irls_pose(camera_points, anchors, weights, *, tau=.25, damping=1e-6, iterations=2):
    """Two-ended centered right-whitened moments, followed by two IRLS updates.

    This is part of the differentiable prediction, never post-prediction fitting.
    No SVD/eigendecomposition or query ground truth is used.
    """
    original_dtype = anchors.dtype
    # Tiny moment systems in double avoid TF32/rank-near-planar cancellation;
    # neural features and rendered state remain float32/bfloat16.
    camera_points, anchors, weights = camera_points.double(), anchors.double(), weights.double()
    eye = torch.eye(3, device=anchors.device, dtype=anchors.dtype)
    def solve(votes):
        votes = votes/votes.sum().clamp_min(1e-8)
        Xbar = (votes[:, None]*anchors).sum(0)
        Pbar = (votes[:, None]*camera_points).sum(0)
        dX, dP = anchors-Xbar, camera_points-Pbar
        H = (votes[:, None]*dX).T @ dP
        C = (votes[:, None]*dP).T @ dP
        lam = damping*C.trace()/3+1e-8
        M = torch.linalg.solve(C+lam*eye, H.T).T
        R = rotation_6d(torch.cat((M[:, 0], M[:, 1])))
        independent = (M[:, 0].norm() > 1e-8) & (torch.linalg.cross(M[:, 0], M[:, 1]).norm() > 1e-10)
        R = torch.where(independent, R, eye)
        t = Xbar-R @ Pbar
        return R, t
    robust = weights
    R, t = solve(robust)
    for _ in range(iterations):
        residual = anchors-(camera_points @ R.T+t)
        robust = weights*tau**2/(tau**2+residual.square().sum(-1))
        R, t = solve(robust)
    T = torch.cat((torch.cat((R, t[:, None]), -1), anchors.new_tensor([[0., 0., 0., 1.]])), 0)
    return T.to(original_dtype), robust.to(original_dtype)


def whitened_irls_pose(camera_points, rays, anchors, weights, *, tau_perp=.08,
                        tau_parallel=.40, translation_damping=.02, iterations=3):
    """Differentiable lateral-ray IRLS and equivariant ray-bundle intersection.

    Predicted radial depth initializes the centered moments and supplies only
    a broad consistency prior. Tight robust weights use lateral ray distance.
    All solves are internal forward operations, with no query supervision.
    """
    dtype = anchors.dtype
    points, rays, anchors, weights = [value.double() for value in (camera_points, rays, anchors, weights)]
    rays = F.normalize(rays, dim=-1)
    initial, _ = point_irls_pose(points, anchors, weights, iterations=0)
    R, t = initial[:3, :3], initial[:3, 3]
    eye = torch.eye(3, device=anchors.device, dtype=anchors.dtype)
    radial_prior = points.norm(dim=-1)
    corrected = points
    robust = weights

    def intersect(rotation, votes, prior_points):
        direction = rays @ rotation.T
        projector = eye-direction[:, :, None]*direction[:, None, :]
        normalized = votes/votes.sum().clamp_min(1e-8)
        Xbar = (normalized[:, None]*anchors).sum(0)
        Pbar = (normalized[:, None]*prior_points).sum(0)
        prior_center = Xbar-rotation @ Pbar
        # Unnormalized votes keep the fixed .02 regularizer small relative to
        # a supported bundle. The transported center prior preserves SE(3).
        A = (votes[:, None, None]*projector).sum(0)+translation_damping*eye
        b = (votes[:, None]*torch.einsum('qij,qj->qi', projector, anchors)).sum(0)
        return torch.linalg.solve(A, b+translation_damping*prior_center)

    for _ in range(iterations):
        direction = rays @ R.T
        delta = anchors-t
        radial = (delta*direction).sum(-1)
        lateral = delta-radial[:, None]*direction
        robust = weights*(tau_perp**2/(tau_perp**2+lateral.square().sum(-1))).square()
        robust = robust*tau_parallel**2/(tau_parallel**2+(radial-radial_prior).square())
        t = intersect(R, robust, corrected)
        positive_depth = ((anchors-t)*direction).sum(-1).clamp_min(.05)
        corrected = rays*positive_depth[:, None]
        updated, _ = point_irls_pose(corrected, anchors, robust, iterations=0)
        R = updated[:3, :3]
        t = intersect(R, robust, corrected)
    T = torch.cat((torch.cat((R, t[:, None]), -1), anchors.new_tensor([[0., 0., 0., 1.]])), 0)
    return T.to(dtype), robust.to(dtype)


class SpatialBootstrap(nn.Module):
    """Memory-independent camera geometry, open-set matches, invariant moments."""
    def __init__(self, token_dim=1024, latent_dim=256, *, ray_bundle=False, anisotropic=False, metric_scale=False, planar_safe=False):
        super().__init__()
        self.ray_bundle = ray_bundle
        self.anisotropic = anisotropic
        self.metric_scale = metric_scale
        self.planar_safe = planar_safe
        self.key_dropout_mode = 'none'
        self.query_residual = nn.Linear((2 if metric_scale else 1)*token_dim+3, latent_dim)
        self.key_residual = nn.Linear(latent_dim+4, latent_dim)
        for layer in (self.query_residual, self.key_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(20.)))
        self.dustbin_logit = nn.Parameter(torch.tensor(10.))
        self.camera_geometry = nn.Sequential(nn.LayerNorm(2*token_dim+3), nn.Linear(2*token_dim+3, 256),
                                             nn.GELU(), nn.Linear(256, 2))
        nn.init.zeros_(self.camera_geometry[-1].weight)
        with torch.no_grad():
            self.camera_geometry[-1].bias.copy_(torch.tensor([2.4, 1.5]))
        if ray_bundle:
            # Residual context preserves a compatible v2j camera-depth start.
            # Two focal channels are normalized to patch-grid dimensions.
            self.camera_context = nn.Sequential(nn.Conv2d(258, 64, 3, padding=1), nn.GELU(),
                                                 nn.Conv2d(64, 2, 3, padding=1))
            self.dustbin_head = nn.Sequential(nn.LayerNorm(2*token_dim), nn.Linear(2*token_dim, 64),
                                               nn.GELU(), nn.Linear(64, 1))
            for layer in (self.camera_context[-1], self.dustbin_head[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def camera_observation(self, tokens_pre_inject, shallow, K, height, width):
        eye = torch.eye(4, device=K.device, dtype=K.dtype)
        _, rays, dz = world_rays(K, eye, height, width)
        inputs = torch.cat((tokens_pre_inject, shallow, rays), -1)
        hidden = self.camera_geometry[:3](inputs)
        raw = self.camera_geometry[3](hidden)
        if self.ray_bundle:
            focal = torch.stack((K[0, 0]/width, K[1, 1]/height)).clamp_min(1e-6).log()
            spatial = hidden.T.reshape(1, 256, height, width)
            context = torch.cat((spatial, focal[None, :, None, None].expand(1, 2, height, width)), 1)
            raw = raw+self.camera_context(context)[0].permute(1, 2, 0).reshape(-1, 2)
        z = F.softplus(raw[:, 0])+.05
        confidence = raw[:, 1].sigmoid()
        return rays/dz[:, None]*z[:, None], z, confidence, rays

    def forward(self, tokens_pre_inject, query_features, memory, K, height, width, *, shallow):
        points, z, c_depth, rays = self.camera_observation(tokens_pre_inject, shallow, K, height, width)
        count = len(points)
        eye = torch.eye(4, device=K.device, dtype=K.dtype)
        if not len(memory):
            return eye, dict(anchors=points.new_zeros(count, 3), confidence=z*0.,
                camera_patch_points=points, z_cam=z, c_depth=c_depth,
                matching_attention=points.new_ones(count, 1), dustbin_probability=z*0.+1,
                bootstrap_attention=points.new_zeros(count, 0), bootstrap_support=torch.zeros(0, device=z.device, dtype=torch.bool),
                anchor_votes=z*0., match_mass=z.new_zeros(()))
        # Rotation-invariant covariance summaries; no eigen/SVD pose path.
        covariance = memory.covars
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        invariants = torch.stack((trace/3, covariance.square().sum((-1, -2)).sqrt()/math.sqrt(3),
                                  torch.linalg.det(covariance).clamp_min(1e-24).pow(1/3)), -1)
        scales = .5*invariants.clamp_min(1e-10).log()
        q = F.normalize(query_features+self.query_residual(torch.cat((tokens_pre_inject, shallow, rays) if self.metric_scale else (shallow, rays), -1)), dim=-1)
        k = F.normalize(memory.features+self.key_residual(torch.cat((memory.features, scales, memory.alpha[:, None]), -1)), dim=-1)
        logits = q @ k.T*self.log_temperature.exp().clamp(max=60.)+memory.alpha.clamp_min(1e-6).log()[None]
        dustbin = self.dustbin_logit.expand(count, 1)
        if self.ray_bundle:
            dustbin = dustbin+self.dustbin_head(torch.cat((tokens_pre_inject, shallow), -1))
        W = torch.cat((logits, dustbin), -1).softmax(-1)
        p_match = 1-W[:, -1]
        inlier = W[:, :-1]/p_match[:, None].clamp_min(1e-6)
        anchors = inlier @ memory.means
        entropy_confidence = 1+(inlier*inlier.clamp_min(1e-9).log()).sum(-1)/math.log(max(2, len(memory)))
        extras = {}
        if self.anisotropic:
            from src.models.ellipsoid_pose import posterior_ellipsoids, anisotropic_ray_pose
            anchors, covariance = posterior_ellipsoids(inlier, memory)
            weights = p_match*c_depth
            T, votes, lm = anisotropic_ray_pose(points, rays, anchors, covariance, weights, return_diagnostic=True, metric_scale=self.metric_scale, planar_safe=self.planar_safe)
            extras = dict(posterior_covars=covariance, lm_energy_history=lm['energy_history'],
                          lm_accepted_scales=lm['accepted_scales'])
        else:
            weights = p_match*c_depth*anchor_vote_weights(inlier, memory, anchors, entropy_confidence)
            T, votes = whitened_irls_pose(points, rays, anchors, weights) if self.ray_bundle else point_irls_pose(points, anchors, weights)
        # On observable matches this is exactly one in floating point. Identity
        # fallback in the unobservable regime is intentionally not gauge equivariant.
        quality = ((p_match.sum()-.25)/.025).sigmoid()
        quality = quality*p_match.sum()/(p_match.sum()+1e-8)
        mixed_R = rotation_6d(torch.cat((quality*T[:3, 0]+(1-quality)*eye[:3, 0],
                                       quality*T[:3, 1]+(1-quality)*eye[:3, 1])))
        top = torch.cat((mixed_R, (quality*T[:3, 3])[:, None]), -1)
        T = torch.cat((top, eye[3:]), 0)
        return T, dict(anchors=anchors, confidence=p_match, camera_patch_points=points, z_cam=z, c_depth=c_depth,
            matching_attention=W, dustbin_probability=W[:, -1], bootstrap_attention=inlier,
            bootstrap_support=torch.ones(len(memory), device=z.device, dtype=torch.bool),
            anchor_votes=votes, match_mass=p_match.sum(), **extras)


class SpatialRefinement(nn.Module):
    def __init__(self, token_dim=2048, *, direct_residual=True, residual_only=False, point_to_plane=False, relative_geometry=False):
        super().__init__()
        self.token_project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 32), nn.GELU())
        # camera point, rendered point in T0 camera, metric residual, ray, O, var
        self.spatial = nn.Sequential(nn.Conv2d(46, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.pose = None if residual_only else PoseHead(64*4*5)
        self.residual_only = residual_only
        self.point_to_plane = point_to_plane
        self.relative_geometry = relative_geometry
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
        if self.point_to_plane:
            if rendered.normal is None:
                raise ValueError('Point-to-plane refinement requires O5 surface normals')
            normal = rendered.normal.reshape(-1, 3)
            residual_world = (residual_world*normal).sum(-1, keepdim=True)*normal
        residual_local = residual_world @ fixed[:3, :3]
        rendered_local = transform_points(rendered.world.reshape(-1, 3), invert_se3(fixed))*opacity
        _, rays, _ = world_rays(K, torch.eye(4, device=K.device), height, width)
        depth_scale = rendered.depth.reshape(-1, 1).clamp_min(.2) if self.relative_geometry else torch.ones_like(opacity)
        variance = rendered.depth_variance.reshape(-1, 1)/depth_scale.square()
        geometry = torch.cat((camera_patch/depth_scale, rendered_local/depth_scale, residual_local/depth_scale, rays, opacity,
                              torch.log1p(variance)), -1)
        inputs = torch.cat((self.token_project(tokens), geometry), -1).T.reshape(1, 46, height, width)
        spatial = F.adaptive_avg_pool2d(self.spatial(inputs), (4, 5)).flatten()
        delta = torch.eye(4, device=K.device, dtype=K.dtype) if self.pose is None else self.pose(spatial)
        if self.residual_pose is not None:
            pose_residual = residual_local/depth_scale
            if self.residual_only:
                # Mixed depth layers and large correspondence errors should not
                # dominate one global camera correction. Context may reweight
                # these metric residuals, but cannot invent a pose on its own.
                variance_weight = 1/(1+variance/.01)
                robust_weight = .2/torch.sqrt((residual_local/depth_scale).square().sum(-1, keepdim=True)+.04)
                pose_residual = pose_residual*variance_weight*robust_weight
            residual_grid = (pose_residual/.1).T.reshape(1, 3, height, width)
            residual_vector = F.adaptive_avg_pool2d(residual_grid, (4, 5)).flatten()
            if self.residual_modulator is not None:
                residual_vector = residual_vector*(1+.5*self.residual_modulator(spatial).tanh())
            value = self.residual_pose(residual_vector)
            if self.residual_only:
                value = value+value.new_tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.])
            if self.relative_geometry:
                metric_depth = (rendered.depth.reshape(-1, 1)*opacity).sum()/opacity.sum().clamp_min(1e-6)
                value = torch.cat((value[:6], value[6:]*metric_depth.clamp(.2, 10.)))
            delta = delta @ pose_from_vector(value)
        return fixed @ delta, X_T0.reshape(height, width, 3), residual_world.reshape(height, width, 3)
