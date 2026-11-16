"""Differentiable camera-local LM against posterior anisotropic ellipsoids.

All observations here are predictions / spatial memory, never query GT.
Closed ray-depth elimination and energy backtracking belong to model.forward.
"""
import math
import torch
from torch.nn import functional as F


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = x*0
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def local_se3_exp(omega, nu):
    """Stable Rodrigues + SE(3) left Jacobian, including zero-increment gradients."""
    angle = torch.linalg.vector_norm(omega)
    square = angle.square()
    W = skew(omega)
    W2 = W @ W
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)
    a = torch.sinc(angle/math.pi)
    b = .5*torch.sinc(angle/(2*math.pi)).square()
    c = torch.where(square < 1e-6, 1/6-square/120+square.square()/5040,
                    (1-a)/square.clamp_min(1e-12))
    return I+a*W+b*W2, (I+b*W+c*W2) @ nu


def sample_read_jitter(device, *, dtype=torch.float32):
    axis = F.normalize(torch.randn(3, device=device, dtype=dtype), dim=0)
    omega = axis*torch.deg2rad(torch.empty((), device=device, dtype=dtype).uniform_(2., 3.))
    direction = F.normalize(torch.randn(3, device=device, dtype=dtype), dim=0)
    translation = direction*torch.empty((), device=device, dtype=dtype).uniform_(.05, .06)
    R, _ = local_se3_exp(omega, translation)
    return torch.cat((torch.cat((R, translation[:, None]), -1), translation.new_tensor([[0., 0., 0., 1.]])), 0)


def posterior_ellipsoids(inlier, memory, *, sigma_floor=.03):
    means = inlier @ memory.means
    delta = memory.means[None]-means[:, None]
    covariance = torch.einsum('qm,mij->qij', inlier, memory.covars)
    covariance = covariance+torch.einsum('qm,qmi,qmj->qij', inlier, delta, delta)
    eye = torch.eye(3, device=means.device, dtype=means.dtype)
    return means, .5*(covariance+covariance.transpose(-1, -2))+sigma_floor**2*eye


def anisotropic_ray_pose(points, rays, anchors, covars, weights, *, iterations=4,
                          kappa=3., radial_precision=4., damping=.01,
                          return_diagnostic=False, metric_scale=False):
    # Import locally: the legacy initialization remains available for restoring
    # older project runs, while the v2l update never uses Gram-Schmidt again.
    from src.models.spatial_pose import point_irls_pose
    dtype, output_device = anchors.dtype, anchors.device
    # These Qx3/6 systems are tiny. CPU float64 avoids hundreds of small CUDA
    # launches per unrolled solve; differentiable copies reconnect every input
    # to its GPU writer / matching graph. VGGT and rasterization stay on GPU.
    points, rays, anchors, covars, weights = [x.to(device='cpu', dtype=torch.float64)
                                            for x in (points, rays, anchors, covars, weights)]
    rays = F.normalize(rays, dim=-1)
    precision = torch.linalg.inv(covars)
    scatter_scale = points.new_ones(())
    if metric_scale:
        normalized = weights/weights.sum().clamp_min(1e-8)
        def scatter(value):
            centered = value-(normalized[:, None]*value).sum(0)
            return (normalized*centered.square().sum(-1)).sum()
        scatter_scale = (scatter(anchors)/scatter(points).clamp_min(1e-6)).clamp_min(1e-12).sqrt().clamp(.25, 4.)
        points = scatter_scale*points
    radial = points.norm(dim=-1)
    alpha = .04/(radial+.5).square() if metric_scale else torch.full_like(radial, radial_precision)
    initial, _ = point_irls_pose(points, anchors, weights, iterations=0)
    R, t = initial[:3, :3], initial[:3, 3]
    radial = points.norm(dim=-1)
    eye3 = torch.eye(3, device=points.device, dtype=points.dtype)

    def state(rotation, center):
        direction = rays @ rotation.T
        delta = anchors-center
        Ad = torch.einsum('qij,qj->qi', precision, direction)
        depth = ((Ad*delta).sum(-1)+alpha*radial)/((Ad*direction).sum(-1)+alpha)
        depth = depth.clamp_min(.05)
        residual = center+depth[:, None]*direction-anchors
        m2 = torch.einsum('qi,qij,qj->q', residual, precision, residual)
        m2 = m2+alpha*(depth-radial).square()
        m2 = m2.clamp_min(0)
        energy = (weights*kappa**2*torch.log1p(m2/kappa**2)).sum()
        return depth, residual, m2, energy

    # One anisotropic center initialization, regularized to a transported
    # centroid prior. Accept only when the SAME robust energy does not rise.
    direction = rays @ R.T
    Ad = torch.einsum('qij,qj->qi', precision, direction)
    denominator = (Ad*direction).sum(-1)+alpha
    W = precision-Ad[:, :, None]*Ad[:, None, :]/denominator[:, None, None]
    A = (weights[:, None, None]*W).sum(0)
    b = (weights[:, None]*(torch.einsum('qij,qj->qi', W, anchors)
                          -alpha[:, None]*radial[:, None]*Ad/denominator[:, None])).sum(0)
    regularizer = 1e-4*A.trace()/3+1e-8
    candidate = torch.linalg.solve(A+regularizer*eye3, b+regularizer*t)
    old_energy = state(R, t)[-1]
    candidate_energy = state(R, candidate)[-1]
    t = torch.where(torch.isfinite(candidate_energy) & (candidate_energy <= old_energy), candidate, t)
    history = [state(R, t)[-1].detach()]
    scales = []
    for _ in range(iterations):
        depth, residual, m2, energy = state(R, t)
        psi = kappa**2/(kappa**2+m2)
        robust = weights*psi
        local_precision = R.T[None] @ precision @ R[None]
        B = local_precision+alpha[:, None, None]*rays[:, :, None]*rays[:, None, :]
        r_loc = residual @ R
        J = torch.cat((-skew(depth[:, None]*rays), eye3.expand(len(points), -1, -1)), -1)
        # Schur-eliminate the same nuisance ray depth in the normal equations.
        # r^T B r is its positive scalar block; using the full 3D point Hessian
        # after depth elimination would incorrectly resist free radial motion.
        Ar = torch.einsum('qij,qj->qi', local_precision, rays)
        depth_block = torch.einsum('qi,qij,qj->q', rays, B, rays)
        reduced_precision = local_precision-Ar[:, :, None]*Ar[:, None, :]/depth_block[:, None, None]
        reduced_precision = torch.where((depth > .05000001)[:, None, None], reduced_precision, local_precision)
        H = torch.einsum('q,qai,qab,qbj->ij', robust, J, reduced_precision, J)
        # Envelope gradient: ds/dpose cancels at the optimal ray depth. The
        # radial prior is already in m2 and s; adding alpha*r*r^T to this
        # gradient would double-count it. B is a positive LM approximation.
        gradient = torch.einsum('q,qai,qab,qb->i', robust, J, local_precision, r_loc)
        regularizer6 = damping*H.diagonal().clamp_min(1e-6)+1e-8
        increment = -torch.linalg.solve(H+torch.diag(regularizer6), gradient)
        # Acceptance is discrete either way; only the accepted branch needs an
        # autograd graph. Rejected trials must not inflate backward computation.
        selected_scale = 0.
        with torch.no_grad():
            for scale in (1., .5, .25):
                delta_R, delta_t = local_se3_exp(scale*increment[:3], scale*increment[3:])
                trial_energy = state(R @ delta_R, t+R @ delta_t)[-1]
                if torch.isfinite(trial_energy) and trial_energy <= energy:
                    selected_scale = scale
                    break
        delta_R, delta_t = local_se3_exp(selected_scale*increment[:3], selected_scale*increment[3:])
        t, R = t+R @ delta_t, R @ delta_R
        history.append(state(R, t)[-1].detach())
        scales.append(points.new_tensor(selected_scale))
    depth, _, m2, final_energy = state(R, t)
    robust = weights*kappa**2/(kappa**2+m2)
    T = torch.cat((torch.cat((R, t[:, None]), -1), points.new_tensor([[0., 0., 0., 1.]])), 0).to(device=output_device, dtype=dtype)
    if return_diagnostic:
        return T, robust.to(device=output_device, dtype=dtype), dict(energy_history=torch.stack(history),
            accepted_scales=torch.stack(scales) if scales else points.new_empty(0),
            mahalanobis2=m2, ray_depth=depth, scatter_scale=scatter_scale.detach(), radial_precision=alpha.detach())
    return T, robust.to(device=output_device, dtype=dtype)
