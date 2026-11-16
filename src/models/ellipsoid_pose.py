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


def sample_read_jitter(device, *, dtype=torch.float32, include_zero=False):
    axis = F.normalize(torch.randn(3, device=device, dtype=dtype), dim=0)
    omega = axis*torch.deg2rad(torch.empty((), device=device, dtype=dtype).uniform_(0. if include_zero else 2., 3.))
    direction = F.normalize(torch.randn(3, device=device, dtype=dtype), dim=0)
    translation = direction*torch.empty((), device=device, dtype=dtype).uniform_(0. if include_zero else .05, .06)
    R, _ = local_se3_exp(omega, translation)
    return torch.cat((torch.cat((R, translation[:, None]), -1), translation.new_tensor([[0., 0., 0., 1.]])), 0)


def posterior_ellipsoids(inlier, memory, *, sigma_floor=.03):
    means = inlier @ memory.means
    delta = memory.means[None]-means[:, None]
    covariance = torch.einsum('qm,mij->qij', inlier, memory.covars)
    covariance = covariance+torch.einsum('qm,qmi,qmj->qij', inlier, delta, delta)
    eye = torch.eye(3, device=means.device, dtype=means.dtype)
    return means, .5*(covariance+covariance.transpose(-1, -2))+sigma_floor**2*eye


def symmetric_polar_rotation(matrix, iterations=6):
    """Three-axis right-handed completion, then matrix-only polar projection.

    One common column scale preserves a symmetric positive camera scatter:
    normalizing each column independently would introduce a new shear bias.
    All operations commute with left multiplication by a world rotation.
    """
    eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    scale = matrix.norm(dim=0).max().clamp_min(1e-12)
    value = matrix/scale
    a, b, c = value.unbind(-1)
    cofactor = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a), torch.linalg.cross(a, b)), -1)
    completed = value+cofactor
    # Exact reflections have a non-unique closest proper rotation. The
    # cofactor provides an equivariant right-handed numerical fallback.
    completed = torch.where(completed.norm() < 1e-10, cofactor, completed)
    rotation = completed/completed.norm().clamp_min(1e-12)
    for _ in range(iterations):
        rotation = .5*rotation @ (3*eye-rotation.T @ rotation)
    # Rare highly elongated supports need additional numerical convergence,
    # never extra data/GT or a different pose objective.
    for _ in range(6):
        if float((rotation.T@rotation-eye).detach().abs().max()) < 1e-7:
            break
        rotation = .5*rotation @ (3*eye-rotation.T @ rotation)
    return torch.where(completed.norm() > 1e-10, rotation, eye)


def planar_safe_initial(points, anchors, weights, *, whiten=True, iterations=2):
    """Robust scatter scale + damped centered moments; camera-to-world pose."""
    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    votes = weights
    for step in range(iterations+1):
        normalized = votes/votes.sum().clamp_min(1e-8)
        Pbar = (normalized[:, None]*points).sum(0)
        Xbar = (normalized[:, None]*anchors).sum(0)
        dP, dX = points-Pbar, anchors-Xbar
        var_P = (normalized*dP.square().sum(-1)).sum().clamp_min(1e-6)
        var_X = (normalized*dX.square().sum(-1)).sum()
        scale = (var_X/var_P).clamp_min(1e-12).sqrt().clamp(.25, 4.)
        scaled = scale*points
        dP = scale*dP
        H = (normalized[:, None]*dX).T @ dP
        C = (normalized[:, None]*dP).T @ dP
        lam = .10*C.trace()/3+1e-6
        matrix = torch.linalg.solve(C+lam*eye, H.T).T if whiten else H
        R = symmetric_polar_rotation(matrix)
        t = Xbar-R@(scale*Pbar)
        if step < iterations:
            error2 = (anchors-(scaled@R.T+t)).square().sum(-1)
            votes = weights*.25**2/(.25**2+error2)
    return R, t, scale, votes


def anisotropic_ray_pose(points, rays, anchors, covars, weights, *, iterations=4,
                          kappa=3., radial_precision=4., damping=.01,
                          return_diagnostic=False, metric_scale=False, planar_safe=False, init_pose=None):
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
    planar_candidate = None
    if init_pose is not None:
        init_pose = init_pose.to(device="cpu", dtype=torch.float64)
        # Refinement starts at the supplied prediction. No moment/center reset.
        planar_safe = False
        metric_scale = False
        direction = rays @ init_pose[:3, :3].T
        radial_reference = ((anchors-init_pose[:3, 3])*direction).sum(-1).clamp_min(.05)
        points = rays*radial_reference[:, None]
    if planar_safe:
        Rwhite, twhite, scatter_scale, initial_votes = planar_safe_initial(points, anchors, weights)
        Rplain, tplain, _, _ = planar_safe_initial(points, anchors, weights, whiten=False)
        planar_candidate = (Rplain, tplain)
        points = scatter_scale*points
    scatter_scale = points.new_ones(()) if not planar_safe else scatter_scale
    if metric_scale and not planar_safe:
        normalized = weights/weights.sum().clamp_min(1e-8)
        def scatter(value):
            centered = value-(normalized[:, None]*value).sum(0)
            return (normalized*centered.square().sum(-1)).sum()
        scatter_scale = (scatter(anchors)/scatter(points).clamp_min(1e-6)).clamp_min(1e-12).sqrt().clamp(.25, 4.)
        points = scatter_scale*points
    radial = points.norm(dim=-1)
    alpha = .04/(radial+.5).square() if metric_scale or init_pose is not None else torch.full_like(radial, radial_precision)
    if init_pose is not None:
        R, t = init_pose[:3, :3], init_pose[:3, 3]
    elif planar_safe:
        R, t = Rwhite, twhite
    else:
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

    if planar_candidate is not None:
        plain_R, plain_t = planar_candidate
        plain_energy = state(plain_R, plain_t)[-1]
        proper = ((plain_R.T@plain_R-eye3).abs().max() < 1e-5) & (torch.linalg.det(plain_R) > .999)
        choose_plain = proper & torch.isfinite(plain_energy) & (plain_energy < state(R, t)[-1])
        R = torch.where(choose_plain, plain_R, R)
        t = torch.where(choose_plain, plain_t, t)

    if init_pose is None:
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
