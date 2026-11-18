"""Prediction-only tracking and rigid seed proposals for streaming v4."""
from functools import lru_cache
from itertools import combinations
import math

import torch
from torch.nn import functional as F

from src.geometry import invert_se3
from src.models.ellipsoid_pose import anisotropic_ray_pose, posterior_ellipsoids
from src.models.spatial_pose import rotation_6d
from src.models.stream_tracking import local_motion


def open_attention(logits, dustbin, *, mutual=False, memory_mask=None, pair_mask=None):
    """Column competition lowers ambiguous matches without competing away the dustbin."""
    if mutual and logits.shape[1]:
        # Equal claims by many patches each get 1/Q. Up to eight well
        # separated query claims can survive without changing their logits.
        reverse = (logits.log_softmax(dim=0)+math.log(8.)).clamp(max=0.)
        logits = logits+reverse
    if memory_mask is not None:
        logits = logits.masked_fill(~memory_mask[None], -80.)
    if pair_mask is not None:
        logits = logits.masked_fill(~pair_mask, -80.)
    return torch.cat((logits, dustbin), dim=-1).softmax(dim=-1)


def bounded_motion(previous, older):
    """Damped camera-local extrapolation; relocalization jumps get no momentum."""
    if older is None:
        return previous
    shift, angle = local_motion(previous, older)
    if shift > .55 or angle > 25.:
        return previous
    delta = invert_se3(older) @ previous
    eye = torch.eye(4, device=previous.device, dtype=previous.dtype)
    damped = eye.lerp(delta,.5)
    damped[:3,:3] = rotation_6d(damped[:3,:2].T.flatten())
    return previous @ damped


def reachable_penalty(means, rays, predicted):
    """A zero-cost angular plateau with smooth tails and soft range limits."""
    relative = (means-predicted[:3,3]) @ predicted[:3,:3]
    distance = relative.norm(dim=-1)
    cosine = F.normalize(rays,dim=-1) @ F.normalize(relative,dim=-1).T
    # Difference in cosine avoids acos singularities. At 25 degrees the
    # quadratic tail joins the flat region with zero derivative.
    angle_tail = (math.cos(math.radians(25))-cosine).clamp_min(0)
    behind = (.05-relative[:,2]).clamp_min(0)/.25
    far = (distance-9.).clamp_min(0)/2.
    return -(angle_tail.square()/.08**2+behind.square()+far.square()).clamp_max(32.)


@torch.no_grad()
def track_membership(logits, means, rays, predicted):
    """Use the motion prior only for the shortlist, never for LM or deep matching weights."""
    penalty = reachable_penalty(means,rays,predicted)
    top = (logits+penalty).topk(min(48,logits.shape[-1]),dim=-1).indices
    ranking = torch.zeros_like(logits,dtype=torch.bool).scatter_(1,top,True)
    return ranking & (penalty > -6.)


@lru_cache(maxsize=48)
def triple_indices(n):
    return torch.tensor(list(combinations(range(n),3)),dtype=torch.int64)


def spatial_triangles(points, centers, weight, *, limit=24):
    """Generate right-handed poses from compatible triangles of unique pairs."""
    if len(points) < 3:
        return points.new_empty((0,4,4))
    triples = triple_indices(len(points)).to(points.device)
    p, x = points[triples], centers[triples]

    def legs(tri):
        return torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],tri[:,2]-tri[:,1]),dim=1)

    u, v = legs(p), legs(x)
    up, vp = u.norm(dim=-1), v.norm(dim=-1)
    scales = (vp.clamp_min(1e-4)/up.clamp_min(1e-4)).log()
    spread = scales.max(dim=-1).values-scales.min(dim=-1).values
    s = scales.mean(dim=-1).exp()
    pa = torch.linalg.cross(u[:,0],u[:,1]).norm(dim=-1)
    xa = torch.linalg.cross(v[:,0],v[:,1]).norm(dim=-1)
    compatible = (spread < .65) & (s >= .25) & (s <= 4.)
    compatible &= (up.min(dim=-1).values > .10) & (vp.min(dim=-1).values > .10)
    compatible &= (pa > .10*up[:,0]*up[:,1]) & (xa > .10*vp[:,0]*vp[:,1])
    if not compatible.any():
        return points.new_empty((0,4,4))
    compact = compatible.nonzero().flatten()
    scores = (weight[triples].clamp_min(1e-6).log().mean(dim=-1)-3*spread
              +.18*xa.clamp_min(.02).log())
    # Keep stream-wide variety: take the best with distinct leading pairs,
    # then fill any spare slots with strong alternatives.
    top = compact[scores[compact].topk(min(8*limit,len(compact))).indices]
    pairs = triples[top,:2].cpu().tolist()
    seen, first, extra = set(), [], []
    for i,pair in enumerate(pairs):
        key = tuple(pair)
        (first if key not in seen else extra).append(i)
        seen.add(key)
    index = top[torch.as_tensor((first+extra)[:limit],device=points.device)]

    def basis(legs_at_index):
        a = F.normalize(legs_at_index[:,0],dim=-1)
        c = F.normalize(torch.linalg.cross(a,legs_at_index[:,1]),dim=-1)
        return torch.stack((a,torch.linalg.cross(c,a),c),dim=-1)

    R = basis(v[index]) @ basis(u[index]).transpose(-1,-2)
    t = x[index].mean(dim=1)-(R @ (s[index,None]*p[index].mean(dim=1))[...,None])[...,0]
    poses = torch.eye(4,device=points.device,dtype=points.dtype).repeat(len(index),1,1)
    poses[:,:3,:3], poses[:,:3,3] = R, t
    return poses


def ray_distances(poses, rays, anchors, precision):
    """Positive-depth ray-to-ellipsoid residual for every query under every pose."""
    direction = rays @ poses[:,:3,:3].transpose(-1,-2)
    offset = anchors[None]-poses[:,None,:3,3]
    Ad = torch.einsum('qij,nqj->nqi',precision,direction)
    depth = ((Ad*offset).sum(-1)/(Ad*direction).sum(-1).clamp_min(1e-8)).clamp_min(.05)
    residual = depth[...,None]*direction-offset
    return torch.einsum('nqi,qij,nqj->nq',residual,precision,residual).clamp_min(0)


@torch.no_grad()
def correspondence_triangles(W, points, memory, mask, *, limit=24):
    selected = torch.where(mask)[0]
    probabilities = W[:,:-1][:,selected]
    max_prob, best = probabilities.max(dim=1)
    reverse = probabilities.argmax(dim=0)
    query_ids = torch.arange(len(W),device=W.device)
    reciprocal = query_ids == reverse[best]
    likely = reciprocal & (max_prob > .02)
    top = query_ids[likely]
    top = top[max_prob[top].topk(min(48,len(top))).indices]
    return spatial_triangles(points[top],memory.means[selected[best[top]]],max_prob[top],limit=limit)


def precision_matrix(covars):
    return torch.linalg.inv(covars+.03**2*torch.eye(3,device=covars.device,dtype=covars.dtype))


def render_evidence(match_p, m2, render, features):
    visible = (render.opacity.flatten() > .20) & (render.depth.flatten() > 0)
    effective = match_p >= .50
    votes = visible & effective & (m2 <= 4.)
    cosine = F.cosine_similarity(render.features.reshape_as(features),features,dim=-1)
    count, coverage, effective_count = int(votes.sum()), int(visible.sum()), int(effective.sum())
    return dict(inliers=count, visible=coverage, effective_matches=effective_count,
                inlier_fraction=count/max(coverage,1), concentration=count/max(effective_count,1),
                cosine=float((cosine*votes).sum()/max(count,1)))


def seed_fit(points, rays, W, c_depth, memory, seed_lock, *, appearance):
    """Bounded proposal search followed by one differentiable four-step LM."""
    p = 1-W[:,-1]
    weights = p*c_depth
    hard_index = W[:,:-1].argmax(dim=-1)
    hard_means = memory.means[hard_index]
    hard_covars = memory.covars[hard_index]
    with torch.no_grad():
        inlier = W[:,:-1][:,seed_lock]/p[:,None].clamp_min(1e-6)
        posterior, covars = posterior_ellipsoids(inlier,memory.subset(seed_lock))
        moment, _ = anisotropic_ray_pose(points,rays,posterior,covars,weights,
            iterations=0,metric_scale=True,planar_safe=True)
        hypotheses = correspondence_triangles(W,points,memory,seed_lock)
        poses = torch.cat((moment[None],hypotheses))
        precision = precision_matrix(hard_covars)
        residuals = ray_distances(poses,rays,hard_means,precision)
        # The score uses all query rays with robust influence. Only the four
        # most plausible geometric poses pay for an O5 verification render.
        weighted_inliers = (weights*(1-residuals/9.).clamp_min(0)).sum(-1)/weights.sum().clamp_min(1e-8)
        top = weighted_inliers.topk(min(4,len(poses))).indices
        best, best_score = top[0], -float('inf')
        for idx in top:
            evidence = appearance(poses[idx],p,residuals[idx])
            score = float(weighted_inliers[idx])+.16*evidence['cosine']*min(evidence['inliers']/16.,1.)
            if score > best_score:
                best, best_score = idx, score
        initial = poses[best]
        selected = residuals[best]
        # Geometry at the selected pose determines the inlier set before LM.
        # Soft mass prevents geometric outliers swamping a small common FOV.
        reliability = (1-selected/12.).clamp_min(0).square()
    fit, votes, diagnostics = anisotropic_ray_pose(points,rays,hard_means,hard_covars,weights*reliability,
                                    init_pose=initial,return_diagnostic=True)
    return fit, hard_means, hard_covars, votes, diagnostics, len(poses)


def select_stream_pose_v4(model, tokens, shallow, features, memory, Kp, hp, wp, previous, older, seed_lock):
    bootstrap = model.initial_pose
    if not len(memory):
        pose, match = bootstrap(tokens,features,memory,Kp,hp,wp,shallow=shallow)
        return previous, dict(match,stream_pose_source='global',stream_pose_evidence={},landmark_selected=False)
    points,z,c_depth,rays = bootstrap.camera_observation(tokens,shallow,Kp,hp,wp)
    logits,dustbin,q,k = bootstrap.matching_logits(tokens,shallow,features,memory,rays)
    temperature = bootstrap.log_temperature.exp().clamp(max=60.)
    evidence = {}
    seed_aux = None

    def auxiliary(W):
        p = 1-W[:,-1]
        conditional = W[:,:-1]/p[:,None].clamp_min(1e-6)
        return dict(matching_attention=W,anchors=conditional @ memory.means[seed_lock],
                    anchor_support_means=memory.means[seed_lock],anchor_support_covars=memory.covars[seed_lock],
                    patch_hw=(hp,wp))

    def joint(source, W, fit, anchors, covars, votes, lm, *, mask=None, pairs=None):
        p = 1-W[:,-1]
        return fit, dict(anchors=anchors,confidence=p,camera_patch_points=points,z_cam=z,c_depth=c_depth,
            matching_attention=W,dustbin_probability=W[:,-1],
            bootstrap_attention=W[:,:-1]/p[:,None].clamp_min(1e-6),
            bootstrap_support=torch.ones(len(memory),device=z.device,dtype=torch.bool) if mask is None else mask,
            anchor_votes=votes,match_mass=p.sum(),matching_query=q,matching_keys=k,
            matching_dustbin_logit=dustbin,matching_temperature=temperature,camera_rays=rays,
            lm_energy_history=lm['energy_history'],lm_accepted_scales=lm['accepted_scales'],
            matching_mask=mask,matching_pair_mask=pairs,matching_column_competition=source!='track',
            stream_pose_source=source,stream_pose_evidence=evidence,landmark_selected=source=='seed',
            seed_match_auxiliary=seed_aux)

    if seed_lock is not None and int(seed_lock.sum()) >= 3:
        W = open_attention(logits,dustbin,mutual=True,memory_mask=seed_lock)
        if model.training:
            # Every eligible forward emits the same prediction; only criterion
            # decides whether the training target physically overlaps the seed.
            seed_aux = auxiliary(torch.cat((W[:,:-1][:,seed_lock],W[:,-1:]),dim=-1))
        if int(seed_lock.sum()) >= 16:
            seed_map = memory.subset(seed_lock)

            def appearance(pose, p, m2):
                rendered = model.render_reader.render(seed_map,pose,Kp,hp,wp)
                return render_evidence(p,m2,rendered,features)

            fit, anchors,covars,votes,lm,count = seed_fit(points,rays,W,c_depth,memory,seed_lock,appearance=appearance)
            with torch.no_grad():
                m2 = ray_distances(fit[None],rays,anchors,precision_matrix(covars))[0]
                s = appearance(fit,1-W[:,-1],m2)
                shift,angle = local_motion(fit,previous)
                evidence['seed'] = dict(s,hypotheses=count,translation_m=shift,rotation_deg=angle)
                if not model.training:
                    evidence['seed']['candidate_c2w'] = fit.cpu().tolist()
            if s['inliers'] >= 16 and s['concentration'] >= .40 and s['cosine'] >= .74:
                return joint('seed',W,fit,anchors,covars,votes,lm,mask=seed_lock)
            del fit,anchors,covars,votes,lm,W

    def fit_posterior(W, initial=None):
        p = 1-W[:,-1]
        anchors,covars = posterior_ellipsoids(W[:,:-1]/p[:,None].clamp_min(1e-6),memory)
        fit,votes,lm = anisotropic_ray_pose(points,rays,anchors,covars,p*c_depth,init_pose=initial,
                                           planar_safe=True,metric_scale=True,return_diagnostic=True)
        return fit,anchors,covars,votes,lm

    with torch.no_grad():
        predicted = bounded_motion(previous,older)
        pairs = track_membership(logits,memory.means,rays,predicted)
    W = open_attention(logits,dustbin,pair_mask=pairs)
    fit,anchors,covars,votes,lm = fit_posterior(W,predicted)
    with torch.no_grad():
        m2 = ray_distances(fit[None],rays,anchors,precision_matrix(covars))[0]
        rendered = model.render_reader.render(memory,fit,Kp,hp,wp)
        t = render_evidence(1-W[:,-1],m2,rendered,features)
        shift,angle = local_motion(fit,previous)
        evidence['track'] = dict(t,translation_m=shift,rotation_deg=angle)
    if t['inliers'] >= 16 and t['concentration'] >= .30 and t['cosine'] >= .68 and shift <= .65 and angle <= 25.:
        return joint('track',W,fit,anchors,covars,votes,lm,pairs=pairs)

    W = open_attention(logits,dustbin,mutual=True)
    if float((1-W[:,-1]).sum().detach()) < .25:
        # In weakly observable frames do not invent a world-axis origin reset.
        init = previous
    else:
        init = None
    return joint('global',W,*fit_posterior(W,init))
