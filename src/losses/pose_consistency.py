"""Pose-only robustness under random missing map regions; no geometry target.

Teacher poses are detached predictions from the same model and full memory.
Query GT is used solely as ordinary pose supervision in this criterion.
"""
import torch
from torch.nn import functional as F
from src.losses.world import rotation_geodesic, open_set_matching_loss


def pose_consistency_loss(partial, teacher, target, *, erased_reference=None, matching_weight=1.):
    matching = partial['T0'].sum()*0
    if 'matching_attention' in partial:
        matching, _, _, observable = open_set_matching_loss(partial, target, erased_reference)
        if 'refined_matching_attention' in partial:
            refined = dict(partial, matching_attention=partial['refined_matching_attention'], anchors=partial['refined_anchors'])
            matching = matching+open_set_matching_loss(refined, target, erased_reference)[0]
        if not observable:
            return matching_weight*matching, dict(counterfactual_pose_loss=0., counterfactual_matching_loss=float(matching.detach()))
    supervised, consistent = [], []
    for key in ('T0', 'T1'):
        T, reference = partial[key], teacher[key].detach()
        for destination, container in ((target.c2w, supervised), (reference, consistent)):
            r = rotation_geodesic(T[:3, :3], destination[:3, :3])
            t = (T[:3, 3]-destination[:3, 3]).norm()
            container.append(F.smooth_l1_loss(r, r.new_zeros(()), beta=.03)+F.smooth_l1_loss(t, t.new_zeros(()), beta=.03))
    loss = .5*sum(supervised)+2.*sum(consistent)+matching_weight*matching
    return loss, dict(counterfactual_pose_loss=float(loss.detach()),
                     counterfactual_t0_shift_m=float((partial['T0'][:3, 3]-teacher['T0'][:3, 3].detach()).detach().norm()),
                     counterfactual_t1_shift_m=float((partial['T1'][:3, 3]-teacher['T1'][:3, 3].detach()).detach().norm()))
