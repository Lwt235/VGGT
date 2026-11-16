#!/usr/bin/env python3
"""Read-only checkpoint evaluation: far-range world and camera depth errors."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage, PROJECT_ROOT, load_config, load_checkpoint_readonly
configure_local_storage()
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.step2_data import EpisodeDataset
from src.models.vggt_memory import VGGTSpatialMemory
from src.vis.step2 import dump_json
from src.losses.world import pose_errors
from src.geometry import patch_intrinsics, world_rays, ray_ellipsoid_distance


@torch.no_grad()
def bootstrap_case_diagnostic(out,query,target):
    """Post-forward physics only. GT candidate is never returned to the model."""
    from src.models.ellipsoid_pose import planar_safe_initial
    points=out['camera_patch_points'].double().cpu()
    anchors=out['anchors'].double().cpu()
    covars=out['posterior_covars'].double().cpu()
    weights=((1-out['dustbin_probability'])*out['c_depth']).double().cpu()
    Rw,tw,scale,votes=planar_safe_initial(points,anchors,weights)
    Rp,tp,_,_=planar_safe_initial(points,anchors,weights,whiten=False)
    h,w=out['rendered'].depth.shape[:2]
    _,rays,_=world_rays(patch_intrinsics(query.K).double().cpu(),torch.eye(4,dtype=torch.float64),h,w)
    radial=(points*scale).norm(dim=-1);alpha=.04/(radial+.5).square()
    precision=torch.linalg.inv(covars)
    def energy(pose):
        R,t=pose[:3,:3],pose[:3,3]
        directions=rays@R.T;delta=anchors-t
        Ad=torch.einsum('qij,qj->qi',precision,directions)
        depth=((Ad*delta).sum(-1)+alpha*radial)/((Ad*directions).sum(-1)+alpha)
        depth=depth.clamp_min(.05)
        residual=t+depth[:,None]*directions-anchors
        m2=torch.einsum('qi,qij,qj->q',residual,precision,residual)+alpha*(depth-radial).square()
        return float((weights*9*torch.log1p(m2.clamp_min(0)/9)).sum())
    candidates={}
    for name,R,t in [('white',Rw,tw),('plain',Rp,tp)]:
        pose=torch.eye(4,dtype=points.dtype);pose[:3,:3]=R;pose[:3,3]=t;candidates[name]=pose
    candidates.update(final_T0=out['T0'].double().cpu(),gt_diagnostic_only=target.c2w.double().cpu())
    rows={}
    for name,pose in candidates.items():
        rotation,translation=pose_errors(pose,target.c2w.double().cpu())
        rows[name]=dict(rotation_error_deg=float(rotation),translation_error_m=float(translation),energy=energy(pose),
                        orthogonality_error=float((pose[:3,:3].T@pose[:3,:3]-torch.eye(3)).abs().max()))
    return dict(candidates=rows,robust_scatter_scale=float(scale),effective_initial_votes=float(votes.sum().square()/votes.square().sum().clamp_min(1e-12)))


@torch.no_grad()
def evaluate_run(name, step, *, case_folder=None):
    folder = PROJECT_ROOT/'results'/name
    config = yaml.safe_load((folder/'config.yaml').read_text())
    runtime = load_config()
    state = load_checkpoint_readonly(folder/f'training_state_{step:05d}.pt')
    if state['step'] != step or state['config'] != config:
        raise ValueError('Checkpoint/config mismatch')
    model = VGGTSpatialMemory(runtime['assets']['official_vggt_source']['path'], runtime['assets']['vggt_weights']['path'],
                              revision=config['architecture_revision'], receive_layer=config['receive_layer'], pose_jitter_probability=0.).cuda().eval()
    parameters = dict(model.named_parameters())
    for key, value in state['trainable'].items(): parameters[key].copy_(value)
    del state
    dataset = EpisodeDataset(PROJECT_ROOT/config['split'], config['image_hw'])
    specs = [s for s in dataset.evaluation_episodes() if s['scene'].startswith('harvard_robotics_lab/')]
    buffers = {label:[] for label in ('near_0.1_3m', 'mid_3_5m', 'far_5_10m')}
    scatter, poses, attention_stats, calibration_stats, case_vis, refinement_lm = [], [], [], [], [], []
    for spec in specs:
        episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device='cuda')
        query, target = episode['query']
        memory = model.build_memory(episode['mapping'])
        out = model(query, memory, return_attention=True)
        if 'local_geometry_confidence' in out and 'rendered_T1' in out:
            visible=out['rendered_T1'].opacity[...,0]>.1
            confidence=out['local_geometry_confidence'][...,0]
            from torch.nn import functional as F
            rendered=out['rendered_T1'];hp,wp=rendered.depth.shape[:2]
            fraction=F.adaptive_avg_pool2d(target.valid.float()[None,None],(hp,wp))[0,0]
            truth_patch=F.adaptive_avg_pool2d((target.depth*target.valid)[None,None],(hp,wp))[0,0]/fraction.clamp_min(1e-6)
            native_patch=F.adaptive_avg_pool2d(out['raw_camera_points'][...,2][None,None],(hp,wp))[0,0]
            good=(fraction>.25)&visible
            rel_var=rendered.depth_variance[...,0]/rendered.depth[...,0].square().clamp_min(.04)
            modulation=.65+.35*confidence if model.render_reader.physical_local else confidence
            effective=modulation*rendered.opacity[...,0]/(1+rel_var)
            calibration_stats.append(dict(query=spec['query_frame_id'],gamma=float(out['camera_scale']) if 'camera_scale' in out else None,confidence_mean=float(confidence.mean()),
                confidence_visible=float(confidence[visible].mean()) if visible.any() else None,visible_fraction=float(visible.float().mean()),
                visible_relative_variance=float(rel_var[good].mean()) if good.any() else None,
                effective_visible_correction_weight=float(effective[good].mean()) if good.any() else None,
                rendered_visible_depth_l1_m=float((rendered.depth[...,0][good]-truth_patch[good]).abs().mean()) if good.any() else None,
                native_visible_patch_depth_l1_m=float((native_patch[good]-truth_patch[good]).abs().mean()) if good.any() else None))
        if 'refinement_lm_energy_history' in out:
            W1=out['refined_matching_attention']
            refinement_lm.append(dict(query=spec['query_frame_id'], energy_history=out['refinement_lm_energy_history'].cpu().tolist(),
                accepted_scales=out['refinement_lm_accepted_scales'].cpu().tolist(),
                matching_normalized_entropy=float(-(W1*W1.clamp_min(1e-9).log()).sum(-1).mean()/np.log(max(2,W1.shape[-1]))),
                dustbin_mean=float(W1[:,-1].mean())))
        info = out['attention']
        weights = info['attention']
        if weights is not None:
            selected = memory.subset(info['indices'])
            origins, directions, _ = world_rays(patch_intrinsics(query.K), out['T0'], *out['rendered'].depth.shape[:2])
            rho2, _ = ray_ellipsoid_distance(origins, directions, selected.means, selected.covars)
            near_ray = (rho2 <= 9).float()
            entropy = -(weights*weights.clamp_min(1e-9).log()).sum(-1)/np.log(max(2,len(selected)))
            attention_stats.append(dict(query=spec['query_frame_id'], normalized_entropy=float(entropy.mean()),
                ray_3sigma_attention_mass=float((weights*near_ray[None]).sum(-1).mean()), uniform_ray_3sigma_mass=float(near_ray.mean())))
        # GT depth/pose only enter evaluation after forward has finished.
        raw = out['raw_camera_points'][..., 2]
        anchored = out['camera_points'][..., 2]
        world_error = (out['world_points']-target.world_points).norm(dim=-1)
        from src.geometry import pixel_grid, transform_points, invert_se3
        grid=pixel_grid(*target.depth.shape,device=query.K.device,dtype=query.K.dtype,center=0.)
        calibrated_rays=grid@torch.linalg.inv(query.K).T
        truth_camera=transform_points(target.world_points,invert_se3(target.c2w))
        native_offray=(out['raw_camera_points']-calibrated_rays*raw[...,None]).norm(dim=-1)
        final_offray=(out['camera_points']-calibrated_rays*anchored[...,None]).norm(dim=-1)
        native_camera_error=(out['raw_camera_points']-truth_camera).norm(dim=-1)
        final_camera_error=(out['camera_points']-truth_camera).norm(dim=-1)
        for label, (lo, hi) in zip(buffers, ((.1,3.),(3.,5.),(5.,10.))):
            mask = target.valid & (target.depth >= lo) & (target.depth < hi)
            if mask.any():
                values = torch.stack((world_error[mask], (raw-target.depth)[mask], (anchored-target.depth)[mask],native_offray[mask],final_offray[mask],native_camera_error[mask],final_camera_error[mask]), -1).cpu().numpy()
                buffers[label].append(values)
        mask = target.valid.flatten().nonzero().flatten()[::40]
        scatter.append(torch.stack((target.depth.flatten()[mask], raw.flatten()[mask], anchored.flatten()[mask]), -1).cpu().numpy())
        r0,t0 = pose_errors(out['T0'],target.c2w);r1,t1 = pose_errors(out['T1'],target.c2w)
        poses.append(dict(query=spec['query_frame_id'],rotation0_deg=float(r0),rotation1_deg=float(r1),translation0_m=float(t0),translation1_m=float(t1)))
        if case_folder is not None and spec['query_frame_id']==362:
            from scripts.train_step2 import intervention_box
            from src.vis.step2 import unseen_visualizations
            empty=model(query,memory.subset(slice(0,0)))
            box=intervention_box(memory,target)
            partial,mask=memory.erase_box(*box)
            erased=model(query,partial)
            wall,erasure=unseen_visualizations(model,episode,memory,out,empty,erased,box,mask,case_folder)
            physics=bootstrap_case_diagnostic(out,query,target) if 'posterior_covars' in out else None
            case_vis.append(dict(query=362,wall=wall,erasure=erasure,bootstrap_physics=physics))
            del empty,partial,erased
        del out,memory,episode
    bins = {}
    for label, arrays in buffers.items():
        values = np.concatenate(arrays)
        bins[label] = dict(valid_pixels=len(values), world_epe_m=float(values[:,0].mean()), raw_depth_bias_m=float(values[:,1].mean()),
                           anchored_depth_bias_m=float(values[:,2].mean()), raw_depth_l1_m=float(np.abs(values[:,1]).mean()), anchored_depth_l1_m=float(np.abs(values[:,2]).mean()),native_off_calibrated_ray_m=float(values[:,3].mean()),final_off_calibrated_ray_m=float(values[:,4].mean()),native_camera_epe_m=float(values[:,5].mean()),final_camera_epe_m=float(values[:,6].mean()))
    del model
    torch.cuda.empty_cache()
    return dict(pixel_weighted_depth_bins=bins,query_pose=poses,o6_attention=attention_stats,camera_calibration=calibration_stats,case_visualizations=case_vis,refinement_lm=refinement_lm), np.concatenate(scatter)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old',default='step2_full_seed17');parser.add_argument('--new',default='step2_v2_seed17');parser.add_argument('--step',type=int,default=3000);parser.add_argument('--middle')
    args=parser.parse_args()
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'): parser.error('Use live GPU launcher')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    results, samples={},{}
    output_folder=PROJECT_ROOT/'results'/args.new
    runs=[('full',args.old),('v2',args.middle),('v3',args.new)] if args.middle else [('old',args.old),('new',args.new)]
    for tag,name in runs:
        results[tag],samples[tag]=evaluate_run(name,args.step,case_folder=output_folder/'vis'/f'step_{args.step:05d}'/'robotics_362_diagnostic' if args.middle and tag=='v3' else None)
    folder=PROJECT_ROOT/'results'/args.new
    dump_json(folder/f'robotics_depth_{args.step:05d}.json',results)
    fig,axes=plt.subplots(len(runs),2,figsize=(10,4.5*len(runs)),layout='constrained')
    for row,(tag,_) in enumerate(runs):
        for col,(index,title) in enumerate(((1,'Native camera depth'),(2,'O5 anchored camera depth'))):
            v=samples[tag];axes[row,col].hexbin(v[:,0],v[:,index],gridsize=55,mincnt=1,cmap='viridis',bins='log')
            axes[row,col].plot([0,10],[0,10],'r--',lw=1)
            axes[row,col].set(xlim=(0,10),ylim=(0,10),xlabel='GT camera z [m]',ylabel='Predicted camera z [m]',title=tag+': '+title)
    fig.suptitle('Robotics: all 11 unseen queries, unchanged splits; no output alignment')
    fig.savefig(folder/f'robotics_depth_{args.step:05d}.png',dpi=160);plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
