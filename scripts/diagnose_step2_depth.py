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
def evaluate_run(name, step):
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
    scatter, poses, attention_stats = [], [], []
    for spec in specs:
        episode = dataset.episode(spec['scene'], spec['mapping_frame_ids'], spec['query_frame_id'], device='cuda')
        query, target = episode['query']
        memory = model.build_memory(episode['mapping'])
        out = model(query, memory, return_attention=True)
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
        for label, (lo, hi) in zip(buffers, ((.1,3.),(3.,5.),(5.,10.))):
            mask = target.valid & (target.depth >= lo) & (target.depth < hi)
            if mask.any():
                values = torch.stack((world_error[mask], (raw-target.depth)[mask], (anchored-target.depth)[mask]), -1).cpu().numpy()
                buffers[label].append(values)
        mask = target.valid.flatten().nonzero().flatten()[::40]
        scatter.append(torch.stack((target.depth.flatten()[mask], raw.flatten()[mask], anchored.flatten()[mask]), -1).cpu().numpy())
        r0,t0 = pose_errors(out['T0'],target.c2w);r1,t1 = pose_errors(out['T1'],target.c2w)
        poses.append(dict(query=spec['query_frame_id'],rotation0_deg=float(r0),rotation1_deg=float(r1),translation0_m=float(t0),translation1_m=float(t1)))
        del out,memory,episode
    bins = {}
    for label, arrays in buffers.items():
        values = np.concatenate(arrays)
        bins[label] = dict(valid_pixels=len(values), world_epe_m=float(values[:,0].mean()), raw_depth_bias_m=float(values[:,1].mean()),
                           anchored_depth_bias_m=float(values[:,2].mean()), raw_depth_l1_m=float(np.abs(values[:,1]).mean()), anchored_depth_l1_m=float(np.abs(values[:,2]).mean()))
    del model
    torch.cuda.empty_cache()
    return dict(pixel_weighted_depth_bins=bins,query_pose=poses,o6_attention=attention_stats), np.concatenate(scatter)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old',default='step2_full_seed17');parser.add_argument('--new',default='step2_v2_seed17');parser.add_argument('--step',type=int,default=3000)
    args=parser.parse_args()
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'): parser.error('Use live GPU launcher')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    results, samples={},{}
    for tag,name in [('old',args.old),('new',args.new)]:
        results[tag],samples[tag]=evaluate_run(name,args.step)
    folder=PROJECT_ROOT/'results'/args.new
    dump_json(folder/f'robotics_depth_{args.step:05d}.json',results)
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for row,tag in enumerate(('old','new')):
        for col,(index,title) in enumerate(((1,'Native camera depth'),(2,'O5 anchored camera depth'))):
            v=samples[tag];axes[row,col].hexbin(v[:,0],v[:,index],gridsize=55,mincnt=1,cmap='viridis',bins='log')
            axes[row,col].plot([0,10],[0,10],'r--',lw=1)
            axes[row,col].set(xlim=(0,10),ylim=(0,10),xlabel='GT camera z [m]',ylabel='Predicted camera z [m]',title=tag+': '+title)
    fig.suptitle('Robotics: all 11 unseen queries, unchanged splits; no output alignment')
    fig.savefig(folder/f'robotics_depth_{args.step:05d}.png',dpi=160);plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
