#!/usr/bin/env python3
"""Read-only final robotics depth bins and same-weight single-map-frame control."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage,load_config,load_checkpoint_readonly,PROJECT_ROOT
configure_local_storage()
import numpy as np
import torch,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.vggt_memory import VGGTSpatialMemory
from src.models.nomemory_vggt import VGGTNoMemory
from src.step2_data import EpisodeDataset
from src.losses.world import pose_errors
from src.vis.step2 import dump_json


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',default='step2_v4_full_seed17');ap.add_argument('--baseline',default='step2_nomemory_seed17');ap.add_argument('--step',type=int,default=18000)
    args=ap.parse_args()
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):ap.error('Use live GPU launcher')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    runtime=load_config();assets=runtime['assets'];output=PROJECT_ROOT/'results'/args.full/'final_comparison';output.mkdir(exist_ok=True)
    labels=['near_0.1_3m','mid_3_5m','far_5_10m'];result={};samples={};control=[]
    for tag,name in [('Full',args.full),('No-Memory',args.baseline)]:
        root=PROJECT_ROOT/'results'/name;c=yaml.safe_load((root/'config.yaml').read_text());state=load_checkpoint_readonly(root/f'training_state_{args.step:05d}.pt')
        if state['step']!=args.step or state['config']!=c:raise ValueError('Checkpoint mismatch')
        kwargs=dict(source=assets['official_vggt_source']['path'],weights=assets['vggt_weights']['path'],receive_layer=c['receive_layer'])
        model=(VGGTSpatialMemory(**kwargs,revision=c['architecture_revision'],pose_jitter_probability=0.) if tag=='Full' else VGGTNoMemory(**kwargs)).cuda().eval()
        params=dict(model.named_parameters())
        for k,v in state['trainable'].items():params[k].copy_(v)
        del state
        dataset=EpisodeDataset(PROJECT_ROOT/c['split'],c['image_hw']);bins={s:[] for s in labels};scatter=[]
        for index,spec in enumerate(dataset.evaluation_episodes()):
            robotics=spec['scene'].startswith('harvard_robotics_lab/')
            if tag!='Full' and not robotics:continue
            ep=dataset.episode(spec['scene'],spec['mapping_frame_ids'],spec['query_frame_id'],device='cuda');q,t=ep['query']
            if tag=='Full':
                one=model.build_memory(ep['mapping'][:1]);single=model(q,one)
                r,tr=pose_errors(single['T1'],t.c2w)
                control.append(dict(**spec,point_epe_m=float((single['world_points'][t.valid]-t.world_points[t.valid]).norm(dim=-1).mean()),rotation_deg=float(r),translation_m=float(tr),gaussians=len(one)))
                del one,single
                if not robotics:continue
                memory=model.build_memory(ep['mapping']);out=model(q,memory);del memory
            else:out=model(q)
            error=(out['world_points']-t.world_points).norm(dim=-1)
            raw=out['raw_camera_points'][...,2];final=out['camera_points'][...,2]
            for label,(lo,hi) in zip(labels,[(.1,3),(3,5),(5,10)]):
                mask=t.valid&(t.depth>=lo)&(t.depth<hi)
                if mask.any():bins[label].append(torch.stack([error[mask],(raw-t.depth)[mask],(final-t.depth)[mask]],-1).cpu().numpy())
            mask=t.valid.flatten().nonzero().flatten()[::40]
            scatter.append(torch.stack([t.depth.flatten()[mask],raw.flatten()[mask],final.flatten()[mask]],-1).cpu().numpy())
            del out,ep
            if index%25==0:print(tag,index,flush=True)
        result[tag]={}
        for label,values in bins.items():
            a=np.concatenate(values);result[tag][label]=dict(valid_pixels=len(a),world_epe_m=float(a[:,0].mean()),world_epe_median_m=float(np.median(a[:,0])),raw_depth_l1_m=float(np.abs(a[:,1]).mean()),final_depth_l1_m=float(np.abs(a[:,2]).mean()),final_depth_bias_m=float(a[:,2].mean()))
        samples[tag]=np.concatenate(scatter);del model;torch.cuda.empty_cache()
    dump_json(output/'robotics_depth_bins.json',result);dump_json(output/'single_map_frame_same_weights.json',dict(step=args.step,description='Only first mapping frame Gaussians; same Full weights; no cross-frame fusion or retraining',episodes=control))
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for i,tag in enumerate(samples):
        a=samples[tag]
        for j,k in enumerate([1,2]):
            ax=axes[i,j];ax.hexbin(a[:,0],a[:,k],gridsize=55,mincnt=1,bins='log',cmap='viridis');ax.plot([0,10],[0,10],'r--');ax.set(xlim=(0,10),ylim=(0,10),xlabel='GT camera z [m]',ylabel='Predicted camera z [m]',title=tag+(' native' if j==0 else ' final'))
    fig.suptitle('Robotics: same 11 unseen queries / 18k updates / no output alignment')
    fig.savefig(output/'robotics_depth_full_vs_nomemory.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
