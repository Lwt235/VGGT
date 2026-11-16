#!/usr/bin/env python3
"""Read-only final depth bins and one-frame, unfused same-weight control."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage,load_config,load_checkpoint_readonly,PROJECT_ROOT
configure_local_storage()
import numpy as np
import torch,yaml
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.vggt_memory import VGGTSpatialMemory,frozen_patch_tokens,aggregate_single
from src.models.nomemory_vggt import VGGTNoMemory
from src.geometry import patch_intrinsics,invert_se3,transform_points
from src.losses.world import pose_errors
from src.vis.step2 import dump_json


@torch.no_grad()
def single_frame_unfused(model,observation):
    """Same writer/context, one permitted map RGB-D view, no fusion call."""
    q,t=observation
    tokens=frozen_patch_tokens(model.native.aggregator,q.rgb)[0].float()
    with torch.autocast(device_type=q.rgb.device.type,dtype=torch.bfloat16,enabled=q.rgb.is_cuda):
        context=aggregate_single(model.native.aggregator,q.rgb,tokens[None],
            receive_layer=model.receive_layer,stop_before_injection=True)
    tokens=torch.cat((tokens,context.float()),-1)
    valid=t.valid.float()[None,None]
    fraction=F.avg_pool2d(valid,14,14)[0,0]
    depth=F.avg_pool2d((t.depth*t.valid)[None,None],14,14)[0,0]/fraction.clamp_min(1e-6)
    return model.writer(tokens,depth,patch_intrinsics(q.K),t.c2w,valid_mask=fraction>.25)


def metrics(out,t):
    raw=out['raw_camera_points'];truth=transform_points(t.world_points,invert_se3(t.c2w))
    r0,t0=pose_errors(out['T0'],t.c2w);r1,t1=pose_errors(out['T1'],t.c2w)
    v=dict(point_l1_m=float((out['world_points'][t.valid]-t.world_points[t.valid]).abs().mean()),
        point_epe_m=float((out['world_points'][t.valid]-t.world_points[t.valid]).norm(dim=-1).mean()),
        coarse_point_epe_m=float((out['coarse_world_points'][t.valid]-t.world_points[t.valid]).norm(dim=-1).mean()),
        rotation0_deg=float(r0),rotation1_deg=float(r1),translation0_m=float(t0),translation1_m=float(t1),
        camera_epe_m=float((raw[t.valid]-truth[t.valid]).norm(dim=-1).mean()),
        anchored_camera_epe_m=float((out['camera_points'][t.valid]-truth[t.valid]).norm(dim=-1).mean()))
    if 'dustbin_probability' in out:
        v.update(dustbin_mean=float(out['dustbin_probability'].mean()),dustbin_rejection_rate=float((out['dustbin_probability']>.5).float().mean()))
    return v


def accumulate(out,t,bins,scatter):
    error=(out['world_points']-t.world_points).norm(dim=-1)
    raw=out['raw_camera_points'][...,2];final=out['camera_points'][...,2]
    for label,(lo,hi) in zip(bins,[(.1,3),(3,5),(5,10)]):
        mask=t.valid&(t.depth>=lo)&(t.depth<hi)
        if mask.any():bins[label].append(torch.stack([error[mask],(raw-t.depth)[mask],(final-t.depth)[mask]],-1).cpu().numpy())
    mask=t.valid.flatten().nonzero().flatten()[::40]
    scatter.append(torch.stack([t.depth.flatten()[mask],raw.flatten()[mask],final.flatten()[mask]],-1).cpu().numpy())


def summarize_bins(bins):
    result={}
    for label,values in bins.items():
        a=np.concatenate(values)
        result[label]=dict(valid_pixels=len(a),world_epe_m=float(a[:,0].mean()),world_epe_median_m=float(np.median(a[:,0])),
            world_epe_p90_m=float(np.quantile(a[:,0],.9)),world_epe_max_m=float(a[:,0].max()),
            raw_depth_l1_m=float(np.abs(a[:,1]).mean()),final_depth_l1_m=float(np.abs(a[:,2]).mean()),
            final_depth_l1_median_m=float(np.median(np.abs(a[:,2]))),final_depth_l1_p90_m=float(np.quantile(np.abs(a[:,2]),.9)),
            final_depth_bias_m=float(a[:,2].mean()))
    return result


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',default='step2_v5_full_seed17');ap.add_argument('--baseline',default='step2_nomemory_seed17');ap.add_argument('--step',type=int,default=18000)
    args=ap.parse_args()
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):ap.error('Use live GPU launcher')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assets=load_config()['assets'];output=PROJECT_ROOT/'results'/args.full/'final_comparison';output.mkdir(exist_ok=True)
    labels=['near_0.1_3m','mid_3_5m','far_5_10m'];result={};samples={};control=[]
    one_bins={s:[] for s in labels};one_scatter=[]
    for tag,name in [('Full',args.full),('No-Memory',args.baseline)]:
        root=PROJECT_ROOT/'results'/name;c=yaml.safe_load((root/'config.yaml').read_text());state=load_checkpoint_readonly(root/f'training_state_{args.step:05d}.pt')
        if state['step']!=args.step or state['config']!=c:raise ValueError('Checkpoint mismatch')
        kwargs=dict(source=assets['official_vggt_source']['path'],weights=assets['vggt_weights']['path'],receive_layer=c['receive_layer'])
        model=(VGGTSpatialMemory(**kwargs,revision=c['architecture_revision'],pose_jitter_probability=0.) if tag=='Full' else VGGTNoMemory(**kwargs)).cuda().eval()
        params=dict(model.named_parameters())
        for k,v in state['trainable'].items():params[k].copy_(v)
        del state
        dataset=EpisodeDataset(PROJECT_ROOT/c['split'],c['image_hw']);bins={s:[] for s in labels};scatter=[];shown=set()
        for index,spec in enumerate(dataset.evaluation_episodes()):
            robotics=spec['scene'].startswith('harvard_robotics_lab/')
            if tag!='Full' and not robotics:continue
            ep=dataset.episode(spec['scene'],spec['mapping_frame_ids'],spec['query_frame_id'],device='cuda');q,t=ep['query']
            if tag=='Full':
                one=single_frame_unfused(model,ep['mapping'][0]);single=model(q,one)
                control.append(dict(**spec,**metrics(single,t),gaussians=len(one)))
                if robotics:accumulate(single,t,one_bins,one_scatter)
                scene=spec['scene']
                if scene.split('/')[0] in ('harvard_c5','harvard_c6','harvard_c11','harvard_robotics_lab') and scene not in shown:
                    from src.vis.smoke import v1
                    path=output/'single_map_frame_vis'/scene.replace('/','__');path.mkdir(parents=True,exist_ok=True)
                    control[-1]['wall']=v1(path,dict(name=ep['name']+' / one unfused map view',mapping=ep['mapping']),q,t,single)
                    shown.add(scene)
                del one,single
                if not robotics:continue
                memory=model.build_memory(ep['mapping']);out=model(q,memory);del memory
            else:out=model(q)
            accumulate(out,t,bins,scatter)
            del out,ep
            if index%25==0:print(tag,index,flush=True)
        result[tag]=summarize_bins(bins);samples[tag]=np.concatenate(scatter);del model;torch.cuda.empty_cache()
    result['Single-map']=summarize_bins(one_bins);samples['Single-map']=np.concatenate(one_scatter)
    dump_json(output/'robotics_depth_bins.json',result)
    dump_json(output/'single_map_frame_same_weights.json',dict(step=args.step,description='Only first map view, original writer/context, no SpatialMomentFusion and no retraining',episodes=control))
    fig,axes=plt.subplots(3,2,figsize=(10,13),layout='constrained')
    for i,tag in enumerate(samples):
        a=samples[tag]
        for j,k in enumerate([1,2]):
            ax=axes[i,j];ax.hexbin(a[:,0],a[:,k],gridsize=55,mincnt=1,bins='log',cmap='viridis');ax.plot([0,10],[0,10],'r--');ax.set(xlim=(0,10),ylim=(0,10),xlabel='GT camera z [m]',ylabel='Predicted camera z [m]',title=tag+(' native' if j==0 else ' final'))
    fig.suptitle('Robotics: same 11 unseen queries / Full and No-Memory 18k each / no alignment')
    fig.savefig(output/'robotics_depth_full_vs_nomemory.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
