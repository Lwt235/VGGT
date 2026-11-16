#!/usr/bin/env python3
"""Read-only final depth bins and one-frame, unfused same-weight control."""
import argparse,hashlib,json,os,sys
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
from src.step2_data import EpisodeDataset
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


DEPTH_LABELS=('near_0.1_3m','mid_3_5m','far_5_10m')
V1_SCENES=('harvard_c5/hv_c5_1','harvard_c6/hv_c6_1',
           'harvard_c11/hv_c11_2','harvard_robotics_lab/hv_s1_1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_metadata(name,step,config,specs):
    """Bind reusable evaluation to the checkpoint and exact fixed protocol."""
    root=PROJECT_ROOT/'results'/name
    checkpoint=root/f'training_state_{step:05d}.pt';stat=checkpoint.stat()
    return dict(cache_version=2,run_name=name,checkpoint_step=step,
        split=config['split'],split_sha256=digest(PROJECT_ROOT/config['split']),
        image_hw=list(config['image_hw']),architecture_revision=config['architecture_revision'],
        config_sha256=digest(root/'config.yaml'),checkpoint_size=stat.st_size,
        checkpoint_mtime_ns=stat.st_mtime_ns,evaluation_episodes=len(specs),
        episode_specs_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest())


def read_full_cache(output,metadata,specs):
    """A partial, stale, or differently configured cache is never reused."""
    try:
        cached=json.loads((output/'full_evaluation_cache.json').read_text())
        control=json.loads((output/'single_map_frame_same_weights.json').read_text())
        if cached['metadata']!=metadata or control['metadata']!=metadata:return None
        if control['step']!=metadata['checkpoint_step']:return None
        identities=lambda rows:[{k:r[k] for k in ('scene','mapping_frame_ids','query_frame_id','split')} for r in rows]
        if identities(control['episodes'])!=identities(specs):return None
        scatter_path=output/'full_evaluation_scatter.npz'
        if cached['scatter_sha256']!=digest(scatter_path):return None
        if cached['control_sha256']!=digest(output/'single_map_frame_same_weights.json'):return None
        with np.load(scatter_path,allow_pickle=False) as arrays:
            samples={'Full':arrays['full'].copy(),'Single-map':arrays['single_map'].copy()}
        if any(a.ndim!=2 or a.shape[1]!=3 or len(a)==0 or not np.isfinite(a).all() for a in samples.values()):return None
        for scene in V1_SCENES:
            if not (output/'single_map_frame_vis'/scene.replace('/','__')/'V1_world_overlap.png').is_file():return None
        if set(cached['robotics_depth_bins'])!={'Full','Single-map'}:return None
        print('Reusing matching Full/Single-map evaluation cache',flush=True)
        return cached['robotics_depth_bins'],samples
    except (OSError,ValueError,KeyError,TypeError):
        return None


def load_model(name,step,config,assets,*,full):
    root=PROJECT_ROOT/'results'/name
    state=load_checkpoint_readonly(root/f'training_state_{step:05d}.pt')
    if state['step']!=step or state['config']!=config:raise ValueError('Checkpoint mismatch')
    kwargs=dict(source=assets['official_vggt_source']['path'],weights=assets['vggt_weights']['path'],receive_layer=config['receive_layer'])
    model=(VGGTSpatialMemory(**kwargs,revision=config['architecture_revision'],pose_jitter_probability=0.) if full else VGGTNoMemory(**kwargs)).cuda().eval()
    params=dict(model.named_parameters())
    for key,value in state['trainable'].items():params[key].copy_(value)
    return model


@torch.no_grad()
def evaluate_full(args,config,dataset,specs,assets,output,metadata):
    model=load_model(args.full,args.step,config,assets,full=True)
    bins={s:[] for s in DEPTH_LABELS};scatter=[]
    one_bins={s:[] for s in DEPTH_LABELS};one_scatter=[];control=[];shown=set()
    for index,spec in enumerate(specs):
        ep=dataset.episode(spec['scene'],spec['mapping_frame_ids'],spec['query_frame_id'],device='cuda');q,t=ep['query']
        one=single_frame_unfused(model,ep['mapping'][0]);single=model(q,one)
        control.append(dict(**spec,**metrics(single,t),gaussians=len(one)))
        robotics=spec['scene'].startswith('harvard_robotics_lab/')
        if robotics:accumulate(single,t,one_bins,one_scatter)
        scene=spec['scene']
        if scene in V1_SCENES and scene not in shown:
            from src.vis.smoke import v1
            path=output/'single_map_frame_vis'/scene.replace('/','__');path.mkdir(parents=True,exist_ok=True)
            # Plot the SAME complete historical reference as Full/No-Memory.
            # This reference is never passed to the single-map model forward.
            control[-1]['wall']=v1(path,dict(name=ep['name']+' / same full-history reference; model: first unfused map view',
                mapping=ep['mapping']),q,t,single)
            shown.add(scene)
        del one,single
        if robotics:
            memory=model.build_memory(ep['mapping']);out=model(q,memory)
            accumulate(out,t,bins,scatter)
            del out,memory
        del ep
        if index%25==0:print('Full/Single-map',index,'/',len(specs),flush=True)
    result={'Full':summarize_bins(bins),'Single-map':summarize_bins(one_bins)}
    samples={'Full':np.concatenate(scatter),'Single-map':np.concatenate(one_scatter)}
    del model;torch.cuda.empty_cache()
    control_path=output/'single_map_frame_same_weights.json'
    dump_json(control_path,dict(step=args.step,metadata=metadata,
        description='Model: only first map view, original writer/context, no SpatialMomentFusion or retraining. V1 plotting reference: same complete historical RGB-D as Full/No-Memory; not model input.',episodes=control))
    scatter_path=output/'full_evaluation_scatter.npz'
    np.savez_compressed(scatter_path,full=samples['Full'],single_map=samples['Single-map'])
    # Write the manifest last: interrupted evaluation cannot become a valid cache.
    dump_json(output/'full_evaluation_cache.json',dict(metadata=metadata,robotics_depth_bins=result,
        scatter_sha256=digest(scatter_path),control_sha256=digest(control_path)))
    return result,samples


@torch.no_grad()
def evaluate_baseline(args,assets,full_specs,full_config):
    root=PROJECT_ROOT/'results'/args.baseline
    config=yaml.safe_load((root/'config.yaml').read_text())
    if config['image_hw']!=full_config['image_hw'] or digest(PROJECT_ROOT/config['split'])!=digest(PROJECT_ROOT/full_config['split']):
        raise ValueError('Full/No-Memory split or image size mismatch')
    dataset=EpisodeDataset(PROJECT_ROOT/config['split'],config['image_hw'])
    if dataset.evaluation_episodes()!=full_specs:raise ValueError('Full/No-Memory evaluation episodes differ')
    model=load_model(args.baseline,args.step,config,assets,full=False)
    bins={s:[] for s in DEPTH_LABELS};scatter=[]
    for spec in full_specs:
        if not spec['scene'].startswith('harvard_robotics_lab/'):continue
        ep=dataset.episode(spec['scene'],spec['mapping_frame_ids'],spec['query_frame_id'],device='cuda');q,t=ep['query']
        out=model(q);accumulate(out,t,bins,scatter);del out,ep
    del model;torch.cuda.empty_cache()
    return summarize_bins(bins),np.concatenate(scatter)


def plot_depth(samples,path,title):
    fig,axes=plt.subplots(len(samples),2,figsize=(10,4.5*len(samples)),layout='constrained')
    for i,(tag,a) in enumerate(samples.items()):
        for j,k in enumerate([1,2]):
            ax=axes[i,j];ax.hexbin(a[:,0],a[:,k],gridsize=55,mincnt=1,bins='log',cmap='viridis');ax.plot([0,10],[0,10],'r--')
            ax.set(xlim=(0,10),ylim=(0,10),xlabel='GT camera z [m]',ylabel='Predicted camera z [m]',title=tag+(' native' if j==0 else ' final'))
    fig.suptitle(title);fig.savefig(path,dpi=160);plt.close(fig)


def plot_full_single_v1(args,output):
    for scene in V1_SCENES:
        key=scene.replace('/','__')
        sources=[PROJECT_ROOT/'results'/args.full/'vis'/f'step_{args.step:05d}'/key/'V1_world_overlap.png',
                 output/'single_map_frame_vis'/key/'V1_world_overlap.png']
        if not all(path.is_file() for path in sources):
            print('Full/Single-map V1 pair unavailable:',scene,flush=True);continue
        fig,axes=plt.subplots(2,1,figsize=(16,10),layout='constrained')
        for tag,path,ax in zip(('Full (same full-history plotting reference)',
                'Single-map (same full-history reference; model uses first map view only, no fusion)'),sources,axes):
            ax.imshow(plt.imread(path));ax.axis('off');ax.set_title(tag)
        fig.suptitle(f'{scene} / Full + Single-map only / step {args.step} / no alignment')
        fig.savefig(output/(key+'_V1_full_vs_single_map.png'),dpi=150);plt.close(fig)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',default='step2_v5_full_seed17');ap.add_argument('--baseline',default='step2_nomemory_seed17');ap.add_argument('--step',type=int,default=18000)
    ap.add_argument('--full-only',action='store_true',help='Evaluate/cache Full and same-weight Single-map only; do not read No-Memory')
    args=ap.parse_args()
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):ap.error('Use live GPU launcher')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assets=load_config()['assets'];root=PROJECT_ROOT/'results'/args.full
    output=root/'final_comparison';output.mkdir(exist_ok=True)
    config=yaml.safe_load((root/'config.yaml').read_text())
    dataset=EpisodeDataset(PROJECT_ROOT/config['split'],config['image_hw']);specs=dataset.evaluation_episodes()
    metadata=evaluation_metadata(args.full,args.step,config,specs)
    full=read_full_cache(output,metadata,specs)
    if full is None:full=evaluate_full(args,config,dataset,specs,assets,output,metadata)
    result,samples=full
    dump_json(output/'robotics_depth_full_vs_single_map.json',dict(metadata=metadata,robotics_depth_bins=result))
    plot_depth(samples,output/'robotics_depth_full_vs_single_map.png',
        f'Robotics: 11 unseen queries / Full + Single-map only / step {args.step} / no alignment')
    plot_full_single_v1(args,output)
    if not args.full_only:
        baseline,baseline_samples=evaluate_baseline(args,assets,specs,config)
        result={'Full':result['Full'],'No-Memory':baseline,'Single-map':result['Single-map']}
        samples={'Full':samples['Full'],'No-Memory':baseline_samples,'Single-map':samples['Single-map']}
        dump_json(output/'robotics_depth_bins.json',result)
        plot_depth(samples,output/'robotics_depth_full_vs_nomemory.png',
            f'Robotics: 11 unseen queries / Full and No-Memory {args.step} steps each; Single-map same Full weights / no alignment')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
