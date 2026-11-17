#!/usr/bin/env python3
"""Strict RGB/K-only read-then-write benchmark; GT is isolated to metrics.

Natural streams use every registered paired frame in source order. Constructed
streams explicitly replay real frames along an outward/backward route; they
are never described as a naturally recorded continuous video.
"""
import argparse
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage, PROJECT_ROOT
configure_local_storage()
import numpy as np
import torch
import yaml
from src.data import Supervision
from src.geometry import invert_se3, transform_points, intrinsics_cv_to_edge
from src.losses.world import pose_errors
from src.vis.streaming import v6, metric_curves, summarize_rows, v1_comparison, training_curves, public_map_diagnostics
from src.vis.step2 import dump_json

C11='harvard_c11/hv_c11_2'
ROBOTICS='harvard_robotics_lab/hv_s1_1'


def segmented_plan(frame_ids, *, steps=80, outward_end=None):
    """Four explicitly labelled phases of an actual-frame replay, no GT use.

    Forward explore -> small local bounce -> outward to end -> backward to
    seed. Integer resampling can repeat an existing source observation; every
    selected index is disclosed in the saved plan.
    """
    if len(frame_ids)<8:raise ValueError('Need at least eight real paired frames')
    if not 60<=steps<=100:raise ValueError('Long-stream replay must have 60–100 steps')
    n=len(frame_ids);cut=max(3,n//3)
    end=n-1 if outward_end is None else int(outward_end)
    if not cut<=end<n:raise ValueError('Outward endpoint must follow the initial explored segment')
    lengths=[steps//4]*3+[steps-3*(steps//4)]
    explore=np.linspace(0,cut-1,lengths[0]).round().astype(int).tolist()
    bounce=list(range(cut-1,max(0,cut-5),-1))+list(range(max(0,cut-5),cut-1))
    loiter=[bounce[i%len(bounce)] for i in range(lengths[1])]
    new=np.linspace(loiter[-1],end,lengths[2]).round().astype(int).tolist()
    revisit=np.linspace(end,0,lengths[3]).round().astype(int).tolist()
    indices=explore+loiter+new+revisit
    labels=sum(([label]*count for label,count in zip(('Explore','Loiter','Explore New','Revisit'),lengths)),[])
    return {'frame_ids':[int(frame_ids[i]) for i in indices],'phases':labels,
            'kind':'constructed real-frame outward/bounce/backward replay; not natural recording order',
            'source_indices':indices,'outward_end_source_index':end}


def default_plans(spec, *, steps=80):
    items={x['id']:x for x in spec['sequences'] if x['partition']!='excluded'}
    plans=[]
    for sid in (C11,ROBOTICS):
        ids=items[sid]['frame_ids'];n=len(ids)
        # Natural streams have no fabricated loiter phase; temporal segments
        # are declared by source index, never selected using predictions/GT.
        phases=(['Explore' if i<n//3 else 'Explore New' if i<5*n//6 else 'Revisit' for i in range(n)]
                if sid==C11 else ['Natural']*n)
        plans.append(dict(name=sid.replace('/','__')+'__natural',scene=sid,frame_ids=ids,phases=phases,
            kind='natural complete paired-frame order',source_indices=list(range(n))))
    for prefix in ('harvard_c11/','harvard_c5/','harvard_c8/'):
        sid=next(k for k in items if k.startswith(prefix))
        ids=items[sid]['frame_ids']
        # C11 already contains a natural return in its final sixth. Its
        # constructed outward leg stops BEFORE that return, then replays the
        # outward route backwards. All 52 frames still occur in natural above.
        endpoint=5*len(ids)//6-1 if prefix=='harvard_c11/' else None
        plans.append(dict(name=sid.replace('/','__')+'__segmented',scene=sid,
            **segmented_plan(ids,steps=steps,outward_end=endpoint)))
    return plans


def prediction_metrics(output,target):
    valid=target.valid;delta=output['world_points'][valid]-target.world_points[valid]
    r0,t0=pose_errors(output['T0'],target.c2w);r1,t1=pose_errors(output['T1'],target.c2w)
    row=dict(valid_pixels=int(valid.sum()),point_epe_m=float(delta.norm(dim=-1).mean()),point_l1_m=float(delta.abs().mean()),
        rotation0_deg=float(r0),translation0_m=float(t0),rotation1_deg=float(r1),translation1_m=float(t1))
    truth=transform_points(target.world_points,invert_se3(target.c2w))
    row['camera_epe_m']=float((output['camera_points'][valid]-truth[valid]).norm(dim=-1).mean())
    if 'dustbin_probability' in output:
        probability=output['dustbin_probability']
        row.update(dustbin_mean=float(probability.mean()),dustbin_rejection_rate=float((probability>.5).float().mean()))
    if 'write_confidence' in output:row['write_confidence_mean']=float(output['write_confidence'].mean())
    if 'stream_pose_gate' in output:row['refinement_gate']=float(output['stream_pose_gate'])
    return row


def seed_surface_overlap(dataset, plan, frame_id):
    """Evaluator-only RGB-D visibility from the fixed read-only split metadata.

    It is not Gaussian support and never enters inference or write gating.
    The stored pair matrix has zero diagonals by construction; an exact replay
    of the seed is instead fully self-visible.
    """
    item=dataset.sequences[plan['scene']]
    seed=plan['frame_ids'][0]
    if frame_id==seed:return 1.
    ids=item['frame_ids']
    return float(item['surface_overlap'][ids.index(frame_id)][ids.index(seed)])


def cpu_prediction(output):
    return {key:output[key].detach().float().cpu() for key in ('world_points','T0','T1')}


def cpu_target(target):
    return Supervision(*(value.detach().cpu() for value in (target.depth,target.c2w,target.world_points,target.valid)))


def display_shells(folder,scene,memory,max_shells=256):
    """Display sampling affects only plotting, never the public memory state."""
    import matplotlib.pyplot as plt
    from src.vis.smoke import array,ellipsoid,axes3d,save
    means,covars,alpha=map(array,(memory.means,memory.covars,memory.alpha))
    indices=np.linspace(0,len(memory)-1,min(max_shells,len(memory))).round().astype(int) if len(memory) else []
    fig=plt.figure(figsize=(12,5),layout='constrained')
    for panel,selection in enumerate((indices,np.argsort(np.linalg.norm(means,axis=-1))[:16])):
        ax=fig.add_subplot(1,2,panel+1,projection='3d')
        for i in selection:
            _,vectors=np.linalg.eigh(covars[i]);color=np.abs(vectors[:,0])
            ellipsoid(ax,means[i],covars[i],color,alpha[i],detail=8)
        axes3d(ax);ax.set_title(f'{len(selection)} displayed / {len(memory)} PUBLIC primitives' if panel==0 else '16 nearest shell details')
    fig.suptitle('V2 · '+scene+' · 1.5 sigma shells; RGB=|thin axis|; alpha=opacity')
    save(fig,folder/'V2_ellipsoid_shells.png')


@torch.no_grad()
def read_only(engine,query,memory,*,return_attention=False,support=None):
    if hasattr(engine,'read'):
        if support is not None:
            return engine.read(query,memory,return_attention=return_attention,support=support)
        return engine.read(query,memory,return_attention=return_attention)
    return engine.model(query,memory,return_attention=return_attention,streaming=True)


@torch.no_grad()
def representative_figures(engine,dataset,plan,t,query,target,memory,full,folder,*,support=None):
    """Interventions are read-only branches, never committed into the stream."""
    from scripts.train_step2 import intervention_box,erasure_metrics
    from src.vis.smoke import v1,v3,v4,v5
    folder.mkdir(parents=True,exist_ok=True)
    empty=read_only(engine,query,memory.subset(slice(0,0)),support=None if support is None else support[:0])
    box=intervention_box(memory,target);partial,mask=memory.erase_box(*box)
    erased=read_only(engine,query,partial,support=None if support is None else support[~mask])
    # Reload only one seed for this offline figure. No engine ever stores it.
    seed=next(iter(dataset.load_stream(plan['scene'],[plan['frame_ids'][0]],device=query.rgb.device)['observations']))
    scene=dict(name=plan['name']+f' / t={t}; pre-write map',mapping=[seed])
    wall=None
    try:wall=v1(folder,scene,query,target,full)
    except RuntimeError:
        v1_comparison(folder,plan['name'],seed[1],target,{'Full persistent':full})
    display_shells(folder,plan['name'],memory)
    h,w=query.rgb.shape[-2:]
    render0=engine.model.render_reader.render(memory,full['T0'],intrinsics_cv_to_edge(query.K),h,w)
    render1=engine.model.render_reader.render(memory,full['T1'],intrinsics_cv_to_edge(query.K),h,w)
    v3(folder,scene,query,target,full,empty,render0,render1)
    v4(folder,scene,query,memory,full,engine.model.receive_layer)
    erase_vis=v5(folder,scene,query,target,memory,full,erased,box,mask)
    return dict(full=prediction_metrics(full,target),empty_same_weights=prediction_metrics(empty,target),
        erased_same_weights=prediction_metrics(erased,target),erasure=erasure_metrics(full,erased,target,box),
        erasure_visualization=erase_vis,wall=wall)


@torch.no_grad()
def evaluate_plan(engine,dataset,plan,folder,*,visualize=True):
    """Every engine step below receives RGB/K + prior 3-D state only."""
    from src.streaming import StreamingEngine
    records={};snapshots={};map_snapshots={};final_centers={};gt_surfaces=[];seed_reference=None
    for label,window in (('Full persistent',None),('Sliding Window W=4',4)):
        runner=type(engine)(engine.model,window=window).to(next(engine.parameters()).device).eval()
        # Share the trained stream-specific gates/fusion too, not only VGGT.
        runner.load_state_dict(engine.state_dict(),strict=True)
        observations=iter(dataset.load_stream(plan['scene'],plan['frame_ids'],device=next(engine.parameters()).device)['observations'])
        q0,t0=next(observations);state=runner.seed(q0,t0.depth,t0.valid)
        seed_geometry=state.memory.means.detach().clone() if hasattr(state,'seed_lock') else None
        if window is None:gt_surfaces.append(t0.world_points[t0.valid][::4].detach().float().cpu().numpy())
        seed_reference=cpu_target(t0);del q0,t0
        rows=[]
        representative=len(plan['frame_ids'])-1
        for t,(query,target) in enumerate(observations,1):
            old_memory=state.memory
            old_support=state.support if hasattr(state,'seed_lock') else None
            full,state=runner(query,state,return_attention=visualize and t==representative and window is None)
            if window is None:gt_surfaces.append(target.world_points[target.valid][::4].detach().float().cpu().numpy())
            row=dict(t=t,frame_id=plan['frame_ids'][t],phase=plan['phases'][t],memory_before=len(old_memory),memory_after=len(state.memory),
                active_count=int(full.get('active_count',len(full.get('attention',{}).get('indices',[])))),**prediction_metrics(full,target))
            row['seed_rgbd_surface_overlap']=seed_surface_overlap(dataset,plan,row['frame_id'])
            row['predicted_c2w']=full['T1'].cpu().tolist();row['target_c2w']=target.c2w.cpu().tolist()
            if old_support is not None:
                row['write_checks']=full['write']
                row['mean_support']=float(state.support.mean()) if len(state.support) else 0.
                row['seed_locked_gaussians']=int(state.seed_lock.sum())
                row['seed_geometry_drift_max_m']=(float((state.memory.means[state.seed_lock]-seed_geometry).norm(dim=-1).max())
                    if state.seed_lock.any() else None)
            if not all(np.isfinite(row[k]) for k in ('point_epe_m','rotation1_deg','translation1_m')):
                raise FloatingPointError(f'Nonfinite stream metric: {plan["name"]} t={t}')
            rows.append(row)
            if t==representative:
                snapshots[label]=cpu_prediction(full)
                from src.memory.gaussians import GaussianMemory
                map_snapshots[label]=GaussianMemory(*(value.detach().float().cpu() for value in
                    (old_memory.means,old_memory.covars,old_memory.alpha,old_memory.features)))
                if visualize and window is None:
                    causal=representative_figures(runner,dataset,plan,t,query,target,old_memory,full,folder/'vis'/'full',support=old_support)
                    dump_json(folder/'representative_interventions.json',causal)
                last_target=cpu_target(target)
            if t%10==0:print(plan['name'],label,t,'/',len(plan['frame_ids'])-1,'memory',len(state.memory),'EPE',round(row['point_epe_m'],3),flush=True)
            del full,old_memory,query,target
        records[label]=rows
        final_centers[label]=state.memory.means.detach().float().cpu()
        # Latest metrics only, no Gaussian/checkpoint snapshots per frame.
        dump_json(folder/(('persistent' if window is None else 'sliding_window')+'_metrics.json'),dict(plan=plan,rows=rows,summary=summarize_rows(rows)))
        del state,runner
        torch.cuda.empty_cache()
    v6(folder/'vis',plan['name'],records,patch_tokens=dataset.image_hw[0]*dataset.image_hw[1]//196)
    metric_curves(folder/'vis',plan['name'],records)
    wall=v1_comparison(folder/'vis',plan['name'],seed_reference,last_target,snapshots,memories=map_snapshots)
    map_quality=public_map_diagnostics(folder/'vis',plan['name'],np.concatenate(gt_surfaces),final_centers)
    summary={label:summarize_rows(rows) for label,rows in records.items()}
    for label,rows in records.items():
        tail=rows[-1]
        summary[label]['closure']=dict(query_frame_id=tail['frame_id'],translation_error_m=tail['translation1_m'],rotation_error_deg=tail['rotation1_deg'],
            actual_seed_to_endpoint_distance_m=float(np.linalg.norm(np.array(tail['target_c2w'])[:3,3])),
            predicted_seed_to_endpoint_distance_m=float(np.linalg.norm(np.array(tail['predicted_c2w'])[:3,3])),
            definition='End pose versus actual GT endpoint in seed gauge; endpoint is not assumed to equal seed')
        summary[label]['wall']=wall[label]
        summary[label]['public_map_quality']=map_quality[label]
        summary[label]['memory']=dict(final=rows[-1]['memory_after'],maximum=max(r['memory_after'] for r in rows),maximum_active=max(r['active_count'] for r in rows),
            loiter_start=next((r['memory_before'] for r in rows if r['phase']=='Loiter'),None),loiter_end=next((r['memory_after'] for r in reversed(rows) if r['phase']=='Loiter'),None))
    dump_json(folder/'comparison.json',dict(plan=plan,variants=summary))
    return dict(plan=plan,variants=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/step3_streaming.yaml')
    parser.add_argument('--checkpoint',default='results/step3_streaming_seed17/training_state.pt')
    parser.add_argument('--output',default='results/step3_streaming_seed17/evaluation_final')
    parser.add_argument('--split',default='configs/step2_splits.yaml')
    parser.add_argument('--plans',help='Optional previously saved exact stream plans JSON')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--no-visualize',action='store_true')
    parser.add_argument('--streams',nargs='*',help='Exact plan names; default is all five streams')
    parser.add_argument('--worker',action='store_true',help='Own plan directories only; coordinator merges root metadata')
    args=parser.parse_args()
    output=PROJECT_ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    spec=yaml.safe_load((PROJECT_ROOT/args.split).read_text())
    plans=json.loads((PROJECT_ROOT/args.plans).read_text()) if args.plans else default_plans(spec)
    if args.streams:plans=[p for p in plans if p['name'] in args.streams]
    if not plans:raise ValueError('No selected stream plans')
    if not args.worker:dump_json(output/'stream_plans.json',plans)
    if args.plan_only:
        print(json.dumps([{k:p[k] for k in ('name','scene','kind')}|{'steps':len(p['frame_ids'])} for p in plans],indent=2));return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; resource checks must be performed by the launcher')
    fraction=os.environ.get('VGGT_GPU_MEMORY_FRACTION')
    if fraction is None:raise RuntimeError('Use run_on_available_gpu.py after live GPU/process checks')
    torch.cuda.set_per_process_memory_fraction(float(fraction),0)
    torch.set_num_threads(2)
    from scripts.train_step3 import load_engine
    from src.streaming_data import StreamingDataset
    config=yaml.safe_load((PROJECT_ROOT/args.config).read_text())
    if config['split']!=args.split:raise ValueError('Evaluation must use unchanged training split')
    engine=load_engine(config,checkpoint=PROJECT_ROOT/args.checkpoint,device='cuda').eval()
    dataset=StreamingDataset(PROJECT_ROOT/args.split,config['image_hw'])
    comparisons=[]
    for plan in plans:
        folder=output/plan['name'];folder.mkdir(exist_ok=True)
        comparisons.append(evaluate_plan(engine,dataset,plan,folder,visualize=not args.no_visualize))
        if not args.worker:
            dump_json(output/'summary.json',dict(checkpoint=args.checkpoint,world_gauge='seed I, metric m; no posterior alignment',
                seed_inputs='RGB/K + seed-only RGB-D; later RGB/K only',streams=comparisons))
    training_log=(PROJECT_ROOT/args.checkpoint).parent/'metrics.jsonl'
    if training_log.exists() and not args.worker:
        rows=[json.loads(line) for line in training_log.read_text().splitlines() if line.strip()]
        training_curves(output/'vis',rows)
    manifest=sorted(str(p.relative_to(output)) for p in output.rglob('*.png'))
    if not args.worker:dump_json(output/'image_manifest.json',{'images':manifest,'opened_by_agent':False})
    print('Streaming evaluation completed:',output,flush=True)


if __name__=='__main__':main()
