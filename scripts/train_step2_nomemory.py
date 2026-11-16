#!/usr/bin/env python3
"""Train the independent single-frame No-Memory baseline after Full 18k.

Reuses only the Full training episode IDs, not its weights/predictions. No mapping observations enter the model. Query forward receives RGB/K only;
training and evaluation target coordinates use the first map camera (G=I).
"""
import argparse,json,math,os,random,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT,configure_local_storage,load_config,load_checkpoint_readonly
configure_local_storage()
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from src.step2_data import EpisodeDataset
from src.geometry import invert_se3,transform_points
from src.losses.world import pose_errors,rotation_geodesic
from src.models.nomemory_vggt import VGGTNoMemory
from src.evaluation import benchmark_observable
from src.vis.step2 import dump_json,curves
from scripts.train_smoke import gradient_norm


def query_loss(output,target,config,*,observable=True):
    valid=target.valid
    delta=output['world_points'][valid]-target.world_points[valid]
    world=delta.abs().mean()
    pose=output['T1']
    rotation=rotation_geodesic(pose[:3,:3],target.c2w[:3,:3])
    translation=(pose[:3,3]-target.c2w[:3,3]).norm()
    camera_truth=transform_points(target.world_points,invert_se3(target.c2w))[valid]
    camera_delta=output['camera_points'][valid]-camera_truth
    camera=camera_delta.abs().mean()
    relative=(camera_delta[:,2].abs()/camera_truth[:,2].clamp_min(.2)).mean()
    loss=config['camera_alignment_weight']*camera+config['relative_depth_weight']*relative
    if observable:
        loss=loss+world+F.smooth_l1_loss(rotation,rotation.new_zeros(()),beta=.03)+F.smooth_l1_loss(translation,translation.new_zeros(()),beta=.03)
    with torch.no_grad():r,t=pose_errors(pose,target.c2w)
    return loss,dict(loss=float(loss.detach()),point_l1_m=float(world.detach()),point_epe_m=float(delta.detach().norm(dim=-1).mean()),
        coarse_point_epe_m=float(delta.detach().norm(dim=-1).mean()),rotation0_deg=float(r),rotation1_deg=float(r),translation0_m=float(t),translation1_m=float(t),
        camera_epe_m=float(camera_delta.detach().norm(dim=-1).mean()),camera_alignment_l1_m=float(camera.detach()),relative_depth_l1=float(relative.detach()),
        world_supervision_enabled=float(observable))


@torch.no_grad()
def evaluate(model,dataset,config,folder,step,reference,*,quick=False,visualize=False):
    model.eval();records=[];shown=set()
    for spec in dataset.evaluation_episodes(max_per_scene=2 if quick else None):
        label=reference[(spec['scene'],spec['query_frame_id'])]
        if label['mapping_frame_ids'] != spec['mapping_frame_ids']:raise ValueError('Reference map ordering changed')
        episode=dataset.episode(spec['scene'],spec['mapping_frame_ids'],spec['query_frame_id'],device='cuda')
        query,target=episode['query']
        output=model(query)
        support=label['surface_support_rate']
        observable=benchmark_observable(spec,{sid:item for sid,(_,item) in dataset.scenes.items()})
        _,metrics=query_loss(output,target,config,observable=observable)
        records.append(dict(**spec,**metrics,surface_support_rate=support,benchmark_observable=observable,reference_support='original fixed RGB-D overlap; learned Full 3sigma support reported separately'))
        if visualize and spec['scene'].split('/')[0] in ('harvard_c5','harvard_c6','harvard_c11','harvard_robotics_lab') and spec['scene'] not in shown:
            from src.vis.smoke import v1,save,cloud,COLD,ORANGE
            path=folder/'vis'/f'step_{step:05d}'/spec['scene'].replace('/','__');path.mkdir(parents=True,exist_ok=True)
            try:records[-1]['wall']=v1(path,dict(name=episode['name'],mapping=episode['mapping']),query,target,output)
            except RuntimeError as error:
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
                for _,truth in episode['mapping']:
                    p=cloud(truth.world_points,truth.valid);ax.scatter(p[:,0],p[:,2],c=COLD,s=.5,alpha=.3)
                p=cloud(output['world_points'],target.valid);ax.scatter(p[:,0],p[:,2],c=ORANGE,s=.5,alpha=.5)
                ax.set(title='Local: direct world coordinates; wall ROI unavailable',xlabel='world x [m]',ylabel='world z [m]');save(fig,path/'V1_world_overlap.png')
                print('Wall ROI unavailable:',error,flush=True)
            shown.add(spec['scene'])
        if len(records)%25==0:print('Local evaluation',step,len(records),flush=True)
        del episode,output
    groups=[]
    for group in ('cross_scene_val','same_scene_heldout'):
        selected=[r for r in records if r['split']==group and r['benchmark_observable']]
        blind=[r for r in records if r['split']==group and not r['benchmark_observable']]
        keys=('point_epe_m','coarse_point_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m','camera_epe_m','surface_support_rate')
        groups.append(dict(step=step,scope='quick' if quick else 'full',group=group,queries=len(selected),blind_queries=len(blind),
                           **{key:float(np.mean([r[key] for r in selected])) for key in keys}))
    dump_json(folder/f'evaluation_{step:05d}_{"quick" if quick else "full"}.json',dict(step=step,architecture='single-frame VGGT+DPT+pose; no mapping input or public spatial memory',
        query_inputs=['rgb','K'],evaluation_gauge='first mapping camera; G=I; no output alignment',aggregates=groups,episodes=records))
    print('Local validation:',json.dumps(groups),flush=True);model.train();return groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/step2_nomemory.yaml');parser.add_argument('--run',required=True)
    parser.add_argument('--until',type=int,default=18000);parser.add_argument('--resume',action='store_true');parser.add_argument('--profile',type=int,default=0)
    args=parser.parse_args();config=yaml.safe_load((PROJECT_ROOT/args.config).read_text())
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):parser.error('Use live GPU resource launcher')
    if config.get('initialize_from') or config['initialization']!='official_vggt_only_no_smoke_checkpoint':parser.error('Official fresh initialization required')
    folder=(PROJECT_ROOT/'results'/args.run).resolve()
    if not folder.is_relative_to(PROJECT_ROOT/'results') or (folder.exists() and not args.resume):parser.error('Use a new results directory or its own explicit resume')
    source=PROJECT_ROOT/'results'/config['source_full_run']
    source_config=yaml.safe_load((source/'config.yaml').read_text())
    for key in ('split','image_hw','steps','seed'):
        if source_config[key]!=config[key]:raise ValueError('Full and Local protocol mismatch: '+key)
    schedule=[json.loads(l) for l in (source/'metrics.jsonl').read_text().splitlines()]
    required=args.profile or args.until
    if len(schedule)<required:parser.error('Full episode schedule is incomplete')
    reference_path=source/f"evaluation_{config['steps']:05d}_full.json"
    if not args.profile and not reference_path.exists():parser.error('Full must finish before Local main training')
    # Profile has no evaluation; an existing initial grouping is sufficient. Main always uses final Full.
    if args.profile and not reference_path.exists():reference_path=source/'evaluation_00000_full.json'
    reference_data=json.loads(reference_path.read_text());reference={(r['scene'],r['query_frame_id']):r for r in reference_data['episodes']}
    subprocess.run([sys.executable,'-B','-m','pytest','tests','-q'],cwd=PROJECT_ROOT,env=dict(os.environ,REQUIRE_GSPLAT='1'),check=True)
    torch.set_num_threads(2);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    folder.mkdir(exist_ok=args.resume);(folder/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    dataset=EpisodeDataset(PROJECT_ROOT/config['split'],config['image_hw']);runtime=load_config()
    print('Fresh official Local baseline; no Full parameters:',runtime['assets']['vggt_weights']['path'],flush=True)
    model=VGGTNoMemory(runtime['assets']['official_vggt_source']['path'],runtime['assets']['vggt_weights']['path'],receive_layer=config['receive_layer']).cuda().train()
    opt=config['optimizer'];used=set();groups=[]
    for module,rate in [(model.pose,opt['pose_lr']),(model.native,opt['native_lr']),(model,opt['heads_lr'])]:
        params=[p for p in module.parameters() if p.requires_grad and id(p) not in used];used.update(id(p) for p in params)
        if params:groups.append(dict(params=params,lr=rate,weight_decay=.01))
    optimizer=torch.optim.AdamW(groups)
    def rate(step):
        warm=opt['warmup_steps']
        if step<warm:return (step+1)/warm
        progress=min(1.,(step-warm)/max(1,config['steps']-warm))
        return opt['end_lr_ratio']+(1-opt['end_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate);start=0
    if args.resume:
        state=load_checkpoint_readonly(folder/'training_state.pt')
        if state['config']!=config or state['initialization']!='fresh_official_vggt_nomemory':raise ValueError('Own Local checkpoint/config required')
        parameters=dict(model.named_parameters())
        with torch.no_grad():
            for key,value in state['trainable'].items():parameters[key].copy_(value)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start=state['step']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
        print('RESUME own Local model/Adam/scheduler/Python/CPU/CUDA RNG:',start,flush=True)
    records=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()] if args.resume else []
    validation=[json.loads(l) for l in (folder/'validation.jsonl').read_text().splitlines()] if args.resume else []
    if not args.resume and not args.profile:
        validation+=evaluate(model,dataset,config,folder,0,reference)
        (folder/'validation.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in validation))
    until=start+args.profile if args.profile else args.until
    modules=dict(dpt=model.native.point_head,pose=model.pose,
                 reception=model.native.aggregator.global_blocks[4],downstream=model.native.aggregator.global_blocks[11])
    started=time.monotonic()
    with (folder/'metrics.jsonl').open('a' if args.resume else 'w',buffering=1) as log:
        for step in range(start,until):
            tic=time.monotonic();torch.cuda.reset_peak_memory_stats();spec=schedule[step]
            item=dataset.scenes[spec['scene']][1];allowed=set(item['train_frame_ids'])
            if item['partition']!='train' or spec['query_frame_id'] not in allowed or not set(spec['mapping_frame_ids'])<=allowed or spec['query_frame_id'] in spec['mapping_frame_ids']:raise ValueError('Invalid Full training schedule entry')
            episode=dataset.episode(spec['scene'],spec['mapping_frame_ids'],spec['query_frame_id'],device='cuda');query,target=episode['query']
            # G=I. Mapping is used only by dataset.episode to define target coordinates.
            optimizer.zero_grad(set_to_none=True)
            output=model(query);loss,metrics=query_loss(output,target,config)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Local loss')
            loss.backward();norms={name:gradient_norm(module) for name,module in modules.items()}
            if not all(math.isfinite(v) for v in norms.values()):raise RuntimeError('Nonfinite Local gradient')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],opt['grad_clip'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();torch.cuda.synchronize()
            metrics.update(loss=float(loss.detach()),step=step+1,scene=spec['scene'],query_frame_id=spec['query_frame_id'],mapping_frame_ids=spec['mapping_frame_ids'],
                module_grad_norms=norms,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,step_seconds=time.monotonic()-tic)
            log.write(json.dumps(metrics,allow_nan=False)+'\n');records.append(metrics)
            if step<3 or (step+1)%config['log_every']==0 or args.profile:print(json.dumps({k:metrics[k] for k in ['step','loss','point_epe_m','rotation1_deg','translation1_m','peak_reserved_gib','step_seconds']}),flush=True)
            if step > 50 and (step+1)%100==0:
                for name in modules:
                    if not any(r['module_grad_norms'][name]>0 for r in records[-50:]):raise RuntimeError('Persistent disconnected No-Memory '+name)
            del loss,output,episode,target
            if not args.profile and ((step+1)%config['checkpoint_every']==0 or step+1==until):
                state=dict(step=step+1,config=config,initialization='fresh_official_vggt_nomemory',trainable={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state,folder/'training_state.tmp');(folder/'training_state.tmp').replace(folder/'training_state.pt');del state
                if step+1 in config['checkpoint_milestones']:
                    snap=folder/f'training_state_{step+1:05d}.pt'
                    if not snap.exists():snap.hardlink_to(folder/'training_state.pt')
            if not args.profile and ((step+1)%config['quick_eval_every']==0 or step+1==until):
                full=step+1 in config['full_eval_steps'] or step+1==until
                validation+=evaluate(model,dataset,config,folder,step+1,reference,quick=not full,visualize=step+1==until)
                (folder/'validation.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in validation));curves(folder/'vis',records,validation)
    summary=dict(completed_steps=until,profile=bool(args.profile),initialization='fresh_official_vggt_nomemory',peak_reserved_gib=max(r['peak_reserved_gib'] for r in records),
        final_loss_mean=float(np.mean([r['loss'] for r in records[-100:]])),elapsed_minutes=round((time.monotonic()-started)/60,1))
    dump_json(folder/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
