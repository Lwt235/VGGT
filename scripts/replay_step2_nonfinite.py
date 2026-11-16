#!/usr/bin/env python3
"""Diagnostic-only exact-state replay; never overwrites the formal run."""
import contextlib,json,math,os,random,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage,load_config,load_checkpoint_readonly,PROJECT_ROOT
configure_local_storage()
import torch,numpy as np
from scripts.train_step2 import build_optimizer,criterion,gradient_norm
from src.training_state import restore_training_state
from src.models.vggt_memory import VGGTSpatialMemory
from src.step2_data import EpisodeDataset
from src.augmentation import augment_training_world
from src.models.ellipsoid_pose import sample_read_jitter
from src.losses.world import surface_support_rate
from src.losses.mapping import supervise_mapping_observation
from src.losses.pose_consistency import pose_consistency_loss


def main():
    if not os.environ.get('VGGT_GPU_MEMORY_FRACTION'):raise RuntimeError('Use live GPU launcher')
    output=PROJECT_ROOT/'results'/'step2_v5_nonfinite_diagnostic'
    output.mkdir(exist_ok=False)
    checkpoint=load_checkpoint_readonly(PROJECT_ROOT/'results'/'step2_v5_full_seed17'/'training_state_13000.pt')
    config=checkpoint['config'];torch.set_num_threads(2)
    torch.set_num_threads(2);torch.manual_seed(17);random.seed(17);np.random.seed(17);torch.cuda.manual_seed_all(17)
    torch.cuda.set_per_process_memory_fraction(float(os.environ['VGGT_GPU_MEMORY_FRACTION']))
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assets=load_config()['assets'];dataset=EpisodeDataset(PROJECT_ROOT/config['split'],config['image_hw'])
    model=VGGTSpatialMemory(assets['official_vggt_source']['path'],assets['vggt_weights']['path'],revision=config['architecture_revision'],receive_layer=config['receive_layer'],pose_jitter_probability=0.).cuda().train()
    optimizer,scheduler=build_optimizer(model,config)
    start=restore_training_state(model,optimizer,scheduler,checkpoint,config);del checkpoint
    modules=dict(writer=model.writer,o5=model.render_reader,o6=model.pe_reader,dpt=model.native.point_head,
        pose0=model.initial_pose,pose1=model.refine_pose,reception=model.native.aggregator.global_blocks[4],downstream=model.native.aggregator.global_blocks[11])
    previous=[json.loads(l) for l in (PROJECT_ROOT/'results'/'step2_v5_full_seed17'/'metrics.jsonl').read_text().splitlines()]
    with (output/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(start,13160):
            anomaly=torch.autograd.detect_anomaly(check_nan=True) if step+1>=13139 else contextlib.nullcontext()
            try:
                with anomaly:
                    episode=dataset.sample(device='cuda');query,target=episode['query']
                    if step<len(previous):
                        r=previous[step]
                        assert (episode['name'],episode['query_frame_id'],episode['mapping_frame_ids'])==(r['scene'],r['query_frame_id'],r['mapping_frame_ids'])
                    mapping,target,_=augment_training_world(episode['mapping'],target)
                    optimizer.zero_grad(set_to_none=True)
                    memory=model.build_memory(mapping);memory.retain_grad()
                    read_jitter=sample_read_jitter('cuda',include_zero=True) if random.random()<config['read_jitter_probability'] and surface_support_rate(memory.means,memory.covars,target)>=.05 else None
                    prediction=model(query,memory,read_jitter=read_jitter)
                    loss,metrics=criterion(prediction,target,config)
                    if config['mapping_self_observation']:
                        map_loss,map_metrics=supervise_mapping_observation(model,memory,mapping[step%len(mapping)])
                        loss=loss+map_loss;metrics.update(map_metrics);del map_loss
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    if config['paired_pose_consistency'] and step%2==0 and read_jitter is None and metrics['world_supervision_enabled']:
                        teacher={k:prediction[k].detach() for k in ('T0','T1')}
                        index=torch.multinomial(prediction['bootstrap_attention'].detach().mean(0)+.001,1)[0]
                        center=memory.means[index].detach();radius=random.uniform(*config['paired_erasure_radius_m'])
                        partial=model.build_memory(mapping);partial,erased=partial.erase_box(center-radius,center+radius)
                        if len(partial)>=.5*len(memory) and erased.any():
                            after=model(query,partial)
                            consistency,extra=pose_consistency_loss(after,teacher,target,erased_reference=(memory.means,memory.covars,erased),matching_weight=config['matching_weight'])
                            if not torch.isfinite(consistency):raise RuntimeError('Nonfinite consistency')
                            consistency.backward();metrics.update(extra);del after,consistency
                        del partial
                    norms={n:gradient_norm(m) for n,m in modules.items()}
                    state={n:float(v.grad.norm()) if v.grad is not None else 0. for n,v in zip(('mu','Sigma','alpha','z'),(memory.means,memory.covars,memory.alpha,memory.features))}
                    if not all(math.isfinite(v) for v in [*norms.values(),*state.values()]):
                        bad=[];large=[]
                        for name,p in model.named_parameters():
                            if p.grad is None:continue
                            if not torch.isfinite(p.grad).all():bad.append(name)
                            elif float(p.grad.detach().abs().max())>1e10:large.append(dict(name=name,max_abs=float(p.grad.detach().abs().max()),norm64=float(p.grad.detach().double().norm())))
                        failure=dict(step=step+1,scene=episode['name'],query=episode['query_frame_id'],maps=episode['mapping_frame_ids'],module_norms=norms,memory_norms=state,nonfinite_parameters=bad,large_finite_parameters=large)
                        (output/'nonfinite_gradients.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure),flush=True)
                        raise RuntimeError('Reproduced nonfinite gradients or float32 norm overflow')
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['optimizer']['grad_clip'],error_if_nonfinite=True)
                    optimizer.step();scheduler.step();torch.cuda.synchronize()
                    row=dict(step=step+1,scene=episode['name'],query=episode['query_frame_id'],maps=episode['mapping_frame_ids'],loss=float(loss.detach()),module_norms=norms)
                    log.write(json.dumps(row)+'\n')
                    if (step+1)%10==0:print(json.dumps(row),flush=True)
                    del loss,prediction,memory,episode,mapping,target
            except Exception as error:
                traceback.print_exc()
                (output/'exception.txt').write_text(f'step={step+1}\n'+traceback.format_exc())
                print('Diagnostic stopped at',step+1,repr(error),flush=True)
                return
    print('Replay finite through 13160; original failure was not reproduced.',flush=True)

if __name__=='__main__':main()
