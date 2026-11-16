#!/usr/bin/env python3
"""Fixed original-overlap v4 midpoint summary; preserve all tail queries."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT,configure_local_storage
configure_local_storage()
import numpy as np,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation import benchmark_observable
from src.vis.step2 import dump_json,causal_summary
from scripts.compare_step2_final import stats

root=PROJECT_ROOT/'results'/'step2_v4_full_seed17'
sequences={s['id']:s for s in yaml.safe_load((PROJECT_ROOT/'configs/step2_splits.yaml').read_text())['sequences']}
names=['step2_full_seed17','step2_v2_seed17','step2_v3_seed17','step2_v4_full_seed17'];tags=['full','v2','v3','v4']
data=[json.loads((PROJECT_ROOT/'results'/n/'evaluation_03000_full.json').read_text()) for n in names]
ids=lambda d:[(r['scene'],r['query_frame_id'],r['mapping_frame_ids'],r['split']) for r in d['episodes']]
assert all(ids(d)==ids(data[0]) for d in data)
select=lambda r,g:benchmark_observable(r,sequences) and (r['scene'].startswith(g) if g.endswith('/') else r['split']==g)
groups=['cross_scene_val','same_scene_heldout','harvard_c11/','harvard_robotics_lab/','harvard_c5/','harvard_c6/']
result={}
for group in groups:
 result[group]={}
 for tag,d in zip(tags,data):
  rows=[r for r in d['episodes'] if select(r,group)]
  value={key:stats([r[key] for r in rows]) for key in ['point_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m','camera_epe_m']}
  if tag=='v4':
   value['interventions']={key:stats([r['point_epe_m'] if key=='full' else r[key]['point_epe_m'] for r in rows]) for key in ['full','empty_same_weights','mismatched_same_weights','erased_same_weights']}
   erased=[r['erasure'] for r in rows if r['erasure']['local_pixels'] and r['erasure']['outside_pixels']]
   value['erasure']=dict(queries=len(erased),locally_degraded=sum(r['epe_increase_inside_m']>0 for r in erased),inside_delta_m=stats([r['epe_increase_inside_m'] for r in erased]),outside_delta_m=stats([r['epe_increase_outside_m'] for r in erased]))
   value['pose_improved_count']=dict(rotation=sum(r['rotation1_deg']<=r['rotation0_deg'] for r in rows),translation=sum(r['translation1_m']<=r['translation0_m'] for r in rows))
  result[group][tag]=value
v4=data[-1]['episodes'];cross=[r for r in v4 if select(r,'cross_scene_val')]
result['gate']=dict(limit_m=.35,mean_world_epe_m=float(np.mean([r['point_epe_m'] for r in cross])),without_robotics362_mean_m=float(np.mean([r['point_epe_m'] for r in cross if not(r['scene'].startswith('harvard_robotics') and r['query_frame_id']==362)])),fixed_covisible_queries=124,zero_overlap_queries=2,completed_steps=3000,planned_steps=18000,continued=False)
result['predicted_support_misclassified']=[r for r in v4 if benchmark_observable(r,sequences) and r['surface_support_rate']<.05]
result['query_regression_vs_v3']={g:sum(new['point_epe_m']>old['point_epe_m'] for old,new in zip(data[-2]['episodes'],v4) if select(new,g)) for g in groups}
dump_json(root/'gate_summary_03000.json',result)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,key,label in zip(axes,['point_epe_m','translation1_m','rotation1_deg'],['World EPE [m]','Translation [m]','Rotation [deg]']):
 for i,tag in enumerate(tags):ax.bar(np.arange(4)+(i-1.5)*.2,[result[g][tag][key]['mean'] for g in groups[:4]],width=.2,label=tag)
 ax.set_xticks(np.arange(4),['Cross-scene','Heldout','C11','Robotics'],rotation=15);ax.set_ylabel(label);ax.legend(fontsize=7)
fig.suptitle('3000 updates / same original-overlap groups and episodes / fresh official VGGT')
fig.savefig(root/'comparison_03000_four.png',dpi=160);plt.close(fig)
validation=[]
for step in [0,1500,3000]:
 d=json.loads((root/f'evaluation_{step:05d}_full.json').read_text())
 for group in groups[:2]:
  rows=[r for r in d['episodes'] if select(r,group)]
  validation.append(dict(step=step,group=group,queries=len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in ['point_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m']}))
(root/'validation_fixed_protocol.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in validation))
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for g in groups[:2]:
 r=[v for v in validation if v['group']==g]
 for ax,key in zip(axes,['point_epe_m','translation1_m','rotation1_deg']):ax.plot([v['step'] for v in r],[v[key] for v in r],'o-',label=g)
for ax,label in zip(axes,['World EPE [m]','Translation [m]','Rotation [deg]']):ax.set(xlabel='Updates',ylabel=label);ax.legend(fontsize=7)
fig.suptitle('Fixed 45 cross-scene / 79 heldout queries; original RGB-D overlap; no alignment')
fig.savefig(root/'validation_fixed_protocol.png',dpi=160);plt.close(fig)
causal_summary(root,data[-1],observable=lambda r:benchmark_observable(r,sequences))
train=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
dump_json(root/'summary.json',dict(completed_steps=3000,planned_steps=18000,profile=False,initialization='fresh_official_vggt',stop_reason='configured_cross_scene_EPE_above_0.35m',cross_scene_epe_m=result['gate']['mean_world_epe_m'],final_loss_mean=float(np.mean([r['loss'] for r in train[-100:]])),peak_reserved_gib=max(r['peak_reserved_gib'] for r in train)))
print(json.dumps({g:result[g]['v4'] for g in groups},indent=2))
