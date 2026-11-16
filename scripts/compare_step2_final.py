#!/usr/bin/env python3
"""Equal-budget Full versus RGB/K-only No-Memory, all fixed unseen queries."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage,PROJECT_ROOT
configure_local_storage()
import numpy as np
import yaml
from src.evaluation import benchmark_observable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vis.step2 import dump_json,causal_summary


def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()),count=len(a))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',default='step2_v5_full_seed17');parser.add_argument('--baseline',default='step2_nomemory_seed17');parser.add_argument('--step',type=int,default=18000)
    args=parser.parse_args();names=[args.full,args.baseline];tags=['Full','No-Memory','Single-map']
    roots=[PROJECT_ROOT/'results'/n for n in names]
    data=[json.loads((p/f'evaluation_{args.step:05d}_full.json').read_text()) for p in roots]
    control=roots[0]/'final_comparison'/'single_map_frame_same_weights.json'
    data.append(json.loads(control.read_text()))
    ids=lambda d:[(r['scene'],r['query_frame_id'],r['mapping_frame_ids'],r['split']) for r in d['episodes']]
    if any(ids(d)!=ids(data[0]) for d in data[1:]):raise ValueError('Episode identities/map order changed')
    config=yaml.safe_load((roots[0]/'config.yaml').read_text())
    split=yaml.safe_load((PROJECT_ROOT/config['split']).read_text());sequences={s['id']:s for s in split['sequences']}
    result={}
    for group,prefix in [('cross_scene_val',None),('same_scene_heldout',None),('C11','harvard_c11/'),('robotics','harvard_robotics_lab/')]:
        result[group]={}
        selector=lambda r:(r['scene'].startswith(prefix) if prefix else r['split']==group) and benchmark_observable(r,sequences)
        keep=[i for i,r in enumerate(data[0]['episodes']) if selector(r)]
        for tag,d in zip(tags,data):
            rows=[d['episodes'][i] for i in keep]
            result[group][tag]={k:stats([r[k] for r in rows]) for k in ['point_l1_m','point_epe_m','coarse_point_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m','camera_epe_m']}
            if tag=='Full':
                result[group][tag]['interventions']={key:stats([r['point_epe_m'] if key=='full' else r[key]['point_epe_m'] for r in rows]) for key in ['full','empty_same_weights','mismatched_same_weights','erased_same_weights']}
                erased=[r['erasure'] for r in rows if r['erasure']['local_pixels'] and r['erasure']['outside_pixels']]
                result[group][tag]['erasure']=dict(queries=len(erased),locally_degraded=sum(r['epe_increase_inside_m']>0 for r in erased),inside_delta_m=stats([r['epe_increase_inside_m'] for r in erased]),outside_delta_m=stats([r['epe_increase_outside_m'] for r in erased]))
            result[group][tag]['wall_examples']=[dict(scene=r['scene'],query=r['query_frame_id'],wall=r['wall']) for r in rows if 'wall' in r]
            result[group][tag]['tail_queries']=sorted([dict(scene=r['scene'],query=r['query_frame_id'],epe_m=r['point_epe_m'],rotation_deg=r['rotation1_deg'],translation_m=r['translation1_m']) for r in rows],key=lambda r:r['epe_m'],reverse=True)[:5]
    output=roots[0]/'final_comparison';output.mkdir(exist_ok=True)
    dump_json(output/'full_vs_nomemory.json',result)
    dump_json(output/'blind_queries.json',{tag:[r for r in d['episodes'] if not benchmark_observable(r,sequences)] for tag,d in zip(tags,data)})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,key,label in zip(axes,['point_epe_m','translation1_m','rotation1_deg'],['World EPE [m]','Translation [m]','Rotation [deg]']):
        for i,tag in enumerate(tags):
            mean=[result[g][tag][key]['mean'] for g in result];median=[result[g][tag][key]['median'] for g in result]
            x=np.arange(len(result))+(i-1)*.25
            ax.bar(x,mean,width=.25,label=tag+' mean');ax.scatter(x,median,marker='_',c='black',s=45,label=tag+' median')
        ax.set_xticks(np.arange(len(result)),list(result),rotation=18,fontsize=8);ax.set_ylabel(label);ax.legend(fontsize=6)
    fig.suptitle('Full and No-Memory 18k each; Single-map = same Full weights, one unfused map view; no alignment')
    fig.savefig(output/'full_vs_nomemory.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for tag,root in zip(tags,roots):
        logs=[json.loads(l) for l in (root/'validation.jsonl').read_text().splitlines()]
        for group,style in [('cross_scene_val','-'),('same_scene_heldout','--')]:
            rows=[dict(r) for r in logs if r['scope']=='full' and r['group']==group]
            for summary in rows:
                source=json.loads((root/f"evaluation_{summary['step']:05d}_full.json").read_text())
                fixed=[r for r in source['episodes'] if r['split']==group and benchmark_observable(r,sequences)]
                for key in ['point_epe_m','translation1_m','rotation1_deg']:
                    summary[key]=float(np.mean([r[key] for r in fixed]))
            for ax,key in zip(axes,['point_epe_m','translation1_m','rotation1_deg']):ax.plot([r['step'] for r in rows],[r[key] for r in rows],style,marker='o',label=tag+' '+group)
    for ax,label in zip(axes,['World EPE [m]','Translation [m]','Rotation [deg]']):ax.set(xlabel='Updates',ylabel=label);ax.legend(fontsize=6)
    fig.savefig(output/'full_vs_nomemory_curves.png',dpi=160);plt.close(fig)
    for scene in ['harvard_c5__hv_c5_1','harvard_c6__hv_c6_1','harvard_c11__hv_c11_2','harvard_robotics_lab__hv_s1_1']:
        fig,axes=plt.subplots(3,1,figsize=(16,15),layout='constrained')
        sources=[root/'vis'/f'step_{args.step:05d}'/scene/'V1_world_overlap.png' for root in roots]
        sources.append(output/'single_map_frame_vis'/scene/'V1_world_overlap.png')
        for tag,source,ax in zip(tags,sources,axes):
            ax.imshow(plt.imread(source));ax.axis('off');ax.set_title(tag+' · same fixed query / world reference; original panels retained')
        fig.savefig(output/(scene+'_V1_full_vs_nomemory.png'),dpi=150);plt.close(fig)
    causal_summary(roots[0],data[0],observable=lambda r:benchmark_observable(r,sequences))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
