#!/usr/bin/env python3
"""Three fresh Step2 runs on the same fixed unseen episodes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage,PROJECT_ROOT
configure_local_storage()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.compare_step2 import summarize
from src.vis.step2 import dump_json,causal_summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--new',default='step2_v3_seed17');parser.add_argument('--step',type=int,default=3000)
    args=parser.parse_args()
    names=['step2_full_seed17','step2_v2_seed17',args.new];tags=['full','v2','v3']
    data=[json.loads((PROJECT_ROOT/'results'/name/f'evaluation_{args.step:05d}_full.json').read_text()) for name in names]
    ids=lambda d:[(r['scene'],r['query_frame_id'],r['mapping_frame_ids'],r['split']) for r in d['episodes']]
    if not all(ids(d)==ids(data[0]) for d in data[1:]):raise ValueError('Evaluation episodes or map ordering differ')
    groups=[('cross_scene_val',None),('same_scene_heldout',None),(None,'harvard_c11'),(None,'harvard_robotics_lab'),(None,'harvard_c5'),(None,'harvard_c6')]
    result={group or scene:{tag:summarize(d,scene,group) for tag,d in zip(tags,data)} for group,scene in groups}
    folder=PROJECT_ROOT/'results'/args.new
    dump_json(folder/f'comparison_{args.step:05d}_three.json',result)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,key,label in zip(axes,['point_epe_m','translation1_m','rotation1_deg'],['World EPE [m]','T1 translation [m]','T1 rotation [deg]']):
        for i,tag in enumerate(tags):ax.bar(np.arange(4)+(i-1)*.25,[result[g][tag][key] for g in list(result)[:4]],width=.25,label=tag)
        ax.set_xticks(np.arange(4),['Cross-scene','Heldout','C11','Robotics'],rotation=15);ax.set_ylabel(label);ax.legend()
    fig.suptitle(f'Fresh official VGGT; unchanged splits / queries / map order / 168x224; step {args.step}')
    fig.savefig(folder/f'comparison_{args.step:05d}_three.png',dpi=160);plt.close(fig)
    if 'empty_same_weights' in data[-1]['episodes'][0]:causal_summary(folder,data[-1])
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
