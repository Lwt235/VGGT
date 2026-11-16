#!/usr/bin/env python3
"""Same-protocol Step2 metric/causal comparison; reads only project outputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage, PROJECT_ROOT
configure_local_storage()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vis.step2 import dump_json, causal_summary


def summarize(data, scene=None, group=None):
    rows = [r for r in data['episodes'] if (scene is None or r['scene'].split('/')[0] == scene)
            and (group is None or r['split'] == group) and r['surface_support_rate'] >= .05]
    keys = ['point_epe_m', 'rotation0_deg', 'rotation1_deg', 'translation0_m', 'translation1_m', 'camera_epe_m', 'z_cam_l1_m']
    result = dict(queries=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in keys})
    if rows and 'empty_same_weights' in rows[0]:
        for key in ('empty_same_weights', 'mismatched_same_weights', 'erased_same_weights'):
            result[key] = float(np.mean([r[key]['point_epe_m'] for r in rows]))
        erasures = [r['erasure'] for r in rows if r['erasure']['local_pixels'] and r['erasure']['outside_pixels']]
        result['erasure'] = dict(queries=len(erasures), locally_degraded=sum(r['epe_increase_inside_m'] > max(0, r['epe_increase_outside_m']) for r in erasures),
            **{k:float(np.mean([r[k] for r in erasures])) for k in ('epe_increase_inside_m', 'epe_increase_outside_m', 't0_change_m')})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', default='step2_full_seed17')
    parser.add_argument('--new', default='step2_v2_seed17')
    parser.add_argument('--step', type=int, default=3000)
    args = parser.parse_args()
    roots = [PROJECT_ROOT/'results'/name for name in (args.old, args.new)]
    data = [json.loads((root/f'evaluation_{args.step:05d}_full.json').read_text()) for root in roots]
    # Enforce identical episodes, map ordering and query membership before comparison.
    identifiers = lambda d: [(r['scene'], r['query_frame_id'], r['mapping_frame_ids'], r['split']) for r in d['episodes']]
    if identifiers(data[0]) != identifiers(data[1]):
        raise ValueError('Evaluation episodes differ; cannot compare this as the same protocol')
    groups = [('cross_scene_val', None), ('same_scene_heldout', None), (None, 'harvard_c11'), (None, 'harvard_robotics_lab'), (None, 'harvard_c5'), (None, 'harvard_c6')]
    summary = {}
    for group, scene in groups:
        summary[group or scene] = {name:summarize(d, scene, group) for name, d in zip(('old', 'new'), data)}
    dump_json(roots[1]/f'comparison_{args.step:05d}.json', summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, key, label in zip(axes, ('point_epe_m', 'translation1_m', 'rotation1_deg'), ('World EPE [m]', 'T1 translation [m]', 'T1 rotation [deg]')):
        x = np.arange(4)
        for i, name in enumerate(('old', 'new')):
            ax.bar(x+(i-.5)*.35, [summary[g][name][key] for g in list(summary)[:4]], width=.35, label=name)
        ax.set_xticks(x, ['Cross-scene', 'Heldout', 'C11', 'Robotics'], rotation=15)
        ax.set_ylabel(label); ax.legend()
    fig.suptitle(f'Identical splits / queries / maps / image size; fresh official initialization; step {args.step}')
    fig.savefig(roots[1]/f'comparison_{args.step:05d}.png', dpi=160); plt.close(fig)
    if 'empty_same_weights' in data[1]['episodes'][0]:
        causal_summary(roots[1], data[1])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
