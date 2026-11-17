#!/usr/bin/env python3
"""Exact-plan v1/v2 streaming comparison, without pose or point alignment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT, configure_local_storage
configure_local_storage()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def stats(rows, field):
    data = np.array([r[field] for r in rows])
    return dict(mean=float(data.mean()), median=float(np.median(data)), p90=float(np.percentile(data, 90)))


def compare(baseline, updated, destination):
    plans = read(baseline/'evaluation_final/stream_plans.json')
    if plans != read(updated/'evaluation_final/stream_plans.json'):
        raise ValueError('v1/v2 evaluation plans differ')
    results, total = [], {key: [] for key in ('v1 Full', 'v1 W4', 'v2 Full', 'v2 W4')}
    fields = ('point_epe_m', 'point_l1_m', 'rotation0_deg', 'rotation1_deg', 'translation0_m', 'translation1_m')
    details = {}
    for plan in plans:
        rows = {}
        for version, folder in (('v1', baseline), ('v2', updated)):
            for variant, filename in (('Full', 'persistent_metrics.json'), ('W4', 'sliding_window_metrics.json')):
                data = read(folder/'evaluation_final'/plan['name']/filename)
                if data['plan'] != plan:
                    raise ValueError('Plan metadata mismatch')
                current = data['rows']
                if [r['frame_id'] for r in current] != plan['frame_ids'][1:]:
                    raise ValueError('A hard frame was lost or reordered')
                rows[f'{version} {variant}'] = current
                total[f'{version} {variant}'].extend(current)
        entry = dict(name=plan['name'], queries=len(plan['frame_ids'])-1, variants={})
        for label, current in rows.items():
            metrics = {field: stats(current, field) for field in fields}
            metrics.update(final_translation_m=current[-1]['translation1_m'], final_rotation_deg=current[-1]['rotation1_deg'],
                final_public=current[-1]['memory_after'], max_active=max(r['active_count'] for r in current),
                phases={phase: {field: stats([r for r in current if r['phase']==phase], field)
                                for field in fields} for phase in dict.fromkeys(r['phase'] for r in current)})
            if label.startswith('v2'):
                checks = [r['write_checks'] for r in current]
                metrics['write_diagnostics'] = dict(rejected_frames=sum(c['candidates']==0 for c in checks),
                    safety_rejected_frames=sum(not c['safety_accepted'] for c in checks),
                    landmark_selected_frames=sum(c['landmark_selected'] for c in checks),
                    mean_feature_cosine=float(np.mean([c['feature_cosine'] for c in checks])),
                    mean_consensus=float(np.mean([c['rigid_consensus'] for c in checks])),
                    locked_seed_drift_max_m=max((r['seed_geometry_drift_max_m'] or 0 for r in current), default=0),
                    wrong_high_write_frames=sum(r['translation1_m'] > 1 and r['write_checks']['frame_quality'] > .4 and r['write_checks']['candidates'] > 0 for r in current))
            entry['variants'][label] = metrics
        if plan['kind'].startswith('natural') and 'harvard_c11' in plan['scene']:
            entry['return_frames'] = {label:[r for r in current if r['frame_id'] in (781,797)] for label,current in rows.items()}
        results.append(entry)
        details[plan['name']] = rows
    counts = {key:len(rows) for key,rows in total.items()}
    if set(counts.values()) != {304}:
        raise ValueError(f'Expected all 304 queries/method: {counts}')
    destination.mkdir(parents=True, exist_ok=True)
    combined = dict(counts=counts, aggregate={label:{field:stats(rows,field) for field in fields} for label,rows in total.items()},
                    streams=results, no_alignment=True)
    (destination/'comparison.json').write_text(json.dumps(combined, indent=2, allow_nan=False))
    lines = ['# Step3 v1 / v2：相同304查询，直接世界系指标', '', '|流|v1 Full EPE|v2 Full EPE|v1 W4 EPE|v2 W4 EPE|v2 Full t末帧|', '|---|---:|---:|---:|---:|---:|']
    for entry in results:
        v = entry['variants']
        lines.append('|'+entry['name']+'|'+ '|'.join(f'{v[k]["point_epe_m"]["mean"]:.4f}' for k in ('v1 Full','v2 Full','v1 W4','v2 W4'))+f'|{v["v2 Full"]["final_translation_m"]:.4f}|')
    lines += ['', '全部均值/中位数/P90、分段位姿、重访781/797、写入校验与种子漂移见comparison.json；没有剔除任何查询。']
    (destination/'comparison.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(len(plans), 3, figsize=(17, 3.8*len(plans)), squeeze=False)
    for axrow, plan in zip(axes, plans):
        for label, current in details[plan['name']].items():
            ts = [r['t'] for r in current]
            for ax, field in zip(axrow, ('point_epe_m','translation1_m','memory_after')):
                ax.plot(ts, [r[field] for r in current], label=label, alpha=.85)
        for ax, name in zip(axrow, ('World EPE (m)', 'Translation drift (m)', 'PUBLIC Gaussians')):
            ax.set_title(plan['name']+'\n'+name, fontsize=9); ax.grid(alpha=.2); ax.set_xlabel('t'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(destination/'v1_v2_all_stream_curves.png', dpi=150); plt.close(fig)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='results/step3_streaming_seed17')
    parser.add_argument('--updated', default='results/step3_v2_seed17')
    args = parser.parse_args()
    output = PROJECT_ROOT/args.updated/'comparison_v1'
    result = compare(PROJECT_ROOT/args.baseline, PROJECT_ROOT/args.updated, output)
    print(json.dumps(dict(counts=result['counts'], aggregate=result['aggregate']), indent=2))


if __name__ == '__main__':
    main()
