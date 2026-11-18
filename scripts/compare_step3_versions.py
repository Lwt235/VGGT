#!/usr/bin/env python3
"""Compare streaming versions on the same plans in the original world gauge."""
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


def compare(baseline, updated, destination, *, third=None):
    versions = {'v1': baseline, 'v2': updated}
    if third is not None:
        versions['v3'] = third
    latest = next(reversed(versions))
    prefix = '_'.join(versions)
    labels = [f'{version} {variant}' for version in versions for variant in ('Full', 'W4')]
    plans = read(baseline/'evaluation_final/stream_plans.json')
    if any(plans != read(folder/'evaluation_final/stream_plans.json') for folder in versions.values()):
        raise ValueError('Evaluation plans differ between versions')
    results, total = [], {label: [] for label in labels}
    fields = ('point_epe_m', 'point_l1_m', 'rotation0_deg', 'rotation1_deg', 'translation0_m', 'translation1_m')
    details = {}
    for plan in plans:
        rows = {}
        for version, folder in versions.items():
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
        for version, folder in versions.items():
            measured = read(folder/'evaluation_final'/plan['name']/'comparison.json')
            entry[version+'_geometry'] = {label:{key:value for key,value in values.items()
                if key in ('wall','public_map_quality','closure')} for label,values in measured['variants'].items()}
            entry[version+'_interventions'] = read(folder/'evaluation_final'/plan['name']/'representative_interventions.json')
        for label, current in rows.items():
            metrics = {field: stats(current, field) for field in fields}
            metrics.update(final_translation_m=current[-1]['translation1_m'], final_rotation_deg=current[-1]['rotation1_deg'],
                final_public=current[-1]['memory_after'], max_active=max(r['active_count'] for r in current),
                phases={phase: {field: stats([r for r in current if r['phase']==phase], field)
                                for field in fields} for phase in dict.fromkeys(r['phase'] for r in current)})
            if 'write_checks' in current[0]:
                checks = [r['write_checks'] for r in current]
                metrics['write_diagnostics'] = dict(rejected_frames=sum(c['candidates']==0 for c in checks),
                    safety_rejected_frames=sum(not c['safety_accepted'] for c in checks),
                    landmark_selected_frames=sum(c['landmark_selected'] for c in checks),
                    mean_feature_cosine=(float(np.mean([c['feature_cosine'] for c in checks]))
                                         if 'feature_cosine' in checks[0] else None),
                    selected_pose_sources={source:sum(c.get('pose_source') == source for c in checks)
                                           for source in ('seed', 'track', 'global')},
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
    lines = ['# Step3 ' + ' / '.join(versions) + '：相同304查询，直接世界系指标',
        '', '|流|' + '|'.join(label + ' EPE [m]' for label in labels) + '|',
        '|---|' + '---:|'*len(labels)]
    for entry in results:
        v = entry['variants']
        lines.append('|' + entry['name'] + '|' + '|'.join(f'{v[k]["point_epe_m"]["mean"]:.4f}' for k in labels) + '|')
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
    fig.tight_layout(); fig.savefig(destination/f'{prefix}_all_stream_curves.png', dpi=150); plt.close(fig)
    # Unaligned world trajectories make inward folding visible independently
    # of scalar EPE. They use saved evaluator targets only for the grey trace.
    fig, axes = plt.subplots(1,len(plans),figsize=(5*len(plans),5),squeeze=False)
    for ax, plan in zip(axes[0],plans):
        data=details[plan['name']]
        target=np.array([r['target_c2w'] for r in data['v2 Full']])[:,:3,3]
        ax.plot(target[:,0],target[:,2],color='.45',lw=2,label='GT; evaluation only')
        colors = dict(zip(versions, ('#e88a28','#1478af','#23914d')))
        for label in labels:
            version, variant = label.split()
            color = colors[version]
            style = '-' if variant == 'Full' else ':'
            path=np.array([r['predicted_c2w'] for r in data[label]])[:,:3,3]
            ax.plot(path[:,0],path[:,2],style,color=color,lw=1.5,label=label)
            ax.scatter(path[-1,0],path[-1,2],s=20,color=color)
        ax.scatter(0,0,marker='*',color='black',s=70,label='Seed gauge origin')
        ax.set(title=plan['name'].replace('__',' / '),xlabel='World x [m]',ylabel='World z [m]')
        ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(destination/f'{prefix}_world_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(plans),3,figsize=(17,3.3*len(plans)),squeeze=False)
    for axrow,plan in zip(axes,plans):
        current=details[plan['name']][f'{latest} Full'];ts=[r['t'] for r in current]
        for field,label in (('rigid_consensus','Rigid inlier fraction'),('feature_cosine','Rendered/query latent cosine'),('frame_quality','Write quality')):
            if field in current[0]['write_checks']:
                axrow[0].plot(ts,[r['write_checks'][field] for r in current],label=label)
        axrow[1].bar(ts,[r['write_checks']['candidates'] for r in current],color='#1478af',label='Predicted Gaussian candidates')
        selected=[r['t'] for r in current if r['write_checks']['landmark_selected']]
        if selected:axrow[1].scatter(selected,[0]*len(selected),marker='*',color='#e88a28',s=45,label='Stable landmark selected')
        axrow[2].plot(ts,[r['seed_geometry_drift_max_m'] or 0 for r in current],label='Locked seed maximum shift [m]')
        axrow[2].plot(ts,[r['translation1_m'] for r in current],label='Camera translation error [m]')
        for ax in axrow:
            ax.set_title(plan['name'],fontsize=9);ax.set_xlabel('t');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(destination/f'{latest}_write_and_landmark_diagnostics.png',dpi=150);plt.close(fig)
    # The extended loiter stream is separate from the 304-query primary set.
    loiter_paths = [folder/'loiter_diagnostic/stream_plans.json' for folder in versions.values()]
    if third is not None or all(path.is_file() for path in loiter_paths):
        loiter_plans = read(loiter_paths[0])
        if any(loiter_plans != read(path) for path in loiter_paths[1:]):
            raise ValueError('Extended loiter plans changed')
        loiter=[]
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for p in loiter_plans:
            for version,folder in versions.items():
                for variant,filename in (('Full','persistent_metrics.json'),('W4','sliding_window_metrics.json')):
                    measured=read(folder/'loiter_diagnostic'/p['name']/filename)
                    current=measured['rows']
                    if measured['plan'] != p or [r['frame_id'] for r in current] != p['frame_ids'][1:]:
                        raise ValueError('Loiter plan or query order changed')
                    selected=[r for r in current if r['phase']=='Loiter']
                    loiter.append(dict(version=version,variant=variant,queries=len(current),epe=stats(current,'point_epe_m'),
                        loiter_start=selected[0]['memory_before'],loiter_end=selected[-1]['memory_after'],
                        last20_growth=selected[-1]['memory_after']-selected[-20]['memory_before'],
                        last10_growth=selected[-1]['memory_after']-selected[-10]['memory_before'],
                        last20_nonzero_write_steps=(sum(r['write_checks']['candidates']>0 for r in selected[-20:])
                            if 'write_checks' in selected[-1] else None),
                        last20_candidates=(sum(r['write_checks']['candidates'] for r in selected[-20:])
                            if 'write_checks' in selected[-1] else None),
                        final_translation_m=current[-1]['translation1_m']))
                    for ax,field in zip(axes,('memory_after','point_epe_m')):
                        ax.plot([r['t'] for r in current],[r[field] for r in current],label=f'{version} {variant}')
        for ax,title in zip(axes,('PUBLIC count (read budget is separate)','World EPE [m]')):
            ax.axvspan(20,79,color='.85',alpha=.3,label='Loiter');ax.set(title=title,xlabel='t');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(destination/f'{prefix}_extended_loiter.png',dpi=150);plt.close(fig)
        combined['extended_loiter']=loiter
        (destination/'comparison.json').write_text(json.dumps(combined,indent=2,allow_nan=False))
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='results/step3_streaming_seed17')
    parser.add_argument('--updated', default='results/step3_v2_seed17')
    parser.add_argument('--third', help='Optional v3 run for the six-way comparison')
    args = parser.parse_args()
    third = PROJECT_ROOT/args.third if args.third else None
    output = third/'comparison_versions' if third else PROJECT_ROOT/args.updated/'comparison_v1'
    result = compare(PROJECT_ROOT/args.baseline, PROJECT_ROOT/args.updated, output, third=third)
    print(json.dumps(dict(counts=result['counts'], aggregate=result['aggregate']), indent=2))


if __name__ == '__main__':
    main()
