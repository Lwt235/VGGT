#!/usr/bin/env python3
"""CPU-only quantitative Step 3 report from actual streaming evaluations.

All source metrics/checkpoints remain read-only. Missing evaluation data is an
error; missing optional wall sections remain explicitly unavailable. No output
alignment, frame exclusion, or model inference is performed here.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import PROJECT_ROOT,configure_local_storage
configure_local_storage()
import numpy as np

VARIANTS={'Full persistent':'persistent_metrics.json','Sliding Window W=4':'sliding_window_metrics.json'}
PHASES=('Explore','Loiter','Explore New','Revisit')
METRICS=('point_epe_m','point_l1_m','camera_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m',
         'write_confidence_mean','dustbin_mean','dustbin_rejection_rate','refinement_gate','memory_before','memory_after','active_count')
CSV_FIELDS=('category','stream','scene','scene_group','variant','phase','metric','count','mean','median','p90','value')


def read_json(path):
    if not path.is_file():raise FileNotFoundError('Missing measured Step 3 artifact: '+str(path))
    return json.loads(path.read_text())


def scene_group(scene):
    return 'unseen cross-scene' if scene.startswith(('harvard_c11/','harvard_robotics_lab/')) else 'known training scene; streaming evaluation'


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    if not np.isfinite(a).all():raise ValueError('Nonfinite measured metric; do not silently omit failed frames')
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)))


def metric_distribution(rows,key,*,required=False):
    missing=[r['t'] for r in rows if r.get(key) is None]
    if missing:
        if required:raise ValueError('Missing required '+key+' at stream steps '+str(missing))
        return None
    return distribution([r[key] for r in rows])


def validate_rows(rows,plan,variant):
    expected=list(range(1,len(plan['frame_ids'])))
    if [r['t'] for r in rows]!=expected:raise ValueError(f'{plan["name"]}/{variant}: incomplete/duplicated stream steps')
    if [r['frame_id'] for r in rows]!=plan['frame_ids'][1:]:raise ValueError('Source frame order differs from saved stream plan')
    if [r['phase'] for r in rows]!=plan['phases'][1:]:raise ValueError('Phase labels differ from saved plan')
    for key in ('point_epe_m','point_l1_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m'):
        metric_distribution(rows,key,required=True)
    for row in rows:
        if row['memory_before']<0 or row['memory_after']<0 or row['active_count']>2048:
            raise ValueError('Invalid public/active memory counts')
        if row['active_count']>row['memory_before']:
            raise ValueError('Active reads cannot exceed the pre-write public memory')


def read_training(path):
    if not path.is_file():raise FileNotFoundError('Missing measured streaming training log: '+str(path))
    rows=[]
    with path.open() as handle:
        for line in handle:
            if line.strip():rows.append(json.loads(line))
    if not rows:raise ValueError('Streaming training log has no measured rows')
    for key in ('step','loss'):
        if any(key not in row for row in rows):raise ValueError('Incomplete streaming training log: '+key)
    if any(not np.isfinite(float(row['loss'])) for row in rows):raise ValueError('Training log contains nonfinite loss')
    if any(a['step']>=b['step'] for a,b in zip(rows,rows[1:])):raise ValueError('Training steps duplicated/out of order')
    return rows


def fmt(value,digits=4):
    return 'N/A' if value is None else f'{float(value):.{digits}f}'


def stats_text(stats):
    return 'N/A' if stats is None else '/'.join(fmt(stats[k]) for k in ('mean','median','p90'))


def build_report(evaluation,training_log):
    """Returns Markdown plus a long-form quantitative CSV, with no inference."""
    evaluation=Path(evaluation)
    summary=read_json(evaluation/'summary.json')
    plans=read_json(evaluation/'stream_plans.json')
    names=[plan['name'] for plan in plans]
    actual=[stream['plan']['name'] for stream in summary['streams']]
    if actual!=names:raise ValueError('Incomplete final evaluation; missing/order mismatch: '+str([n for n in names if n not in actual]))
    train=read_training(Path(training_log))
    tables=[];lines=['# Step 3 quantitative streaming results','',
        'Measured final stream artifacts only. The seed is not scored as a prediction. All later queries, including failures and repeated real-frame observations, are retained. No posterior alignment or ICP.',
        '',f"Checkpoint: `{summary.get('checkpoint','not recorded')}`. World gauge: {summary.get('world_gauge','not recorded')}.",'',
        'C11/robotics are unseen cross-scene streams. C5/C8 are known training scenes evaluated along a new streaming route. Natural streams preserve every paired frame; segmented streams explicitly replay real frames and are not described as naturally recorded continuous video.',
        '',f"Training log: {len(train)} recorded updates, steps {train[0]['step']}–{train[-1]['step']}; loss {fmt(train[0]['loss'])} → {fmt(train[-1]['loss'])}.",
        '']
    for key in ('point_epe_m','mean_write_confidence','mean_frame_write_quality','write_confidence_mean'):
        if all(key in row for row in train):
            lines.append(f"Training `{key}`: {fmt(train[0][key])} → {fmt(train[-1][key])}.")
    lines+=['','World EPE entries below are mean / median / P90 across query frames. Point L1 and pose distributions are retained in metrics.csv. Pose pairs are mean R0→R1 [deg] and t0→t1 [m].','',
        '| Stream / frame count | Method | Scope | Queries | World EPE [m] | R0 → R1 [deg] | t0 → t1 [m] |',
        '|---|---|---|---:|---|---|---|']
    extras=[]
    for stream,plan in zip(summary['streams'],plans):
        if stream['plan']!=plan:raise ValueError('Summary plan differs from stream_plans.json')
        folder=evaluation/plan['name'];all_rows={}
        base=dict(stream=plan['name'],scene=plan['scene'],scene_group=scene_group(plan['scene']))
        for variant,filename in VARIANTS.items():
            data=read_json(folder/filename)
            if data['plan']!=plan:raise ValueError('Per-stream metrics have a different plan')
            rows=data['rows'];validate_rows(rows,plan,variant);all_rows[variant]=rows
            for phase in ('All',*PHASES):
                selected=rows if phase=='All' else [r for r in rows if r['phase']==phase]
                stats={key:metric_distribution(selected,key) if selected else None for key in METRICS}
                for key,value in stats.items():
                    tables.append(dict(category='query_distribution',**base,variant=variant,phase=phase,metric=key,
                        **(value or dict(count=0,mean=None,median=None,p90=None)),value=None))
                means=lambda key:None if stats[key] is None else stats[key]['mean']
                lines.append('| '+f"{plan['name']} / {len(plan['frame_ids'])}"+' | '+variant+' | '+phase+' | '+str(len(selected))+' | '+stats_text(stats['point_epe_m'])+' | '+fmt(means('rotation0_deg'))+' → '+fmt(means('rotation1_deg'))+' | '+fmt(means('translation0_m'))+' → '+fmt(means('translation1_m'))+' |')
            tail=rows[-1]
            closure=stream['variants'][variant].get('closure',{})
            detail=dict(actual_seed_to_endpoint_distance_m=closure.get('actual_seed_to_endpoint_distance_m'),
                predicted_seed_to_endpoint_distance_m=closure.get('predicted_seed_to_endpoint_distance_m'),closure_translation_error_m=tail['translation1_m'],closure_rotation_error_deg=tail['rotation1_deg'],
                final_public_gaussians=tail['memory_after'],maximum_public_gaussians=max(r['memory_after'] for r in rows),
                maximum_active_gaussians=max(r['active_count'] for r in rows),active_read_budget=2048,
                analytic_framewise_kv_tokens_at_end=(tail['t']+1)*192,
                mean_write_confidence=metric_distribution(rows,'write_confidence_mean'),
                mean_pose_gate=metric_distribution(rows,'refinement_gate'))
            loiter=[r for r in rows if r['phase']=='Loiter']
            detail.update(loiter_start_public=loiter[0]['memory_before'] if loiter else None,
                loiter_end_public=loiter[-1]['memory_after'] if loiter else None,
                loiter_public_change=loiter[-1]['memory_after']-loiter[0]['memory_before'] if loiter else None,
                loiter_analytic_kv_growth=(len(loiter)*192) if loiter else None)
            for phase in PHASES:
                phase_rows=[r for r in rows if r['phase']==phase]
                for key,value in dict(public_start=phase_rows[0]['memory_before'] if phase_rows else None,
                        public_end=phase_rows[-1]['memory_after'] if phase_rows else None,
                        public_change=phase_rows[-1]['memory_after']-phase_rows[0]['memory_before'] if phase_rows else None,
                        active_max=max(r['active_count'] for r in phase_rows) if phase_rows else None,
                        analytic_kv_growth=192*len(phase_rows) if phase_rows else None).items():
                    tables.append(dict(category='phase_memory_growth',**base,variant=variant,phase=phase,metric=key,
                        count=len(phase_rows),mean=None,median=None,p90=None,value=value))
            wall=stream['variants'][variant].get('wall',{})
            detail.update(wall_p90_p10_m=wall.get('wall_p90_p10_m'),wall_signed_bias_m=wall.get('wall_signed_bias_m'))
            for key,value in detail.items():
                tables.append(dict(category='stream_detail',**base,variant=variant,phase='All',metric=key,
                    **(value if isinstance(value,dict) else dict(count=None,mean=None,median=None,p90=None)),value=None if isinstance(value,dict) else value))
            extras+=['',f"## {plan['name']} · {variant}",'',
                f"Scene category: {base['scene_group']}. Source: {plan['kind']}.",
                f"Actual final frame {tail['frame_id']}; closure endpoint error {fmt(tail['translation1_m'])} m / {fmt(tail['rotation1_deg'])}°. Actual/predicted seed-to-endpoint distance: {fmt(detail['actual_seed_to_endpoint_distance_m'])}/{fmt(detail['predicted_seed_to_endpoint_distance_m'])} m. This compares the prediction to the real endpoint, not an assumed identity pose.",
                f"Public Gaussians final/max: {detail['final_public_gaussians']}/{detail['maximum_public_gaussians']}; maximum active read: {detail['maximum_active_gaussians']} (read budget 2048). Analytical retained-frame KV baseline at end: {detail['analytic_framewise_kv_tokens_at_end']} tokens = 192 × number of observed frames; no actual historical KV is retained.",
                f"Loiter public count: {detail['loiter_start_public']} → {detail['loiter_end_public']} (change {detail['loiter_public_change']}); analytical KV growth during loiter: {detail['loiter_analytic_kv_growth']}. N/A loiter counts mean the natural stream has no artificial loiter phase.",
                f"Mean write confidence distribution: {stats_text(detail['mean_write_confidence'])}; conservative pose gate: {stats_text(detail['mean_pose_gate'])}.",
                f"Fixed GT reference-plane wall P90–P10: {fmt(detail['wall_p90_p10_m'])} m; signed median wall bias: {fmt(detail['wall_signed_bias_m'])} m."]
            if wall.get('unavailable_reason'):extras.append('Wall section unavailable: '+wall['unavailable_reason']+'.')
            if plan['scene'].startswith('harvard_c11/') and 'natural' in plan['name']:
                extras+=['','C11 natural return frames (every occurrence, no selection):','',
                    '| Source frame | Stream t | World EPE [m] | R0 → R1 [deg] | t0 → t1 [m] | Public / active |','|---:|---:|---:|---|---|---|']
                for row in [r for r in rows if r['frame_id'] in (781,797)]:
                    extras.append(f"| {row['frame_id']} | {row['t']} | {fmt(row['point_epe_m'])} | {fmt(row['rotation0_deg'])} → {fmt(row['rotation1_deg'])} | {fmt(row['translation0_m'])} → {fmt(row['translation1_m'])} | {row['memory_after']} / {row['active_count']} |")
                    for key in METRICS:
                        if key in row:tables.append(dict(category='C11_natural_return',**base,variant=variant,phase=row['phase'],metric=f"frame_{row['frame_id']}_t{row['t']}_{key}",count=1,mean=None,median=None,p90=None,value=row[key]))
            extras+=['','Largest world-EPE query failures remain included:']
            for row in sorted(rows,key=lambda r:r['point_epe_m'],reverse=True)[:3]:
                extras.append(f"- t={row['t']}, source frame {row['frame_id']}, phase {row['phase']}: EPE {fmt(row['point_epe_m'])} m, translation {fmt(row['translation1_m'])} m, rotation {fmt(row['rotation1_deg'])}°.")
        causal=read_json(folder/'representative_interventions.json')
        extras+=['',f"## {plan['name']} · fixed-weight read-only interventions",'',
            '| Condition | World EPE [m] | Rotation [deg] | Translation [m] |','|---|---:|---:|---:|']
        for condition in ('full','empty_same_weights','erased_same_weights'):
            values=causal[condition]
            extras.append(f"| {condition} | {fmt(values['point_epe_m'])} | {fmt(values['rotation1_deg'])} | {fmt(values['translation1_m'])} |")
            for key,value in values.items():
                if isinstance(value,(float,int)):
                    tables.append(dict(category='representative_intervention',**base,variant=condition,phase='Revisit representative',metric=key,count=1,mean=None,median=None,p90=None,value=value))
        erasure=causal['erasure']
        inside,outside=erasure.get('epe_increase_inside_m'),erasure.get('epe_increase_outside_m')
        extras+=['',f"BBox ΔEPE inside / outside = {fmt(inside)} / {fmt(outside)} m; local/outside pixels {erasure.get('local_pixels')} / {erasure.get('outside_pixels')}; T0 shift {fmt(erasure.get('t0_change_m'))} m. Negative ΔEPE is retained as measured improvement; no local-causality success is inferred from a negative or near-zero denominator."]
        for key,value in erasure.items():
            if isinstance(value,(float,int)) or value is None:
                tables.append(dict(category='representative_erasure',**base,variant='erased_same_weights',phase='Revisit representative',metric=key,count=1,mean=None,median=None,p90=None,value=value))
    for key in ('loss','point_epe_m','point_l1_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m',
                'mean_write_confidence','mean_frame_write_quality','peak_allocated_gib','peak_reserved_gib'):
        if all(key in row for row in train):
            stats=distribution([row[key] for row in train])
            tables.append(dict(category='training_distribution',stream='all training clips',scene='train partition only',
                scene_group='heldout and val excluded',variant='Full persistent',phase='train optimizer updates',metric=key,**stats,value=None))
    lines+=extras
    lines+=['','No plateau is attributed to the 2048 active-read ceiling. Natural saturation must be assessed using the public-memory counts, especially their measured loiter and revisit changes. The analytical KV baseline counts all observed frames including the seed.','']
    return '\n'.join(lines),tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='results/step3_streaming_seed17')
    parser.add_argument('--evaluation',help='Default RUN/evaluation_final')
    parser.add_argument('--output',help='Default evaluation folder')
    args=parser.parse_args()
    run=PROJECT_ROOT/args.run
    evaluation=PROJECT_ROOT/args.evaluation if args.evaluation else run/'evaluation_final'
    output=PROJECT_ROOT/args.output if args.output else evaluation
    try:markdown,rows=build_report(evaluation,run/'metrics.jsonl')
    except (FileNotFoundError,KeyError,ValueError) as exc:parser.exit(2,'Step 3 quantitative report not complete: '+str(exc)+'\n')
    output.mkdir(parents=True,exist_ok=True)
    (output/'quantitative_summary.md').write_text(markdown)
    with (output/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=CSV_FIELDS);writer.writeheader();writer.writerows(rows)
    print('Wrote',output/'quantitative_summary.md','and',output/'metrics.csv')


if __name__=='__main__':main()
