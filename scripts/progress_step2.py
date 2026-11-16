#!/usr/bin/env python3
"""Compact read-only training monitor; no GPU context or administrative state."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'results'/a.run
file=root/'metrics.jsonl'
if not file.exists():print(json.dumps(dict(run=a.run,status='initializing')));raise SystemExit
rows=[]
for line in file.read_text().splitlines():
 try:rows.append(json.loads(line))
 except json.JSONDecodeError:pass  # concurrent writer may be mid-line
if not rows:print(json.dumps(dict(run=a.run,status='initial_evaluation')));raise SystemExit
recent=rows[-100:]
mean=lambda key:sum(r[key] for r in recent)/len(recent)
value=dict(run=a.run,step=rows[-1]['step'],loss_100=round(mean('loss'),4),world_epe_100_m=round(mean('point_epe_m'),4),peak_gib=round(max(r['peak_reserved_gib'] for r in rows),2))
if 'depth_confidence_mean' in recent[-1]:value['depth_confidence_100']=round(mean('depth_confidence_mean'),3)
vfile=root/'validation.jsonl'
if vfile.exists():
 validation=[json.loads(l) for l in vfile.read_text().splitlines()]
 value['latest_validation']=[{k:r[k] for k in ['step','scope','group','point_epe_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m']} for r in validation[-2:]]
print(json.dumps(value))
