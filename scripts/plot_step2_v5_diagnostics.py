#!/usr/bin/env python3
"""Plot saved predictions/LM energies; no model forward or query selection."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import configure_local_storage,PROJECT_ROOT
configure_local_storage()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=PROJECT_ROOT/'results'/'step2_v5_full_seed17'
d=json.loads((root/'robotics_depth_03000.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for r in d['new']['refinement_lm']:
 e=np.asarray(r['energy_history']);axes[0].plot(np.arange(len(e)),e/max(e[0],1e-12),'-o',ms=3,label=str(r['query']))
axes[0].set(xlabel='Refinement LM iteration',ylabel='Energy / initial energy',title='W1 objective: monotone for every query');axes[0].legend(fontsize=6,ncol=3)
for tag,label in [('old','v4'),('new','v5')]:
 p=d[tag]['query_pose'];axes[1].scatter([r['translation0_m'] for r in p],[r['translation1_m'] for r in p],label=label,alpha=.8)
axes[1].plot([0,2.2],[0,2.2],'--',c='grey');axes[1].set(xlabel='T0 translation error [m]',ylabel='T1 translation error [m]',title='GT evaluation after forward');axes[1].legend()
labels=['near_0.1_3m','mid_3_5m','far_5_10m']
for i,(tag,label) in enumerate([('old','v4'),('new','v5')]):
 axes[2].bar(np.arange(3)+(i-.5)*.35,[d[tag]['pixel_weighted_depth_bins'][b]['anchored_depth_l1_m'] for b in labels],width=.35,label=label)
axes[2].set_xticks(np.arange(3),['0.1–3 m','3–5 m','5–10 m']);axes[2].set(ylabel='Final camera depth L1 [m]',title='All 11 robotics queries');axes[2].legend()
fig.suptitle('Step2 v5 / 3000 updates / unchanged queries / no output alignment')
fig.savefig(root/'refinement_lm_diagnostic.png',dpi=160);plt.close(fig)
