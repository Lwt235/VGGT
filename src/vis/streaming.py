"""Streaming plots use saved predictions/3-D state, never alter model gauges."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vis.smoke import save, cloud, COLD, ORANGE

PHASES = ('Explore', 'Loiter', 'Explore New', 'Revisit')
PHASE_COLORS = {'Explore':'#cce5ff','Loiter':'#fff0be','Explore New':'#daf0d0','Revisit':'#ead9f4'}


def phase_spans(rows):
    """Half-open plotted intervals from the actual per-step phase labels."""
    if not rows:
        return []
    spans, start, label = [], rows[0]['t']-.5, rows[0]['phase']
    for previous, row in zip(rows, rows[1:]):
        if row['phase'] != label:
            spans.append((start, previous['t']+.5, label))
            start, label = row['t']-.5, row['phase']
    spans.append((start, rows[-1]['t']+.5, label))
    return spans


def shade_phases(ax, rows):
    for left, right, phase in phase_spans(rows):
        ax.axvspan(left, right, color=PHASE_COLORS.get(phase,'#eeeeee'), alpha=.38, zorder=-10)
        ax.text((left+right)/2,.98,phase,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7)


def summarize_rows(rows):
    """All predicted steps included; seed is not counted as a prediction."""
    keys=('point_epe_m','point_l1_m','rotation0_deg','rotation1_deg','translation0_m','translation1_m')
    def one(part):
        result={'queries':len(part)}
        for key in keys:
            values=[r[key] for r in part if r.get(key) is not None and np.isfinite(r[key])]
            result[key] = dict(mean=float(np.mean(values)),median=float(np.median(values)),p90=float(np.quantile(values,.9))) if values else None
        return result
    return {'all':one(rows),'phases':{phase:one([r for r in rows if r['phase']==phase]) for phase in dict.fromkeys(r['phase'] for r in rows)}}


def v6(folder, sequence, records, *, active_budget=2048, patch_tokens=192):
    """Theoretical KV counts are analytic, not retained by either engine."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig, axes=plt.subplots(3,1,figsize=(13,10),layout='constrained',sharex=True)
    for label,rows in records.items():
        t=np.array([r['t'] for r in rows])
        with_seed=np.r_[0,t]
        axes[0].plot(with_seed,[rows[0]['memory_before']]+[r['memory_after'] for r in rows],label=label+' public total')
        # Seed RGB-D writes the initial map; it does not perform a memory read.
        axes[1].plot(with_seed,[0]+[r['active_count'] for r in rows],label=label+' active read')
        axes[2].plot(t,[r['translation1_m'] for r in rows],label=label+' translation')
    base=next(iter(records.values()))
    t=np.r_[0,np.array([r['t'] for r in base])]
    axes[0].plot(t,(t+1)*patch_tokens,'--',color='#555555',label='Analytic frame KV tokens = (t+1) x patch count')
    axes[1].axhline(active_budget,ls='--',color='#555555',lw=1,label=f'Read budget {active_budget}; NOT public capacity')
    phase_rows=[dict(t=0,phase=base[0]['phase'])]+base
    for ax in axes:shade_phases(ax,phase_rows);ax.legend(fontsize=7,loc='upper left')
    axes[0].set(ylabel='3DGS primitives / 2D token count',title='V6 · total public memory vs analytical frame KV growth')
    axes[1].set(ylabel='Active Gaussian primitives',title='Active-read count is separate from persistent public memory')
    axes[2].set(ylabel='Unaligned translation error [m]',xlabel='Stream step t (seed at t=0)',title='Accumulated pose error in the fixed seed-camera metric gauge')
    fig.suptitle(sequence+' · real RGB frames; no historical RGB / 2D KV retained by engines')
    save(fig,folder/'V6_memory_growth_and_drift.png')


def metric_curves(folder, sequence, records):
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharex=True)
    for label,rows in records.items():
        x=[r['t'] for r in rows]
        axes[0,0].plot(x,[r['point_epe_m'] for r in rows],label=label)
        axes[0,1].plot(x,[r['rotation1_deg'] for r in rows],label=label)
        axes[1,0].plot(x,[r.get('write_confidence_mean',0) for r in rows],label=label)
        axes[1,1].plot(x,[r.get('dustbin_mean',0) for r in rows],label=label)
    for ax,title in zip(axes.flat,('World EPE [m]','Rotation error [deg]','Write confidence','Dustbin probability')):
        ax.set(title=title,xlabel='Stream step t');shade_phases(ax,next(iter(records.values())));ax.legend(fontsize=7)
    fig.suptitle(sequence+' · all query frames retained, no alignment')
    save(fig,Path(folder)/'stream_metrics.png')


def fixed_wall(reference, target):
    """A GT-only reference diagnostic plane; never fitted to predictions."""
    p=reference.world_points.detach().cpu().numpy();v=reference.valid.detach().cpu().numpy().astype(bool)
    q=target.world_points.detach().cpu().numpy();valid=target.valid.detach().cpu().numpy().astype(bool)
    h,w=v.shape;candidates=[]
    for row in range(3):
        for col in range(4):
            patch=p[row*h//4:(row+1)*h//4,col*w//4:(col+1)*w//4][v[row*h//4:(row+1)*h//4,col*w//4:(col+1)*w//4]]
            if len(patch)<30:continue
            center=np.median(patch,axis=0)
            _,_,basis=np.linalg.svd(patch-center,full_matrices=False)
            normal=basis[-1]
            if abs(normal[1])>.75:continue
            selected=(np.abs((q-center)@normal)<.15)&valid
            if selected.sum()<30:continue
            signed=(patch-center)@normal
            candidates.append((float(np.ptp(np.quantile(signed,[.1,.9]))),center,normal,selected,signed))
    return min(candidates,key=lambda x:x[0]) if candidates else None


def v1_comparison(folder, sequence, reference, target, predictions, *, memories=None):
    """Same seed GT plane/cloud in both columns; outputs are never transformed."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    labels=list(predictions);wall=fixed_wall(reference,target);stats={}
    fig,axes=plt.subplots(2,len(labels),figsize=(6*len(labels),8),layout='constrained',squeeze=False)
    history=cloud(reference.world_points,reference.valid,5000)
    limits=[]
    for label in labels:
        p=cloud(predictions[label]['world_points'],target.valid,6000);limits.append(p)
    combined=np.concatenate([history,*limits]);low,high=np.quantile(combined[:,[0,2]],[.005,.995],axis=0)
    for col,label in enumerate(labels):
        p=limits[col];ax=axes[0,col]
        ax.scatter(history[:,0],history[:,2],s=.6,c=COLD,alpha=.45,label='Seed RGB-D reference (GT, plotting only)')
        if memories and label in memories:
            means=memories[label].means.detach().float().cpu().numpy()
            ax.scatter(means[:,0],means[:,2],s=.5,c='#5995ba',alpha=.2,label='Predicted persistent map centers')
        ax.scatter(p[:,0],p[:,2],s=.6,c=ORANGE,alpha=.5,label='Current predicted world surface')
        ax.set(xlabel='world x [m]',ylabel='world z [m]',title=label+' · direct metric world overlap',xlim=(low[0]-.1,high[0]+.1),ylim=(low[1]-.1,high[1]+.1));ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=6)
        ax=axes[1,col]
        if wall is None:
            ax.text(.05,.5,'No adequately sampled vertical seed/query shared wall ROI',wrap=True);ax.axis('off');stats[label]={'unavailable_reason':'No shared vertical seed/query wall ROI'}
        else:
            thickness,center,normal,mask,signed=wall
            pred=predictions[label]['world_points'].detach().float().cpu().numpy()
            error=(pred[mask]-center)@normal
            ax.hist(signed,bins=40,density=True,color=COLD,alpha=.5,label='Seed reference')
            ax.hist(error,bins=40,density=True,color=ORANGE,alpha=.6,label='Current query prediction')
            width=float(np.ptp(np.quantile(error,[.1,.9])))
            # A displaced but individually thin query wall must count as a
            # double layer against the seed. Balance the two surface samples
            # so unequal image coverage cannot hide either layer below P10.
            quantiles=np.linspace(0,1,1024)
            combined=np.r_[np.quantile(signed,quantiles),np.quantile(error,quantiles)]
            union_width=float(np.ptp(np.quantile(combined,[.1,.9])))
            stats[label]={'wall_p90_p10_m':width,'seed_query_union_p90_p10_m':union_width,
                'wall_signed_bias_m':float(np.median(error)),'wall_pixels':int(mask.sum()),'reference_wall_p90_p10_m':thickness,'reference_plane_center':center.tolist(),'reference_plane_normal':normal.tolist()}
            ax.set(xlabel='Signed distance to SAME fixed seed plane [m]',ylabel='Density',
                title=f'P90-P10: query {width:.3f} m; balanced seed/query union {union_width:.3f} m');ax.legend(fontsize=7)
    fig.suptitle('V1 · '+sequence+' · Full vs Sliding Window; no output fitting / alignment')
    save(fig,folder/'V1_persistent_vs_sliding_window.png')
    return stats


def training_curves(folder, rows):
    if not rows:return
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key in zip(axes,('loss','point_epe_m','mean_write_confidence')):
        values=[r for r in rows if key in r]
        ax.plot([r['step'] for r in values],[r[key] for r in values],alpha=.3)
        if len(values)>=40:
            ax.plot([r['step'] for r in values][39:],np.convolve([r[key] for r in values],np.ones(40)/40,'valid'))
        ax.set(title=key,xlabel='Streaming optimizer updates')
    fig.suptitle('Step 3 · train scenes only, heldout frames excluded')
    save(fig,folder/'stream_training_curves.png')
