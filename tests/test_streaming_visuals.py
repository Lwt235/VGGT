"""CPU tests of actual-frame stream plans and quantitative streaming plots."""
import numpy as np
import pytest
from scripts.evaluate_step3 import segmented_plan, default_plans, seed_surface_overlap
from src.vis.streaming import phase_spans, summarize_rows, v6, metric_curves


def rows():
    return [dict(t=i,phase=('Explore' if i<3 else 'Loiter' if i<5 else 'Explore New' if i<7 else 'Revisit'),
        point_epe_m=.1*i,point_l1_m=.05*i,rotation0_deg=i,rotation1_deg=i*.9,
        translation0_m=i*.01,translation1_m=i*.009,memory_before=min(20,i+9),memory_after=min(20,i+10),active_count=min(10,i+5)) for i in range(1,9)]


def test_segmented_real_frame_plan_is_explicit_and_complete():
    frames=[i*17+3 for i in range(52)]
    plan=segmented_plan(frames)
    assert len(plan['frame_ids'])==len(plan['phases'])==80
    assert set(plan['frame_ids'])<=set(frames)
    assert list(dict.fromkeys(plan['phases']))==['Explore','Loiter','Explore New','Revisit']
    assert plan['frame_ids'][0]==plan['frame_ids'][-1]==frames[0]
    assert all(plan['frame_ids'][i]==frames[j] for i,j in enumerate(plan['source_indices']))
    assert 'not natural' in plan['kind']


def test_natural_streams_never_drop_difficult_or_blind_frames():
    ids=['harvard_c11/hv_c11_2','harvard_robotics_lab/hv_s1_1','harvard_c5/hv_c5_1','harvard_c8/hv_c8_3']
    spec={'sequences':[dict(id=s,partition='val' if i<2 else 'train',frame_ids=list(range(52 if i==0 else 17))) for i,s in enumerate(ids)]}
    plans=default_plans(spec)
    assert plans[0]['frame_ids']==list(range(52))
    assert plans[1]['frame_ids']==list(range(17))
    assert plans[1]['phases']==['Natural']*17
    assert 'Loiter' not in plans[0]['phases']
    assert len(plans)==5
    # C11's natural return must not be relabelled as new outward exploration.
    c11=plans[2]
    assert c11['outward_end_source_index']==42
    assert max(c11['source_indices'])==42
    assert c11['source_indices'][59]==c11['source_indices'][60]==42


def test_phase_intervals_and_summary_keep_tail_errors():
    data=rows();data[-1]['point_epe_m']=12.
    spans=phase_spans(data)
    assert spans==[(.5,2.5,'Explore'),(2.5,4.5,'Loiter'),(4.5,6.5,'Explore New'),(6.5,8.5,'Revisit')]
    summary=summarize_rows(data)
    assert summary['all']['queries']==8
    assert summary['all']['point_epe_m']['mean']==pytest.approx(np.mean([r['point_epe_m'] for r in data]))
    assert summary['phases']['Revisit']['point_epe_m']['mean']>6


def test_v6_and_metrics_plot_public_total_separate_from_active(tmp_path):
    records={'Full persistent':rows(),'Sliding Window W=4':rows()}
    records['Full persistent'][-1]['memory_after']=2500
    v6(tmp_path,'Synthetic plot fixture, not a measured experiment',records)
    metric_curves(tmp_path,'Synthetic plot fixture',records)
    assert (tmp_path/'V6_memory_growth_and_drift.png').stat().st_size>20000
    assert (tmp_path/'stream_metrics.png').stat().st_size>10000


def test_wall_comparison_uses_one_fixed_gt_plane_and_keeps_ghost_layers(tmp_path):
    import torch
    from src.data import Supervision
    from src.vis.streaming import v1_comparison
    y,x=torch.meshgrid(torch.linspace(-1,1,32),torch.linspace(-1,1,48),indexing='ij')
    points=torch.stack((x,y,torch.ones_like(x)*3),-1)
    truth=Supervision(torch.ones_like(x)*3,torch.eye(4),points,torch.ones_like(x,dtype=torch.bool))
    full=points.clone();full[...,2]+=.005*torch.sin(x*5)
    ghost=points.clone();ghost[::2,:,2]+=.20
    translated=points.clone();translated[...,2]+=.20
    result=v1_comparison(tmp_path,'Synthetic fixture, not measured data',truth,truth,
        {'Full persistent':{'world_points':full},'Sliding Window W=4':{'world_points':ghost},
         'Thin displaced layer':{'world_points':translated}})
    assert result['Full persistent']['wall_p90_p10_m']<.02
    assert result['Sliding Window W=4']['wall_p90_p10_m']>.19
    assert result['Thin displaced layer']['wall_p90_p10_m']<1e-6
    assert result['Thin displaced layer']['seed_query_union_p90_p10_m']>.19
    assert result['Full persistent']['reference_plane_center']==result['Sliding Window W=4']['reference_plane_center']
    assert (tmp_path/'V1_persistent_vs_sliding_window.png').stat().st_size>20000


def test_training_curve_includes_actual_logged_write_confidence(tmp_path):
    from src.vis.streaming import training_curves
    data=[dict(step=i,loss=1/i,point_epe_m=.5/i,mean_write_confidence=.3+i*.001)
          for i in range(1,81)]
    training_curves(tmp_path/'new_training_directory',data)
    assert (tmp_path/'new_training_directory'/'stream_training_curves.png').stat().st_size>20000


def test_evaluator_seed_visibility_preserves_zero_overlap_and_exact_replay():
    from types import SimpleNamespace
    dataset=SimpleNamespace(sequences={'room':dict(frame_ids=[31,430,797],
        surface_overlap=[[0.,0.,.25],[0.,0.,0.],[.30,0.,0.]])})
    plan=dict(scene='room',frame_ids=[31,430,797,31])
    assert seed_surface_overlap(dataset,plan,430)==0.
    assert seed_surface_overlap(dataset,plan,797)==.30
    assert seed_surface_overlap(dataset,plan,31)==1.
