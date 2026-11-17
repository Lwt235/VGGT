"""CPU reporting tests use labelled synthetic artifacts, never real results."""
import json
import numpy as np
import pytest
from scripts.summarize_step3 import build_report,validate_rows,distribution


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    evaluation=tmp_path/'evaluation_final'
    plan=dict(name='harvard_c11__fixture__natural',scene='harvard_c11/hv_c11_2',frame_ids=[31,781,797],
        phases=['Explore','Revisit','Revisit'],kind='synthetic test fixture, not measured video')
    variants=('Full persistent','Sliding Window W=4')
    wall=dict(wall_p90_p10_m=.08,wall_signed_bias_m=-.02)
    write(evaluation/'stream_plans.json',[plan])
    write(evaluation/'summary.json',dict(checkpoint='synthetic',world_gauge='seed camera, no alignment',
        streams=[dict(plan=plan,variants={v:dict(wall=wall) for v in variants})]))
    for variant,name in zip(variants,('persistent_metrics.json','sliding_window_metrics.json')):
        rows=[]
        for t,frame in enumerate(plan['frame_ids'][1:],1):
            rows.append(dict(t=t,frame_id=frame,phase='Revisit',point_epe_m=.2 if t==1 else 9.,point_l1_m=.1 if t==1 else 4.,
                rotation0_deg=2.,rotation1_deg=3.,translation0_m=.1,translation1_m=.2 if t==1 else 8.,
                memory_before=30,memory_after=35,active_count=22,write_confidence_mean=.3,refinement_gate=.04))
        write(evaluation/plan['name']/name,dict(plan=plan,rows=rows))
    measured=dict(point_epe_m=.2,rotation1_deg=3.,translation1_m=.2)
    write(evaluation/plan['name']/'representative_interventions.json',dict(full=measured,
        empty_same_weights=dict(measured,point_epe_m=1.3),erased_same_weights=dict(measured,point_epe_m=.3),
        erasure=dict(local_pixels=35,outside_pixels=60,epe_increase_inside_m=.1,epe_increase_outside_m=-.005,t0_change_m=.008)))
    log=tmp_path/'metrics.jsonl'
    log.write_text(json.dumps(dict(step=1,loss=3.,point_epe_m=.5,mean_write_confidence=.2))+'\n'+json.dumps(dict(step=3000,loss=.8,point_epe_m=.2,mean_write_confidence=.4))+'\n')
    return evaluation,log,plan


def test_report_preserves_large_failures_return_frames_and_absent_phases(tmp_path):
    evaluation,log,plan=fixture(tmp_path)
    text,rows=build_report(evaluation,log)
    assert '781' in text and '797' in text
    assert '9.0000 m' in text
    assert 'unseen cross-scene' in text
    assert '0.1000 / -0.0050 m' in text
    assert 'Loiter public count: None → None' in text
    epe=next(r for r in rows if r['category']=='query_distribution' and r['phase']=='All' and r['metric']=='point_epe_m')
    assert epe['mean']==pytest.approx(4.6)
    assert epe['p90']>8
    loiter=next(r for r in rows if r['category']=='query_distribution' and r['phase']=='Loiter' and r['metric']=='point_epe_m')
    assert loiter['count']==0 and loiter['mean'] is None
    kv=next(r for r in rows if r['metric']=='analytic_framewise_kv_tokens_at_end')
    assert kv['value']==3*192


def test_missing_evaluation_is_explicit_and_never_fabricated(tmp_path):
    with pytest.raises(FileNotFoundError,match='Missing measured Step 3 artifact'):
        build_report(tmp_path/'missing',tmp_path/'missinglog')


def test_incomplete_stream_and_missing_intervention_are_rejected(tmp_path):
    evaluation,log,plan=fixture(tmp_path)
    path=evaluation/plan['name']/'persistent_metrics.json'
    data=json.loads(path.read_text());data['rows'].pop()
    write(path,data)
    with pytest.raises(ValueError,match='incomplete/duplicated'):
        build_report(evaluation,log)
    data['rows'].append(dict(data['rows'][0],t=2,frame_id=797))
    write(path,data)
    (evaluation/plan['name']/'representative_interventions.json').unlink()
    with pytest.raises(FileNotFoundError,match='representative_interventions'):
        build_report(evaluation,log)


def test_nonfinite_failure_is_an_error_instead_of_silent_row_exclusion():
    with pytest.raises(ValueError,match='Nonfinite measured'):
        distribution([.2,float('nan'),.3])
