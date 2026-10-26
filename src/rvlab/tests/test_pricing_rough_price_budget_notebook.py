"""Price-budget displays preserve the complete record without invented readiness."""
from pathlib import Path
import pandas as pd
import pytest


def _cells():
    nbformat=pytest.importorskip('nbformat')
    path=Path(__file__).resolve().parents[3]/'notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb'
    notebook=nbformat.read(path,as_version=4)
    cells=[c.source for c in notebook.cells if c.cell_type=='code' and
        c.source.startswith('if results["manifest"].get("rough_price_budget_included"')]
    assert len(cells)==3
    return cells


def _env(manifest,tables,frames,text):
    return {'results':{'manifest':manifest,'figures':{}},'tables':tables,'pd':pd,
        'display':text.append,'Markdown':str,
        'present_table':lambda table,**options:frames.append((table.copy(),options)),
        'present_figure':lambda *a,**k:pytest.fail('No figures in this fixture')}


def test_disabled_price_budget_needs_no_new_estimates():
    frames,text=[],[];env=_env({}, {},frames,text)
    for cell in _cells():exec(cell,env)
    assert not frames
    assert text==['The price-budget attribution and common-control study is disabled in this execution.']


def test_failed_attempt_stays_visible_without_a_fabricated_price_or_uncertainty():
    failures=pd.DataFrame([{'error':'bounded common-control attempt failed'}])
    frames,text=[],[];env=_env({'rough_price_budget_executed':True},
        {'rough_price_budget_failures':failures},frames,text)
    for cell in _cells():exec(cell,env)
    assert any('No completed independent price-budget evidence' in value for value in text)
    assert len(frames)==1 and frames[0][1]['collapsed'] is True
    pd.testing.assert_frame_equal(frames[0][0],failures)


def test_main_budget_keeps_primary_failure_and_every_final_estimator_and_technical_row():
    data=pd.DataFrame([{'estimator':name,'primary':i==1,'baseline':i==0,'paths':8192,
        'price':2.86,'price_se':.0002,'upper':2.861,'upper_se':.0002,
        'gap':.001,'gap_se':.00005,'lower_sampling_term':.000392,
        'gap_sampling_term':.000098,'price_error_budget':.00149,
        'price_allowance':.001286,'price_budget_pass':False}
        for i,name in enumerate(('unchanged control','shared control'))])
    changes=pd.DataFrame([{'estimator':'shared control','primary':True,'metric':metric,
        'estimate':0.,'interval_low':0.,'interval_high':0.}
        for metric in ('lower change','upper change','gap change')])
    full=pd.DataFrame({'record':range(200),'all_paths_retained':True})
    frames,text=[],[];env=_env({'rough_price_budget_executed':True},
        {'rough_price_budget_price_comparison':data,'rough_price_budget_paired_changes':changes,
        'rough_price_budget_full_record':full},frames,text)
    for cell in _cells():exec(cell,env)
    main=next(frame for frame,k in frames if k['title']=='What consumes the unchanged price budget?')
    pd.testing.assert_frame_equal(main,data)
    assert not main.price_budget_pass.any() and set(main.paths)=={8192}
    paired=next(frame for frame,k in frames if k['title']=='A common endpoint shift is not a new exercise policy')
    pd.testing.assert_frame_equal(paired,changes)
    original=next(frame for frame,k in frames if k['title'].endswith('· price comparison'))
    pd.testing.assert_frame_equal(original,data)
    record,options=next((frame,k) for frame,k in frames if k['title'].endswith('· full record'))
    pd.testing.assert_frame_equal(record,full)
    assert len(record)==200 and options['max_rows']==30
    assert '30 of 200' in options['caption'] and 'not an acceptance subset' in options['caption']


def test_development_figure_is_labelled_spent_and_all_fifty_trial_rows_remain_visible():
    trials=pd.DataFrame({'trial_row':range(50),'used_for_fresh_acceptance':False})
    frames,text=[],[];env=_env({'rough_price_budget_included':True},
        {'rough_price_budget_spent_trials':trials},frames,text)
    shown=[];env['results']['figures']['rough_price_budget_spent_control_trials']=object()
    env['present_figure']=lambda result,name,**options:shown.append((name,options))
    for cell in _cells():exec(cell,env)
    assert len(shown)==1 and shown[0][0]=='rough_price_budget_spent_control_trials'
    caption=shown[0][1]['caption']
    assert 'Spent development evidence only' in caption and 'not confidence intervals' in caption
    assert 'frozen before fresh paths' in caption and 'Unknown-mean Z projections are excluded' in caption
    original,options=next((frame,k) for frame,k in frames if k['title'].endswith('· spent trials'))
    pd.testing.assert_frame_equal(original,trials)
    assert options['collapsed'] is True and options['max_rows']==50
