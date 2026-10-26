"""Report summaries preserve saved training choices and adverse validation errors."""
from copy import deepcopy

import pandas as pd
import pytest

from rvlab.experiments._pricing_first_stage_reporting import (
    BASE_CASES,add_first_stage_reporting,evidence_status,generic_case_summary,literature_sources,
    matched_maturity_recovery,method_matrix,selected_calibrations)


def _tables():
    attempts=pd.DataFrame([{'model':'heston','dataset':'noisy','start':i,'stage':stage,
        'test_clean_rmse':error,'status':'exhausted' if i==1 else 'converged'}
        for i,stage,error in ((1,'exploratory',.00001),(2,'exploratory',.1),(2,'continuation',.2))])
    choice=pd.DataFrame([{'model':'heston','dataset':'noisy','selected_start':2,
        'continuation_executed':True,'heldout_used_for_selection':False}])
    recovery=pd.DataFrame([{'design':d,'attempt_id':d+str(i),'split':split,
        'horizon_cohort':'common_14_30_60','contract_count':6,'clean_rmse':value,
        'refined_clean_rmse':refined}
        for d in ('short','augmented') for i in (1,2) for split in ('train','heldout')
        for value,refined in ((.0025 if d=='short' else .0024,.0023 if d=='short' else .0026),)])
    mchoice=pd.DataFrame([{'design':d,'selected_attempt_id':d+'2',
        'heldout_used_for_selection':False} for d in ('short','augmented')])
    return {'american_joint_summary':attempts,'american_joint_selection':choice,
        'maturity_extension_common_horizon_recovery':recovery,'maturity_extension_selection':mchoice}


def test_saved_training_stage_wins_even_with_worse_heldout_error():
    tables=_tables();before=deepcopy(tables);out=selected_calibrations(tables)
    assert len(out)==1 and out.iloc[0].start==2 and out.iloc[0].stage=='continuation'
    assert out.iloc[0].test_clean_rmse==.2
    for key in tables:pd.testing.assert_frame_equal(tables[key],before[key])
    out.loc[0,'test_clean_rmse']=999
    pd.testing.assert_frame_equal(tables['american_joint_summary'],before['american_joint_summary'])


def test_matched_maturity_summary_preserves_adverse_refined_comparison():
    out=matched_maturity_recovery(_tables())
    assert list(out.design)==['short','augmented']
    assert list(out.attempt_id)==['short2','augmented2']
    assert out.iloc[1].clean_rmse<out.iloc[0].clean_rmse
    assert out.iloc[1].refined_clean_rmse>out.iloc[0].refined_clean_rmse
    assert out.contract_count.eq(6).all()


@pytest.mark.parametrize('which',['calibration','maturity'])
def test_heldout_selected_or_missing_attempts_are_rejected(which):
    tables=_tables()
    name='american_joint_selection' if which=='calibration' else 'maturity_extension_selection'
    func=selected_calibrations if which=='calibration' else matched_maturity_recovery
    tables[name]['heldout_used_for_selection']=True
    with pytest.raises(ValueError,match='training-only'):func(tables)
    tables[name]['heldout_used_for_selection']=False
    tables[name][('selected_start' if which=='calibration' else 'selected_attempt_id')]='missing'
    with pytest.raises(ValueError,match='unique'):func(tables)


def test_method_and_source_tables_state_the_actual_target_without_novelty_claims():
    methods=method_matrix();sources=literature_sources()
    assert len(methods)==6 and len(sources)==7
    assert sources.source.str.startswith('https://').all()
    assert 'finite' in ' '.join(methods.target).lower()
    assert 'not a generic six-case' in ' '.join(methods.scope)
    assert 'mean-reversion level' in ' '.join(methods.scope)
    assert 'no optimal-American Greek claim' in ' '.join(methods.scope)
    assert selected_calibrations({}).empty and matched_maturity_recovery({}).empty


def test_report_completion_does_not_promote_failed_gates_or_readiness():
    manifest={'rough_benchmark':{'complete':True,'all_price_gates_pass':False},
        'historical_focused_evidence':{'rough_mc_precision':{'sampling_gate_pass':True},
            'rough_shared_delta':{'fixed_policy_delta_checks_pass':True}},'stock_delta_ready':False}
    result={'manifest':manifest,'tables':_tables(),'figures':{}};before=deepcopy(result)
    out=add_first_stage_reporting(result)
    status=out['tables']['report_evidence_status'].set_index('evidence')
    assert status.loc['Six-case generic panel','recorded_state']=='Complete'
    assert 'does not mean all' in status.loc['Six-case generic panel','interpretation']
    assert status.loc['Broader market/hedging readiness','recorded_state']=='False'
    assert out['manifest']==before['manifest']
    for key in result['tables']:pd.testing.assert_frame_equal(result['tables'][key],before['tables'][key])
    assert not result['figures'] and evidence_status({}).empty


def _generic():
    prices=pd.DataFrame([{'trial_id':case,'control':control,'primary':control=='european_premium',
        'paths':1024,'price':2.+i,'upper':2.01+i,'price_se':.001*(3-c),
        'upper_se':.001,'gap':.01,'gap_se':.0002,'price_error_budget':.015-c*.001,
        'price_allowance':.0012,'price_budget_pass':False}
        for i,case in enumerate(BASE_CASES) for c,control in enumerate(('raw','european','european_premium'))])
    status=pd.DataFrame([{'trial_id':case,'complete':True,'declared_paths':1024} for case in BASE_CASES])
    refinement=pd.DataFrame([{'coarse_trial':'put_90d_k100','fine_trial':'put_90d_k100_'+kind,
        'same_full_policy':kind=='physical','control':'european_premium','primary':True,
        'quantity':quantity,'bump_relative':bump,'change_plus_uncertainty':.002,
        'allowance':.001,'complete':True,'refinement_check_pass':False}
        for kind in ('physical','exercise','kernel','joint')
        for quantity,bump in (('price',0.),('delta',.005),('delta',.0025))])
    return {'rough_benchmark_price_bounds':prices,'rough_benchmark_case_status':status,
        'rough_benchmark_refinement':refinement}


def test_generic_figures_keep_all_cases_controls_and_construction_refinements():
    import matplotlib.pyplot as plt
    tables=_generic();before=deepcopy(tables)
    result=add_first_stage_reporting({'tables':tables,'manifest':{},'figures':{}})
    assert list(result['tables']['report_generic_cases'].trial_id)==list(BASE_CASES)
    assert len(result['figures'])==3
    controls=result['figures']['rough_benchmark_generic_control_comparison']
    assert [len(line.get_xdata()) for line in controls.axes[0].lines]==[6,6,6]
    refinement=result['figures']['rough_benchmark_generic_refinement_checks'].axes[0]
    assert len(refinement.patches)==12 and all(p.get_width()==2. for p in refinement.patches)
    assert refinement.get_xscale()=='symlog'
    assert any('fixed full policy' in t.get_text() for t in refinement.get_yticklabels())
    assert any('retrained construction' in t.get_text() for t in refinement.get_yticklabels())
    for key in tables:pd.testing.assert_frame_equal(tables[key],before[key])
    for fig in result['figures'].values():plt.close(fig)


def test_missing_or_incomplete_base_case_cannot_be_dropped_or_filled_with_zero():
    tables=_generic();case=BASE_CASES[0]
    tables['rough_benchmark_case_status'].loc[0,'complete']=False
    with pytest.raises(ValueError,match='partial control'):generic_case_summary(tables)
    tables['rough_benchmark_price_bounds']=tables['rough_benchmark_price_bounds'].loc[
        lambda t:~t.trial_id.eq(case)]
    out=generic_case_summary(tables)
    assert len(out)==6 and out.iloc[0].price!=out.iloc[0].price and not out.iloc[0].complete
    tables['rough_benchmark_case_status']=tables['rough_benchmark_case_status'].iloc[1:]
    with pytest.raises(ValueError,match='every declared'):generic_case_summary(tables)


def test_zero_variance_fixed_policy_keeps_case_and_reports_undefined_ratio():
    import matplotlib.pyplot as plt
    tables=_generic();tables['rough_benchmark_price_bounds'].loc[
        lambda t:t.trial_id.eq(BASE_CASES[0]),'price_se']=0.
    result=add_first_stage_reporting({'tables':tables,'manifest':{},'figures':{}})
    axis=result['figures']['rough_benchmark_generic_control_comparison'].axes[0]
    assert all(pd.isna(line.get_xdata()[0]) for line in axis.lines)
    assert any('zero raw SE' in text.get_text() for text in axis.texts)
    assert len(result['tables']['report_generic_cases'])==6
    for fig in result['figures'].values():plt.close(fig)


def _generic_delta():
    tables=_generic()
    tables['rough_benchmark_delta_sampling']=pd.DataFrame([
        {'trial_id':case,'control':control,'primary':control=='european_premium',
         'bump_relative':bump,'estimate':-.5+case_index*.01+bump,
         'half_width':.02,'delta_allowance':.004,'sampling_pass':False}
        for case_index,case in enumerate(BASE_CASES)
        for control in ('raw','european','european_premium') for bump in (.005,.0025)])
    tables['rough_benchmark_bump_refinement']=pd.DataFrame([
        {'trial_id':case,'control':'european_premium','primary':True,'estimate':.002,
         'change_plus_uncertainty':.005,'tolerance_resolved':False} for case in BASE_CASES])
    return tables


def test_six_case_delta_summary_keeps_fixed_small_bump_and_failed_gates():
    tables=_generic_delta();before=deepcopy(tables);result=generic_case_summary(tables)
    assert list(result.trial_id)==list(BASE_CASES)
    assert list(result.delta)==pytest.approx([-.5+i*.01+.0025 for i in range(6)])
    assert not result.delta_sampling_pass.any() and not result.bump_pass.any()
    assert result.bump_budget.eq(.005).all()
    for key in tables:pd.testing.assert_frame_equal(tables[key],before[key])


def test_complete_case_cannot_silently_lose_or_reselect_delta_or_bump():
    tables=_generic_delta();tables['rough_benchmark_delta_sampling']=tables[
        'rough_benchmark_delta_sampling'].loc[lambda t:~(t.trial_id.eq(BASE_CASES[0])&t.bump_relative.eq(.0025))]
    with pytest.raises(ValueError,match='primary small-bump'):generic_case_summary(tables)
    tables=_generic_delta();tables['rough_benchmark_bump_refinement']=pd.concat([
        tables['rough_benchmark_bump_refinement'],tables['rough_benchmark_bump_refinement'].iloc[[0]]])
    with pytest.raises(ValueError,match='reselect a bump'):generic_case_summary(tables)
