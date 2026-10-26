"""Frozen Bellman control protocol and selection cannot use final observations."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rvlab.experiments._pricing_rough_bellman_control_collection import (
    read,validate_design,validate_selection,BASELINE)
from rvlab.experiments._pricing_rough_bellman_control_tables import summarize_control_streams,row


@pytest.fixture
def design():
    # Small protocol snapshot only; full archived path samples are not required.
    return read(Path(__file__).with_name('fixtures')/'bellman_control_protocol_2026_10_19.json')


def test_current_predeclared_design_is_supported(design):validate_design(design)


@pytest.mark.parametrize('key,value',[('frozen_policy','GH32 blend1'),('proxy_correction_bound',.1),
    ('physical_steps',256),('exercise_intervals',128),('extra_shrink_candidates',True),
    ('price_allowance','.002+1e-4*abs(V)'),('new_control_stock_delta_validated',True),
    ('nodal_european_grid','65 dates'),('all_target_primitives_retained',False),
    ('selection_rule','min final gap'),('proxy_fit_seed',1094005042),('operator_steps',256)])
def test_changed_protocol_rejected(design,key,value):
    design[key]=value
    with pytest.raises(ValueError):validate_design(design)


def test_prior_seed_overlap_rejected(design):
    design['forbidden_prior_seeds'].append(1094005042)
    with pytest.raises(ValueError,match='overlap'):validate_design(design)


def selection_case():
    controls={BASELINE:{},'challenger':{}}
    base={'lower':np.array([1.,1.,1.,1.]),'upper':np.array([1.1,1.2,1.3,1.4])}
    other={'lower':base['lower']+np.array([.01,-.01,.02,-.02]),'upper':base['upper']-.05}
    stream={'values':{BASELINE:base,'challenger':other}}
    seal={'selected_control':'challenger','designated_challenger':'challenger'}
    rows=[]
    for name,v in stream['values'].items():
        stat=row([base['upper']-base['lower']-v['upper']+v['lower']])
        rows.append({'control':name,'paired_gap_gain':stat['estimate'],'gap_gain_se':stat['standard_error'],
            'gain_low':stat['interval_low'],'gain_high':stat['interval_high'],'selection_score':stat['interval_low'],
            'selection_seed':1093005042,'final_paths_used_for_selection':False,
            'selected_control':name=='challenger','designated_challenger':name=='challenger'})
    return stream,controls,seal,pd.DataFrame(rows)


def test_selection_reconstructed_from_selection_only_paths():
    args=selection_case();frame=validate_selection(*args)
    assert frame.selected_control.tolist()==[False,True]


@pytest.mark.parametrize('key,value',[('gain_low',1.),('final_paths_used_for_selection',True),('selection_seed',1094005042)])
def test_selection_statistic_or_role_tampering_rejected(key,value):
    stream,controls,seal,frame=selection_case(); frame.loc[1,key]=value
    with pytest.raises(ValueError):validate_selection(stream,controls,seal,frame)


def test_final_price_budget_and_majority_use_joint_path_samples():
    streams=[]
    for seed in [1,2]:
        n=8; base={'lower':np.ones(n),'upper':np.ones(n)+np.linspace(.01,.03,n)}
        other={'lower':base['lower']+np.linspace(-.001,.001,n),'upper':base['upper']-.015}
        streams.append({'seed':seed,'values':{BASELINE:base,'challenger':other},'raw':np.ones(n),
            'targets':np.zeros((n,65,4)), 'pure_targets':{i:deepcopy(base) for i in range(4)}})
    selection={'controls':{BASELINE:{},'challenger':{}},'selected_control':'challenger','designated_challenger':'challenger'}
    tables=summarize_control_streams(streams,selection,['a','b','c','d'])
    frame=tables['gap_comparison']; selected=frame.loc[frame.selected_control].iloc[0]
    assert selected.price_error_budget==pytest.approx(selected.gap+1.96*(selected.price_se+selected.gap_se))
    assert not selected.optimal_value_ready and not selected.new_control_delta_validated
    majority=tables['gap_dual_majority'].iloc[1]
    assert majority.estimate==pytest.approx(.005)
    assert majority.dual_majority_demonstrated
    assert len(tables['gap_per_seed'])==4 and frame.paths.tolist()==[16,16]


def test_whole_strata_standard_error_keeps_within_seed_variance():
    chunks=[np.array([0.,2.]),np.array([10.,12.])]
    stat=row(chunks)
    assert stat['estimate']==6.
    assert stat['standard_error']==pytest.approx(np.sqrt(8.)/4.)


def test_duplicate_final_seed_rejected():
    with pytest.raises(ValueError):summarize_control_streams([{'seed':1},{'seed':1}],{},[])


def diagnostic_case():
    mask=np.zeros((3,65),bool); mask[:2,1]=True; mask[2,63]=True
    arrays={'actual_time_grid':np.linspace(0,1,129)}
    dates=[{'exercise_index':j,'states':int(mask[:,j].sum()),'orders':[16],'physical_increments':2,
        'terminal_exact_payoff':j==63,'only_premium_capped':True,'actual_future_state_used':False,
        'all_factor_branches_and_stock_orderings':True,'max_probability_error':1e-16,
        'max_cv_direct_difference':[.0001]} for j in (1,63)]
    diagnostic={'operator_steps':128,'outer_steps':128,'computed_states':3,
        'only_alive_ITM_evaluated':True,'all_paths_retained':True,'all_bumps_recompute_continuation_and_stops':True,
        'operator_grid':arrays['actual_time_grid'].tolist(),'dates':dates}
    return {'diagnostics':{'fixed_policy':diagnostic}}, {'computed_mask':mask}, arrays


def test_policy_diagnostics_independently_match_every_computed_date():
    from rvlab.experiments._pricing_rough_bellman_control_collection import validate_policy_diagnostics
    validate_policy_diagnostics(*diagnostic_case())


@pytest.mark.parametrize('key,value',[('operator_steps',256),('computed_states',2),
    ('only_alive_ITM_evaluated',False),('all_paths_retained',False),('dates',[])])
def test_policy_diagnostic_law_counts_and_coverage_cannot_change(key,value):
    from rvlab.experiments._pricing_rough_bellman_control_collection import validate_policy_diagnostics
    record,result,arrays=diagnostic_case();record['diagnostics']['fixed_policy'][key]=value
    with pytest.raises(ValueError):validate_policy_diagnostics(record,result,arrays)


@pytest.mark.parametrize('key,value',[('physical_increments',1),('actual_future_state_used',True),
    ('states',3),('orders',[32]),('only_premium_capped',False),('terminal_exact_payoff',True),
    ('max_probability_error',1e-3),('max_cv_direct_difference',[-1.])])
def test_conditional_operator_flags_cannot_change(key,value):
    from rvlab.experiments._pricing_rough_bellman_control_collection import validate_policy_diagnostics
    record,result,arrays=diagnostic_case();record['diagnostics']['fixed_policy']['dates'][0][key]=value
    with pytest.raises(ValueError):validate_policy_diagnostics(record,result,arrays)


@pytest.fixture
def proxy_lineage(tmp_path,monkeypatch):
    import json
    import rvlab.experiments._pricing_rough_bellman_control_collection as module
    output=tmp_path/'output'; proxy=output/'proxy';proxy.mkdir(parents=True)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    (tmp_path/'source.py').write_text('sealed source')
    (proxy/'source').mkdir();(proxy/'source/source.py').write_text('sealed source')
    names=['rough_larger_policy_v1/32768_paths_two_starts/'+n for n in
        ('frozen_policy_before_final.json','frozen_policy_coefficients.npz')]
    names+=['rough_bellman_policy_v1/one_step_v1/'+n for n in ('selection_before_final.json','manifest.json')]
    for name in names:
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('input')
    for i,seed in enumerate((1077005042,1078005042)):
        p=output/f'rough_bellman_policy_v1/one_step_v1/final_{i}_samples.npz';p.write_text('sample')
        record={'seed':seed,'paths':4096,'sample_sha256':module._sha(p)}
        q=p.with_name(f'final_{i}_manifest.json');q.write_text(json.dumps(record))
        names += [str(p.relative_to(output)),str(q.relative_to(output))]
    manifest={'input_sha256':{name:module._sha(output/name) for name in names},
        'source_sha256':{'source.py':module._sha(tmp_path/'source.py')},'created_at':'2026-01-01T00:00:00+00:00',
        'fit_seed':1077005042,'diagnostic_seed':1078005042,'new_outer_paths':0,'complete':True,
        'regression_row_clip_or_deletion':False}
    for name in ('manifest.json','proxy_before_fresh_paths.json','design_before_fit.json'):
        (proxy/name).write_text(json.dumps(manifest))
    return module,proxy,output


def test_nested_spent_proxy_lineage_has_exact_eight_immutable_inputs(proxy_lineage):
    m,p,o=proxy_lineage
    assert len(m.validate_proxy_lineage(p,o,'2026-01-02T00:00:00+00:00'))==8


@pytest.mark.parametrize('field,value',[('new_outer_paths',1),('fit_seed',1094005042),
    ('diagnostic_seed',1093005042),('regression_row_clip_or_deletion',True),('input_sha256',{})])
def test_proxy_training_role_input_set_cannot_change(proxy_lineage,field,value):
    import json
    m,p,o=proxy_lineage;path=p/'manifest.json';manifest=m.read(path);manifest[field]=value;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):m.validate_proxy_lineage(p,o,'2026-01-02T00:00:00+00:00')


def test_proxy_nested_sample_tampering_rejected(proxy_lineage):
    m,p,o=proxy_lineage
    (o/'rough_bellman_policy_v1/one_step_v1/final_0_samples.npz').write_text('altered')
    with pytest.raises(ValueError,match='nested'):m.validate_proxy_lineage(p,o,'2026-01-02T00:00:00+00:00')


def test_proxy_seal_must_precede_fresh_design(proxy_lineage):
    m,p,o=proxy_lineage
    with pytest.raises(ValueError):m.validate_proxy_lineage(p,o,'2025-01-02T00:00:00+00:00')
