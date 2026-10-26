"""Unchanged finite-bump/time gates and honest representation contrasts."""
import copy

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_shared_delta_statistics import (
    RELATIVE,delta,summarize_shared_delta)


SPECS=[{'seed':1133005042,'steps':128,'paths':8,'representations':['original','micro']},
       {'seed':1134005042,'steps':128,'paths':8,'representations':['original','micro']},
       {'seed':1135005042,'steps':256,'paths':12,'representations':['micro']}]


def streams():
    result=[]
    for spec in SPECS:
        n=spec['paths'];x=100.*RELATIVE
        z=np.tile([-1.,1.],n//2);raw=2.86-.4*x[None,:]+.01*z[:,None]*(1.+x[None,:])
        stable=2.86-.4*x[None,:]+.001*z[:,None]*(1.+x[None,:])
        reps={}
        for rep in spec['representations']:
            slope=-.4+(.001 if rep=='micro' else 0.)+(.0005 if spec['steps']==256 else 0.)
            shared=2.86+slope*x[None,:]+.0005*z[:,None]*(1.+x[None,:])+.0005*x[None,:]**3
            reps[rep]={'raw':raw.copy(),'stable':stable.copy(),'shared':shared,
                       'stops':np.tile([1,2,64,2,1],(n,1))}
        result.append({'seed':spec['seed'],'steps':spec['steps'],'relative_spots':RELATIVE.copy(),
            'initial_spots':100.*(1.+RELATIVE),'representations':reps})
    return result


def test_whole_path_delta_SE_and_all_declared_representations_are_retained():
    s=streams();t=summarize_shared_delta(s,expected_streams=SPECS)
    assert len(t['delta_sampling'])==18 and len(t['bump_refinement'])==9
    assert len(t['raw_control'])==18 and len(t['representation_change'])==6
    assert len(t['time_refinement'])==6 and len(t['combined_refinement'])==6
    r=t['delta_sampling'].query("physical_steps == 128 and representation == 'original' and estimator == 'shared' and bump_relative == .0025").iloc[0]
    chunks=[delta(a,'original','shared',.0025) for a in s[:2]]
    expected=np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in chunks))/16
    assert r.paths==16 and r.standard_error==pytest.approx(expected)
    assert r.delta_allowance==pytest.approx(.002+.005*abs(r.estimate))
    assert not t['delta_sampling'].optimal_delta_ready.any()


def test_zero_in_interval_is_distinct_from_tolerance_equivalence():
    t=summarize_shared_delta(streams(),expected_streams=SPECS)
    r=t['representation_change'].query("estimator == 'shared' and bump_relative == .0025").iloc[0]
    assert r.estimate==pytest.approx(.001)
    assert not r.zero_in_interval and r.tolerance_bias_resolved
    assert r.change_plus_uncertainty<=r.delta_allowance


def test_bump_check_is_paired_and_time_check_uses_independent_resolution_SEs():
    s=streams();t=summarize_shared_delta(s,expected_streams=SPECS)
    row=t['bump_refinement'].query("physical_steps == 128 and representation == 'micro' and estimator == 'shared'").iloc[0]
    difference=[delta(x,'micro','shared',.0025)-delta(x,'micro','shared',.005) for x in s[:2]]
    se=np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in difference))/16
    assert row.standard_error==pytest.approx(se)
    assert row.change_plus_uncertainty==pytest.approx(abs(row.estimate)+1.96*se)
    a=t['delta_sampling'].query("physical_steps == 128 and representation == 'micro' and estimator == 'shared' and bump_relative == .0025").iloc[0]
    b=t['delta_sampling'].query("physical_steps == 256 and representation == 'micro' and estimator == 'shared' and bump_relative == .0025").iloc[0]
    time=t['time_refinement'].query("estimator == 'shared' and bump_relative == .0025").iloc[0]
    assert time.standard_error==pytest.approx(np.hypot(a.standard_error,b.standard_error))
    assert time.delta_change==pytest.approx(b.estimate-a.estimate)
    combined=t['combined_refinement'].query("estimator == 'shared' and bump_relative == .0025").iloc[0]
    assert combined.control_representation_changed and combined.physical_law_changed


def test_all_raw_control_diagnostics_and_own_stop_switches_remain_visible():
    t=summarize_shared_delta(streams(),expected_streams=SPECS)
    assert set(t['raw_control'].comparison)=={'raw minus shared','stable minus shared','raw minus stable'}
    assert 'tolerance_bias_resolved' in t['raw_control'] and 'zero_in_interval' in t['raw_control']
    assert t['raw_control'].marginal_interval.all()
    assert t['exercise_changes'].all_bumps_recomputed_own_stops.all()


@pytest.mark.parametrize('kind',['missingstream','reordered','reusedseed','missingrepresentation','missingpath',
    'missingestimator','nonfinite','changedspots','changedstops','changedraw','changedstable'])
def test_changed_path_pool_rule_or_comparator_is_rejected(kind):
    s=streams();spec=copy.deepcopy(SPECS)
    if kind=='missingstream':s.pop()
    elif kind=='reordered':s=s[::-1]
    elif kind=='reusedseed':spec[1]['seed']=spec[0]['seed'];s[1]['seed']=s[0]['seed']
    elif kind=='missingrepresentation':del s[0]['representations']['micro']
    elif kind=='missingpath':s[0]['representations']['original']['raw']=s[0]['representations']['original']['raw'][:-1]
    elif kind=='missingestimator':del s[0]['representations']['micro']['stable']
    elif kind=='nonfinite':s[0]['representations']['micro']['shared'][0,0]=np.nan
    elif kind=='changedspots':s[0]['relative_spots'][0]=-.006
    elif kind=='changedstops':s[0]['representations']['micro']['stops'][0,0]=3
    elif kind=='changedraw':s[0]['representations']['micro']['raw'][0,0]+=.1
    else:s[0]['representations']['micro']['stable'][0,0]+=.1
    with pytest.raises(ValueError):summarize_shared_delta(s,expected_streams=spec)

