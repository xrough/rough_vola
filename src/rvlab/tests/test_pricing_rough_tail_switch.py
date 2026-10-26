"""Tail switches keep predictable centering and the poststop lower identity."""
import numpy as np
import pytest
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_tail_switch import tail_switch_corrections,FEATURES


def current(trace):return trace['european_martingale'][:,:,0]+.7*trace['premium_martingale'][:,:,0]


def test_switches_retain_all_states_and_exact_poststop_lower_identity(local_case):
    proxy,selected,trace=local_case;m=current(trace)
    before={k:trace[k].copy() for k in ('stocks','factors','discounted_payoffs')}
    result=tail_switch_corrections(proxy,selected,trace,m)
    c=result['corrections'];stops=result['stops'];path=np.arange(len(m))
    assert c.shape==m.shape+(len(FEATURES),) and np.all(c[:,0]==0)
    assert np.all(c[path,stops,:3]==0)
    # This identity holds for any deterministic combination, not just a
    # coefficient eventually preferred by an optimizer.
    added=c[:,:,:3]@np.array([.2,.3,.5])
    raw=trace['discounted_payoffs'][path,stops,0]
    np.testing.assert_array_equal(raw-m[path,stops],raw-(m+added)[path,stops])
    for key,value in before.items():np.testing.assert_array_equal(trace[key],value)


def test_switch_increment_arithmetic_and_left_gates(local_case):
    proxy,selected,trace=local_case;m=current(trace)
    r=tail_switch_corrections(proxy,selected,trace,m)
    diff=np.diff(trace['intrinsic_martingale'][:,:,0]-m,axis=1)
    eu=np.diff(trace['european_martingale'][:,:,0]-m,axis=1)
    expected=r['gates']*np.concatenate((np.repeat(diff[...,None],6,axis=-1),eu[...,None]),axis=-1)
    np.testing.assert_allclose(np.diff(r['corrections'],axis=1),expected,atol=2e-15,rtol=0.)
    assert np.all((r['gates']>=0)&(r['gates']<=1))
    for i,tau in enumerate(r['stops']):assert np.all(r['gates'][i,:tau,:3]==0)


def test_future_martingale_changes_cannot_affect_earlier_corrections(local_case):
    proxy,selected,trace=local_case;m=current(trace)
    first=tail_switch_corrections(proxy,selected,trace,m)
    changed={**trace,'intrinsic_martingale':trace['intrinsic_martingale'].copy()}
    changed['intrinsic_martingale'][:,-1]+=100.
    second=tail_switch_corrections(proxy,selected,changed,m)
    np.testing.assert_array_equal(first['gates'],second['gates'])
    np.testing.assert_array_equal(first['corrections'][:,:-1],second['corrections'][:,:-1])


@pytest.mark.parametrize('mutation',['seed','calendar','spot','nonfinite','origin'])
def test_trace_and_independence_guards(local_case,mutation):
    proxy,selected,trace=local_case;trace=dict(trace);m=current(trace).copy();forbidden=()
    if mutation=='seed':forbidden=(trace['seed'],)
    elif mutation=='calendar':trace['actual_time_grid']=trace['actual_time_grid']+.0001
    elif mutation=='spot':trace['initial_spots']=np.array([101.])
    elif mutation=='nonfinite':m[0,1]=np.nan
    elif mutation=='origin':m[0,0]=1.
    with pytest.raises(ValueError):tail_switch_corrections(proxy,selected,trace,m,forbidden_seeds=forbidden)
