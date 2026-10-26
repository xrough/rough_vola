"""The inner Bellman rule is fixed while outer weak paths are refined."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_bellman_policy import BellmanEvaluator
from rvlab.experiments._pricing_rough_bellman_delta import (
    bellman_policy_stops,bellman_delta_samples,summarize_bellman_delta)
from rvlab.pricing.rough_simulation import _time_grid

def test_refining_outer_grid_cannot_change_the_frozen_inner_operator(local_case):
    proxy,selected,trace=local_case;e=BellmanEvaluator(proxy,selected,orders=(16,))
    before=e.grid.copy();a=bellman_policy_stops(trace,e,outer_steps=8)
    refined={**trace,'actual_time_grid':_time_grid(proxy.params,proxy.market,proxy.exercise_times,16)}
    b=bellman_policy_stops(refined,e,outer_steps=16)
    np.testing.assert_array_equal(e.grid,before)
    np.testing.assert_array_equal(a['stops'],b['stops'])
    np.testing.assert_array_equal(a['computed_mask'],b['computed_mask'])
    np.testing.assert_allclose(a['continuation'],b['continuation'],rtol=0.,atol=0.,equal_nan=True)
    assert b['diagnostics']['operator_steps']==8 and b['diagnostics']['outer_steps']==16

def test_every_bump_recomputes_its_own_stopping_rule_and_ignores_poststop_states(local_case):
    proxy,selected,trace=local_case;relative=np.array([-.005,-.0025,0.,.0025,.005]);n=len(trace['stocks'])
    stocks=trace['stocks']*(1+relative)[None,None,:]
    sign=1 if proxy.is_call else -1
    stocks[:,1]=proxy.strike+sign*relative*proxy.market.spot
    stocks[:,2:]=proxy.strike+sign*10.
    trace={**trace,'stocks':stocks,'initial_spots':proxy.market.spot*(1+relative)}
    e=BellmanEvaluator(proxy,selected,orders=(16,));calls=[]
    def constant(j,s,u):
        calls.append((j,s.copy(),u.copy()))
        return {'continuation':np.full((len(s),1),.3),'diagnostics':{}}
    e.conditional_expectations=constant
    result=bellman_policy_stops(trace,e,outer_steps=8)
    np.testing.assert_array_equal(result['stops'],np.broadcast_to([2,2,2,2,1],(n,5)))
    assert len(calls)==2 and len(calls[0][1])==2*n and len(calls[1][1])==4*n
    assert not result['computed_mask'][:,2,4].any()
    assert np.isnan(result['continuation'][~result['computed_mask']]).all()
    assert np.isfinite(result['continuation'][result['computed_mask']]).all()

@pytest.mark.parametrize('kind',['policyseed','reservedseed','calendar','factorreset','order'])
def test_invalid_or_dependent_trace_fails(local_case,kind):
    proxy,selected,trace=local_case;e=BellmanEvaluator(proxy,selected,orders=(16,));trace=dict(trace);forbidden=()
    if kind=='policyseed':trace['seed']=selected.training_seeds[0]
    elif kind=='reservedseed':forbidden=(trace['seed'],)
    elif kind=='calendar':trace['actual_time_grid']=trace['actual_time_grid']+.001
    elif kind=='factorreset':trace['factors']=trace['factors'][:,:,:1]
    else:e.orders=(8,)
    with pytest.raises(ValueError):bellman_policy_stops(trace,e,outer_steps=8,forbidden_seeds=forbidden)

def test_nonfinite_continuation_is_identifiable_and_not_zero_substituted(local_case):
    proxy,selected,trace=local_case;e=BellmanEvaluator(proxy,selected,orders=(16,))
    e.conditional_expectations=lambda j,s,u:{'continuation':np.full((len(s),1),np.nan),'diagnostics':{}}
    with pytest.raises(FloatingPointError,match='Nonfinite'):bellman_policy_stops(trace,e,outer_steps=8)

def test_summaries_use_whole_path_paired_bump_and_control_differences():
    x=np.arange(1.,9.)/100.;arrays={}
    for prefix,shift in (('baseline',0.),('bellman',.01)):
        for estimator,extra in (('raw',.1),('controlled',0.)):
            arrays[prefix+'_'+estimator+'_deltas']=np.column_stack((-.4+x+shift+extra,-.4+.5*x+shift+extra))
    summary,comparisons=summarize_bellman_delta(arrays)
    assert len(summary)==8 and len(comparisons)==12
    bumps=comparisons.query("comparison == 'bump refinement'")
    np.testing.assert_allclose(bumps['mean'],-.5*x.mean(),atol=1e-15)
    controlled=comparisons.query("comparison == 'raw minus controlled'")
    np.testing.assert_allclose(controlled['mean'],.1,atol=1e-15)
    assert not summary.optimal_delta_ready.any()

def test_predeclared_rule_rejects_changed_original_operator_before_simulation(local_case):
    proxy,selected,_=local_case
    bundle={'proxy':proxy,'selected_policy':selected,'manifest':{'all_selection_training_seeds':()},
        'weights':SimpleNamespace(training_seed=915)}
    with pytest.raises(ValueError,match='original128'):
        bellman_delta_samples(bundle,seed=916,paths=2,steps=256)

def test_stock_derivative_of_bellman_operator_in_constant_variance_limit(local_case):
    from rvlab.tests.test_pricing_rough_bellman_policy import ConstantPremium
    from rvlab.pricing import DiscountCurve,BlackScholesParams
    from rvlab.pricing.black_scholes import black_scholes_price
    from scipy.special import ndtr
    proxy,selected,_=local_case;zero=tuple(ConstantPremium() for _ in selected.times)
    params=replace(proxy.params,v0=.04,theta=.04,vol_of_vol=0.)
    market=replace(proxy.market,discount_curve=DiscountCurve.flat(0.,1.))
    proxy=replace(proxy,params=params,market=market,continuations=zero,value_premiums=zero)
    selected=replace(selected,base_policy=proxy,continuations=zero)
    evaluator=BellmanEvaluator(proxy,selected,orders=(16,));j=1;tau=evaluator.times[-1]-evaluator.times[j]
    for bump in (.5,.25):
        stocks=np.array([100-bump,100+bump]);factors=np.zeros((2,len(proxy.kernel.nodes)))
        values=evaluator.conditional_expectations(j,stocks,factors)['continuation'][:,0]
        expected=black_scholes_price(BlackScholesParams(.2),forward=stocks,discount=1.,
            strike=100.,maturity=tau,is_call=proxy.is_call)
        actual=(values[1]-values[0])/(2*bump);reference=(expected[1]-expected[0])/(2*bump)
        assert actual==pytest.approx(reference,abs=2e-10)
        analytic=ndtr(.5*.2*np.sqrt(tau))-(not proxy.is_call)
        assert abs(actual-analytic)<.002+.005*abs(analytic)
