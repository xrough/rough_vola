"""Conditional Bellman policies retain the real obstacle and fixed controls."""
from dataclasses import dataclass,replace
import numpy as np
import pytest
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_bellman_policy import (
    BellmanEvaluator,BellmanNumericalError,bellman_candidate_specs,evaluate_bellman_candidates,select_bellman_policy)

@dataclass(frozen=True)
class ConstantPremium:
    amount:float=0.
    knots:tuple=()
    def value(self,stocks,factors,strike):return np.full_like(stocks,self.amount,dtype=float)
    def coefficients(self,factors):
        result=np.zeros(np.shape(factors)[:-1]+(3,));result[...,0]=self.amount;return result

def test_endpoint_caps_only_premium_and_never_the_snell_envelope(local_case):
    proxy,selected,_=local_case;e=BellmanEvaluator(proxy,selected)
    class NegativeEuropean:
        def values(self,time,factors,means,clocks,active):return np.full(means.shape+(len(active),),-3.*e.discounts[2])
    e.affine=NegativeEuropean();c=e.endpoint_components(2,np.array([90.,100.,110.]),np.zeros((3,len(proxy.kernel.nodes))))
    assert np.all(c['premium']<=.7) and np.all(c['premium']>=0)
    np.testing.assert_array_equal(c['envelope'],c['payoff'])
    assert np.all(c['envelope']-c['european']>.7)

def test_exact_terminal_step_uses_conditional_payoff_not_quadrature(local_case):
    from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
    from rvlab.pricing.rough_estimators import _conditional_payoffs
    proxy,selected,trace=local_case;e=BellmanEvaluator(proxy,selected);j=len(e.times)-2
    s=trace['stocks'][:3,j,0];u=trace['factors'][:3,j];actual=e.conditional_expectations(j,s,u)
    a,b=np.searchsorted(e.grid,e.times[j:j+2]);law=exercise_block_law(proxy,u,times=e.grid[a:b+1])
    values=_conditional_payoffs(np.log(s)[:,None,None]+law['log_stock_shifts'],law['orthogonal'],proxy.strike,proxy.is_call)
    expected=e.discounts[j+1]/e.discounts[j]*np.sum(values.mean(axis=2)*law['probabilities'],axis=1)
    np.testing.assert_allclose(actual['continuation'],np.repeat(expected[:,None],3,axis=1),atol=2e-14,rtol=0.)
    np.testing.assert_array_equal(actual['continuation'],actual['direct_continuation'])

def test_zero_volvol_zero_rate_european_semigroup_and_no_exercise_limit(local_case):
    from rvlab.pricing import DiscountCurve,BlackScholesParams
    from rvlab.pricing.black_scholes import black_scholes_price
    proxy,selected,_=local_case;zero=tuple(ConstantPremium() for _ in selected.times)
    params=replace(proxy.params,v0=.04,theta=.04,vol_of_vol=0.)
    market=replace(proxy.market,discount_curve=DiscountCurve.flat(0.,1.))
    proxy=replace(proxy,params=params,market=market,continuations=zero,value_premiums=zero)
    selected=replace(selected,base_policy=proxy,continuations=zero)
    e=BellmanEvaluator(proxy,selected);stocks=np.array([90.,99.,101.]);factors=np.zeros((3,len(proxy.kernel.nodes)))
    for j in (1,len(e.times)-2):
        value=e.conditional_expectations(j,stocks,factors)['continuation']
        expected=black_scholes_price(BlackScholesParams(.2),forward=stocks,discount=1.,strike=100.,maturity=e.times[-1]-e.times[j],is_call=False)
        np.testing.assert_allclose(value,np.repeat(expected[:,None],3,axis=1),rtol=0.,atol=2e-10)

def test_trace_mask_common_upper_fallback_decisions_and_future_independence(local_case):
    from rvlab.pricing.rough_policy_diagnostics import _policy_stops
    proxy,selected,trace=local_case;e=BellmanEvaluator(proxy,selected,batch_size=8)
    before={k:trace[k].copy() for k in ('stocks','factors','european_proxy')};look=e.trace_continuations(trace)
    assert np.isnan(look['bellman_continuations'][~look['bellman_computed_mask']]).all()
    assert np.isfinite(look['bellman_continuations'][look['bellman_computed_mask']]).all()
    m=trace['european_martingale'][:,:,0];result=evaluate_bellman_candidates(trace,look,m,is_call=proxy.is_call)
    expected=_policy_stops(selected,trace['times'],trace['eligible'],trace['stocks'],trace['factors'],trace['european_proxy'])[:,0]
    np.testing.assert_array_equal(result['new_stops'][:,0],expected)
    assert result['new_lower'].shape==(32,7) and np.all(result['common_upper'][:,None]>=result['new_lower']-2e-11)
    changed={**trace,'stocks':trace['stocks'].copy(),'factors':trace['factors'].copy(),'european_proxy':trace['european_proxy'].copy()}
    changed['stocks'][:,-1]*=1.2;changed['factors'][:,-1]+=1.;changed['european_proxy'][:,-1]+=5.
    other=e.trace_continuations(changed)
    np.testing.assert_allclose(look['bellman_continuations'][:,:-1],other['bellman_continuations'][:,:-1],atol=0.,rtol=0.,equal_nan=True)
    for k,v in before.items():np.testing.assert_array_equal(trace[k],v)

def test_selection_keeps_fallback_and_still_names_a_nonbaseline_challenger():
    names=list(bellman_candidate_specs());lower=np.tile(-np.arange(7),(8,1)).astype(float)
    result={'new_lower':lower,'new_stops':np.full((8,7),64),'policy_order':names}
    selected,challenger,rows=select_bellman_policy(result,selection_seed=42)
    assert selected==names[0] and challenger==names[1] and rows.selected_challenger_before_final.sum()==1

def test_nonfinite_conditional_calculation_raises_instead_of_becoming_zero(local_case):
    proxy,selected,trace=local_case;e=BellmanEvaluator(proxy,selected)
    class InvalidEuropean:
        def values(self,time,factors,means,clocks,active):return np.full(means.shape+(len(active),),np.nan)
    e.affine=InvalidEuropean()
    with pytest.raises(BellmanNumericalError):e.conditional_expectations(1,trace['stocks'][:1,1,0],trace['factors'][:1,1])

@pytest.mark.parametrize('kind',['policyseed','controlseed','calendar','nonfinite'])
def test_invalid_or_reused_traces_fail(local_case,kind):
    proxy,selected,trace=local_case;trace=dict(trace);e=BellmanEvaluator(proxy,selected);forbidden=()
    if kind=='policyseed':trace['seed']=selected.training_seeds[0]
    elif kind=='controlseed':forbidden=(trace['seed'],)
    elif kind=='calendar':trace['actual_time_grid']=trace['actual_time_grid']+.001
    else:trace['european_proxy']=np.full_like(trace['european_proxy'],np.nan)
    with pytest.raises(ValueError):e.trace_continuations(trace,forbidden_seeds=forbidden)
