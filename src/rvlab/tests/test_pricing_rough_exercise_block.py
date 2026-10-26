"""Composed exact weak laws do not coarsen physical paths or exercise dates."""
from dataclasses import replace
import numpy as np
import pytest
from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law,block_control_trace
from rvlab.experiments._pricing_rough_foundation import _market
from rvlab.pricing import RoughHestonParams,approximate_fractional_kernel
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy,value_fourier_stopping_policy
from rvlab.pricing.rough_estimators import _conditional_payoffs

@pytest.fixture(scope='module')
def policy():
    p=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    k=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    c=RoughAmericanConfig(training_paths=48,valuation_paths=64,steps=4,exercise_steps=4,
        batch_size=16,training_seed=823,valuation_seed=824,stock_knots=(.8,.95,1.,1.1),premium_cap=.7)
    return train_fourier_stopping_policy(p,_market(),k,strike=100.,maturity=60/365,config=c,premium_cap=.7)

def test_one_step_block_matches_complete_core_trace(policy):
    original=value_fourier_stopping_policy(policy,trace=True)['trace']
    changed=block_control_trace(policy,original,batch_size=16)
    for key in ('stocks','factors','discounted_payoffs','stop_indices','times','eligible'):
        np.testing.assert_array_equal(changed[key],original[key])
    for key in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale'):
        np.testing.assert_allclose(changed[key],original[key],rtol=0.,atol=4e-11)
    assert changed['block_max_physical_steps']==1
    assert changed['block_max_probability_error']<2e-15

def test_two_step_law_matches_independent_recursive_logstock_moments(policy):
    from rvlab.pricing.rough_simulation import _weak_transition_branches,_drift_flow_with_integral,_theta
    t=np.array([0.,.001,.003]);f=np.array([[0.,0.],[.0002,.0005]])
    law=exercise_block_law(policy,f,times=t)
    assert law['new_factors'].shape==(2,9,2) and law['orthogonal'].shape==(2,9,4)
    p=policy.params;nodes=np.array(policy.kernel.nodes);w=np.array(policy.kernel.weights)
    matrix=-np.diag(nodes)-p.kappa*np.ones((2,1))*w
    def step(old,left,right):
        dt=right-left;theta=_theta(p,(left+right)/2)
        flow=_drift_flow_with_integral(matrix,np.full(2,p.kappa*(theta-p.v0)),w,p.v0,dt/2)
        growth=np.log(policy.market.discount_curve.discount(left)/policy.market.discount_curve.discount(right))
        return _weak_transition_branches(old,p,nodes=nodes,weights=w,flow=flow[:2,:2],shift=flow[:2,-1],integrated_flow=flow,dt=dt,theta=theta,growth=growth,stock_integral=policy.config.stock_integral)
    first=step(f,t[0],t[1])
    for power in (1.,2.,.5):
        reference=np.zeros(2)
        for b in range(3):
            second=step(first['new_factors'][:,b],t[1],t[2])
            for c in range(3):
                for o in range(2):
                    for q in range(2):
                        clock=first['orthogonal'][:,b,o]+second['orthogonal'][:,c,q]
                        shift=first['correlated'][:,b]+second['correlated'][:,c]-.5*clock
                        reference+=first['probabilities'][:,b]*second['probabilities'][:,c]*.25*np.exp(power*shift+.5*power*power*clock)
        actual=np.sum(law['probabilities']*np.exp(power*law['log_stock_shifts']+.5*power*power*law['orthogonal']).mean(axis=2),axis=1)
        np.testing.assert_allclose(actual,reference,atol=2e-15,rtol=2e-15)

def test_constant_variance_block_is_exact_gaussian_and_preserves_core_european_control(policy):
    from rvlab.pricing.rough_lift import ExponentialKernel
    p=replace(policy.params,v0=.04,theta=.04,vol_of_vol=0.,hurst=.5)
    k=ExponentialKernel((0.,),(1.,),.5,1.)
    c=replace(policy.config,steps=8,training_seed=825,valuation_seed=826)
    bs=train_fourier_stopping_policy(p,_market(),k,strike=100.,maturity=60/365,config=c,premium_cap=.7)
    law=exercise_block_law(bs,np.zeros((3,1)),times=[0.,.001,.003])
    np.testing.assert_allclose(law['log_stock_shifts'],(.04-.5*.04)*.003,rtol=0.,atol=1e-15)
    np.testing.assert_allclose(law['orthogonal'],.04*.003,rtol=0.,atol=1e-15)
    trace=value_fourier_stopping_policy(bs,trace=True)['trace'];changed=block_control_trace(bs,trace)
    np.testing.assert_allclose(changed['european_martingale'],trace['european_martingale'],atol=1e-11,rtol=0.)
    assert changed['block_max_physical_steps']==2

@pytest.mark.parametrize('times',[[0.],[0.,.1,.2,.3],[.1,0.],[0.,np.nan]])
def test_invalid_composition_is_identified(policy,times):
    with pytest.raises(ValueError):exercise_block_law(policy,np.zeros((2,2)),times=times)

def test_block_evaluator_rejects_three_physical_steps_and_keeps_source_unchanged(policy):
    c=replace(policy.config,steps=12)
    trace=value_fourier_stopping_policy(policy,config=c,trace=True)['trace']
    before=trace['premium_martingale'].copy()
    with pytest.raises(ValueError,match='one or two|physical law'):block_control_trace(policy,trace)
    np.testing.assert_array_equal(trace['premium_martingale'],before)


@pytest.mark.parametrize('field,value',[('strike',101.),('kernel_fingerprint','changed'),('policy_fingerprint','changed')])
def test_mismatched_physical_trace_is_rejected(policy,field,value):
    trace=dict(value_fourier_stopping_policy(policy,trace=True)['trace'])
    trace[field]=value
    with pytest.raises(ValueError,match='matching|identify'):
        block_control_trace(policy,trace)
