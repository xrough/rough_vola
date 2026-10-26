"""Signed Gaussian integration and convex local-control safeguards."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from rvlab.experiments._pricing_rough_local_control import (
    signed_linear_expectation,_linear_coefficients,correction_basis,
    fit_correction_weights,correction_endpoints,local_correction_martingales)


@pytest.mark.parametrize("mean,clock",[(math.log(96),.014),(math.log(101),.001),(math.log(91),0.)])
def test_signed_spline_against_independent_scalar_gaussian(mean,clock):
    nodes=np.array([0.,.8,.95,1.,1.1,1.3]);values=np.array([-.2,.1,-.05,.12,-.08,.03])
    coefficient=_linear_coefficients(nodes,values)
    expected=signed_linear_expectation(mean,clock,coefficient,strike=100.,knots=nodes[1:])
    def value(z):return np.interp(np.exp(mean+np.sqrt(clock)*z)/100.,nodes,values)
    if clock==0:reference=value(0.)
    else:
        points=[-12.]+[(np.log(100*k)-mean)/np.sqrt(clock) for k in nodes[1:] if -12<(np.log(100*k)-mean)/np.sqrt(clock)<12]+[12.]
        reference=sum(quad(lambda z:value(z)*np.exp(-.5*z*z)/np.sqrt(2*np.pi),a,b,epsabs=1e-12)[0] for a,b in zip(points[:-1],points[1:]))
    assert expected==pytest.approx(reference,abs=2e-12)


def test_signed_spline_preserves_negative_values_and_broadcasts():
    coeff=np.array([[-.2,0.,0.],[-.2,0.,0.]])
    actual=signed_linear_expectation(np.log([80.,100.,120.]),np.array([0.,.02,.1]),coeff,strike=100.,knots=(1.,))
    np.testing.assert_allclose(actual,-.2,rtol=0.,atol=1e-15)


def test_calendar_basis_reconstructs_all_increments():
    rng=np.random.default_rng(42);increments=rng.normal(size=(8,8,3))
    cumulative=np.concatenate((np.zeros((8,1,3)),increments.cumsum(axis=1)),axis=1)
    basis=correction_basis(cumulative,np.arange(9),4)
    np.testing.assert_allclose(basis.reshape(8,9,4,3).sum(axis=2),cumulative,rtol=0.,atol=2e-15)


def test_convex_lp_improves_known_upper_and_zero_is_exact_fallback():
    payoff=np.tile([0.,2.,1.],(16,1));base=np.zeros_like(payoff);stops=np.full(16,2)
    basis=np.tile(np.array([[[0.],[1.],[0.]]]),(16,1,1))
    weights,record=fit_correction_weights(payoff,base,stops,basis,bins=1)
    assert record['status']=='completed' and 0<=weights[0]<=1.
    lower,upper=correction_endpoints(payoff,base,stops,basis,weights)
    np.testing.assert_allclose(upper-lower,0.,atol=1e-12)
    oldlower,oldupper=correction_endpoints(payoff,base,stops,basis,np.zeros(1))
    np.testing.assert_array_equal(oldlower,np.ones(16));np.testing.assert_array_equal(oldupper,np.full(16,2.))


def test_right_state_reliability_requires_integration_not_afterward():
    # A negative control demonstrates the exact issue this implementation
    # avoids: multiplying a centered shock by its RIGHT state is not centered.
    shock=np.array([-1.,1.]);prob=np.array([.5,.5]);reliability=np.array([.1,1.])
    assert abs(prob@(reliability*shock))>.4
    actual=reliability*shock;centered=actual-prob@actual
    assert abs(prob@centered)<1e-15


@pytest.fixture(scope="module")
def local_case():
    from dataclasses import replace
    from rvlab.experiments._pricing_rough_foundation import _market
    from rvlab.pricing import RoughHestonParams,approximate_fractional_kernel
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy,value_fourier_stopping_policy
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    from rvlab.pricing.rough_policy_diagnostics import fit_controlled_policy_iteration
    from rvlab.pricing.rough_tensor_premium import tensor_premium_fitter
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    config=RoughAmericanConfig(training_paths=64,valuation_paths=32,steps=8,exercise_steps=4,
        training_seed=910,valuation_seed=913,batch_size=16,stock_knots=(.85,.95,1.,1.1),premium_cap=.7)
    base=train_fourier_stopping_policy(params,_market(),kernel,strike=100.,maturity=60/365,config=config,premium_cap=.7)
    proxy=build_rough_obstacle_proxy(base,stock_knots=(.8,.9,.95,1.,1.05,1.2))
    training=value_fourier_stopping_policy(proxy,config=replace(proxy.config,valuation_seed=911),trace=True)["trace"]
    selected=fit_controlled_policy_iteration(proxy,training,fit_continuation=tensor_premium_fitter(degree=2),
        stock_knots=proxy.config.stock_knots,premium_floor=0.,premium_cap=.7)
    trace=value_fourier_stopping_policy(proxy,trace=True)["trace"]
    return proxy,selected,trace


def test_actual_right_reliability_composite_center_matches_scalar_gaussian_quadrature(local_case):
    from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
    proxy,selected,trace=local_case
    correction,_=local_correction_martingales(proxy,selected,trace,batch_size=16)
    path,j=0,1;left,right=trace['times'][j:j+2];grid=trace['actual_time_grid']
    a,b=np.searchsorted(grid,[left,right]);factor=trace['factors'][path: path+1,j]
    law=exercise_block_law(proxy,factor,times=grid[a:b+1]);model=proxy.value_premiums[j+1]
    nodes=model.stock_nodes;stocks=proxy.strike*nodes
    def nodal(f):
        f=np.asarray(f);euro=model.european_values(f);old=model.continuation_values(f)
        new=np.clip(selected.continuations[j+1].value(np.broadcast_to(stocks,f.shape[:-1]+stocks.shape),f,proxy.strike),0.,.7)
        payoff=np.maximum(proxy.strike-stocks,0.)
        old=np.clip(np.maximum(old,payoff-euro),0.,.7);new=np.clip(np.maximum(new,payoff-euro),0.,.7)
        distance=np.max(abs((f-selected.continuations[j+1].center)/selected.continuations[j+1].scale),axis=-1)
        return (new-old)*np.exp(-(distance/3.)**8)[...,None]
    vals=nodal(law['new_factors'])[0];reference=0.
    for branch in range(vals.shape[0]):
        for order in range(law['orthogonal'].shape[2]):
            mean=np.log(trace['stocks'][path,j,0])+law['log_stock_shifts'][0,branch,order]
            clock=law['orthogonal'][0,branch,order];root=np.sqrt(clock)
            if clock==0:integral=np.interp(np.exp(mean)/proxy.strike,nodes,vals[branch])
            else:
                points=[-12.]+[(np.log(proxy.strike*k)-mean)/root for k in nodes[1:] if -12<(np.log(proxy.strike*k)-mean)/root<12]+[12.]
                integral=sum(quad(lambda z:np.interp(np.exp(mean+root*z)/proxy.strike,nodes,vals[branch])*np.exp(-.5*z*z)/np.sqrt(2*np.pi),lo,hi,epsabs=1e-12)[0]
                    for lo,hi in zip(points[:-1],points[1:]))
            reference+=law['probabilities'][0,branch]*integral/law['orthogonal'].shape[2]
    actual=np.interp(trace['stocks'][path,j+1,0]/proxy.strike,nodes,nodal(trace['factors'][path: path+1,j+1])[0])
    implied=actual-(correction[path,j+1,0]-correction[path,j,0])/proxy.market.discount_curve.discount(right)
    assert implied==pytest.approx(reference,abs=2e-11)


def test_identical_rule_zero_correction_and_trace_guards(local_case):
    from dataclasses import replace
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
    proxy,selected,trace=local_case
    original=ControlledExercisePolicy(proxy,proxy.continuation_times,proxy.continuations,
        proxy.exercise_at_inception,(proxy.config.training_seed,),0.,proxy.premium_cap,{})
    correction,_=local_correction_martingales(proxy,original,trace)
    np.testing.assert_allclose(correction,0.,atol=2e-12,rtol=0.)
    for changes in ({'kernel_fingerprint':'wrong'},{'initial_spots':np.array([101.])},{'strike':101.}):
        with pytest.raises(ValueError,match='model, market, kernel'):
            local_correction_martingales(proxy,selected,{**trace,**changes})
    for options in ({'widths':()},{'widths':(.1,.1)},{'batch_size':0},{'batch_size':True}):
        with pytest.raises(ValueError):local_correction_martingales(proxy,selected,trace,**options)
