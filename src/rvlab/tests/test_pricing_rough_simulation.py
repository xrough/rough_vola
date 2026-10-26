"""Layered controls for inception-only weak lifted rough-Heston dynamics.

Statistical consistency checks are not accuracy certificates: a confidence
interval that covers a reference can still be far wider than pricing tolerance.
The research notebook reports the strict error-plus-uncertainty gate separately.
"""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DiscountCurve, DividendSchedule,
    MeanReversionCurve, PricingConfig, RoughHestonCurveParams, RoughHestonParams,
    SpotMarket, UnsupportedModelError, black_scholes_price, heston_price,
)
from rvlab.pricing.rough_lift import (
    ExponentialKernel, lifted_forward_variance, lifted_heston_price,
    lifted_integrated_variance,
)
from rvlab.pricing.rough_simulation import (
    RoughSimulationConfig, RoughSimulationError, _check_state, _cone_translation,
    _moment_support, european_payoff_samples, simulate_rough_heston,
)


def _params(**changes):
    return replace(RoughHestonParams(.04,.04,1.5,.3,-.6,.15),**changes)


def _market(events=(), rate=.04, horizon=1.):
    return SpotMarket(100.,DiscountCurve.flat(rate,horizon),DividendSchedule(events,horizon))


def _kernel():
    return ExponentialKernel((.1,4.),(.5,1.5),hurst=.15,horizon=1.)


@pytest.mark.parametrize("field,value",[
    ("paths",0),("steps",True),("batch_size",1.2),("seed",-1),("seed",True),
    ("memory_budget_mb",0),("memory_budget_mb",float("inf")),
])
def test_invalid_config(field,value):
    with pytest.raises(ValueError,match=field):
        RoughSimulationConfig(**{field:value})


def test_three_point_rule_exact_first_four_and_audited_fifth_moment_discrepancy():
    x=np.array([0.,1e-16,1e-8,.04,1.,1e5])
    for z in (1e-12,.001,1.,1e6):
        support,p=_moment_support(x,z)
        assert np.min(support)>=0 and np.min(p)>=0
        np.testing.assert_allclose(p.sum(-1),1.,rtol=0,atol=3e-16)
        moments=[x,x*x+x*z,x**3+3*x*x*z+1.5*x*z*z,
                 x**4+6*x**3*z+9*x*x*z*z+3*x*z**3]
        for order,expected in enumerate(moments,1):
            np.testing.assert_allclose((support**order*p).sum(-1),expected,rtol=2e-13,atol=1e-90)
        a=(6+np.sqrt(3))/4
        exact_fifth=x**5+10*x**4*z+30*x**3*z*z+30*x*x*z**3+7.5*x*z**4
        discrepancy=(6*a-10.5)*x*x*z**3+(11.25*a-3*a*a-9.75)*x*z**4
        np.testing.assert_allclose((support**5*p).sum(-1),exact_fifth+discrepancy,rtol=2e-13,atol=1e-90)


def test_expiration_and_saved_array_shapes_are_readonly():
    result=simulate_rough_heston(_params(),_market(),_kernel(),observation_times=[0.],
        config=RoughSimulationConfig(paths=5,steps=1))
    assert result.spots.shape==(5,1) and result.factors.shape==(5,1,2)
    np.testing.assert_array_equal(result.spots,100.)
    np.testing.assert_array_equal(result.variances,.04)
    np.testing.assert_array_equal(result.factors,0.)
    np.testing.assert_array_equal(european_payoff_samples(result,strike=[90,110],maturity=0.,is_call=[True,False]),10.)
    assert result.diagnostics["actual_time_steps"]==0
    with pytest.raises(ValueError):
        result.factors[0,0,0]=1.


def test_random_stream_and_results_reproduce_across_batches():
    config=RoughSimulationConfig(paths=129,steps=19,batch_size=7)
    a=simulate_rough_heston(_params(),_market(),_kernel(),observation_times=[.03,.1],config=config)
    b=simulate_rough_heston(_params(),_market(),_kernel(),observation_times=[.03,.1],config=replace(config,batch_size=64))
    np.testing.assert_allclose(a.spots,b.spots,rtol=2e-14,atol=1e-12)
    np.testing.assert_allclose(a.factors,b.factors,rtol=2e-13,atol=2e-16)
    assert a.diagnostics["batch_size_invariant_random_stream"]
    assert a.diagnostics["minimum_saved_factor"]<0
    assert not a.diagnostics["individual_factor_clipping"]


def test_cash_events_rates_and_curve_knots_align_exactly():
    params=RoughHestonCurveParams(.04,1.5,.3,-.6,MeanReversionCurve((.023,),(.03,.06)),.15)
    market=SpotMarket(100.,DiscountCurve((0.,.017,.1),(1.,1.001,.998)),
                      DividendSchedule((CashDividend(.041,1.5),),.1))
    r=simulate_rough_heston(params,market,_kernel(),observation_times=[.041,.1],
        config=RoughSimulationConfig(paths=64,steps=11))
    assert all(t in r.diagnostics["time_grid"] for t in (.017,.023,.041,.1))
    assert r.conditional_log_spots is None
    assert r.diagnostics["dividend_events"][0]["time"]==.041
    with pytest.raises(ValueError,match="positive cash dividends"):
        european_payoff_samples(r,strike=100,maturity=.1)


def test_zero_noise_constant_variance_conditional_prices_are_analytic_bs():
    params=_params(vol_of_vol=0.)
    r=simulate_rough_heston(params,_market(rate=-.02),_kernel(),observation_times=[.2],
        config=RoughSimulationConfig(paths=13,steps=3))
    expected=black_scholes_price(BlackScholesParams(.2),forward=100*np.exp(-.02*.2),
        discount=np.exp(.02*.2),strike=[90,100,110],maturity=.2,is_call=[True,False,True])
    samples=european_payoff_samples(r,strike=[90,100,110],maturity=.2,is_call=[True,False,True])
    np.testing.assert_allclose(samples,np.broadcast_to(expected,samples.shape),rtol=2e-14,atol=2e-13)
    np.testing.assert_allclose(r.variances,.04,rtol=0,atol=2e-16)


@pytest.mark.parametrize("eta",[1e-16,1e-12,np.nextafter(1e-8,0.)])
def test_underresolved_positive_vol_of_vol_raises_instead_of_biased_stock(eta):
    with pytest.raises(RoughSimulationError,match="factor-increment cancellation"):
        simulate_rough_heston(_params(vol_of_vol=eta),_market(),_kernel(),observation_times=[.1],
            config=RoughSimulationConfig(paths=8,steps=4))
    # Expiration performs no divided stock increment and remains well-defined.
    expired=simulate_rough_heston(_params(vol_of_vol=eta),_market(),_kernel(),observation_times=[0.],
        config=RoughSimulationConfig(paths=2,steps=1))
    np.testing.assert_array_equal(expired.spots,100.)


def test_smallest_supported_positive_vol_of_vol_runs_and_zero_is_not_substituted():
    result=simulate_rough_heston(_params(vol_of_vol=1e-8),_market(),_kernel(),observation_times=[.1],
        config=RoughSimulationConfig(paths=8,steps=4))
    assert np.isfinite(result.spots).all()
    assert result.diagnostics["minimum_supported_positive_vol_of_vol"]==1e-8
    assert np.any(result.factors!=0.)


def test_deterministic_curve_uses_exact_lifted_variance_path_and_integral():
    params=RoughHestonCurveParams(.04,1.2,0.,-.6,MeanReversionCurve((.031,),(.08,.025)),.15)
    times=np.array([.03,.031,.12])
    r=simulate_rough_heston(params,_market(),_kernel(),observation_times=times,
        config=RoughSimulationConfig(paths=9,steps=7))
    np.testing.assert_allclose(r.variances[0],lifted_forward_variance(params,times,_kernel()),rtol=2e-13,atol=2e-16)
    np.testing.assert_allclose(r.conditional_variances[0],lifted_integrated_variance(params,times,_kernel()),rtol=2e-13,atol=2e-16)
    assert r.variances[0,0]!=r.variances[0,-1]


def test_limited_liability_cash_jump_preserves_gain_pathwise_and_variance():
    params=_params(v0=0.,theta=0.,vol_of_vol=0.)
    market=_market((CashDividend(.05,1000.),),rate=-.02)
    r=simulate_rough_heston(params,market,_kernel(),observation_times=[.05,.1],
        config=RoughSimulationConfig(paths=8,steps=4))
    np.testing.assert_array_equal(r.spots,0.)
    np.testing.assert_allclose(r.discounted_gains,100.,rtol=2e-15,atol=2e-13)
    np.testing.assert_array_equal(r.factors,0.)
    assert r.diagnostics["dividend_events"][0]["zero_stock_count"]==8


def test_missing_coverage_memory_and_conditional_reset_are_rejected():
    args=(_params(),_market(),_kernel())
    with pytest.raises(UnsupportedModelError,match="conditional"):
        simulate_rough_heston(*args,observation_times=[.1],conditional_state={})
    with pytest.raises(MemoryError,match="exceeding"):
        simulate_rough_heston(*args,observation_times=[.1],config=RoughSimulationConfig(memory_budget_mb=.00001))
    short=SpotMarket(100,DiscountCurve.flat(.04,1),DividendSchedule((),.01))
    with pytest.raises(ValueError,match="dividend expectations"):
        simulate_rough_heston(_params(),short,_kernel(),observation_times=[.1])
    with pytest.raises(ValueError,match="Hurst"):
        simulate_rough_heston(_params(hurst=.2),_market(),_kernel(),observation_times=[.1])
    with pytest.raises(ValueError,match="kernel approximation horizon"):
        simulate_rough_heston(_params(),_market(horizon=2),_kernel(),observation_times=[1.1])


@pytest.mark.parametrize("times",[[],[-.1],[.2,.1],[.1,.1],[float("nan")],[[.1]],
                                [True],[".1"],[.1+0j],[.1+.2j],np.array([1,0],dtype=np.uint64)])
def test_invalid_observation_times(times):
    with pytest.raises(ValueError,match="observation_times"):
        simulate_rough_heston(_params(),_market(),_kernel(),observation_times=times)


def test_material_cone_failure_is_identifiable_not_clipped():
    nodes,weights=np.array([.1,4.]),np.array([.5,1.5])
    shift,cone=_cone_translation(.04,nodes,weights)
    diagnostics={"minimum_cone_coordinate":0.,"minimum_raw_variance":0.,"aggregate_roundoff_count":0}
    with pytest.raises(RoughSimulationError,match="admissible-cone violation"):
        _check_state(np.array([[-1.,-1.]]),.04,weights,shift,cone,diagnostics,stage="test",time=.1)


def test_forced_one_factor_heston_european_statistics_without_delegation(monkeypatch):
    import rvlab.pricing.heston as heston_module
    params=_params(hurst=.5)
    horizon=.25
    expected=heston_price(params,forward=100*np.exp(.04*horizon),discount=np.exp(-.04*horizon),
        strike=[95,100,105],maturity=horizon,config=PricingConfig(256,160,512))
    monkeypatch.setattr(heston_module,"heston_price",lambda *a,**k:pytest.fail("simulator delegated"))
    result=simulate_rough_heston(params,_market(),ExponentialKernel((0.,),(1.,),hurst=.5,horizon=1.),
        observation_times=[horizon],config=RoughSimulationConfig(paths=32768,steps=128,seed=42))
    samples=european_payoff_samples(result,strike=[95,100,105],maturity=horizon)
    se=samples.std(axis=0,ddof=1)/np.sqrt(len(samples))
    # Four-SE statistical control, explicitly not the strict pricing gate.
    assert np.all(np.abs(samples.mean(axis=0)-expected)<4*se)
    assert np.any(1.96*se > .001+1e-4*np.abs(expected))
    assert result.diagnostics["backend"]=="finite_factor_weak_simulation"


def test_finite_lift_european_and_variance_moments_against_independent_references():
    params=_params()
    kernel=_kernel()
    t=.12
    result=simulate_rough_heston(params,_market(),kernel,observation_times=[t],
        config=RoughSimulationConfig(paths=32768,steps=128,seed=43))
    expected=lifted_heston_price(params,forward=100*np.exp(.04*t),discount=np.exp(-.04*t),
        strike=[95,100,105],maturity=t,kernel=kernel,config=PricingConfig(256,160,1024))
    samples=european_payoff_samples(result,strike=[95,100,105],maturity=t)
    errors=np.abs(samples.mean(0)-expected)
    assert np.all(errors<4*samples.std(0,ddof=1)/np.sqrt(len(samples)))
    # This constant-theta case has exactly constant mean and its factor
    # covariance solves a linear Lyapunov ODE, independent of the simulator.
    nodes,weights=np.asarray(kernel.nodes),np.asarray(kernel.weights)
    a=-np.diag(nodes)-params.kappa*np.ones((2,1))*weights
    covariance=solve_ivp(lambda time,c:(a@c.reshape(2,2)+c.reshape(2,2)@a.T
        +params.vol_of_vol**2*params.v0*np.ones((2,2))).ravel(),(0,t),np.zeros(4),rtol=1e-11,atol=1e-14).y[:,-1].reshape(2,2)
    u=result.factors[:,0]
    assert np.all(np.abs(u.mean(0))<4*np.sqrt(np.diag(covariance)/len(u)))
    products=u[:,:,None]*u[:,None,:]
    assert np.all(np.abs(products.mean(0)-covariance)<5*products.std(0,ddof=1)/np.sqrt(len(u)))
    assert result.diagnostics["minimum_raw_variance"]>=-1e-13


def test_cash_dividend_discounted_gain_martingale_statistical_control():
    result=simulate_rough_heston(_params(),_market((CashDividend(.05,1.5),),rate=-.02),
        _kernel(),observation_times=[.05,.12],config=RoughSimulationConfig(paths=32768,steps=128,seed=44))
    gains=result.discounted_gains
    assert np.all(np.abs(gains.mean(0)-100)<4*gains.std(0,ddof=1)/np.sqrt(len(gains)))
    raw=european_payoff_samples(result,strike=[95,100],maturity=.12,conditional=False)
    assert raw.shape==(32768,2) and np.min(raw)>=0


def test_payoff_validation_and_conditional_vs_raw_bs_noise():
    result=simulate_rough_heston(_params(vol_of_vol=0),_market(),_kernel(),observation_times=[.1],
        config=RoughSimulationConfig(paths=2048,steps=8))
    for kwargs in ({"strike":0,"maturity":.1},{"strike":100,"maturity":.09},
                   {"strike":100,"maturity":.1,"is_call":1}):
        with pytest.raises(ValueError):
            european_payoff_samples(result,**kwargs)
    conditional=european_payoff_samples(result,strike=100,maturity=.1)
    raw=european_payoff_samples(result,strike=100,maturity=.1,conditional=False)
    assert conditional.std()<1e-12 and raw.std()>1


@pytest.mark.parametrize("strike",[True,"100",100+0j,[True,False],["99","101"],[99+1j,101+0j]])
def test_payoff_strike_rejects_nonreal_coercions(strike):
    result=simulate_rough_heston(_params(vol_of_vol=0),_market(),_kernel(),observation_times=[.1],
        config=RoughSimulationConfig(paths=2,steps=1))
    with pytest.raises(ValueError,match="strikes"):
        european_payoff_samples(result,strike=strike,maturity=.1)
