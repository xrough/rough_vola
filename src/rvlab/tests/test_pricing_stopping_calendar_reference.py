"""Independent calendar benchmarks, distinct from stopping regression/controls."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import quad
from rvlab.pricing import (BlackScholesParams, CashDividend, DiscountCurve,
    DividendSchedule, SpotMarket, price)
from rvlab.pricing.rough_stopping_reference import (
    _gaussian_stock_linear_step, black_scholes_stopping_reference, black_scholes_stopping_surface)
from rvlab.pricing.vanilla_types import UnsupportedModelError


def market(rate=.04):
    return SpotMarket(100., DiscountCurve.flat(rate, 1.), DividendSchedule((), 1.))


@pytest.mark.parametrize("call", [False, True])
def test_one_gaussian_transition_integrates_vanilla_payoff_exactly(call):
    stocks=100*np.exp(np.linspace(-2.,2.,513)); dt=.2; rate=-.02; vol=.3
    values=np.maximum(stocks-100 if call else 100-stocks,0.)
    tails=dict(left_intercept=0. if call else 100.,left_slope=0. if call else -1.,
        right_intercept=-100. if call else 0.,right_slope=1. if call else 0.)
    actual=np.exp(-rate*dt)*_gaussian_stock_linear_step(values,stocks,
        (rate-.5*vol**2)*dt,vol**2*dt,**tails)
    exact=price(BlackScholesParams(vol),forward=stocks*np.exp(rate*dt),
        discount=np.exp(-rate*dt),strike=100.,maturity=dt,is_call=call)
    np.testing.assert_allclose(actual,exact,atol=4e-10,rtol=0)


def test_fft_moment_convolution_matches_independent_scalar_density_integration():
    stocks=100*np.exp(np.linspace(-1.,1.,65)); values=.2+np.sin(np.log(stocks))**2
    mean,variance=.003,.025
    actual=_gaussian_stock_linear_step(values,stocks,mean,variance,
        left_intercept=values[0],left_slope=0.,right_intercept=values[-1],right_slope=0.)
    for i in (1,32,63):
        def integrand(z):
            s=stocks[i]*np.exp(mean+np.sqrt(variance)*z)
            return np.interp(s,stocks,values)*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        points=(np.log(stocks/stocks[i])-mean)/np.sqrt(variance)
        exact=quad(integrand,-10.,10.,points=points[(points>-10)&(points<10)],
            epsabs=1e-10,limit=200)[0]
        assert actual[i]==pytest.approx(exact,abs=2e-10)


@pytest.mark.parametrize("rate", [-.02,.04])
@pytest.mark.parametrize("call", [False,True])
def test_multistep_european_reference_and_calendar_american_bounds(rate,call):
    m=market(rate); times=np.linspace(0.,.2,17)
    euro=black_scholes_stopping_reference(m,volatility=.2,strike=100.,
        exercise_times=times,is_call=call,exercise_style="european",spot_nodes=4097)
    american=black_scholes_stopping_reference(m,volatility=.2,strike=100.,
        exercise_times=times,is_call=call,spot_nodes=4097)
    exact=price(BlackScholesParams(.2),forward=100*np.exp(rate*.2),discount=np.exp(-rate*.2),
        strike=100.,maturity=.2,is_call=call)
    assert abs(euro.price-exact)<.001+1e-4*abs(exact)
    assert american.price>=euro.price-2e-10
    assert american.diagnostics["price_stable"] and american.diagnostics["delta_stable"]
    if call and rate>=0:
        assert american.price==pytest.approx(euro.price,abs=2e-10)


def _calendar_tree(spot,strike,vol,rate,maturity,exercise_intervals,substeps):
    n=exercise_intervals*substeps;dt=maturity/n;h=vol*np.sqrt(dt)
    u,d=np.exp(h),np.exp(-h);p=(np.exp(rate*dt)-d)/(u-d)
    values=np.maximum(strike-spot*np.exp((2*np.arange(n+1)-n)*h),0.)
    for j in range(n-1,-1,-1):
        values=np.exp(-rate*dt)*(p*values[1:]+(1-p)*values[:-1])
        if j%substeps==0:
            values=np.maximum(values,strike-spot*np.exp((2*np.arange(j+1)-j)*h))
    return float(values[0])


def test_american_reference_agrees_with_independent_calendar_aligned_tree():
    t=60/365; intervals=32
    ref=black_scholes_stopping_reference(market(),volatility=.2,strike=100.,
        exercise_times=np.linspace(0.,t,intervals+1),spot_nodes=8193)
    coarse=_calendar_tree(100.,100.,.2,.04,t,intervals,128)
    fine=_calendar_tree(100.,100.,.2,.04,t,intervals,256)
    assert abs(fine-coarse)<.001+1e-4*abs(fine)
    assert abs(ref.price-fine)<.001+1e-4*abs(fine)


@pytest.mark.parametrize("call", [False,True])
def test_deterministic_variance_expiration_and_negative_rate_dates(call):
    m=market(-.1); times=np.linspace(0.,.2,17)
    r=black_scholes_stopping_reference(m,volatility=0.,strike=100.,
        exercise_times=times,is_call=call,spot_nodes=4097)
    d=m.discount_curve.discount(times)
    exact=np.max(np.maximum(100-100*d if call else 100*d-100,0.))
    assert r.price==pytest.approx(exact,abs=2e-8)
    at_expiry=black_scholes_stopping_reference(m,volatility=.2,strike=100.,
        exercise_times=[0.],is_call=call)
    assert at_expiry.price==0. and at_expiry.delta==(.5 if call else -.5)


@pytest.mark.parametrize("changes",[
    {"volatility":-.1},{"volatility":True},{"strike":0.},{"spot_nodes":64},
    {"spot_nodes":True},{"log_half_width":0.},{"exercise_times":[.1,.2]},
    {"exercise_times":[0.,.2,.1]},{"exercise_times":[0.,np.nan]},
    {"is_call":1},{"check_refinement":1},{"exercise_style":"bermudan"},
])
def test_reference_rejects_malformed_inputs(changes):
    kwargs=dict(volatility=.2,strike=100.,exercise_times=[0.,.2]);kwargs.update(changes)
    with pytest.raises((ValueError,TypeError)):
        black_scholes_stopping_reference(market(),**kwargs)


def test_reference_preserves_cash_exclusion_and_explicit_coverage():
    m=replace(market(),dividends=DividendSchedule((CashDividend(.1,1.),),1.))
    with pytest.raises(UnsupportedModelError,match="cash"):
        black_scholes_stopping_reference(m,volatility=.2,strike=100.,exercise_times=[0.,.2])
    m=replace(m,dividends=DividendSchedule((),.1))
    with pytest.raises(ValueError,match="cover"):
        black_scholes_stopping_reference(m,volatility=.2,strike=100.,exercise_times=[0.,.2])


def test_calendar_surface_retains_date_values_and_is_immutable():
    times=np.linspace(0.,.2,17)
    kwargs=dict(volatility=.2,strike=100.,exercise_times=times,spot_nodes=2049)
    surface=black_scholes_stopping_surface(market(),**kwargs)
    reference=black_scholes_stopping_reference(market(),check_refinement=False,**kwargs)
    assert surface['values'].shape==(17,2049)
    assert surface['values'][0,1024]==reference.price
    payoff=np.maximum(100.-surface['stocks'],0.)
    np.testing.assert_allclose(surface['values'][-1],payoff)
    assert np.min(surface['values']-payoff[None,:])>=-2e-10
    with pytest.raises(ValueError):surface['values'][0,0]=0.
