"""Obstacle nodes retain original factor/Fourier values and exact centering."""
import math
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad

from rvlab.pricing import RoughHestonParams, approximate_fractional_kernel
from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy, value_fourier_stopping_policy
from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
from rvlab.pricing.rough_stopping_controls import stopping_value, expected_stopping_value
from rvlab.experiments._pricing_rough_foundation import _market


@pytest.fixture(scope="module")
def policies():
    params = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
    kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003)
    config = RoughAmericanConfig(training_paths=64, valuation_paths=64, steps=8,
        exercise_steps=4, batch_size=64)
    base = train_fourier_stopping_policy(params, _market(), kernel, strike=100.,
        maturity=60/365, config=config, premium_cap=.7)
    return base, build_rough_obstacle_proxy(base)


def test_obstacle_nodes_match_original_affine_and_continuation(policies):
    base, proxy = policies
    affine = _AffineSurrogate(base.params, base.market, base.kernel, np.array([100.]),
        np.array([base.maturity]), np.array([False]), base.continuation_times, base.control_config)
    factors = np.array([[0., 0., 0., 0.], [.0001, .0002, .0001, 0.], [.0002, 0., 0., .0001]])
    for index in [0, 1, 3, 4]:
        time = float(base.continuation_times[index])
        model = proxy.value_premiums[index]
        stocks = 100*model.stock_nodes
        logmean = np.broadcast_to(np.log(stocks[1:]), (len(factors), 1, len(stocks)-1))
        euro = affine.values(time, factors[:, None, :], logmean, np.zeros_like(logmean), np.array([0]))[:, 0, :, 0]
        euro /= base.market.discount_curve.discount(time)
        np.testing.assert_allclose(model.european_values(factors)[:, 1:], euro, atol=3e-13)
        q = base.continuations[index].value(np.broadcast_to(stocks, (len(factors), len(stocks))), factors, 100.)
        np.testing.assert_allclose(model.continuation_values(factors), np.clip(q, 0., .7), atol=3e-13)
        if index < 4:
            expected = np.clip(np.maximum(np.clip(q, 0., .7),
                np.maximum(100-stocks, 0)-model.european_values(factors)), 0., .7)
            np.testing.assert_allclose(model.nodal_premiums(factors), expected, atol=3e-13)
        else:
            assert np.all(model.nodal_premiums(factors) == 0)
    assert proxy.continuations is base.continuations
    assert proxy.exercise_at_inception == base.exercise_at_inception


def test_obstacle_gaussian_expectation_against_scalar_integration(policies):
    _, proxy = policies
    model = proxy.value_premiums[2]
    coefficient = model.coefficients(np.zeros(4))
    mean, variance = math.log(96), .014
    expected = expected_stopping_value(mean, variance, coefficient, strike=100.,
        is_call=False, exercise=False, stock_knots=proxy.config.stock_knots, value_cap=.7)
    knots = [(math.log(100*k)-mean)/math.sqrt(variance) for k in proxy.config.stock_knots]
    points = [-12.]+[v for v in knots if -12 < v < 12]+[12.]
    def integrand(z):
        spot = math.exp(mean+math.sqrt(variance)*z)
        value = stopping_value(spot, coefficient, strike=100., is_call=False,
            exercise=False, stock_knots=proxy.config.stock_knots, value_cap=.7)
        return float(value)*math.exp(-z*z/2)/math.sqrt(2*math.pi)
    reference = sum(quad(integrand, a, b, epsabs=1e-12)[0] for a,b in zip(points[:-1], points[1:]))
    assert expected == pytest.approx(reference, abs=2e-11)
    assert not model.affine_exponent.flags.writeable
    assert model.coefficients(np.zeros((3, 2, 4))).shape == (3, 2, len(proxy.config.stock_knots)+1, 3)


def test_obstacle_transform_preserves_independent_cashflows_and_stops(policies):
    base, proxy = policies
    first = value_fourier_stopping_policy(base, trace=True)
    second = value_fourier_stopping_policy(proxy, trace=True)
    np.testing.assert_array_equal(first["raw_samples"], second["raw_samples"])
    np.testing.assert_array_equal(first["stop_times"], second["stop_times"])
    np.testing.assert_array_equal(first["trace"]["stocks"], second["trace"]["stocks"])
    for name in ("primal_samples", "dual_samples"):
        assert np.isfinite(second[name]).all()
    assert not second["diagnostics"]["american_pricing_ready"]


@pytest.mark.parametrize("knots", [(), (1., .9), (0., 1.), (.9, float("nan"))])
def test_obstacle_rejects_invalid_nodes(policies, knots):
    with pytest.raises(ValueError):
        build_rough_obstacle_proxy(policies[0], stock_knots=knots)


def test_public_obstacle_proxy_opt_in_preserves_policy_and_reports_settings(policies):
    from rvlab.pricing import evaluate_vanilla, RoughDualConfig
    from rvlab.pricing.rough_stopping import train_rough_stopping_policy
    original,_=policies
    knobs=dict(training_method='fourier_premium',training_paths=64,valuation_paths=64,
        steps=8,exercise_steps=4,batch_size=32)
    base=RoughAmericanConfig(**knobs)
    dual=RoughDualConfig(training_paths=32,training_seed=44,candidate_bins=(1,),
        european_bounds=(1.,1.),premium_bounds=(1.,1.))
    enhanced=replace(base,dual_proxy='obstacle',dual_proxy_knots=(.8,.9,.95,.975,1.,1.05,1.2),
        dual_config=dual)
    options=dict(market=original.market,kernel=original.kernel,strike=100.,
        maturity=original.maturity,is_call=False,exercise_style='american')
    baseline=evaluate_vanilla(original.params,config=base,**options)
    result=evaluate_vanilla(original.params,config=enhanced,**options)
    np.testing.assert_array_equal(result.raw_prices,baseline.raw_prices)
    np.testing.assert_array_equal(result.exercise_probabilities,baseline.exercise_probabilities)
    assert result.config.dual_proxy=='obstacle'
    assert result.policies[0].config.stock_knots==enhanced.dual_proxy_knots
    assert result.diagnostics['dual_training_evaluations']==1
    assert result.policies[0].diagnostics['exercise_policy_unchanged']
    assert not result.diagnostics['american_pricing_ready']
    # The training basis and control interpolation knots have separate roles.
    assert result.policies[0].continuations[1].knots==base.stock_knots
    for changes in ({'dual_proxy':'unknown'}, {'dual_proxy':'obstacle'},
            {'dual_proxy_knots':(.9,1.)}):
        with pytest.raises(ValueError):RoughAmericanConfig(**changes)
    with pytest.raises(ValueError):replace(enhanced,dual_proxy_knots=(1.,.9))
    expired=evaluate_vanilla(original.params,config=enhanced,**{**options,'maturity':0.})
    assert expired.prices==0. and expired.diagnostics['dual_training_evaluations']==0


@pytest.mark.parametrize('case,is_call,rate',[
    ('rough',True,.04),('rough',True,-.03),('rough',False,-.03),
    ('curve',False,.04),('heston',True,-.03),('deterministic_curve',False,-.03)])
def test_obstacle_node_european_parity_and_original_policy_invariance_across_models(case,is_call,rate):
    from rvlab.pricing import (RoughHestonCurveParams,MeanReversionCurve,
        DiscountCurve,DividendSchedule,SpotMarket)
    from rvlab.pricing.rough_lift import ExponentialKernel
    curve=MeanReversionCurve((.04,.11),(.035,.07,.025))
    if case in ('curve','deterministic_curve'):
        model=RoughHestonCurveParams(.04,1.5,0. if case=='deterministic_curve' else .3,
            -.6,curve,.15)
    else:model=RoughHestonParams(.04,.045,1.5,.3,-.6,.5 if case=='heston' else .15)
    kernel=(ExponentialKernel((0.,),(1.,),.5,1.) if case=='heston'
        else approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003))
    market=SpotMarket(100.,DiscountCurve.flat(rate,1.),DividendSchedule((),1.))
    config=RoughAmericanConfig(training_paths=48,valuation_paths=64,steps=8,
        exercise_steps=4,batch_size=32,stock_knots=(.85,.95,1.,1.05,1.15))
    base=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.18,
        is_call=is_call,config=config,premium_cap=.7)
    proxy=build_rough_obstacle_proxy(base,stock_knots=(.75,.88,.96,1.02,1.1,1.4))
    affine=_AffineSurrogate(model,market,kernel,np.array([100.]),np.array([.18]),
        np.array([is_call]),base.continuation_times,base.control_config)
    factors=np.array([np.zeros(len(kernel.nodes)),np.full(len(kernel.nodes),.0001)])
    for j,time in enumerate(base.continuation_times):
        nodal=proxy.value_premiums[j];stocks=100*nodal.stock_nodes
        mean=np.broadcast_to(np.log(stocks[1:]),(2,1,len(stocks)-1))
        expected=affine.values(float(time),factors[:,None,:],mean,np.zeros_like(mean),
            np.array([0]))[:,0,:,0]/market.discount_curve.discount(time)
        value=nodal.european_values(factors)
        np.testing.assert_allclose(value[:,1:],expected,rtol=0.,atol=8e-13)
        opposite=replace(nodal,is_call=not is_call).european_values(factors)
        relative_discount=market.discount_curve.discount(.18)/market.discount_curve.discount(time)
        parity=stocks-100*relative_discount
        np.testing.assert_allclose(value-opposite,(1 if is_call else -1)*np.broadcast_to(parity,value.shape),
            rtol=0.,atol=8e-13)
        assert value[0,0]==pytest.approx(0. if is_call else 100*relative_discount,abs=2e-13)
    # The proxy uses a different interpolation grid from the exercise rule.
    # Exercise decisions, factor/stock paths and raw payments must remain the
    # original policy's, including at both sides of the paired delta bumps.
    spots=np.array([99.5,99.75,100.,100.25,100.5])
    original=value_fourier_stopping_policy(base,spots=spots,trace=True)
    transformed=value_fourier_stopping_policy(proxy,spots=spots,trace=True)
    assert proxy.continuations is base.continuations
    for name in ('raw_samples','stop_times'):
        np.testing.assert_array_equal(transformed[name],original[name])
    for name in ('stocks','factors','stop_indices','micro_boundary_martingale'):
        np.testing.assert_array_equal(transformed['trace'][name],original['trace'][name])
    assert np.isfinite(transformed['dual_samples']).all()
    assert np.all(transformed['dual_samples']>=transformed['primal_samples']-2e-12)


def test_public_obstacle_four_component_dual_retrained_bumps_use_their_own_frozen_policies(policies):
    from rvlab.pricing import evaluate_vanilla,RoughDualConfig
    original,_=policies
    dual=RoughDualConfig(training_paths=32,training_seed=44,candidate_bins=(1,),
        components=('european','premium','micro_boundary','micro_poststop'),
        european_bounds=(1.,1.),premium_bounds=(1.,1.),intrinsic_bounds=(0.,.5),
        poststop_bounds=(0.,1.5))
    config=RoughAmericanConfig(training_method='fourier_premium',training_paths=48,
        valuation_paths=64,steps=8,exercise_steps=4,batch_size=32,retrain_bumps=True,
        stock_knots=(.85,.95,1.,1.05,1.15),dual_proxy='obstacle',
        dual_proxy_knots=(.75,.88,.96,1.02,1.1,1.4),dual_config=dual)
    result=evaluate_vanilla(original.params,market=original.market,kernel=original.kernel,
        strike=100.,maturity=original.maturity,is_call=False,exercise_style='american',config=config)
    assert result.diagnostics['dual_training_evaluations']==3
    record=result.diagnostics['contracts'][0]
    for item in [record['dual_training']]+[v['dual_training'] for v in record['retrained_bump_dual_training']]:
        assert item['components']==list(dual.components)
        assert item['independent_policy_dual_valuation_streams']
        assert all(c['status'] in ('baseline','completed') for c in item['candidates'])
    assert len(record['retrained_bump_dual_training'])==2
    for name in ('prices','dual_prices','deltas','delta_standard_errors','bump_deltas',
            'bump_delta_standard_errors','retrained_deltas','retrained_delta_standard_errors'):
        assert np.isfinite(getattr(result,name)).all()
    assert not result.diagnostics['american_pricing_ready']
