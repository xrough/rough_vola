"""Refreshed controls preserve a frozen tensor rule and exact Gaussian means."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad

from rvlab.experiments._pricing_rough_stopping_refresh import (
    refresh_tensor_obstacle_control,fit_microdate_value_control)
from rvlab.experiments._pricing_rough_foundation import _market
from rvlab.pricing import RoughHestonParams, approximate_fractional_kernel
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy, value_fourier_stopping_policy
from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
from rvlab.pricing.rough_policy_diagnostics import fit_controlled_policy_iteration, evaluate_trace_policy
from rvlab.pricing.rough_tensor_premium import tensor_premium_fitter
from rvlab.pricing.rough_stopping_controls import expected_stopping_value, stopping_value


@pytest.fixture(scope="module")
def refreshed():
    model = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
    kernel = approximate_fractional_kernel(.15, 1., 2, min_time_ratio=.003)
    config = RoughAmericanConfig(training_paths=64, valuation_paths=96, steps=8,
        exercise_steps=4, batch_size=32, training_seed=510, valuation_seed=513,
        stock_knots=(.8, .95, 1., 1.1), premium_cap=.7)
    base = train_fourier_stopping_policy(model, _market(), kernel, strike=100.,
        maturity=60/365, config=config, premium_cap=.7)
    old = build_rough_obstacle_proxy(base, stock_knots=(.75,.85,.9,.95,1.,1.05,1.2))
    training = value_fourier_stopping_policy(old,
        config=replace(old.config, valuation_seed=511), trace=True)["trace"]
    decision = fit_controlled_policy_iteration(old, training,
        fit_continuation=tensor_premium_fitter(degree=2),
        stock_knots=old.config.stock_knots, premium_cap=.7)
    return old, decision, refresh_tensor_obstacle_control(old, decision)


def test_refreshed_rule_and_paths_are_exact(refreshed):
    old, selected, new = refreshed
    old_trace = value_fourier_stopping_policy(old, trace=True)["trace"]
    expected = evaluate_trace_policy(old_trace, policy=selected)
    actual = value_fourier_stopping_policy(new, trace=True)
    assert new.continuations is selected.continuations
    np.testing.assert_array_equal(expected.raw_samples, actual["raw_samples"])
    np.testing.assert_array_equal(expected.stop_indices, actual["trace"]["stop_indices"])
    for name in ("stocks", "factors", "discounted_payoffs", "european_martingale"):
        np.testing.assert_array_equal(old_trace[name], actual["trace"][name])
    assert new.diagnostics["selected_exercise_policy_preserved"]
    assert not new.diagnostics["american_pricing_ready"]


def test_tensor_nodal_envelope_matches_selected_rule(refreshed):
    _, selected, new = refreshed
    factors = np.array([[0.,0.], [.001,.0004], [-.0001,.0001]])
    for j in (1,2,3):
        premium = new.value_premiums[j]
        stock = np.broadcast_to(100*premium.stock_nodes, (3,len(premium.stock_nodes)))
        expected = np.clip(selected.continuations[j].value(stock, factors, 100.),0.,.7)
        np.testing.assert_allclose(premium.continuation_values(factors), expected, rtol=0.,atol=1e-11)
        obstacle = np.maximum(expected, np.maximum(100-stock,0.)-premium.european_values(factors))
        np.testing.assert_allclose(premium.nodal_premiums(factors),np.clip(obstacle,0.,.7),rtol=0.,atol=1e-11)


def test_tensor_obstacle_independent_gaussian_quadrature(refreshed):
    _, _, new = refreshed
    coefficients = new.value_premiums[2].coefficients(np.zeros(2))
    mean, clock = math.log(96.), .014
    analytic = expected_stopping_value(mean,clock,coefficients,strike=100.,is_call=False,
        exercise=False,stock_knots=new.config.stock_knots,value_cap=.7)
    points = [-12.]+[(math.log(100*k)-mean)/math.sqrt(clock) for k in new.config.stock_knots]+[12.]
    def integrand(z):
        return float(stopping_value(math.exp(mean+math.sqrt(clock)*z),coefficients,strike=100.,
            is_call=False,exercise=False,stock_knots=new.config.stock_knots,value_cap=.7))*math.exp(-z*z/2)/math.sqrt(2*math.pi)
    reference = sum(quad(integrand,a,b,epsabs=1e-12)[0] for a,b in zip(points[:-1],points[1:]))
    assert analytic == pytest.approx(reference,abs=2e-11)


def test_refreshed_control_propagates_training_firewall(refreshed):
    _, selected, new = refreshed
    assert set(selected.training_seeds).issubset(new.diagnostics["additional_control_training_seeds"])
    with pytest.raises(ValueError,match="additional control training"):
        value_fourier_stopping_policy(new,config=replace(new.config,valuation_seed=511))
    with pytest.raises(ValueError,match="zero premium floor"):
        refresh_tensor_obstacle_control(selected.base_policy,replace(selected,premium_floor=-.1))


def test_all_microdate_refit_keeps_law_and_every_exercise_decision(refreshed):
    old,selected,_=refreshed
    new=fit_microdate_value_control(old,selected,training_paths=96,
        training_seed=514,value_stock_knots=(.8,.95,1.,1.1))
    before=value_fourier_stopping_policy(old,trace=True)["trace"]
    expected=evaluate_trace_policy(before,policy=selected)
    after=value_fourier_stopping_policy(new,trace=True)
    assert len(new.continuation_times)==9
    np.testing.assert_array_equal(new.exercise_times,old.exercise_times)
    np.testing.assert_array_equal(expected.raw_samples,after["raw_samples"])
    np.testing.assert_array_equal(expected.stop_indices,after["trace"]["stop_indices"])
    for key in ("stocks","factors","discounted_payoffs","european_martingale"):
        np.testing.assert_array_equal(before[key],after["trace"][key])
    np.testing.assert_allclose(before["actual_time_grid"],after["trace"]["actual_time_grid"],rtol=0.,atol=0.)
    # Only the selected rule's predictable boundary differs from the old rule;
    # its exercise-date interpolation is identical to direct materialization.
    direct=value_fourier_stopping_policy(refreshed[2],trace=True)["trace"]
    np.testing.assert_allclose(direct["micro_boundary_martingale"],after["trace"]["micro_boundary_martingale"],rtol=0.,atol=1e-12)
    assert new.diagnostics["original_exercise_calendar_preserved"]
    assert 514 in new.diagnostics["additional_control_training_seeds"]
    with pytest.raises(ValueError,match="independent|additional control training"):
        value_fourier_stopping_policy(new,config=replace(new.config,valuation_seed=514))
    with pytest.raises(ValueError,match="overlaps"):
        fit_microdate_value_control(old,selected,training_paths=64,training_seed=511)


def test_mixture_hashes_coefficient_changes_and_control_dates_are_not_exercise_dates(refreshed):
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    old,selected,_=refreshed
    new=fit_microdate_value_control(old,selected,training_paths=64,training_seed=515)
    assert sum(model.exercise_obstacle for model in new.value_premiums)==len(new.exercise_times)
    mixture=new.continuations[1]
    assert mixture.left_beta.ndim==1 and mixture.right_beta.ndim==2
    changed=mixture.right_beta.copy();changed.flat[-1]+=1e-12
    replacement=replace(mixture,right_beta=changed)
    altered=replace(new,continuations=(new.continuations[0],replacement)+new.continuations[2:])
    assert fourier_policy_fingerprint(new)!=fourier_policy_fingerprint(altered)


@pytest.mark.parametrize("constant_variance",[True,False])
def test_refit_preserves_frozen_bumped_cashflows_in_constant_variance_and_heston_limits(constant_variance):
    from rvlab.pricing.rough_lift import ExponentialKernel
    model=RoughHestonParams(.04,.04 if constant_variance else .045,1.5,
        0. if constant_variance else .3,-.6,.5)
    kernel=ExponentialKernel((0.,),(1.,),.5,1.)
    config=RoughAmericanConfig(training_paths=48,valuation_paths=64,steps=8,
        exercise_steps=4,batch_size=32,training_seed=610,valuation_seed=613,
        stock_knots=(.8,.95,1.,1.1),premium_cap=.7)
    base=train_fourier_stopping_policy(model,_market(),kernel,strike=100.,
        maturity=60/365,config=config,premium_cap=.7)
    old=build_rough_obstacle_proxy(base,stock_knots=(.75,.85,.9,.95,1.,1.05,1.2))
    training=value_fourier_stopping_policy(old,config=replace(old.config,valuation_seed=611),trace=True)["trace"]
    selected=fit_controlled_policy_iteration(old,training,fit_continuation=tensor_premium_fitter(degree=2),
        stock_knots=old.config.stock_knots,premium_cap=.7)
    new=fit_microdate_value_control(old,selected,training_paths=48,training_seed=612)
    spots=np.array([99.5,99.75,100.,100.25,100.5])
    before=value_fourier_stopping_policy(old,spots=spots,trace=True)["trace"]
    expected=evaluate_trace_policy(before,policy=selected)
    after=value_fourier_stopping_policy(new,spots=spots,trace=True)
    np.testing.assert_array_equal(after["raw_samples"],expected.raw_samples)
    np.testing.assert_array_equal(after["trace"]["stop_indices"],expected.stop_indices)
    for key in ("factors","stocks","european_martingale"):
        np.testing.assert_array_equal(before[key],after["trace"][key])
    assert np.isfinite(after["primal_samples"]).all()
    assert np.all(after["dual_samples"]>=after["primal_samples"]-2e-12)
