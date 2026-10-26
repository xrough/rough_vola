"""Independent stopping policies and exact-discrete martingale bounds."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DividendSchedule, DiscountCurve, RoughHestonParams, SpotMarket,
)
from rvlab.pricing.black_scholes import black_scholes_price
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_simulation import (
    RoughSimulationConfig, _drift_flow_with_integral, _weak_transition_branches,
    simulate_rough_heston,
)
from rvlab.pricing.rough_stopping import (
    RoughAmericanConfig, _fit_continuation, estimate_rough_american,
    train_rough_stopping_policy, value_rough_stopping_policy,
)
from rvlab.pricing.vanilla_types import UnsupportedModelError


@pytest.fixture
def inputs():
    return (RoughHestonParams(.04, .045, 1.5, .3, -.6, .5),
        SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.)),
        ExponentialKernel((0.,), (1.,), .5, 1.))


@pytest.fixture(params=["least_squares", "conditional"])
def tiny(request):
    return RoughAmericanConfig(training_paths=128, valuation_paths=64, steps=8,
        exercise_steps=4, batch_size=32, training_method=request.param)


@pytest.mark.parametrize("kwargs", [
    {"training_paths": 1}, {"valuation_paths": True}, {"steps": 0},
    {"exercise_steps": 2.5}, {"batch_size": 0}, {"training_seed": -1},
    {"valuation_seed": 42}, {"ridge": -1.}, {"ridge": np.inf},
    {"memory_budget_mb": 0.}, {"bump_relative": .25}, {"bump_relative": False},
    {"stock_knots": (1., .9)}, {"stock_knots": (0., 1.)},
    {"stock_knots": (1., 1.)}, {"retrain_bumps": 1}, {"stock_integral": "unknown"},
    {"training_method": "in_sample"},
    {"bounded_surrogate": 1},
    {"surrogate_interpolation": "terminal_zero"},
])
def test_invalid_config_rejects_without_allocating(kwargs):
    with pytest.raises((ValueError, TypeError)):
        RoughAmericanConfig(**kwargs)


def test_refinements_keep_independent_axes_and_tolerances(tiny):
    assert tiny.refined("time").steps == 2*tiny.steps
    assert tiny.refined("time").exercise_steps == tiny.exercise_steps
    assert tiny.refined("exercise").exercise_steps == 2*tiny.exercise_steps
    assert tiny.refined("exercise").steps == tiny.steps
    assert tiny.refined("bump").bump_relative == tiny.bump_relative/2
    assert tiny.refined("training").training_paths == 2*tiny.training_paths
    assert tiny.refined("sampling").valuation_paths == 2*tiny.valuation_paths
    with pytest.raises(ValueError):
        tiny.refined("precision")


def test_value_bridge_keeps_terminal_payoff_anchor_and_policy_cashflows(inputs):
    from rvlab.pricing.rough_stopping import _surrogate_terms
    params, market, kernel = inputs
    config = RoughAmericanConfig(training_paths=128, valuation_paths=64,
        steps=8, exercise_steps=2, batch_size=32)
    policy = train_rough_stopping_policy(params, market, kernel,
        strike=100., maturity=.25, is_call=False, config=config)
    state = np.zeros((3, 1))
    bridge = replace(config, surrogate_interpolation="value_bridge")
    terms = _surrogate_terms(policy, .1875, state, bridge, False)
    assert [term[0] for term in terms] == [.5, .5]
    assert [term[2] for term in terms] == [False, True]
    # The right term is the true terminal payoff, not the zero continuation.
    from rvlab.pricing.rough_stopping_controls import stopping_value
    terminal = stopping_value(np.array([90., 100., 110.]), terms[1][1],
        strike=100., is_call=False, exercise=terms[1][2],
        stock_knots=config.stock_knots)
    np.testing.assert_allclose(terminal, [10., 0., 0.])
    old = value_rough_stopping_policy(policy)
    new = value_rough_stopping_policy(policy, config=bridge)
    # Changing an arbitrary martingale function leaves the frozen policy and
    # actual simulated cashflows untouched; each endpoint remains pathwise valid.
    np.testing.assert_array_equal(old["raw_samples"], new["raw_samples"])
    np.testing.assert_array_equal(old["stop_times"], new["stop_times"])
    assert np.min(new["dual_samples"]-new["primal_samples"]) >= -2e-12


def test_put_surrogate_cap_includes_future_negative_rate_extrema():
    from rvlab.pricing.rough_stopping import _surrogate_cap, _decision_continuation
    market = SpotMarket(100., DiscountCurve((0., .1, .3, .5, 1.),
        (1., 1.02, .99, 1.1, 1.05)), DividendSchedule((), 1.))
    cap = _surrogate_cap(market, strike=100., maturity=.8, time=.2,
        is_call=False, enabled=True)
    assert cap == pytest.approx(100*1.1/market.discount_curve.discount(.2), abs=2e-13)
    assert cap >= 100.
    assert _surrogate_cap(market, strike=100., maturity=.8, time=.8,
        is_call=False, enabled=True) == 100.
    values, spots = np.array([-2., 5., 1e12]), np.array([90., 100., 110.])
    bounded = _decision_continuation(values, spots, market, strike=100., maturity=.8,
        time=.2, is_call=False, enabled=True)
    np.testing.assert_allclose(bounded, [0., 5., cap])
    calls = _decision_continuation(values, spots, market, strike=100., maturity=.8,
        time=.2, is_call=True, enabled=True)
    np.testing.assert_array_equal(calls, [0., 5., 110.])
    np.testing.assert_array_equal(_decision_continuation(values, spots, market,
        strike=100., maturity=.8, time=.2, is_call=True, enabled=False), values)


def test_full_factor_spline_regression_recovers_cross_state_values():
    rng = np.random.default_rng(712)
    spots = rng.uniform(60., 140., 2048)
    factors = rng.normal(size=(2048, 3))
    z = spots/100.
    target = (2.+z+.2*z*z+3*np.maximum(z-.9, 0.)**2
        -2*np.maximum(z-1.1, 0.)**2+factors[:, 0]-.5*factors[:, 1]
        +2*z*factors[:, 2]+.3*factors[:, 0]*factors[:, 2])
    config = RoughAmericanConfig(stock_knots=(.9, 1.1), ridge=0.)
    fitted = _fit_continuation(spots, factors, target, 100., config)
    np.testing.assert_allclose(fitted.value(spots, factors, 100.), target, atol=2e-10)
    # Equal aggregate variance does not collapse the Markovian memory state.
    states = np.array([[1., -1., 0.], [-1., 1., 0.]])
    assert fitted.value(np.array([100., 100.]), states, 100.)[0] != pytest.approx(
        fitted.value(np.array([100., 100.]), states, 100.)[1])
    # Bumped-stock broadcasting preserves those same factor coordinates.
    bumped = np.array([[99., 100., 101.], [99., 100., 101.]])
    evaluated = fitted.value(bumped, states, 100.)
    assert evaluated.shape == (2, 3)
    np.testing.assert_allclose(evaluated[:, 1], fitted.value(np.full(2, 100.), states, 100.))


@pytest.mark.parametrize("is_call", [True, False])
def test_one_step_deterministic_variance_exact_bs_and_paired_deltas(inputs, tiny, is_call):
    params, market, kernel = inputs
    params = replace(params, theta=.04, vol_of_vol=0.)
    config = replace(tiny, steps=1, exercise_steps=1)
    result = estimate_rough_american(params, market, kernel, strike=100., maturity=.25,
        is_call=is_call, config=config)
    reference = black_scholes_price(BlackScholesParams(.2), forward=100*np.exp(.04*.25), discount=np.exp(-.04*.25),
        strike=100., maturity=.25, is_call=is_call)
    assert result.prices == pytest.approx(reference, abs=2e-12)
    assert result.dual_prices == pytest.approx(reference, abs=2e-12)
    assert result.standard_errors < 2e-13
    assert result.dual_gaps == pytest.approx(0., abs=2e-13)
    d1 = (.04+.5*.04)*.25/(.2*np.sqrt(.25))
    delta = ndtr(d1)-(not is_call)
    # This is a central finite difference, independently checked against BS.
    assert result.deltas == pytest.approx(delta, abs=3e-5)
    assert result.delta_standard_errors < 2e-12
    assert result.diagnostics["american_pricing_ready"] is False
    assert result.diagnostics["stock_delta_ready"] is False
    assert result.diagnostics["price_absolute_tolerance"] == .001
    assert result.diagnostics["delta_absolute_tolerance"] == .002


def _independent_one_step_price(params, market, kernel, strike, tenor, call, cash, american):
    nodes, weights = np.array(kernel.nodes), np.array(kernel.weights)
    count = len(nodes)
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    exact = _drift_flow_with_integral(matrix, np.full(count, params.kappa*(params.theta-params.v0)),
        weights, params.v0, tenor/2)
    law = _weak_transition_branches(np.zeros((1, count)), params, nodes=nodes, weights=weights,
        flow=exact[:count, :count], shift=exact[:count, -1], integrated_flow=exact,
        dt=tenor, theta=params.theta, growth=.04*tenor, stock_integral="exact_drift")
    price = 0.
    for branch in range(3):
        for order in range(2):
            variance = law["orthogonal"][0, branch, order]
            mean = np.log(market.spot)+law["correlated"][0, branch]-.5*variance
            def integrand(z):
                pre = np.exp(mean+np.sqrt(variance)*z)
                post = max(pre-cash, 0.)
                payoff = max(post-strike if call else strike-post, 0.)
                if american:
                    payoff = max(payoff, max(pre-strike if call else strike-pre, 0.))
                return payoff*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            points = [(np.log(x)-mean)/np.sqrt(variance)
                for x in (strike, cash, strike+cash) if x > 0]
            integral = quad(integrand, -12., 12., epsabs=2e-10,
                points=[point for point in points if -12. < point < 12.], limit=200)[0]
            price += .5*law["probabilities"][0, branch]*integral
    return market.discount_curve.discount(tenor)*price


@pytest.mark.parametrize("cash,call,american", [(0., False, True), (5., True, True),
    (5., False, True), (150., False, False), (150., True, False)])
def test_exact_one_step_discrete_price_independent_scalar_integration(inputs, tiny, cash, call, american):
    params, market, kernel = inputs
    tenor = .25
    market = replace(market, dividends=DividendSchedule((CashDividend(tenor, cash),), 1.))
    result = estimate_rough_american(params, market, kernel, strike=100., maturity=tenor,
        is_call=call, exercise_style="american" if american else "european",
        config=replace(tiny, steps=1, exercise_steps=1))
    reference = _independent_one_step_price(params, market, kernel, 100., tenor, call, cash, american)
    assert result.prices == pytest.approx(reference, abs=3e-8)
    assert result.standard_errors < 3e-13
    assert result.dual_prices == pytest.approx(reference, abs=3e-8)


def test_training_valuation_stream_separation_and_frozen_state(inputs, tiny):
    params, market, kernel = inputs
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny)
    assert policy.diagnostics["normalization_source"] == "training paths only"
    assert policy.diagnostics["independent_training_valuation"] is True
    snapshot = [continuation.beta.copy() for continuation in policy.continuations]
    one = value_rough_stopping_policy(policy)
    two = value_rough_stopping_policy(policy,
        config=replace(tiny, valuation_seed=44))
    assert not np.array_equal(one["raw_samples"], two["raw_samples"])
    for continuation, saved in zip(policy.continuations, snapshot):
        np.testing.assert_array_equal(continuation.beta, saved)
    # Construct a valid config whose valuation stream overlaps the old policy.
    with pytest.raises(ValueError, match="overlaps"):
        value_rough_stopping_policy(policy, config=replace(tiny, training_seed=44, valuation_seed=42))
    with pytest.raises(ValueError):
        policy.continuations[1].beta[0] = 1.


def test_dual_gap_is_paired_nonnegative_and_control_is_at_every_microstep(inputs, tiny):
    params, market, kernel = inputs
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny)
    value = value_rough_stopping_policy(policy)
    assert np.min(value["dual_samples"]-value["primal_samples"]) >= -3e-13
    assert value["diagnostics"]["martingale_every_integration_step"] is True
    assert value["diagnostics"]["actual_time_steps"] > tiny.exercise_steps
    assert "discrete" in value["diagnostics"]["bound_scope"]
    assert not value["primal_samples"].flags.writeable


def test_european_terminal_only_primal_equals_dual_even_with_cash(inputs, tiny):
    params, market, kernel = inputs
    market = replace(market, dividends=DividendSchedule((CashDividend(.125, 150.),), 1.))
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny, exercise_style="european")
    value = value_rough_stopping_policy(policy)
    np.testing.assert_array_equal(value["primal_samples"], value["dual_samples"])
    np.testing.assert_array_equal(value["stop_times"], .25)
    assert np.isfinite(value["primal_samples"]).all()


def test_pre_dividend_call_exercise_and_post_dividend_put_exercise(inputs, tiny):
    params, market, kernel = inputs
    params = replace(params, v0=0., theta=0., vol_of_vol=0.)
    market = SpotMarket(120., DiscountCurve.flat(0., 1.),
        DividendSchedule((CashDividend(.125, 30.),), 1.))
    # With no time value or positive rate, the call should retain its pre-event
    # intrinsic value; the put gains from waiting until after the cash event.
    result = estimate_rough_american(params, market, kernel, strike=100., maturity=.25,
        is_call=True, config=tiny)
    assert result.prices == pytest.approx(20., abs=1e-12)
    assert result.exercise_probabilities == 1.
    put = estimate_rough_american(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny)
    assert put.prices == pytest.approx(10., abs=1e-12)
    assert put.dual_prices == pytest.approx(10., abs=1e-12)


def test_cash_and_curve_dates_align_without_resetting_factor_memory(tiny):
    from rvlab.pricing.variance_curve import RoughHestonCurveParams, MeanReversionCurve
    params = RoughHestonCurveParams(.04, 1.5, .3, -.6,
        MeanReversionCurve((.07, .18), (.035, .065, .045)), .15)
    market = SpotMarket(100., DiscountCurve((0., .11, 1.), (1., 1.001, .96)),
        DividendSchedule((CashDividend(.09, 1.5),), 1.))
    kernel = ExponentialKernel((.3, 3.), (1., 1.), .15, 1.)
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=True, config=tiny)
    assert .09 in policy.exercise_times
    assert policy.diagnostics["factor_state_dimension"] == 2
    value = value_rough_stopping_policy(policy)
    for time in (.07, .09, .11, .18):
        assert time in value["diagnostics"]["time_grid"]


def test_broadcasting_order_expiration_one_sided_sensitivities(inputs, tiny):
    params, market, kernel = inputs
    result = estimate_rough_american(params, market, kernel, strike=[[95., 100., 105.]],
        maturity=0., is_call=[[True, True, False]], config=tiny)
    np.testing.assert_array_equal(result.prices, [[5., 0., 5.]])
    assert result.prices.shape == (1, 3)
    assert result.one_sided_deltas.shape == (1, 3, 2)
    assert result.delta_left[0, 1] == 0.
    assert result.delta_right[0, 1] == 1.
    assert result.deltas[0, 1] == .5
    np.testing.assert_array_equal(result.primal_standard_errors, result.standard_errors)
    np.testing.assert_array_equal(result.paired_gaps, result.dual_gaps)
    with pytest.raises(ValueError):
        result.prices[0, 0] = 3.


def test_batch_invariance_and_sampling_diagnostics(inputs, tiny):
    params, market, kernel = inputs
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny)
    one = value_rough_stopping_policy(policy, config=replace(tiny, batch_size=64))
    two = value_rough_stopping_policy(policy, config=replace(tiny, batch_size=7))
    np.testing.assert_allclose(one["primal_samples"], two["primal_samples"], atol=2e-12)
    np.testing.assert_array_equal(one["raw_samples"], two["raw_samples"])
    np.testing.assert_allclose(one["dual_samples"], two["dual_samples"], atol=2e-12)


def test_retrained_bumps_are_visible_and_use_training_not_valuation_paths(inputs, tiny):
    params, market, kernel = inputs
    params = replace(params, theta=.04, vol_of_vol=0.)
    result = estimate_rough_american(params, market, kernel, strike=100., maturity=.25,
        config=replace(tiny, steps=1, exercise_steps=1, retrain_bumps=True))
    assert result.diagnostics["retrained_bump_check_executed"] is True
    assert result.retrained_deltas == pytest.approx(result.deltas, abs=2e-12)
    assert result.retrained_delta_standard_errors < 2e-12


@pytest.mark.parametrize("kwargs", [
    {"strike": True}, {"maturity": -.1}, {"is_call": 1},
    {"exercise_style": "bermudan"}, {"conditional_state": np.zeros(1)},
])
def test_invalid_contract_or_conditional_memory_rejects(inputs, tiny, kwargs):
    params, market, kernel = inputs
    request = dict(strike=100., maturity=.25, config=tiny)
    request.update(kwargs)
    with pytest.raises((ValueError, UnsupportedModelError)):
        estimate_rough_american(params, market, kernel, **request)


def test_memory_budget_rejects_before_valuation_allocation(inputs, tiny):
    params, market, kernel = inputs
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        config=tiny)
    with pytest.raises(MemoryError, match="exceeding"):
        value_rough_stopping_policy(policy, config=replace(tiny, memory_budget_mb=1e-6))


@pytest.mark.parametrize("cash", [0., 1.5, 150.])
def test_frozen_policy_valuation_matches_public_simulator_raw_path_law(inputs, tiny, cash):
    params, market, kernel = inputs
    market = replace(market, dividends=DividendSchedule((CashDividend(.125, cash),), 1.))
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny, exercise_style="european")
    value = value_rough_stopping_policy(policy)
    simulation = simulate_rough_heston(params, market, kernel,
        observation_times=policy.continuation_times,
        config=RoughSimulationConfig(tiny.valuation_paths, tiny.steps, tiny.valuation_seed,
            tiny.batch_size, tiny.memory_budget_mb))
    raw = np.exp(-.04*.25)*np.maximum(100.-simulation.spots[:, -1], 0.)
    np.testing.assert_allclose(value["raw_samples"][:, 0], raw, atol=2e-12)


def test_nonnested_european_training_dates_remain_in_valuation_law(inputs, tiny):
    params, market, kernel = inputs
    config = replace(tiny, steps=3, exercise_steps=2)
    policy = train_rough_stopping_policy(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=config, exercise_style="european")
    value = value_rough_stopping_policy(policy)
    for time in policy.continuation_times:
        assert time in value["diagnostics"]["time_grid"]
    assert .125 in value["diagnostics"]["time_grid"]
    assert value["diagnostics"]["exercise_dates"] == [.25]
    simulation = simulate_rough_heston(params, market, kernel,
        observation_times=policy.continuation_times,
        config=RoughSimulationConfig(config.valuation_paths, config.steps, config.valuation_seed,
            config.batch_size, config.memory_budget_mb))
    np.testing.assert_allclose(value["raw_samples"][:, 0],
        np.exp(-.04*.25)*np.maximum(100.-simulation.spots[:, -1], 0.), atol=2e-12)


def test_conditional_bellman_labels_have_no_terminal_cashflow_noise(inputs, tiny):
    from rvlab.pricing.rough_stopping import _conditional_training_labels, _zero_continuation
    params, market, kernel = inputs
    params = replace(params, theta=.04, vol_of_vol=0.)
    spots = np.array([90., 100., 110.])
    factors = np.array([[.01], [0.], [-.01]])
    left, right = .1, .25
    continuation = _zero_continuation(factors, tiny)
    labels = _conditional_training_labels(params, market, kernel, spots, factors,
        left=left, right=right, continuation=continuation, config=tiny,
        strike=100., is_call=False, exercise=True, payoff_side="post")
    integrated = (params.theta*(right-left)
        +(params.v0+factors[:, 0]-params.theta)*(-np.expm1(-params.kappa*(right-left)))/params.kappa)
    reference = np.array([black_scholes_price(BlackScholesParams(np.sqrt(clock/(right-left))),
        forward=spot*np.exp(.04*(right-left)), discount=np.exp(-.04*(right-left)),
        strike=100., maturity=right-left, is_call=False) for spot, clock in zip(spots, integrated)])
    np.testing.assert_allclose(labels, reference, atol=3e-12)
    repeated = _conditional_training_labels(params, market, kernel, spots, factors,
        left=left, right=right, continuation=continuation,
        config=replace(tiny, training_seed=11, valuation_seed=12),
        strike=100., is_call=False, exercise=True, payoff_side="post")
    np.testing.assert_array_equal(labels, repeated)


def test_general_deterministic_variance_uses_exact_path_integral(inputs, tiny):
    from rvlab.pricing.rough_lift import lifted_integrated_variance
    params, market, kernel = inputs
    params = replace(params, v0=.01, theta=.16, kappa=3., vol_of_vol=0.)
    tenor = .5
    result = estimate_rough_american(params, market, kernel, strike=100., maturity=tenor,
        is_call=False, config=replace(tiny, steps=1, exercise_steps=1), exercise_style="european")
    integrated = lifted_integrated_variance(params, tenor, kernel)
    expected = params.theta*tenor+(params.v0-params.theta)*(-np.expm1(-params.kappa*tenor))/params.kappa
    assert integrated == pytest.approx(expected, abs=2e-16)
    reference = black_scholes_price(BlackScholesParams(np.sqrt(expected/tenor)),
        forward=100*np.exp(.04*tenor), discount=np.exp(-.04*tenor), strike=100.,
        maturity=tenor, is_call=False)
    assert result.prices == pytest.approx(reference, abs=3e-12)
    assert result.standard_errors < 3e-13


def test_flat_mean_reversion_curve_matches_constant_params_same_path_law(inputs, tiny):
    from rvlab.pricing.variance_curve import RoughHestonCurveParams, MeanReversionCurve
    params, market, kernel = inputs
    curve = RoughHestonCurveParams(params.v0, params.kappa, params.vol_of_vol,
        params.rho, MeanReversionCurve((.125,), (params.theta, params.theta)), .5)
    a = estimate_rough_american(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny)
    b = estimate_rough_american(curve, market, kernel, strike=100., maturity=.25,
        is_call=False, config=tiny)
    np.testing.assert_allclose(a.prices, b.prices, atol=2e-12)
    np.testing.assert_allclose(a.deltas, b.deltas, atol=2e-12)
    np.testing.assert_allclose(a.dual_prices, b.dual_prices, atol=2e-12)


def test_heston_limit_runs_new_stopping_path_not_existing_fd_backend(inputs, monkeypatch):
    from rvlab.pricing._vanilla_heston import solve_heston
    from rvlab.pricing import FDConfig, HestonParams
    params, market, kernel = inputs
    reference = solve_heston(HestonParams(params.v0, params.theta, params.kappa,
        params.vol_of_vol, params.rho), market, np.array([100.]), .25,
        np.array([False]), FDConfig(240, 80, 300))
    def forbidden(*args, **kwargs):
        raise AssertionError("H=.5 test must execute the new rough stopping scheme")
    monkeypatch.setattr("rvlab.pricing._vanilla_heston.solve_heston", forbidden)
    config = RoughAmericanConfig(training_paths=512, valuation_paths=512,
        steps=64, exercise_steps=16)
    result = estimate_rough_american(params, market, kernel, strike=100., maturity=.25,
        is_call=False, config=config)
    # This stochastic consistency test is deliberately separate from strict
    # production acceptance.  The study reports the original price/delta
    # gates; a broad confidence envelope here never sets readiness true.
    assert result.prices-5*result.standard_errors <= reference["prices"][0]
    assert reference["prices"][0] <= result.dual_prices+5*result.dual_standard_errors
    assert abs(result.deltas-reference["deltas"][0]) <= 5*result.delta_standard_errors
    assert result.diagnostics["american_pricing_ready"] is False
