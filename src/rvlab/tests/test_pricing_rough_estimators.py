"""The estimator controls the discrete law, never a continuous reference."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.pricing import CashDividend, DividendSchedule, DiscountCurve, PricingConfig, RoughHestonParams, SpotMarket
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig, _AffineSurrogate, _centered_increment, _conditional_payoffs, _estimate_result, estimate_rough_european
from rvlab.pricing.rough_lift import ExponentialKernel, lifted_heston_price
from rvlab.pricing.rough_simulation import RoughSimulationConfig, RoughSimulationError, _drift_flow_with_integral, _weak_transition_branches, european_payoff_samples, simulate_rough_heston
from rvlab.pricing.vanilla_types import UnsupportedModelError


@pytest.fixture
def inputs():
    return (RoughHestonParams(.04, .045, 1.5, .3, -.6, .5),
        SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.)),
        ExponentialKernel((0.,), (1.,), .5, 1.))


def test_centered_control_has_exact_discrete_conditional_mean():
    rng = np.random.default_rng(17)
    values = rng.normal(size=(7, 3, 2, 4))
    probabilities = rng.dirichlet([1., 1., 1.], size=7)
    expectation = np.zeros((7, 4))
    for branch in range(3):
        for endpoint in range(2):
            increment = _centered_increment(values, probabilities,
                np.full(7, branch), np.full(7, endpoint))
            expectation += probabilities[:, branch, None]*.5*increment
    np.testing.assert_allclose(expectation, 0., atol=4e-16)


@pytest.mark.parametrize("stock_integral", ["trapezoidal", "exact_drift"])
def test_single_step_is_exact_discrete_price_and_preserves_time_bias(inputs, stock_integral):
    params, market, kernel = inputs
    tenor = .5
    strikes, calls = np.array([95., 100., 105.]), np.array([False, True, True])
    nodes, weights = np.array(kernel.nodes), np.array(kernel.weights)
    matrix = -np.diag(nodes)-params.kappa*np.ones((1, 1))*weights
    exact = _drift_flow_with_integral(matrix, np.full(1, params.kappa*(params.theta-params.v0)), weights, params.v0, tenor/2)
    branches = _weak_transition_branches(np.zeros((1, 1)), params, nodes=nodes, weights=weights,
        flow=exact[:1, :1], shift=exact[:1, -1], integrated_flow=exact, dt=tenor,
        theta=params.theta, growth=.04*tenor, stock_integral=stock_integral)
    mean = np.log(market.spot)+branches["correlated"][:, :, None]-.5*branches["orthogonal"]
    payoffs = _conditional_payoffs(mean[..., None], branches["orthogonal"][..., None], strikes, calls)
    reference = market.discount_curve.discount(tenor)*np.sum(branches["probabilities"][:, :, None]*payoffs.mean(axis=2), axis=1)[0]
    result = estimate_rough_european(params, market, kernel, strike=strikes, maturity=tenor,
        is_call=calls, config=RoughSimulationConfig(paths=64, steps=1, stock_integral=stock_integral),
        control_config=RoughEuropeanControlConfig(2, 1., 2))
    np.testing.assert_allclose(result.prices, reference, atol=2e-13)
    assert np.max(result.standard_errors) < 2e-14
    continuous = lifted_heston_price(params, forward=100*np.exp(.04*tenor), discount=np.exp(-.04*tenor),
        strike=strikes, maturity=tenor, is_call=calls, kernel=kernel, config=PricingConfig(192, 160., 512))
    assert np.max(abs(result.prices-continuous)) > .01
    assert result.diagnostics["continuous_reference_used_to_center"] is False


@pytest.mark.parametrize("stock_integral", ["trapezoidal", "exact_drift"])
def test_conditional_samples_match_public_simulator_same_uniforms(inputs, stock_integral):
    params, market, kernel = inputs
    config = RoughSimulationConfig(paths=64, steps=12, stock_integral=stock_integral)
    strikes, maturities, calls = np.array([105., 95., 100.]), np.array([.5, .125, .5]), np.array([True, False, True])
    result = estimate_rough_european(params, market, kernel, strike=strikes, maturity=maturities, is_call=calls, config=config)
    simulation = simulate_rough_heston(params, market, kernel, observation_times=np.unique(maturities), config=config)
    samples = np.column_stack([european_payoff_samples(simulation, strike=k, maturity=t, is_call=bool(call))
        for k, t, call in zip(strikes, maturities, calls)])
    np.testing.assert_allclose(result.conditional_prices, samples.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(result.conditional_standard_errors, samples.std(axis=0, ddof=1)/8, atol=2e-13)
    np.testing.assert_allclose(result.prices+result.control_means, result.conditional_prices, atol=2e-14)
    np.testing.assert_allclose(result.standard_errors**2, result.sample_covariances[:, 2, 2]/config.paths, atol=1e-18)


def test_surrogate_initial_value_agrees_with_independent_lift_price(inputs):
    params, market, kernel = inputs
    tenor = .5
    strikes, calls = np.array([95., 100., 105.]), np.array([False, True, True])
    proxy = _AffineSurrogate(params, market, kernel, strikes, np.full(3, tenor), calls,
        np.array([0., tenor]), RoughEuropeanControlConfig(192, 160., 1024))
    values = proxy.values(0., np.zeros((1, 1, 1)), np.full((1, 1, 1), np.log(100.)), np.zeros((1, 1, 1)), np.arange(3))[0, 0, 0]
    reference = lifted_heston_price(params, forward=100*np.exp(.04*tenor), discount=np.exp(-.04*tenor),
        strike=strikes, maturity=tenor, is_call=calls, kernel=kernel, config=PricingConfig(192, 160., 1024))
    np.testing.assert_allclose(values, reference, atol=8e-7)


def test_conditional_affine_coefficients_and_future_moment_match_heston(inputs):
    from rvlab.pricing.heston import heston_cf
    from rvlab.pricing.types import HestonParams
    params, market, kernel = inputs
    time, tenor, state = .125, .5, .02
    proxy = _AffineSurrogate(params, market, kernel, np.array([100.]), np.array([tenor]),
        np.array([True]), np.array([0., time, tenor]), RoughEuropeanControlConfig(24, 40., 2048))
    a, b = proxy.coefficients[(tenor, time)]
    conditional_cf = np.exp(a+b[:, 0]*state)
    reference = heston_cf(HestonParams(params.v0+state, params.theta, params.kappa,
        params.vol_of_vol, params.rho), proxy.contour, tenor-time)
    np.testing.assert_allclose(conditional_cf, reference, atol=6e-8)
    coefficient, constant = proxy._remaining_integral(time, tenor)
    tau = tenor-time
    expected_integral = params.theta*tau+(params.v0+state-params.theta)*(-np.expm1(-params.kappa*tau))/params.kappa
    assert coefficient[0]*state+constant == pytest.approx(expected_integral, abs=2e-16)


def test_multifactor_calendar_curve_surrogate_matches_lift_at_inception():
    from rvlab.experiments._pricing_rough_foundation import _market, _truths
    from rvlab.pricing.rough_lift import approximate_fractional_kernel
    params, market = _truths()["curve"], _market()
    kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003)
    tenor, strikes = .5, np.array([95., 100., 105.])
    calls = np.array([False, True, True])
    proxy = _AffineSurrogate(params, market, kernel, strikes, np.full(3, tenor), calls,
        np.array([0., tenor]), RoughEuropeanControlConfig(192, 160., 1024))
    values = proxy.values(0., np.zeros((1, 1, 4)), np.full((1, 1, 1), np.log(100.)),
        np.zeros((1, 1, 1)), np.arange(3))[0, 0, 0]
    reference = lifted_heston_price(params, forward=100*np.exp(.04*tenor), discount=np.exp(-.04*tenor),
        strike=strikes, maturity=tenor, is_call=calls, kernel=kernel, config=PricingConfig(192, 160., 1024))
    np.testing.assert_allclose(values, reference, atol=8e-7)


def test_conditional_curve_coefficients_retain_calendar_clock():
    from rvlab.experiments._pricing_rough_foundation import _market, _truths
    from rvlab.pricing.variance_curve import HestonCurveParams, MeanReversionCurve, heston_curve_cf
    params, market = replace(_truths()["curve"], hurst=.5), _market()
    kernel = ExponentialKernel((0.,), (1.,), .5, 1.)
    time, tenor, state = .08, .5, .02
    proxy = _AffineSurrogate(params, market, kernel, np.array([100.]), np.array([tenor]),
        np.array([True]), np.array([0., time, tenor]), RoughEuropeanControlConfig(24, 40., 2048))
    a, b = proxy.coefficients[(tenor, time)]
    future = HestonCurveParams(params.v0+state, params.kappa, params.vol_of_vol, params.rho,
        MeanReversionCurve((params.curve.knot_times[-1]-time,), (.065, .045)))
    reference = heston_curve_cf(future, proxy.contour, tenor-time)
    np.testing.assert_allclose(np.exp(a+b[:, 0]*state), reference, atol=7e-8)


def test_remaining_time_rounding_does_not_drop_calendar_targets(inputs):
    params, market, kernel = inputs
    tiny = 1e-30
    assert 1.-tiny == 1.
    proxy = _AffineSurrogate(params, market, kernel, np.array([100.]), np.array([1.]),
        np.array([True]), np.array([0., tiny, 1.]), RoughEuropeanControlConfig(8, 20., 16))
    assert (1., 0.) in proxy.coefficients
    assert (1., tiny) in proxy.coefficients
    np.testing.assert_array_equal(proxy.coefficients[(1., 0.)][0], proxy.coefficients[(1., tiny)][0])


@pytest.mark.parametrize("payoffs,controls", [
    ([[np.inf], [1.]], [[0.], [0.]]),
    ([[1.], [1.]], [[0.], [np.nan]]),
    ([[1e308], [-1e308]], [[0.], [0.]]),
    ([[1e308], [1e308]], [[-1e308], [-1e308]]),
])
def test_nonfinite_samples_or_summary_raise_identifiable_failure(payoffs, controls):
    with pytest.raises(RoughSimulationError):
        _estimate_result(np.array(payoffs), np.array(controls), (), RoughSimulationConfig(paths=2),
            RoughEuropeanControlConfig(), {})


def test_memory_budget_covers_summary_peak(inputs):
    params, market, kernel = inputs
    result = estimate_rough_european(params, market, kernel, strike=[95., 100., 105.],
        maturity=.125, config=RoughSimulationConfig(paths=16, steps=2))
    # Summary alone retains both inputs, controlled values, stacked values,
    # and centered values. The reported budget must cover at least 9 vectors.
    assert result.diagnostics["estimated_memory_bytes"] >= 9*16*3*8


def test_control_reduces_sampling_uncertainty_without_fitting_coefficients(inputs):
    params, market, kernel = inputs
    result = estimate_rough_european(params, market, kernel, strike=[95., 100., 105.], maturity=.5,
        is_call=[False, True, True], config=RoughSimulationConfig(paths=256, steps=32))
    assert np.all(result.standard_errors < .15*result.conditional_standard_errors)
    assert result.diagnostics["control_coefficient"] == 1.
    assert result.diagnostics["american_pricing_ready"] is False
    assert result.diagnostics["stock_delta_ready"] is False
    with pytest.raises(ValueError):
        result.prices[0] = 0.


def test_broadcasting_contract_order_and_batch_invariance(inputs):
    params, market, kernel = inputs
    config = RoughSimulationConfig(paths=32, steps=8, batch_size=32)
    options = dict(strike=np.array([[95., 100., 105.]]), maturity=np.array([[.125], [.5]]), is_call=np.array([[False, True, True]]))
    result = estimate_rough_european(params, market, kernel, **options, config=config)
    batched = estimate_rough_european(params, market, kernel, **options, config=replace(config, batch_size=7))
    assert result.prices.shape == (2, 3)
    assert result.sample_covariances.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(result.prices, batched.prices, atol=3e-13)
    np.testing.assert_allclose(result.standard_errors, batched.standard_errors, atol=1e-14)
    scalar = estimate_rough_european(params, market, kernel, strike=100., maturity=.5, config=config)
    assert scalar.prices.shape == ()


def test_deterministic_and_expiry_controls(inputs):
    params, market, kernel = inputs
    params = replace(params, vol_of_vol=0.)
    result = estimate_rough_european(params, market, kernel, strike=[95., 105.], maturity=[0., .5],
        is_call=[False, True], config=RoughSimulationConfig(paths=8, steps=8))
    tenors = np.array([0., .5])
    reference = lifted_heston_price(params, forward=100*np.exp(.04*tenors), discount=np.exp(-.04*tenors),
        strike=[95., 105.], maturity=tenors, is_call=[False, True], kernel=kernel)
    np.testing.assert_allclose(result.prices, reference, atol=2e-12)
    assert np.max(result.standard_errors) < 1e-14
    assert np.all(result.control_means == 0)


@pytest.mark.parametrize("kwargs", [{"strike": True}, {"strike": 1+2j}, {"strike": 0.}, {"strike": []},
    {"maturity": True}, {"maturity": -.1}, {"maturity": float("nan")}, {"is_call": 1}, {"is_call": "call"}])
def test_malformed_contracts_rejected(inputs, kwargs):
    params, market, kernel = inputs
    with pytest.raises(ValueError):
        estimate_rough_european(params, market, kernel, **{"strike": 100., "maturity": .5, **kwargs})


@pytest.mark.parametrize("kwargs", [{"quadrature_nodes": True}, {"quadrature_nodes": 1},
    {"quadrature_nodes": 2.5}, {"riccati_steps": 1}, {"fourier_cutoff": 0.}, {"fourier_cutoff": float("inf")}])
def test_malformed_control_config_rejected(kwargs):
    with pytest.raises(ValueError):
        RoughEuropeanControlConfig(**kwargs)


def test_coverage_cash_events_memory_and_config_rejected(inputs):
    params, market, kernel = inputs
    with pytest.raises(ValueError, match="cover"):
        estimate_rough_european(params, replace(market, dividends=DividendSchedule((), .1)), kernel, strike=100., maturity=.5)
    with pytest.raises(UnsupportedModelError, match="cash"):
        estimate_rough_european(params, replace(market, dividends=DividendSchedule((CashDividend(.1, 1.),), 1.)), kernel, strike=100., maturity=.5)
    with pytest.raises(MemoryError):
        estimate_rough_european(params, market, kernel, strike=100., maturity=.5, config=RoughSimulationConfig(paths=32, steps=8, memory_budget_mb=.001))
    with pytest.raises(ValueError, match="two"):
        estimate_rough_european(params, market, kernel, strike=100., maturity=.5, config=RoughSimulationConfig(paths=1))
    with pytest.raises(TypeError, match="control_config"):
        estimate_rough_european(params, market, kernel, strike=100., maturity=.5, control_config={})
