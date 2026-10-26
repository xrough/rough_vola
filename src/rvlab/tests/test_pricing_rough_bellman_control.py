"""Bellman value proxies remain bounded functions with exact spline centering."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_bellman_control import (
    fit_bellman_correction, target_nodal_premiums, bellman_target_martingales)
from rvlab.experiments._pricing_rough_bellman_audit import adaptive_block_expectation
from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
from rvlab.experiments._pricing_rough_local_control import _linear_coefficients
from rvlab.experiments._pricing_rough_node_refinement_fast import refined_stock_knots, signed_shared_expectations
from rvlab.pricing import DiscountCurve, DividendSchedule, ExponentialKernel, RoughHestonParams, SpotMarket
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig, _AffineSurrogate
from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
from rvlab.pricing.rough_stopping import RoughAmericanConfig, _Continuation
from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy, fourier_policy_fingerprint
from rvlab.pricing.rough_simulation import _time_grid
from rvlab.pricing.rough_tensor_premium import TensorPremium


def constant_tensor(value, knots=(.7, 1., 1.3)):
    beta = np.zeros((3+len(knots), 3)); beta[0, 0] = value
    return TensorPremium(np.zeros(2), np.ones(2), beta, knots, 1, 3., 1, 1.)


@pytest.fixture(scope="module")
def control_case():
    params = RoughHestonParams(.04, .04, 1.5, .3, -.6, .15)
    market = SpotMarket(100., DiscountCurve.flat(1., 1.), DividendSchedule((), 1.))
    kernel = ExponentialKernel((.5, 2.), (.4, .6), .15, 1.)
    config = RoughAmericanConfig(training_paths=8, valuation_paths=4, training_seed=10,
        valuation_seed=11, steps=4, exercise_steps=2, batch_size=2, stock_knots=(.7, 1., 1.3))
    times = np.array([0., .1, .2]); n = 2; h = len(config.stock_knots)
    beta = np.zeros(3+h+2*n+n*(n+1)//2)
    original = _Continuation(np.zeros(2), np.ones(2), beta, config.stock_knots, 1, 1.)
    base = FourierStoppingPolicy(params, market, kernel, 100., .2, False, "american", times, times,
        (original,)*3, (original,)*3, config, RoughEuropeanControlConfig(), False, {}, .7)
    proxy = build_rough_obstacle_proxy(base, stock_knots=config.stock_knots)
    models = (constant_tensor(.1), constant_tensor(.1), constant_tensor(0.))
    selected = ControlledExercisePolicy(proxy, times, models, False, (10, 12), 0., .7, {})
    delta = (None, constant_tensor(.2), None)
    grid = _time_grid(params, market, times, 4)
    stock = np.array([[[100.], [99.], [98.]], [[100.], [101.], [102.]]])
    factors = np.zeros((2, 3, 2)); affine = _AffineSurrogate(params, market, kernel,
        np.array([100.]), np.array([.2]), np.array([False]), grid, proxy.control_config)
    euro = np.column_stack([affine.values(float(t), factors[:, j:j+1], np.log(stock[:, j:j+1, 0])[:, :, None],
        np.zeros((2, 1, 1)), np.array([0]))[:, 0, 0, 0] for j, t in enumerate(times)])[:, :, None]
    trace = {"stocks": stock, "factors": factors, "european_proxy": euro, "times": times,
        "actual_time_grid": grid, "discounts": market.discount_curve.discount(times), "seed": 100,
        "policy_fingerprint": fourier_policy_fingerprint(proxy), "kernel_fingerprint": kernel.fingerprint,
        "strike": 100., "initial_spots": np.array([100.])}
    return proxy, selected, delta, trace


def test_nodal_q_clip_endpoint_trials_and_uncapped_exercise_obstacle(control_case):
    proxy, selected, models, _ = control_case
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    values = target_nodal_premiums(proxy, selected, models, 1, np.zeros((2, 2)), fine_nodes=nodes)
    assert values.shape == (2, 4, len(nodes))
    # The naive upper premium survives; constrained targets have zero q there.
    np.testing.assert_allclose(values[:, 0, -1], .1, atol=1e-12)
    np.testing.assert_array_equal(values[:, 1:, -1], 0.)
    # At S=0 the immediate-exercise premium greatly exceeds the .7 q cap.
    assert np.all(values[..., 0] > .7)
    atmoney = int(np.flatnonzero(nodes == 1.)[0])
    np.testing.assert_allclose(values[:, 3, atmoney]-values[:, 1, atmoney], .05, atol=1e-12)


def test_right_factor_reliability_limits_correction_without_clipping_states(control_case):
    proxy, selected, models, _ = control_case
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    memory = np.array([[0., 0.], [.01, .02]])
    before = memory.copy()
    base = target_nodal_premiums(proxy, selected, models, 1, memory, fine_nodes=nodes)
    suppressed = list(models); suppressed[1] = replace(models[1], center=np.full(2, 100.))
    out = target_nodal_premiums(proxy, selected, suppressed, 1, memory, fine_nodes=nodes)
    np.testing.assert_array_equal(memory, before)
    np.testing.assert_allclose(out[:, 3], out[:, 1], atol=1e-12)
    assert np.max(abs(base[:, 3]-out[:, 3])) > .01


def test_four_signed_spline_expectations_match_independent_scalar_quadrature(control_case):
    proxy, selected, models, trace = control_case
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    law = exercise_block_law(proxy, np.zeros((1, 2)), times=trace["actual_time_grid"][:3])
    values = target_nodal_premiums(proxy, selected, models, 1, law["new_factors"], fine_nodes=nodes)
    coefficients = _linear_coefficients(nodes, values)
    mean = np.log(100.)+law["log_stock_shifts"]
    expected = signed_shared_expectations(mean, law["orthogonal"], coefficients, strike=100., knots=nodes[1:])
    expected = np.sum(expected.mean(axis=-1)*law["probabilities"][..., None], axis=1)[0]
    cache = {}
    for target in range(4):
        def function(stocks, memory):
            key = tuple(memory[0])
            if key not in cache:
                cache[key] = target_nodal_premiums(proxy, selected, models, 1, memory[:1], fine_nodes=nodes)[0]
            return np.interp(stocks/100., nodes, cache[key][target])
        reference = adaptive_block_expectation(law, [100.], function, stock_breakpoints=100.*nodes[1:])
        np.testing.assert_allclose(expected[target], reference["expectation"][0], atol=2e-11, rtol=0.)


def test_terminal_premiums_zero_and_batch_changes_preserve_every_martingale(control_case):
    proxy, selected, models, trace = control_case
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    np.testing.assert_array_equal(target_nodal_premiums(proxy, selected, models, 2, np.zeros((2, 2)), fine_nodes=nodes), 0.)
    first, diagnostic = bellman_target_martingales(proxy, selected, models, trace, batch_size=1)
    second, _ = bellman_target_martingales(proxy, selected, models, trace, batch_size=2)
    np.testing.assert_allclose(first, second, atol=2e-13, rtol=0.)
    np.testing.assert_array_equal(first[:, 0], 0.)
    assert first.shape == (2, 3, 4) and diagnostic["right_factor_reliability_inside_expectation"]
    assert not diagnostic["snell_premium_capped"]


@pytest.mark.parametrize("seed", [10, 12, 100])
def test_target_paths_cannot_reuse_any_proxy_fit_or_prior_seed(control_case, seed):
    proxy, selected, models, trace = control_case
    with pytest.raises(ValueError, match="independent"):
        bellman_target_martingales(proxy, selected, models, {**trace, "seed": seed}, forbidden_seeds=(100,))


def test_spent_regression_preserves_outliers_and_all_rows(control_case):
    proxy, _, _, _ = control_case; n = 32
    stocks = np.zeros((n, 3, 1))+99.; memory = np.zeros((n, 3, 2))
    memory[:, 1, 0] = np.linspace(-.01, .01, n); memory[:, 1, 1] = np.linspace(-.02, .02, n)**2
    mask = np.zeros((n, 3), bool); mask[:, 1] = True
    baseline = np.ones((n, 3)); continuation = np.full((n, 3, 3), np.nan)
    target = np.linspace(-.01, .01, n); target[-1] = .15
    continuation[:, 1] = 1.+target[:, None]
    arrays = dict(stocks=stocks, factors=memory, bellman_computed_mask=mask,
        baseline_continuation=baseline, bellman_continuations=continuation)
    before = continuation.copy()
    models, diagnostics = fit_bellman_correction(proxy, arrays, arrays)
    assert models[0] is models[-1] is None and models[1] is not None
    assert diagnostics.rows.tolist() == [32, 32]
    np.testing.assert_allclose(diagnostics.label_maximum, .15)
    assert not diagnostics.regression_rows_clipped_or_removed.any()
    np.testing.assert_array_equal(continuation, before)
