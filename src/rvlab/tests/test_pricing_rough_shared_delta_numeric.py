"""Exact transition sharing and stock-state controls for shared-price deltas."""
from dataclasses import replace
import itertools
import numpy as np
import pytest

from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_shared_delta import (
    composed_law, _actual_splines, _expected_splines, physical_trace,
    RELATIVE_SPOTS,
)
from rvlab.experiments._pricing_rough_local_control import _linear_coefficients
from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
from rvlab.experiments._pricing_rough_mc_precision import shared_expectations


@pytest.mark.parametrize('steps', [1, 2])
def test_composed_law_preserves_original_tree_bit_exact(local_case, steps):
    proxy, _, trace = local_case
    times = np.linspace(0., proxy.maturity/64, steps+1)
    factors = trace['factors'][:3, 1]
    expected = exercise_block_law(proxy, factors, times=times)
    actual = composed_law(proxy, factors, times=times)
    for key in ('new_factors', 'probabilities', 'log_stock_shifts', 'orthogonal'):
        np.testing.assert_array_equal(actual[key], expected[key])


def test_four_step_composition_preserves_every_branch_and_order(local_case):
    proxy, _, trace = local_case
    times = np.linspace(0., proxy.maturity/64, 5)
    actual = composed_law(proxy, trace['factors'][:2, 1], times=times)
    assert actual['new_factors'].shape == (2, 81, 2)
    assert actual['orthogonal'].shape == (2, 81, 16)
    np.testing.assert_allclose(actual['probabilities'].sum(axis=1), 1., atol=1e-15, rtol=0.)
    # Independently join two existing two-step trees, without averaging clocks.
    left = exercise_block_law(proxy, trace['factors'][:2, 1], times=times[:3])
    right = exercise_block_law(proxy, left['new_factors'].reshape(18, 2), times=times[2:])
    clocks = (left['orthogonal'][:, :, None, :, None]
        +right['orthogonal'].reshape(2, 9, 9, 4)[:, :, :, None]).reshape(2, 81, 16)
    means = (left['log_stock_shifts'][:, :, None, :, None]
        +right['log_stock_shifts'].reshape(2, 9, 9, 4)[:, :, :, None]).reshape(2, 81, 16)
    np.testing.assert_allclose(actual['orthogonal'], clocks, atol=2e-18, rtol=0.)
    np.testing.assert_allclose(actual['log_stock_shifts'], means, atol=2e-17, rtol=0.)


@pytest.mark.parametrize('clock', [0., 1e-12, .02])
def test_multistock_splines_match_five_independent_scalar_integrals(clock):
    nodes = np.array([0., .8, 1., 1.3])
    values = np.array([[.3, -.2, .1, 0.], [.1, .2, -.1, .4]])
    coefficient = _linear_coefficients(nodes, values)[None, None]
    stocks = 100*(1+RELATIVE_SPOTS)[None]
    mean = np.log(stocks)[:, None, None]
    variance = np.full((1, 1, 1, 1), clock)
    result = _expected_splines(mean, variance, coefficient, strike=100., knots=nodes[1:])
    for k in range(5):
        scalar = shared_expectations(mean[..., k], np.full((1, 1, 1), clock),
            coefficient, strike=100., knots=nodes[1:])
        np.testing.assert_array_equal(result[..., k], scalar)
    actual = _actual_splines(coefficient[:, 0], stocks, strike=100., knots=nodes[1:])
    for k in range(5):
        for family in range(2):
            assert actual[0, k, family] == pytest.approx(
                np.interp(stocks[0, k]/100., nodes, values[family]), abs=2e-16)


def test_microstep_control_is_centered_but_is_not_the_block_control():
    # All four histories of a two-step symmetric walk are retained.
    histories = np.array(list(itertools.product((-1., 1.), repeat=2)))
    x1, x2 = histories[:, 0], histories.sum(axis=1)
    # F(x)=x^3: one-step conditional means are x^3+3x.
    micro = x1**3 + x2**3-(x1**3+3*x1)
    block = x2**3
    assert micro.mean() == 0. and block.mean() == 0.
    assert np.max(abs(micro-block)) == 3.
    # A calendar-left constant multiplier preserves both expectations.
    assert (2.5*micro).mean() == 0.


@pytest.mark.parametrize('steps', [128, 256])
def test_physical_five_stock_rng_and_martingales_match_independent_valuer(local_case, steps):
    from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy
    proxy, _, _ = local_case
    proxy = replace(proxy, exercise_times=np.linspace(0., proxy.maturity, 65),
        config=replace(proxy.config, steps=128, batch_size=512))
    bundle = {'proxy':proxy}
    actual, conditional = physical_trace(bundle, seed=15201, paths=8, steps=steps)
    config = replace(proxy.config, valuation_seed=15201, valuation_paths=8, steps=steps)
    expected = value_fourier_stopping_policy(proxy, config=config,
        spots=proxy.market.spot*(1+RELATIVE_SPOTS), trace=True)['trace']
    for key in ('stocks', 'factors', 'discounted_payoffs', 'european_martingale',
                'premium_martingale', 'intrinsic_martingale', 'micro_boundary_martingale'):
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-13, rtol=0.)
    np.testing.assert_array_equal(conditional['physical_stocks'][:, ::steps//64], actual['stocks'])
    np.testing.assert_array_equal(conditional['physical_factors'][:, ::steps//64], actual['factors'])
    np.testing.assert_allclose(np.diff(conditional['conditional_log_means'], axis=-1),
        np.broadcast_to(np.diff(np.log(actual['initial_spots'])),
            np.diff(conditional['conditional_log_means'], axis=-1).shape), atol=1e-14, rtol=0.)


@pytest.mark.parametrize('bad', [True, -1, 3.5])
def test_physical_stream_rejects_malformed_seed_before_work(local_case, bad):
    with pytest.raises(ValueError, match='seed'):
        physical_trace({'proxy':local_case[0]}, seed=bad, paths=8, steps=128)

