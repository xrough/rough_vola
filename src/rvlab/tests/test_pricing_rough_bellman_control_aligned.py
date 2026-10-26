"""Physical-grid target alignment and primitive centering audits."""
from dataclasses import replace
import numpy as np
import pytest

from rvlab.tests.test_pricing_rough_bellman_control import control_case
from rvlab.experiments._pricing_rough_bellman_control_aligned import (
    aligned_node_proxy, target_nodal_premiums, bellman_target_martingales)
from rvlab.experiments._pricing_rough_bellman_audit import adaptive_block_expectation
from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
from rvlab.experiments._pricing_rough_local_control import _linear_coefficients
from rvlab.experiments._pricing_rough_node_refinement_fast import refined_stock_knots, signed_shared_expectations
from rvlab.pricing.rough_estimators import _AffineSurrogate


def test_nodal_european_matches_physical_affine_including_zero_and_upper_tail(control_case):
    proxy, _, _, trace = control_case
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([100.]),
        np.array([.2]), np.array([False]), trace['actual_time_grid'], proxy.control_config)
    aligned = aligned_node_proxy(proxy, affine)
    nodes = np.array([0., .1, 1., 2., 10.]); factors = np.array([[0., 0.], [.001, .002]])
    nodal = replace(aligned.value_premiums[1], stock_nodes=nodes).european_values(factors)
    direct = affine.values(.1, factors[:, None], np.broadcast_to(np.log(100.*nodes[1:])[None, None, :], (2, 1, len(nodes)-1)),
        np.zeros((2, 1, len(nodes)-1)), np.array([0]))[:, 0, :, 0]
    np.testing.assert_allclose(nodal[:, 1:]*trace['discounts'][1], direct, atol=2e-12, rtol=0.)
    np.testing.assert_allclose(nodal[:, 0], 100.*trace['discounts'][-1]/trace['discounts'][1])
    assert aligned is not proxy


@pytest.mark.parametrize('stock', [2., 100., 2000.])
def test_aligned_signed_spline_moments_against_scalar_integration(control_case, stock):
    proxy, selected, models, trace = control_case
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    law = exercise_block_law(proxy, np.zeros((1, 2)), times=trace['actual_time_grid'][:3])
    values = target_nodal_premiums(proxy, selected, models, 1, law['new_factors'], fine_nodes=nodes)
    coefficients = _linear_coefficients(nodes, values)
    expected = signed_shared_expectations(np.log(stock)+law['log_stock_shifts'], law['orthogonal'],
        coefficients, strike=100., knots=nodes[1:])
    expected = np.sum(expected.mean(axis=-1)*law['probabilities'][..., None], axis=1)[0]
    cache = {}
    for target in range(4):
        def function(stocks, memory):
            key = tuple(memory[0])
            if key not in cache:
                cache[key] = target_nodal_premiums(proxy, selected, models, 1, memory[:1], fine_nodes=nodes)[0]
            return np.interp(stocks/100., nodes, cache[key][target])
        reference = adaptive_block_expectation(law, [stock], function, stock_breakpoints=100.*nodes[1:])
        np.testing.assert_allclose(expected[target], reference['expectation'][0], atol=2e-10, rtol=0.)


def test_all_target_martingales_reconstruct_from_full_primitives(control_case):
    proxy, selected, models, trace = control_case
    result, diagnostic, components = bellman_target_martingales(proxy, selected, models, trace,
        batch_size=1, return_components=True)
    increments = ((trace['european_proxy'][:, 1:, 0]-components['target_expected_european'][:, 1:])[:, :, None]
        +trace['discounts'][None, 1:, None]*(components['target_actual_premiums'][:, 1:]-components['target_expected_premiums'][:, 1:]))
    np.testing.assert_allclose(result[:, 1:], np.cumsum(increments, axis=1), atol=2e-13, rtol=0.)
    np.testing.assert_array_equal(components['target_actual_premiums'][:, -1], 0.)
    np.testing.assert_array_equal(components['target_expected_premiums'][:, -1], 0.)
    assert diagnostic['nodal_european_grid_points'] == 5
    assert diagnostic['nodal_european_grid_matches_physical_grid'] and diagnostic['actual_expected_primitives_saved']
    other, _ = bellman_target_martingales(proxy, selected, models, trace, batch_size=2)
    np.testing.assert_allclose(result, other, atol=2e-13, rtol=0.)


@pytest.mark.parametrize('seed', [None, True, -1, 2.1])
def test_target_seed_is_recorded_nonnegative_integer(control_case, seed):
    proxy, selected, models, trace = control_case
    with pytest.raises(ValueError, match='nonnegative'):
        bellman_target_martingales(proxy, selected, models, {**trace, 'seed': seed})


def test_selected_base_model_identity_is_explicitly_guarded(control_case):
    proxy, selected, models, trace = control_case
    different = replace(proxy, premium_cap=.6)
    with pytest.raises(ValueError, match='matching frozen'):
        bellman_target_martingales(proxy, replace(selected, base_policy=different), models, trace)
