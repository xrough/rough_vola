"""Physical-grid-aligned successor to the preserved mixed-grid control trial.

Both sides of each Snell maximum and its conditional expectation use the same
original129-point affine surrogate. The previous helper remains byte-for-byte
preserved as a valid centered, but differently specified, exploratory control.
"""
from dataclasses import replace
from numbers import Integral

import numpy as np

from ._pricing_rough_bellman_control import (
    BellmanControlError, fit_bellman_correction, target_nodal_premiums as _old_nodal)
from ._pricing_rough_exercise_block import exercise_block_law
from ._pricing_rough_local_control import _linear_coefficients
from ._pricing_rough_node_refinement_fast import refined_stock_knots, signed_shared_expectations
from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_simulation import _time_grid
from rvlab.pricing.rough_stopping_fourier import _affine_values, fourier_policy_fingerprint


TARGETS = ("naive old-q grid8 constant upper", "old-q grid8 q-endpoint-zero",
    "Bellman-q grid4 q-endpoint-zero", "Bellman-q grid8 q-endpoint-zero")


def aligned_node_proxy(proxy, affine):
    """Precompute every date's physical-grid European coefficients once."""
    models = []
    for j, time in enumerate(proxy.continuation_times):
        exponent, factor = affine.coefficients[(proxy.maturity, float(time))]
        coefficient, constant = affine._remaining_integral(float(time), proxy.maturity)
        models.append(replace(proxy.value_premiums[j], affine_exponent=exponent, affine_factor=factor,
            remaining_integral_coefficient=coefficient, remaining_integral_constant=float(constant)))
    return replace(proxy, value_premiums=tuple(models))


def target_nodal_premiums(proxy, selected, models, j, factors, *, fine_nodes, node_proxy=None):
    if node_proxy is None:
        grid = _time_grid(proxy.params, proxy.market, proxy.continuation_times, proxy.config.steps)
        affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([proxy.strike]),
            np.array([proxy.maturity]), np.array([proxy.is_call]), grid, proxy.control_config)
        node_proxy = aligned_node_proxy(proxy, affine)
    return _old_nodal(node_proxy, selected, models, j, factors, fine_nodes=fine_nodes)


def bellman_target_martingales(proxy, selected, models, trace, *, forbidden_seeds=(),
        require_independent=True, batch_size=64, return_components=False):
    """Four exact-centered processes, optionally retaining primitive audits."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("Positive integer target-control batch required")
    if not isinstance(require_independent, (bool, np.bool_)) or not isinstance(return_components, (bool, np.bool_)):
        raise TypeError("Explicit independence and component-audit flags required")
    seed = trace.get("seed")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
            or fourier_policy_fingerprint(selected.base_policy) != fourier_policy_fingerprint(proxy)):
        raise ValueError("Recorded nonnegative seed and matching frozen policy required")
    if require_independent and seed in set(selected.training_seeds)|set(forbidden_seeds):
        raise ValueError("Target paths must be independent of all proxy fitting and prior selection")
    times = np.asarray(trace["times"]); grid = np.asarray(trace["actual_time_grid"])
    stock, factors, euro = (np.asarray(trace[k]) for k in ("stocks", "factors", "european_proxy"))
    if (not np.array_equal(times, proxy.continuation_times) or not np.array_equal(times, selected.times)
            or not np.array_equal(grid, _time_grid(proxy.params, proxy.market, times, proxy.config.steps))
            or trace.get("policy_fingerprint") != fourier_policy_fingerprint(proxy)
            or trace.get("kernel_fingerprint") != proxy.kernel.fingerprint or trace.get("strike") != proxy.strike
            or not np.array_equal(trace.get("initial_spots"), [proxy.market.spot])
            or not np.array_equal(trace.get("discounts"), proxy.market.discount_curve.discount(times))
            or proxy.market.dividends.events or stock.ndim != 3 or stock.shape[-1] != 1
            or stock.shape[1] != len(times) or euro.shape != stock.shape
            or factors.shape != (len(stock), len(times), len(proxy.kernel.nodes))
            or not all(np.isfinite(v).all() for v in (stock, factors, euro)) or np.any(stock <= 0)):
        raise ValueError("Matching frozen full-state trace and original physical law required")
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([proxy.strike]),
        np.array([proxy.maturity]), np.array([proxy.is_call]), grid, proxy.control_config)
    node_proxy = aligned_node_proxy(proxy, affine)
    result = np.zeros(stock.shape[:2]+(4,)); maxmass = 0.
    expected_euros = np.zeros(stock.shape[:2]); actual_premiums = np.zeros_like(result); expected_premiums = np.zeros_like(result)
    for j, (left, right) in enumerate(zip(times[:-1], times[1:])):
        a, b = np.searchsorted(grid, [left, right])
        if grid[a] != left or grid[b] != right or b-a != 2:
            raise ValueError("Exactly two original physical steps per exercise interval required")
        for begin in range(0, len(stock), batch_size):
            end = min(begin+batch_size, len(stock)); old = factors[begin:end, j]; s = stock[begin:end, j, 0]
            law = exercise_block_law(proxy, old, times=grid[a:b+1]); new = law["new_factors"]
            p = law["probabilities"]; maxmass = max(maxmass, float(np.max(abs(p.sum(axis=1)-1))))
            values = target_nodal_premiums(proxy, selected, models, j+1, new, fine_nodes=nodes, node_proxy=node_proxy)
            coefficients = _linear_coefficients(nodes, values)
            mean = np.log(s)[:, None, None]+law["log_stock_shifts"]; clock = law["orthogonal"]
            premium = signed_shared_expectations(mean, clock, coefficients, strike=proxy.strike, knots=nodes[1:])
            expected_premium = np.sum(premium.mean(axis=-1)*p[..., None], axis=1)
            expected_euro = np.sum(_affine_values(affine, float(right), new, mean[..., None], clock[..., None])[..., 0]
                .mean(axis=2)*p, axis=1)
            actual_values = target_nodal_premiums(proxy, selected, models, j+1, factors[begin:end, j+1], fine_nodes=nodes, node_proxy=node_proxy)
            actual_coefficients = _linear_coefficients(nodes, actual_values)
            z = stock[begin:end, j+1, 0]/proxy.strike; index = np.searchsorted(nodes[1:], z, side="right")
            positions = np.broadcast_to(index[:, None, None, None], actual_coefficients.shape[:2]+(1, 1))
            c = np.take_along_axis(actual_coefficients, positions, axis=-2)[..., 0, :]
            actual_premium = c[..., 0]+c[..., 1]*z[:, None]
            increment = ((euro[begin:end, j+1, 0]-expected_euro)[:, None]
                +float(proxy.market.discount_curve.discount(right))*(actual_premium-expected_premium))
            result[begin:end, j+1] = result[begin:end, j]+increment
            expected_euros[begin:end, j+1] = expected_euro
            actual_premiums[begin:end, j+1] = actual_premium
            expected_premiums[begin:end, j+1] = expected_premium
    if not all(np.isfinite(v).all() for v in (result, expected_euros, actual_premiums, expected_premiums)):
        raise BellmanControlError("Aligned target martingales or primitive audit arrays are nonfinite")
    diagnostic = {"targets": list(TARGETS), "max_probability_error": maxmass, "fine_positive_nodes": len(nodes)-1,
        "physical_increments_per_interval": 2, "only_q_capped": True, "snell_premium_capped": False,
        "right_factor_reliability_inside_expectation": True, "all_paths_retained": True,
        "stock_interpolation_is_actual_control": True, "conditional_mixture_expectation_exact": True,
        "new_exercise_policy": False, "nested_Bellman_integration_in_centering": False,
        "naive_upper_endpoint_retained": True, "proxy_correction_bound": .05,
        "no_state_or_regression_row_deletion": True, "nodal_european_grid_matches_physical_grid": True,
        "nodal_european_grid_points": len(grid), "actual_expected_primitives_saved": return_components}
    if return_components:
        return result, diagnostic, {"target_expected_european": expected_euros,
            "target_actual_premiums": actual_premiums, "target_expected_premiums": expected_premiums}
    return result, diagnostic


__all__ = ["TARGETS", "fit_bellman_correction", "aligned_node_proxy", "target_nodal_premiums", "bellman_target_martingales"]
