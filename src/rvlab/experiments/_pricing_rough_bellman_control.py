"""Exact-centered stock-spline controls for a frozen Bellman exercise rule.

Only the control proxy is fitted. The GH16 exercise operator, physical law,
cash assumptions and acceptance tolerances remain fixed. Fitted continuation
corrections are bounded functions; neither states nor regression rows are
clipped or removed. The actual control is its explicitly declared stock spline.
"""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral

import numpy as np
import pandas as pd

from ._pricing_rough_exercise_block import exercise_block_law
from ._pricing_rough_local_control import _linear_coefficients, _state_reliability
from ._pricing_rough_node_refinement_fast import (
    interpolate_nodes, polynomial_node_values, refined_stock_knots, signed_shared_expectations)
from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_simulation import _time_grid
from rvlab.pricing.rough_stopping_fourier import _affine_values, fourier_policy_fingerprint
from rvlab.pricing.rough_tensor_premium import fit_tensor_premium


TARGETS = ("naive old-q grid8 constant upper", "old-q grid8 endpoint-zero",
    "Bellman-q grid4 endpoint-zero", "Bellman-q grid8 endpoint-zero")
CORRECTION_BOUND = .05
RELIABILITY_SCALE = 3.
RIDGE = 1e-3


class BellmanControlError(FloatingPointError):
    """A declared proxy fit or exact-centered control failed numerically."""


def _label_arrays(arrays, dates, factors):
    stock = np.asarray(arrays["stocks"])
    memory = np.asarray(arrays["factors"])
    mask = np.asarray(arrays["bellman_computed_mask"])
    continuation = np.asarray(arrays["bellman_continuations"])
    baseline = np.asarray(arrays["baseline_continuation"])
    if (stock.ndim != 3 or stock.shape[1:] != (dates, 1) or memory.shape != (len(stock), dates, factors)
            or mask.shape != (len(stock), dates) or mask.dtype.kind != "b"
            or continuation.shape != (len(stock), dates, 3) or baseline.shape != (len(stock), dates)
            or np.any(stock <= 0) or not all(np.isfinite(v).all() for v in (stock, memory, baseline))
            or not np.isfinite(continuation[mask]).all() or not np.isnan(continuation[~mask]).all()
            or np.any(mask[:, [0, dates-1]])):
        raise ValueError("Complete spent Bellman labels, full states and explicit computed mask required")
    return stock[:, :, 0], memory, mask, continuation[:, :, 1]-baseline


def fit_bellman_correction(proxy, fit_arrays, validation_arrays):
    """One fixed degree1 ridge fit per date, preserving every ITM label.

    The fixed source streams have become spent training/diagnostic evidence;
    they must never re-enter fresh control selection or final acceptance.
    """
    dates = len(proxy.continuation_times); count = len(proxy.kernel.nodes)
    train = _label_arrays(fit_arrays, dates, count)
    validation = _label_arrays(validation_arrays, dates, count)
    models = [None]*dates; rows = []
    for j in range(1, dates-1):
        s, u, mask, delta = train; active = mask[:, j]
        if np.sum(active) < 2:
            raise BellmanControlError(f"Date{j} has fewer than two declared training labels")
        model = fit_tensor_premium(s[active, j], u[active, j], delta[active, j], proxy.strike,
            proxy.config, degree=1, ridge=RIDGE, factor_clip=RELIABILITY_SCALE)
        models[j] = model
        for role, data in (("spent fit1077", train), ("spent diagnostic1078", validation)):
            ds, du, dm, dd = data; used = dm[:, j]; observed = dd[used, j]
            predicted = model.value(ds[used, j], du[used, j], proxy.strike)
            if not np.isfinite(predicted).all():
                raise BellmanControlError(f"Date{j} correction fit is nonfinite")
            difference = predicted-observed
            rows.append({"exercise_index": j, "role": role, "rows": len(observed), "status": "completed",
                "label_mean": float(np.mean(observed)), "label_minimum": float(np.min(observed)),
                "label_maximum": float(np.max(observed)), "label_above_proxy_bound_fraction": float(np.mean(abs(observed) > CORRECTION_BOUND)),
                "raw_fit_rmse": float(np.sqrt(np.mean(difference*difference))),
                "raw_fit_mae": float(np.mean(abs(difference))), "degree": 1, "ridge": RIDGE,
                "correction_bound": CORRECTION_BOUND, "factor_clip": RELIABILITY_SCALE,
                "rank": model.rank, "condition": model.condition,
                "regression_rows_clipped_or_removed": False, "used_for_fresh_acceptance": False})
    return tuple(models), pd.DataFrame(rows)


def target_nodal_premiums(proxy, selected, models, j, factors, *, fine_nodes):
    """Four actual proxy functions on one shared grid; cap only q.

    The reliability factor multiplies the learned continuation correction at
    each RIGHT factor state before the obstacle maximum. It does not damp
    intrinsic exercise value or multiply a precomputed conditional mean.
    """
    factors = np.asarray(factors, float); fine_nodes = np.asarray(fine_nodes, float)
    if (not isinstance(j, Integral) or isinstance(j, bool) or not 0 <= j < len(models)
            or factors.shape[-1] != len(proxy.kernel.nodes) or not np.isfinite(factors).all()
            or fine_nodes.ndim != 1 or fine_nodes[0] != 0 or np.any(np.diff(fine_nodes) <= 0)):
        raise ValueError("Declare a valid date, full factors and increasing zero-start stock nodes")
    if j == len(models)-1:
        return np.zeros(factors.shape[:-1]+(4, len(fine_nodes)))
    if models[j] is None:
        raise ValueError("Every noninitial, nonterminal target date requires its frozen correction model")
    euro = replace(proxy.value_premiums[j], stock_nodes=fine_nodes).european_values(factors)
    old = np.clip(polynomial_node_values(selected.continuations[j], fine_nodes, factors), 0., selected.premium_cap)
    raw_delta = polynomial_node_values(models[j], fine_nodes, factors)
    reliability = _state_reliability(models[j], factors, RELIABILITY_SCALE)
    corrected = np.clip(old+reliability[..., None]*np.clip(raw_delta, -CORRECTION_BOUND, CORRECTION_BOUND),
        0., selected.premium_cap)
    old_zero = old.copy(); old_zero[..., [0, -1]] = 0.
    corrected[..., [0, -1]] = 0.
    payoff = np.maximum((1 if proxy.is_call else -1)*(proxy.strike*fine_nodes-proxy.strike), 0.)
    naive = np.maximum.reduce((old, np.broadcast_to(payoff-euro, old.shape), np.zeros_like(old)))
    uncorrected = np.maximum.reduce((old_zero, payoff-euro, np.zeros_like(old)))
    corrected = np.maximum.reduce((corrected, payoff-euro, np.zeros_like(old)))
    coarse_nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 4)]
    positions = np.searchsorted(fine_nodes, coarse_nodes)
    if not np.array_equal(fine_nodes[positions], coarse_nodes):
        raise ValueError("The declared grid4 nodes must be exact members of grid8")
    coarse = interpolate_nodes(coarse_nodes, corrected[..., positions], fine_nodes)
    result = np.stack((naive, uncorrected, coarse, corrected), axis=-2)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise BellmanControlError("A Bellman target premium is negative or nonfinite")
    return result


def bellman_target_martingales(proxy, selected, models, trace, *, forbidden_seeds=(),
        require_independent=True, batch_size=64):
    """Exact finite-mixture centering for all four declared spline processes."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("Positive integer target-control batch required")
    if not isinstance(require_independent, (bool, np.bool_)):
        raise TypeError("require_independent must be boolean")
    times = np.asarray(trace["times"]); grid = np.asarray(trace["actual_time_grid"])
    stock, factors, euro = (np.asarray(trace[k]) for k in ("stocks", "factors", "european_proxy"))
    forbidden = set(selected.training_seeds)|set(forbidden_seeds)
    if (require_independent and (trace.get("seed") is None or trace["seed"] in forbidden)):
        raise ValueError("Target control paths must be independent of proxy fitting and prior selection")
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
        raise ValueError("Matching frozen full-state no-cash trace and physical law required")
    nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([proxy.strike]),
        np.array([proxy.maturity]), np.array([proxy.is_call]), grid, proxy.control_config)
    result = np.zeros(stock.shape[:2]+(4,)); maxmass = 0.
    for j, (left, right) in enumerate(zip(times[:-1], times[1:])):
        a, b = np.searchsorted(grid, [left, right])
        if grid[a] != left or grid[b] != right or b-a != 2:
            raise ValueError("This target study retains exactly two original physical steps per exercise date")
        for begin in range(0, len(stock), batch_size):
            end = min(begin+batch_size, len(stock)); old = factors[begin:end, j]; s = stock[begin:end, j, 0]
            law = exercise_block_law(proxy, old, times=grid[a:b+1]); new = law["new_factors"]
            p = law["probabilities"]; maxmass = max(maxmass, float(np.max(abs(p.sum(axis=1)-1))))
            values = target_nodal_premiums(proxy, selected, models, j+1, new, fine_nodes=nodes)
            coefficients = _linear_coefficients(nodes, values)
            mean = np.log(s)[:, None, None]+law["log_stock_shifts"]; clock = law["orthogonal"]
            premium = signed_shared_expectations(mean, clock, coefficients, strike=proxy.strike, knots=nodes[1:])
            expected_premium = np.sum(premium.mean(axis=-1)*p[..., None], axis=1)
            expected_euro = np.sum(_affine_values(affine, float(right), new, mean[..., None], clock[..., None])[..., 0]
                .mean(axis=2)*p, axis=1)
            actual_values = target_nodal_premiums(proxy, selected, models, j+1, factors[begin:end, j+1], fine_nodes=nodes)
            actual_coefficients = _linear_coefficients(nodes, actual_values)
            z = stock[begin:end, j+1, 0]/proxy.strike; index = np.searchsorted(nodes[1:], z, side="right")
            positions = np.broadcast_to(index[:, None, None, None], actual_coefficients.shape[:2]+(1, 1))
            c = np.take_along_axis(actual_coefficients, positions, axis=-2)[..., 0, :]
            actual_premium = c[..., 0]+c[..., 1]*z[:, None]
            increment = ((euro[begin:end, j+1, 0]-expected_euro)[:, None]
                +float(proxy.market.discount_curve.discount(right))*(actual_premium-expected_premium))
            result[begin:end, j+1] = result[begin:end, j]+increment
    if not np.isfinite(result).all():
        raise BellmanControlError("The target Snell martingales are nonfinite")
    return result, {"targets": list(TARGETS), "max_probability_error": maxmass, "fine_positive_nodes": len(nodes)-1,
        "physical_increments_per_interval": 2, "only_q_capped": True, "snell_premium_capped": False,
        "right_factor_reliability_inside_expectation": True, "all_paths_retained": True,
        "stock_interpolation_is_actual_control": True, "conditional_mixture_expectation_exact": True,
        "new_exercise_policy": False, "nested_Bellman_integration_in_centering": False,
        "naive_upper_endpoint_retained": True, "proxy_correction_bound": CORRECTION_BOUND,
        "no_state_or_regression_row_deletion": True}


__all__ = ["TARGETS", "BellmanControlError", "fit_bellman_correction", "target_nodal_premiums", "bellman_target_martingales"]
