"""One-dimensional backward equation, obstacle and discrete cash events."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import PchipInterpolator

from .types import BlackScholesParams, PricingError
from .vanilla_types import NumericalConvergenceError
from ._vanilla_common import build_grid, time_grid, zero_spot_put_value, derivative_coefficients


def deterministic_variance(params, t):
    if isinstance(params, BlackScholesParams):
        return params.volatility**2
    kappa = params.kappa
    value, start = params.v0, 0.
    knots = params.curve.knot_times if hasattr(params, "curve") else ()
    levels = params.curve.levels if hasattr(params, "curve") else (params.theta,)
    for end, theta in zip((*knots, float("inf")), levels):
        stop = min(t, end)
        value = theta + (value-theta)*np.exp(-kappa*(stop-start))
        if t <= end:
            return float(value)
        start = end
    return float(value)


def _operator(spots, rate, variance):
    first, second = derivative_coefficients(spots)
    interior = spots[1:-1]
    diffusion, drift = .5*variance*interior**2, rate*interior
    lower = diffusion*second[0] + drift*first[0]
    diag = diffusion*second[1] + drift*first[1] - rate
    upper = diffusion*second[2] + drift*first[2]
    # A monotone upwind derivative is used only where convection dominates.
    bad = (lower < 0) | (upper < 0)
    left, right = np.diff(spots)[:-1], np.diff(spots)[1:]
    fl = np.where(drift < 0, -1/left, 0.)
    fu = np.where(drift >= 0, 1/right, 0.)
    fd = -fl-fu
    lower[bad] = (diffusion*second[0] + drift*fl)[bad]
    diag[bad] = (diffusion*second[1] + drift*fd-rate)[bad]
    upper[bad] = (diffusion*second[2] + drift*fu)[bad]
    boundary = lower[0]
    lower[0] = 0.
    ratio = (spots[-1]-spots[-2])/(spots[-2]-spots[-3])
    lower[-1] -= ratio*upper[-1]
    diag[-1] += (1+ratio)*upper[-1]
    upper[-1] = 0.
    return lower, diag, upper, boundary


def _multiply(lower, diag, upper, values):
    result = diag[:, None]*values
    result[1:] += lower[1:, None]*values[:-1]
    result[:-1] += upper[:-1, None]*values[1:]
    return result


def _band(lower, diag, upper):
    band = np.zeros((3, len(diag)))
    band[0, 1:], band[1], band[2, :-1] = upper[:-1], diag, lower[1:]
    return band


def _lcp(lower, diag, upper, rhs, obstacle, start, config):
    values = np.maximum(start, obstacle).copy()
    solves = 0
    for column in range(values.shape[1]):
        x, b, g = values[:, column], rhs[:, column], obstacle[:, column]
        for iteration in range(config.max_lcp_iterations):
            residual = _multiply(lower, diag, upper, x[:, None])[:, 0]-b
            gap = x-g
            tolerance = config.lcp_tolerance*max(1., float(np.max(np.abs(b))), float(np.max(np.abs(g))))
            if np.max(np.abs(np.minimum(residual, gap))) <= tolerance:
                break
            active = gap <= residual
            l, d, u, y = lower.copy(), diag.copy(), upper.copy(), b.copy()
            l[active], d[active], u[active], y[active] = 0., 1., 0., g[active]
            x = solve_banded((1, 1), _band(l, d, u), y, check_finite=False)
            solves += 1
        else:
            residual = _multiply(lower, diag, upper, x[:, None])[:, 0]-b
            if np.max(np.abs(np.minimum(residual, x-g))) > tolerance:
                raise NumericalConvergenceError(f"American BS policy iteration exhausted {config.max_lcp_iterations} iterations")
        values[:, column] = x
    return values, solves


def extract_values(spots, values, spot):
    interpolation = PchipInterpolator(spots, values, axis=0)
    prices = interpolation(spot)
    deltas = interpolation.derivative()(spot)
    # Left/right secants expose unresolved local kinks, instead of claiming
    # finite-difference smoothness around a scheduled event or exercise edge.
    i = int(np.searchsorted(spots, spot))
    if i < len(spots) and abs(spots[i]-spot) < 1e-10*max(1, spot):
        left = (values[i]-values[i-1])/(spots[i]-spots[i-1]) if i else deltas
        right = (values[i+1]-values[i])/(spots[i+1]-spots[i]) if i+1 < len(spots) else deltas
    else:
        left = right = deltas
    return prices, deltas, left, right


def solve_bs(params, market, strike, maturity, is_call, config, grid=None, *, exercise_style="american"):
    grid = build_grid(market, strike, config, variance=False) if grid is None else grid
    spots = grid.spot
    payoff = np.maximum(np.where(is_call, 1., -1.)[None, :]*(spots[:, None]-strike[None, :]), 0.)
    values = payoff.copy()
    clock = time_grid(params, market, maturity, config)
    boundaries, events = [], []
    solves = 0
    damping = True
    cash = {event.time: event.amount for event in market.dividends.events if event.time <= maturity}

    def boundary(t):
        return np.where(is_call, 0., strike*zero_spot_put_value(market, t, maturity, exercise_style))

    def enforce_edges(t):
        values[0] = boundary(t)
        ratio = (spots[-1]-spots[-2])/(spots[-2]-spots[-3])
        values[-1] = (1+ratio)*values[-2]-ratio*values[-3]

    def exercise_record(t):
        if exercise_style != "american":
            return
        for column, (k, call) in enumerate(zip(strike, is_call)):
            active = (payoff[:, column] > 0) & (np.abs(values[:, column]-payoff[:, column]) <= 1e-7*max(1., k))
            # An imposed domain boundary alone is not a resolved stopping
            # frontier. Match the 2D engine's interior-only reporting.
            active[0] = active[-1] = False
            eligible = spots[active]
            edge = (float(np.min(eligible)) if call else float(np.max(eligible))) if len(eligible) else float("nan")
            boundaries.append({"time": float(t), "strike": float(k), "contract_index": column,
                               "is_call": bool(call), "spot_boundary": edge})

    def dividend(t):
        nonlocal values, damping
        amount = cash.get(t)
        if amount is None or amount == 0:
            return
        post_spot = max(market.spot-amount, 0.)
        post = extract_values(spots, values, post_spot)
        shifted = np.maximum(spots-amount, 0.)
        remapped = np.column_stack([np.interp(shifted, spots, values[:, i]) for i in range(len(strike))])
        values = np.maximum(remapped, payoff) if exercise_style == "american" else remapped
        pre = extract_values(spots, values, market.spot)
        events.append({"time": float(t), "reference_spot": market.spot,
                       "post_spot": post_spot, "pre_delta": pre[1].tolist(), "post_delta": post[1].tolist()})
        damping = True
        exercise_record(t)

    def advance(right, left, theta):
        nonlocal values, solves
        dt = right-left
        rate = market.discount_curve.rate((right+left)/2)
        old = _operator(spots, rate, deterministic_variance(params, right))
        new = _operator(spots, rate, deterministic_variance(params, left))
        rhs = values[1:-1] + (1-theta)*dt*_multiply(*old[:3], values[1:-1])
        rhs[0] += dt*((1-theta)*old[3]*boundary(right)+theta*new[3]*boundary(left))
        lower, diag, upper = -theta*dt*new[0], 1-theta*dt*new[1], -theta*dt*new[2]
        if exercise_style == "american":
            values[1:-1], count = _lcp(lower, diag, upper, rhs, payoff[1:-1], values[1:-1], config)
            solves += count
        else:
            values[1:-1] = solve_banded((1, 1), _band(lower, diag, upper), rhs, check_finite=False)
            solves += 1
        enforce_edges(left)
        if not np.isfinite(values).all():
            raise PricingError("BS finite-difference solve produced nonfinite prices")

    dividend(float(maturity))
    for i in range(len(clock)-1, 0, -1):
        right, left = float(clock[i]), float(clock[i-1])
        if damping:
            middle = .5*(right+left)
            advance(right, middle, 1.)
            advance(middle, left, 1.)
            damping = False
        else:
            advance(right, left, .5)
        dividend(left)
        exercise_record(left)
    prices, deltas, dl, dr = extract_values(spots, values, market.spot)
    return {"prices": prices, "deltas": deltas, "delta_left": dl, "delta_right": dr,
            "exercise_boundaries": boundaries, "grid": grid,
            "diagnostics": {"backend": "bs_cn_policy_iteration", "linear_solves": solves,
                            "time_intervals": len(clock)-1, "dividend_event_sensitivities": events}}


__all__ = ["solve_bs", "deterministic_variance"]
