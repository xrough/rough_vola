"""Finite-difference vanilla Heston pricing with MCS--IT time stepping.

The mixed derivative is explicit and the two directional operators are
implicit.  Each factorization is shared by all strikes in a maturity batch.
The degenerate variance-zero equation is included in the system, including
when the Feller condition is violated.  No Feller restriction or price
clipping is imposed.

The ADI and obstacle splitting follow Haentjens and in 't Hout (2013),
https://arxiv.org/abs/1309.0110, equation (4.3), with theta=1/3.  Directional
drifts use centered differences where monotone and upwinding otherwise.
Two fully implicit Euler half steps damp each payoff/dividend discontinuity.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import splu

from .types import PricingError
from ._vanilla_common import build_grid, time_grid, zero_spot_put_value


def _derivatives(nodes):
    """Second-order nonuniform centered first and second derivatives."""
    nodes = np.asarray(nodes, float)
    n = len(nodes)
    first = sparse.lil_matrix((n, n))
    second = sparse.lil_matrix((n, n))
    left, right = np.diff(nodes)[:-1], np.diff(nodes)[1:]
    rows = np.arange(1, n - 1)
    first[rows, rows - 1] = -right / (left * (left + right))
    first[rows, rows] = (right - left) / (left * right)
    first[rows, rows + 1] = left / (right * (left + right))
    second[rows, rows - 1] = 2 / (left * (left + right))
    second[rows, rows] = -2 / (left * right)
    second[rows, rows + 1] = 2 / (right * (left + right))
    return first.tocsr(), second.tocsr()


def _direction(nodes, drift, diffusion, killing=0.0, *, lower="fixed", upper="fixed"):
    """Monotone-centered drift/diffusion stencil and degenerate boundaries."""
    n = len(nodes)
    drift = np.broadcast_to(drift, (n,))
    diffusion = np.broadcast_to(diffusion, (n,))
    d1, d2 = _derivatives(nodes)
    matrix = (sparse.diags(drift) @ d1 + sparse.diags(diffusion) @ d2
              - killing * sparse.eye(n)).tolil()
    # Preserve second order whenever its off-diagonal entries are nonnegative.
    # At a drift-dominated node an upwind generator retains positivity.
    for i in range(1, n - 1):
        hm, hp = nodes[i] - nodes[i - 1], nodes[i + 1] - nodes[i]
        if matrix[i, i - 1] < 0 or matrix[i, i + 1] < 0:
            a = 2 * diffusion[i] / (hm * (hm + hp))
            c = 2 * diffusion[i] / (hp * (hm + hp))
            if drift[i] >= 0:
                c += drift[i] / hp
            else:
                a -= drift[i] / hm
            matrix[i, i - 1], matrix[i, i], matrix[i, i + 1] = a, -a - c - killing, c
    if lower == "degenerate":
        # At v=0 the PDE is first order in v, rather than a prescribed value.
        h = nodes[1] - nodes[0]
        matrix[0, :] = 0
        matrix[0, 0], matrix[0, 1] = -drift[0] / h - killing, drift[0] / h
    else:
        matrix[0, :] = 0
    if upper == "neumann":
        # A reflected ghost point imposes u_v(Vmax)=0.  Drift and mixed
        # derivatives vanish there; the second derivative remains present.
        h = nodes[-1] - nodes[-2]
        weight = 2 * diffusion[-1] / h**2
        matrix[-1, :] = 0
        matrix[-1, -2], matrix[-1, -1] = weight, -weight - killing
    elif upper == "linear":
        # The known asymptotic slope enters as a source outside the matrix.
        matrix[-1, :] = 0
        matrix[-1, -1] = -killing
    else:
        matrix[-1, :] = 0
    return matrix.tocsr()


def _operators(params, spot, variance, rate, mean_level):
    ns, nv = len(spot), len(variance)
    s1, _ = _derivatives(spot)
    v1, _ = _derivatives(variance)
    # Mixed terms vanish at s=0, the asymptotically linear upper-spot
    # boundary, v=0, and the upper-variance Neumann boundary.
    mixed = sparse.diags((params.rho * params.vol_of_vol
                         * variance[:, None] * spot[None, :]).ravel()) @ sparse.kron(v1, s1)
    spot_operator = sparse.block_diag([
        _direction(spot, rate * spot, 0.5 * v * spot**2,
                   0.5 * rate, upper="linear")
        for v in variance
    ], format="csr")
    variance_operator = _direction(
        variance, params.kappa * (mean_level - variance),
        0.5 * params.vol_of_vol**2 * variance, 0.5 * rate,
        lower="degenerate", upper="neumann")
    variance_operator = sparse.kron(variance_operator, sparse.eye(ns), format="csr")
    # Prescribed stock-zero values must not undergo variance diffusion or
    # killing.  Their calendar-dependent values are pinned in each stage.
    free = np.ones(ns * nv)
    free[::ns] = 0
    variance_operator = sparse.diags(free) @ variance_operator
    return mixed.tocsr(), spot_operator, variance_operator.tocsr()


def _theta_at(params, time):
    if hasattr(params, "curve"):
        return float(params.curve(time))
    return float(params.theta)


def _terminal_values(spot, strikes, calls, payoff):
    """Cell-average only the payoff kink (the exercise obstacle stays exact)."""
    terminal = payoff.copy()
    for contract, (strike, call) in enumerate(zip(strikes, calls)):
        j = int(np.argmin(abs(spot - strike)))
        if j == 0 or j == len(spot) - 1:
            continue
        # A centered cell retains call-put linearity even when insertion of a
        # strike has made its two neighboring spacings unequal.
        half = 0.5 * min(spot[j] - spot[j - 1], spot[j + 1] - spot[j])
        left, right = spot[j] - half, spot[j] + half
        if left < strike < right:
            if call:
                integral = 0.5 * (max(right - strike, 0)**2 - max(left - strike, 0)**2)
            else:
                integral = 0.5 * (max(strike - left, 0)**2 - max(strike - right, 0)**2)
            terminal[j, contract] = integral / (right - left)
    return terminal


def _pin_zero(values, market, calendar_time, maturity, strikes, calls, style, ns):
    multiplier = zero_spot_put_value(market, calendar_time, maturity, style)
    values[::ns] = np.where(calls, 0.0, strikes * multiplier)
    return values


def _exercise_records(values, obstacle, spot, variance, v0, time, calls, tolerance, *, event=None):
    """Exercise boundary at the initial variance, avoiding zero-payoff tails."""
    surface = values.reshape(len(variance), len(spot), -1)
    if v0 <= variance[0]:
        line = surface[0]
    else:
        j = min(np.searchsorted(variance, v0), len(variance) - 1)
        weight = (v0 - variance[j - 1]) / (variance[j] - variance[j - 1])
        line = (1 - weight) * surface[j - 1] + weight * surface[j]
    payoff = obstacle[:len(spot)]
    records = []
    for k, call in enumerate(calls):
        active = ((payoff[:, k] > tolerance)
                  & (line[:, k] - payoff[:, k] <= tolerance))
        active[0] = active[-1] = False
        selected = spot[active]
        boundary = (float(selected.min() if call else selected.max())
                    if len(selected) else None)
        record = {"time": float(time), "contract_index": k,
                  "strike": float(spot[-1] - payoff[-1, k] if call else payoff[0, k]),
                  "variance": float(v0), "spot_boundary": boundary,
                  "is_call": bool(call)}
        if event is not None:
            record["event_side"] = event
        records.append(record)
    return records


def _interpolate(values, spot, variance, initial_spot, initial_variance):
    """Cubic interpolation in both state coordinates on a fixed mesh.

    In particular v0 is not inserted into the mesh as an optimizer candidate
    changes. Cubic variance interpolation avoids artificial kinks at the
    variance-grid nodes in the calibration objective.
    """
    surface = values.reshape(len(variance), len(spot), -1)
    if initial_variance <= variance[0]:
        line = surface[0]
    else:
        line = CubicSpline(variance, surface, axis=0)(initial_variance)
    spline = CubicSpline(spot, line, axis=0)
    prices, deltas = spline(initial_spot), spline(initial_spot, 1)
    right = min(max(np.searchsorted(spot, initial_spot, side="right"), 1), len(spot) - 1)
    left = min(max(np.searchsorted(spot, initial_spot, side="left") - 1, 0), len(spot) - 2)
    left_delta = (line[left + 1] - line[left]) / (spot[left + 1] - spot[left])
    right_delta = (line[right] - line[right - 1]) / (spot[right] - spot[right - 1])
    return prices, deltas, left_delta, right_delta


def solve_heston(params, market, strike, maturity, is_call, config, grid=None,
                 *, exercise_style="american"):
    """Price a maturity batch, with fixed inputs and a reusable spatial grid.

    ``grid`` is structural: optimization callers must freeze it for the whole
    attempt.  Changing model parameters cannot silently expand its domains.
    Dividend events occur strictly after valuation.  The event map uses
    ``S_after=max(S_before-D,0)`` and American exercise immediately before it.
    """
    if exercise_style not in ("american", "european"):
        raise ValueError("exercise_style must be 'american' or 'european'")
    strikes = np.asarray(strike, float).reshape(-1)
    calls = np.broadcast_to(np.asarray(is_call, bool), strikes.shape)
    if grid is None:
        grid = build_grid(market, strikes, config, variance=True)
    spot, variance = np.asarray(grid.spot), np.asarray(grid.variance)
    if (variance.ndim != 1 or len(variance) < 3 or variance[0] != 0
            or params.v0 > variance[-1]):
        raise PricingError("Heston variance grid must start at zero and contain v0")
    if maturity <= 0:
        payoff = np.maximum(np.where(calls, market.spot - strikes, strikes - market.spot), 0)
        delta = np.where(calls, (market.spot > strikes).astype(float),
                         -(market.spot < strikes).astype(float))
        return {"prices": payoff, "deltas": delta, "delta_left": delta.copy(),
                "delta_right": delta.copy(), "exercise_boundaries": [],
                "diagnostics": {"method": "MCS-IT", "time_steps": 0, "linear_solves": 0,
                                "factorizations": 0}, "grid": grid}
    ns, nv, nc = len(spot), len(variance), len(strikes)
    payoff = np.maximum(np.where(calls[None], spot[:, None] - strikes[None],
                                 strikes[None] - spot[:, None]), 0)
    obstacle = np.tile(payoff, (nv, 1))
    values = np.tile(_terminal_values(spot, strikes, calls, payoff), (nv, 1))
    multiplier = np.zeros_like(values)
    identity = sparse.eye(ns * nv, format="csc")
    american = exercise_style == "american"
    times = np.asarray(time_grid(params, market, maturity, config), float)
    events = {float(event.time): float(event.amount)
              for event in market.dividends.events if 0 < event.time <= maturity and event.amount > 0}
    operator_cache, factor_cache = {}, {}
    solves = damped_steps = 0
    records, event_deltas = [], []
    damping = True

    def get_operators(rate, mean):
        key = (float(rate), float(mean))
        if key not in operator_cache:
            operator_cache[key] = _operators(params, spot, variance, rate, mean)
            if any(not np.isfinite(operator.data).all()
                   for operator in operator_cache[key]):
                raise PricingError("Heston finite-difference operator contains nonfinite coefficients")
        return key, operator_cache[key]

    def factor(key, operator, dt, direction):
        cache_key = (key, float(f"{dt:.13g}"), direction)
        if cache_key not in factor_cache:
            try:
                factor_cache[cache_key] = splu(identity - dt * operator)
            except RuntimeError as exc:
                raise PricingError("Heston finite-difference factorization failed") from exc
        return factor_cache[cache_key]

    def pin(array, calendar):
        return _pin_zero(array, market, calendar, maturity, strikes, calls,
                         exercise_style, ns)

    def obstacle_step(candidate, old_multiplier, dt, calendar):
        if american:
            value = np.maximum(candidate - dt * old_multiplier, obstacle)
            lagrange = np.maximum(0.0, old_multiplier + (obstacle - candidate) / dt)
        else:
            value, lagrange = candidate, old_multiplier
        pin(value, calendar)
        lagrange[::ns] = 0
        return value, lagrange

    def apply_dividend(value, event_time):
        post_spot = max(market.spot - events[event_time], 0.)
        after = _interpolate(value, spot, variance, post_spot, params.v0)
        if american:
            records.extend(_exercise_records(value, obstacle, spot, variance, params.v0,
                                             event_time, calls, config.lcp_tolerance, event="after"))
        amount = events[event_time]
        mapped_spot = np.maximum(spot - amount, 0)
        surface = value.reshape(nv, ns, nc)
        # Piecewise-linear interpolation is monotone at the cash-dividend
        # jump and does not introduce overshoots into the stopping value.
        j = np.clip(np.searchsorted(spot, mapped_spot, side="right") - 1, 0, ns - 2)
        w = (mapped_spot - spot[j]) / (spot[j + 1] - spot[j])
        value = ((1 - w)[None, :, None] * surface[:, j, :]
                 + w[None, :, None] * surface[:, j + 1, :]).reshape(nv * ns, nc)
        if american:
            value = np.maximum(value, obstacle)
        pin(value, event_time)
        before = _interpolate(value, spot, variance, market.spot, params.v0)
        event_deltas.append({"time": float(event_time), "cash_dividend": amount,
                             "reference_spot": market.spot, "post_spot": post_spot,
                             "post_delta": after[1].tolist(),
                             "pre_delta": before[1].tolist()})
        return value

    # The convention for a dividend exactly at expiry is an ex-dividend
    # terminal payoff, preceded by the American holder's exercise decision.
    # time_grid inserts the exact event values. An isclose matcher would apply
    # one dividend twice if a nearby linspace node differs only by roundoff.
    terminal_event = maturity if maturity in events else None
    if terminal_event is not None:
        values = apply_dividend(values, terminal_event)
        if american:
            records.extend(_exercise_records(values, obstacle, spot, variance, params.v0,
                                             maturity, calls, config.lcp_tolerance, event="before"))

    for high, low in zip(times[:0:-1], times[-2::-1]):
        dt = float(high - low)
        middle = 0.5 * (high + low)
        rate = float(market.discount_curve.rate(middle))
        mean = _theta_at(params, middle)
        key, (a0, a1, a2) = get_operators(rate, mean)
        total = a0 + a1 + a2
        source = np.zeros_like(values)
        source[ns - 1::ns] = rate * spot[-1] * calls
        if damping:
            step = 0.5 * dt
            solver = factor(key, total, step, "euler")
            for end in (high - step, low):
                rhs = values + step * (source + multiplier)
                pin(rhs, end)
                candidate = solver.solve(rhs)
                solves += 1
                values, multiplier = obstacle_step(candidate, multiplier, step, end)
                damped_steps += 1
            damping = False
        else:
            theta_dt = dt / 3
            s_solver = factor(key, a1, theta_dt, "spot")
            v_solver = factor(key, a2, theta_dt, "variance")
            a1_old, a2_old = a1 @ values, a2 @ values
            y0 = values + dt * (total @ values + source + multiplier)
            pin(y0, low)
            y1 = s_solver.solve(pin(y0 - theta_dt * a1_old, low))
            y2 = v_solver.solve(pin(y1 - theta_dt * a2_old, low))
            difference = y2 - values
            corrected = y0 + theta_dt * (a0 @ difference) + dt / 6 * (total @ difference)
            pin(corrected, low)
            y1 = s_solver.solve(pin(corrected - theta_dt * a1_old, low))
            candidate = v_solver.solve(pin(y1 - theta_dt * a2_old, low))
            solves += 4
            values, multiplier = obstacle_step(candidate, multiplier, dt, low)
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1e10 * max(1, spot[-1]):
            raise PricingError("Heston finite-difference solution diverged or became nonfinite")
        event_time = float(low) if float(low) in events else None
        if event_time is not None:
            values = apply_dividend(values, event_time)
            multiplier.fill(0)
            damping = True
        if american:
            records.extend(_exercise_records(values, obstacle, spot, variance, params.v0,
                                             low, calls, config.lcp_tolerance,
                                             event="before" if event_time is not None else None))
    prices, deltas, delta_left, delta_right = _interpolate(
        values, spot, variance, market.spot, params.v0)
    if not np.all(np.isfinite(np.r_[prices, deltas])):
        raise PricingError("Heston price/delta interpolation produced nonfinite values")
    return {"prices": prices, "deltas": deltas, "delta_left": delta_left,
            "delta_right": delta_right, "exercise_boundaries": records,
            "diagnostics": {"method": "MCS-IT", "adi_theta": 1 / 3,
                            "time_steps": len(times) - 1, "linear_solves": solves,
                            "factorizations": len(factor_cache),
                            "fully_implicit_damping_steps": damped_steps,
                            "terminal_payoff_smoothing": "cell average at the payoff kink only",
                            "contract_count": nc, "dividend_events": event_deltas,
                            "backend": "heston_mcs_ikonen_toivanen",
                            "dividend_event_sensitivities": event_deltas,
                            "spot_max": float(spot[-1]),
                            "variance_max": float(variance[-1]),
                            "variance_zero_boundary": "degenerate PDE",
                            "upper_spot_boundary": "linear, slope call=1/put=0",
                            "upper_variance_boundary": "zero first derivative"},
            "grid": grid}
