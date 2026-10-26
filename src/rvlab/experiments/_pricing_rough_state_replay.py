"""Internal state-only replay of the Fourier pricer's inception random stream.

No valuation, regression, exercise optimization or conditional-state reset is
performed here. The same three draws and floating-point stock update recover
states omitted from a sealed trace. Existing payoffs must agree before using
these states for spent-path diagnostics.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from scipy.special import ndtri

from rvlab.pricing.rough_simulation import (
    _drift_flow_with_integral, _nonnegative_clock, _theta, _time_grid,
    _weak_transition_branches, _MIN_POSITIVE_VOL_OF_VOL, RoughSimulationError)


def replay_inception_states(proxy, *, seed, paths, expected_payoffs=None):
    """Reproduce the single-stock/no-cash Fourier path law, without its controls."""
    for name, value, lower in (("seed", seed, 0), ("paths", paths, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if proxy.market.dividends.events:
        raise ValueError("State replay requires the declared empty cash schedule")
    params, market, kernel, config = proxy.params, proxy.market, proxy.kernel, proxy.config
    if 0 < params.vol_of_vol < _MIN_POSITIVE_VOL_OF_VOL:
        raise RoughSimulationError("The replay does not substitute the deterministic limit")
    times = np.asarray(proxy.exercise_times, float)
    if not np.array_equal(times, proxy.continuation_times):
        raise ValueError("Replay currently requires the common frozen exercise/continuation calendar")
    grid = _time_grid(params, market, times, config.steps)
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    count = len(nodes)
    memory = 8*paths*(len(times)*(count+2)+count+1)
    if memory > config.memory_budget_mb*1024**2:
        raise MemoryError("State-only replay exceeds the declared memory budget")
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    stocks = np.empty((paths, len(times), 1))
    factors = np.empty((paths, len(times), count))
    stock, current = np.full((paths, 1), market.spot), np.zeros((paths, count))
    stocks[:, 0], factors[:, 0] = stock, current
    indices, cache, rng = {float(t): i for i, t in enumerate(times)}, {}, np.random.default_rng(seed)
    for left, right in zip(grid[:-1], grid[1:]):
        dt, theta = float(right-left), _theta(params, float((left+right)/2))
        key = (dt, theta)
        if key not in cache:
            cache[key] = _drift_flow_with_integral(matrix, np.full(count, params.kappa*(theta-params.v0)),
                weights, params.v0, dt if params.vol_of_vol == 0 else dt/2)
        exact = cache[key]
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        for begin in range(0, paths, config.batch_size):
            end = min(paths, begin+config.batch_size)
            old = current[begin:end]
            random = rng.random((end-begin, 3))
            normal = ndtri(np.clip(random[:, 2], np.finfo(float).tiny, 1-np.finfo(float).eps))
            if params.vol_of_vol == 0:
                new = old@exact[:count, :count].T+exact[:count, -1]
                clock = _nonnegative_clock(old@exact[count, :count]+exact[count, -1])
                branches = new[:, None]
                means = np.log(stock[begin:end])[:, None, None]+growth-.5*clock[:, None, None, None]
                clocks = clock[:, None, None, None]
                choice, endpoint = np.zeros(end-begin, int), np.zeros(end-begin, int)
            else:
                law = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
                    flow=exact[:count, :count], shift=exact[:count, -1], integrated_flow=exact,
                    dt=dt, theta=theta, growth=growth, stock_integral=config.stock_integral)
                branches, probabilities = law["new_factors"], law["probabilities"]
                choice = ((random[:, 0] > probabilities[:, 0]).astype(int)
                    +(random[:, 0] > probabilities[:, :2].sum(axis=1)).astype(int))
                endpoint = (random[:, 1] >= .5).astype(int)
                clocks = law["orthogonal"][..., None]
                means = np.log(stock[begin:end])[:, None, None]+law["correlated"][:, :, None, None]-.5*clocks
            index = np.arange(end-begin)
            next_mean = means[index, choice, endpoint]
            next_clock = np.broadcast_to(clocks, means.shape)[index, choice, endpoint]
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                next_stock = np.exp(next_mean+np.sqrt(next_clock)*normal[:, None])
            if not np.isfinite(next_stock).all() or np.any(next_stock <= 0):
                raise RoughSimulationError("State-only stock replay overflowed or underflowed")
            current[begin:end], stock[begin:end] = branches[index, choice], next_stock
        if float(right) in indices:
            j = indices[float(right)]
            stocks[:, j], factors[:, j] = stock, current
    payoff = market.discount_curve.discount(times)[None, :, None]*np.maximum(
        (1 if proxy.is_call else -1)*(stocks-proxy.strike), 0.)
    exact_match = None
    if expected_payoffs is not None:
        expected = np.asarray(expected_payoffs)
        if expected.ndim == 2:
            expected = expected[..., None]
        exact_match = expected.shape == payoff.shape and np.array_equal(expected, payoff)
        if not exact_match:
            raise ValueError("Replayed discounted payoffs differ from the sealed stream; states cannot be attributed")
    return {"times": times.copy(), "actual_time_grid": grid, "stocks": stocks, "factors": factors,
        "discounted_payoffs": payoff, "diagnostics": {"seed": int(seed), "paths": int(paths),
            "actual_steps": len(grid)-1, "same_three_uniform_draws": True,
            "same_fourier_stock_update": True, "sealed_payoffs_bit_exact": exact_match,
            "no_valuation_or_fitting": True, "inception_memory_only": True}}


__all__ = ["replay_inception_states"]
