"""Reconstruct saved finite-rule Greek samples without simulation or refitting."""
from __future__ import annotations

import numpy as np

from rvlab.pricing.rough_simulation import _time_grid

RELATIVE = np.array([-.005, -.0025, 0., .0025, .005])


def validate_delta_samples(arrays, bundle, *, paths, steps, seed):
    proxy, selected, fit = bundle["proxy"], bundle["selected_policy"], bundle["weights"]
    shape = (paths, 65, 5)
    required = {"relative_spots", "times", "discounts", "eligible", "actual_time_grid",
        "operator_time_grid", "strike", "is_call", "stable_weights", "stable_time_edges",
        "stable_components", "initial_spots", "stocks", "factors", "european_proxy", "payoff",
        "european_martingale", "premium_martingale", "micro_boundary_martingale",
        "micro_poststop_martingale", "intrinsic_martingale", "stable_martingale",
        "bellman_computed_mask", "bellman_continuation", "baseline_continuation", "baseline_european_proxy",
        *[p+"_"+e for p in ("baseline", "bellman") for e in ("stops", "raw", "controlled")],
        *[p+"_"+e+"_deltas" for p in ("baseline", "bellman") for e in ("raw", "controlled")]}
    if not required.issubset(arrays):
        raise ValueError("Missing declared Bellman delta sample arrays")
    times = np.asarray(proxy.exercise_times)
    discount = proxy.market.discount_curve.discount(times)
    if (proxy.market.spot != 100. or proxy.strike != 100. or proxy.is_call
            or proxy.market.dividends.events or proxy.config.steps != 128 or len(times) != 65
            or fit.config.predictable_features):
        raise ValueError("Saved Greek target differs from the frozen no-dividend ATM put")
    for name, expected in (("relative_spots", RELATIVE), ("times", times), ("discounts", discount),
            ("eligible", np.ones(65, bool)), ("initial_spots", 100*(1+RELATIVE)),
            ("operator_time_grid", _time_grid(proxy.params, proxy.market, times, 128)),
            ("actual_time_grid", _time_grid(proxy.params, proxy.market, times, steps)),
            ("stable_weights", fit.weights), ("stable_time_edges", fit.time_edges),
            ("stable_components", np.asarray(fit.config.components)),
            ("strike", np.array(100.)), ("is_call", np.array(False))):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"Changed fixed-rule delta law or identity: {name}")
    fields = ("stocks", "european_proxy", "payoff", "european_martingale", "premium_martingale",
        "micro_boundary_martingale", "micro_poststop_martingale", "intrinsic_martingale",
        "stable_martingale", "baseline_continuation", "baseline_european_proxy")
    if any(arrays[n].shape != shape or not np.isfinite(arrays[n]).all() for n in fields):
        raise ValueError("Malformed or nonfinite full multistock Greek trace")
    stocks, factors = arrays["stocks"], arrays["factors"]
    if (factors.shape != (paths, 65, 4) or not np.isfinite(factors).all() or np.any(factors[:, 0] != 0.)
            or np.any(stocks <= 0) or not np.array_equal(stocks[:, 0], np.broadcast_to(
                arrays["initial_spots"], (paths, 5)))
            or not np.allclose(stocks, stocks[:, :, 2:3]*(1+RELATIVE), rtol=3e-14, atol=0.)):
        raise ValueError("Changed common variance factors or multiplicative stock bumps")
    intrinsic = np.maximum(100-stocks, 0.)
    if not np.array_equal(arrays["payoff"], intrinsic*discount[None, :, None]):
        raise ValueError("Greek cashflows are not discounted payoffs on their own stock paths")
    if steps == 128 and not np.array_equal(arrays["baseline_european_proxy"], arrays["european_proxy"]):
        raise ValueError("Original-grid regression policy changed its European state function")
    for j in range(65):
        q = np.clip(selected.continuations[j].value(stocks[:, j], factors[:, j], 100.), 0., selected.premium_cap)
        expected = arrays["baseline_european_proxy"][:, j]/discount[j]+q
        if not np.array_equal(arrays["baseline_continuation"][:, j], expected):
            raise ValueError("Regression policy changed its frozen continuation premium")
    if (arrays["bellman_computed_mask"].shape != shape or arrays["bellman_computed_mask"].dtype.kind != "b"
            or arrays["bellman_continuation"].shape != shape):
        raise ValueError("Malformed alive-ITM Bellman mask or continuation")
    # Independent component/bin reconstruction, rather than calling the production weight applier.
    mapping = {"european": "european_martingale", "premium": "premium_martingale",
        "micro_boundary": "micro_boundary_martingale"}
    parts = np.stack([arrays[mapping[n]] for n in fit.config.components], axis=-1)
    if np.any(parts[:, 0] != 0):
        raise ValueError("Component martingales must start at zero")
    bins = np.searchsorted(fit.time_edges[1:], times[1:], side="left")
    weighted = np.sum(np.diff(parts, axis=1)*fit.weights[bins][None, :, None, :], axis=-1)
    martingale = np.concatenate((np.zeros((paths, 1, 5)), np.cumsum(weighted, axis=1)), axis=1)
    if not np.allclose(arrays["stable_martingale"], martingale, rtol=0., atol=4e-13):
        raise ValueError("Greek stopped martingale differs from its frozen component weights")
    result = {"seed": seed, "steps": steps, "relative_spots": RELATIVE}
    for policy in ("baseline", "bellman"):
        stops = np.full((paths, 5), 64, dtype=np.int64)
        expected_mask = np.zeros(shape, bool)
        for j in range(1, 64):
            active = (stops == 64) & (intrinsic[:, j] > 0)
            continuation = arrays[policy+"_continuation"][:, j]
            if policy == "bellman":
                expected_mask[:, j] = active
                if not np.isfinite(continuation[active]).all() or np.any(continuation[active] < 0):
                    raise ValueError("Active Bellman continuations must be finite and nonnegative")
            exercise = active & (intrinsic[:, j] >= continuation)
            stops[exercise] = j
        if (arrays[policy+"_stops"].dtype.kind not in "iu"
                or not np.array_equal(stops, arrays[policy+"_stops"])):
            raise ValueError("Stock bump used another bump's stopping time or exercise decision")
        if policy == "bellman" and (not np.array_equal(expected_mask, arrays["bellman_computed_mask"])
                or not np.isnan(arrays["bellman_continuation"][~expected_mask]).all()):
            raise ValueError("Bellman computations omit alive ITM states or fabricate unused values")
        path, bump = np.arange(paths)[:, None], np.arange(5)[None, :]
        raw = arrays["payoff"][path, stops, bump]
        controlled = raw-martingale[path, stops, bump]
        result[policy] = {"stops": stops, "raw": raw, "controlled": controlled}
        for estimator, prices in (("raw", raw), ("controlled", controlled)):
            if not np.allclose(arrays[policy+"_"+estimator], prices, rtol=0., atol=5e-13):
                raise ValueError("Greek endpoint differs from its own stopping cashflow and stopped martingale")
            delta = np.column_stack(((prices[:, 4]-prices[:, 0])/1., (prices[:, 3]-prices[:, 1])/.5))
            if not np.allclose(arrays[policy+"_"+estimator+"_deltas"], delta, rtol=0., atol=2e-12):
                raise ValueError("Saved stock differences do not match the declared symmetric bumps")
    return result
