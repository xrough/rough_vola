"""Independent decisions/endpoints audit for masked Bellman state evaluations."""
from __future__ import annotations

import numpy as np

from ._pricing_rough_tail_policy import margin_policy_specs
from ._pricing_rough_tail_samples import validate_tail_samples, _weighted_addition


NEW_ARRAYS = {"bellman_continuations", "bellman_direct_continuations", "bellman_computed_mask",
    "baseline_continuation", "new_raw", "new_stops", "new_lower", "common_upper"}


def validate_bellman_samples(arrays, bundle, prior, grid, tail_seal, policy_specs, *, paths, orders=(8,16,32)):
    if not NEW_ARRAYS.issubset(arrays):
        raise ValueError("A declared Bellman value, mask, policy or endpoint array is missing")
    base = {k: v for k, v in arrays.items() if k not in NEW_ARRAYS}
    old = validate_tail_samples(base, tail_seal["controls"], bundle, prior, grid,
        margin_policy_specs(), paths=paths)
    times, discounts = base["times"], base["discounts"]
    proxy = bundle["proxy"]
    intrinsic = np.maximum((1 if proxy.is_call else -1)*(base["stocks"][:,:,0]-proxy.strike),0.)
    mask = arrays["bellman_computed_mask"]
    expected_mask = intrinsic > 0
    expected_mask[:, [0,64]] = False
    if mask.dtype.kind != "b" or mask.shape != (paths,65) or not np.array_equal(mask, expected_mask):
        raise ValueError("The Bellman evaluation mask must include every ITM internal date on every path")
    for key in ("bellman_continuations", "bellman_direct_continuations"):
        value = arrays[key]
        if (value.shape != (paths,65,len(orders)) or not np.isfinite(value[mask]).all()
                or not np.isnan(value[~mask]).all() or np.any(value[mask] < 0)):
            raise ValueError("Bellman values must be finite nonnegative on mask and explicitly NaN elsewhere")
    baseline = arrays["baseline_continuation"]
    expected = intrinsic-base["tail_margin"]
    if (baseline.shape != (paths,65) or not np.isfinite(baseline).all()
            or not np.allclose(baseline, expected, rtol=0., atol=2e-12)):
        raise ValueError("The Bellman baseline continuation changed the frozen rule")
    specs = list(policy_specs.values())
    expected_specs = [{"order": None, "blend": 0.}]+[
        {"order": order, "blend": blend} for order in orders for blend in (.5,1.)]
    if specs != expected_specs:
        raise ValueError("Bellman candidates changed their declared order, integration resolution or blend")
    expected_stops = np.full((paths,len(specs)), 64, dtype=np.int64)
    for p, spec in enumerate(specs):
        alive = np.ones(paths, bool)
        for j in range(1,64):
            continuation = baseline[:,j]
            if p:
                q = arrays["bellman_continuations"][:,j,list(orders).index(spec["order"])]
                continuation = (1-spec["blend"])*continuation+spec["blend"]*q
            stop = alive & (intrinsic[:,j] > 0) & (intrinsic[:,j] >= continuation)
            expected_stops[stop,p] = j
            alive &= ~stop
    stops = arrays["new_stops"]
    if stops.dtype.kind not in "iu" or not np.array_equal(stops, expected_stops):
        raise ValueError("A Bellman policy does not follow its own frozen state-dependent exercise decisions")
    if not np.array_equal(stops[:,0], base["stops"][:,0]):
        raise ValueError("Bellman fallback differs from the frozen baseline stopping rule")
    path = np.arange(paths)[:,None]
    raw = base["payoff"][path,stops]
    if not np.array_equal(arrays["new_raw"], raw):
        raise ValueError("Bellman raw cashflows do not match each policy's own stopping dates")
    common_control = tail_seal["selected_control"]
    control = tail_seal["controls"][common_control]
    martingale = base["current_martingale"]+_weighted_addition(base["tail_corrections"],control)
    lower = raw-martingale[path,stops]
    upper = old["values"][common_control]["upper"]
    for key, expected in (("new_lower", lower),("common_upper",upper)):
        value = arrays[key]
        if value.shape != expected.shape or not np.isfinite(value).all() or not np.allclose(
                value, expected, rtol=0., atol=2e-13):
            raise ValueError("A Bellman lower or common upper endpoint changed its fixed martingale")
    if np.any(upper[:,None] < lower-2e-11):
        raise FloatingPointError("Common dual upper violates pathwise Bellman lower/upper ordering")
    terminal_interval = mask[:,63]
    for key in ("bellman_continuations", "bellman_direct_continuations"):
        terminal = arrays[key][terminal_interval,63]
        if not np.allclose(terminal, terminal[:,:1], rtol=0., atol=2e-13):
            raise ValueError("The last Bellman interval must use the same exact payoff integral for every order")
    return {"raw": raw, "lower": lower, "upper": upper, "stops": stops,
        "continuation": arrays["bellman_continuations"],
        "direct_continuation": arrays["bellman_direct_continuations"],
        "computed_mask": mask, "intrinsic": intrinsic, "baseline_continuation": baseline,
        "stocks": base["stocks"], "factors": base["factors"], "times": times,
        "actual_time_grid": base["actual_time_grid"]}
