"""Independent reconstruction of tail-control paths, decisions and endpoints."""
from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import expit


def _coefficient_rows(control, features, intervals=64):
    bins, weights = control.get("bins"), np.asarray(control.get("weights"), float)
    if (isinstance(bins, bool) or not isinstance(bins, Integral) or bins not in (1, 4)
            or weights.shape != (bins*features,) or not np.isfinite(weights).all()
            or np.any(weights < -1e-12) or np.any(weights > 1+1e-12)
            or np.any(weights.reshape(bins, features).sum(axis=1) > 1+1e-12)):
        raise ValueError("Tail control violates its declared coefficient dimensions or per-bin budget")
    cuts = np.linspace(0, intervals, bins+1, dtype=int)
    return np.concatenate([np.broadcast_to(row, (b-a, features))
        for row, a, b in zip(weights.reshape(bins, features), cuts[:-1], cuts[1:])])


def _weighted_addition(corrections, control):
    rows = _coefficient_rows(control, corrections.shape[-1], corrections.shape[1]-1)
    increment = np.sum(np.diff(corrections, axis=1)*rows, axis=-1)
    return np.column_stack((np.zeros(len(corrections)), np.cumsum(increment, axis=1)))


def validate_control_specs(controls, completed):
    expected = {"current baseline"} | set(completed)
    for name in completed:
        if name.startswith("extended"):
            expected.update(f"{name} shrink{s:g}" for s in (.25, .5))
    if set(controls) != expected or next(iter(controls), None) != "current baseline":
        raise ValueError("A fitted control, failed attempt or declared shrinkage candidate was omitted")
    fallback = controls["current baseline"]
    if (fallback.get("family") != "fallback" or fallback.get("bins") != 1
            or fallback.get("penalty") != 0. or fallback.get("shrink_scale") != 1.
            or not np.array_equal(fallback.get("weights"), np.zeros(7))):
        raise ValueError("The unchanged current-control fallback was altered")
    for name, control in controls.items():
        _coefficient_rows(control, 7)
        if name == "current baseline":
            continue
        family = control.get("family")
        if family not in ("poststop", "extended"):
            raise ValueError("Unknown tail-control family")
        if family == "poststop" and np.any(np.asarray(control["weights"]).reshape(control["bins"], 7)[:, 3:] != 0):
            raise ValueError("A strictly post-stop control contains pre-stop features")
        parent = control.get("parent_control")
        if parent is not None:
            scale = control.get("shrink_scale")
            if (family != "extended" or parent not in completed or scale not in (.25, .5)
                    or name != f"{parent} shrink{scale:g}"
                    or not np.array_equal(control["weights"], np.asarray(controls[parent]["weights"])*scale)
                    or any(control.get(k) != controls[parent].get(k) for k in ("family", "bins", "penalty"))):
                raise ValueError("Changed deterministic shrinkage of a fitted parent control")
        elif (control.get("shrink_scale") != 1.
                or name != f"{family} {control['bins']} bins penalty{control.get('penalty'):g}"
                or control.get("penalty") not in (0., 1e-4)):
            raise ValueError("Changed predeclared tail-control candidate")


def validate_tail_samples(arrays, controls, bundle, prior_control, grid_control, policy_specs, *, paths):
    """Check full states, predictable gates, every decision and every saved price."""
    from rvlab.pricing.rough_simulation import _time_grid
    names = {"times", "discounts", "eligible", "actual_time_grid", "stocks", "factors", "european_proxy",
        "payoff", "european_martingale", "intrinsic_martingale", "premium_martingale", "micro_boundary_martingale",
        "stable_old_martingale", "prior_local_corrections", "prior_martingale", "grid_corrections", "current_martingale",
        "tail_corrections", "tail_gates", "tail_margin", "raw", "stops"}
    names.update(f"control{i}_{suffix}" for i in range(len(controls)) for suffix in ("lower", "upper"))
    if set(arrays) != names or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Missing, unexpected or nonfinite tail-attribution path arrays")
    proxy, policy, fit = bundle["proxy"], bundle["selected_policy"], bundle["weights"]
    times, discount, eligible = (arrays[k] for k in ("times", "discounts", "eligible"))
    if (not np.array_equal(times, proxy.exercise_times) or len(times) != 65
            or not np.array_equal(times, policy.times) or not np.array_equal(times, proxy.continuation_times)
            or not np.array_equal(discount, proxy.market.discount_curve.discount(times))
            or eligible.dtype.kind != "b" or eligible.shape != (65,) or not eligible.all()
            or not np.array_equal(arrays["actual_time_grid"], _time_grid(proxy.params, proxy.market, times, 128))):
        raise ValueError("Tail-attribution simulation or exercise calendars changed")
    shapes = {"stocks": (paths, 65, 1), "factors": (paths, 65, len(proxy.kernel.nodes)),
        "european_proxy": (paths, 65, 1), "payoff": (paths, 65), "raw": (paths, 7), "stops": (paths, 7),
        "prior_local_corrections": (paths, 65, 3), "grid_corrections": (paths, 65, 6),
        "tail_corrections": (paths, 65, 7), "tail_gates": (paths, 64, 7), "tail_margin": (paths, 65)}
    martingale_names = ("european_martingale", "intrinsic_martingale", "premium_martingale", "micro_boundary_martingale",
        "stable_old_martingale", "prior_martingale", "current_martingale")
    shapes.update({name: (paths, 65) for name in martingale_names})
    if any(arrays[k].shape != v for k, v in shapes.items()):
        raise ValueError("Changed tail-study path count, factor count or declared array dimensions")
    stock, factors, euro, payoff = (arrays[k] for k in ("stocks", "factors", "european_proxy", "payoff"))
    if (np.any(stock <= 0) or np.any(payoff < 0) or not np.all(stock[:, 0] == proxy.market.spot)
            or np.any(factors[:, 0] != 0) or any(np.any(arrays[k][:, 0] != 0) for k in martingale_names)
            or any(np.any(arrays[k][:, 0] != 0) for k in ("prior_local_corrections", "grid_corrections", "tail_corrections"))):
        raise ValueError("Invalid tail-study prices, inception memory or martingale origins")
    intrinsic = np.maximum((1 if proxy.is_call else -1)*(stock[:, :, 0]-proxy.strike), 0.)
    if not np.array_equal(payoff, discount*intrinsic):
        raise ValueError("Saved payoff does not equal its discounted vanilla payoff at every state")
    # The frozen baseline uses three deterministic calendar-bin components.
    if fit.config.predictable_features or tuple(fit.config.components) != ("european", "premium", "micro_boundary"):
        raise ValueError("Changed frozen baseline martingale specification")
    components = np.stack([arrays[k] for k in ("european_martingale", "premium_martingale", "micro_boundary_martingale")], axis=-1)
    bins = np.searchsorted(fit.time_edges[1:], times[1:], side="left")
    increments = np.sum(np.diff(components, axis=1)*fit.weights[bins], axis=-1)
    stable = np.column_stack((np.zeros(paths), np.cumsum(increments, axis=1)))
    prior = stable+_weighted_addition(arrays["prior_local_corrections"], prior_control)
    current = prior+_weighted_addition(arrays["grid_corrections"], grid_control)
    for name, expected in (("stable_old_martingale", stable), ("prior_martingale", prior), ("current_martingale", current)):
        if not np.allclose(arrays[name], expected, rtol=0., atol=2e-13):
            raise ValueError("Saved current martingale differs from its independently reconstructed frozen components")
    q = np.column_stack([np.clip(model.value(stock[:, j, 0], factors[:, j], proxy.strike), 0., policy.premium_cap)
        for j, model in enumerate(policy.continuations)])
    margin = (payoff-euro[:, :, 0])/discount-q
    if not np.allclose(arrays["tail_margin"], margin, rtol=0., atol=2e-13):
        raise ValueError("Tail gate margin does not match the frozen continuation rule")
    stops = np.full((paths, 7), 64, dtype=np.int64)
    if len(policy_specs) != 7:
        raise ValueError("The declared seven policy challenges are incomplete")
    for p, spec in enumerate(policy_specs.values()):
        alive = np.ones(paths, bool)
        for j in range(1, 64):
            premium = np.clip(q[:, j]*spec["multiplier"]+spec["offset"]*max(0., (proxy.maturity-times[j])/proxy.maturity), 0., policy.premium_cap)
            action = alive & (intrinsic[:, j] > 0) & (intrinsic[:, j] >= euro[:, j, 0]/discount[j]+premium)
            stops[action, p] = j
            alive &= ~action
    if arrays["stops"].dtype.kind not in "iu" or not np.array_equal(arrays["stops"], stops):
        raise ValueError("A saved exercise decision uses a different rule, future information or contract ordering")
    row = np.arange(paths)[:, None]
    raw = payoff[row, stops]
    if not np.array_equal(arrays["raw"], raw):
        raise ValueError("Raw policy payoffs do not equal each policy's own stopping decision")
    post = (np.arange(64)[None, :] >= stops[:, :1]).astype(float)
    left = margin[:, :-1]
    gates = np.stack((post, post*expit(left/.1), post*expit(left/.3), expit(left/.03),
        expit(left/.1), expit(left/.3), (payoff[:, :-1] == 0).astype(float)), axis=-1)
    if not np.allclose(arrays["tail_gates"], gates, rtol=0., atol=2e-13):
        raise ValueError("Tail-control gates differ from their strictly predictable left-state specification")
    # Use the stored current process after its separate reconstruction guard,
    # retaining its exact floating-point arithmetic in this second identity.
    current = arrays["current_martingale"]
    di = np.diff(arrays["intrinsic_martingale"]-current, axis=1)
    de = np.diff(arrays["european_martingale"]-current, axis=1)
    target = np.stack([di]*6+[de], axis=-1)*gates
    corrections = np.concatenate((np.zeros((paths, 1, 7)), np.cumsum(target, axis=1)), axis=1)
    if (not np.allclose(arrays["tail_corrections"], corrections, rtol=0., atol=2e-13)
            or np.any(arrays["tail_corrections"][np.arange(paths), stops[:, 0], :3] != 0)):
        raise ValueError("Tail martingale increments or the post-stop lower-sample identity changed")
    values = {}
    for i, (name, control) in enumerate(controls.items()):
        m = current+_weighted_addition(arrays["tail_corrections"], control)
        lower, upper = raw-m[row, stops], np.max(payoff-m, axis=1)
        if np.any(upper[:, None] < lower-2e-11):
            raise ValueError("The joint tail-study endpoints violate pathwise ordering")
        for suffix, expected in (("lower", lower), ("upper", upper)):
            recorded = arrays[f"control{i}_{suffix}"]
            if recorded.shape != expected.shape or not np.allclose(recorded, expected, rtol=0., atol=2e-13):
                raise ValueError("A tail-study endpoint changed or omitted a policy/path")
        if control["family"] == "poststop" and not np.array_equal(lower[:, 0], values["current baseline"]["lower"][:, 0]):
            raise ValueError("A post-stop-only control changed the frozen policy's lower sample")
        values[name] = {"lower": lower, "upper": upper}
    return {"values": values, "raw": raw, "stops": stops,
        "payoff": payoff, "current_martingale": current}


__all__ = ["validate_control_specs", "validate_tail_samples"]
