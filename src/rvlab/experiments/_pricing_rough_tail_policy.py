"""Nonanticipating margin challenges and descriptive spent-path gap timing.

Policy challenges change only the exercise rule. Every rule uses one common
fixed martingale, so its upper estimator is exactly unchanged. Timing of an
ex-post maximum is descriptive and cannot identify optimal policy error.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
import numpy as np
import pandas as pd

BASELINE_NAME = "frozen selected policy"


def margin_policy_specs():
    return {
        BASELINE_NAME: {"offset": 0., "multiplier": 1.},
        "premium offset -0.020": {"offset": -.02, "multiplier": 1.},
        "premium offset -0.005": {"offset": -.005, "multiplier": 1.},
        "premium offset +0.005": {"offset": .005, "multiplier": 1.},
        "premium offset +0.020": {"offset": .02, "multiplier": 1.},
        "premium multiplier 0.9": {"offset": 0., "multiplier": .9},
        "premium multiplier 1.1": {"offset": 0., "multiplier": 1.1},
    }


@dataclass(frozen=True)
class MarginContinuation:
    base: object
    offset: float
    multiplier: float
    remaining_fraction: float
    cap: float

    def value(self, stocks, factors, strike):
        value = np.asarray(self.base.value(stocks, factors, strike), float)
        if self.remaining_fraction == 0.:
            return np.zeros_like(value)
        clipped = np.clip(value, 0., self.cap)
        return np.clip(clipped * self.multiplier + self.offset * self.remaining_fraction, 0., self.cap)


def build_margin_policies(base_policy):
    """Adjust already clipped continuation premiums; reclip to the same cap.

    Offsets decay linearly to zero at expiration. Multipliers act on the
    clipped premium. These are research challenges of the fixed inception
    policy; the inception exercise decision remains the sealed decision.
    """
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
    if not isinstance(base_policy, ControlledExercisePolicy):
        raise TypeError("Margin challenges require a frozen controlled exercise policy")
    proxy = base_policy.base_policy
    if (proxy.exercise_style != "american" or base_policy.premium_floor != 0.
            or base_policy.premium_cap is None or not np.isfinite(base_policy.premium_cap)
            or base_policy.premium_cap <= 0 or proxy.maturity <= 0):
        raise ValueError("Margin challenges require positive-horizon American premiums with floor0 and finite cap")
    # No initial-exercise recalibration is implied by a fitted t>0 rule.
    intrinsic = max(proxy.market.spot - proxy.strike if proxy.is_call else proxy.strike - proxy.market.spot, 0.)
    if intrinsic > 0 or base_policy.exercise_at_inception:
        raise ValueError("This bounded margin study supports an inception-OTM/ATM nonexercised contract")
    specs = margin_policy_specs(); result = {BASELINE_NAME: base_policy}
    for name, spec in specs.items():
        if name == BASELINE_NAME:
            continue
        models = tuple(MarginContinuation(model, spec["offset"], spec["multiplier"],
            max(0., (proxy.maturity - float(t)) / proxy.maturity), base_policy.premium_cap)
            for t, model in zip(base_policy.times, base_policy.continuations))
        diagnostics = dict(base_policy.diagnostics)
        diagnostics.update(margin_challenge=spec, margin_time_profile="(T-t)/T for additive offsets",
            margin_clip_order="clip base premium, apply multiplier/offset, reclip to existing floor0/cap",
            margin_uses_future_states=False, terminal_premium_zero=True,
            american_pricing_ready=False, optimal_stock_delta_ready=False)
        result[name] = replace(base_policy, continuations=models, diagnostics=diagnostics)
    return result


def evaluate_policy_candidates(trace, candidates, common_martingale, *,
        require_independent=True, forbidden_seeds=()):
    """Reapply every decision rule under exactly one supplied fixed control.

    The caller constructs and seals the martingale. Its conditional centering
    is not inferred from an arbitrary array. Spent exploration must explicitly
    disable independence; no output is an automatic readiness decision.
    """
    from rvlab.pricing.rough_policy_diagnostics import _trace_arrays, _policy_stops
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    if not isinstance(require_independent, (bool, np.bool_)):
        raise TypeError("require_independent must be boolean")
    if not isinstance(candidates, dict) or BASELINE_NAME not in candidates or not candidates:
        raise ValueError("Policy challenges require the declared fallback")
    times, eligible, payoff, _, _, _, stocks, factors, euro = _trace_arrays(trace, need_states=True)
    if payoff.shape[2] != 1:
        raise ValueError("This policy-side attribution accepts one inception stock")
    martingale = np.asarray(common_martingale, float)
    if martingale.shape != payoff.shape[:2] or not np.isfinite(martingale).all() or np.any(martingale[:, 0] != 0):
        raise ValueError("Finite aligned common martingale starting at zero required")
    upper = np.max(np.where(eligible, payoff[:, :, 0] - martingale, -np.inf), axis=1)
    upper = np.maximum(upper, payoff[:, 0, 0])
    results = {}
    for name, policy in candidates.items():
        if (not np.array_equal(policy.times, times)
                or trace.get("policy_fingerprint") != fourier_policy_fingerprint(policy.base_policy)):
            raise ValueError("A policy challenge changes the base model or exercise dates")
        forbidden = set(policy.training_seeds) | set(forbidden_seeds)
        if require_independent and (trace.get("seed") is None or trace["seed"] in forbidden):
            raise ValueError("Policy challenge valuation must be independent of fitting and selection")
        stops = _policy_stops(policy, times, eligible, stocks, factors, euro)
        raw = payoff[np.arange(len(payoff)), stops[:, 0], 0]
        lower = raw - martingale[np.arange(len(payoff)), stops[:, 0]]
        if np.any(upper < lower - 2e-11):
            raise FloatingPointError("Common upper no longer dominates the challenged policy")
        results[name] = {"raw": raw, "lower": lower, "upper": upper.copy(), "stops": stops[:, 0]}
    return results


def select_margin_policy(evaluations, candidates, *, selection_seed):
    """Paired lower-gain LCB selects a rule; fresh final evidence is separate."""
    from ._pricing_rough_stock_delta import _summary
    if (isinstance(selection_seed, bool) or not isinstance(selection_seed, Integral)
            or selection_seed < 0 or any(selection_seed in p.training_seeds for p in candidates.values())):
        raise ValueError("Policy selection needs a distinct independent nonnegative integer seed")
    if set(evaluations) != set(candidates) or BASELINE_NAME not in candidates:
        raise ValueError("Selection requires all predeclared policy candidates and fallback")
    baseline = evaluations[BASELINE_NAME]
    selected, best, rows = BASELINE_NAME, 0., []
    for name, result in evaluations.items():
        if not np.array_equal(result["upper"], baseline["upper"]):
            raise ValueError("Policy attribution requires an exactly identical upper estimator")
        stat = _summary(result["lower"] - baseline["lower"])
        score = stat["mean"] - 1.96 * stat["standard_error"]
        rows.append({"policy": name, "paired_lower_gain": stat["mean"],
            "paired_lower_gain_se": stat["standard_error"], "selection_score": score,
            "exercise_change_fraction": float(np.mean(result["stops"] != baseline["stops"])),
            "selection_seed": selection_seed, "final_paths_used_for_selection": False})
        if score > best:
            selected, best = name, score
    sealed = {name: replace(policy, training_seeds=tuple(dict.fromkeys(policy.training_seeds + (int(selection_seed),))))
        for name, policy in candidates.items()}
    for row in rows:
        row["selected_before_final"] = row["policy"] == selected
    return selected, sealed, pd.DataFrame(rows)


def spent_gap_timing(payoff, martingale, stops, *, times, seed):
    """Describe ex-post dual peaks; no peak time is a deployable stopping rule."""
    payoff, martingale, stops, times = map(np.asarray, (payoff, martingale, stops, times))
    if (payoff.ndim != 2 or payoff.shape != martingale.shape or stops.shape != (len(payoff),)
            or times.shape != (payoff.shape[1],) or stops.dtype.kind not in "iu"
            or np.any(stops < 0) or np.any(stops >= len(times))
            or not np.isfinite(payoff).all() or not np.isfinite(martingale).all()):
        raise ValueError("Aligned finite payoff/control arrays and valid stops/times required")
    adjusted = payoff - martingale
    peak = np.argmax(adjusted, axis=1); paths = np.arange(len(payoff))
    lower = adjusted[paths, stops]; upper = adjusted[paths, peak]; gap = upper - lower
    if np.any(gap < -2e-11): raise FloatingPointError("Negative gap in spent-path attribution")
    return pd.DataFrame({"seed": seed, "path_index": paths, "gap": gap,
        "stop_index": stops, "dual_peak_index": peak, "stop_time": times[stops],
        "dual_peak_time": times[peak], "peak_before_stop": peak < stops,
        "peak_after_stop": peak > stops, "peak_equals_stop": peak == stops,
        "stop_at_maturity": stops == len(times)-1, "peak_at_maturity": peak == len(times)-1,
        "payoff_at_stop": payoff[paths, stops], "payoff_at_peak": payoff[paths, peak],
        "martingale_at_stop": martingale[paths, stops], "martingale_at_peak": martingale[paths, peak],
        "lower": lower, "upper": upper, "acceptance_input": False, "used_for_selection": False})
