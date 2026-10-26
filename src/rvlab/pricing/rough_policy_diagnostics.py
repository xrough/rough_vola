"""Policy attribution on frozen inception traces and controlled policy iteration.

These diagnostics hold the kernel, simulation law, exercise dates, valuation
paths and martingale fixed.  A paired increase in a policy's controlled payoff
identifies an improvement in that discrete lower bound. It does not identify
the total policy bias or turn a lower-bound stock delta into an optimal delta.

The trace protocol is deliberately internal: it does not expose conditional
rough-Heston simulation or reset historical factor memory. Policy improvement
uses a third stream independent of both the base-policy and valuation streams.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType

import numpy as np

from .rough_stopping import _fit_continuation, _zero_continuation


def _readonly(value, dtype=float):
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _weights(value):
    weights = np.asarray(value)
    if weights.dtype.kind not in "iuf" or weights.shape != (2,) or not np.isfinite(weights).all():
        raise ValueError("martingale_weights must contain two finite real constants")
    return weights.astype(float)


def _martingale(trace, value):
    """Use fixed constants or provenance-checked train-selected bin weights."""
    from .rough_dual import RoughDualWeights, apply_rough_dual_weights
    if isinstance(value, RoughDualWeights):
        weighted = apply_rough_dual_weights(trace, value)
        return weighted["martingale"], value, {
            "kind": "independent train-selected deterministic calendar bins",
            "weights": value.weights.tolist(), "time_edges": value.time_edges.tolist(),
            "dual_training_seed": value.training_seed}
    weights = _weights(value)
    with np.errstate(over="ignore", invalid="ignore"):
        martingale = weights[0]*trace["european_martingale"]+weights[1]*trace["premium_martingale"]
    if not np.isfinite(martingale).all():
        raise ValueError("weighted martingale overflowed")
    return martingale, tuple(weights), {"kind": "caller-fixed constants", "weights": weights.tolist()}


def _trace_arrays(trace, *, need_states=False):
    if not isinstance(trace, dict) and not isinstance(trace, MappingProxyType):
        raise TypeError("trace must be a mapping returned by the Fourier stopping evaluator")
    try:
        times = np.asarray(trace["times"], float)
        eligible = np.asarray(trace["eligible"])
        payoff, euro_m, premium_m = (np.asarray(trace[name], float) for name in
            ("discounted_payoffs", "european_martingale", "premium_martingale"))
        stops = np.asarray(trace["stop_indices"])
    except KeyError as error:
        raise ValueError(f"missing trace field: {error.args[0]}") from error
    if (times.ndim != 1 or len(times) == 0 or times[0] != 0 or
            not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("trace times must start at zero and increase strictly")
    if eligible.dtype.kind != "b" or eligible.shape != times.shape or not eligible[-1]:
        raise ValueError("trace eligible must be a boolean time vector including maturity")
    if (payoff.ndim != 3 or payoff.shape[0] < 2 or payoff.shape[1] != len(times)
            or payoff.shape[2] == 0 or euro_m.shape != payoff.shape or premium_m.shape != payoff.shape
            or not all(np.isfinite(a).all() for a in (payoff, euro_m, premium_m)) or np.any(payoff < 0)):
        raise ValueError("payoffs and martingales must be finite aligned (path,time,stock-bump) arrays")
    if np.any(euro_m[:, 0] != 0) or np.any(premium_m[:, 0] != 0):
        raise ValueError("trace martingales must start at zero")
    if (stops.dtype.kind not in "iu" or stops.shape != payoff.shape[::2]
            or np.any(stops < 0) or np.any(stops >= len(times))):
        raise ValueError("trace stop_indices must align with paths and stock bumps")
    if np.any((stops != 0) & ~eligible[stops]):
        raise ValueError("a recorded stopping index is not an eligible exercise date")
    if need_states:
        try:
            stocks, factors, euro = (np.asarray(trace[name], float) for name in
                ("stocks", "factors", "european_proxy"))
        except KeyError as error:
            raise ValueError(f"policy diagnostics require trace field: {error.args[0]}") from error
        if (stocks.shape != payoff.shape or euro.shape != payoff.shape or factors.ndim != 3
                or factors.shape[:2] != payoff.shape[:2] or factors.shape[2] == 0
                or not all(np.isfinite(a).all() for a in (stocks, factors, euro))
                or np.any(stocks <= 0)):
            raise ValueError("stocks, factors and discounted European values must align with the trace")
        return times, eligible, payoff, euro_m, premium_m, stops, stocks, factors, euro
    return times, eligible, payoff, euro_m, premium_m, stops


def _summary(samples):
    return {"mean": float(np.mean(samples)),
        "standard_error": float(np.std(samples, ddof=1)/np.sqrt(len(samples)))}


@dataclass(frozen=True)
class TracePolicyEvaluation:
    """Frozen-policy lower samples and a policy-independent common upper sample."""
    raw_samples: np.ndarray
    controlled_samples: np.ndarray
    dual_samples: np.ndarray
    stop_indices: np.ndarray
    martingale_weights: object
    diagnostics: object

    def __post_init__(self):
        for name in ("raw_samples", "controlled_samples", "dual_samples"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "stop_indices", _readonly(self.stop_indices, dtype=int))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class ControlledExercisePolicy:
    """An inception policy fitted using an independent controlled-label stream."""
    base_policy: object
    times: np.ndarray
    continuations: tuple
    exercise_at_inception: bool
    training_seeds: tuple[int, ...]
    premium_floor: float | None
    premium_cap: float | None
    diagnostics: object

    def __post_init__(self):
        object.__setattr__(self, "times", _readonly(self.times))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _clip(values, floor, cap):
    if floor is not None:
        values = np.maximum(values, floor)
    if cap is not None:
        values = np.minimum(values, cap)
    return values


def _policy_stops(policy, times, eligible, stocks, factors, euro):
    custom = isinstance(policy, ControlledExercisePolicy)
    base = policy.base_policy if custom else policy
    if custom and not np.array_equal(policy.times, times):
        raise ValueError("controlled policy and valuation trace must share the same time/exercise grid")
    n, _, bumps = stocks.shape
    stops = np.full((n, bumps), len(times)-1, dtype=int)
    alive = np.ones_like(stops, bool)
    if policy.exercise_at_inception:
        stops[:] = 0
        return stops
    if base.exercise_style == "european":
        return stops
    floor = policy.premium_floor if custom else getattr(base, "premium_floor", 0.)
    cap = policy.premium_cap
    discounts = base.market.discount_curve.discount(times)
    for j in range(1, len(times)-1):
        if not eligible[j]:
            continue
        index = j if custom else int(np.searchsorted(base.continuation_times, times[j]))
        if not custom and (index >= len(base.continuation_times) or base.continuation_times[index] != times[j]):
            raise ValueError("trace exercise date is missing from the frozen policy")
        premium = _clip(policy.continuations[index].value(stocks[:, j], factors[:, j], base.strike), floor, cap)
        continuation = euro[:, j]/discounts[j]+premium
        intrinsic = np.maximum(np.where(base.is_call, stocks[:, j]-base.strike, base.strike-stocks[:, j]), 0.)
        stop = alive & (intrinsic > 0) & (intrinsic >= continuation)
        stops[stop] = j
        alive &= ~stop
    return stops


def evaluate_trace_policy(trace, *, policy=None, martingale_weights=(1., 1.),
                          require_independent=True):
    """Evaluate a policy on a common trace without simulating or retraining.

    ``policy=None`` uses the stopping indices recorded for the original policy.
    The upper sample depends only on the trace and fixed martingale weights;
    changing the exercise policy cannot change that upper sample.
    """
    if not isinstance(require_independent, (bool, np.bool_)):
        raise ValueError("require_independent must be boolean")
    values = _trace_arrays(trace, need_states=policy is not None)
    times, eligible, payoff, euro_m, premium_m, recorded = values[:6]
    martingale, weights, weight_description = _martingale(trace, martingale_weights)
    if policy is None:
        stops = recorded
        training = (trace.get("policy_training_seed"),)
    else:
        stops = _policy_stops(policy, times, eligible, *values[6:])
        training = (policy.training_seeds if isinstance(policy, ControlledExercisePolicy)
            else (policy.config.training_seed,))
    if require_independent:
        training = tuple(dict.fromkeys(tuple(training)+
            tuple(trace.get("additional_control_training_seeds", ()))))
        if "seed" not in trace or any(seed is None for seed in training):
            raise ValueError("independent valuation requires recorded valuation and training seeds")
        if trace["seed"] in training:
            raise ValueError("valuation seed overlaps a policy training stream")
    path, bump = np.arange(payoff.shape[0])[:, None], np.arange(payoff.shape[2])[None, :]
    raw = payoff[path, stops, bump]
    controlled = raw-martingale[path, stops, bump]
    upper = np.max(np.where(eligible[None, :, None], payoff-martingale, -np.inf), axis=1)
    # American exercise at inception is allowed independently of whether the
    # trace recorder marked its common initial date as an exercise event.
    style = trace.get("exercise_style")
    if policy is not None:
        style = (policy.base_policy if isinstance(policy, ControlledExercisePolicy) else policy).exercise_style
    if style == "american":
        upper = np.maximum(upper, payoff[:, 0])
    return TracePolicyEvaluation(raw, controlled, upper, stops, weights,
        {"valuation_seed": trace.get("seed"), "policy_training_seeds": list(training),
         "independent_training_valuation": bool(require_independent),
         "fixed_kernel_law_exercise_grid": True, "same_martingale_for_policy_comparison": True,
         "martingale": weight_description,
         "bound_scope": "declared finite-lift discrete law and exercise grid",
         "optimal_policy_bias_identified": False, "optimal_stock_delta_ready": False})


def compare_trace_policies(trace, candidate_policy, *, baseline_policy=None,
                           martingale_weights=(1., 1.)):
    """Paired lower-bound changes with an exactly common upper estimator."""
    baseline = evaluate_trace_policy(trace, policy=baseline_policy, martingale_weights=martingale_weights)
    candidate = evaluate_trace_policy(trace, policy=candidate_policy, martingale_weights=martingale_weights)
    if not np.array_equal(baseline.dual_samples, candidate.dual_samples):
        raise ValueError("policy attribution requires an identical upper estimator")
    rows = []
    for bump in range(baseline.raw_samples.shape[1]):
        difference = candidate.controlled_samples[:, bump]-baseline.controlled_samples[:, bump]
        raw_difference = candidate.raw_samples[:, bump]-baseline.raw_samples[:, bump]
        rows.append({"stock_bump_index": bump,
            "baseline_lower": _summary(baseline.controlled_samples[:, bump]),
            "candidate_lower": _summary(candidate.controlled_samples[:, bump]),
            "paired_lower_gain": _summary(difference),
            "paired_raw_gain": _summary(raw_difference),
            "common_upper": _summary(baseline.dual_samples[:, bump]),
            "exercise_decision_change_fraction": float(np.mean(candidate.stop_indices[:, bump] != baseline.stop_indices[:, bump]))})
    return {"rows": rows, "baseline": baseline, "candidate": candidate,
        "diagnostics": {"valuation_outcomes_used_for_selection": False,
            "scope": "paired lower-bound improvement; not an exact decomposition of total stopping bias",
            "same_martingale_weights": dict(baseline.diagnostics["martingale"]),
            "same_upper_samples": True}}


def fit_controlled_policy_iteration(base_policy, training_trace, *, martingale_weights=(1., 1.),
                                    premium_floor=0., premium_cap=None, fit_continuation=None,
                                    stock_knots=None, ridge=None):
    """Fit one policy-iteration step with independently frozen total-M labels.

    The base policy is held fixed while a third stream supplies the new labels.
    No valuation trace or validation objective enters fitting. The resulting
    policy can be compared with its base using ``compare_trace_policies``.
    Exercise regression uses the original decision knots by default. A finer
    control interpolation grid attached to an obstacle proxy does not silently
    redefine the exercise basis. Explicit ``stock_knots``/``ridge`` overrides
    affect only the regression, never the simulation law or exercise calendar.
    """
    times, eligible, payoff, euro_m, premium_m, _, stocks, factors, euro = _trace_arrays(training_trace, need_states=True)
    if payoff.shape[2] != 1:
        raise ValueError("policy improvement accepts one inception stock per training trace")
    seed = training_trace.get("seed")
    additional = tuple(base_policy.diagnostics.get("additional_control_training_seeds", ())) if hasattr(base_policy, "diagnostics") else ()
    if seed is None or seed in (base_policy.config.training_seed, base_policy.config.valuation_seed)+additional:
        raise ValueError("policy improvement requires a third independent training seed")
    from .rough_stopping_fourier import FourierStoppingPolicy, fourier_policy_fingerprint
    if (isinstance(base_policy, FourierStoppingPolicy) and
            training_trace.get("policy_fingerprint") != fourier_policy_fingerprint(base_policy)):
        raise ValueError("policy-improvement trace belongs to another frozen base policy")
    if base_policy.exercise_style != "american":
        raise ValueError("policy improvement currently accepts American policies only")
    if any(event.amount > 0 and event.time <= base_policy.maturity for event in base_policy.market.dividends.events):
        raise ValueError("controlled policy improvement currently requires an explicitly empty cash schedule")
    for name, value in (("premium_floor", premium_floor), ("premium_cap", premium_cap)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)):
            raise ValueError(f"{name} must be None or finite real")
    if premium_cap is None:
        premium_cap = base_policy.premium_cap
    if premium_floor is not None and premium_cap is not None and premium_floor > premium_cap:
        raise ValueError("premium_floor cannot exceed premium_cap")
    decision_knots = (getattr(base_policy.continuations[0], "knots", base_policy.config.stock_knots)
        if stock_knots is None else stock_knots)
    fitting_config = replace(base_policy.config, stock_knots=decision_knots,
        ridge=base_policy.config.ridge if ridge is None else ridge)
    martingale, weights, weight_description = _martingale(training_trace, martingale_weights)
    martingale = martingale[..., 0]
    discounts = base_policy.market.discount_curve.discount(times)
    target = payoff[:, -1, 0]-martingale[:, -1]
    fit = _fit_continuation if fit_continuation is None else fit_continuation
    continuations = [None]*len(times)
    continuations[-1] = _zero_continuation(factors[:, -1], fitting_config)
    for j in range(len(times)-2, 0, -1):
        labels = (target+martingale[:, j]-euro[:, j, 0])/discounts[j]
        continuation = fit(stocks[:, j, 0], factors[:, j], labels, base_policy.strike, fitting_config)
        continuations[j] = continuation
        if not eligible[j]:
            continue
        premium = _clip(continuation.value(stocks[:, j, 0], factors[:, j], base_policy.strike), premium_floor, premium_cap)
        future = euro[:, j, 0]/discounts[j]+premium
        intrinsic = payoff[:, j, 0]/discounts[j]
        stop = (intrinsic > 0) & (intrinsic >= future)
        target = np.where(stop, payoff[:, j, 0]-martingale[:, j], target)
    continuations[0] = _zero_continuation(factors[:, 0], fitting_config)
    intrinsic = float(payoff[0, 0, 0])
    inception_stop = intrinsic > 0 and intrinsic >= float(np.mean(target))
    training_seeds = tuple(dict.fromkeys((int(base_policy.config.training_seed), int(seed))+additional))
    if "dual_training_seed" in weight_description:
        training_seeds += (int(weight_description["dual_training_seed"]),)
    return ControlledExercisePolicy(base_policy, times, tuple(continuations), inception_stop,
        training_seeds, premium_floor, premium_cap,
        {"method": "one policy-iteration step using independently frozen total-martingale labels",
         "base_policy_training_seed": int(base_policy.config.training_seed),
         "improvement_training_seed": int(seed), "training_paths": payoff.shape[0],
         "martingale_weights": weight_description, "premium_floor": premium_floor, "premium_cap": premium_cap,
         "fit_stock_knots": tuple(fitting_config.stock_knots), "fit_ridge": fitting_config.ridge,
         "control_stock_knots": tuple(base_policy.config.stock_knots),
         "exercise_and_control_grids_separate": True,
         "regression_ranks": tuple(model.rank if hasattr(model, "rank") else None for model in continuations),
         "regression_conditions": tuple(model.condition if hasattr(model, "condition") else None for model in continuations),
         "valuation_used_in_training": False, "initial_training_continuation": float(np.mean(target)),
         "initial_factor_state": "inception zero memory; trace only",
         "american_pricing_ready": False, "optimal_stock_delta_ready": False})


def _seal_policy_selection(policy, trace, selection_seed, martingale_description):
    """Include every used selection outcome in the later valuation firewall."""
    if isinstance(policy, ControlledExercisePolicy):
        seeds = policy.training_seeds
        diagnostics = dict(policy.diagnostics)
        original = policy
    else:
        times = np.asarray(trace["times"], float)
        indices = np.searchsorted(policy.continuation_times, times)
        if (np.any(indices >= len(policy.continuation_times)) or
                not np.array_equal(policy.continuation_times[indices], times)):
            raise ValueError("selection dates are missing from the frozen baseline policy")
        seeds = (int(policy.config.training_seed),)+tuple(
            policy.diagnostics.get("additional_control_training_seeds", ()))
        diagnostics = dict(policy.diagnostics)
        original = ControlledExercisePolicy(policy, times,
            tuple(policy.continuations[index] for index in indices),
            policy.exercise_at_inception, seeds, getattr(policy, "premium_floor", 0.),
            policy.premium_cap, diagnostics)
    dual_seed = martingale_description.get("dual_training_seed")
    seeds = tuple(dict.fromkeys(seeds+(() if dual_seed is None else (int(dual_seed),))+
        (int(selection_seed),)))
    diagnostics.update({"policy_selection_seed": int(selection_seed),
        "selection_paths_used_for_policy_choice": True,
        "final_valuation_used_for_selection": False,
        "american_pricing_ready": False, "optimal_stock_delta_ready": False})
    return replace(original, training_seeds=seeds, diagnostics=diagnostics)


def select_trace_policy(trace, candidates, *, baseline_name="original",
                        martingale_weights=(1., 1.),
                        selection_rule="paired_lower_confidence"):
    """Select among predeclared policies on an independent training holdout.

    The default score is the paired controlled lower gain minus 1.96 standard
    errors relative to the mandatory baseline fallback. ``lower_mean`` instead
    uses the paired mean gain. These are selection scores, not acceptance gates
    or simultaneous confidence statements. Final evidence requires a fresh
    valuation stream after all candidates and this selection have been frozen.
    The original trace's complete martingale, including its original policy's
    history gates, remains fixed; upper samples must be exactly identical.
    """
    if (not isinstance(candidates, dict) or not candidates or
            any(not isinstance(name, str) or not name or policy is None
                for name, policy in candidates.items())):
        raise ValueError("candidates must map distinct nonempty names to explicit frozen policies")
    if baseline_name not in candidates:
        raise ValueError("policy selection requires a declared baseline fallback")
    if selection_rule not in ("paired_lower_confidence", "lower_mean"):
        raise ValueError("selection_rule must be 'paired_lower_confidence' or 'lower_mean'")
    seed = trace.get("seed")
    if seed is None:
        raise ValueError("independent policy selection requires a recorded selection seed")
    evaluations = {name: evaluate_trace_policy(trace, policy=policy,
        martingale_weights=martingale_weights) for name, policy in candidates.items()}
    baseline = evaluations[baseline_name]
    if baseline.controlled_samples.shape[1] != 1:
        raise ValueError("policy selection accepts one inception stock per selection trace")
    _, _, martingale_description = _martingale(trace, martingale_weights)
    selected, score, rows = baseline_name, 0., []
    for name, value in evaluations.items():
        if not np.array_equal(value.dual_samples, baseline.dual_samples):
            raise ValueError("policy selection requires exactly identical upper samples")
        gain = _summary((value.controlled_samples-baseline.controlled_samples)[:, 0])
        raw_gain = _summary((value.raw_samples-baseline.raw_samples)[:, 0])
        candidate_score = gain["mean"]-(1.96*gain["standard_error"]
            if selection_rule == "paired_lower_confidence" else 0.)
        if candidate_score > score:
            selected, score = name, candidate_score
        lower = _summary(value.controlled_samples[:, 0])
        rows.append({"policy": name, "controlled_lower": lower["mean"],
            "controlled_lower_standard_error": lower["standard_error"],
            "paired_lower_gain": gain["mean"], "paired_lower_gain_standard_error": gain["standard_error"],
            "paired_raw_gain": raw_gain["mean"], "paired_raw_gain_standard_error": raw_gain["standard_error"],
            "selection_score": candidate_score, "selection_rule": selection_rule,
            "exercise_decision_change_fraction": float(np.mean(value.stop_indices != baseline.stop_indices)),
            "same_upper_samples": True, "selection_paths": len(value.raw_samples),
            "selection_seed": int(seed), "independent_final_acceptance_evidence": False})
    sealed = {name: _seal_policy_selection(policy, trace, seed, martingale_description)
        for name, policy in candidates.items()}
    for row in rows:
        row["selected_before_valuation"] = row["policy"] == selected
    return {"selected_name": selected, "selected_policy": sealed[selected],
        "candidates": sealed, "rows": rows,
        "diagnostics": {"baseline_name": baseline_name, "selection_seed": int(seed),
            "selection_rule": selection_rule, "selection_confidence_multiplier": 1.96,
            "selection_used_for_fitting": False, "final_valuation_used_for_selection": False,
            "same_upper_samples": True, "baseline_fallback_included": True,
            "deterministic_tie_rule": "baseline then first declared strictly better score",
            "unchanged_acceptance_tolerances": True,
            "optimal_policy_bias_identified": False, "optimal_stock_delta_ready": False}}


def fit_frozen_policy_value_proxy(base_policy, training_trace, *,
                                  martingale_weights=(1., 1.), fit_continuation=None):
    """Refit only the dual proxy to the value of one frozen exercise policy.

    Decisions at every training state use the original continuation rule.
    Labels are policy cashflows controlled with the independently frozen
    martingale, including the current exercise decision. The result retains
    the exact original ``continuations`` and initial exercise decision;
    only ``value_premiums`` change. A new independent control-training seed
    is propagated to the valuation firewall.
    """
    from .rough_stopping_fourier import FourierStoppingPolicy, fourier_policy_fingerprint
    if not isinstance(base_policy, FourierStoppingPolicy):
        raise TypeError("value-proxy refitting requires FourierStoppingPolicy")
    values = _trace_arrays(training_trace, need_states=True)
    times, eligible, payoff, _, _, _, stocks, factors, euro = values
    if payoff.shape[2] != 1:
        raise ValueError("value-proxy refitting accepts one inception stock per training trace")
    if not np.array_equal(times, base_policy.continuation_times):
        raise ValueError("value-proxy training trace must contain every continuation date")
    if training_trace.get("policy_fingerprint") != fourier_policy_fingerprint(base_policy):
        raise ValueError("value-proxy trace belongs to another frozen base policy")
    seed = training_trace.get("seed")
    existing = tuple(base_policy.diagnostics.get("additional_control_training_seeds", ()))
    forbidden = (base_policy.config.training_seed, base_policy.config.valuation_seed)+existing
    if seed is None or seed in forbidden:
        raise ValueError("value-proxy fitting requires an independent control-training seed")
    martingale, _, description = _martingale(training_trace, martingale_weights)
    martingale = martingale[..., 0]
    discounts = base_policy.market.discount_curve.discount(times)
    target = payoff[:, -1, 0]-martingale[:, -1]
    fit = _fit_continuation if fit_continuation is None else fit_continuation
    models = [None]*len(times)
    models[-1] = _zero_continuation(factors[:, -1], base_policy.config)
    base_floor = getattr(base_policy, "premium_floor", 0.)
    for j in range(len(times)-2, 0, -1):
        # Evaluate the frozen rule counterfactually even on paths which
        # exercised earlier: this is the policy's value when started at j.
        if eligible[j] and base_policy.exercise_style == "american":
            premium = _clip(base_policy.continuations[j].value(
                stocks[:, j, 0], factors[:, j], base_policy.strike), base_floor, base_policy.premium_cap)
            future = euro[:, j, 0]/discounts[j]+premium
            intrinsic = payoff[:, j, 0]/discounts[j]
            stop = (intrinsic > 0) & (intrinsic >= future)
            target = np.where(stop, payoff[:, j, 0]-martingale[:, j], target)
        labels = (target+martingale[:, j]-euro[:, j, 0])/discounts[j]
        models[j] = fit(stocks[:, j, 0], factors[:, j], labels,
            base_policy.strike, base_policy.config)
    models[0] = _zero_continuation(factors[:, 0], base_policy.config)
    control_seeds = tuple(dict.fromkeys(existing+(int(seed),)+
        ((int(description["dual_training_seed"]),) if "dual_training_seed" in description else ())))
    diagnostics = dict(base_policy.diagnostics)
    diagnostics.update({"control_proxy_method": "regression of independently controlled frozen-policy value premiums",
        "exercise_policy_unchanged": True, "base_policy_fingerprint": fourier_policy_fingerprint(base_policy),
        "additional_control_training_seeds": control_seeds,
        "control_training_seed": int(seed), "control_training_paths": payoff.shape[0],
        "control_label_martingale": description, "valuation_used_in_control_fitting": False,
        "american_pricing_ready": False, "optimal_stock_delta_ready": False})
    return replace(base_policy, value_premiums=tuple(models), diagnostics=diagnostics)


__all__ = ["TracePolicyEvaluation", "ControlledExercisePolicy", "evaluate_trace_policy",
    "compare_trace_policies", "fit_controlled_policy_iteration", "select_trace_policy",
    "fit_frozen_policy_value_proxy"]
