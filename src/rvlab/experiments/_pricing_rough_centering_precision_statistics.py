"""Whole-path inference for an independent predictable-compensator audit.

The projected quantity integrates the *actual* next-state control using an
independent implementation and subtracts the center used by the simulator.
Predictable stopping gates preserve its expectation.  Its mean is never set
to zero, and a small standard error does not certify quadrature accuracy.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from ._pricing_rough_centering_statistics import summarize_independent_centering_runs


_REQUIRED_SAMPLES = frozenset(("raw_contrast", "projected_coarse", "projected_fine",
    "quadrature_error", "tail_error", "arithmetic_error"))
_ALLOWANCES = ("quadrature_error", "tail_error", "arithmetic_error")


def summarize_precision_runs(runs, *, price_allowance, confidence_multiplier=1.96,
        forbidden_seeds=()):
    """Summarize aligned raw and independently integrated path quantities.

    Every run has exactly ``seed`` and ``samples`` fields.  Samples contain
    the six arrays in ``_REQUIRED_SAMPLES``; the final three are nonnegative
    *whole-path* numerical error allowances supplied by the audit.  No date
    is treated as an independent observation.  Raw, coarse and fine values
    on a path must share its dynamics and frozen stopping gates.

    The diagnostic fine-estimator interval adds the path-averaged absolute
    refinement change, quadrature error estimate, tail allowance and
    arithmetic allowance to its marginal normal sampling half-width.  This
    is an explicit numerical uncertainty budget, not a rigorous global
    error bound: refinement agreement and adaptive-quadrature estimates can
    both miss error.  Additional lower/upper pseudo-outcome intervals retain
    sampling variation and covariance of the numerical allowances themselves.
    Sampling and numerical diagnostics remain separate.
    """
    if (isinstance(price_allowance, (bool, np.bool_))
            or not isinstance(price_allowance, Real)
            or not np.isfinite(price_allowance) or price_allowance <= 0):
        raise ValueError("price_allowance must be finite and positive")
    if not isinstance(runs, (list, tuple)) or not runs:
        raise ValueError("Declare a nonempty independent-run list")
    prepared, inferential_runs = [], []
    for run in runs:
        if not isinstance(run, dict) or set(run) != {"seed", "samples"}:
            raise ValueError("Each run must declare exactly seed and samples")
        samples = run["samples"]
        if not isinstance(samples, dict) or set(samples) != _REQUIRED_SAMPLES:
            raise ValueError("Each precision run must contain exactly the six required sample arrays")
        arrays, size = {}, None
        for name, values in samples.items():
            a = np.asarray(values, dtype=float)
            if (a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
                    or (size is not None and len(a) != size)):
                raise ValueError("Each precision run requires finite aligned path arrays, with at least two paths")
            if name in _ALLOWANCES and np.any(a < 0):
                raise ValueError("Numerical path allowances must be nonnegative")
            size, arrays[name] = len(a), a
        prepared.append((run["seed"], size, arrays))
        numerical_path = np.abs(arrays["projected_fine"]-arrays["projected_coarse"])
        numerical_path = numerical_path+sum(arrays[name] for name in _ALLOWANCES)
        inferential_runs.append({"seed": run["seed"], "samples": {
            "raw_contrast": arrays["raw_contrast"],
            "projected_coarse": arrays["projected_coarse"],
            "projected_fine": arrays["projected_fine"],
            "raw_minus_projected": arrays["raw_contrast"]-arrays["projected_fine"],
            "signed_refinement_change": arrays["projected_fine"]-arrays["projected_coarse"],
            "lower_numerical_endpoint": arrays["projected_fine"]-numerical_path,
            "upper_numerical_endpoint": arrays["projected_fine"]+numerical_path}})
    statistics = summarize_independent_centering_runs(inferential_runs,
        confidence_multiplier=confidence_multiplier, forbidden_seeds=forbidden_seeds)
    per_seed, pooled = statistics["per_seed"].copy(), statistics["pooled"].copy()
    budgets = []
    budget_names = ("absolute_refinement_change", *_ALLOWANCES)
    totals = {name: 0. for name in budget_names}
    maximums = {name: 0. for name in budget_names}
    total_paths = sum(size for _, size, _ in prepared)
    for seed, size, arrays in prepared:
        budget = {"absolute_refinement_change": np.abs(arrays["projected_fine"]-arrays["projected_coarse"]),
            **{name: arrays[name] for name in _ALLOWANCES}}
        means = {name: float(values.mean()) for name, values in budget.items()}
        for name, values in budget.items():
            totals[name] += float(values.sum())
            maximums[name] = max(maximums[name], float(values.max()))
        fine = per_seed.loc[(per_seed.seed == seed) & (per_seed.quantity == "projected_fine")].iloc[0]
        numerical = sum(means.values())
        sampling = float(confidence_multiplier*fine.standard_error)
        low, high = float(fine.estimate-sampling-numerical), float(fine.estimate+sampling+numerical)
        lower = per_seed.loc[(per_seed.seed == seed) & (per_seed.quantity == "lower_numerical_endpoint")].iloc[0]
        upper = per_seed.loc[(per_seed.seed == seed) & (per_seed.quantity == "upper_numerical_endpoint")].iloc[0]
        endpoint_low, endpoint_high = float(lower.confidence_low), float(upper.confidence_high)
        budgets.append({"scope": "stream", "seed": seed, "paths": size,
            **means, "numerical_envelope": numerical, "sampling_half_width": sampling,
            "diagnostic_half_width": sampling+numerical,
            "diagnostic_low": low, "diagnostic_high": high,
            "diagnostic_interval_within_price_allowance": max(abs(low), abs(high)) <= price_allowance,
            "uncertainty_endpoint_low": endpoint_low, "uncertainty_endpoint_high": endpoint_high,
            "lower_endpoint_standard_error": float(lower.standard_error),
            "upper_endpoint_standard_error": float(upper.standard_error),
            "endpoint_interval_within_price_allowance": max(abs(endpoint_low), abs(endpoint_high)) <= price_allowance})
    means = {name: total/total_paths for name, total in totals.items()}
    fine = pooled.loc[pooled.quantity == "projected_fine"].iloc[0]
    raw = pooled.loc[pooled.quantity == "raw_contrast"].iloc[0]
    numerical = sum(means.values())
    sampling = float(confidence_multiplier*fine.standard_error)
    low, high = float(fine.estimate-sampling-numerical), float(fine.estimate+sampling+numerical)
    lower = pooled.loc[pooled.quantity == "lower_numerical_endpoint"].iloc[0]
    upper = pooled.loc[pooled.quantity == "upper_numerical_endpoint"].iloc[0]
    endpoint_low, endpoint_high = float(lower.confidence_low), float(upper.confidence_high)
    budget_row = {"scope": "pooled", "seed": None, "paths": total_paths, **means,
        **{"maximum_path_"+name: value for name, value in maximums.items()},
        "numerical_envelope": numerical, "sampling_half_width": sampling,
        "diagnostic_half_width": sampling+numerical,
        "diagnostic_low": low, "diagnostic_high": high,
        "diagnostic_interval_within_price_allowance": max(abs(low), abs(high)) <= price_allowance,
        "uncertainty_endpoint_low": endpoint_low, "uncertainty_endpoint_high": endpoint_high,
        "lower_endpoint_standard_error": float(lower.standard_error),
        "upper_endpoint_standard_error": float(upper.standard_error),
        "endpoint_interval_within_price_allowance": max(abs(endpoint_low), abs(endpoint_high)) <= price_allowance,
        "sampling_half_width_within_price_allowance": sampling <= price_allowance,
        "variance_reduction_factor": (float((raw.standard_error/fine.standard_error)**2)
            if fine.standard_error > 0 else None)}
    budgets.append(budget_row)
    for table in (per_seed, pooled):
        table["price_allowance"] = float(price_allowance)
        table["predictable_compensator_expected_value_identity"] = True
        table["projection_mean_assumed_zero"] = False
        table["quadrature_validated_by_sampling_interval"] = False
    numerical_budget = pd.DataFrame(budgets)
    numerical_budget["price_allowance"] = float(price_allowance)
    numerical_budget["numerical_budget_is_certified_bound"] = False
    numerical_budget["american_pricing_ready"] = False
    numerical_budget["stock_delta_ready"] = False
    metadata = {**statistics["metadata"],
        "scope": "Fixed-policy stopped-control expected-contrast audit using an independently evaluated predictable compensator",
        "predictable_compensator_expected_value_identity": True,
        "projection_mean_assumed_zero": False,
        "variance_reduction_guaranteed": False,
        "sampling_interval_certifies_numerical_accuracy": False,
        "numerical_budget_is_certified_bound": False,
        "whole_path_numerical_allowances_required": True,
        "numerical_budget_method": "mean_absolute_refinement_plus_quadrature_plus_tail_plus_arithmetic",
        "endpoint_interval_method": "normal_lower_and_upper_pseudo_outcomes_with_declared_multiplier",
        "tolerances_changed": False, "american_pricing_ready": False,
        "stock_delta_ready": False}
    return {"per_seed": per_seed, "pooled": pooled,
        "numerical_budget": numerical_budget, "metadata": metadata}


__all__ = ["summarize_precision_runs"]
