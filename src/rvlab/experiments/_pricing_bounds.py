"""Training-only Heston bound and short-maturity identification experiments.

Search boxes in this module are research scenarios, never replacements for
the pricing API's defaults. Every attempt, including exhaustion and numerical
failure, is retained. Held-out prices are diagnostic outputs only.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd

from rvlab.pricing import HestonParams, PricingConfig, PricingError, price
from ._pricing_fits import (
    ACCURATE_SPY_CONFIG, DEMO_BOUNDS, FitAttempt, PRICE_SCALE_FLOOR,
    _observation_arrays, fit_attempt, select_attempt,
)


# The first start is the earlier accurate, constrained SPY fit. The second is
# a declared round-number start, rather than a cached expanded-box optimum.
HESTON_BASELINE_START = HestonParams(
    .01997056817515708, .25, .16867520917564147, .6, -.8281903495287395,
)
HESTON_EXPANDED_START = HestonParams(.02, .04, 6., 1.2, -.65)
HESTON_CONTROLLED_START = HestonParams(.02, .045, 1., 1., -.65)
HESTON_ORIGINAL_BOUNDS = {
    field.name: DEMO_BOUNDS[field.name] for field in fields(HESTON_BASELINE_START)
}
HESTON_EXPANDED_BOUNDS = {
    "v0": (.0025, .25), "theta": (.0025, 2.), "kappa": (.01, 20.),
    "vol_of_vol": (.01, 2.), "rho": (-.95, 0.),
}
THETA_PROFILE_VALUES = (.045, .1, .25, .5, 1., 2.)


def relative_price_jacobian(
    params, quotes: pd.DataFrame, *, config: PricingConfig = ACCURATE_SPY_CONFIG,
    parameters=None, relative_step: float = 1e-4,
) -> dict:
    """Describe the local training-price Jacobian in declared relative units.

    Column ``j`` differentiates scaled training prices with respect to
    ``parameter_j / max(abs(parameter_j), .01)``. Bid/ask observations use the
    same half-spread floor as calibration; price-only observations use dollar
    units. SVD values therefore depend on this coordinate choice. They describe
    local sensitivity, not confidence
    intervals or evidence of global parameter identification. Physically
    invalid centred perturbations use a one-sided difference instead.
    """
    if (isinstance(relative_step, bool) or not isinstance(relative_step, Real)
            or not np.isfinite(relative_step) or relative_step <= 0):
        raise ValueError("relative_step must be finite and positive")
    if not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig")
    names = [field.name for field in fields(params)]
    requested = names if parameters is None else list(parameters)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(names):
        raise ValueError("parameters must contain unique model parameter names")
    _, scales, train = _observation_arrays(quotes)
    training = quotes.loc[train]
    kwargs = {name: training[name].to_numpy() for name in
              ("forward", "discount", "strike", "maturity", "is_call")}
    price_scales = scales[train]

    def evaluate(parameters):
        values = np.asarray(price(parameters, config=config, **kwargs), dtype=float)
        if values.shape != (len(training),) or not np.all(np.isfinite(values)):
            raise PricingError("Jacobian evaluation produced invalid training prices")
        return values / price_scales

    base = evaluate(params)
    columns, schemes, parameter_scales = [], {}, {}
    for name in requested:
        parameter_scale = max(abs(float(getattr(params, name))), .01)
        parameter_scales[name] = parameter_scale
        perturbation = float(relative_step) * parameter_scale
        candidates = []
        for sign in (1., -1.):
            try:
                candidates.append(replace(params, **{
                    name: float(getattr(params, name)) + sign * perturbation}))
            except ValueError:
                candidates.append(None)
        positive, negative = candidates
        if positive is not None and negative is not None:
            column = (evaluate(positive) - evaluate(negative)) / (2. * relative_step)
            schemes[name] = "central"
        elif positive is not None:
            column = (evaluate(positive) - base) / relative_step
            schemes[name] = "forward"
        elif negative is not None:
            column = (base - evaluate(negative)) / relative_step
            schemes[name] = "backward"
        else:
            raise ValueError(f"relative_step leaves no valid perturbation for {name}")
        columns.append(column)
    jacobian = np.column_stack(columns)
    _, singular, right = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = (float(singular[0] / singular[-1])
                 if rank == len(requested) and singular[-1] > 0 else None)
    return {
        "status": "ok", "error": None, "parameters": requested,
        "parameter_scales": parameter_scales, "relative_step": float(relative_step),
        "finite_difference_schemes": schemes,
        "singular_values": singular.tolist(), "rank": rank,
        "condition_number": condition,
        "weak_direction": dict(zip(requested, right[-1].tolist())),
        "jacobian": jacobian.tolist(), "n_training": int(train.sum()),
        "n_parameters": len(requested), "price_scale_floor": PRICE_SCALE_FLOOR,
        "config": asdict(config),
        "interpretation": "Local relative-coordinate sensitivity; not confidence intervals or global identification",
    }


def _tag_attempt(attempt: FitAttempt, group: str, provenance: str, **metadata) -> FitAttempt:
    return replace(attempt, summary={**attempt.summary,
        "study_group": group, "initial_provenance": provenance, **metadata})


def heston_bounds_study(
    quotes: pd.DataFrame, *, config: PricingConfig = ACCURATE_SPY_CONFIG,
    max_nfev: int = 80,
) -> dict:
    """Run declared bound ablations, theta profiles and expiry-specific fits.

    The five joint-surface attempts comprise the original box, theta-only and
    vol-of-vol-only expansions, and two starts in the complete expanded box.
    The two expanded attempts are selected by training objective only; an
    exhausted or unstable result remains eligible as an illustration and
    keeps its status. Six independent theta profiles reoptimise the remaining
    parameters with vol-of-vol capped at .6. Separate-expiry fits hold theta
    and kappa fixed and estimate three parameters. No per-expiry five-parameter
    fit is presented as identified from four training contracts.
    """
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig")
    _observation_arrays(quotes)
    started = time.perf_counter()
    source = quotes.copy(deep=True)
    original = dict(HESTON_ORIGINAL_BOUNDS)
    expanded = dict(HESTON_EXPANDED_BOUNDS)
    cases = (
        ("heston_original_bounds", original, HESTON_BASELINE_START,
         "frozen earlier accurate constrained SPY fit"),
        ("heston_theta_expanded", {**original, "theta": expanded["theta"]},
         HESTON_BASELINE_START, "same frozen constrained start; theta box alone expanded"),
        ("heston_eta_expanded", {**original, "vol_of_vol": expanded["vol_of_vol"]},
         HESTON_BASELINE_START, "same frozen constrained start; vol_of_vol box alone expanded"),
        ("heston_expanded_original_start", expanded, HESTON_BASELINE_START,
         "same frozen constrained start in expanded box"),
        ("heston_expanded_declared_start", expanded, HESTON_EXPANDED_START,
         "declared deterministic round-number start; no cached expanded optimum"),
    )
    attempts, bounds_attempts, profile_attempts, expiry_attempts = [], [], [], []
    for label, bounds, initial, provenance in cases:
        attempt = _tag_attempt(fit_attempt(label, source, initial,
            bounds=bounds, config=config, max_nfev=max_nfev), "bounds", provenance)
        attempts.append(attempt)
        bounds_attempts.append(attempt)
    for theta in THETA_PROFILE_VALUES:
        bounds = {**original, "theta": (theta, theta), "kappa": (.001, 20.)}
        initial = replace(HESTON_BASELINE_START, theta=theta)
        attempt = _tag_attempt(fit_attempt(f"heston_theta_profile_{theta:g}", source,
            initial, bounds=bounds, config=config, max_nfev=max_nfev),
            "theta_profile", "independent frozen constrained start with declared fixed theta",
            profile_theta=theta)
        attempts.append(attempt)
        profile_attempts.append(attempt)
    grouping = "expiry_date" if "expiry_date" in source else "maturity"
    for expiry, subset in source.groupby(grouping, sort=True):
        bounds = {**expanded, "theta": (.045, .045), "kappa": (1., 1.)}
        attempt = _tag_attempt(fit_attempt(f"heston_expiry_controlled_{expiry}", subset,
            HESTON_CONTROLLED_START, bounds=bounds, config=config, max_nfev=max_nfev),
            "per_expiry_controlled", "declared common controlled start, independent per expiry",
            expiry=str(expiry), expiry_grouping=grouping)
        attempts.append(attempt)
        expiry_attempts.append(attempt)

    expanded_attempts = [a for a in bounds_attempts if a.label.startswith("heston_expanded_")]
    selected = select_attempt(expanded_attempts)
    identification = []
    for attempt in [bounds_attempts[0], *([selected] if selected is not None else [])]:
        try:
            if attempt.result is None:
                raise PricingError(attempt.summary.get("error") or "No fitted parameters are available")
            diagnostic = relative_price_jacobian(attempt.result.params, source,
                                                  config=config)
        except PricingError as exc:
            diagnostic = {"status": "numerical_failure", "error": str(exc),
                          "condition_number": None, "singular_values": None}
        identification.append({"label": attempt.label, **diagnostic})

    profiles = []
    for attempt in profile_attempts:
        row = dict(attempt.summary)
        fitted = row.get("params")
        row["kappa_theta"] = fitted["kappa"] * fitted["theta"] if fitted else None
        profiles.append(row)
    tables = {
        "bounds_summary": pd.DataFrame([a.summary for a in bounds_attempts]),
        "theta_profile": pd.DataFrame(profiles),
        "per_expiry_summary": pd.DataFrame([a.summary for a in expiry_attempts]),
        "fitted_prices": pd.concat([a.predictions.assign(
            label=a.label, study_group=a.summary["study_group"]) for a in attempts], ignore_index=True),
        "theta_profile_prices": pd.concat([a.predictions.assign(label=a.label,
            profile_theta=a.summary["profile_theta"]) for a in profile_attempts], ignore_index=True),
        "identification": pd.DataFrame(identification),
    }
    return {"attempts": attempts, "selected_expanded": selected, "tables": tables,
        "manifest": {
            "config": asdict(config), "max_nfev_per_attempt": int(max_nfev),
            "original_bounds": original, "expanded_bounds": expanded,
            "baseline_start": asdict(HESTON_BASELINE_START),
            "declared_expanded_start": asdict(HESTON_EXPANDED_START),
            "controlled_expiry_start": asdict(HESTON_CONTROLLED_START),
            "theta_profile_values": list(THETA_PROFILE_VALUES),
            "selected_expanded": selected.label if selected is not None else None,
            "selection_rule": "minimum finite training objective among the two expanded-box starts only",
            "attempts": [a.summary for a in attempts],
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": ["Empirical numerical refinement, not certified error bounds",
                "Local Jacobian sensitivity is not a confidence interval or proof of identification",
                "No default calibration bounds or market-data readiness are changed",
                "Expiry-specific parameters are a diagnostic relaxation of one common model"],
        }}


__all__ = ["HESTON_BASELINE_START", "HESTON_EXPANDED_START", "HESTON_CONTROLLED_START",
    "HESTON_ORIGINAL_BOUNDS", "HESTON_EXPANDED_BOUNDS", "THETA_PROFILE_VALUES",
    "relative_price_jacobian", "heston_bounds_study"]
