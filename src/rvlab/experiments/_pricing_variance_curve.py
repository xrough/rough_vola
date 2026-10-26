"""Controlled term-variance experiments on the unchanged SPY demonstration.

The fitted curve is the nonnegative, time-dependent mean-reversion level
theta(t). Its implied forward variance xi_0(t)=E[V_t] is derived from the
Heston/Volterra mean equation; the two curves are deliberately not conflated.
No inferred input in this module is independent stock or exercise data.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from rvlab.pricing import (
    HestonParams, PricingConfig, PricingError, RoughHestonParams,
    price,
)
from ._pricing_data import load_spy_demo
from ._pricing_exercise import exercise_adjusted_surface
from ._pricing_fits import fit_attempt
from ._pricing_spy import _fingerprints


# Explicit previous *training-only* estimates, not a held-out start selection.
# H is fixed throughout the new rough-model comparisons, so adding a curve
# cannot exploit a simultaneous change in the definition of kernel roughness.
HESTON_START = HestonParams(.020039964, .0392007765, 6.3861676, 1.1983434, -.66350526)
ROUGH_START = RoughHestonParams(.0204296763, .25, .2183764, .548515, -.6570358, .31031935)
CURVE_BOUNDS = {
    "v0": (.0025, .25), "theta": (0., 2.), "kappa": (.01, 20.),
    "vol_of_vol": (.01, 2.), "rho": (-.95, 0.),
}


def _parameter_values(params):
    """Flatten numerical parameters while keeping fixed knot locations separate."""
    values = {field.name: float(getattr(params, field.name))
              for field in fields(params) if field.name != "curve"}
    if hasattr(params, "curve"):
        values.update({f"theta_{i}": float(value)
                       for i, value in enumerate(params.curve.levels)})
    return values


def _attempt(label, quotes, initial_params, bounds, config, max_nfev):
    """Use the common workflow and retain flat curve-parameter diagnostics."""
    attempt = fit_attempt(label, quotes, initial_params, bounds=bounds,
                          config=config, max_nfev=max_nfev)
    attempt.summary["parameter_values"] = (_parameter_values(attempt.result.params)
                                           if attempt.result is not None else None)
    return attempt


def _as_curve(params, knot_times):
    from rvlab.pricing.variance_curve import (
        HestonCurveParams, MeanReversionCurve, RoughHestonCurveParams,
    )
    kwargs = {key: getattr(params, key) for key in ("v0", "kappa", "vol_of_vol", "rho")}
    kwargs["curve"] = MeanReversionCurve(tuple(knot_times),
                                         (float(params.theta),) * (len(knot_times) + 1))
    if isinstance(params, RoughHestonParams):
        return RoughHestonCurveParams(**kwargs, hurst=params.hurst)
    return HestonCurveParams(**kwargs)


def _bounds(params, *, curve_only=False):
    values = _parameter_values(params)
    chosen = {key: CURVE_BOUNDS["theta" if key.startswith("theta_") else key]
              for key in values if key != "hurst"}
    if "hurst" in values:
        chosen["hurst"] = (values["hurst"], values["hurst"])
    if curve_only:
        chosen.update({key: (value, value) for key, value in values.items()
                       if not key.startswith("theta_")})
    return chosen


def _curve_tables(attempts, knot_times, horizon):
    from rvlab.pricing.variance_curve import forward_variance
    times = np.unique(np.r_[np.linspace(0., horizon, 161), knot_times, horizon])
    curves, levels, parameter_rows, errors = [], [], [], []
    for attempt in attempts:
        summary = attempt.summary
        initial_values = _parameter_values_from_summary(summary["initial_params"])
        fitted_values = summary["parameter_values"] or {}
        for key, value in initial_values.items():
            parameter_rows.append({"label": attempt.label,
                "input_scenario": summary["input_scenario"], "model": summary["model"],
                "specification": summary["specification"], "parameter": key,
                "initial_value": value, "fitted_value": fitted_values.get(key),
                "free": key in summary["free_parameters"],
                "lower_bound": summary["bounds"][key][0], "upper_bound": summary["bounds"][key][1],
                "active_bound": summary["active_bounds"].get(key)})
        if attempt.result is None:
            continue
        params = attempt.result.params
        curve_params = params if hasattr(params, "curve") else _as_curve(params, knot_times)
        try:
            xi = np.asarray(forward_variance(curve_params, times), float)
        except PricingError as exc:
            errors.append({"label": attempt.label, "status": "numerical_failure", "error": str(exc)})
            continue
        for t, value in zip(times, xi):
            curves.append({"label": attempt.label, "input_scenario": summary["input_scenario"],
                "model": summary["model"], "specification": summary["specification"],
                "maturity": t, "forward_variance": value})
        starts, ends = (0., *knot_times), (*knot_times, horizon)
        for i, (start, end, value) in enumerate(zip(starts, ends, curve_params.curve.levels)):
            levels.append({"label": attempt.label, "input_scenario": summary["input_scenario"],
                "model": summary["model"], "specification": summary["specification"],
                "bucket": i, "start_years": start, "end_years": end,
                "theta_level": value, "immigration_kappa_theta": curve_params.kappa * value})
    return (pd.DataFrame(curves), pd.DataFrame(levels), pd.DataFrame(parameter_rows),
            pd.DataFrame(errors, columns=["label", "status", "error"]))


def _parameter_values_from_summary(values):
    result = {name: value for name, value in values.items() if name != "curve"}
    if "curve" in values:
        result.update({f"theta_{i}": value for i, value in enumerate(values["curve"]["levels"])})
    return result




__all__ = ["HESTON_START", "ROUGH_START", "CURVE_BOUNDS", "variance_curve_demo"]


def variance_curve_demo(context, **kwargs):
    """American replacement; historical European implementation is retired."""
    from ._pricing_american import american_spy_demo
    return american_spy_demo(context, study='variance-curve', **kwargs)
