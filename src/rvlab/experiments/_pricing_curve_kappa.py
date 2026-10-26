"""Declared Heston curve profiles separating kappa and volatility bounds.

These are research search boxes on the unchanged training contracts. Neither
profile selection nor continuation uses held-out observations. Widening a box
does not certify parameter identification or a useful hedging model.
"""

from __future__ import annotations

from dataclasses import asdict
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd

from rvlab.pricing import PricingConfig, PricingError, parameter_values, price, replace_parameters
from ._pricing_fits import (
    ACCURATE_SPY_CONFIG, PRICE_SCALE_FLOOR, _observation_arrays, fit_attempt,
)
from ._pricing_variance_curve import HESTON_START, _as_curve, _bounds


KAPPA_PROFILE = (5., 10., 20., 40., 80., 160.)
ETA_CAPS = (2., 4.)


def curve_relative_price_jacobian(params, quotes, *, config=ACCURATE_SPY_CONFIG,
                                  parameters=None, relative_step=1e-4):
    """Local scaled-price sensitivity, including flattened theta coordinates.

    The declared coordinates are parameter/max(abs(parameter), .01). This is
    a local numerical conditioning diagnostic, not a confidence interval.
    Fixed profile parameters should be omitted by the caller. Physical
    boundaries use one-sided differences, with the scheme recorded.
    """
    if (isinstance(relative_step, bool) or not isinstance(relative_step, Real)
            or not np.isfinite(relative_step) or relative_step <= 0):
        raise ValueError("relative_step must be finite and positive")
    if not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig")
    values = parameter_values(params)
    requested = list(values) if parameters is None else list(parameters)
    if not requested or len(set(requested)) != len(requested) or set(requested)-set(values):
        raise ValueError("parameters must contain unique model parameter names")
    _, scales, train = _observation_arrays(quotes)
    selected = quotes.loc[train]
    kwargs = {name: selected[name].to_numpy() for name in
              ("forward", "discount", "strike", "maturity", "is_call")}

    def evaluate(parameters):
        result = np.asarray(price(parameters, config=config, **kwargs), float)
        if result.shape != (len(selected),) or not np.all(np.isfinite(result)):
            raise PricingError("Curve Jacobian produced invalid training prices")
        return result / scales[train]

    base, columns, coordinate_scales, schemes = evaluate(params), [], {}, {}
    for name in requested:
        coordinate_scales[name] = max(abs(float(values[name])), .01)
        step = relative_step*coordinate_scales[name]
        candidates = []
        for sign in (1., -1.):
            try:
                candidates.append(replace_parameters(params, **{name: values[name]+sign*step}))
            except ValueError:
                candidates.append(None)
        plus, minus = candidates
        if plus is not None and minus is not None:
            column = (evaluate(plus)-evaluate(minus))/(2*relative_step)
            schemes[name] = "central"
        elif plus is not None:
            column = (evaluate(plus)-base)/relative_step
            schemes[name] = "forward"
        elif minus is not None:
            column = (base-evaluate(minus))/relative_step
            schemes[name] = "backward"
        else:
            raise ValueError(f"relative_step leaves no valid perturbation for {name}")
        columns.append(column)
    jacobian = np.column_stack(columns)
    _, singular, right = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = (float(singular[0]/singular[-1])
                 if rank == len(requested) and singular[-1] > 0 else None)
    return {"status": "ok", "error": None, "parameters": requested,
        "parameter_scales": coordinate_scales, "relative_step": float(relative_step),
        "finite_difference_schemes": schemes, "singular_values": singular.tolist(),
        "rank": rank, "condition_number": condition,
        "weak_direction": dict(zip(requested, right[-1].tolist())),
        "jacobian": jacobian.tolist(), "n_training": int(train.sum()),
        "price_scale_floor": PRICE_SCALE_FLOOR, "config": asdict(config),
        "interpretation": "Local relative-coordinate sensitivity; not confidence intervals or global identification"}


def heston_curve_kappa_study(quotes, *, config=ACCURATE_SPY_CONFIG, max_nfev=80,
                             kappa_values=KAPPA_PROFILE, eta_caps=ETA_CAPS,
                             include_joint=True):
    """Reoptimise six curve coordinates at fixed kappa and two eta caps.

    Each cap has independent declared starts at every kappa plus a continuation
    from the previous *training-objective-selected* profile. Both attempts are
    retained, and profile summaries choose only their training objective. The
    first point has one attempt because a duplicate start adds no information.
    """
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or not 1 <= max_nfev <= 80:
        raise ValueError("max_nfev must be an integer in [1, 80]")
    if not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig")
    if not isinstance(include_joint, bool):
        raise ValueError("include_joint must be boolean")
    kappa_values, eta_caps = tuple(kappa_values), tuple(eta_caps)
    for label, values in (("kappa_values", kappa_values), ("eta_caps", eta_caps)):
        if (not values or any(isinstance(x, bool) or not isinstance(x, Real)
                or not np.isfinite(x) or x <= 0 for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{label} must contain distinct finite positive values")
    if min(eta_caps) < .01:
        raise ValueError("eta_caps must be at least .01")
    _observation_arrays(quotes)
    maturities = tuple(sorted(quotes.maturity.unique()))
    if len(maturities) != 3:
        raise ValueError("Heston curve profile requires three maturities")
    source = quotes.copy(deep=True)
    declared = _as_curve(HESTON_START, maturities[:-1])
    started, attempts, profiles, jacobians = time.perf_counter(), [], [], []
    selected_parameters, predictions = [], []
    for cap in eta_caps:
        previous = None
        for kappa in kappa_values:
            initial = replace_parameters(declared, kappa=float(kappa),
                vol_of_vol=min(declared.vol_of_vol, float(cap)))
            bounds = {**_bounds(initial), "kappa": (float(kappa), float(kappa)),
                      "vol_of_vol": (.01, float(cap))}
            candidates = [("declared", initial)]
            if previous is not None:
                candidates.append(("continuation", replace_parameters(previous, kappa=float(kappa))))
            current = []
            for start_name, start in candidates:
                label = f"heston_curve_kappa_{kappa:g}_eta_cap_{cap:g}_{start_name}"
                attempt = fit_attempt(label, source, start, bounds=bounds,
                                      config=config, max_nfev=max_nfev)
                attempt.summary.update(profile_kappa=float(kappa), eta_cap=float(cap),
                    profile_start=start_name, model="heston_curve", study_group="kappa_profile",
                    initial_provenance="declared flat constant-theta start" if start_name == "declared"
                    else "previous profile selected by training objective only")
                attempts.append(attempt)
                current.append(attempt)
            eligible = [a for a in current if a.result is not None and np.isfinite(a.result.objective)]
            selected = min(eligible, key=lambda a: a.result.objective) if eligible else None
            row = {"profile_kappa": float(kappa), "eta_cap": float(cap),
                   "selected_label": selected.label if selected is not None else None}
            if selected is not None:
                previous = selected.result.params
                row.update(selected.summary)
                row.update(parameter_values(previous))
                row["eta_squared_over_kappa"] = previous.vol_of_vol**2/previous.kappa
                row["kappa_horizon"] = previous.kappa*maturities[-1]
                predictions.append(selected.predictions.assign(label=selected.label,
                    profile_kappa=float(kappa), eta_cap=float(cap)))
                for name, value in parameter_values(previous).items():
                    selected_parameters.append({"label": selected.label, "profile_kappa": float(kappa),
                        "eta_cap": float(cap), "parameter": name, "value": value,
                        "active_bound": selected.summary["active_bounds"].get(name)})
                # Full-coordinate sensitivity includes kappa even though it was
                # fixed in this profile; it exposes cross-parameter ridges.
                try:
                    diagnostic = curve_relative_price_jacobian(previous, source, config=config)
                except PricingError as exc:
                    diagnostic = {"status": "numerical_failure", "error": str(exc),
                                  "condition_number": None, "singular_values": None}
                jacobians.append({"label": selected.label, "profile_kappa": float(kappa),
                    "eta_cap": float(cap), **diagnostic})
                row["condition_number"] = diagnostic.get("condition_number")
            else:
                row.update(status="numerical_failure", error="No finite profile candidate")
            profiles.append(row)
    joint_attempts = []
    if include_joint:
        # These three joint boxes distinguish a kappa cap from a coupled eta
        # cap without optimizing a box on the held-out observations.
        for kappa_cap, eta_cap in ((20., 2.), (80., 2.), (80., 4.)):
            bounds = {**_bounds(declared), "kappa": (.01, kappa_cap),
                      "vol_of_vol": (.01, eta_cap)}
            attempt = fit_attempt(f"heston_curve_joint_kappa_cap_{kappa_cap:g}_eta_cap_{eta_cap:g}",
                source, declared, bounds=bounds, config=config, max_nfev=max_nfev)
            attempt.summary.update(kappa_cap=kappa_cap, eta_cap=eta_cap,
                model="heston_curve", study_group="joint_bounds", profile_start="declared",
                initial_provenance="same declared flat constant-theta start in every joint box")
            attempts.append(attempt)
            joint_attempts.append(attempt)
            if attempt.result is not None:
                fitted = attempt.result.params
                attempt.summary.update(parameter_values(fitted))
                attempt.summary["eta_squared_over_kappa"] = fitted.vol_of_vol**2/fitted.kappa
                for name, value in parameter_values(fitted).items():
                    selected_parameters.append({"label": attempt.label, "profile_kappa": np.nan,
                        "kappa_cap": kappa_cap, "eta_cap": eta_cap, "parameter": name,
                        "value": value, "active_bound": attempt.summary["active_bounds"].get(name)})
                try:
                    diagnostic = curve_relative_price_jacobian(fitted, source, config=config)
                except PricingError as exc:
                    diagnostic = {"status": "numerical_failure", "error": str(exc),
                                  "condition_number": None, "singular_values": None}
                jacobians.append({"label": attempt.label, "profile_kappa": np.nan,
                    "kappa_cap": kappa_cap, "eta_cap": eta_cap, **diagnostic})
                attempt.summary["condition_number"] = diagnostic.get("condition_number")
                predictions.append(attempt.predictions.assign(label=attempt.label,
                    profile_kappa=np.nan, eta_cap=eta_cap))
    tables = {"kappa_profile": pd.DataFrame(profiles),
        "kappa_all_attempts": pd.DataFrame([a.summary for a in attempts]),
        "kappa_joint_bounds": pd.DataFrame([a.summary for a in joint_attempts]),
        "kappa_parameters": pd.DataFrame(selected_parameters),
        "kappa_identification": pd.DataFrame(jacobians),
        "kappa_fitted_prices": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()}
    return {"attempts": attempts, "tables": tables, "manifest": {
        "config": asdict(config), "max_nfev_per_attempt": int(max_nfev),
        "kappa_values": list(kappa_values), "eta_caps": list(eta_caps),
        "joint_boxes": [[20., 2.], [80., 2.], [80., 4.]] if include_joint else [],
        "declared_start": asdict(declared), "selection_rule": "minimum finite training objective only",
        "bounds": {**_bounds(declared), "kappa": "fixed to each profile value", "vol_of_vol": "[.01, eta_cap]"},
        "attempts": [a.summary for a in attempts], "elapsed_seconds": time.perf_counter()-started,
        "limitations": ["Search boxes are experiment inputs; pricing defaults remain unchanged",
            "Fixed kappa profiles have six free coordinates on twelve training contracts",
            "Held-out outcomes do not select profile starts or stop the profile",
            "Local relative-coordinate Jacobian is not an identification guarantee"]}}


__all__ = ["KAPPA_PROFILE", "ETA_CAPS", "curve_relative_price_jacobian", "heston_curve_kappa_study"]
