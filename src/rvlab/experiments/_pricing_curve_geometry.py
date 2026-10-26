"""Training-only derivative and parameter-geometry diagnostics for curve fits.

The affine drift inputs are ``a_i = kappa * theta_i``. Their Jacobian is an
algebraic coordinate change at one fixed parameter point, not a different
model, a different calibration domain, or an identification certificate.
Finite differences use absolute scale floors and feasible one-sided formulas
at bounds; no private SciPy differentiation API is required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from numbers import Real
import time

import numpy as np
import pandas as pd

from rvlab.pricing import PricingConfig, PricingError, price
from rvlab.pricing.optimization import _finite_difference_jacobian
from rvlab.pricing.calibration import _parameter_bounds, _prepare_quotes
from rvlab.pricing.parameters import parameter_values, replace_parameters
from rvlab.pricing.variance_curve import HestonCurveParams




def _spectrum(jacobian, coordinate):
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    full_column_rank = rank == jacobian.shape[1]
    condition = (float(singular[0] / singular[-1])
                 if full_column_rank and singular[-1] > 0 else None)
    summary = {"coordinates": coordinate, "n_training": jacobian.shape[0],
               "n_free_parameters": jacobian.shape[1], "rank": rank,
               "condition_number": condition, "full_column_rank": full_column_rank}
    spectrum = [{"coordinates": coordinate, "singular_index": index,
                 "singular_value": float(value)} for index, value in enumerate(singular)]
    return summary, spectrum


def curve_geometry_diagnostics(params, quotes: pd.DataFrame, *, bounds: Mapping,
                               config: PricingConfig, coordinate_floor: float = .01):
    """Compare local numerical derivatives and kappa/theta geometry.

    ``bounds`` is the declared physical calibration box; equal endpoints fix
    parameters. Only rows labelled ``train`` are read for pricing and targets.
    A table without a split column uses every row as training. Numerical
    failures propagate so the notebook can record an identifiable failed trial.

    ``gradient`` differentiates half the sum of squared calibration residuals.
    ``projected_gradient`` removes outward gradients at active bounds, using
    the displayed 1e-9 physical-bound proximity tolerance. Positive gradients
    at a lower bound satisfy the corresponding directional KKT sign.
    """
    started = time.perf_counter()
    if not isinstance(params, HestonCurveParams):
        raise TypeError("params must be HestonCurveParams or RoughHestonCurveParams")
    if not isinstance(quotes, pd.DataFrame) or quotes.empty:
        raise ValueError("quotes must be a nonempty DataFrame")
    if not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig")
    if not isinstance(bounds, Mapping):
        raise TypeError("bounds must be an explicit physical parameter mapping")
    if (isinstance(coordinate_floor, bool) or not isinstance(coordinate_floor, Real)
            or not np.isfinite(coordinate_floor) or coordinate_floor <= 0):
        raise ValueError("coordinate_floor must be finite and positive")
    split = (quotes["split"].to_numpy() if "split" in quotes
             else np.full(len(quotes), "train"))
    if set(split) - {"train", "heldout", "validation", "test"}:
        raise ValueError("quotes have unknown split labels")
    selected = split == "train"
    if not np.any(selected):
        raise ValueError("at least one training row is required")
    training = quotes.loc[selected].copy(deep=True)
    required_inputs = {"forward", "discount"} - set(training)
    if required_inputs:
        raise ValueError(f"quotes are missing columns: {sorted(required_inputs)}")
    strike, maturity, calls, forward, discount, target, scale = _prepare_quotes(
        training, training.forward, training.discount, .005)
    names, _, lower, upper = _parameter_bounds(params, bounds)
    if not names:
        raise ValueError("geometry diagnostics require at least one free parameter")
    values = parameter_values(params)
    x = np.array([values[name] for name in names])
    lo = np.array([lower[name] for name in names])
    hi = np.array([upper[name] for name in names])
    cache, pricing_evaluations = {}, 0

    def residuals(point):
        nonlocal pricing_evaluations
        key = tuple(point)
        if key not in cache:
            candidate = replace_parameters(params, **dict(zip(names, point)))
            calculated = np.asarray(price(candidate, forward=forward, discount=discount,
                strike=strike, maturity=maturity, is_call=calls, config=config), float)
            if calculated.shape != target.shape or not np.all(np.isfinite(calculated)):
                raise PricingError("Geometry diagnostic produced invalid training prices")
            cache[key] = (calculated - target) / scale
            pricing_evaluations += 1
        return cache[key]

    base = residuals(x)
    two, details_two = _finite_difference_jacobian(residuals, x, lower=lo, upper=hi,
                                                  method="2-point")
    three, details_three = _finite_difference_jacobian(residuals, x, lower=lo, upper=hi,
                                                      method="3-point")
    gradient = three.T @ base
    relative_scales = np.maximum(abs(x), coordinate_floor)
    lower_active = x - lo <= 1e-9 * np.maximum(1., abs(lo))
    upper_active = hi - x <= 1e-9 * np.maximum(1., abs(hi))
    projected = gradient.copy()
    projected[(lower_active & (gradient >= 0)) | (upper_active & (gradient <= 0))] = 0.
    column_records, naive_records, jacobian_records = [], [], []
    for index, name in enumerate(names):
        norm = float(np.linalg.norm(three[:, index]))
        disagreement = float(np.linalg.norm(two[:, index] - three[:, index]))
        column_records.append({"parameter": name, "value": float(x[index]),
            "lower_bound": lower[name], "upper_bound": upper[name],
            "two_point_norm": float(np.linalg.norm(two[:, index])), "three_point_norm": norm,
            "relative_derivative_disagreement": disagreement / max(norm, np.finfo(float).tiny),
            "gradient": float(gradient[index]), "projected_gradient": float(projected[index]),
            "lower_active": bool(lower_active[index]), "upper_active": bool(upper_active[index]),
            "relative_coordinate_scale": float(relative_scales[index]),
            "two_point_scheme": details_two[index]["scheme"],
            "two_point_actual_steps": details_two[index]["actual_steps"],
            "three_point_scheme": details_three[index]["scheme"],
            "three_point_actual_steps": details_three[index]["actual_steps"]})
        for row, original_position in enumerate(np.flatnonzero(selected)):
            jacobian_records.append({"row_position": int(original_position),
                "index_label": str(training.index[row]), "parameter": name,
                "two_point_derivative": float(two[row, index]),
                "three_point_derivative": float(three[row, index])})
        # Preserve the tempting but defective raw-relative rule as a named
        # diagnostic trial. It never supplies an optimizer Jacobian.
        if name.startswith("theta_") and abs(x[index]) < 1e-12:
            step = 1e-4 * abs(x[index])
            candidate = x.copy()
            candidate[index] = np.clip(x[index] + step, lo[index], hi[index])
            actual = candidate[index] - x[index]
            naive_norm = (float(np.linalg.norm((residuals(candidate) - base) / actual))
                          if actual else None)
            naive_records.append({"parameter": name, "value": float(x[index]),
                "raw_relative_step": float(step), "actual_step": float(actual),
                "naive_derivative_norm": naive_norm, "floored_derivative_norm": norm,
                "status": "zero_step" if not actual else
                    "lost_sensitivity" if naive_norm == 0. and norm > 0. else "computed",
                "used_for_optimization": False})

    matrices = {"physical": three, "relative": three * relative_scales}
    ridge = {"kappa": float(params.kappa), "training_cost_half_sum_squares": float(.5 * base @ base),
             "kappa_column_norm_fixed_theta": None, "kappa_column_norm_fixed_immigration": None,
             "immigration_coordinates_available": params.kappa > 0,
             "projected_gradient_infinity_norm": float(np.max(abs(projected)))}
    if params.kappa > 0:
        immigration = three.copy()
        theta_indices = [index for index, name in enumerate(names) if name.startswith("theta_")]
        immigration[:, theta_indices] /= params.kappa
        immigration_values = x.copy()
        immigration_values[theta_indices] *= params.kappa
        if "kappa" in names:
            kappa_index = names.index("kappa")
            # theta levels fixed by the physical box cannot change under this
            # local coordinate map; only free theta coordinates are converted.
            immigration[:, kappa_index] = three[:, kappa_index] - (
                three[:, theta_indices] @ x[theta_indices]) / params.kappa
            ridge.update(kappa_column_norm_fixed_theta=float(np.linalg.norm(three[:, kappa_index])),
                         kappa_column_norm_fixed_immigration=float(np.linalg.norm(immigration[:, kappa_index])))
        matrices.update(immigration=immigration,
            relative_immigration=immigration * np.maximum(abs(immigration_values), coordinate_floor))
    spectra, summaries = [], []
    for coordinate, matrix in matrices.items():
        summary, singular = _spectrum(matrix, coordinate)
        summaries.append(summary)
        spectra.extend(singular)
    return {"tables": {"jacobian_columns": pd.DataFrame(column_records),
        "jacobian_values": pd.DataFrame(jacobian_records),
        "geometry_summary": pd.DataFrame(summaries),
        "jacobian_singular_values": pd.DataFrame(spectra),
        "ridge_summary": pd.DataFrame([ridge]),
        "naive_relative_steps": pd.DataFrame(naive_records, columns=["parameter", "value",
            "raw_relative_step", "actual_step", "naive_derivative_norm", "floored_derivative_norm",
            "status", "used_for_optimization"])}, "metadata": {
        "parameters": list(names), "fixed_parameters": {name: value for name, value in values.items()
                                                         if name not in names},
        "bounds": {name: (lower[name], upper[name]) for name in values},
        "config": asdict(config), "coordinate_floor": float(coordinate_floor),
        "n_training": int(selected.sum()), "pricing_evaluations": pricing_evaluations,
        "elapsed_seconds": time.perf_counter() - started,
        "price_scale_floor": .005, "active_bound_proximity_tolerance": 1e-9,
        "gradient_objective": "half sum of squared training calibration residuals",
        "interpretation": "Local coordinate-dependent sensitivity, not confidence intervals or global identification",
        "immigration_mapping": "a_i=kappa*theta_i for free theta coordinates; fixed theta levels remain fixed",
        "domain_change": False, "heldout_used": False}}


__all__ = ["curve_geometry_diagnostics"]
