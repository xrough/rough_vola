"""Local identification algebra for the synthetic maturity comparison.

The derivatives are dollar-price derivatives on a caller's frozen numerical
grid.  This module performs no pricing or optimization.  The covariance is the
linearized weighted-least-squares sandwich for the declared dollar noise, not
the inverse Hessian of a likelihood, a nonlinear confidence interval, or proof
of global identification.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def linearized_geometry(dollar_jacobian, parameters, half_spreads, *,
                        noise_std=.005, bounds=None, rcond=1e-12):
    """Summarize local sensitivity with the study's actual residual weights.

    ``parameters`` is an ordered name-to-value mapping, and Jacobian columns
    must follow that order. ``half_spreads`` must already include any spread
    floor. A deficient Jacobian is reported explicitly: minimum-norm noise
    maps are retained for audit, but unidentified coordinate standard
    deviations are infinite rather than spuriously precise.
    """
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a nonempty ordered name-to-value mapping")
    names = list(parameters)
    values = np.asarray(list(parameters.values()), dtype=float)
    jacobian = np.asarray(dollar_jacobian, dtype=float)
    spreads = np.asarray(half_spreads, dtype=float)
    if (jacobian.ndim != 2 or jacobian.shape[1] != len(names) or
            jacobian.shape[0] == 0 or not np.all(np.isfinite(jacobian))):
        raise ValueError("dollar_jacobian must be finite with one column per parameter")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameter values must be finite")
    if (spreads.shape != (len(jacobian),) or not np.all(np.isfinite(spreads)) or
            np.any(spreads <= 0)):
        raise ValueError("half_spreads must be positive, finite and contract aligned")
    if not np.isscalar(noise_std) or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be a finite nonnegative dollar standard deviation")
    if not np.isscalar(rcond) or not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie strictly between zero and one")
    lower, upper = _bounds_arrays(names, bounds)

    weighted = jacobian / spreads[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    nonzero = norms > 0
    normalized = np.divide(weighted, norms, out=np.zeros_like(weighted), where=nonzero)
    _, singular, right = np.linalg.svd(normalized, full_matrices=True)
    cutoff = rcond * singular[0] if len(singular) and singular[0] else 0.
    rank = int(np.sum(singular > cutoff))
    full_rank = rank == len(names)
    condition = float(singular[0]/singular[-1]) if full_rank else float("inf")
    # Scaling before inversion prevents physical units from choosing the SVD
    # truncation. Map back to physical parameter coordinates after inversion.
    inverse = np.divide(np.linalg.pinv(normalized, rcond=rcond), norms[:, None],
                        out=np.zeros((len(names), len(jacobian))), where=nonzero[:, None])
    noise_map = inverse * (noise_std/spreads)[None, :]
    covariance = noise_map @ noise_map.T
    local_sd = np.sqrt(np.maximum(np.diag(covariance), 0.))
    null_directions = right[rank:]
    unidentified = (np.linalg.norm(null_directions, axis=0) > 1e-8
                    if len(null_directions) else np.zeros(len(names), bool))
    reported_sd = np.where(unidentified, np.inf, local_sd)

    sensitivities = []
    for index, name in enumerate(names):
        others = [other for other in range(len(names)) if other != index]
        nuisance = weighted[:, others]
        column = weighted[:, index]
        compensation = np.linalg.lstsq(nuisance, -column, rcond=rcond)[0]
        projected_norm = float(np.linalg.norm(column+nuisance@compensation))
        sensitivities.append({
            "parameter": name, "generating": float(values[index]),
            "weighted_column_norm": float(norms[index]),
            "nuisance_projected_norm": projected_norm,
            "nuisance_projected_fraction": projected_norm/norms[index] if nonzero[index] else 0.,
            "local_noise_sd": float(reported_sd[index]),
            "locally_identified_coordinate": not bool(unidentified[index]),
            "lower": float(lower[index]), "upper": float(upper[index]),
        })
    weakest = right[-1]
    physical_weakest = np.divide(weakest, norms, out=np.zeros_like(weakest), where=nonzero)
    return {
        "contract_count": len(jacobian), "parameter_count": len(names),
        "parameter_names": names, "parameters": dict(zip(names, values.tolist())),
        "noise_std": float(noise_std), "rcond": float(rcond),
        "rank": rank, "full_column_rank": full_rank,
        "column_normalized_condition_number": condition,
        "column_normalized_singular_values": singular.tolist(),
        "weighted_column_norms": dict(zip(names, norms.tolist())),
        "weakest_normalized_direction": dict(zip(names, weakest.tolist())),
        "weakest_physical_direction": dict(zip(names, physical_weakest.tolist())),
        "local_parameter_noise_sd": dict(zip(names, reported_sd.tolist())),
        "minimum_norm_noise_covariance": covariance.tolist(),
        "minimum_norm_noise_map": noise_map.tolist(),
        "physical_target_to_parameter_map": (inverse/spreads[None, :]).tolist(),
        "sensitivities": sensitivities,
        "limitations": [
            "Local linearization at the declared parameter point; not global identification.",
            "Spread-weighted objectives and homoscedastic dollar noise differ; sandwich covariance is used.",
            "Unconstrained perturbations can cross physical bounds and are not nonlinear fitted parameters.",
            "Rank-deficient minimum-norm covariance is retained for audit, not interpreted as identified uncertainty.",
        ],
    }


def _bounds_arrays(names, bounds):
    if bounds is None:
        return np.full(len(names), -np.inf), np.full(len(names), np.inf)
    if not isinstance(bounds, Mapping) or set(bounds) != set(names):
        raise ValueError("bounds must supply exactly the parameter names")
    try:
        array = np.asarray([bounds[name] for name in names], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("bounds must contain lower/upper pairs") from error
    if (array.shape != (len(names), 2) or np.any(np.isnan(array)) or
            np.any(array[:, 0] >= array[:, 1])):
        raise ValueError("bounds must contain increasing lower/upper pairs")
    return array[:, 0], array[:, 1]


def coupled_linearized_noise(geometries, *, draws=10000, seed=314159):
    """Compare local noise maps with shared draws for their common first rows.

    A count-matched repeated-short control may be supplied as another geometry.
    Its repeated quotes are independent hypothetical measurements, not extra
    contracts or an executed repeated calibration. Different maps must order
    their shared rows first for this coupling to be meaningful.
    """
    if not isinstance(geometries, Mapping) or not geometries:
        raise ValueError("geometries must be a nonempty labelled mapping")
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws < 2:
        raise ValueError("draws must be an integer of at least two")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    maximum = max(int(geometry["contract_count"]) for geometry in geometries.values())
    shocks = np.random.default_rng(seed).normal(size=(maximum, draws))
    records = []
    for design, geometry in geometries.items():
        names = geometry["parameter_names"]
        values = np.asarray([geometry["parameters"][name] for name in names])
        maps = np.asarray(geometry["minimum_norm_noise_map"])
        shifts = maps@shocks[:maps.shape[1]]
        for index, row in enumerate(geometry["sensitivities"]):
            estimates = values[index]+shifts[index]
            identified = row["locally_identified_coordinate"]
            records.append({"design": design, "parameter": names[index],
                "draws": int(draws), "seed": int(seed),
                "analytic_sd": row["local_noise_sd"],
                "simulated_sd": float(np.std(estimates, ddof=1)) if identified else np.inf,
                "q025": float(np.quantile(estimates, .025)) if identified else np.nan,
                "median": float(np.median(estimates)) if identified else np.nan,
                "q975": float(np.quantile(estimates, .975)) if identified else np.nan,
                "fraction_below_lower": float(np.mean(estimates < row["lower"])) if identified else np.nan,
                "fraction_above_upper": float(np.mean(estimates > row["upper"])) if identified else np.nan,
                "locally_identified_coordinate": identified})
    return pd.DataFrame(records)
