"""Local identification diagnostics on the frozen American curve study mesh.

This script does not optimize or change the study's cached fits.  Exact objective
slices hold the other six parameters fixed.  Noise propagation and nuisance
projection are explicitly local linear approximations, not nonlinear profile
fits, confidence intervals, or repeated full calibrations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

from rvlab.experiments import _pricing_american_identification as study
from rvlab.pricing import FDConfig, evaluate_vanilla, parameter_values, replace_parameters
from rvlab.pricing._vanilla_common import build_grid
from rvlab.pricing.optimization import _finite_difference_jacobian


def _read(path):
    return json.loads(path.read_text())


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_diagnostics(checkpoint_directory, output_directory, *, noise_draws=10000):
    started = time.perf_counter()
    directory, output = Path(checkpoint_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    input_paths = [directory/name for name in ("heston_curve_observations.json",
        "heston_curve_clean_3_exploratory.json", "heston_curve_noisy_1_exploratory.json")]
    cache_specification = {"script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "engine_sha256": study._engine_fingerprint(), "noise_draws": noise_draws,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__},
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in input_paths]}
    fingerprint = study._fingerprint(cache_specification)
    cached_path = output/"local_identification.json"
    csv_names = ("local_sensitivities.csv", "conditional_theta2_slices.csv", "linearized_noise.csv")
    if cached_path.exists():
        saved = _read(cached_path)
        if (saved.get("fingerprint") == fingerprint and all(
                (output/name).exists() and hashlib.sha256((output/name).read_bytes()).hexdigest()
                == saved.get("output_sha256", {}).get(name) for name in csv_names)):
            print(f"Reusing exact local identification fingerprint {fingerprint[:16]}", flush=True)
            return saved
    contracts, market = study.joint_contracts(), study.joint_market()
    training = contracts.loc[contracts.split.eq("train")].copy()
    truths, _, bounds = study.joint_specifications()
    truth = truths["heston_curve"]
    config = FDConfig(192, 48, 180, spot_concentration=.05)
    grid = build_grid(market, training.strike.to_numpy(), config, variance=True)
    observations_path = directory / "heston_curve_observations.json"
    clean = np.asarray(_read(observations_path)["payload"]["prices"])
    noise = np.random.default_rng(42).normal(0., study.NOISE_STD, len(contracts))
    quote_sets = study.synthetic_quotes(contracts, clean, noise)
    names = list(parameter_values(truth))
    lower = np.array([bounds["heston_curve"][name][0] for name in names])
    upper = np.array([bounds["heston_curve"][name][1] for name in names])
    inputs, records, sensitivities, slices, noise_records = [], [], [], [], []
    points = [("generating", truth, "clean", None)]
    for dataset, selected_start in (("clean", 3), ("noisy", 1)):
        path = directory / f"heston_curve_{dataset}_{selected_start}_exploratory.json"
        cached = _read(path)
        points.append((f"selected_{dataset}", replace_parameters(truth,
            **cached["payload"]["summary"]["params"]), dataset, cached))
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for label, params, dataset, cached in points:
        begin = time.perf_counter()
        quotes = quote_sets[dataset].loc[lambda frame: frame.split.eq("train")]
        scales = quotes.effective_half_spread.to_numpy()
        target = quotes.market_price.to_numpy()
        values = np.array(list(parameter_values(params).values()))
        evaluation_count = 0

        def prices(candidate):
            nonlocal evaluation_count
            evaluation_count += 1
            result = evaluate_vanilla(replace_parameters(params, **dict(zip(names, candidate))),
                market=market, strike=training.strike.to_numpy(),
                maturity=training.maturity.to_numpy(), is_call=training.is_call.to_numpy(),
                exercise_style="american", config=config, grid=grid)
            return np.asarray(result.prices)

        def weighted(candidate):
            return prices(candidate) / scales

        base_price = prices(values)
        weighted_base = base_price / scales
        two, two_details = _finite_difference_jacobian(weighted, values,
            lower=lower, upper=upper, method="2-point", base=weighted_base)
        jacobian, three_details = _finite_difference_jacobian(weighted, values,
            lower=lower, upper=upper, method="3-point", base=weighted_base)
        norms = np.linalg.norm(jacobian, axis=0)
        normalized = jacobian / norms
        _, singular, right = np.linalg.svd(normalized, full_matrices=False)
        theta_index = names.index("theta_2")
        others = [index for index in range(len(names)) if index != theta_index]
        nuisance = jacobian[:, others]
        theta_column = jacobian[:, theta_index]
        compensation = np.linalg.lstsq(nuisance, -theta_column, rcond=None)[0]
        projected = theta_column + nuisance @ compensation
        inverse = np.linalg.pinv(normalized, rcond=1e-12) / norms[:, None]
        # Target perturbations have heteroscedastic normalized magnitudes even
        # though the declared synthetic dollar noise is homoscedastic.
        noise_map = inverse * (study.NOISE_STD / scales)[None, :]
        local_sd = np.linalg.norm(noise_map, axis=1)
        residual = (base_price-target) / scales
        gradient = jacobian.T @ residual
        kkt_distance = np.where(gradient < 0, upper-values, values-lower)
        record = {"point": label, "dataset": dataset, "parameters": dict(zip(names, values)),
            "weighted_objective": float(np.mean(residual**2)),
            "weighted_singular_values_column_normalized": singular,
            "column_normalized_condition_number": float(singular[0]/singular[-1]),
            "max_relative_two_three_point_difference": float(np.max(
                np.linalg.norm(two-jacobian, axis=0)/norms)),
            "weighted_gradient_half_sum_squares": dict(zip(names, gradient)),
            "physical_kkt_infinity_norm": float(np.max(np.abs(gradient)*kkt_distance)),
            "two_point_details": dict(zip(names, two_details)),
            "three_point_details": dict(zip(names, three_details)),
            "theta2_raw_column_norm": float(np.linalg.norm(theta_column)),
            "theta2_nuisance_projected_norm": float(np.linalg.norm(projected)),
            "theta2_nuisance_projected_fraction": float(np.linalg.norm(projected)/np.linalg.norm(theta_column)),
            "theta2_compensation_per_unit": dict(zip([names[index] for index in others], compensation)),
            "local_parameter_noise_sd": dict(zip(names, local_sd)),
            "weakest_normalized_coordinate_direction": dict(zip(names, right[-1])),
            "weakest_physical_coordinate_direction": dict(zip(names, right[-1]/norms)),
            "training_grid_sha256": hashlib.sha256(grid.spot.tobytes()+grid.variance.tobytes()).hexdigest(),
        }
        for index, name in enumerate(names):
            sensitivities.append({"point": label, "parameter": name,
                "weighted_column_norm": norms[index], "local_noise_sd": local_sd[index],
                "gradient_half_sum_squares": gradient[index],
                "weakest_normalized_direction": right[-1, index],
                "relative_two_three_point_difference": np.linalg.norm(two-jacobian, axis=0)[index]/norms[index]})
        for maturity in sorted(training.maturity.unique()):
            mask = training.maturity.to_numpy() == maturity
            record.setdefault("theta2_dollar_sensitivity_by_maturity", []).append({
                "days": int(round(maturity*365)),
                "maximum_abs_price_change_per_theta2_01": float(np.max(np.abs(
                    jacobian[mask, theta_index]*scales[mask]))*.01)})
        # Exact conditional slices are inexpensive and explicitly not profiles.
        for level in (.0025, .01, .02, .035, .045, .065, .08, .12):
            changed = values.copy()
            changed[theta_index] = level
            model_price = prices(changed)
            difference = (model_price-target)/scales
            slices.append({"point": label, "theta_2": level,
                "objective": float(np.mean(difference**2)),
                "train_dollar_rmse": float(np.sqrt(np.mean((model_price-target)**2))),
                "max_price_move_from_base": float(np.max(np.abs(model_price-base_price))),
                "other_parameters": "fixed; conditional slice, not a reoptimized profile"})
        if label == "generating":
            actual_noise = noise[contracts.split.eq("train").to_numpy()]
            clean_bias = clean[contracts.split.eq("train").to_numpy()]-base_price
            for role, perturbation in (("noise_only", actual_noise),
                    ("generation_discretization_only", clean_bias),
                    ("noise_and_generation_discretization", actual_noise+clean_bias)):
                shift = inverse @ (perturbation/scales)
                record.setdefault("actual_seed42_local_unconstrained_shifts", []).append({
                    "role": role, "shift": dict(zip(names, shift)),
                    "predicted_theta_2": float(values[theta_index]+shift[theta_index])})
            # Deterministic algebraic propagation; no additional PDE solves.
            z = np.random.default_rng(314159).normal(size=(len(training), noise_draws))
            shifts = noise_map @ z
            for index, name in enumerate(names):
                estimates = values[index]+shifts[index]
                noise_records.append({"parameter": name, "draws": noise_draws,
                    "mean": float(estimates.mean()), "std": float(estimates.std()),
                    "q025": float(np.quantile(estimates, .025)), "q5": float(np.median(estimates)),
                    "q975": float(np.quantile(estimates, .975)),
                    "fraction_below_lower": float(np.mean(estimates < lower[index])),
                    "fraction_above_upper": float(np.mean(estimates > upper[index]))})
        record["pricing_evaluations"] = evaluation_count
        record["elapsed_seconds"] = time.perf_counter()-begin
        records.append(record)
        print(f"{label}: normalized condition={record['column_normalized_condition_number']:.2f}; "
              f"theta2 local noise sd={local_sd[theta_index]:.6f}; "
              f"gradient={gradient[theta_index]:.6g}", flush=True)
    metadata = {"study": "Frozen-grid American Heston-curve local identification audit",
        "fingerprint": fingerprint, "cache_specification": cache_specification,
        "config": asdict(config), "engine_sha256": study._engine_fingerprint(),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inputs": inputs+[{"path": str(observations_path),
            "sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest()}],
        "noise_std": study.NOISE_STD, "noise_draws": noise_draws, "noise_seed": 314159,
        "market": asdict(market), "bounds": bounds["heston_curve"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__},
        "limitations": ["Local derivatives and linearized noise propagation do not establish global identification.",
            "The half-spread weights differ from the synthetic noise standard deviation; objectives are not likelihood statistics.",
            "Slices hold all other parameters fixed; they are not nuisance-reoptimized objective profiles.",
            "Linearized unconstrained parameter estimates can leave physical bounds and have no pricing interpretation there.",
            "No optimizer convergence or independent price/delta acceptance criterion is changed."],
        "points": records, "elapsed_seconds": time.perf_counter()-started}
    pd.DataFrame(sensitivities).to_csv(output/"local_sensitivities.csv", index=False)
    pd.DataFrame(slices).to_csv(output/"conditional_theta2_slices.csv", index=False)
    pd.DataFrame(noise_records).to_csv(output/"linearized_noise.csv", index=False)
    metadata["output_sha256"] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                 for name in csv_names}
    (output/"local_identification.json").write_text(json.dumps(_jsonable(metadata), indent=2)+"\n")
    return metadata


def main():
    root = Path(__file__).resolve().parents[1]
    default = root/"notebooks/_output/pricing/research_v1/02_synthetic_calibration"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", type=Path, default=default/
        "american_joint_identification_v2_spot_accuracy/837b8286fe4df229")
    parser.add_argument("--output-directory", type=Path, default=default/"american_curve_diagnosis_v1/identification")
    parser.add_argument("--noise-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.noise_draws < 100:
        parser.error("--noise-draws must be at least 100")
    run_diagnostics(args.checkpoint_directory, args.output_directory, noise_draws=args.noise_draws)


if __name__ == "__main__":
    main()
