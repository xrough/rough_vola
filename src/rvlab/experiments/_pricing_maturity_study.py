"""Matched short/long maturity American Heston-curve identification study.

One actual noisy surface (seed 42 by default) is calibrated from three starts.
The 10,000 algebraic perturbations are local linearized noise propagation, not
10,000 nonlinear calibrations.  Numerical acceptance is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from numbers import Integral
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy

from rvlab.pricing import (
    FDConfig, OptimizerConfig, PricingError, evaluate_vanilla,
    parameter_values, replace_parameters,
)
from rvlab.pricing._vanilla_common import build_grid
from rvlab.pricing.optimization import _finite_difference_jacobian
from . import _pricing_american_identification as joint
from ._pricing_maturity_accuracy import maturity_truth_accuracy
from ._pricing_maturity_geometry import linearized_geometry, coupled_linearized_noise


STUDY_VERSION = "maturity_extension_v1"
DESIGNS = ("short", "augmented")
TENORS = (14, 30, 60, 90, 180, 365)
LINEAR_NOISE_DRAWS = 10000
LINEAR_NOISE_SEED = 314159


def maturity_contracts():
    """Append horizons without changing any original contract or weight."""
    original = joint.joint_contracts()
    added = pd.DataFrame([
        {"contract_id": f"american-joint-{days}-{strike}", "maturity": days/365,
         "strike": float(strike), "is_call": strike >= 100,
         "split": "heldout" if strike in (97, 103) else "train",
         "half_spread": (.003, .008, .012, .018, .025, .035)[j]*(1+.25*i)}
        for i, days in enumerate(TENORS[3:], 3)
        for j, strike in enumerate((94, 97, 99, 101, 103, 106))])
    return pd.concat([original, added], ignore_index=True)


def _generate(truth, contracts, market, config):
    begin = time.perf_counter()
    result = evaluate_vanilla(truth, market=market, strike=contracts.strike.to_numpy(),
        maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
        exercise_style="american", config=config)
    return {"prices": result.prices.tolist(), "deltas": result.deltas.tolist(),
        "grid_sha256": result.diagnostics["grid_sha256"],
        "elapsed_seconds": time.perf_counter()-begin}


def _generation_or_preflight(context, truth, contracts, market, config, fit):
    """Reuse a separately audited reference only with exact provenance checks."""
    output = getattr(context, "output_dir", None)
    path = (Path(output)/STUDY_VERSION/"reference_640_128_600.json"
            if output is not None else None)
    if path is not None and path.is_file():
        record = json.loads(path.read_text())
        expected = {"config": asdict(config),
            "contracts": contracts.to_dict("records"),
            "generating_parameters": parameter_values(truth), "market": asdict(market),
            "engine_sha256": joint._engine_fingerprint(), "acceptance": _acceptance(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pandas": pd.__version__}}
        exact = all(joint._fingerprint(record.get(key)) == joint._fingerprint(value)
                    for key, value in expected.items())
        runner = Path(record.get("runner_source_path", ""))
        runner_ok = (runner.is_file() and hashlib.sha256(runner.read_bytes()).hexdigest()
                     == record.get("runner_source_sha256"))
        prices, deltas = np.asarray(record.get("prices", [])), np.asarray(record.get("deltas", []))
        if (exact and runner_ok and prices.shape == (len(contracts),) and deltas.shape == prices.shape
                and np.isfinite(prices+deltas).all()):
            return {"prices": prices.tolist(), "deltas": deltas.tolist(),
                "grid_sha256": record["grid_sha256"], "elapsed_seconds": record["elapsed_seconds"],
                "preflight_reference_path": str(path),
                "preflight_reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return _generate(truth, contracts, market, config)


def _accuracy_or_preflight(context, truth, contracts, market, fit, observed):
    """Reaggregate independently captured PDE values after exact input checks."""
    output = getattr(context, "output_dir", None)
    name = f"preflight_{fit.spot_steps}_{fit.variance_steps}_{fit.time_steps}.json"
    path = Path(output)/STUDY_VERSION/name if output is not None else None
    cached_result = None
    if path is not None and path.is_file():
        record = json.loads(path.read_text())
        expected = {"config": asdict(fit), "contracts": contracts.to_dict("records"),
            "generating_parameters": parameter_values(truth), "market": asdict(market),
            "engine_sha256": joint._engine_fingerprint(), "acceptance": _acceptance(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pandas": pd.__version__}}
        exact = all(joint._fingerprint(record.get(key)) == joint._fingerprint(value)
                    for key, value in expected.items())
        runner = Path(record.get("runner_source_path", ""))
        runner_ok = (runner.is_file() and hashlib.sha256(runner.read_bytes()).hexdigest()
                     == record.get("runner_source_sha256"))
        if exact and runner_ok and record.get("grid_sha256") is not None:
            cached_result = SimpleNamespace(prices=np.asarray(record["prices"]),
                deltas=np.asarray(record["deltas"]), price_stable=record["price_stable"],
                delta_stable=record["delta_stable"], diagnostics={
                    "checks": pd.DataFrame(record["checks"]), "price_changes": record["price_changes"],
                    "pricing_evaluations": record["pricing_evaluations"],
                    "grid_sha256": record["grid_sha256"]})
    result = maturity_truth_accuracy(context, truth, contracts, market, fit,
        observed["prices"], observed["deltas"], precomputed_result=cached_result)
    if cached_result is not None:
        result["preflight_path"] = str(path)
        result["preflight_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _local_geometry(truth, contracts, quotes, market, config, bounds):
    begin = time.perf_counter()
    training = contracts.loc[contracts.split.eq("train")]
    grid = build_grid(market, training.strike.to_numpy(), config, variance=True)
    parameters = parameter_values(truth)
    names, values = list(parameters), np.asarray(list(parameters.values()))
    lower = np.asarray([bounds[name][0] for name in names])
    upper = np.asarray([bounds[name][1] for name in names])
    count = 0

    def dollar_prices(candidate):
        nonlocal count
        count += 1
        priced = evaluate_vanilla(replace_parameters(truth, **dict(zip(names, candidate))),
            market=market, strike=training.strike.to_numpy(),
            maturity=training.maturity.to_numpy(), is_call=training.is_call.to_numpy(),
            exercise_style="american", config=config, grid=grid)
        return np.asarray(priced.prices)

    base = dollar_prices(values)
    two, two_details = _finite_difference_jacobian(dollar_prices, values,
        lower=lower, upper=upper, method="2-point", base=base)
    three, three_details = _finite_difference_jacobian(dollar_prices, values,
        lower=lower, upper=upper, method="3-point", base=base)
    scales = quotes.loc[quotes.split.eq("train"), "effective_half_spread"].to_numpy()
    short_rows = int(joint.joint_contracts().split.eq("train").sum())
    geometries = {
        "short": linearized_geometry(three[:short_rows], parameters, scales[:short_rows],
            noise_std=joint.NOISE_STD, bounds=bounds),
        "augmented": linearized_geometry(three, parameters, scales,
            noise_std=joint.NOISE_STD, bounds=bounds),
        "repeated_short_control": linearized_geometry(np.tile(three[:short_rows], (2, 1)),
            parameters, np.tile(scales[:short_rows], 2), noise_std=joint.NOISE_STD, bounds=bounds)}
    relative = np.linalg.norm((two-three)/scales[:, None], axis=0)/np.maximum(
        np.linalg.norm(three/scales[:, None], axis=0), np.finfo(float).tiny)
    for design, geometry in geometries.items():
        size = short_rows if design != "augmented" else len(training)
        design_relative = np.linalg.norm((two[:size]-three[:size])/scales[:size, None], axis=0)/np.maximum(
            np.linalg.norm(three[:size]/scales[:size, None], axis=0), np.finfo(float).tiny)
        geometry["max_relative_two_three_point_difference"] = float(np.max(design_relative))
        if design != "repeated_short_control":
            actual = quotes.loc[quotes.split.eq("train"), "noise"].to_numpy()[:size]
            clean = quotes.loc[quotes.split.eq("train"), "clean_price"].to_numpy()[:size]
            inverse = np.asarray(geometry["physical_target_to_parameter_map"])
            geometry["actual_noise_local_shift"] = dict(zip(names, (inverse@actual).tolist()))
            geometry["finite_generation_local_shift"] = dict(zip(names, (inverse@(clean-base[:size])).tolist()))
    by_maturity = []
    for tenor in sorted(training.maturity.unique()):
        mask = training.maturity.to_numpy() == tenor
        for index, name in enumerate(names):
            by_maturity.append({"days": int(round(tenor*365)), "parameter": name,
                "max_abs_dollar_derivative": float(np.max(np.abs(three[mask, index]))),
                "weighted_column_norm": float(np.linalg.norm(three[mask, index]/scales[mask]))})
    noise = coupled_linearized_noise(geometries, draws=LINEAR_NOISE_DRAWS, seed=LINEAR_NOISE_SEED)
    return {"geometries": geometries, "linearized_noise": noise.to_dict("records"),
        "by_maturity": by_maturity, "dollar_jacobian_two_point": two.tolist(),
        "dollar_jacobian_three_point": three.tolist(), "base_prices": base.tolist(),
        "parameter_names": names, "two_point_details": two_details,
        "three_point_details": three_details,
        "relative_two_three_point_difference": dict(zip(names, relative.tolist())),
        "training_contract_ids": training.contract_id.tolist(),
        "training_grid_sha256": hashlib.sha256(grid.spot.tobytes()+grid.variance.tobytes()).hexdigest(),
        "pricing_evaluations": count, "elapsed_seconds": time.perf_counter()-begin}


def _old_short_attempt(context, quotes, truth, initial, bounds, fit, generation,
                       market, optimizer, seed, budget, start):
    """Import previous evidence only after exact mathematical-input equality."""
    root = getattr(context, "output_dir", None)
    if root is None or seed != 42:
        return None
    directory = Path(root)/joint.STUDY_VERSION/"837b8286fe4df229"
    path = directory/f"heston_curve_noisy_{start}_exploratory.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    specification = record.get("specification", {})
    if record.get("fingerprint") != joint._fingerprint(specification):
        raise ValueError(f"Corrupt prior short-fit checkpoint fingerprint: {path}")
    expected = {"engine_sha256": joint._engine_fingerprint(), "seed": seed,
        "contracts": joint.joint_contracts().to_dict("records"),
        "market": asdict(market), "truth": parameter_values(truth),
        "generation_config": asdict(generation), "fit_config": asdict(fit),
        "optimizer": asdict(optimizer), "bounds": bounds,
        "initial": parameter_values(initial), "max_nfev": budget,
        "noise_std": joint.NOISE_STD, "spread_floor": joint.SPREAD_FLOOR,
        "quotes": quotes.to_dict("records"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__},
        "acceptance": _acceptance()}
    if any(joint._fingerprint(specification.get(key)) != joint._fingerprint(value)
           for key, value in expected.items()):
        return None
    result = deepcopy(record["payload"])
    result["summary"].update(prior_profile_reused=True, prior_profile_path=str(path),
        prior_profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def _calibrate(context, design, quotes, truth, initial, bounds, market, fit,
               generation, optimizer, seed, budget, start, stage):
    imported = (_old_short_attempt(context, quotes, truth, initial, bounds, fit, generation,
        market, optimizer, seed, budget, start) if design == "short" and stage == "exploratory" else None)
    result = imported if imported is not None else joint._attempt("heston_curve", "noisy", start,
        stage, quotes, market, initial, bounds, truth, fit, budget, optimizer)
    result["summary"].setdefault("prior_profile_reused", False)
    source_id = result["summary"]["attempt_id"]
    for key in ("summary", "parameters", "predictions", "refinement"):
        rows = [result[key]] if key == "summary" else result[key]
        for row in rows:
            row.update(design=design, source_attempt_id=source_id,
                       attempt_id=f"maturity_{design}_{start}_{stage}")
    return result


def _acceptance():
    return {"price_atol": joint.PRICE_ATOL, "price_rtol": joint.PRICE_RTOL,
        "delta_atol": joint.DELTA_ATOL, "delta_rtol": joint.DELTA_RTOL,
        "spread_refinement_fraction": joint.SPREAD_REFINEMENT_FRACTION}


def _validate(fit, generation, budget, seed, reuse, designs, starts, compute_figures, run_calibration):
    joint._validate_settings(fit, generation, fit, budget, budget, seed, True, reuse)
    if designs is None:
        designs = DESIGNS
    if (not isinstance(designs, (list, tuple)) or not designs
            or any(not isinstance(name, str) or name not in DESIGNS for name in designs)
            or len(set(designs)) != len(designs)):
        raise ValueError("designs must be distinct short/augmented names")
    if starts is None:
        starts = (1, 2, 3)
    if (not isinstance(starts, (list, tuple)) or not starts
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                   or value not in (1, 2, 3) for value in starts)
            or len(set(starts)) != len(starts)):
        raise ValueError("start_numbers must be distinct integers from 1, 2, 3")
    if any(not isinstance(value, (bool, np.bool_)) for value in (compute_figures, run_calibration)):
        raise ValueError("compute_figures and run_calibration must be boolean")
    return tuple(designs), tuple(starts)


def maturity_extension(context, *, seed=42, fit_config=None, generation_config=None,
        max_nfev=80, reuse=True, designs=None, start_numbers=None, compute_figures=True,
        run_calibration=True):
    """Compare identical short rows against an augmented 90/180/365-day surface.

    Filters permit disjoint workers and are excluded from numerical keys.
    Prefill shared observations, accuracy and local geometry with
    ``run_calibration=False`` before running parallel start workers. Selection
    and a same-grid parameter restart occur only in a call requesting all three
    starts, and use training objectives alone.
    """
    begin = time.perf_counter()
    fit = FDConfig(192, 64, 180, spot_concentration=.05) if fit_config is None else fit_config
    generation = FDConfig(640, 128, 600, spot_concentration=.05) if generation_config is None else generation_config
    requested_designs, requested_starts = _validate(fit, generation, max_nfev, seed,
        reuse, designs, start_numbers, compute_figures, run_calibration)
    contracts, market = maturity_contracts(), joint.joint_market()
    truths, initial_starts, all_bounds = joint.joint_specifications()
    truth, starts, bounds = truths["heston_curve"], initial_starts["heston_curve"], all_bounds["heston_curve"]
    optimizer = OptimizerConfig(parameterization="asinh", ftol=1e-8, xtol=1e-8, gtol=1e-6)
    noise = np.random.default_rng(seed).normal(0., joint.NOISE_STD, len(contracts))
    numerical_source = "\n".join(inspect.getsource(function) for function in (
        maturity_contracts, _generate, _generation_or_preflight, _accuracy_or_preflight,
        _local_geometry, _old_short_attempt,
        _calibrate, _acceptance, joint.joint_contracts, joint.joint_market,
        joint.joint_specifications, joint.synthetic_quotes, joint._attempt,
        joint.select_training_attempt))
    common = {"study_version": STUDY_VERSION, "backend": "american_vanilla_fd",
        "data_kind": "synthetic", "exercise_style": "american",
        "engine_sha256": joint._engine_fingerprint(),
        "numeric_source_sha256": hashlib.sha256(numerical_source.encode()).hexdigest(),
        "geometry_source_sha256": hashlib.sha256(Path(__file__).with_name("_pricing_maturity_geometry.py").read_bytes()).hexdigest(),
        "accuracy_source_sha256": hashlib.sha256(Path(__file__).with_name("_pricing_maturity_accuracy.py").read_bytes()).hexdigest(),
        "seed": int(seed), "noise_std": joint.NOISE_STD, "noise": noise.tolist(),
        "linearized_noise_draws": LINEAR_NOISE_DRAWS, "linearized_noise_seed": LINEAR_NOISE_SEED,
        "contracts": contracts.to_dict("records"), "market": asdict(market),
        "truth": parameter_values(truth), "starts": [parameter_values(value) for value in starts],
        "bounds": bounds, "fit_config": asdict(fit), "generation_config": asdict(generation),
        "max_nfev": int(max_nfev), "restart_max_nfev": int(max_nfev),
        "optimizer": asdict(optimizer), "acceptance": _acceptance(),
        "spread_floor": joint.SPREAD_FLOOR,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = joint._fingerprint(common)
    output = getattr(context, "output_dir", None)
    directory = Path(output)/STUDY_VERSION/fingerprint[:16] if output is not None else None
    def cached(name, specification, compute):
        return joint._cached(directory/f"{name}.json" if directory is not None else None,
                             {**common, **specification}, compute, reuse)
    payloads, selections, failures, provenance = [], [], [], []
    observations, audit, geometry, quotes = None, {}, {}, None
    try:
        context.progress("Maturity extension: finer observations for 14/30/60/90/180/365 days")
        observations, reused = cached("observations", {},
            lambda: _generation_or_preflight(context, truth, contracts, market, generation, fit))
        provenance.append({"stage": "observation generation", "checkpoint_reused": reused})
        quotes = joint.synthetic_quotes(contracts, observations["prices"], noise, observations["deltas"])["noisy"]
        context.progress("Maturity extension: unchanged all-contract truth accuracy gate")
        observation_fingerprint = joint._fingerprint({key: observations[key] for key in ("prices", "deltas", "grid_sha256")})
        audit, reused = cached("truth_accuracy", {"observation_fingerprint": observation_fingerprint},
            lambda: _accuracy_or_preflight(context, truth, contracts, market, fit, observations))
        provenance.append({"stage": "truth accuracy", "checkpoint_reused": reused})
        if audit["accepted"]:
            context.progress("Maturity extension: frozen-grid two/three-point local geometry")
            geometry, reused = cached("local_geometry", {"observation_fingerprint": observation_fingerprint},
                lambda: _local_geometry(truth, contracts, quotes, market, fit, bounds))
            provenance.append({"stage": "local geometry", "checkpoint_reused": reused})
        else:
            failures.append({"stage": "truth accuracy gate", "error":
                "Calibration blocked: the declared common grid failed unchanged all-contract price, delta or spread checks."})
    except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        failures.append({"stage": "shared generation, accuracy or geometry", "error": f"{type(error).__name__}: {error}"})
    if run_calibration and audit.get("accepted") and geometry:
        for design in requested_designs:
            selected_quotes = quotes.iloc[:18].copy() if design == "short" else quotes.copy()
            design_payloads = []
            for start in requested_starts:
                initial = starts[start-1]
                context.progress(f"Maturity extension · {design} · start {start}: at most {max_nfev} evaluations")
                specification = {"design": design, "quotes": selected_quotes.to_dict("records"),
                    "start": int(start), "stage": "exploratory", "initial": parameter_values(initial)}
                result, reused = cached(f"{design}_{start}_exploratory", specification,
                    lambda: _calibrate(context, design, selected_quotes, truth, initial, bounds,
                        market, fit, generation, optimizer, seed, max_nfev, start, "exploratory"))
                result["summary"]["checkpoint_reused"] = reused
                design_payloads.append(result)
                payloads.append(result)
            if set(requested_starts) != {1, 2, 3}:
                continue
            chosen = joint.select_training_attempt([item["summary"] for item in design_payloads])
            restart = bool(chosen and not chosen["optimizer_success"])
            final = chosen
            if restart:
                initial = replace_parameters(truth, **chosen["params"])
                start = int(chosen["start"])
                context.progress(f"Maturity extension · {design}: same-grid parameter restart of training-selected start {start}")
                specification = {"design": design, "quotes": selected_quotes.to_dict("records"),
                    "start": start, "stage": "parameter_restart", "initial": parameter_values(initial),
                    "source_attempt_id": chosen["attempt_id"]}
                result, reused = cached(f"{design}_{start}_parameter_restart", specification,
                    lambda: _calibrate(context, design, selected_quotes, truth, initial, bounds,
                        market, fit, generation, optimizer, seed, max_nfev, start, "parameter_restart"))
                result["summary"]["checkpoint_reused"] = reused
                payloads.append(result)
                final = joint.select_training_attempt([chosen, result["summary"]])
            selections.append({"design": design, "selected_start": chosen["start"] if chosen else None,
                "initial_selected_attempt_id": chosen["attempt_id"] if chosen else None,
                "selected_attempt_id": final["attempt_id"] if final else None,
                "selected_training_objective": final["training_objective"] if final else None,
                "restart_executed": restart, "selection_rule": "Minimum training objective; deterministic start tie break",
                "heldout_used_for_selection": False, "complete_three_start_comparison": True})
    tables = _tables(payloads, selections, failures, contracts, quotes, geometry, audit, common, provenance)
    manifest = {**common, "run_fingerprint": fingerprint,
        "checkpoint_directory": str(directory) if directory is not None else None,
        "requested_designs": list(requested_designs), "requested_start_numbers": list(requested_starts),
        "run_calibration": bool(run_calibration), "truth_accuracy_accepted": bool(audit.get("accepted", False)),
        "observation_fingerprint": joint._fingerprint({key: observations[key] for key in ("prices", "deltas", "grid_sha256")}) if observations else None,
        "observations": observations, "geometry": geometry, "provenance": provenance,
        "attempts": tables["summary"].to_dict("records"), "failures": failures,
        **_timing_summary(tables["summary"]),
        "elapsed_seconds": time.perf_counter()-begin,
        "limitations": ["One actual nonlinear noise realization per design; seed is matched across their common rows.",
            "10,000 local algebraic draws are not repeated nonlinear calibration outcomes.",
            "Repeated-short control consists of hypothetical independent repeat measurements, not new contracts.",
            "Fixed strikes become closer to ATM in standardized moneyness at longer maturities.",
            "Synthetic schedule declares one cash dividend at .08 years and no others through one year.",
            "A parameter restart resets optimizer history; it does not isolate extra evaluations alone.",
            "No historical SPY readiness or hedge replay is established by synthetic numerical acceptance."]}
    return {"tables": tables, "figures": _figures(tables) if compute_figures else {}, "manifest": manifest}


def _timing_summary(summary):
    """Separate cumulative numerical work from this call's cache replay."""
    if summary.empty:
        return {"recorded_attempt_seconds": 0., "recorded_pricing_evaluations": 0,
            "current_attempt_seconds": 0., "current_pricing_evaluations": 0,
            "reused_attempt_count": 0, "executed_attempt_count": 0}
    newly_executed = ~summary.checkpoint_reused.fillna(False).astype(bool)
    prior_import = summary.get("prior_profile_reused", pd.Series(False, index=summary.index)).fillna(False).astype(bool)
    actual_new_work = newly_executed & ~prior_import
    return {"recorded_attempt_seconds": float(summary.elapsed_seconds.sum()),
        "recorded_pricing_evaluations": int(summary.pricing_evaluations.sum()),
        "current_attempt_seconds": float(summary.loc[actual_new_work, "elapsed_seconds"].sum()),
        "current_pricing_evaluations": int(summary.loc[actual_new_work, "pricing_evaluations"].sum()),
        "reused_attempt_count": int((~newly_executed).sum()),
        "imported_prior_attempt_count": int(prior_import.sum()),
        "executed_attempt_count": int(actual_new_work.sum()),
        "pricing_evaluations_complete": bool(summary.pricing_evaluations_complete.all()),
        "timing_scope": "Cumulative attempt times are not parallel wall time; shared generation/audits/geometry are separate."}


def _price_recovery_tables(predictions):
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = predictions.copy()
    frame["days"] = np.rint(frame.maturity*365).astype(int)
    frame["horizon_cohort"] = np.where(frame.days <= 60, "common_14_30_60", "added_90_180_365")
    measures = {"price_residual": "observed_rmse", "clean_price_error": "clean_rmse",
        "scaled_residual": "spread_normalized_rmse", "delta_error": "clean_delta_rmse",
        "refined_price_residual": "refined_observed_rmse", "refined_clean_price_error": "refined_clean_rmse"}
    results = []
    identity = ["design", "attempt_id", "start", "stage", "split"]
    for grouping in ("days", "horizon_cohort"):
        records = []
        for key, part in frame.groupby(identity+[grouping], sort=False):
            row = dict(zip(identity+[grouping], key))
            row["contract_count"] = len(part)
            for column, label in measures.items():
                if column in part:
                    row[label] = float(np.sqrt(np.mean(part[column].to_numpy(float)**2)))
            records.append(row)
        results.append(pd.DataFrame(records))
    return tuple(results)


def _parameter_recovery_table(parameters, geometry):
    if parameters.empty or geometry.empty:
        return pd.DataFrame()
    local = geometry.loc[geometry.design.isin(DESIGNS),
        ["design", "parameter", "local_noise_sd", "nuisance_projected_fraction"]]
    result = parameters.merge(local, on=["design", "parameter"], how="left", validate="many_to_one")
    result["parameter_error"] = result.fitted-result.generating
    result["error_in_local_noise_sd"] = result.parameter_error/result.local_noise_sd
    result["interpretation"] = "Descriptive local scaling; one nonlinear noise draw, finite-grid bias, bounds and nonlinearity prevent a z-test."
    return result


def _tables(payloads, selections, failures, contracts, quotes, geometry, audit, settings, provenance):
    tables = {name: pd.DataFrame([row for item in payloads for row in item[name]])
              for name in ("parameters", "predictions", "refinement")}
    geometry_rows = []
    for design, record in geometry.get("geometries", {}).items():
        for row in record["sensitivities"]:
            geometry_rows.append({"design": design, **row, "contract_count": record["contract_count"],
                "rank": record["rank"], "column_normalized_condition_number": record["column_normalized_condition_number"],
                "max_relative_two_three_point_difference": record["max_relative_two_three_point_difference"]})
    tables.update(summary=joint._reporting_summary(payloads), selection=pd.DataFrame(selections),
        geometry=pd.DataFrame(geometry_rows), noise=pd.DataFrame(geometry.get("linearized_noise", [])),
        accuracy=pd.DataFrame(audit.get("accuracy", [])), accuracy_refinement=pd.DataFrame(audit.get("refinement", [])),
        accuracy_contracts=pd.DataFrame(audit.get("contracts", [])),
        by_maturity=pd.DataFrame(geometry.get("by_maturity", [])),
        accuracy_by_maturity=pd.DataFrame(audit.get("by_maturity", [])),
        contracts=contracts, observations=quotes if quotes is not None else pd.DataFrame(),
        failures=pd.DataFrame(failures), provenance=pd.DataFrame(provenance),
        settings=pd.DataFrame([{"setting": key, "value": json.dumps(joint._jsonable(value), sort_keys=True)}
            for key, value in settings.items() if key not in ("contracts", "noise")]))
    tables["price_recovery"], tables["common_horizon_recovery"] = _price_recovery_tables(tables["predictions"])
    tables["parameter_recovery"] = _parameter_recovery_table(tables["parameters"], tables["geometry"])
    return tables


def _figures(tables):
    import matplotlib.pyplot as plt
    figures = {}
    geometry = tables["geometry"]
    if not geometry.empty:
        names = list(dict.fromkeys(geometry.parameter))
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for design, group in geometry.groupby("design", sort=False):
            rows = group.set_index("parameter").reindex(names)
            axes[0].plot(names, rows.local_noise_sd, "o-", label=design.replace("_", " "))
            axes[1].plot(names, rows.nuisance_projected_fraction, "o-", label=design.replace("_", " "))
        axes[0].set(yscale="log", ylabel="Local parameter SD under $0.005 noise", title="Physical-noise sandwich propagation")
        axes[1].set(yscale="log", ylabel="Sensitivity remaining after nuisance compensation", title="Information beyond more quotes")
        for axis in axes:
            axis.tick_params(axis="x", rotation=25)
            axis.legend(fontsize=8)
        figure.suptitle("Longer maturities: local geometry at the generating parameters")
        figures["identification"] = figure
    summary, parameters = tables["summary"], tables["parameters"]
    if not summary.empty and not parameters.empty:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        names = list(dict.fromkeys(parameters.parameter))
        colors = {"short": "#b46b43", "augmented": "#31798f"}
        for (design, start, stage), group in parameters.groupby(["design", "start", "stage"], sort=False):
            rows = group.set_index("parameter").reindex(names)
            attempt = summary.loc[summary.attempt_id.eq(group.attempt_id.iloc[0])].iloc[0]
            status = "" if attempt.status == "converged" else f" ({attempt.status})"
            axes[0].plot(names, (rows.fitted-rows.generating)/(rows.upper-rows.lower),
                "o--" if stage == "exploratory" else "o-", color=colors[design],
                alpha=.6 if stage == "exploratory" else 1., label=f"{design} · {start}"+(" restart" if stage != "exploratory" else "")+status)
        axes[0].axhline(0., color="black", lw=.8)
        axes[0].set(ylabel="(Estimate − truth) / bound width", title="All starts; bounds and failures remain visible")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].legend(fontsize=7, ncols=2)
        x = np.arange(len(summary))
        common = tables["common_horizon_recovery"]
        if common.empty:
            common = pd.DataFrame(index=summary.attempt_id, columns=("train", "heldout"), dtype=float)
            axes[1].text(.5, .5, "No completed price comparisons", transform=axes[1].transAxes, ha="center")
        else:
            common = common.loc[common.horizon_cohort.eq("common_14_30_60")]
            common = common.pivot(index="attempt_id", columns="split", values="clean_rmse").reindex(summary.attempt_id)
        axes[1].bar(x-.18, common.get("train", pd.Series(np.nan, index=common.index)), width=.35, label="Training")
        axes[1].bar(x+.18, common.get("heldout", pd.Series(np.nan, index=common.index)), width=.35, label="Held out")
        axes[1].set(xticks=x, xticklabels=[f"{row.design}\n{row.start}"+(" R" if row.stage != "exploratory" else "") for row in summary.itertuples()],
            ylabel="RMSE versus finer clean prices ($/share)", title="Price recovery on identical 14/30/60-day contracts")
        axes[1].legend(fontsize=8)
        figure.suptitle("Matched noisy American Heston-curve calibration")
        figures["recovery"] = figure
    return figures
