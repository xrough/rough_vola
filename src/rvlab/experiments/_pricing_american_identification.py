"""Reproducible joint American Heston calibration, without market-data access.

The short-tenor surface intentionally resembles the SPY pilot's geometry. It
tests numerical calibration and price recovery, not uniqueness of coefficients.
Each expensive attempt is checkpointed using its exact numerical specification.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

from rvlab.pricing import (
    CashDividend, DividendSchedule, DiscountCurve, FDConfig, HestonCurveParams,
    HestonParams, MeanReversionCurve, OptimizerConfig, PricingError, SpotMarket,
    calibrate_vanilla_surface, evaluate_vanilla, parameter_values,
    replace_parameters,
)
from rvlab.pricing._vanilla_common import build_grid


STUDY_VERSION = "american_joint_identification_v2_spot_accuracy"
MODEL_LABELS = {"heston": "Heston", "heston_curve": "Heston curve"}
SPREAD_FLOOR = .005
NOISE_STD = .005
PRICE_ATOL = .001
PRICE_RTOL = 1e-4
DELTA_ATOL = .002
DELTA_RTOL = .005
SPREAD_REFINEMENT_FRACTION = .1
MAX_ATTEMPT_NFEV = 160


def joint_contracts():
    """Fixed 12/6 train/held-out split; neither quotes nor starts select rows."""
    return pd.DataFrame([
        {"contract_id": f"american-joint-{days}-{strike}", "maturity": days/365,
         "strike": float(strike), "is_call": strike >= 100,
         "split": "heldout" if strike in (97, 103) else "train",
         "half_spread": (.003, .008, .012, .018, .025, .035)[j]*(1+.25*i)}
        for i, days in enumerate((14, 30, 60))
        for j, strike in enumerate((94, 97, 99, 101, 103, 106))
    ])


def joint_market():
    return SpotMarket(100., DiscountCurve.flat(.04, 1.),
                      DividendSchedule((CashDividend(.08, 1.5),), 1.))


def joint_specifications():
    """All five/seven coefficients free; calendar-time knots remain structural."""
    curve = lambda v0, kappa, eta, rho, levels: HestonCurveParams(
        v0, kappa, eta, rho, MeanReversionCurve((20/365, 45/365), tuple(levels)))
    truths = {
        "heston": HestonParams(.04, .045, 1.5, .3, -.6),
        "heston_curve": curve(.04, 1.5, .3, -.6, (.035, .065, .045)),
    }
    starts = {
        "heston": [HestonParams(.055, .035, .7, .2, -.35),
                    HestonParams(.025, .08, 3., .5, -.8),
                    HestonParams(.045, .055, 1.8, .35, -.55)],
        "heston_curve": [curve(.055, .7, .2, -.35, (.055, .035, .08)),
                          curve(.025, 3., .5, -.8, (.08, .025, .065)),
                          curve(.045, 1.8, .35, -.55, (.045, .055, .055))],
    }
    bounds = {model: {name: ((.1, 5.) if name == "kappa" else
                            (.01, .6) if name == "vol_of_vol" else
                            (-.95, 0.) if name == "rho" else (.0025, .25))
                      for name in parameter_values(truth)}
              for model, truth in truths.items()}
    return truths, starts, bounds


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fingerprint(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _engine_fingerprint():
    directory = Path(__file__).resolve().parents[1]/"pricing"
    digest = hashlib.sha256()
    for name in ("_vanilla_heston.py", "_vanilla_bs.py", "_vanilla_common.py", "vanilla.py",
                 "vanilla_types.py", "vanilla_calibration.py", "optimization.py",
                 "parameters.py", "variance_curve.py", "types.py"):
        digest.update(name.encode())
        digest.update((directory/name).read_bytes())
    return digest.hexdigest()


def _cached(path, specification, compute, reuse):
    """Reuse only exact-specification evidence, including unsuccessful attempts."""
    key = _fingerprint(specification)
    if path is not None and reuse and path.exists():
        record = json.loads(path.read_text())
        if record.get("fingerprint") == key:
            return record["payload"], True
    payload = _jsonable(compute())
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"fingerprint": key,
            "specification": _jsonable(specification), "payload": payload},
            indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    return payload, False


def _validate_settings(fit, generation, continuation, max_nfev, continuation_max_nfev,
                       seed, run_continuation, reuse):
    for name, value in (("seed", seed), ("max_nfev", max_nfev),
                        ("continuation_max_nfev", continuation_max_nfev)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
    if max_nfev > MAX_ATTEMPT_NFEV or continuation_max_nfev > MAX_ATTEMPT_NFEV:
        raise ValueError(f"each joint American attempt is bounded at {MAX_ATTEMPT_NFEV} optimizer evaluations")
    if not isinstance(run_continuation, (bool, np.bool_)) or not isinstance(reuse, (bool, np.bool_)):
        raise ValueError("run_continuation and reuse_checkpoints must be boolean")
    if any(not isinstance(c, FDConfig) for c in (fit, generation, continuation)):
        raise TypeError("joint calibration requires FDConfig settings")
    for candidate in (fit, continuation) if run_continuation else (fit,):
        if (generation.spot_steps <= candidate.spot_steps or
                generation.variance_steps <= candidate.variance_steps or
                generation.time_steps <= candidate.time_steps or
                generation.spot_max_multiple < candidate.spot_max_multiple or
                generation.variance_max < candidate.variance_max):
            raise ValueError("observation generation must be strictly finer than every fitting grid with no smaller domains")


def synthetic_quotes(contracts, clean_prices, noise, clean_deltas=None):
    """Produce one clean/noisy pair without clipping prices, bids or noise."""
    clean = np.asarray(clean_prices, float)
    noise = np.asarray(noise, float)
    if clean.shape != (len(contracts),) or noise.shape != clean.shape or not np.all(np.isfinite(clean+noise)):
        raise ValueError("generation and noise must be finite and contract aligned")
    result = {}
    for name, perturbation in (("clean", np.zeros(len(clean))), ("noisy", noise)):
        frame = contracts.assign(clean_price=clean, noise=perturbation,
            market_price=clean+perturbation, exercise_style="american")
        if clean_deltas is not None:
            deltas = np.asarray(clean_deltas, float)
            if deltas.shape != clean.shape or not np.all(np.isfinite(deltas)):
                raise ValueError("generating deltas must be finite and contract aligned")
            frame["clean_delta"] = deltas
        frame["bid"] = frame.market_price-frame.half_spread
        frame["ask"] = frame.market_price+frame.half_spread
        frame["effective_half_spread"] = np.maximum(frame.half_spread, SPREAD_FLOOR)
        intrinsic = np.maximum(np.where(frame.is_call, 100-frame.strike, frame.strike-100), 0.)
        bad = (frame.bid < 0) | (frame.market_price < intrinsic)
        if bad.any():
            identifiers = frame.loc[bad, "contract_id"].tolist()
            raise ValueError(f"{name} synthetic observations violate bid/intrinsic constraints at {identifiers}; observations were not clipped")
        result[name] = frame
    return result


def select_training_attempt(rows):
    """Training objectives alone select a continuation, including exhausted fits."""
    eligible = [row for row in rows if row.get("params") is not None and
                row.get("training_objective") is not None and
                np.isfinite(row["training_objective"])]
    return min(eligible, key=lambda row: (row["training_objective"], row["start"])) if eligible else None


def _empty_attempt(model, dataset, start, stage, initial, bounds, truth, config, budget):
    return {"attempt_id": f"{model}_{dataset}_{start}_{stage}", "model": model,
        "dataset": dataset, "start": start, "stage": stage, "status": "numerical_failure",
        "optimizer_success": False, "price_stable": False, "delta_stable": False,
        "spread_stable": False, "ready": False, "hedge_delta_ready": False,
        "initial_params": parameter_values(initial), "params": None,
        "generating_params": parameter_values(truth), "bounds": bounds,
        "free_parameters": list(parameter_values(initial)),
        "free_parameter_count": len(parameter_values(initial)), "fixed_parameters": {},
        "config": asdict(config), "max_nfev": budget, "nfev": 0, "njev": None,
        "pricing_evaluations": 0, "active_bounds": {}, "active_bound_count": 0,
        "training_objective": None, "train_rmse": None, "test_rmse": None,
        "train_scaled_rmse": None, "test_scaled_rmse": None,
        "train_clean_rmse": None, "test_clean_rmse": None,
        "max_price_change": None, "max_delta_change": None,
        "physical_optimality": None, "optimality": None,
        "error": None, "elapsed_seconds": 0.}


def _attempt(model, dataset, start, stage, quotes, market, initial, bounds,
             truth, config, budget, optimizer):
    started = time.perf_counter()
    row = _empty_attempt(model, dataset, start, stage, initial, bounds, truth, config, budget)
    parameters, prediction, checks = [], pd.DataFrame(), []
    try:
        training = quotes.loc[quotes.split.eq("train")]
        fit = calibrate_vanilla_surface(training, initial, market=market,
            exercise_style="american", bounds=bounds, config=config, max_nfev=budget,
            optimizer=optimizer, price_scale_floor=SPREAD_FLOOR,
            stability_atol=PRICE_ATOL, stability_rtol=PRICE_RTOL,
            delta_atol=DELTA_ATOL, delta_rtol=DELTA_RTOL,
            spread_refinement_fraction=SPREAD_REFINEMENT_FRACTION)
        row.update(optimizer_success=fit.optimizer_success, params=parameter_values(fit.params),
            nfev=fit.nfev, njev=fit.njev, pricing_evaluations=fit.pricing_evaluations,
            optimizer_status=fit.optimizer_status, optimizer_message=fit.optimizer_message,
            optimizer_settings=dict(fit.optimizer_settings), active_bounds=dict(fit.active_bounds),
            active_bound_count=len(fit.active_bounds), training_objective=fit.objective,
            optimality=fit.optimality, physical_optimality=fit.physical_optimality,
            fixed_parameters=dict(fit.fixed_parameters),
            training_grid_sha256=(fit.diagnostics.get("grid_fingerprints") or [None])[0],
            frozen_grid=bool(fit.diagnostics.get("fixed_grid", False)))
        checks.extend(fit.diagnostics["checks"].assign(coverage="training").to_dict("records"))
        for name, value in parameter_values(fit.params).items():
            parameters.append({"parameter": name, "generating": parameter_values(truth)[name],
                "initial": parameter_values(initial)[name], "fitted": value,
                "lower": bounds[name][0], "upper": bounds[name][1], "fixed": False,
                "active_bound": dict(fit.active_bounds).get(name)})
        # Retain exactly the training mesh during held-out base pricing. Merely
        # inserting validation strikes must not change the fitted objective.
        grid = build_grid(market, training.strike.to_numpy(), config, variance=True)
        values = evaluate_vanilla(fit.params, market=market, strike=quotes.strike.to_numpy(),
            maturity=quotes.maturity.to_numpy(), is_call=quotes.is_call.to_numpy(),
            exercise_style="american", config=config, grid=grid, check_refinement=True,
            price_atol=PRICE_ATOL, price_rtol=PRICE_RTOL,
            delta_atol=DELTA_ATOL, delta_rtol=DELTA_RTOL)
        row["pricing_evaluations"] += values.diagnostics["pricing_evaluations"]
        validation_checks = values.diagnostics["checks"].copy()
        scales = quotes.effective_half_spread.to_numpy()
        spread_pass = True
        for axis, change in values.diagnostics["price_changes"].items():
            passed = bool(np.all(np.asarray(change) <= SPREAD_REFINEMENT_FRACTION*scales))
            validation_checks.loc[validation_checks.axis.eq(axis), "spread_stable"] = passed
            spread_pass &= passed
        checks.extend(validation_checks.assign(coverage="training and held-out").to_dict("records"))
        prediction = quotes.assign(model_price=values.prices, delta=values.deltas,
            refined_model_price=values.diagnostics["reference_prices"],
            refined_delta=values.diagnostics["reference_deltas"])
        prediction["price_residual"] = prediction.model_price-prediction.market_price
        prediction["scaled_residual"] = prediction.price_residual/scales
        prediction["clean_price_error"] = prediction.model_price-prediction.clean_price
        prediction["refined_price_residual"] = prediction.refined_model_price-prediction.market_price
        prediction["refined_clean_price_error"] = prediction.refined_model_price-prediction.clean_price
        if "clean_delta" in prediction:
            prediction["delta_error"] = prediction.delta-prediction.clean_delta
            prediction["refined_delta_error"] = prediction.refined_delta-prediction.clean_delta
        row["train_reprice_difference"] = float(np.max(np.abs(
            prediction.loc[prediction.split.eq("train"), "model_price"].to_numpy()-fit.fitted_quotes.model_price.to_numpy())))
        row["validation_grid_matches_training"] = values.diagnostics["grid_sha256"] == row["training_grid_sha256"]
        row.update(price_stable=bool(fit.price_stable and values.price_stable and spread_pass),
            delta_stable=bool(fit.delta_stable and values.delta_stable),
            spread_stable=bool(fit.diagnostics["spread_stable"] and spread_pass),
            max_price_change=max(fit.max_price_change, float(validation_checks.max_price_change.max())),
            max_delta_change=max(fit.max_delta_change, float(validation_checks.max_delta_change.max())))
        row["ready"] = bool(row["optimizer_success"] and row["price_stable"])
        row["hedge_delta_ready"] = bool(row["ready"] and row["delta_stable"])
        row["status"] = ("converged" if row["ready"] else "converged_unstable"
                         if row["optimizer_success"] else "exhausted"
                         if fit.optimizer_status == 0 else "optimizer_failed")
        for split, prefix in (("train", "train"), ("heldout", "test")):
            part = prediction.loc[prediction.split.eq(split)]
            for column, suffix in (("price_residual", "rmse"), ("scaled_residual", "scaled_rmse"),
                    ("clean_price_error", "clean_rmse"), ("refined_price_residual", "refined_rmse"),
                    ("refined_clean_price_error", "refined_clean_rmse")):
                row[f"{prefix}_{suffix}"] = float(np.sqrt(np.mean(part[column]**2)))
            if "delta_error" in part:
                row[f"{prefix}_delta_rmse"] = float(np.sqrt(np.mean(part.delta_error**2)))
                row[f"{prefix}_refined_delta_rmse"] = float(np.sqrt(np.mean(part.refined_delta_error**2)))
    except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row.update(status="numerical_failure", ready=False, hedge_delta_ready=False,
                   error=f"{type(exc).__name__}: {exc}")
    row["elapsed_seconds"] = time.perf_counter()-started
    metadata = {key: row[key] for key in ("attempt_id", "model", "dataset", "start", "stage")}
    return {"summary": row, "parameters": [{**metadata, **item} for item in parameters],
        "predictions": prediction.assign(**metadata).to_dict("records") if not prediction.empty else [],
        "refinement": [{**metadata, **item} for item in checks]}


def _findings(summary, selection):
    rows = []
    for (model, dataset), group in summary.groupby(["model", "dataset"], sort=False):
        coarse = group.loc[group.stage.eq("exploratory")]
        finite = coarse.loc[coarse.training_objective.notna()]
        fine = group.loc[group.stage.eq("continuation")]
        selected = fine.iloc[0] if len(fine) else finite.sort_values(["training_objective", "start"]).iloc[0] if len(finite) else None
        sentence = (f"{int(coarse.optimizer_success.sum())}/{len(coarse)} initial optimizers converged; "
            f"{int(coarse.price_stable.sum())}/{len(coarse)} passed independent price/spread checks; "
            f"{int(coarse.delta_stable.sum())}/{len(coarse)} passed delta checks; "
            f"{int(coarse.ready.sum())}/{len(coarse)} were accepted including optimizer convergence.")
        if selected is not None:
            stage_label = "initial" if selected.stage == "exploratory" else "continuation"
            sentence += (f" Training-selected {stage_label} start {int(selected.start)}: "
                f"held-out clean-price RMSE ${selected.test_clean_rmse:.6f}; "
                f"status {selected.status}, price stable={bool(selected.price_stable)}, "
                f"delta stable={bool(selected.delta_stable)}.") if pd.notna(selected.test_clean_rmse) else f" Selected trial failed: {selected.error}."
        rows.append({"model": model, "dataset": dataset, "finding": sentence})
    rows.append({"model": "all", "dataset": "all", "finding":
        "Short maturities weakly identify kappa and theta levels; close prices and low objectives do not imply unique parameter recovery. Neither held-out errors nor refinement results select the initial start."})
    rows.append({"model": "all", "dataset": "noisy", "finding":
        "Gaussian noise has standard deviation $0.005/share. Declared heterogeneous spreads are research residual weights, not estimates of the noise variance; the $0.005 floor is active."})
    return pd.DataFrame(rows)


def _reporting_summary(payloads):
    """Annotate saved evidence without changing cached numerical payloads.

    A calibration exception cannot return the optimizer's internal pricing
    counter. A validation exception retains the known completed calibration
    count, which is explicitly incomplete. Neither case means zero work.
    """
    summary = pd.DataFrame([dict(item["summary"]) for item in payloads])
    if summary.empty:
        return summary
    failed = summary.status.eq("numerical_failure")
    summary["pricing_evaluations_complete"] = ~failed
    unknown = failed & pd.to_numeric(summary.pricing_evaluations, errors="coerce").fillna(0).eq(0)
    summary.loc[unknown, "pricing_evaluations"] = np.nan
    return summary


def _spot_accuracy(context, model, truth, contracts, market, observed, fit):
    from ._pricing_spot_accuracy import model_spot_accuracy
    return model_spot_accuracy(context, model, truth, contracts, market,
        observed["prices"], observed["deltas"], fit)


def _spot_accuracy_source_sha256():
    return hashlib.sha256(Path(__file__).with_name("_pricing_spot_accuracy.py").read_bytes()).hexdigest()


def american_identification_study(context, *, seed=42, fit_config=None,
        generation_config=None, continuation_config=None, max_nfev=80,
        continuation_max_nfev=160, run_continuation=True, reuse_checkpoints=True,
        models=None, datasets=None, compute_figures=True, run_calibration=True):
    """Twelve joint fits and up to four selected budget continuations.

    Starts, failures, optimization exhaustion and failed refinement are retained.
    All five constant-model and seven curve coefficients are genuinely free.
    Already-converged selected fits do not repeat optimization. ``models`` can
    restrict execution to one model without changing its checkpoint keys; this
    permits separate workers to populate one complete reproducible study.
    ``datasets`` similarly restricts clean/noisy execution, using the same
    deterministic observations, weights and per-attempt checkpoint keys.
    """
    fit = FDConfig(192, 48, 180, spot_concentration=.05) if fit_config is None else fit_config
    generation = FDConfig(640, 128, 600, spot_concentration=.05) if generation_config is None else generation_config
    # Continuing on exactly the same objective isolates extra optimizer work
    # from the separate independent spatial/time/domain refinement checks.
    continuation = fit if continuation_config is None else continuation_config
    _validate_settings(fit, generation, continuation, max_nfev, continuation_max_nfev,
                       seed, run_continuation, reuse_checkpoints)
    if models is None:
        requested_models = tuple(MODEL_LABELS)
    elif (not isinstance(models, (tuple, list)) or not models or
          any(not isinstance(model, str) or model not in MODEL_LABELS for model in models) or
          len(set(models)) != len(models)):
        raise ValueError("models must be a nonempty sequence of distinct heston/heston_curve names")
    else:
        requested_models = tuple(models)
    if datasets is None:
        requested_datasets = ("clean", "noisy")
    elif (not isinstance(datasets, (tuple, list)) or not datasets or
          any(not isinstance(dataset, str) or dataset not in ("clean", "noisy") for dataset in datasets) or
          len(set(datasets)) != len(datasets)):
        raise ValueError("datasets must be a nonempty sequence of distinct clean/noisy names")
    else:
        requested_datasets = tuple(datasets)
    for name, value in (("compute_figures", compute_figures), ("run_calibration", run_calibration)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    started = time.perf_counter()
    contracts, market = joint_contracts(), joint_market()
    truths, starts, bounds = joint_specifications()
    optimizer = OptimizerConfig(parameterization="asinh", ftol=1e-8, xtol=1e-8, gtol=1e-6)
    noise = np.random.default_rng(seed).normal(0., NOISE_STD, len(contracts))
    output_directory = getattr(context, "output_dir", None)
    engine = _engine_fingerprint()
    numeric_source = "\n".join(inspect.getsource(function) for function in (
        joint_contracts, joint_market, joint_specifications, synthetic_quotes,
        select_training_attempt, _attempt, _spot_accuracy, american_identification_study))
    common = {"study_version": STUDY_VERSION, "engine_sha256": engine,
        "experiment_numerics_sha256": hashlib.sha256(numeric_source.encode()).hexdigest(), "seed": int(seed),
        "contracts": contracts.to_dict("records"), "market": asdict(market),
        "noise_std": NOISE_STD, "noise": noise.tolist(), "spread_floor": SPREAD_FLOOR,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pandas": pd.__version__},
        "acceptance": {"price_atol": PRICE_ATOL, "price_rtol": PRICE_RTOL,
            "delta_atol": DELTA_ATOL, "delta_rtol": DELTA_RTOL,
            "spread_refinement_fraction": SPREAD_REFINEMENT_FRACTION}}
    # A new grid, budget, engine or library version starts a separate directory.
    # In particular, neither this rerun nor later experiments overwrite the
    # original v1 trials or an earlier completed v2 numerical specification.
    run_specification = {**common, "truths": {name: parameter_values(value) for name, value in truths.items()},
        "starts": {name: [parameter_values(value) for value in values] for name, values in starts.items()},
        "bounds": bounds, "fit_config": asdict(fit), "generation_config": asdict(generation),
        "continuation_config": asdict(continuation), "max_nfev": int(max_nfev),
        "continuation_max_nfev": int(continuation_max_nfev), "run_continuation": bool(run_continuation),
        "optimizer": asdict(optimizer)}
    run_fingerprint = _fingerprint(run_specification)
    directory = (Path(output_directory)/STUDY_VERSION/run_fingerprint[:16]
                 if output_directory is not None else None)
    payloads, selection, failures, generations, baselines = [], [], [], [], []
    spot_accuracy, spot_refinement, spot_errors = [], [], []
    for model in requested_models:
        truth = truths[model]
        context.progress(f"Joint American {MODEL_LABELS[model]}: generating finer-grid clean observations")
        spec = {**common, "model": model, "truth": parameter_values(truth),
                "generation_config": asdict(generation)}
        def generate():
            begin = time.perf_counter()
            values = evaluate_vanilla(truth, market=market, strike=contracts.strike.to_numpy(),
                maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
                exercise_style="american", config=generation)
            return {"prices": values.prices.tolist(), "deltas": values.deltas.tolist(),
                    "elapsed_seconds": time.perf_counter()-begin,
                    "grid_sha256": values.diagnostics["grid_sha256"]}
        try:
            observed, cached = _cached(None if directory is None else directory/f"{model}_observations.json",
                                      spec, generate, reuse_checkpoints)
            generations.append({"model": model, "cached": cached, **observed})
            quote_sets = synthetic_quotes(contracts, observed["prices"], noise, observed["deltas"])
        except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append({"model": model, "stage": "observation generation", "error": f"{type(exc).__name__}: {exc}"})
            continue
        # Accuracy must be demonstrated at the generating coefficients before
        # a parameter search can compensate for discretization bias. This is
        # a numerical screen only: no held-out market error selects a start.
        accuracy_sha = _spot_accuracy_source_sha256()
        accuracy_spec = {**spec, "fit_config": asdict(fit),
            "observations_fingerprint": _fingerprint(observed["prices"]),
            "accuracy_source_sha256": accuracy_sha}
        try:
            accuracy, cached = _cached(None if directory is None else
                directory/f"{model}_spot_accuracy_{accuracy_sha[:12]}.json", accuracy_spec,
                lambda: _spot_accuracy(context, model, truth, contracts, market, observed, fit),
                reuse_checkpoints)
            spot_accuracy.extend({**row, "checkpoint_reused": cached} for row in accuracy["accuracy"])
            spot_refinement.extend({**row, "checkpoint_reused": cached} for row in accuracy["refinement"])
            spot_errors.extend({**row, "checkpoint_reused": cached} for row in accuracy["contracts"])
            accuracy_accepted = bool(accuracy["accepted"])
        except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            accuracy_accepted = False
            failures.append({"model": model, "stage": "truth-grid accuracy audit", "error": f"{type(exc).__name__}: {exc}"})
        # Before optimization, measure the discretization bias at the actual
        # generating coefficients. Joint fitting can otherwise hide that bias
        # by moving model parameters away from their generating values.
        for stage, settings in (("exploratory", fit), ("continuation", continuation)):
            if stage == "continuation" and not run_continuation:
                continue
            baseline_spec = {**spec, "stage": stage, "config": asdict(settings)}
            def baseline():
                begin = time.perf_counter()
                grid = build_grid(market, contracts.loc[contracts.split.eq("train"), "strike"].to_numpy(),
                                  settings, variance=True)
                values = evaluate_vanilla(truth, market=market, strike=contracts.strike.to_numpy(),
                    maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
                    exercise_style="american", config=settings, grid=grid)
                rows = []
                errors = np.asarray(values.prices)-np.asarray(observed["prices"])
                scales = np.maximum(contracts.half_spread.to_numpy(), SPREAD_FLOOR)
                for split in ("train", "heldout"):
                    mask = contracts.split.eq(split).to_numpy()
                    rows.append({"model": model, "stage": stage, "split": split,
                        "rmse": float(np.sqrt(np.mean(errors[mask]**2))),
                        "scaled_rmse": float(np.sqrt(np.mean((errors[mask]/scales[mask])**2))),
                        "max_price_error": float(np.max(np.abs(errors[mask]))),
                        "elapsed_seconds": time.perf_counter()-begin,
                        "grid_sha256": values.diagnostics["grid_sha256"]})
                return {"rows": rows}
            try:
                result, cached = _cached(None if directory is None else directory/f"{model}_{stage}_truth_baseline.json",
                                        baseline_spec, baseline, reuse_checkpoints)
                baselines.extend([{**row, "checkpoint_reused": cached} for row in result["rows"]])
            except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                failures.append({"model": model, "stage": f"{stage} truth baseline", "error": f"{type(exc).__name__}: {exc}"})
        if not accuracy_accepted:
            failures.append({"model": model, "stage": "calibration blocked by truth-grid accuracy",
                "error": "The selected fitting grid failed unchanged price, delta or spread refinement tolerances at the generating parameters."})
            continue
        if not run_calibration:
            continue
        for dataset, quotes in quote_sets.items():
            if dataset not in requested_datasets:
                continue
            dataset_payloads = []
            for start, initial in enumerate(starts[model], 1):
                context.progress(f"Joint American {MODEL_LABELS[model]} · {dataset} · start {start}/3: at most {max_nfev} evaluations")
                attempt_spec = {**spec, "dataset": dataset, "quotes": quotes.to_dict("records"),
                    "start": start, "initial": parameter_values(initial), "bounds": bounds[model],
                    "fit_config": asdict(fit), "max_nfev": int(max_nfev), "optimizer": asdict(optimizer)}
                result, cached = _cached(None if directory is None else directory/f"{model}_{dataset}_{start}_exploratory.json",
                    attempt_spec, lambda: _attempt(model, dataset, start, "exploratory", quotes,
                        market, initial, bounds[model], truth, fit, max_nfev, optimizer), reuse_checkpoints)
                result["summary"]["checkpoint_reused"] = cached
                dataset_payloads.append(result)
                payloads.append(result)
            chosen = select_training_attempt([part["summary"] for part in dataset_payloads])
            continue_chosen = bool(run_continuation and chosen and not chosen["optimizer_success"])
            selected = {"model": model, "dataset": dataset, "selection_rule": "Minimum exploratory training objective; deterministic start-number tie break",
                "selected_start": chosen["start"] if chosen else None,
                "selected_training_objective": chosen["training_objective"] if chosen else None,
                "selected_status": chosen["status"] if chosen else "no finite fitted attempt",
                "heldout_used_for_selection": False, "continuation_executed": continue_chosen,
                "continuation_reason": ("Selected optimizer has not converged" if continue_chosen else
                    "Selected optimizer already converged" if chosen and chosen["optimizer_success"] else
                    "No finite fitted attempt" if chosen is None else "Continuation disabled")}
            selection.append(selected)
            if continue_chosen:
                initial = replace_parameters(truth, **chosen["params"])
                start = int(chosen["start"])
                context.progress(f"Joint American {MODEL_LABELS[model]} · {dataset}: budget continuation of training-selected start {start}")
                continuation_spec = {**spec, "dataset": dataset, "quotes": quotes.to_dict("records"),
                    "start": start, "initial": parameter_values(initial), "bounds": bounds[model],
                    "fit_config": asdict(continuation), "max_nfev": int(continuation_max_nfev),
                    "optimizer": asdict(optimizer), "selection": selected}
                result, cached = _cached(None if directory is None else directory/f"{model}_{dataset}_{start}_continuation.json",
                    continuation_spec, lambda: _attempt(model, dataset, start, "continuation", quotes,
                        market, initial, bounds[model], truth, continuation, continuation_max_nfev, optimizer), reuse_checkpoints)
                result["summary"]["checkpoint_reused"] = cached
                payloads.append(result)
    tables = {name: pd.DataFrame([row for item in payloads for row in item[name]])
              for name in ("parameters", "predictions", "refinement")}
    summary = _reporting_summary(payloads)
    selections = pd.DataFrame(selection)
    tables.update(summary=summary, selection=selections, contracts=contracts,
        truth_baseline=pd.DataFrame(baselines),
        spot_accuracy=pd.DataFrame(spot_accuracy), spot_refinement=pd.DataFrame(spot_refinement),
        spot_errors=pd.DataFrame(spot_errors),
        noise=contracts[["contract_id", "split"]].assign(noise=noise), failures=pd.DataFrame(failures),
        settings=pd.DataFrame([
            {"setting": "seed", "value": seed}, {"setting": "noise standard deviation ($/share)", "value": NOISE_STD},
            {"setting": "spread floor ($/share)", "value": SPREAD_FLOOR},
            {"setting": "exploratory grid", "value": str(asdict(fit))},
            {"setting": "continuation grid", "value": str(asdict(continuation))},
            {"setting": "generation grid", "value": str(asdict(generation))},
            {"setting": "exploratory / continuation nfev caps", "value": f"{max_nfev} / {continuation_max_nfev}"},
            {"setting": "curve knots (days; fixed)", "value": "20, 45"},
            {"setting": "optimizer", "value": str(asdict(optimizer))},
            {"setting": "acceptance tolerances (unchanged)", "value": str(common["acceptance"])},
            {"setting": "checkpoint run fingerprint", "value": run_fingerprint},
            {"setting": "weighting", "value": "midpoint residual / max(half spread, .005); same weights on clean/noisy surfaces"},
            {"setting": "market", "value": str(asdict(market))}]),
        findings=_findings(summary, selections) if not summary.empty else pd.DataFrame(
            [{"model": "all", "dataset": "all", "finding": (
                "Truth-grid accuracy audit only; calibration was not requested." if not run_calibration else
                "No calibration result exists; inspect observation-generation and truth-grid accuracy failures.")}]))
    figures = _figures(tables) if compute_figures else {}
    if compute_figures and not tables["spot_accuracy"].empty:
        from ._pricing_spot_accuracy import spot_accuracy_figure
        figures["spot_accuracy"] = spot_accuracy_figure(tables["spot_accuracy"])
    return {"tables": tables, "figures": figures, "manifest": {
        **common, "backend": "american_vanilla_fd", "data_kind": "synthetic",
        "exercise_style": "american", "truths": {k: parameter_values(v) for k, v in truths.items()},
        "starts": {k: [parameter_values(v) for v in values] for k, values in starts.items()},
        "bounds": bounds, "curve_knot_times": [20/365, 45/365],
        "fit_config": asdict(fit), "generation_config": asdict(generation),
        "continuation_config": asdict(continuation), "max_nfev": int(max_nfev),
        "continuation_max_nfev": int(continuation_max_nfev), "run_continuation": bool(run_continuation),
        "requested_models": list(requested_models),
        "requested_datasets": list(requested_datasets),
        "compute_figures": bool(compute_figures),
        "run_calibration": bool(run_calibration),
        "optimizer": asdict(optimizer), "selection_rule": "Training objective only; held-out errors never select starts",
        "run_fingerprint": run_fingerprint,
        "checkpoint_directory": str(directory) if directory is not None else None,
        "generations": generations, "attempts": summary.to_dict("records"),
        "input_fingerprint": _fingerprint(common), "observation_fingerprints": {
            row["model"]: _fingerprint(row["prices"]) for row in generations},
        "pricing_evaluations": int(summary.pricing_evaluations.sum()) if not summary.empty else 0,
        "recorded_attempt_seconds": float(summary.elapsed_seconds.sum()) if not summary.empty else 0.,
        "recorded_generation_seconds": float(sum(row["elapsed_seconds"] for row in generations)),
        "reused_attempt_count": int(summary.checkpoint_reused.sum()) if not summary.empty else 0,
        "executed_attempt_count": int((~summary.checkpoint_reused).sum()) if not summary.empty else 0,
        "current_execution_pricing_evaluations": int(summary.loc[
            ~summary.checkpoint_reused, "pricing_evaluations"].sum()) if not summary.empty else 0,
        "current_execution_pricing_evaluations_complete": bool(summary.loc[
            ~summary.checkpoint_reused, "pricing_evaluations_complete"].all()) if not summary.empty else True,
        "pricing_evaluations_scope": "Attempt calls including Jacobians and refinement; generation and truth-grid baselines are separate",
        "elapsed_seconds": time.perf_counter()-started,
        "limitations": ["Synthetic American prices have finite-grid error; fine generation is not exact truth.",
            "Only 14/30/60-day tenors are studied; weak kappa/theta identification is expected.",
            "Fixed curve knots are structural assumptions, not recovered parameters.",
            "Noise is independent of declared spread widths; weighted errors are not likelihood diagnostics.",
            "Numerical acceptance, optimizer convergence and parameter identification are separate.",
            "No archived European SPY fit or market-data substitute enters this study."]}}


def _figures(tables):
    import matplotlib.pyplot as plt
    summary, parameters, predictions = (tables[key] for key in ("summary", "parameters", "predictions"))
    if summary.empty:
        return {}
    figures = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    groups = list(summary.groupby(["model", "dataset"], sort=False))
    for ax, ((model, dataset), group) in zip(axes.flat, groups):
        x = np.arange(len(group))
        labels = [f"start {row.start}" if row.stage == "exploratory" else f"continued · {row.start}"
                  for row in group.itertuples()]
        for column, shift, label in (("train_clean_rmse", -.16, "Training"), ("test_clean_rmse", .16, "Held out")):
            ax.bar(x+shift, pd.to_numeric(group[column], errors="coerce"), width=.3, label=label)
        errors = group[["train_clean_rmse", "test_clean_rmse"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite_errors = errors[np.isfinite(errors)]
        if finite_errors.size:
            ax.set_ylim(0., max(float(finite_errors.max())*1.23, 1e-8))
        for i, row in enumerate(group.itertuples()):
            ax.text(i, .97, "✓" if row.ready else "unstable" if row.optimizer_success else "budget" if row.status == "exhausted" else "failed",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.set(title=f"{MODEL_LABELS[model]} · {dataset}", xticks=x, xticklabels=labels,
               ylabel="RMSE versus clean prices ($/share)")
        ax.tick_params(axis="x", labelsize=8)
    axes[0, 0].legend(fontsize=8)
    recovery_title = ("Joint price recovery: three starts and the training-selected budget continuation"
                      if summary.stage.eq("continuation").any() else
                      "Joint price recovery: all three initial starts")
    fig.suptitle(recovery_title, y=1.04)
    figures["recovery"] = fig
    if not parameters.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
        for ax, model in zip(axes, MODEL_LABELS):
            subset = parameters.loc[parameters.model.eq(model)]
            names = list(dict.fromkeys(subset.parameter))
            for (dataset, start, stage), part in subset.groupby(["dataset", "start", "stage"], sort=False):
                part = part.set_index("parameter").reindex(names)
                ax.plot(np.arange(len(names)), (part.fitted-part.generating)/(part.upper-part.lower),
                    "o-" if stage == "continuation" else "o--", alpha=1. if stage == "continuation" else .45,
                    lw=2 if stage == "continuation" else 1,
                    label=f"{dataset} · {start}"+(" · continued" if stage == "continuation" else ""))
            ax.axhline(0, color="black", lw=.8)
            ax.set(title=MODEL_LABELS[model], xticks=np.arange(len(names)), xticklabels=names,
                   ylabel="(Estimate − truth) / declared bound width")
            ax.tick_params(axis="x", labelsize=8, rotation=20)
            ax.legend(fontsize=6, ncols=2)
        fig.suptitle("Parameter dispersion can remain after prices become close")
        figures["parameter_dispersion"] = fig
    if not predictions.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for ax, ((model, dataset), group) in zip(axes.flat, groups):
            fine = group.loc[group.stage.eq("continuation")]
            chosen = fine.iloc[0] if len(fine) else select_training_attempt(group.to_dict("records"))
            if chosen is None:
                ax.text(.5, .5, "No fitted attempt", transform=ax.transAxes, ha="center")
                continue
            attempt_id = chosen["attempt_id"]
            selected = predictions.loc[predictions.attempt_id.eq(attempt_id)]
            for tenor, part in selected.groupby("maturity", sort=True):
                ax.plot(part.strike, part.scaled_residual, "o-", label=f"{tenor*365:.0f}d")
                hold = part.loc[part.split.eq("heldout")]
                ax.scatter(hold.strike, hold.scaled_residual, s=75, facecolors="none", edgecolors="black", zorder=4)
            ax.axhline(0, color="black", lw=.7)
            ax.axhspan(-1, 1, color="#4c956c", alpha=.08)
            ax.set(title=f"{MODEL_LABELS[model]} · {dataset}", xlabel="Strike ($)",
                   ylabel="Price residual / effective half-spread")
            ax.legend(fontsize=7)
        fig.suptitle("Training-selected fits; black rings mark held-out contracts")
        figures["residuals"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for (model, dataset), group in groups:
        label = f"{MODEL_LABELS[model]} · {dataset}"
        axes[0].scatter(group.nfev, group.training_objective, label=label, s=np.where(group.stage.eq("continuation"), 90, 35))
        axes[1].scatter(group.max_price_change, group.test_clean_rmse, label=label,
                        s=np.where(group.stage.eq("continuation"), 90, 35))
    axes[0].set(xlabel="Optimizer evaluations (Jacobian calls excluded)", ylabel="Training mean squared weighted residual",
                title="A budget limit is not convergence")
    axes[1].set(xlabel="Maximum independent price refinement ($)", ylabel="Held-out clean-price RMSE ($)",
                title="Low fitting error is not numerical acceptance")
    axes[0].set_yscale("symlog", linthresh=1e-6)
    # Accepted refinements can occupy a narrow range between log ticks.
    # Keep their dollar values readable instead of displaying an empty axis.
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=False)
    axes[1].legend(fontsize=7)
    figures["convergence"] = fig
    return figures
