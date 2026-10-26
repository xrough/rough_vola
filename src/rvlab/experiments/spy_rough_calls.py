"""Sealed, bounded American-call calibration research campaign.

This experiment uses the exact no-early-exercise reduction only after the
public pricing boundary validates the market assumptions.  It never changes
canonical data readiness, chooses a start on held-out observations, or claims
historical rough-state or hedge readiness.  No notebook is read or written.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from hashlib import sha256
import argparse
import html
import json
import math
import os
import platform
from pathlib import Path
import time
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import scipy

from rvlab.pricing import (
    BlackScholesParams, HestonParams, RoughHestonParams, PricingConfig,
    SpotMarket, DiscountCurve, DividendSchedule, parameter_values,
    OptimizerConfig, PricingError,
)


ROOT = Path(__file__).resolve().parents[3]
DATES = ("2025-08-07", "2025-08-08", "2025-08-11")
EXPIRIES = ("2025-08-22", "2025-08-29", "2025-09-05")
TARGETS = (-.03, -.02, -.01, .01, .02, .03)
FIT_CONFIG = PricingConfig(192, 160, 512)
GENERATING_CONFIG = PricingConfig(384, 320, 2048)
EXTRA_CONFIG = PricingConfig(256, 240, 1024)
OPTIMIZER = OptimizerConfig(parameterization="asinh", jacobian="2-point")
GENERATOR = RoughHestonParams(.04, .045, 1.5, .30, -.60, .15)
STARTS = (
    (.04, .045, 1.5, .30, -.60, .15),
    (.025, .03, .5, .15, -.30, .10),
    (.075, .08, 3., .50, -.85, .30),
)
BOUNDS = {"v0": (.0025, .25), "theta": (.0025, .25), "kappa": (.1, 5.),
          "vol_of_vol": (.01, .6), "rho": (-.95, 0.), "hurst": (.05, .5)}


class CampaignDeadline(RuntimeError):
    """The declared numerical budget expired; completed evidence is retained."""


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.to_list())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _bytes(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_bytes(value) + b"\n")
    temporary.replace(path)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_inventory(folder, filename="campaign_inventory.json", *, required=False):
    path = Path(folder) / filename
    if not path.exists():
        if required:
            raise ValueError(f"required sealed inventory is missing: {path.name}")
        return
    receipt = json.loads(path.read_text())
    actual = {str(p.relative_to(folder)) for p in Path(folder).rglob("*")
              if p.is_file() and p != path}
    if actual != set(receipt["files"]):
        raise ValueError("sealed inventory file set mismatch")
    for relative, expected in receipt["files"].items():
        item = Path(folder) / relative
        if not item.is_file() or _hash(item) != expected:
            raise ValueError(f"sealed inventory mismatch: {relative}")


def _close_inventory(folder, seal, elapsed):
    receipt_path = Path(folder) / "campaign_inventory.json"
    _write_json(Path(folder) / "campaign_inventory.json", {"files": {str(p.relative_to(folder)): _hash(p)
        for p in sorted(Path(folder).rglob("*")) if p.is_file() and p != receipt_path},
        "campaign_identity": seal["identity_sha256"], "elapsed_numerical_seconds": elapsed,
        "hedging_ready": False})


def _close_attempt_inventory(folder):
    folder = Path(folder)
    receipt_path = folder / "attempt_inventory.json"
    _write_json(receipt_path, {"files": {str(p.relative_to(folder)): _hash(p)
        for p in sorted(folder.rglob("*")) if p.is_file() and p != receipt_path}})


def _validated_values(values, count):
    values = np.asarray(values, dtype=float)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise PricingError("price array must be finite with exactly one value per contract")
    return values


def _source_identity(root):
    """Bind all local pricing sources and this experiment, before fitting."""
    paths = sorted((root / "src/rvlab/pricing").rglob("*.py"))
    paths += [Path(__file__), root / "examples/spy_rough_calls.py"]
    paths += sorted((root / "src/rvlab/market_data").rglob("*.py"))
    return {str(path.relative_to(root)): _hash(path) for path in sorted(set(paths))
            if path.exists()}


def market_record(market):
    return {"spot": market.spot, "discount_curve": asdict(market.discount_curve),
            "dividends": asdict(market.dividends)}


def _targets_scales(panel, floor=.005):
    if "bid" in panel and "ask" in panel:
        bid, ask = panel.bid.to_numpy(float), panel.ask.to_numpy(float)
        return (bid + ask) / 2, np.maximum((ask - bid) / 2, floor)
    return panel.market_price.to_numpy(float), np.ones(len(panel))


def training_objective(prices, panel):
    """Only the provided training rows enter start/candidate selection."""
    target, scale = _targets_scales(panel)
    prices = np.asarray(prices, float)
    if prices.shape != target.shape or not np.isfinite(prices).all():
        raise ValueError("pricing output must be finite and contract-aligned")
    return float(np.mean(((prices - target) / scale) ** 2))


@dataclass
class NumericalBudget:
    """One campaign clock, including all pricing calls but not manual waiting."""
    launch_seconds: float = 2400.
    total_seconds: float = 3600.
    elapsed_before: float = 0.
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self):
        if not 0 < self.launch_seconds <= self.total_seconds:
            raise ValueError("require 0 < fit-launch budget <= numerical budget")
        self.started = self.clock()

    @property
    def elapsed(self):
        return self.elapsed_before + self.clock() - self.started

    def check(self, *, fitting=False):
        limit = self.launch_seconds if fitting else self.total_seconds
        if self.elapsed >= limit:
            raise CampaignDeadline(f"{'fit' if fitting else 'numerical'} budget exhausted "
                                   f"at {self.elapsed:.3f}s (limit {limit:.3f}s)")


def design_record():
    return {"version": "spy_rough_calls_v1", "dates": DATES, "expiries": EXPIRIES,
            "targets": TARGETS, "seed": 42, "noise_std": .005,
            "fit_config": FIT_CONFIG, "generating_config": GENERATING_CONFIG,
            "extra_config": EXTRA_CONFIG, "generator": GENERATOR,
            "optimizer": OPTIMIZER, "synthetic_half_spread": .025,
            "starts": STARTS, "bounds": BOUNDS, "max_nfev": 120,
            "hurst_profile": [.05, .10, .15, .25, .50], "profile_max_nfev": 80,
            "price_atol": .001, "price_rtol": 1e-4,
            "spread_refinement_fraction": .1, "price_scale_floor": .005,
            "fit_launch_seconds": 2400., "total_numerical_seconds": 3600.,
            "selection": "minimum training objective only; never held-out error",
            "hedging_ready": False,
            "synthetic_gate": "at least one converged numerically stable controlled or full rough fit "
                "on each clean/noisy surface, with refined training and held-out RMSE "
                "<= .001 and .015 respectively (test-only recovered-price sanity checks, "
                "not numerical pricing acceptance tolerances); parameter equality is not required",
            "interpretation": "independent implied surfaces, not historical rough-state reconstruction"}


def _numerical_environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "threads": {name: os.environ.get(name) for name in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}


def attempt_specs():
    """Predeclare a deterministic queue before generating observations."""
    out = []
    for dataset in ("synthetic_clean", "synthetic_noisy"):
        out.append({"id": f"{dataset}_controlled_0", "dataset": dataset,
                    "model": "rough_controlled", "start": 0, "max_nfev": 120})
        for index in range(3):
            out.append({"id": f"{dataset}_full_{index}", "dataset": dataset,
                        "model": "rough_full", "start": index, "max_nfev": 120})
    for date in DATES:
        for model in ("bs", "heston", "rough_controlled", "rough_full"):
            for index in range(3) if model == "rough_full" else (0,):
                out.append({"id": f"{date}_{model}_{index}", "dataset": date,
                            "model": model, "start": index, "max_nfev": 120})
    for hurst in (.05, .10, .15, .25, .50):
        out.append({"id": f"2025-08-07_hurst_{hurst:.2f}", "dataset": DATES[0],
                    "model": "rough_profile", "hurst": hurst, "start": None,
                    "max_nfev": 80})
    return out


def _initial_bounds(spec, selected=None):
    model = spec["model"]
    if model == "bs":
        return BlackScholesParams(.2), {"volatility": (.01, 1.)}
    if model == "heston":
        return HestonParams(*STARTS[0][:5]), {k: v for k, v in BOUNDS.items() if k != "hurst"}
    values = STARTS[spec.get("start") or 0]
    if model == "rough_profile" and selected is not None:
        values = tuple(selected[k] for k in ("v0", "theta", "kappa", "vol_of_vol", "rho", "hurst"))
    params = RoughHestonParams(*values)
    bounds = BOUNDS.copy()
    if model == "rough_controlled":
        params = replace(params, theta=.045, kappa=1.5, hurst=.15)
        bounds.update(theta=(.045, .045), kappa=(1.5, 1.5), hurst=(.15, .15))
    if model == "rough_profile":
        params = replace(params, hurst=spec["hurst"])
        bounds["hurst"] = (spec["hurst"], spec["hurst"])
    return params, bounds


def synthetic_panels(pricer, budget):
    """Same 18-call design as the pilot; no market-data fallback is possible."""
    market = SpotMarket(100., DiscountCurve.flat(.04, 60 / 365),
                        DividendSchedule((), 60 / 365))
    rows = []
    for days in (14, 30, 60):
        maturity = days / 365
        forward = market.spot / market.discount_curve.discount(maturity)
        for target in TARGETS:
            rows.append({"contract_id": f"synthetic_{days}_{target:+.2f}",
                         "strike": forward * math.exp(target), "maturity": maturity,
                         "is_call": True, "option_type": "call", "exercise_style": "american",
                         "target_log_moneyness": target,
                         "split": "heldout" if abs(target) == .02 else "train"})
    panel = pd.DataFrame(rows)
    budget.check()
    clean = _validated_values(pricer(GENERATOR, market=market, strike=panel.strike.to_numpy(),
                             maturity=panel.maturity.to_numpy(), config=GENERATING_CONFIG), len(panel))
    rng = np.random.default_rng(42)
    output = {}
    for name, values in (("synthetic_clean", clean),
                         ("synthetic_noisy", clean + rng.normal(0., .005, len(clean)))):
        q = panel.copy()
        q["market_price"] = values
        # A fixed declared synthetic spread tests the identical weighted workflow.
        q["bid"], q["ask"] = values - .025, values + .025
        output[name] = q
    return output, {name: market for name in output}


def _pending(spec, reason):
    return {**spec, "status": "pending", "reason": reason,
            "optimizer_success": False, "numerical_stable": False,
            "ready": False, "market_fit_accepted": False}


def _model_from_record(record):
    values = record["parameters"]
    cls = BlackScholesParams if "volatility" in values else RoughHestonParams if "hurst" in values else HestonParams
    return cls(**values)


def _probe_configs():
    return {"quadrature": replace(FIT_CONFIG, quadrature_nodes=384),
            "cutoff": replace(FIT_CONFIG, fourier_cutoff=200),
            "cutoff_double": replace(FIT_CONFIG, fourier_cutoff=320),
            "riccati": replace(FIT_CONFIG, riccati_steps=1024),
            "joint": FIT_CONFIG.refined(), "extra": EXTRA_CONFIG}


def refine_attempt(params, panel, market, pricer, budget):
    """Independent axes; one failure invalidates stability, never drops a row."""
    target, scale = _targets_scales(panel)
    budget.check()
    base = _validated_values(pricer(params, market=market, strike=panel.strike.to_numpy(),
                             maturity=panel.maturity.to_numpy(), config=FIT_CONFIG), len(panel))
    output = panel.copy()
    output["target_price"], output["price_scale"] = target, scale
    output["model_price"] = base
    probes, all_stable = [], True
    prices_by_axis = {"baseline": base}
    for axis, config in _probe_configs().items():
        item = {"axis": axis, "config": asdict(config), "pricing_attempted": False}
        try:
            budget.check()
            item["pricing_attempted"] = True
            refined = _validated_values(pricer(params, market=market, strike=panel.strike.to_numpy(),
                maturity=panel.maturity.to_numpy(), config=config), len(panel))
            changes = np.abs(refined - base)
            price_pass = changes <= .001 + 1e-4 * np.abs(base)
            spread_pass = changes <= .1 * scale if "bid" in panel else np.ones(len(panel), bool)
            passed = price_pass & spread_pass & np.isfinite(refined)
            item.update(status="complete", max_price_change=float(changes.max()),
                        price_stable=bool(price_pass.all()), spread_stable=bool(spread_pass.all()),
                        stable=bool(passed.all()))
            output[f"{axis}_price"] = refined
            output[f"{axis}_change"] = changes
            output[f"{axis}_pass"] = passed
            prices_by_axis[axis] = refined
            all_stable &= bool(passed.all())
        except Exception as exc:
            item.update(status="deadline" if isinstance(exc, CampaignDeadline) else "numerical_failure",
                        error=f"{type(exc).__name__}: {exc}", stable=False)
            all_stable = False
        probes.append(item)
    # Independent errors can cancel relative to the baseline. Require every
    # successful pair, including both cutoff probes and the joint grid, to pass.
    axes = list(prices_by_axis)
    for i, first in enumerate(axes):
        for second in axes[i+1:]:
            a, b = prices_by_axis[first], prices_by_axis[second]
            changes = np.abs(a - b)
            price_pass = changes <= .001 + 1e-4 * np.minimum(np.abs(a), np.abs(b))
            spread_pass = changes <= .1 * scale if "bid" in panel else np.ones(len(panel), bool)
            passed = price_pass & spread_pass & np.isfinite(a) & np.isfinite(b)
            probes.append({"axis": f"pair:{first}:{second}", "status": "complete",
                           "pricing_attempted": False, "max_price_change": float(changes.max()),
                           "price_stable": bool(price_pass.all()), "spread_stable": bool(spread_pass.all()),
                           "stable": bool(passed.all()), "failed_row_positions": np.flatnonzero(~passed).tolist()})
            output[f"pair_{first}_{second}_pass"] = passed
            all_stable &= bool(passed.all())
    output["price_residual"] = base - target
    output["scaled_residual"] = (base - target) / scale
    reference = output["extra_price"].to_numpy() if "extra_price" in output else np.full(len(panel), np.nan)
    output["reference_price"] = reference
    output["reference_residual"] = reference - target
    output["reference_scaled_residual"] = (reference - target) / scale
    return output, probes, all_stable


def split_metrics(table):
    metrics = {}
    for split in ("train", "heldout"):
        q = table.loc[table.split.eq(split)]
        for prefix, column in (("dollar", "price_residual"), ("spread", "scaled_residual"),
                               ("refined_dollar", "reference_residual"),
                               ("refined_spread", "reference_scaled_residual")):
            values = q[column].to_numpy(float)
            metrics[f"{split}_{prefix}_rmse"] = float(np.sqrt(np.mean(values ** 2))) if np.isfinite(values).all() else None
            metrics[f"{split}_{prefix}_mae"] = float(np.mean(np.abs(values))) if np.isfinite(values).all() else None
        metrics[f"{split}_count"] = len(q)
    return metrics


def run_attempt(spec, panel, market, folder, budget, calibrator, pricer, *, selected=None):
    """Retain the best training candidate even if a subsequent evaluation fails."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = budget.elapsed
    initial, bounds = _initial_bounds(spec, selected)
    train = panel.loc[panel.split.eq("train")].copy()
    state = {"actual_pricing_evaluations": 0, "successful_pricing_evaluations": 0,
             "best_training_objective": math.inf,
             "best_parameters": None}
    record = {**spec, "status": "running", "initial_parameters": parameter_values(initial),
              "bounds": bounds, "config": asdict(FIT_CONFIG), "optimizer_success": False,
              "ready": False, "numerical_stable": False, "market_fit_accepted": False}
    _write_json(folder / "attempt.json", record)

    def before(params, settings, stage):
        # API stages use 'fit' for parameter evaluations and separate diagnostic stages.
        fitting = stage in {"calibration pricing", "fit", "optimization", "residual", "jacobian"}
        budget.check(fitting=fitting)
        state["actual_pricing_evaluations"] += 1
        state["campaign_elapsed_seconds"] = budget.elapsed
        _write_json(folder / "evaluation_clock.json", {"campaign_elapsed_seconds": budget.elapsed,
                                                      "stage": stage})

    def after(params, settings, stage, prices):
        state["successful_pricing_evaluations"] += 1
        if settings == FIT_CONFIG and stage in {"calibration pricing", "final calibration pricing",
                "fit", "optimization", "residual", "jacobian", "baseline"}:
            objective = training_objective(prices, train)
            if objective < state["best_training_objective"]:
                state.update(best_training_objective=objective, best_parameters=parameter_values(params))
                _write_json(folder / "best_training_checkpoint.json", state)

    fit = None
    try:
        budget.check(fitting=True)
        fit = calibrator(train, initial, market=market, bounds=bounds, config=FIT_CONFIG,
            max_nfev=spec["max_nfev"], price_scale_floor=.005, stability_atol=.001,
            stability_rtol=1e-4, spread_refinement_fraction=.1,
            optimizer=OPTIMIZER,
            evaluation_callback=before, evaluation_observer=after)
        record.update(status="converged" if fit.optimizer_success else "optimizer_exhausted",
            parameters=parameter_values(fit.params), optimizer_success=bool(fit.optimizer_success),
            optimizer_status=int(fit.optimizer_status), optimizer_message=str(fit.optimizer_message),
            nfev=int(fit.nfev), njev=fit.njev, objective=float(fit.objective),
            active_bounds=dict(fit.active_bounds), fixed_parameters=dict(fit.fixed_parameters),
            optimality=fit.optimality, physical_optimality=fit.physical_optimality,
            free_parameters=tuple(getattr(fit, "free_parameters", ())),
            jacobian_singular_values=tuple(getattr(fit, "jacobian_singular_values", ())),
            backend_metadata=getattr(fit, "metadata", None),
            optimizer_settings=getattr(fit, "optimizer_settings", None))
        if getattr(fit, "final_jacobian", None) is not None:
            pd.DataFrame(fit.final_jacobian, columns=fit.free_parameters).to_csv(folder / "physical_jacobian.csv", index=False)
        if getattr(fit, "diagnostics", None) is not None:
            _write_json(folder / "api_refinement_diagnostics.json", fit.diagnostics)
            fit.diagnostics.checks.to_csv(folder / "api_refinement_checks.csv", index=False)
            record["api_numerical_stable"] = bool(fit.numerical_stable)
            record["api_max_price_change"] = fit.max_price_change
    except Exception as exc:
        status = ("budget_interrupted" if isinstance(exc, CampaignDeadline) else
                  "reporting_failure" if fit is not None else "numerical_failure")
        record.update(status=status, optimizer_success=False, ready=False,
                      error=f"{type(exc).__name__}: {exc}")
        if fit is not None:
            record["reported_optimizer_success_before_reporting_failure"] = bool(fit.optimizer_success)
        if state["best_parameters"] is not None:
            record["parameters"] = state["best_parameters"]
            record["objective"] = state["best_training_objective"]
            record["candidate_source"] = "best successful training evaluation; attempt did not complete"
    if "parameters" in record:
        refinement_count = 0
        def counted_pricer(params, **kwargs):
            nonlocal refinement_count
            budget.check()
            refinement_count += 1
            return pricer(params, **kwargs)
        try:
            table, probes, stable = refine_attempt(_model_from_record(record), panel, market, counted_pricer, budget)
            table.to_csv(folder / "contracts.csv", index=False)
            stable = stable and bool(getattr(fit, "numerical_stable", True))
            record.update(refinement=probes, numerical_stable=stable,
                          metrics=split_metrics(table), refinement_pricing_evaluations=1 + sum(p["pricing_attempted"] for p in probes))
            record["ready"] = bool(record["optimizer_success"] and stable)
        except Exception as exc:
            record.update(numerical_stable=False, ready=False,
                          refinement_error=f"{type(exc).__name__}: {exc}")
        record["refinement_pricing_evaluations"] = refinement_count
    record.update(calibration_pricing_evaluations=state["actual_pricing_evaluations"],
                  actual_pricing_evaluations=state["actual_pricing_evaluations"] + record.get("refinement_pricing_evaluations", 0),
                  successful_pricing_evaluations=state["successful_pricing_evaluations"],
                  best_training_objective=state["best_training_objective"],
                  elapsed_seconds=budget.elapsed - started,
                  campaign_elapsed_seconds=budget.elapsed)
    _write_json(folder / "attempt.json", record)
    _close_attempt_inventory(folder)
    return record


def choose_training_start(records, dataset, model="rough_full"):
    candidates = [row for row in records if row.get("dataset") == dataset
                  and row.get("model") == model and "parameters" in row
                  and row.get("objective") is not None and np.isfinite(row["objective"])]
    return min(candidates, key=lambda row: (row["objective"], row["id"])) if candidates else None


def synthetic_gate(records):
    """Numerical/recovered-price validation, independent of parameter recovery."""
    checks = {}
    for dataset, limit in (("synthetic_clean", .001), ("synthetic_noisy", .015)):
        candidates = [r for r in records if r.get("dataset") == dataset
                      and r.get("model") in {"rough_full", "rough_controlled"}]
        accepted = [r for r in candidates if r.get("ready") and
                    all(r.get("metrics", {}).get(f"{split}_refined_dollar_rmse") is not None and
                        r["metrics"][f"{split}_refined_dollar_rmse"] <= limit
                        for split in ("train", "heldout"))]
        checks[dataset] = {"passed": bool(accepted), "rmse_limit": limit,
                           "passing_attempts": [r["id"] for r in accepted]}
    return {"passed": all(v["passed"] for v in checks.values()), "checks": checks}


def input_sensitivities(records, bundle, folder, pricer, budget):
    """No refitting, ranking or held-out selection; record fixed-contract changes."""
    rows, failures = [], []
    for date in DATES:
        for model in ("bs", "heston", "rough_controlled", "rough_full"):
            chosen = choose_training_start(records, date, model)
            if chosen is None:
                continue
            params = _model_from_record(chosen)
            panel, market = bundle["panels"][date], bundle["markets"][date]
            previous_market = bundle.get("previous_markets", {}).get(date)
            variants = {}
            if previous_market is not None:
                variants["fixed_contracts_previous_endpoint_stock"] = (panel, replace(market, spot=previous_market.spot))
            for bp in (-10, 10):
                curve = market.discount_curve
                bumped = DiscountCurve(curve.times, tuple(d * math.exp(-bp * 1e-4 * t)
                                                          for t, d in zip(curve.times, curve.discounts)))
                variants[f"fixed_contracts_rate_{bp:+d}bp"] = (panel, replace(market, discount_curve=bumped))
            if date in bundle.get("previous_panels", {}) and previous_market is not None:
                variants["synchronized_previous_endpoint"] = (bundle["previous_panels"][date], previous_market)
            try:
                budget.check()
                base = _validated_values(pricer(params, market=market, strike=panel.strike.to_numpy(),
                                         maturity=panel.maturity.to_numpy(), config=EXTRA_CONFIG), len(panel))
                for name, (q, m) in variants.items():
                    budget.check()
                    p = _validated_values(pricer(params, market=m, strike=q.strike.to_numpy(),
                                          maturity=q.maturity.to_numpy(), config=EXTRA_CONFIG), len(q))
                    target, scale = _targets_scales(q)
                    for i, (_, contract) in enumerate(q.iterrows()):
                        rows.append({"date": date, "model": model, "attempt_id": chosen["id"],
                                     "sensitivity": name, "contract_id": contract.get("contract_id", i),
                                     "split": contract["split"], "baseline_model_price": base[i],
                                     "model_price": p[i], "model_price_change": p[i] - base[i],
                                     "observed_midpoint": target[i], "price_residual": p[i] - target[i],
                                     "spread_residual": (p[i] - target[i]) / scale[i],
                                     "interpretation": "reconstructed interval-end input sensitivity; no measured latency or executable price claim"})
            except Exception as exc:
                failures.append({"date": date, "model": model, "error": f"{type(exc).__name__}: {exc}"})
    pd.DataFrame(rows).to_csv(Path(folder) / "input_sensitivities.csv", index=False)
    _write_json(Path(folder) / "sensitivity_failures.json", failures)
    return rows, failures


def _summary_row(row):
    fields = ("id", "dataset", "model", "start", "hurst", "status", "reason", "objective",
              "optimizer_success", "optimizer_status", "optimizer_message", "nfev", "njev",
              "actual_pricing_evaluations", "calibration_pricing_evaluations", "refinement_pricing_evaluations", "elapsed_seconds",
              "numerical_stable", "ready", "market_fit_accepted", "error")
    result = {key: row.get(key) for key in fields}
    result.update(row.get("metrics", {}))
    for key in ("parameters", "active_bounds", "jacobian_singular_values", "fixed_parameters"):
        result[key] = json.dumps(_jsonable(row.get(key)), sort_keys=True)
    return result


def write_report(root, records, bundle, elapsed):
    """Compact report with diagnostic failures and clearly separated acceptance gates."""
    root = Path(root)
    summary = pd.DataFrame([_summary_row(row) for row in records])
    if summary.empty:
        summary = pd.DataFrame(columns=["id", "status"])
    summary.to_csv(root / "fit_summary.csv", index=False)
    selected = {date: choose_training_start(records, date) for date in DATES}
    _write_json(root / "training_only_selection.json", {k: v["id"] if v else None for k, v in selected.items()})
    dispersion = []
    for dataset in ("synthetic_clean", "synthetic_noisy") + DATES:
        fits = [r for r in records if r.get("dataset") == dataset and r.get("model") == "rough_full" and "parameters" in r]
        if fits:
            for name in parameter_values(GENERATOR):
                values = [r["parameters"][name] for r in fits]
                dispersion.append({"dataset": dataset, "parameter": name, "attempt_count": len(values),
                                   "minimum": min(values), "maximum": max(values), "range": max(values)-min(values)})
    pd.DataFrame(dispersion).to_csv(root / "parameter_dispersion.csv", index=False)
    gate = synthetic_gate(records)
    synthetic_ran = any(str(r.get("dataset", "")).startswith("synthetic") and
                        ("parameters" in r or r.get("actual_pricing_evaluations", 0) > 0) for r in records)
    gate_status = ("passed" if gate["passed"] else "did not pass") if synthetic_ran else "not run"
    independent_verified = bundle.get("diagnostics", {}).get("independent_intraday_prices_verified", False)
    independent_status = "verified" if independent_verified else "unavailable; independent intraday price agreement is not established"
    text = ["SPY rough-Heston American-call pilot", "", f"Numerical elapsed time: {elapsed:.2f} seconds.",
            f"Repeat-acquisition stock gate: {bundle.get('ready', False)} ({bundle.get('status', 'unknown')}).",
            f"Independent intraday cross-source validation: {independent_status}.",
            f"Synthetic recovered-price gate: {gate_status}.",
            "American calls retain their exercise labels. The exact European-value reduction requires",
            "explicit no-dividend coverage, nonnegative rates and the frictionless martingale model.",
            "The archived dividend calendar is accepted subject to the stated no-subsequent-notice assumption.",
            "Treasury yields are discount proxies; interval-end snapshots are reconstructed, not executable.",
            "Historical conditional rough state, stock deltas, puts and hedging readiness are not established.",
            "", "Attempts: " + ", ".join(f"{k}={v}" for k, v in summary.status.value_counts().items()),
            "", "A converged optimizer is not numerical stability, market fit, or identified parameters.",
            "Declared starts and bounds are retained in the design/queue; completed or failed attempts retain",
            "their results in fit_summary.csv and attempts/<attempt>/.",
            "Held-out outcomes never select a start or Hurst profile initialization."]
    if not bundle.get("ready", False):
        text += ["", "Real-data calibration remains pending until the repeat-acquisition stock gate passes."]
    if not gate["passed"]:
        text += ["", "Synthetic workflow validation has not run; no scientific failure is inferred." if not synthetic_ran
                 else "Real-data calibration remains pending because synthetic workflow validation did not pass."]
    report = "\n".join(text) + "\n"
    (root / "report.txt").write_text(report)
    (root / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>SPY rough-call study</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:3rem auto;line-height:1.5}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccc}pre{white-space:pre-wrap}img{max-width:100%}</style>"
        "<h1>SPY American-call calibration</h1><pre>" + html.escape(report) + "</pre>" + summary.to_html(index=False)
        + "<h2>Diagnostics</h2><img src='figures/calibration_errors.png'><img src='figures/identification.png'>")
    _write_json(root / "summary.json", {"synthetic_gate": gate, "synthetic_gate_status": gate_status,
        "independent_intraday_prices_verified": bool(independent_verified), "data_ready": bool(bundle.get("ready", False)),
        "hedging_ready": False, "elapsed_numerical_seconds": elapsed, "attempts": records})
    _figures(root, records)


def _figures(root, records):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    folder = Path(root) / "figures"
    folder.mkdir(exist_ok=True)
    completed = [r for r in records if r.get("metrics")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, key, title in zip(axes, ("dollar", "spread"), ("Dollar RMSE per share", "Half-spread-normalised RMSE")):
        for i, row in enumerate(completed):
            metrics = row["metrics"]
            ax.scatter(i, metrics[f"train_{key}_rmse"], marker="o", color="#24648a", label="Training" if i == 0 else None)
            ax.scatter(i, metrics[f"heldout_{key}_rmse"], marker="x", color="#b35b26", label="Held out" if i == 0 else None)
        if completed:
            ax.set_xticks(range(len(completed)), [r["id"] for r in completed], rotation=90, fontsize=7)
            ax.set_yscale("symlog", linthresh=.001)
            ax.legend()
        else:
            ax.text(.5, .5, "No completed candidate prices", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        ax.grid(alpha=.2)
    fig.suptitle("Recovered prices and failures remain separate from convergence")
    fig.savefig(folder / "calibration_errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    profiles = [r for r in records if r.get("model") == "rough_profile" and r.get("objective") is not None]
    if profiles:
        axes[0].plot([r["hurst"] for r in profiles], [r["objective"] for r in profiles], "o-")
    else:
        axes[0].text(.5, .5, "Hurst profile pending", ha="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="Fixed Hurst", ylabel="Training objective", title="Refitted Hurst profile")
    for row in completed:
        if row.get("model") == "rough_full":
            singular = row.get("jacobian_singular_values", ())
            if singular:
                axes[1].semilogy(range(1, len(singular)+1), singular, "o-", alpha=.6, label=row["id"])
    axes[1].set(xlabel="Singular-value index", ylabel="Physical residual Jacobian singular value",
                title="Local identification depends on units and the fitted point")
    fig.savefig(folder / "identification.png", dpi=150)
    plt.close(fig)


def run_campaign(campaign_root=None, *, root=ROOT, bundle=None, calibrator=None, pricer=None,
                 resume=False, budget=None):
    """Execute or safely resume exactly one sealed one-hour campaign."""
    root = Path(root).resolve()
    folder = Path(campaign_root or root / "lab/reports/artifacts/spy_rough_calls_v1").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    _verify_inventory(folder)
    if bundle is None:
        from rvlab.market_data.spy_call_pilot import load_spy_call_pilot
        bundle = load_spy_call_pilot(folder, root=root)
    if not bundle.get("ready", False) and not (folder / "campaign_seal.json").exists():
        # Stock acquisition/validation is completed before spending the one
        # numerical hour. This preflight is deliberately not a numerical seal.
        pending = [_pending(spec, "stock repeat-acquisition/OHLC preflight remains blocked; numerical campaign has not started")
                   for spec in attempt_specs()]
        _write_json(folder / "preflight.json", {"status": "blocked_preflight", "numerical_campaign_started": False,
                    "diagnostics": bundle.get("diagnostics", {}), "design": design_record()})
        _write_json(folder / "queue.json", attempt_specs())
        write_report(folder, pending, bundle, 0.)
        return {"root": str(folder), "status": "blocked_preflight", "attempts": 0,
                "data_ready": False, "elapsed_seconds": 0., "numerical_campaign_started": False}
    if calibrator is None or pricer is None:
        from rvlab.pricing import price_american_calls, calibrate_american_call_surface
        calibrator = calibrator or calibrate_american_call_surface
        pricer = pricer or price_american_calls
    identity = {"design": design_record(), "source_hashes": _source_identity(root),
                "environment": _numerical_environment(),
                "input_diagnostics": bundle.get("diagnostics", {}), "data_ready": bundle.get("ready", False),
                "panels": {date: sha256(panel.to_csv(index=False).encode()).hexdigest()
                           for date, panel in bundle.get("panels", {}).items()},
                "markets": {date: market_record(m) for date, m in bundle.get("markets", {}).items()},
                "previous_panels": {date: sha256(panel.to_csv(index=False).encode()).hexdigest()
                                    for date, panel in bundle.get("previous_panels", {}).items()},
                "previous_markets": {date: market_record(m) for date, m in bundle.get("previous_markets", {}).items()}}
    seal = {"identity": identity, "identity_sha256": sha256(_bytes(identity)).hexdigest()}
    existing = folder / "campaign_seal.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if not resume:
            raise FileExistsError("campaign already sealed; use resume with identical sources and data, or a new root")
        if old["identity_sha256"] != seal["identity_sha256"]:
            raise ValueError("campaign source/data/design identity changed; resume is forbidden")
        if sha256(_bytes(old["identity"])).hexdigest() != old["identity_sha256"]:
            raise ValueError("campaign seal identity hash is invalid")
        for relative, expected in old["identity"]["source_hashes"].items():
            snapshot = folder / "source_snapshot" / relative
            if not snapshot.is_file() or _hash(snapshot) != expected:
                raise ValueError(f"source snapshot mismatch: {relative}")
    else:
        seal["sealed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        for relative, expected in identity["source_hashes"].items():
            destination = folder / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root / relative).read_bytes())
            if _hash(destination) != expected:
                raise RuntimeError("source changed while creating the pre-fit snapshot")
        _write_json(existing, seal)
    records = []
    elapsed = 0.
    if (folder / "progress.json").exists():
        elapsed = float(json.loads((folder / "progress.json").read_text()).get("campaign_elapsed_seconds", 0.))
    for prior_summary in (folder / "summary.json", folder / "campaign_inventory.json"):
        if prior_summary.exists():
            elapsed = max(elapsed, float(json.loads(prior_summary.read_text()).get("elapsed_numerical_seconds", 0.)))
    phase_path = folder / "active_phase.json"
    if phase_path.exists():
        previous_phase = json.loads(phase_path.read_text())
        if previous_phase.get("active"):
            elapsed = max(elapsed, 3600.)
        elapsed = max(elapsed, previous_phase.get("campaign_elapsed_seconds", 0.))
    for path in sorted((folder / "attempts").glob("*/attempt.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "running":
            _verify_inventory(path.parent, "attempt_inventory.json",
                required=record.get("status") not in {"pending", "process_interrupted"})
        if record.get("status") == "running":
            # A killed evaluation has unknown elapsed cost. Conservatively spend
            # the remaining clock rather than silently grant a fresh hour.
            checkpoint = path.parent / "best_training_checkpoint.json"
            state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            record.update(status="process_interrupted", optimizer_success=False,
                          ready=False, numerical_stable=False,
                          reason="Previous process ended during an evaluation; unknown remaining time is conservatively consumed.")
            if state.get("best_parameters"):
                record.update(parameters=state["best_parameters"], objective=state["best_training_objective"])
            record["actual_pricing_evaluations"] = state.get("actual_pricing_evaluations", 0)
            elapsed = max(elapsed, 3600.)
            _write_json(path, record)
        records.append(record)
        elapsed = max(elapsed, record.get("campaign_elapsed_seconds", 0.))
        clock_path = path.parent / "evaluation_clock.json"
        if clock_path.exists():
            elapsed = max(elapsed, json.loads(clock_path.read_text()).get("campaign_elapsed_seconds", 0.))
    budget = budget or NumericalBudget(elapsed_before=elapsed)
    _write_json(folder / "queue.json", attempt_specs())
    synthetic_path = folder / "synthetic_observations.csv"
    if synthetic_path.exists():
        generation = folder / "synthetic_generation.json"
        if not generation.exists() or json.loads(generation.read_text()).get("sha256") != _hash(synthetic_path):
            raise ValueError("sealed synthetic observations are missing their matching generation fingerprint")
        all_synthetic = pd.read_csv(synthetic_path)
        panels = {name: q.drop(columns="dataset").reset_index(drop=True)
                  for name, q in all_synthetic.groupby("dataset", sort=False)}
        synthetic_market = SpotMarket(100., DiscountCurve.flat(.04, 60/365), DividendSchedule((), 60/365))
        markets = {name: synthetic_market for name in panels}
    else:
        _write_json(phase_path, {"phase": "synthetic_generation", "active": True,
                                 "campaign_elapsed_seconds": budget.elapsed})
        try:
            panels, markets = synthetic_panels(pricer, budget)
        except Exception as exc:
            failure = {"status": "budget_exhausted" if isinstance(exc, CampaignDeadline) else "numerical_failure",
                       "error": f"{type(exc).__name__}: {exc}", "campaign_elapsed_seconds": budget.elapsed}
            _write_json(folder / "synthetic_generation_failure.json", failure)
            _write_json(phase_path, {"phase": "synthetic_generation", "active": False,
                                     "campaign_elapsed_seconds": budget.elapsed})
            records = [_pending(s, "synthetic observation generation failed or numerical budget expired")
                       for s in attempt_specs()]
            write_report(folder, records, bundle, budget.elapsed)
            _close_inventory(folder, seal, budget.elapsed)
            return {"root": str(folder), "status": failure["status"], "attempts": 0,
                    "data_ready": bool(bundle.get("ready", False)), "elapsed_seconds": budget.elapsed,
                    "synthetic_gate": synthetic_gate(records)}
        pd.concat([q.assign(dataset=name) for name, q in panels.items()], ignore_index=True).to_csv(synthetic_path, index=False)
        _write_json(folder / "synthetic_generation.json", {"sha256": _hash(synthetic_path),
                    "parameters": GENERATOR, "config": GENERATING_CONFIG, "seed": 42, "noise_std": .005})
        _write_json(phase_path, {"phase": "synthetic_generation", "active": False,
                                 "campaign_elapsed_seconds": budget.elapsed})
    panels.update(bundle.get("panels", {}))
    markets.update(bundle.get("markets", {}))
    by_id = {r["id"]: r for r in records}
    for spec in attempt_specs():
        if spec["id"] in by_id and by_id[spec["id"]]["status"] != "pending":
            continue
        is_market = spec["dataset"] in DATES
        reason = None
        if is_market and not bundle.get("ready", False):
            reason = "independent stock repeat-acquisition/OHLC validation gate pending"
        elif is_market and not synthetic_gate(records)["passed"]:
            reason = "synthetic recovered-price/numerical workflow gate failed or pending"
        elif spec["dataset"] not in panels:
            reason = "required observed input panel unavailable; no synthetic fallback"
        elif budget.elapsed >= budget.launch_seconds:
            reason = "40-minute fit-launch budget exhausted"
        selected = choose_training_start(records, DATES[0]) if spec["model"] == "rough_profile" else None
        if spec["model"] == "rough_profile" and selected is None:
            reason = reason or "no successful training-objective candidate for profile initialization"
        records = [r for r in records if r["id"] != spec["id"]]
        if reason:
            record = _pending(spec, reason)
            _write_json(folder / "attempts" / spec["id"] / "attempt.json", record)
        else:
            _write_json(phase_path, {"phase": spec["id"], "active": True,
                                     "campaign_elapsed_seconds": budget.elapsed})
            record = run_attempt(spec, panels[spec["dataset"]], markets[spec["dataset"]],
                folder / "attempts" / spec["id"], budget, calibrator, pricer,
                selected=selected["parameters"] if selected else None)
            _write_json(phase_path, {"phase": spec["id"], "active": False,
                                     "campaign_elapsed_seconds": budget.elapsed})
        records.append(record)
        _write_json(folder / "progress.json", {"campaign_elapsed_seconds": budget.elapsed,
                                               "completed_attempts": len(records), "last_attempt": spec["id"]})
    _write_json(phase_path, {"phase": "input_sensitivities", "active": True,
                             "campaign_elapsed_seconds": budget.elapsed})
    try:
        input_sensitivities(records, bundle, folder, pricer, budget)
    finally:
        _write_json(phase_path, {"phase": "input_sensitivities", "active": False,
                                 "campaign_elapsed_seconds": budget.elapsed})
    live_sources = _source_identity(root)
    if live_sources != identity["source_hashes"]:
        _write_json(folder / "source_identity_failure.json", {"sealed": identity["source_hashes"],
                    "live": live_sources, "ready": False})
        for row in records:
            row["status_before_source_invalidation"] = row["status"]
            row["status"] = "source_identity_failure"
            row["ready"] = False
            row["source_identity_valid"] = False
            attempt_folder = folder / "attempts" / row["id"]
            _write_json(attempt_folder / "attempt.json", row)
            _close_attempt_inventory(attempt_folder)
    write_report(folder, records, bundle, budget.elapsed)
    _close_inventory(folder, seal, budget.elapsed)
    return {"root": str(folder), "attempts": len(records), "synthetic_gate": synthetic_gate(records),
            "data_ready": bool(bundle.get("ready", False)), "elapsed_seconds": budget.elapsed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(_jsonable(run_campaign(args.output, root=args.root, resume=args.resume)), indent=2))


if __name__ == "__main__":
    main()
