"""Small, explicit numerical experiments around :mod:`rvlab.pricing`.

These helpers retain unsuccessful and unstable fits as experimental outcomes.
They neither alter the numerical pricers nor infer market-data readiness.
Prices and errors are per underlying unit throughout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
import time

import numpy as np
import pandas as pd

from rvlab.pricing import (
    BlackScholesParams, CalibrationResult, HestonParams, ModelParams,
    PricingConfig, PricingError, RoughHestonParams, calibrate_surface, price, parameter_values, OptimizerConfig,
)
from rvlab.pricing.diagnostics import pricing_diagnostics


SYNTHETIC_TRUTHS = {
    "bs": BlackScholesParams(.20),
    "heston": HestonParams(.04, .045, 1.5, .30, -.60),
    "rough_heston": RoughHestonParams(.04, .045, 1., .12, -.60, .15),
}
SYNTHETIC_INITIALS = {
    "bs": BlackScholesParams(.30),
    "heston": HestonParams(.035, .05, 1.2, .25, -.45),
    "rough_heston": RoughHestonParams(.03, .045, 1., .12, -.60, .15),
}
DEMO_BOUNDS = {
    "volatility": (.001, 2.), "v0": (.0025, .25), "theta": (.0025, .25),
    "kappa": (.1, 5.), "vol_of_vol": (.01, .6), "rho": (-.95, 0.),
    "hurst": (.05, .5),
}
COARSE_CONFIG = PricingConfig(96, 60., 256)
SPY_CONFIG = PricingConfig(192, 160., 512)
ACCURATE_SPY_CONFIG = PricingConfig(192, 400., 1024)
EXTENDED_CONFIG = PricingConfig(256, 240., 1024)
REFERENCE_CONFIG = PricingConfig(128, 75., 512)
STABILITY_ATOL = .005
STABILITY_RTOL = 1e-4
PRICE_SCALE_FLOOR = .005


@dataclass(frozen=True)
class FitAttempt:
    """One calibration attempt, including failed attempts and all observations.

    ``result`` is absent if the core calibration raised a numerical error.
    ``summary['ready']`` additionally checks refinement on held-out observations;
    ``result.ready`` only describes the training contracts. ``predictions``
    preserves the incoming row order and index. NaNs denote unavailable prices,
    never replacements for a divergent price.
    """

    label: str
    result: CalibrationResult | None
    summary: dict
    predictions: pd.DataFrame


def synthetic_surface(
    model: str = "rough_heston", *, noise_std: float = 0., seed: int = 42,
    config: PricingConfig | None = None,
) -> pd.DataFrame:
    """Return 18 OTM synthetic observations, generated on the reference grid.

    The 97/103 strikes are held out at each maturity. All other strikes train
    the model. Noise is added to prices, not fabricated bid/ask spreads.
    """
    if model not in SYNTHETIC_TRUTHS:
        raise ValueError(f"model must be one of {tuple(SYNTHETIC_TRUTHS)}")
    if (isinstance(noise_std, bool) or not isinstance(noise_std, Real)
            or not np.isfinite(noise_std) or noise_std < 0):
        raise ValueError("noise_std must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if config is None:
        config = REFERENCE_CONFIG
    elif not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    maturity = np.repeat(np.array([14, 30, 60]) / 365., 6)
    strike = np.tile(np.array([94., 97., 99., 101., 103., 106.]), 3)
    forward = 100. * np.exp(.02 * maturity)
    discount = np.exp(-.03 * maturity)
    is_call = strike >= forward
    truth = SYNTHETIC_TRUTHS[model]
    reference = np.asarray(price(
        truth, forward=forward, discount=discount, strike=strike,
        maturity=maturity, is_call=is_call, config=config,
    ), dtype=float)
    noise = np.random.default_rng(seed).normal(0., noise_std, len(strike))
    out = pd.DataFrame({
        "contract_id": [f"synthetic-{model}-{i:02d}" for i in range(len(strike))],
        "strike": strike, "maturity": maturity, "is_call": is_call,
        "forward": forward, "discount": discount,
        "market_price": reference + noise, "reference_price": reference,
        "split": np.where(np.isin(strike, [97., 103.]), "heldout", "train"),
    })
    out.attrs.update({
        "model": model, "truth": asdict(truth), "seed": int(seed),
        "noise_std": float(noise_std), "reference_config": asdict(config),
        "observation_source": "finer-resolution synthetic prices",
        "price_units": "per underlying unit",
    })
    return out


def _bounds_for(params: ModelParams, overrides=None) -> dict:
    chosen = {name: DEMO_BOUNDS["theta" if name.startswith("theta_") else name]
              for name in parameter_values(params)}
    if overrides is not None:
        chosen.update(dict(overrides))
    return chosen


def _observation_arrays(quotes: pd.DataFrame):
    if not isinstance(quotes, pd.DataFrame) or quotes.empty:
        raise ValueError("quotes must be a nonempty DataFrame")
    required = {"strike", "maturity", "is_call", "forward", "discount"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes are missing columns: {sorted(missing)}")
    if ("bid" in quotes) != ("ask" in quotes):
        raise ValueError("provide both bid and ask, or neither")
    if "bid" in quotes:
        bid, ask = quotes["bid"].to_numpy(float), quotes["ask"].to_numpy(float)
        target = bid + .5 * (ask - bid)
        scale = np.maximum(.5 * (ask - bid), PRICE_SCALE_FLOOR)
    elif "market_price" in quotes:
        target = quotes["market_price"].to_numpy(float)
        scale = np.ones(len(quotes))
    else:
        raise ValueError("quotes require market_price or bid and ask")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(scale)):
        raise ValueError("observation prices and scales must be finite")
    split = quotes["split"].to_numpy() if "split" in quotes else np.full(len(quotes), "train")
    unknown = set(split) - {"train", "heldout", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown split labels: {sorted(unknown)}")
    training = split == "train"
    if not np.any(training):
        raise ValueError("at least one training contract is required")
    return target, scale, training


def _evaluate(params, quotes, config):
    values = np.asarray(price(
        params, forward=quotes["forward"].to_numpy(float),
        discount=quotes["discount"].to_numpy(float),
        strike=quotes["strike"].to_numpy(float),
        maturity=quotes["maturity"].to_numpy(float),
        is_call=quotes["is_call"].to_numpy(), config=config,
    ), dtype=float)
    if values.shape != (len(quotes),) or not np.all(np.isfinite(values)):
        raise PricingError("experiment pricing produced invalid or nonfinite prices")
    return values


def _rmse(values, mask):
    if not np.any(mask) or not np.all(np.isfinite(values[mask])):
        return None
    return float(np.sqrt(np.mean(values[mask] ** 2)))


def _check_records(checks: pd.DataFrame, **columns) -> list[dict]:
    """Keep missing diagnostic comparisons JSON-null rather than NaN."""
    table = checks.assign(**columns).astype(object)
    return table.where(table.notna(), None).to_dict("records")


def fit_attempt(
    label: str, quotes: pd.DataFrame, initial_params: ModelParams, *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    config: PricingConfig = COARSE_CONFIG, max_nfev: int = 80,
    refine_configs: Sequence[PricingConfig] | None = None,
    optimizer: OptimizerConfig | None = None,
) -> FitAttempt:
    """Fit training rows only, retaining numerical failures as diagnostic data.

    Bounds override the demonstration box; equal endpoints fix parameters.
    Invalid caller inputs raise normally. Only identifiable numerical pricing
    failures are captured. The held-out prices cannot influence optimization
    or the choice of starting point. Optional ``refine_configs`` supplies the
    remaining grids for bounded warm-start calibration; every stage is kept.
    """
    started = time.perf_counter()
    target, scale, train = _observation_arrays(quotes)
    chosen_bounds = _bounds_for(initial_params, bounds)
    fixed = {name: float(pair[0]) for name, pair in chosen_bounds.items()
             if pair[0] == pair[1]}
    free = [name for name in parameter_values(initial_params) if name not in fixed]
    predictions = quotes.copy(deep=True)
    predictions["target_price"], predictions["price_scale"] = target, scale
    for name in ("model_price", "refined_model_price", "price_residual", "residual",
                 "refined_price_residual", "refined_residual", "refinement_difference"):
        predictions[name] = np.nan
    summary = {
        "label": label, "status": "numerical_failure", "error": None,
        "initial_params": asdict(initial_params), "params": None,
        "bounds": chosen_bounds, "config": asdict(config), "initial_config": asdict(config),
        "final_config": None, "total_nfev": None,
        "refined_config": None, "max_nfev": int(max_nfev),
        "free_parameters": free, "fixed_parameters": fixed,
        "free_parameter_count": len(free), "fixed_parameter_count": len(fixed),
        "n_train": int(train.sum()), "n_test": int((~train).sum()),
        "optimizer_success": False, "optimizer_status": None,
        "optimizer_message": None, "nfev": None, "njev": None,
        "optimality": None, "physical_optimality": None,
        "optimizer_settings": asdict(optimizer if optimizer is not None else OptimizerConfig()),
        "pricing_evaluations": 0,
        "calibration_numerical_stable": False, "calibration_ready": False,
        "numerical_stable": False, "ready": False, "active_bounds": {},
        "training_objective": None, "refined_training_objective": None,
        "train_rmse": None, "test_rmse": None,
        "train_scaled_rmse": None, "test_scaled_rmse": None,
        "refined_train_rmse": None, "refined_test_rmse": None,
        "refined_train_scaled_rmse": None, "refined_test_scaled_rmse": None,
        "max_price_change": None, "elapsed_seconds": None,
        "selection_rule": "minimum training objective only; held-out outcomes never select a start",
        "stability_atol": STABILITY_ATOL, "stability_rtol": STABILITY_RTOL,
        "price_scale_floor": PRICE_SCALE_FLOOR,
        "numerical_checks": [], "stages": [],
    }
    result = None
    continuation_ready = True
    try:
        fit_kwargs = dict(
            forward=quotes["forward"].to_numpy(float)[train],
            discount=quotes["discount"].to_numpy(float)[train],
            bounds=chosen_bounds, config=config, max_nfev=max_nfev,
            price_scale_floor=PRICE_SCALE_FLOOR,
            stability_atol=STABILITY_ATOL, stability_rtol=STABILITY_RTOL,
        )
        if optimizer is not None:
            fit_kwargs["optimizer"] = optimizer
        if refine_configs is None:
            result = calibrate_surface(quotes.loc[train], initial_params, **fit_kwargs)
        else:
            from rvlab.pricing.refinement import calibrate_surface_refined
            continuation = calibrate_surface_refined(
                quotes.loc[train], initial_params,
                configs=[config, *refine_configs],
                **{name: value for name, value in fit_kwargs.items() if name != "config"},
            )
            result = continuation.result
            continuation_ready = continuation.ready
            summary["stages"] = [{
                "config": asdict(stage.config), "initial_params": asdict(stage.initial_params),
                "error": str(stage.error) if stage.error is not None else None,
                "elapsed_seconds": stage.elapsed_seconds,
                "params": asdict(stage.result.params) if stage.result is not None else None,
                "ready": bool(stage.result.ready) if stage.result is not None else False,
                "nfev": stage.result.nfev if stage.result is not None else None,
                "optimizer_success": stage.result.optimizer_success if stage.result is not None else False,
                "optimizer_status": stage.result.optimizer_status if stage.result is not None else None,
                "optimizer_message": stage.result.optimizer_message if stage.result is not None else None,
                "optimality": stage.result.optimality if stage.result is not None else None,
                "physical_optimality": stage.result.physical_optimality if stage.result is not None else None,
                "optimizer_settings": (dict(stage.result.optimizer_settings)
                    if stage.result is not None and stage.result.optimizer_settings is not None else None),
                "objective": stage.result.objective if stage.result is not None else None,
                "numerical_stable": stage.result.numerical_stable if stage.result is not None else False,
                "active_bounds": dict(stage.result.active_bounds) if stage.result is not None else {},
                "numerical_checks": (_check_records(stage.result.diagnostics.checks)
                                     if stage.result is not None else []),
            } for stage in continuation.stages]
            summary["total_nfev"] = continuation.total_nfev
            if continuation.stages and continuation.stages[-1].error:
                summary["error"] = str(continuation.stages[-1].error)
    except PricingError as exc:
        summary["error"] = str(exc)
    if result is not None:
        summary.update({
            "params": asdict(result.params),
            "optimizer_success": bool(result.optimizer_success),
            "optimizer_status": int(result.optimizer_status),
            "optimizer_message": result.optimizer_message,
            "nfev": int(result.nfev), "njev": result.njev,
            "optimality": float(result.optimality) if np.isfinite(result.optimality) else None,
            "physical_optimality": float(result.physical_optimality) if np.isfinite(result.physical_optimality) else None,
            "optimizer_settings": dict(result.optimizer_settings) if result.optimizer_settings is not None else None,
            "pricing_evaluations": int(result.pricing_evaluations),
            "config": asdict(result.config),
            "final_config": asdict(result.config),
            "refined_config": asdict(result.refined_config),
            "calibration_numerical_stable": bool(result.numerical_stable),
            "calibration_ready": bool(result.ready),
            "active_bounds": dict(result.active_bounds),
            "training_objective": float(result.objective),
            "refined_training_objective": float(result.refined_objective),
        })
        if summary["total_nfev"] is None:
            summary["total_nfev"] = int(result.nfev)
        elif summary["stages"]:
            summary["pricing_evaluations"] = sum(
                stage.result.pricing_evaluations for stage in continuation.stages
                if stage.result is not None)
        core_checks = _check_records(result.diagnostics.checks, scope="training")
        summary["numerical_checks"] = core_checks
        try:
            checked = pricing_diagnostics(
                result.params, forward=quotes.forward.to_numpy(float),
                discount=quotes.discount.to_numpy(float), strike=quotes.strike.to_numpy(float),
                maturity=quotes.maturity.to_numpy(float), is_call=quotes.is_call.to_numpy(),
                config=result.config, atol=STABILITY_ATOL, rtol=STABILITY_RTOL,
            )
            coarse = checked.base_prices
            predictions["model_price"] = coarse
            predictions["price_residual"] = coarse - target
            predictions["residual"] = (coarse - target) / scale
            fine = checked.reference_prices
            if fine is not None:
                predictions["refined_model_price"] = fine
                predictions["refined_price_residual"] = fine - target
                predictions["refined_residual"] = (fine - target) / scale
                predictions["refinement_difference"] = np.abs(fine - coarse)
            summary["pricing_evaluations"] += checked.pricing_evaluations
            summary["numerical_checks"] += _check_records(checked.checks, scope="all_contracts")
            stable = bool(result.numerical_stable and checked.stable)
            failed = [row["error"] for row in summary["numerical_checks"] if row["error"]]
            if failed:
                summary["error"] = "; ".join(dict.fromkeys(failed))
            available_changes = [value for value in (result.max_price_change, checked.max_price_change)
                                 if np.isfinite(value)]
            max_change = max(available_changes) if available_changes else None
            summary.update({
                "numerical_stable": stable,
                "ready": bool(result.optimizer_success and stable and continuation_ready),
                "max_price_change": float(max_change) if max_change is not None else None,
                "status": ("refinement_failure" if failed or not continuation_ready and summary["error"] else
                           "converged" if result.optimizer_success and stable and continuation_ready else
                           "unstable" if result.optimizer_success else "exhausted"),
            })
        except PricingError as exc:
            summary.update(status="refinement_failure", error=str(exc))
        for prefix, mask in (("train", train), ("test", ~train)):
            summary[f"{prefix}_rmse"] = _rmse(predictions["price_residual"].to_numpy(), mask)
            summary[f"{prefix}_scaled_rmse"] = _rmse(predictions["residual"].to_numpy(), mask)
            summary[f"refined_{prefix}_rmse"] = _rmse(
                predictions["refined_price_residual"].to_numpy(), mask)
            summary[f"refined_{prefix}_scaled_rmse"] = _rmse(
                predictions["refined_residual"].to_numpy(), mask)
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    return FitAttempt(label, result, summary, predictions)


def deterministic_starts() -> tuple[RoughHestonParams, ...]:
    """Three declared starts; perturbations use seed 42 and interior box clips."""
    first = SYNTHETIC_INITIALS["rough_heston"]
    rng = np.random.default_rng(42)
    starts = [first]
    for _ in range(2):
        values = {}
        for name, value in asdict(first).items():
            lo, hi = DEMO_BOUNDS[name]
            margin = 1e-6 * (hi - lo)
            values[name] = float(np.clip(value * rng.uniform(.5, 1.5), lo + margin, hi - margin))
        starts.append(RoughHestonParams(**values))
    return tuple(starts)


def select_attempt(attempts: Iterable[FitAttempt]) -> FitAttempt | None:
    """Choose the lowest finite training objective, without hiding bad status.

    Exhausted or unstable results remain eligible for an illustrative best fit;
    callers must still display ``ready``. A numerical failure has no result and
    cannot be selected. Ties preserve the declared start order.
    """
    candidates = [attempt for attempt in attempts
                  if attempt.result is not None and np.isfinite(attempt.result.objective)]
    return min(candidates, key=lambda attempt: attempt.result.objective) if candidates else None


def refinement_table(
    params: ModelParams, quotes: pd.DataFrame, configs: Sequence[PricingConfig],
) -> pd.DataFrame:
    """Compare numerical settings to the last supplied grid, keeping failures.

    ``max_change_vs_previous`` also shows sequential changes. A successful
    last grid is a comparison reference, not proof of numerical exactness.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("configs must contain at least one PricingConfig")
    target, scale, _ = _observation_arrays(quotes)
    rows, values = [], []
    previous = None
    for config in configs:
        started = time.perf_counter()
        row = {**asdict(config), "params": asdict(params), "status": "ok", "error": None}
        try:
            current = _evaluate(params, quotes, config)
            row.update({
                "price_rmse": float(np.sqrt(np.mean((current - target) ** 2))),
                "scaled_rmse": float(np.sqrt(np.mean(((current - target) / scale) ** 2))),
                "max_change_vs_previous": (float(np.max(np.abs(current - previous)))
                                           if previous is not None else None),
            })
        except PricingError as exc:
            current = None
            row.update(status="numerical_failure", error=str(exc), price_rmse=None,
                       scaled_rmse=None, max_change_vs_previous=None)
        row["elapsed_seconds"] = float(time.perf_counter() - started)
        values.append(current)
        rows.append(row)
        previous = current
    reference = values[-1]
    for row, current in zip(rows, values):
        change = (np.abs(current - reference)
                  if current is not None and reference is not None else None)
        row["max_change_vs_finest"] = float(np.max(change)) if change is not None else None
        row["stable_vs_finest"] = (bool(np.all(
            change <= STABILITY_ATOL + STABILITY_RTOL * np.abs(reference)))
            if change is not None else False)
    return pd.DataFrame(rows)


__all__ = [
    "SYNTHETIC_TRUTHS", "SYNTHETIC_INITIALS", "DEMO_BOUNDS", "COARSE_CONFIG",
    "SPY_CONFIG", "ACCURATE_SPY_CONFIG", "EXTENDED_CONFIG", "REFERENCE_CONFIG", "FitAttempt",
    "synthetic_surface", "fit_attempt", "deterministic_starts", "select_attempt",
    "refinement_table",
]
