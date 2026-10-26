"""Truth-on-grid audit for the longer-maturity identification experiment.

The finite-resolution synthetic observations and the independent refinement
tests answer different questions. Both must pass before a fitting grid is
accepted. Every contract is evaluated on the mesh made from training strikes;
held-out strikes never change the base objective.
"""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import pandas as pd

from rvlab.pricing import PricingError, evaluate_vanilla
from rvlab.pricing._vanilla_common import build_grid
from ._pricing_spot_accuracy import (
    PRICE_ATOL, PRICE_RTOL, DELTA_ATOL, DELTA_RTOL,
    SPREAD_FRACTION, SPREAD_FLOOR,
)


def maturity_truth_accuracy(context, truth, contracts, market, fit_config,
                            observed_prices, observed_deltas, *, precomputed_result=None):
    """Audit one declared fitting grid without changing any tolerances.

    ``observed_prices`` and ``observed_deltas`` must be generated independently
    at the declared finer resolution and retain the exact input row ordering.
    Numerical failures remain explicit records and never select another grid.
    A workflow may supply ``precomputed_result`` only after it has verified the
    cached engine, contracts, market, parameters, versions and settings. The
    full acceptance screen is recomputed here from its prices and diagnostics.
    """
    needed = {"contract_id", "strike", "maturity", "is_call", "split", "half_spread"}
    if not isinstance(contracts, pd.DataFrame) or not needed.issubset(contracts):
        raise ValueError("maturity accuracy contracts require " + ", ".join(sorted(needed)))
    if contracts.empty or contracts.contract_id.duplicated().any():
        raise ValueError("maturity accuracy contract identifiers must be nonempty and unique")
    if not contracts.split.isin(("train", "heldout")).all() or not contracts.split.eq("train").any():
        raise ValueError("maturity accuracy requires training rows and train/heldout split labels")
    reference = np.asarray(observed_prices, float)
    reference_delta = np.asarray(observed_deltas, float)
    if (reference.shape != (len(contracts),) or reference_delta.shape != reference.shape
            or not np.isfinite(reference + reference_delta).all()):
        raise ValueError("maturity accuracy references must be finite and contract aligned")
    spreads = contracts.half_spread.to_numpy(float)
    if not np.isfinite(spreads).all() or (spreads <= 0).any():
        raise ValueError("maturity accuracy half spreads must be finite and positive")
    scales = np.maximum(spreads, SPREAD_FLOOR)
    train = contracts.split.eq("train").to_numpy()
    strikes = contracts.strike.to_numpy(float)
    started = time.perf_counter()
    row = {"status": "numerical_failure", "accepted": False,
        "reference_price_pass": False, "reference_delta_pass": False,
        "price_stable": False, "delta_stable": False, "spread_stable": False,
        "config": asdict(fit_config), "error": None, "pricing_evaluations": 0,
        "contract_count": len(contracts), "training_count": int(train.sum()),
        "max_maturity": float(contracts.maturity.max())}
    detail = pd.DataFrame()
    checks = pd.DataFrame()
    by_maturity = pd.DataFrame()
    context.progress("Long-maturity truth grid · independent spot/time/variance/domain/joint checks")
    try:
        grid = build_grid(market, strikes[train], fit_config, variance=True)
        result = precomputed_result
        if result is None:
            result = evaluate_vanilla(truth, market=market, strike=strikes,
                maturity=contracts.maturity.to_numpy(float), is_call=contracts.is_call.to_numpy(),
                exercise_style="american", config=fit_config, grid=grid, check_refinement=True,
                price_atol=PRICE_ATOL, price_rtol=PRICE_RTOL,
                delta_atol=DELTA_ATOL, delta_rtol=DELTA_RTOL)
        prices, deltas = np.asarray(result.prices), np.asarray(result.deltas)
        if (prices.shape != reference.shape or deltas.shape != reference.shape
                or not np.isfinite(prices+deltas).all()):
            raise PricingError("truth-grid prices and deltas must be finite and contract aligned")
        error, delta_error = prices-reference, deltas-reference_delta
        allowance = np.minimum(PRICE_ATOL+PRICE_RTOL*np.abs(prices), SPREAD_FRACTION*scales)
        delta_allowance = DELTA_ATOL+DELTA_RTOL*np.abs(deltas)
        price_ratio, delta_ratio = np.abs(error)/allowance, np.abs(delta_error)/delta_allowance
        checks = result.diagnostics["checks"].copy()
        checks["max_half_spread_fraction"] = np.nan
        checks["spread_stable"] = False
        for axis, change in result.diagnostics["price_changes"].items():
            fraction = float(np.max(np.asarray(change)/scales))
            checks.loc[checks.axis.eq(axis), "max_half_spread_fraction"] = fraction
            checks.loc[checks.axis.eq(axis), "spread_stable"] = fraction <= SPREAD_FRACTION
        complete = set(checks.axis) == {"spot", "time", "variance", "domain", "joint"}
        spread_pass = bool(complete and checks.spread_stable.all())
        row.update(status="priced", reference_price_pass=bool(np.all(price_ratio <= 1.)),
            reference_delta_pass=bool(np.all(delta_ratio <= 1.)),
            price_stable=bool(result.price_stable), delta_stable=bool(result.delta_stable),
            spread_stable=spread_pass, max_price_error=float(np.abs(error).max()),
            max_delta_error=float(np.abs(delta_error).max()),
            max_price_error_ratio=float(price_ratio.max()), max_delta_error_ratio=float(delta_ratio.max()),
            pricing_evaluations=result.diagnostics["pricing_evaluations"],
            training_grid_sha256=result.diagnostics["grid_sha256"])
        row["accepted"] = bool(complete and all(row[name] for name in
            ("reference_price_pass", "reference_delta_pass", "price_stable", "delta_stable", "spread_stable")))
        detail = contracts.assign(reference_price=reference, model_price=prices,
            reference_delta=reference_delta, delta=deltas, price_error=error, delta_error=delta_error,
            effective_half_spread=scales, price_allowance=allowance, delta_allowance=delta_allowance,
            price_error_ratio=price_ratio, delta_error_ratio=delta_ratio)
        by_maturity = detail.groupby("maturity", sort=True).agg(
            max_price_error=("price_error", lambda v: float(np.abs(v).max())),
            max_price_error_ratio=("price_error_ratio", "max"),
            max_delta_error=("delta_error", lambda v: float(np.abs(v).max())),
            max_delta_error_ratio=("delta_error_ratio", "max")).reset_index()
        by_maturity["reference_price_pass"] = by_maturity.max_price_error_ratio <= 1.
        by_maturity["reference_delta_pass"] = by_maturity.max_delta_error_ratio <= 1.
    except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_seconds"] = time.perf_counter()-started
    return {"accuracy": [row], "refinement": checks.to_dict("records"),
        "contracts": detail.to_dict("records"), "by_maturity": by_maturity.to_dict("records"),
        "accepted": bool(row["accepted"])}


__all__ = ["maturity_truth_accuracy"]
