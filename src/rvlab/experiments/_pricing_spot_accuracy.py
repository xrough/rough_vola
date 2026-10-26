"""Truth-on-grid screens before expensive joint American calibration.

Every comparison uses the training mesh for both training and held-out prices.
The finer observations are finite-resolution synthetic references, not exact
American prices. Independent refinement remains a separate acceptance check.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time

import numpy as np
import pandas as pd

from rvlab.pricing import PricingError, evaluate_vanilla
from rvlab.pricing._vanilla_common import build_grid


PRICE_ATOL, PRICE_RTOL = .001, 1e-4
DELTA_ATOL, DELTA_RTOL = .002, .005
SPREAD_FRACTION, SPREAD_FLOOR = .1, .005


def model_spot_accuracy(context, model, truth, contracts, market,
                        observed_prices, observed_deltas, fit_config):
    """Screen one declared fitting grid and show the surrounding spot sequence.

    Reference-price errors and independent refinement must both pass on the
    chosen grid. No check changes the requested tolerances or chooses quotes.
    """
    reference, reference_delta = np.asarray(observed_prices, float), np.asarray(observed_deltas, float)
    if (reference.shape != (len(contracts),) or reference_delta.shape != reference.shape
            or not np.isfinite(reference+reference_delta).all()):
        raise ValueError("spot-accuracy references must be finite and contract aligned")
    scales = np.maximum(contracts.half_spread.to_numpy(float), SPREAD_FLOOR)
    train = contracts.split.eq("train").to_numpy()
    strikes = contracts.strike.to_numpy(float)
    rows, checks, errors, accepted = [], [], [], False
    steps = sorted({80, 120, 160, 200, 240, 320, fit_config.spot_steps})
    for intervals in steps:
        config = replace(fit_config, spot_steps=intervals)
        chosen = intervals == fit_config.spot_steps
        begin = time.perf_counter()
        row = {"model": model, "spot_steps": intervals,
            "variance_steps": config.variance_steps, "time_steps": config.time_steps,
            "spot_concentration": config.spot_concentration, "selected_grid": chosen,
            "status": "numerical_failure", "accepted": False, "error": None,
            "max_price_error": None, "max_delta_error": None,
            "max_price_error_ratio": None, "max_delta_error_ratio": None,
            "reference_price_pass": False, "reference_delta_pass": False,
            "price_stable": None, "delta_stable": None, "spread_stable": None,
            "pricing_evaluations": None, "config": asdict(config)}
        context.progress(f"Truth-grid screen · {model} · {intervals} spot intervals"
                         + (" · full independent checks" if chosen else ""))
        try:
            grid = build_grid(market, strikes[train], config, variance=True)
            result = evaluate_vanilla(truth, market=market, strike=strikes,
                maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
                exercise_style="american", config=config, grid=grid, check_refinement=chosen,
                price_atol=PRICE_ATOL, price_rtol=PRICE_RTOL,
                delta_atol=DELTA_ATOL, delta_rtol=DELTA_RTOL)
            price_error, delta_error = result.prices-reference, result.deltas-reference_delta
            allowance = np.minimum(PRICE_ATOL+PRICE_RTOL*np.abs(result.prices), SPREAD_FRACTION*scales)
            delta_allowance = DELTA_ATOL+DELTA_RTOL*np.abs(result.deltas)
            price_ratio, delta_ratio = np.abs(price_error)/allowance, np.abs(delta_error)/delta_allowance
            row.update(status="priced", max_price_error=float(np.max(np.abs(price_error))),
                max_delta_error=float(np.max(np.abs(delta_error))),
                max_price_error_ratio=float(price_ratio.max()), max_delta_error_ratio=float(delta_ratio.max()),
                reference_price_pass=bool(np.all(price_ratio <= 1.)),
                reference_delta_pass=bool(np.all(delta_ratio <= 1.)),
                pricing_evaluations=result.diagnostics["pricing_evaluations"],
                training_grid_sha256=result.diagnostics["grid_sha256"])
            for split, mask in (("train", train), ("heldout", ~train)):
                row[f"{split}_rmse"] = float(np.sqrt(np.mean(price_error[mask]**2)))
            if chosen:
                diagnostic = result.diagnostics["checks"].copy()
                spread_pass = True
                for axis, change in result.diagnostics["price_changes"].items():
                    fraction = float(np.max(np.asarray(change)/scales))
                    passed = fraction <= SPREAD_FRACTION
                    diagnostic.loc[diagnostic.axis.eq(axis), "spread_stable"] = passed
                    diagnostic.loc[diagnostic.axis.eq(axis), "max_half_spread_fraction"] = fraction
                    spread_pass &= passed
                checks.extend(diagnostic.assign(model=model, spot_steps=intervals,
                    variance_steps=config.variance_steps, time_steps=config.time_steps,
                    spot_concentration=config.spot_concentration).to_dict("records"))
                row.update(price_stable=bool(result.price_stable), delta_stable=bool(result.delta_stable),
                           spread_stable=bool(spread_pass))
                accepted = bool(result.price_stable and result.delta_stable and spread_pass
                                and row["reference_price_pass"] and row["reference_delta_pass"])
                row["accepted"] = accepted
            detail = contracts.assign(model=model, spot_steps=intervals,
                selected_grid=chosen, reference_price=reference, model_price=result.prices,
                reference_delta=reference_delta, delta=result.deltas,
                price_error=price_error, delta_error=delta_error,
                effective_half_spread=scales, price_allowance=allowance,
                delta_allowance=delta_allowance, price_error_ratio=price_ratio,
                delta_error_ratio=delta_ratio)
            errors.extend(detail.to_dict("records"))
        except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = time.perf_counter()-begin
        rows.append(row)
    return {"accuracy": rows, "refinement": checks, "contracts": errors, "accepted": accepted}


def spot_accuracy_figure(table):
    """The horizontal line is the unchanged contract-specific allowance."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, rows in table.groupby("model", sort=False):
        rows = rows.sort_values("spot_steps")
        label = "Heston curve" if model == "heston_curve" else "Heston"
        for ax, column in zip(axes, ("max_price_error_ratio", "max_delta_error_ratio")):
            line, = ax.plot(rows.spot_steps, rows[column], "o-", label=label)
            selected = rows.loc[rows.selected_grid]
            ax.scatter(selected.spot_steps, selected[column], s=110,
                       facecolors="none", edgecolors=line.get_color(), linewidths=2)
    for ax in axes:
        ax.axhline(1., color="black", ls="--", lw=1, label="Allowance")
        ax.set(xlabel="Spot intervals", yscale="log", ylabel="Worst reference error / allowance")
        ax.legend(fontsize=8)
    axes[0].set_title("Prices: native and spread allowances")
    axes[1].set_title("Stock deltas")
    figure.suptitle("Truth-on-grid screen; rings mark the declared fitting grid")
    return figure
