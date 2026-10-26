"""Recheck selected historical calibration prices with the current FD source.

This is a source-compatibility check, not a new fit or a repeat of historical
refinement tests. Selection uses the recorded training decision exclusively.
The same training-strike mesh and recorded fitted parameters are reconstructed.
"""
from time import perf_counter

import numpy as np
import pandas as pd

from rvlab.pricing import (CashDividend, DiscountCurve, DividendSchedule, FDConfig,
    HestonCurveParams, HestonParams, MeanReversionCurve, SpotMarket, evaluate_vanilla)
from rvlab.pricing._vanilla_common import build_grid


def _market(record):
    return SpotMarket(record["spot"],
        DiscountCurve(tuple(record["discount_curve"]["times"]),
                      tuple(record["discount_curve"]["discounts"])),
        DividendSchedule(tuple(CashDividend(**event) for event in record["dividends"]["events"]),
                         record["dividends"]["coverage"]))


def _params(model, values, knots):
    if model == "heston":
        return HestonParams(**values)
    if model != "heston_curve":
        raise ValueError("Historical compatibility check supports Heston and Heston curves")
    levels = tuple(values[f"theta_{i}"] for i in range(len(knots) + 1))
    return HestonCurveParams(values["v0"], values["kappa"], values["vol_of_vol"],
                            values["rho"], MeanReversionCurve(tuple(knots), levels))


def reprice_selected_calibration(tables, manifest, *, roundoff_allowance=2e-11):
    """Reprice all four recorded model/dataset selections, including exhaustion."""
    if not np.isfinite(roundoff_allowance) or roundoff_allowance <= 0:
        raise ValueError("The implementation roundoff allowance must be positive")
    study = manifest["american_joint"]
    market = _market(study["market"])
    selections = tables["american_joint_selection"]
    summary = tables["american_joint_summary"]
    parameters = tables["american_joint_parameters"]
    predictions = tables["american_joint_predictions"]
    expected = {(model, dataset) for model in ("heston", "heston_curve")
                for dataset in ("clean", "noisy")}
    if (len(selections) != 4 or set(zip(selections.model, selections.dataset)) != expected or
            selections.heldout_used_for_selection.any()):
        raise ValueError("All original training-only model/dataset selections are required")
    rows = []
    for selected in selections.to_dict("records"):
        model, dataset = selected["model"], selected["dataset"]
        stage = "continuation" if selected["continuation_executed"] else "exploratory"
        attempt = summary.loc[(summary.model == model) & (summary.dataset == dataset) &
            (summary.start == selected["selected_start"]) & (summary.stage == stage)]
        if len(attempt) != 1:
            raise ValueError("Recorded training selection does not identify exactly one attempt")
        attempt = attempt.iloc[0]
        name = attempt.attempt_id
        fitted = parameters.loc[parameters.attempt_id.eq(name)]
        if fitted.empty or fitted.parameter.duplicated().any():
            raise ValueError("Selected fitted parameters are incomplete or duplicated")
        value = _params(model, dict(zip(fitted.parameter, fitted.fitted)), study["curve_knot_times"])
        quotes = predictions.loc[predictions.attempt_id.eq(name)]
        if len(quotes) != len(study["contracts"]) or quotes.contract_id.duplicated().any():
            raise ValueError("The full original contract ordering is required for repricing")
        config = FDConfig(**study["continuation_config" if stage == "continuation" else "fit_config"])
        grid = build_grid(market, quotes.loc[quotes.split.eq("train"), "strike"].to_numpy(),
                          config, variance=True)
        start = perf_counter()
        values = evaluate_vanilla(value, market=market, strike=quotes.strike.to_numpy(),
            maturity=quotes.maturity.to_numpy(), is_call=quotes.is_call.to_numpy(),
            exercise_style="american", config=config, grid=grid, check_refinement=False)
        price_error = float(np.max(abs(values.prices - quotes.model_price.to_numpy())))
        delta_error = float(np.max(abs(values.deltas - quotes.delta.to_numpy())))
        grid_matches = values.diagnostics["grid_sha256"] == attempt.training_grid_sha256
        passed = grid_matches and max(price_error, delta_error) <= roundoff_allowance
        rows.append(dict(model=model, dataset=dataset, attempt_id=name,
            contracts=len(quotes), heldout_used_for_selection=False,
            max_price_difference=price_error, max_delta_difference=delta_error,
            grid_matches_recorded_training=grid_matches, roundoff_allowance=roundoff_allowance,
            passed=passed, elapsed_seconds=perf_counter() - start,
            calibration_recomputed=False, refinement_recomputed=False,
            scope="Current-source base-grid repricing of all contracts for the recorded training-selected fit"))
        if not passed:
            raise ArithmeticError(f"Current calibration repricing differs from recorded source: {name}; "
                f"price={price_error}, delta={delta_error}, grid_matches={grid_matches}")
    return pd.DataFrame(rows)
