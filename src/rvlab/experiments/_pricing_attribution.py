"""Controlled SPY attribution: bounds, exercise/forward inputs and model shape."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rvlab.pricing import PricingConfig, RoughHestonParams
from ._pricing_benchmarks import forward_sensitivity, quadratic_smile_benchmark
from ._pricing_bounds import (
    HESTON_BASELINE_START, HESTON_EXPANDED_START, HESTON_EXPANDED_BOUNDS,
    heston_bounds_study,
)
from ._pricing_data import load_spy_demo
from ._pricing_exercise import exercise_adjusted_surface
from ._pricing_fits import fit_attempt, select_attempt
from ._pricing_spy import _fingerprints


# Reuse the earlier training-only fit as an explicit warm start, not as a
# newly selected start. The investigation estimates no independent stock data.
ROUGH_BASELINE_START = RoughHestonParams(.0204296763, .25, .2183764,
                                         .548515, -.6570358, .31031935)


def _row(attempt, inputs, model, box):
    return {**attempt.summary, "input_scenario": inputs, "model": model, "box": box}


def _aggregate(predictions, *, label, inputs, model, ready=None, status=None):
    target = predictions.target_price.to_numpy(float)
    residual = predictions.model_price.to_numpy(float) - target
    scales = predictions.price_scale.to_numpy(float)
    training = predictions.split.eq("train").to_numpy()
    row = {"label": label, "input_scenario": inputs, "model": model,
           "ready": ready, "status": status}
    for name, mask in (("train", training), ("test", ~training)):
        valid = np.any(mask) and np.all(np.isfinite(residual[mask]))
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(residual[mask]**2))) if valid else None
        row[f"{name}_scaled_rmse"] = float(np.sqrt(np.mean((residual[mask]/scales[mask])**2))) if valid else None
        half = .5 * (predictions.ask.to_numpy(float) - predictions.bid.to_numpy(float))
        row[f"{name}_interval_fraction"] = float(np.mean(np.abs(residual[mask]) <= half[mask] + 1e-10)) if valid else None
    return row


def _input_checks(quotes, anchors):
    rows = []
    for expiry, group in quotes.groupby("expiry_date", sort=True):
        group = group.sort_values("strike")
        a = anchors.loc[anchors.expiry_date == expiry]
        strike, discount = group.strike.to_numpy(float), float(group.discount.iloc[0])
        forward = float(group.forward.iloc[0])
        # This is a local European-equivalent diagnostic under the supplied
        # proxy, not an arbitrage claim about the original American contracts.
        calls = group.market_price.to_numpy(float) + np.where(
            group.is_call.to_numpy(), 0., discount * (forward - strike))
        slopes = np.diff(calls) / np.diff(strike)
        implied_discount = -float(np.polyfit(a.strike,
            a.market_price_call - a.market_price_put, 1)[0])
        rows.append({"expiry_date": expiry, "maturity": float(group.maturity.iloc[0]),
            "discount": discount, "continuous_rate": -np.log(discount)/float(group.maturity.iloc[0]),
            "quote_interval_age_seconds_max": float(group.quote_age_seconds.max()),
            "minimum_bid_size": int(group.bid_size.min()), "minimum_ask_size": int(group.ask_size.min()),
            "minimum_half_spread": float(group.half_spread.min()),
            "maximum_half_spread": float(group.half_spread.max()),
            "cross_strike_parity_implied_discount": implied_discount,
            "local_converted_curve_decreasing": bool(np.all(slopes <= 1e-10)),
            "local_converted_curve_convex": bool(np.all(np.diff(slopes) >= -1e-10)),
            "local_converted_curve_slope_bounds": bool(np.all((slopes >= -discount-1e-10) & (slopes <= 1e-10))),
            "quote_change_freshness_known": False})
    return pd.DataFrame(rows)




__all__ = ["ROUGH_BASELINE_START", "attribution_demo"]


def attribution_demo(context, **kwargs):
    """American replacement; historical European implementation is retired."""
    from ._pricing_american import american_spy_demo
    return american_spy_demo(context, study='attribution', **kwargs)
