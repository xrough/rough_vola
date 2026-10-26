"""Synthetic European pricing and calibration, without market data or downloads.

Run from the project root::

    python examples/pricing_calibration.py
    python examples/pricing_calibration.py --model rough_heston

The rough-Heston example is a controlled one-parameter recovery: its other
parameters are fixed at their known synthetic values. This demonstrates the
numerical workflow, not empirical parameter identification or trading value.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Permit direct execution from a source checkout; no sibling repositories.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvlab.pricing import (
    BlackScholesParams, HestonParams, RoughHestonParams, PricingConfig,
    calibrate_surface, price,
)


def run_example(model: str) -> bool:
    maturity = np.repeat(np.array([14, 30, 60]) / 365.0, 5)
    strike = np.tile(np.array([94., 97., 100., 103., 106.]), 3)
    forward = 100.0 * np.exp((0.03 - 0.01) * maturity)
    discount = np.exp(-0.03 * maturity)
    is_call = strike >= forward
    config = PricingConfig(quadrature_nodes=192, fourier_cutoff=160., riccati_steps=512)
    generation_config = PricingConfig(384, 320., 2048)
    bounds = None
    if model == "bs":
        truth, initial = BlackScholesParams(.20), BlackScholesParams(.30)
    elif model == "heston":
        truth = HestonParams(v0=.04, theta=.045, kappa=1.5, vol_of_vol=.30, rho=-.60)
        initial = HestonParams(v0=.035, theta=.05, kappa=1.2, vol_of_vol=.25, rho=-.45)
    else:
        truth = RoughHestonParams(v0=.04, theta=.045, kappa=1.,
                                  vol_of_vol=.12, rho=-.60, hurst=.15)
        initial = RoughHestonParams(**{**asdict(truth), "v0": .03})
        bounds = {name: (value, value) for name, value in asdict(truth).items()
                  if name != "v0"}
        bounds["v0"] = (.005, .15)

    observed = price(truth, forward=forward, discount=discount, strike=strike,
                     maturity=maturity, is_call=is_call, config=generation_config)
    half_spread = np.minimum(.02, observed * .25)
    quotes = pd.DataFrame({
        "contract_id": [f"synthetic-{i}" for i in range(len(strike))],
        "strike": strike, "maturity": maturity, "is_call": is_call,
        "market_price": observed, "bid": observed - half_spread,
        "ask": observed + half_spread,
    })
    result = calibrate_surface(quotes, initial, forward=forward, discount=discount,
                               bounds=bounds, config=config, max_nfev=120,
                               stability_atol=.001, stability_rtol=1e-4)
    print(f"\n{model}: {len(quotes)} synthetic quotes")
    print("Generation settings:", asdict(generation_config))
    print("Fit settings:", asdict(config))
    print("Refinement tolerance: .001 + 1e-4 * abs(price), plus 10% effective half-spread")
    if model == "rough_heston":
        print("Controlled v0 recovery; the other five parameters are fixed.")
    print("Parameters:", asdict(result.params))
    print(f"Optimizer converged: {result.success}; numerical stability: {result.numerical_stable}")
    print(f"Price RMSE: {result.price_rmse:.6g}; max refinement change: {result.max_price_change:.6g}")
    effective_half_spread = np.maximum(half_spread, .005)
    spread_change_limit = .1 * float(effective_half_spread.min())
    spread_stable = bool(np.isfinite(result.max_price_change)
                         and result.max_price_change <= spread_change_limit)
    demo_ready = result.ready and spread_stable
    print(f"Conservative spread check: {spread_stable}; maximum change <= {spread_change_limit:.6g}")
    print(f"Ready under these tolerances: {demo_ready}")
    print(result.fitted_quotes[["contract_id", "target_price", "model_price",
                                "refined_model_price", "price_residual"]].head().to_string(index=False))
    return demo_ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "bs", "heston", "rough_heston"), default="all")
    args = parser.parse_args()
    models = ("bs", "heston", "rough_heston") if args.model == "all" else (args.model,)
    outcomes = [run_example(model) for model in models]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
