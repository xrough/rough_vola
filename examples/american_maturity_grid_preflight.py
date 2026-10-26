"""Reproduce the independent numerical gate before adding longer maturities.

This is an axis-refinement screen, not the final fine-observation audit. Run
from MVP with, for example::

    OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/american_maturity_grid_preflight.py --config original

Previously captured trials stay intact unless an explicit new output is given.
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

from rvlab.experiments._pricing_american_identification import (
    _engine_fingerprint, _jsonable, joint_market, joint_specifications,
)
from rvlab.experiments._pricing_spot_accuracy import (
    PRICE_ATOL, PRICE_RTOL, DELTA_ATOL, DELTA_RTOL, SPREAD_FLOOR, SPREAD_FRACTION,
)
from rvlab.pricing import FDConfig, evaluate_vanilla
from rvlab.pricing._vanilla_common import build_grid


def long_contracts():
    """The original short-tenor contracts form an unchanged economic prefix."""
    return pd.DataFrame([
        {"contract_id": f"american-long-{days}-{strike}", "maturity": days/365,
         "strike": float(strike), "is_call": strike >= 100,
         "split": "heldout" if strike in (97, 103) else "train",
         "half_spread": (.003, .008, .012, .018, .025, .035)[j]*(1+.25*i)}
        for i, days in enumerate((14, 30, 60, 90, 180, 365))
        for j, strike in enumerate((94, 97, 99, 101, 103, 106))])


def preflight(config):
    contracts = long_contracts()
    market = joint_market()
    truth = joint_specifications()[0]["heston_curve"]
    grid = build_grid(market, contracts.loc[contracts.split.eq("train"), "strike"].to_numpy(),
                      config, variance=True)
    begin = time.perf_counter()
    result = evaluate_vanilla(truth, market=market, strike=contracts.strike.to_numpy(),
        maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
        exercise_style="american", config=config, grid=grid, check_refinement=True,
        price_atol=PRICE_ATOL, price_rtol=PRICE_RTOL,
        delta_atol=DELTA_ATOL, delta_rtol=DELTA_RTOL)
    checks = result.diagnostics["checks"].copy()
    scale = np.maximum(contracts.half_spread.to_numpy(), SPREAD_FLOOR)
    for axis, change in result.diagnostics["price_changes"].items():
        fraction = float(np.max(np.asarray(change)/scale))
        checks.loc[checks.axis.eq(axis), "max_halfspread_fraction"] = fraction
        checks.loc[checks.axis.eq(axis), "spread_stable"] = fraction <= SPREAD_FRACTION
    accepted = bool(result.price_stable and result.delta_stable and checks.spread_stable.all())
    return {"config": asdict(config), "checks": checks.to_dict("records"),
        "contracts": contracts.to_dict("records"), "prices": result.prices,
        "deltas": result.deltas, "price_changes": result.diagnostics["price_changes"],
        "price_stable": result.price_stable, "delta_stable": result.delta_stable,
        "independent_axes_accepted": accepted, "elapsed_seconds": time.perf_counter()-begin,
        "acceptance": {"price_atol": PRICE_ATOL, "price_rtol": PRICE_RTOL,
            "delta_atol": DELTA_ATOL, "delta_rtol": DELTA_RTOL,
            "spread_fraction": SPREAD_FRACTION, "spread_floor": SPREAD_FLOOR},
        "engine_sha256": _engine_fingerprint(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("original", "variance", "spatial", "candidate"), default="original")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    intervals = {"original": (192, 48, 180), "variance": (192, 64, 180),
                 "spatial": (240, 64, 180), "candidate": (240, 64, 600)}
    config = FDConfig(*intervals[args.config], spot_concentration=.05)
    output = args.output or Path("notebooks/_output/pricing/research_v1/02_synthetic_calibration/"
        "maturity_extension_v1/reproduction")/f"preflight_{args.config}.json"
    if output.exists():
        raise SystemExit(f"Preserved trial exists at {output}; choose --output for a new run")
    print("Independent truth-grid audit", asdict(config), flush=True)
    record = preflight(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(record), indent=2, allow_nan=False)+"\n")
    print(pd.DataFrame(record["checks"]).to_string(index=False))
    print(f"Wrote {output}; elapsed {record['elapsed_seconds']:.1f} s")
