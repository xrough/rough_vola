"""Reproduce the preliminary American spot-mesh investigation.

Run a cheap independent Fourier check::

    .venv/bin/python examples/american_spot_grid_accuracy.py --suite fourier

Run every slow American trial (typically several minutes)::

    .venv/bin/python examples/american_spot_grid_accuracy.py --suite all

The naive grid is an explicitly labelled numerical control, implemented here
with exact node insertion and no pruning. It does not import archived source,
load fitted results, change the active pricing backend, or supply a fallback.
All outputs go into a new reproduction directory unless --output-dir is given.
The captured original trials and this script's new reproductions are distinct.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from rvlab.pricing import (
    CashDividend, DiscountCurve, DividendSchedule, FDConfig, FrozenFDGrid,
    HestonCurveParams, HestonParams, MeanReversionCurve, PricingConfig,
    SpotMarket, evaluate_vanilla, heston_price,
)
from rvlab.pricing._vanilla_common import build_grid

OUTPUT = ROOT / "notebooks/_output/pricing/research_v1/02_synthetic_calibration/spot_grid_trials_v2"


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict("records"))
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def contracts():
    return pd.DataFrame([
        {"contract_id": f"american-joint-{days}-{strike}", "maturity": days/365,
         "strike": float(strike), "is_call": strike >= 100,
         "split": "heldout" if strike in (97, 103) else "train",
         "half_spread": (.003, .008, .012, .018, .025, .035)[j]*(1+.25*i)}
        for i, days in enumerate((14, 30, 60))
        for j, strike in enumerate((94, 97, 99, 101, 103, 106))
    ])


def specification():
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.),
                       DividendSchedule((CashDividend(.08, 1.5),), 1.))
    truths = {
        "heston": HestonParams(.04, .045, 1.5, .3, -.6),
        "heston_curve": HestonCurveParams(.04, 1.5, .3, -.6,
            MeanReversionCurve((20/365, 45/365), (.035, .065, .045))),
    }
    return market, truths


def trial_grid(market, strikes, config, *, prune):
    """Current mesh or explicit pre-fix insertion control on the same domain."""
    corrected = build_grid(market, strikes, config, variance=True)
    if prune:
        return corrected
    center = market.spot
    maximum = config.spot_max_multiple*max(center, float(np.max(strikes)))
    inner = min(maximum, 8.*max(center, float(np.max(strikes))))
    width = config.spot_concentration*center
    coordinates = np.linspace(np.arcsinh(-center/width),
        np.arcsinh((inner-center)/width), config.spot_steps+1)
    spots = center+width*np.sinh(coordinates)
    spots[0], spots[-1] = 0., inner
    if maximum > inner:
        spacing = spots[-1]-spots[-2]
        spots = np.r_[spots, np.linspace(inner, maximum,
            int(np.ceil((maximum-inner)/spacing))+1)[1:]]
    spots = np.unique(np.r_[spots, center, np.asarray(strikes, float)])
    return FrozenFDGrid(spots, corrected.variance)


def _price(params, market, quotes, config, grid, *, refine=False, style="american"):
    return evaluate_vanilla(params, market=market,
        strike=quotes.strike.to_numpy(), maturity=quotes.maturity.to_numpy(),
        is_call=quotes.is_call.to_numpy(), exercise_style=style,
        config=config, grid=grid, check_refinement=refine)


def fourier_trials():
    """Independent no-dividend European reference isolates spatial pricing bias."""
    params = HestonParams(.04, .045, 1.5, .3, -.6)
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    quotes = contracts().iloc[:6].copy()
    reference_config = PricingConfig(512, 500., 1024)
    reference = heston_price(params, forward=100*np.exp(.04*14/365),
        discount=np.exp(-.04*14/365), strike=quotes.strike.to_numpy(),
        maturity=14/365, is_call=quotes.is_call.to_numpy(), config=reference_config)
    rows = []
    # Original broad-width trials plus the corrected default/pruned controls.
    for intervals in (80, 160, 240, 320):
        for width in (.1, .05, .02, .01):
            for prune in (False, True):
                config = FDConfig(intervals, 64, 200, spot_concentration=width)
                grid = trial_grid(market, quotes.strike.to_numpy(), config, prune=prune)
                started = time.perf_counter()
                result = _price(params, market, quotes, config, grid, style="european")
                error = np.asarray(result.prices)-reference
                rows.append({"spot_steps": intervals, "spot_concentration": width,
                    "pruned": prune, "maximum_price_error": float(np.max(abs(error))),
                    "errors": error.tolist(), "reference_prices": np.asarray(reference).tolist(),
                    "elapsed_seconds": time.perf_counter()-started,
                    "config": asdict(config), "grid_sha256": result.diagnostics["grid_sha256"]})
    return {"exercise_style": "european", "cash_dividends": [],
        "reference": "Independent characteristic-function Fourier pricing",
        "reference_config": asdict(reference_config), "trials": rows}


def american_trials():
    """Old/pruned controls and width trials against independently finer prices."""
    quotes = contracts()
    market, truths = specification()
    training_strikes = quotes.loc[quotes.split.eq("train"), "strike"].to_numpy()
    scales = np.maximum(quotes.half_spread.to_numpy(), .005)
    result = {"exercise_style": "american", "market": asdict(market),
              "contracts": quotes.to_dict("records"), "trials": [], "references": {}}
    for model, params in truths.items():
        generation = FDConfig(640, 128, 600)
        grid = trial_grid(market, training_strikes, generation, prune=True)
        begin = time.perf_counter()
        reference = _price(params, market, quotes, generation, grid)
        result["references"][model] = {"params": asdict(params),
            "config": asdict(generation), "prices": reference.prices,
            "elapsed_seconds": time.perf_counter()-begin,
            "grid_sha256": reference.diagnostics["grid_sha256"]}
        cases = [(n, .1, prune) for n in (80, 160, 240, 320) for prune in (False, True)]
        if model == "heston":
            cases += [(n, width, True) for n in (120, 160, 240) for width in (.1, .05, .02)]
        for intervals, width, prune in cases:
            config = FDConfig(intervals, 48, 180, spot_concentration=width)
            grid = trial_grid(market, training_strikes, config, prune=prune)
            begin = time.perf_counter()
            values = _price(params, market, quotes, config, grid)
            error = np.asarray(values.prices)-reference.prices
            result["trials"].append({"model": model, "pruned": prune,
                "config": asdict(config), "prices": values.prices, "errors": error,
                "maximum_price_error": float(np.max(abs(error))),
                "maximum_effective_halfspread_fraction": float(np.max(abs(error)/scales)),
                "elapsed_seconds": time.perf_counter()-begin,
                "grid_sha256": values.diagnostics["grid_sha256"]})
            print(f"American {model}: {intervals} spot, width {width}, prune={prune}", flush=True)
    return result


def refinement_trials():
    """Check all 18 contracts on a frozen training mesh and independent refinements."""
    quotes = contracts()
    market, truths = specification()
    training_strikes = quotes.loc[quotes.split.eq("train"), "strike"].to_numpy()
    scales = np.maximum(quotes.half_spread.to_numpy(), .005)
    rows = []
    for model, params in truths.items():
        for intervals, width in ((160, .05), (160, .02), (192, .05), (240, .05)):
            config = FDConfig(intervals, 48, 180, spot_concentration=width)
            grid = build_grid(market, training_strikes, config)
            begin = time.perf_counter()
            values = _price(params, market, quotes, config, grid, refine=True)
            fractions = {axis: float(np.max(change/scales))
                for axis, change in values.diagnostics["price_changes"].items()}
            rows.append({"model": model, "config": asdict(config),
                "prices": values.prices, "deltas": values.deltas,
                "price_stable": values.price_stable, "delta_stable": values.delta_stable,
                "spread_stable": bool(fractions and all(x <= .1 for x in fractions.values())),
                "max_effective_halfspread_fractions": fractions,
                "diagnostics": values.diagnostics, "elapsed_seconds": time.perf_counter()-begin})
            print(f"Refinement {model}: {intervals} spot, width {width}; price={values.price_stable}", flush=True)
    return {"market": asdict(market), "contracts": quotes.to_dict("records"),
        "thresholds": {"price_atol": .001, "price_rtol": 1e-4,
            "delta_atol": .002, "delta_rtol": .005,
            "effective_halfspread_fraction": .1, "spread_floor": .005}, "trials": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("fourier", "american", "refinement", "all"), default="fourier")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output_dir or OUTPUT / "reproductions" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for name, function in (("fourier", fourier_trials), ("american", american_trials),
                           ("refinement", refinement_trials)):
        if args.suite in (name, "all"):
            _write(directory / f"{name}.json", function())
    sources = [Path(__file__), *sorted((ROOT / "src/rvlab/pricing").glob("*.py"))]
    _write(directory / "manifest.json", {"kind": "reproduction", "suite": args.suite,
        "created_at": stamp, "interpreter": sys.executable,
        "elapsed_seconds": time.perf_counter()-started,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(directory.glob("*.json"))},
        "limitations": ["American generation uses finer instances of the same finite-difference engine, not exact truth.",
            "Truth refinement acceptance does not establish convergence or accuracy at fitted parameters."]})
    print(directory)


if __name__ == "__main__":
    main()
