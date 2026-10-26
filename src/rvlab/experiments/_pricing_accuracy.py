"""Reproduce the SPY truncation diagnosis and bounded numerical repair."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from rvlab.pricing import HestonParams, PricingConfig, PricingError, RoughHestonParams, pricing_diagnostics
from ._pricing_data import load_spy_demo
from ._pricing_fits import fit_attempt, refinement_table
from ._pricing_spy import _fingerprints


# Frozen snapshots from the original executed notebooks, not newly selected
# using held-out errors. They let the accuracy audit run without old CSV files.
HISTORICAL_SPY_PARAMS = {
    "heston": HestonParams(.0199782115770668, .249999999999844,
        .1694991802105628, .5999999999999999, -.8331920907620385),
    "rough_controlled": RoughHestonParams(.0204102372579376, .045, 1.,
        .2999999999999923, -.6845172910552568, .15),
    "rough_all_six": RoughHestonParams(.0204521713902941, .2499999999995925,
        .2227999224029153, .559707382304462, -.6531390004544548, .3129978921774439),
}


def _diagnostic_rows(params, *, config, **kwargs):
    """Retain a failed base grid as an experiment result, without fake prices."""
    started = time.perf_counter()
    try:
        return pricing_diagnostics(params, config=config, **kwargs).checks.copy(deep=True)
    except PricingError as exc:
        return pd.DataFrame([{
            "check": "base", "baseline_config": asdict(config),
            "config": asdict(config), **asdict(config),
            "status": "numerical_failure", "error": str(exc),
            "max_price_change": None, "numerical_stable": False,
            "elapsed_seconds": time.perf_counter() - started,
        }])




__all__ = ["HISTORICAL_SPY_PARAMS", "accuracy_demo"]


def accuracy_demo(context, **kwargs):
    """American replacement; historical European implementation is retired."""
    from ._pricing_american import american_spy_demo
    return american_spy_demo(context, study='accuracy', **kwargs)
