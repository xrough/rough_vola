"""Small SPY demonstration; option-implied forwards never become stock data."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from rvlab.pricing import (
    BlackScholesParams, HestonParams, PricingConfig, PricingError,
    RoughHestonParams, price,
)
from rvlab.pricing.diagnostics import pricing_diagnostics
from ._pricing_data import load_spy_demo, implied_volatility
from ._pricing_fits import ACCURATE_SPY_CONFIG, COARSE_CONFIG, SPY_CONFIG, fit_attempt


def _fingerprints(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            if (root / name).is_file() else None
            for name in ("readiness.json", "manifest.json")}








__all__ = ["spy_demo"]


def spy_demo(context, **kwargs):
    """American replacement; historical European implementation is retired."""
    from ._pricing_american import american_spy_demo
    return american_spy_demo(context, study='pricing', **kwargs)
