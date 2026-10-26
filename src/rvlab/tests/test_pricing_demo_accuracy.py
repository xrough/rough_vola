"""Accuracy demonstrations preserve numerical failures without hiding input errors."""

from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import _pricing_accuracy as accuracy
from rvlab.pricing import HestonParams, PricingConfig, PricingError


CONFIG = PricingConfig(16, 40., 32)
PARAMS = HestonParams(.04, .045, 1., .2, -.5)
INPUTS = dict(forward=100., discount=.99, strike=[95., 100., 105.], maturity=.1)


def test_failed_diagnostic_base_becomes_an_explicit_failed_row(monkeypatch):
    def fail(*args, **kwargs):
        raise PricingError("base pricing diagnostic failed: integration violated price bounds")

    monkeypatch.setattr(accuracy, "pricing_diagnostics", fail)
    table = accuracy._diagnostic_rows(PARAMS, config=CONFIG, **INPUTS)
    assert len(table) == 1
    row = table.iloc[0]
    assert row.check == "base" and row.status == "numerical_failure"
    assert row.config == row.baseline_config == asdict(CONFIG)
    assert "integration violated price bounds" in row.error
    assert not row.numerical_stable
    assert pd.isna(row.max_price_change)
    assert row.elapsed_seconds >= 0.
    assert "model_price" not in table


def test_successful_checks_are_preserved_and_detached(monkeypatch):
    checks = pd.DataFrame({"check": ["quadrature", "cutoff_cumulative"],
        "status": ["stable", "unstable"], "error": [None, None],
        "max_price_change": [1e-10, .02], "numerical_stable": [True, False]})
    original = checks.copy(deep=True)
    seen = []

    def success(params, **kwargs):
        seen.append((params, kwargs))
        return SimpleNamespace(checks=checks)

    monkeypatch.setattr(accuracy, "pricing_diagnostics", success)
    table = accuracy._diagnostic_rows(PARAMS, config=CONFIG, **INPUTS)
    pd.testing.assert_frame_equal(table, original)
    assert seen[0][0] is PARAMS and seen[0][1] == {"config": CONFIG, **INPUTS}
    table.loc[0, "status"] = "changed"
    pd.testing.assert_frame_equal(checks, original)


def test_malformed_inputs_are_not_misrepresented_as_numerical_failures(monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("strike must be finite and positive")

    monkeypatch.setattr(accuracy, "pricing_diagnostics", invalid)
    with pytest.raises(ValueError, match="strike must be finite and positive"):
        accuracy._diagnostic_rows(PARAMS, config=CONFIG, **INPUTS)
