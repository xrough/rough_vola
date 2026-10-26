"""Bound-profile bookkeeping and local curve sensitivity diagnostics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_curve_kappa as study
from rvlab.experiments._pricing_fits import FitAttempt, synthetic_surface
from rvlab.experiments._pricing_variance_curve import HESTON_START, _as_curve
from rvlab.pricing import PricingConfig, replace_parameters


@pytest.fixture
def quotes():
    return synthetic_surface("heston", config=PricingConfig(32, 60., 32))


def test_relative_curve_jacobian_uses_only_training_contracts(quotes):
    params = _as_curve(HESTON_START, sorted(quotes.maturity.unique())[:-1])
    params = replace_parameters(params, theta_2=0.)
    kwargs = dict(config=PricingConfig(48, 100., 32), parameters=["v0", "theta_0", "theta_2"])
    result = study.curve_relative_price_jacobian(params, quotes, **kwargs)
    damaged = quotes.copy(deep=True)
    heldout = damaged.split.eq("heldout")
    damaged.loc[heldout, "strike"] = -1.
    repeated = study.curve_relative_price_jacobian(params, damaged, **kwargs)
    np.testing.assert_array_equal(result["jacobian"], repeated["jacobian"])
    assert result["n_training"] == 12
    assert result["finite_difference_schemes"]["theta_2"] == "forward"
    assert result["rank"] == 3
    assert np.asarray(result["jacobian"]).shape == (12, 3)
    assert result["condition_number"] > 1


@pytest.mark.parametrize("kwargs,match", [
    ({"relative_step": 0.}, "relative_step"),
    ({"relative_step": True}, "relative_step"),
    ({"parameters": ["theta_0", "theta_0"]}, "parameters"),
    ({"parameters": ["curve"]}, "parameters"),
])
def test_curve_jacobian_rejects_invalid_controls(quotes, kwargs, match):
    params = _as_curve(HESTON_START, sorted(quotes.maturity.unique())[:-1])
    with pytest.raises(ValueError, match=match):
        study.curve_relative_price_jacobian(params, quotes, **kwargs)


def test_profile_selection_uses_training_objective_and_preserves_attempts(monkeypatch, quotes):
    calls = []

    def fake_fit(label, source, initial, **kwargs):
        calls.append((label, initial, kwargs))
        continuation = label.endswith("continuation")
        objective = 1. if continuation else 2.
        params = replace_parameters(initial, theta_0=.1 if continuation else .2)
        return FitAttempt(label, SimpleNamespace(params=params, objective=objective),
            {"status": "exhausted" if continuation else "converged", "active_bounds": {},
             "training_objective": objective, "test_rmse": 999. if continuation else 0.},
            source.copy(deep=True))

    monkeypatch.setattr(study, "fit_attempt", fake_fit)
    monkeypatch.setattr(study, "curve_relative_price_jacobian", lambda *a, **k: {
        "status": "ok", "condition_number": 5., "singular_values": [5., 1.]})
    before = quotes.copy(deep=True)
    result = study.heston_curve_kappa_study(quotes, kappa_values=(5., 10.),
                                           eta_caps=(2.,), include_joint=False)
    profile = result["tables"]["kappa_profile"]
    assert len(result["attempts"]) == 3
    assert profile.iloc[1].selected_label.endswith("continuation")
    assert profile.iloc[1].status == "exhausted"
    assert profile.iloc[1].test_rmse == 999.
    assert len(result["tables"]["kappa_all_attempts"]) == 3
    assert calls[2][1].curve.levels[0] == .2
    assert calls[2][2]["bounds"]["kappa"] == (10., 10.)
    pd.testing.assert_frame_equal(quotes, before)


@pytest.mark.parametrize("kwargs,match", [
    ({"max_nfev": 81}, "max_nfev"),
    ({"max_nfev": True}, "max_nfev"),
    ({"kappa_values": ()}, "kappa_values"),
    ({"kappa_values": (5., 5.)}, "kappa_values"),
    ({"eta_caps": (.001,)}, "eta_caps"),
    ({"include_joint": 1}, "include_joint"),
])
def test_profile_controls_rejected_before_fitting(quotes, kwargs, match):
    with pytest.raises(ValueError, match=match):
        study.heston_curve_kappa_study(quotes, **kwargs)
