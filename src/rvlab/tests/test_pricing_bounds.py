"""Bound research keeps training selection, provenance and bad outcomes visible."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from rvlab.experiments import _pricing_bounds as study
from rvlab.experiments._pricing_fits import DEMO_BOUNDS, FitAttempt, synthetic_surface
from rvlab.pricing import BlackScholesParams, HestonParams, PricingConfig, PricingError


CONFIG = PricingConfig(128, 400., 128)


def _quotes():
    quotes = synthetic_surface("heston", config=CONFIG)
    quotes["expiry_date"] = np.repeat(["2025-08-22", "2025-08-29", "2025-09-05"], 6)
    return quotes


def _mock_fit(monkeypatch, *, fail_label=None):
    calls = []

    def fake_fit(label, quotes, initial_params, **kwargs):
        calls.append((label, quotes.copy(deep=True), initial_params, kwargs))
        # Deliberately make the exhausted first expanded start the training
        # winner and the converged second one the held-out winner.
        objective = 1. if label == "heston_expanded_original_start" else 2.
        status = "exhausted" if label == "heston_expanded_original_start" else "converged"
        summary = {"label": label, "params": asdict(initial_params),
            "initial_params": asdict(initial_params), "bounds": kwargs["bounds"],
            "max_nfev": kwargs["max_nfev"], "nfev": kwargs["max_nfev"],
            "training_objective": objective, "test_rmse": 999. if objective == 1. else 0.,
            "status": status, "ready": status == "converged", "error": None}
        result = SimpleNamespace(params=initial_params, objective=objective)
        if label == fail_label:
            summary.update(status="numerical_failure", error="declared failure", params=None)
            result = None
        return FitAttempt(label, result, summary, quotes.copy(deep=True))

    monkeypatch.setattr(study, "fit_attempt", fake_fit)
    monkeypatch.setattr(study, "relative_price_jacobian", lambda *args, **kwargs: {
        "status": "ok", "error": None, "condition_number": 1., "singular_values": [1.]})
    return calls


def test_declared_bounds_budgets_and_start_provenance(monkeypatch):
    calls = _mock_fit(monkeypatch)
    original_bounds = dict(DEMO_BOUNDS)
    quotes = _quotes()
    untouched = quotes.copy(deep=True)
    result = study.heston_bounds_study(quotes, config=CONFIG, max_nfev=80)
    assert len(calls) == len(result["attempts"]) == 14
    assert all(call[3]["config"] is CONFIG and call[3]["max_nfev"] == 80 for call in calls)
    assert calls[0][3]["bounds"] == study.HESTON_ORIGINAL_BOUNDS
    assert calls[1][3]["bounds"] == {**study.HESTON_ORIGINAL_BOUNDS, "theta": (.0025, 2.)}
    assert calls[2][3]["bounds"] == {**study.HESTON_ORIGINAL_BOUNDS, "vol_of_vol": (.01, 2.)}
    assert calls[3][2] == study.HESTON_BASELINE_START
    assert calls[4][2] == HestonParams(.02, .04, 6., 1.2, -.65)
    assert calls[3][3]["bounds"] == calls[4][3]["bounds"] == study.HESTON_EXPANDED_BOUNDS
    assert calls[3][3]["bounds"]["rho"] == (-.95, 0.)
    profile_calls = calls[5:11]
    assert [call[2].theta for call in profile_calls] == list(study.THETA_PROFILE_VALUES)
    for call in profile_calls:
        theta = call[2].theta
        assert call[3]["bounds"]["theta"] == (theta, theta)
        assert call[3]["bounds"]["kappa"] == (.001, 20.)
        assert call[3]["bounds"]["vol_of_vol"] == (.01, .6)
    for call in calls[11:]:
        assert len(call[1]) == 6
        assert call[3]["bounds"]["theta"] == (.045, .045)
        assert call[3]["bounds"]["kappa"] == (1., 1.)
        assert call[3]["bounds"]["vol_of_vol"] == (.01, 2.)
    assert "round-number" in result["attempts"][4].summary["initial_provenance"]
    assert result["manifest"]["max_nfev_per_attempt"] == 80
    assert DEMO_BOUNDS == original_bounds
    pd.testing.assert_frame_equal(quotes, untouched)


def test_selection_uses_training_objective_and_preserves_exhaustion(monkeypatch):
    _mock_fit(monkeypatch)
    result = study.heston_bounds_study(_quotes(), config=CONFIG, max_nfev=3)
    selected = result["selected_expanded"]
    assert selected.label == "heston_expanded_original_start"
    assert selected.summary["status"] == "exhausted"
    assert selected.summary["test_rmse"] == 999.
    assert not selected.summary["ready"]
    summary = result["tables"]["bounds_summary"]
    assert len(summary) == 5
    assert summary.iloc[3].status == "exhausted"
    profile = result["tables"]["theta_profile"]
    assert len(profile) == 6
    assert np.allclose(profile.kappa_theta,
                       np.array(study.THETA_PROFILE_VALUES) * study.HESTON_BASELINE_START.kappa)
    assert len(result["tables"]["fitted_prices"]) == 5 * 18 + 6 * 18 + 3 * 6


def test_failed_attempt_is_retained_with_missing_profile_product(monkeypatch):
    _mock_fit(monkeypatch, fail_label="heston_theta_profile_0.25")
    result = study.heston_bounds_study(_quotes(), config=CONFIG, max_nfev=1)
    row = result["tables"]["theta_profile"].query("profile_theta == .25").iloc[0]
    assert row.status == "numerical_failure" and row.error == "declared failure"
    assert pd.isna(row.kappa_theta)
    assert any(a.label == row.label and a.result is None for a in result["attempts"])


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected_before_fitting(monkeypatch, budget):
    calls = _mock_fit(monkeypatch)
    with pytest.raises(ValueError, match="max_nfev"):
        study.heston_bounds_study(_quotes(), config=CONFIG, max_nfev=budget)
    assert not calls


def test_relative_jacobian_matches_bs_vega_and_ignores_heldout_prices():
    quotes = pd.DataFrame({"forward": [100., 100.], "discount": [1., 1.],
        "strike": [100., 100.], "maturity": [.25, .25], "is_call": [True, True],
        "market_price": [4., 999.], "split": ["train", "heldout"]})
    diagnostic = study.relative_price_jacobian(BlackScholesParams(.2), quotes, config=CONFIG)
    expected = .2 * 100. * norm.pdf(.05) * .5
    assert diagnostic["jacobian"][0][0] == pytest.approx(expected, rel=1e-8)
    assert diagnostic["n_training"] == diagnostic["rank"] == diagnostic["n_parameters"] == 1
    assert diagnostic["parameter_scales"] == {"volatility": .2}
    assert diagnostic["condition_number"] == 1.
    assert diagnostic["finite_difference_schemes"] == {"volatility": "central"}
    quotes.loc[1, "market_price"] = -999.
    changed = study.relative_price_jacobian(BlackScholesParams(.2), quotes, config=CONFIG)
    assert changed == diagnostic
    assert "not confidence intervals" in diagnostic["interpretation"]


def test_relative_jacobian_uses_valid_one_sided_difference_at_boundary():
    quotes = pd.DataFrame({"forward": [100.], "discount": [1.], "strike": [100.],
        "maturity": [.25], "is_call": [True], "market_price": [0.], "split": ["train"]})
    result = study.relative_price_jacobian(BlackScholesParams(0.), quotes, config=CONFIG)
    assert result["finite_difference_schemes"] == {"volatility": "forward"}
    assert result["parameter_scales"] == {"volatility": .01}
    assert result["jacobian"][0][0] == pytest.approx(.01 * 100. * norm.pdf(0.) * .5, rel=1e-8)


def test_underdetermined_jacobian_reports_null_direction_without_infinite_json():
    quotes = _quotes().iloc[:1].copy()
    result = study.relative_price_jacobian(HestonParams(.04, .045, 1., .3, -.6), quotes,
                                            config=CONFIG)
    jacobian = np.array(result["jacobian"])
    weak = np.array(list(result["weak_direction"].values()))
    assert result["rank"] == 1 and result["n_parameters"] == 5
    assert result["condition_number"] is None
    assert np.linalg.norm(jacobian @ weak) < 1e-12


def test_jacobian_numerical_failure_is_not_replaced_by_zero(monkeypatch):
    def fail(*args, **kwargs):
        raise PricingError("integration failed")

    monkeypatch.setattr(study, "price", fail)
    with pytest.raises(PricingError, match="integration failed"):
        study.relative_price_jacobian(study.HESTON_BASELINE_START, _quotes(), config=CONFIG)


def test_actual_bounded_study_retains_all_one_evaluation_attempts():
    quotes = _quotes()
    result = study.heston_bounds_study(quotes, config=CONFIG, max_nfev=1)
    assert len(result["attempts"]) == 14
    assert all(a.summary["max_nfev"] == 1 for a in result["attempts"])
    assert all(a.summary["nfev"] in (None, 1) for a in result["attempts"])
    assert any(a.summary["status"] == "exhausted" for a in result["attempts"])
    assert len(result["tables"]["theta_profile_prices"]) == 6 * len(quotes)
    assert len(result["tables"]["identification"]) == 2
    assert all(a.predictions.contract_id.isin(quotes.contract_id).all() for a in result["attempts"])
