"""Research narratives keep compute scope and market-input gates explicit."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_research as study
from rvlab.pricing import BlackScholesParams, FDConfig, PricingConfig, PricingError


@pytest.fixture(autouse=True)
def plots():
    pytest.importorskip("matplotlib").use("Agg", force=True)
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def context():
    return SimpleNamespace(progress=lambda message: None)


def _blocked():
    return {"audit": pd.DataFrame([{"date": day, "ready": False, "contracts": 0,
        "blocker": "Independent stock; historically available cash dividends",
        "stock_observed": False, "dividend_expectations_available": False} for day in study.DATES]),
        "snapshots": {}, "quotes": pd.DataFrame(), "exclusions": pd.DataFrame(),
        "fingerprints": [], "processed_root": "unavailable"}


def test_pricing_gallery_never_calibrates_and_keeps_both_exercise_styles(context, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pricing notebook must not run calibration")
    monkeypatch.setattr(study, "fit_attempt", forbidden)
    monkeypatch.setattr(study, "calibrate_vanilla_surface", forbidden)
    result = study.pricing_research(context, fourier_config=PricingConfig(32, 40., 64),
                                    fd_config=FDConfig(24, 8, 20))
    assert result["manifest"]["optimizer_calls"] == 0
    assert result["manifest"]["evidence_version"] == "pricing_research_v1"
    assert result["manifest"]["data_kind"] == "synthetic"
    assert len(result["tables"]["european_prices"]) == 54
    prices = result["tables"]["american_prices"]
    assert len(prices) == 144
    assert set(prices.exercise_style) == {"american", "european"}
    assert set(prices.scenario) == {"cash dividend", "no dividend"}
    assert not result["tables"]["exercise_boundaries"].empty
    assert set(result["tables"]["refinement"].exercise_style) == {"american", "european"}


def test_spy_gate_audits_once_without_fits_or_proxy_substitutes(context, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "load_american_spy_inputs", lambda *args, **kwargs: calls.append(kwargs) or _blocked())
    monkeypatch.setattr(study, "_spy_fits", lambda *args, **kwargs: pytest.fail("blocked SPY must not fit"))
    result = study.spy_research(context)
    assert len(calls) == 1 and result["manifest"]["input_audit_count"] == 1
    assert result["manifest"]["backend"] == "american_vanilla_fd"
    assert result["manifest"]["status"] == "blocked_inputs"
    assert not result["manifest"]["calibration_ready"]
    assert not result["manifest"]["hedging_ready"]
    assert set(result["tables"]) == {"input_audit", "observed_quotes", "input_exclusions", "study_design"}
    assert "fit_summary" not in result["tables"]


def _mock_fit(initial):
    return SimpleNamespace(params=initial, optimizer_success=True, price_stable=True, delta_stable=True,
        nfev=2, pricing_evaluations=4, optimizer_status=1, optimizer_message="mock convergence",
        optimizer_settings={}, active_bounds={}, objective=0., physical_optimality=0.,
        max_price_change=0., max_delta_change=0.)


def test_all_contract_validation_failure_cannot_enter_spy_sensitivity(context, monkeypatch):
    market = study._markets()["cash dividend"]
    quotes = study._contracts().assign(market_price=1., exercise_style="american")
    monkeypatch.setattr(study, "calibrate_vanilla_surface", lambda quotes, initial, **kwargs: _mock_fit(initial))
    monkeypatch.setattr(study, "evaluate_vanilla", lambda *args, **kwargs: (_ for _ in ()).throw(PricingError("validation divergence")))
    row, prediction, parameters, checks, fitted = study._american_attempt("BS", "test", quotes, market,
        BlackScholesParams(.2), None, FDConfig(20, 6, 20), 2)
    assert row["status"] == "numerical_failure" and not row["ready"]
    assert "validation divergence" in row["error"]
    assert fitted is None and prediction.empty


def test_calibration_keeps_generation_failures_without_loading_optional_results(context, monkeypatch):
    import rvlab.experiments._pricing_extended as extended
    monkeypatch.setattr(extended, "extended_demo", lambda *a, **k: pytest.fail("optional study is disabled"))
    monkeypatch.setattr(study, "pricing_research", lambda *a, **k: pytest.fail("calibration must not run gallery"))
    monkeypatch.setattr(study, "evaluate_vanilla", lambda *a, **k: (_ for _ in ()).throw(PricingError("reference generation failed")))
    result = study.calibration_research(context, max_nfev=1, american_max_nfev=1,
        run_controlled_american=True, run_american_joint=False,
        coarse_config=PricingConfig(24, 30., 32), fine_config=PricingConfig(32, 40., 64),
        fd_config=FDConfig(16, 4, 12),
        calibration_configs={"bs": FDConfig(20, 6, 20), "heston": FDConfig(20, 6, 20)})
    summary = result["tables"]["american_summary"]
    assert len(summary) == 2 and summary.status.eq("numerical_failure").all()
    assert not summary.ready.any() and summary.error.str.contains("reference generation failed").all()
    assert {"free_parameters", "train_rmse", "test_rmse", "price_stable", "delta_stable", "optimizer_success"} <= set(summary)
    assert not result["manifest"]["extended_executed"]
    assert not any(name.startswith("extended_") for name in result["tables"])


def test_joint_calibration_bundle_is_explicit_and_controlled_generation_is_optional(context, monkeypatch):
    import rvlab.experiments._pricing_american_identification as joint
    calls = []
    monkeypatch.setattr(study, "evaluate_vanilla", lambda *a, **k: pytest.fail("controlled generation is disabled"))
    def identification(actual_context, **kwargs):
        calls.append((actual_context, kwargs))
        return {"tables": {"summary": pd.DataFrame([{"status": "exhausted", "ready": False}])},
                "figures": {}, "manifest": {"selection_rule": "training objective only"}}
    monkeypatch.setattr(joint, "american_identification_study", identification)
    result = study.calibration_research(context, max_nfev=1,
        coarse_config=PricingConfig(24, 30., 32), fine_config=PricingConfig(32, 40., 64),
        joint_max_nfev=3, joint_run_continuation=False, joint_reuse_checkpoints=False)
    assert len(calls) == 1 and calls[0][0] is context
    assert calls[0][1]["max_nfev"] == 3
    assert calls[0][1]["run_continuation"] is False
    assert calls[0][1]["reuse_checkpoints"] is False
    assert result["tables"]["american_summary"].empty
    assert result["tables"]["american_joint_summary"].status.eq("exhausted").all()
    assert result["manifest"]["american_joint_executed"]
    assert not result["manifest"]["controlled_american_executed"]
    assert "american_recovery" not in result["figures"]


def test_generation_must_be_finer_than_coarse_and_requested_fit_grids():
    fits = {"bs": FDConfig(20, 6, 20), "heston": FDConfig(20, 6, 20)}
    with pytest.raises(ValueError, match="strictly finer"):
        study._calibration_fd_settings(FDConfig(40, 12, 40), fits, None)
    with pytest.raises(ValueError, match="strictly finer"):
        study._calibration_fd_settings(FDConfig(16, 4, 12), fits, fits)


@pytest.mark.parametrize("dividends", [True, False])
def test_ready_spy_reuses_six_fits_and_preserves_failed_input_sensitivities(context, monkeypatch, dividends):
    market = study._markets()["cash dividend" if dividends else "no dividend"]
    quotes = study._contracts().assign(market_price=1., bid=.98, ask=1.02, exercise_style="american")
    monkeypatch.setattr(study, "select_american_contracts", lambda snapshot: (quotes.copy(), market))
    calls = []
    def attempt(label, which, frame, actual_market, initial, bounds, config, budget, **kwargs):
        calls.append((label, which, frame.contract_id.tolist(), kwargs["optimizer"].parameterization))
        assert actual_market is market and frame.exercise_style.eq("american").all()
        row = {"model": label, "status": "converged", "ready": True}
        return row, frame.assign(model=label, model_price=1., price_residual=0.), [], pd.DataFrame(), initial
    monkeypatch.setattr(study, "_american_attempt", attempt)
    def failed_sensitivity(params, **kwargs):
        assert kwargs["exercise_style"] == "american"
        actual = kwargs["market"]
        if actual.dividends.events:
            assert actual.dividends.coverage == market.dividends.coverage
            assert actual.dividends.events[0].time == market.dividends.events[0].time
        raise PricingError("retained sensitivity failure")
    monkeypatch.setattr(study, "evaluate_vanilla", failed_sensitivity)
    tables, figures, manifest = {}, {}, {}
    study._spy_fits(context, object(), FDConfig(20, 6, 20), 2, tables, figures, manifest)
    assert len(calls) == 6 and len({item[0] for item in calls}) == 6
    assert len({tuple(item[2]) for item in calls}) == 1
    assert calls[-1][3] == "asinh"
    assert manifest["fit_attempt_count"] == 6
    sensitivity = tables["input_sensitivity"]
    assert not sensitivity.refitted.any()
    assert sensitivity.loc[sensitivity.status.eq("numerical_failure"), "max_price_change"].isna().all()
    if dividends:
        assert sensitivity.scenario.str.contains("counterfactual").sum() == 6
        assert len(sensitivity) == 18
    else:
        assert sensitivity.status.eq("not_applicable").sum() == 3
        assert len(sensitivity) == 15


@pytest.mark.parametrize("value", [True, 0, 1.5])
def test_invalid_optimizer_budget_is_rejected_before_data_access(context, value):
    with pytest.raises(ValueError, match="positive integer"):
        study.spy_research(context, max_nfev=value)
