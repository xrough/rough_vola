"""Fast campaign orchestration checks; no market requests or expensive fits."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import spy_rough_calls as study
from rvlab.pricing import SpotMarket, DiscountCurve, DividendSchedule, PricingError


def market():
    return SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))


def panel():
    return pd.DataFrame({"strike": [98., 100., 102.], "maturity": [.05, .1, .2],
        "is_call": True, "option_type": "call", "exercise_style": "american",
        "market_price": [3., 2., 1.], "bid": [2.975, 1.975, .975],
        "ask": [3.025, 2.025, 1.025], "split": ["train", "train", "heldout"]})


def fake_price(params, *, market, strike, maturity, config):
    return np.array([3., 2., 1.])[:len(strike)]


def fake_calibrator(quotes, initial, *, market, bounds, config, max_nfev,
                    evaluation_callback, evaluation_observer, **kwargs):
    assert quotes.split.eq("train").all()
    evaluation_callback(initial, config, "calibration pricing")
    evaluation_observer(initial, config, "calibration pricing", np.array([3., 2.]))
    return SimpleNamespace(params=initial, optimizer_success=True, optimizer_status=1,
        optimizer_message="mock converged", nfev=1, njev=1, objective=0.,
        active_bounds={}, fixed_parameters={}, optimality=0., physical_optimality=0.,
        final_jacobian=np.ones((2, 1)), free_parameters=("v0",),
        jacobian_singular_values=(2.,), metadata={"exercise_style": "american"},
        optimizer_settings={})


def test_queue_and_declared_bounds_are_fixed():
    specs = study.attempt_specs()
    assert len(specs) == 31
    assert len({s["id"] for s in specs}) == 31
    assert all(s["dataset"].startswith("synthetic") for s in specs[:8])
    assert all(s["max_nfev"] == 80 for s in specs[-5:])
    assert study.BOUNDS["hurst"] == (.05, .5)
    p, bounds = study._initial_bounds(specs[0])
    assert p.theta == .045 and p.kappa == 1.5 and p.hurst == .15
    assert bounds["theta"] == (.045, .045)


def test_budget_uses_single_monotonic_clock_and_reserves_refinement():
    now = [0.]
    budget = study.NumericalBudget(launch_seconds=4, total_seconds=6, clock=lambda: now[0])
    now[0] = 4.1
    with pytest.raises(study.CampaignDeadline, match="fit budget"):
        budget.check(fitting=True)
    budget.check()
    now[0] = 6.
    with pytest.raises(study.CampaignDeadline, match="numerical budget"):
        budget.check()


def test_contract_metrics_and_start_selection_never_select_on_holdout():
    rows = [{"id": "a", "dataset": "x", "model": "rough_full", "parameters": {},
             "objective": 2., "metrics": {"heldout_dollar_rmse": 0.}},
            {"id": "b", "dataset": "x", "model": "rough_full", "parameters": {},
             "objective": 1., "metrics": {"heldout_dollar_rmse": 100.}}]
    assert study.choose_training_start(rows, "x")["id"] == "b"
    train = panel().query("split == 'train'")
    assert study.training_objective([3., 2.], train) == 0.
    with pytest.raises(ValueError, match="contract-aligned"):
        study.training_objective([3.], train)


def test_attempt_preserves_contracts_and_independent_refinement(tmp_path):
    spec = study.attempt_specs()[0]
    result = study.run_attempt(spec, panel(), market(), tmp_path,
        study.NumericalBudget(), fake_calibrator, fake_price)
    assert result["status"] == "converged"
    assert result["ready"] and result["numerical_stable"]
    assert result["actual_pricing_evaluations"] == 8
    assert result["calibration_pricing_evaluations"] == 1
    assert result["refinement_pricing_evaluations"] == 7
    assert len(result["refinement"]) == 27
    table = pd.read_csv(tmp_path / "contracts.csv")
    assert table["split"].tolist() == ["train", "train", "heldout"]
    assert "extra_price" in table
    assert (tmp_path / "physical_jacobian.csv").exists()
    assert json.loads((tmp_path / "best_training_checkpoint.json").read_text())["best_training_objective"] == 0.


def test_numerical_failure_keeps_best_training_checkpoint_without_false_success(tmp_path):
    def failure(*args, **kwargs):
        fake_calibrator(*args, **kwargs)
        raise PricingError("deliberate divergent point")
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
        study.NumericalBudget(), failure, fake_price)
    assert result["status"] == "numerical_failure"
    assert result["parameters"]
    assert result["numerical_stable"]
    assert not result["optimizer_success"] and not result["ready"]
    assert "deliberate divergent" in result["error"]


def test_exhaustion_and_refinement_failure_are_separate(tmp_path):
    def exhausted(*args, **kwargs):
        fit = fake_calibrator(*args, **kwargs)
        fit.optimizer_success, fit.optimizer_status = False, 0
        return fit
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
        study.NumericalBudget(), exhausted, fake_price)
    assert result["status"] == "optimizer_exhausted"
    assert result["numerical_stable"] and not result["ready"]
    def unstable(params, **kwargs):
        values = fake_price(params, **kwargs)
        return values + (.01 if kwargs["config"] != study.FIT_CONFIG else 0.)
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path / "unstable",
        study.NumericalBudget(), fake_calibrator, unstable)
    assert result["optimizer_success"] and not result["ready"]
    assert not result["numerical_stable"]


def test_refinement_numerical_failure_remains_visible(tmp_path):
    def fail(params, **kwargs):
        if kwargs["config"] == study.EXTRA_CONFIG:
            raise PricingError("extra grid failed")
        return fake_price(params, **kwargs)
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
        study.NumericalBudget(), fake_calibrator, fail)
    probe = next(p for p in result["refinement"] if p["axis"] == "extra")
    assert probe["status"] == "numerical_failure"
    assert probe["pricing_attempted"] and not result["ready"]
    assert result["refinement_pricing_evaluations"] == 7


def test_pairwise_refinement_detects_cancellation_against_baseline():
    def cancelling(params, **kwargs):
        values = fake_price(params, **kwargs)
        config = kwargs["config"]
        shift = .001 if config.quadrature_nodes == 384 and config.riccati_steps == 512 else 0.
        shift = -.001 if config.fourier_cutoff == 200 and config.quadrature_nodes == 192 else shift
        return values + shift
    _, probes, stable = study.refine_attempt(study.GENERATOR, panel(), market(), cancelling, study.NumericalBudget())
    assert all(p["stable"] for p in probes if not p["axis"].startswith("pair:"))
    assert not stable
    assert any(not p["stable"] for p in probes if p["axis"].startswith("pair:"))


def test_api_numerical_failure_cannot_be_overridden_by_runner(tmp_path):
    def api_unstable(*args, **kwargs):
        result = fake_calibrator(*args, **kwargs)
        result.numerical_stable = False
        return result
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
        study.NumericalBudget(), api_unstable, fake_price)
    assert result["optimizer_success"] and not result["numerical_stable"]
    assert not result["ready"]


def test_deadline_before_fit_retains_failure_not_optimizer_success(tmp_path):
    now = [0.]
    budget = study.NumericalBudget(launch_seconds=1., total_seconds=2., clock=lambda: now[0])
    now[0] = 1.1
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
        budget, fake_calibrator, fake_price)
    assert result["status"] == "budget_interrupted"
    assert not result["optimizer_success"]
    assert result["actual_pricing_evaluations"] == 0
    assert "parameters" not in result


def test_synthetic_gate_uses_recovered_prices_and_allows_parameter_dispersion():
    rows = []
    for dataset in ("synthetic_clean", "synthetic_noisy"):
        rows.append({"id": dataset, "dataset": dataset, "model": "rough_full", "ready": True,
            "parameters": {"hurst": .49}, "metrics": {"train_refined_dollar_rmse": .0005,
                                                         "heldout_refined_dollar_rmse": .0007}})
    assert study.synthetic_gate(rows)["passed"]
    rows[0]["ready"] = False
    assert not study.synthetic_gate(rows)["passed"]


def test_synthetic_design_noise_reproducibility_and_no_market_fallback():
    def synthetic(params, *, strike, **kwargs):
        return np.ones(len(strike)) * 2.
    first, _ = study.synthetic_panels(synthetic, study.NumericalBudget())
    second, _ = study.synthetic_panels(synthetic, study.NumericalBudget())
    pd.testing.assert_frame_equal(first["synthetic_noisy"], second["synthetic_noisy"])
    for name, q in first.items():
        assert len(q) == 18
        assert q.split.value_counts().to_dict() == {"train": 12, "heldout": 6}
        assert q.exercise_style.eq("american").all()
        assert (q.ask > q.bid).all()


def test_pending_report_is_explicit_and_serializes_no_nan(tmp_path):
    records = [study._pending(s, "stock repeat required") for s in study.attempt_specs()]
    study.write_report(tmp_path, records, {"ready": False, "status": "pending"}, 0.)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert not summary["hedging_ready"] and not summary["data_ready"]
    assert not summary["synthetic_gate"]["passed"]
    assert summary["synthetic_gate_status"] == "not run"
    assert "stock" in (tmp_path / "report.txt").read_text()
    assert "Repeat-acquisition stock gate" in (tmp_path / "report.txt").read_text()
    assert "Independent intraday cross-source validation: unavailable" in (tmp_path / "report.txt").read_text()
    assert (tmp_path / "figures/calibration_errors.png").exists()


def test_negative_or_overlapping_budget_rejected():
    with pytest.raises(ValueError):
        study.NumericalBudget(launch_seconds=3, total_seconds=2)
    with pytest.raises(ValueError):
        study.NumericalBudget(launch_seconds=-1, total_seconds=2)


def test_campaign_seal_blocks_unvalidated_workflow_and_rejects_changed_inputs(tmp_path, monkeypatch):
    spec = {"id": "2025-08-07_bs_0", "dataset": study.DATES[0], "model": "bs", "start": 0, "max_nfev": 120}
    monkeypatch.setattr(study, "attempt_specs", lambda: [spec])
    monkeypatch.setattr(study, "_figures", lambda *args: None)
    bundle = {"ready": True, "status": "ready", "panels": {study.DATES[0]: panel()},
              "markets": {study.DATES[0]: market()}, "diagnostics": {"input_sha256": "test-source"}}
    def no_fit(*args, **kwargs):
        raise AssertionError("missing observed inputs must never launch calibration")
    def price(params, *, strike, **kwargs):
        return np.full(len(strike), 2.)
    result = study.run_campaign(tmp_path, bundle=bundle, calibrator=no_fit, pricer=price)
    assert result["data_ready"]
    row = json.loads((tmp_path / "attempts" / spec["id"] / "attempt.json").read_text())
    assert row["status"] == "pending" and "synthetic" in row["reason"]
    assert list((tmp_path / "source_snapshot").rglob("*.py"))
    with pytest.raises(FileExistsError, match="sealed"):
        study.run_campaign(tmp_path, bundle=bundle, calibrator=no_fit, pricer=price)
    study.run_campaign(tmp_path, bundle=bundle, calibrator=no_fit, pricer=price, resume=True)
    bundle["markets"][study.DATES[0]] = replace(market(), spot=101.)
    with pytest.raises(ValueError, match="identity changed"):
        study.run_campaign(tmp_path, bundle=bundle, calibrator=no_fit, pricer=price, resume=True)


def test_campaign_resume_rejects_modified_synthetic_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "attempt_specs", lambda: [])
    monkeypatch.setattr(study, "_figures", lambda *args: None)
    bundle = {"ready": True, "status": "ready", "panels": {}, "markets": {}}
    def price(params, *, strike, **kwargs):
        return np.full(len(strike), 2.)
    study.run_campaign(tmp_path, bundle=bundle, calibrator=fake_calibrator, pricer=price)
    source = tmp_path / "synthetic_observations.csv"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="inventory mismatch"):
        study.run_campaign(tmp_path, bundle=bundle, calibrator=fake_calibrator, pricer=price, resume=True)


def test_blocked_data_preflight_never_starts_numerical_clock_or_seals_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_figures", lambda *args: None)
    def no_price(*args, **kwargs):
        raise AssertionError("blocked acquisition must never invoke a pricer")
    result = study.run_campaign(tmp_path, bundle={"ready": False, "status": "blocked", "diagnostics": {}},
                                calibrator=no_price, pricer=no_price)
    assert result["status"] == "blocked_preflight" and result["elapsed_seconds"] == 0.
    assert not (tmp_path / "campaign_seal.json").exists()
    assert not (tmp_path / "synthetic_observations.csv").exists()
    assert not (tmp_path / "source_snapshot").exists()


@pytest.mark.parametrize("bad", [1., [1.], [1., 2., float("nan")]])
def test_scalar_wrong_shape_and_nonfinite_prices_rejected(bad):
    with pytest.raises(PricingError, match="one value per contract"):
        study.refine_attempt(study.GENERATOR, panel(), market(), lambda *args, **kwargs: bad,
                             study.NumericalBudget())


def test_failed_baseline_refinement_is_counted_as_actual_attempt(tmp_path):
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
        study.NumericalBudget(), fake_calibrator,
        lambda *args, **kwargs: (_ for _ in ()).throw(PricingError("baseline failed")))
    assert result["refinement_pricing_evaluations"] == 1
    assert result["actual_pricing_evaluations"] == 2
    assert not result["ready"]


def test_deadline_after_successful_training_call_retains_candidate(tmp_path):
    now = [0.]
    budget = study.NumericalBudget(launch_seconds=1., total_seconds=3., clock=lambda: now[0])
    def interrupted(*args, **kwargs):
        fake_calibrator(*args, **kwargs)
        now[0] = 1.1
        kwargs["evaluation_callback"](args[1], kwargs["config"], "calibration pricing")
        raise AssertionError("second fit evaluation must not run")
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
                               budget, interrupted, fake_price)
    assert result["status"] == "budget_interrupted"
    assert result["parameters"] and result["objective"] == 0.
    assert result["calibration_pricing_evaluations"] == 1
    assert result["refinement_pricing_evaluations"] == 7
    assert result["numerical_stable"] and not result["ready"]


def test_actual_api_attempt_serializes_refinement_dataclass_and_dataframe(tmp_path):
    from rvlab.pricing import BlackScholesParams, price_american_calls, calibrate_american_call_surface
    q, m = panel(), market()
    values = price_american_calls(BlackScholesParams(.2), market=m,
        strike=q.strike.to_numpy(), maturity=q.maturity.to_numpy(), config=study.FIT_CONFIG)
    q["market_price"], q["bid"], q["ask"] = values, values-.025, values+.025
    spec = {"id": "real_bs", "dataset": "synthetic_api", "model": "bs", "start": 0, "max_nfev": 1}
    result = study.run_attempt(spec, q, m, tmp_path, study.NumericalBudget(),
                               calibrate_american_call_surface, price_american_calls)
    assert result["status"] == "converged"
    assert result["optimizer_success"] and result["ready"]
    diagnostic = json.loads((tmp_path / "api_refinement_diagnostics.json").read_text())
    assert len(diagnostic["checks"]) >= 5
    assert len(diagnostic["base_prices"]) == 2
    assert pd.read_csv(tmp_path / "api_refinement_checks.csv").shape[0] >= 5
    study._verify_inventory(tmp_path, "attempt_inventory.json", required=True)


def test_postfit_reporting_failure_cannot_remain_ready(tmp_path, monkeypatch):
    original = study._write_json
    def fail_diagnostic(path, value):
        if path.name == "api_refinement_diagnostics.json":
            raise OSError("diagnostic write deliberately failed")
        return original(path, value)
    monkeypatch.setattr(study, "_write_json", fail_diagnostic)
    def with_diagnostics(*args, **kwargs):
        result = fake_calibrator(*args, **kwargs)
        result.diagnostics = SimpleNamespace(checks=pd.DataFrame({"stable": [True]}))
        return result
    result = study.run_attempt(study.attempt_specs()[0], panel(), market(), tmp_path,
                               study.NumericalBudget(), with_diagnostics, fake_price)
    assert result["status"] == "reporting_failure"
    assert result["reported_optimizer_success_before_reporting_failure"]
    assert not result["optimizer_success"] and not result["ready"]


def test_terminal_attempt_requires_receipt_after_interruption(tmp_path, monkeypatch):
    spec = {"id": "2025-08-07_bs_0", "dataset": study.DATES[0], "model": "bs", "start": 0, "max_nfev": 1}
    monkeypatch.setattr(study, "attempt_specs", lambda: [spec])
    monkeypatch.setattr(study, "_figures", lambda *args: None)
    bundle = {"ready": True, "panels": {study.DATES[0]: panel()}, "markets": {study.DATES[0]: market()}}
    def price(params, *, strike, **kwargs):
        return np.full(len(strike), 2.)
    study.run_campaign(tmp_path, bundle=bundle, calibrator=fake_calibrator, pricer=price)
    (tmp_path / "campaign_inventory.json").unlink()  # represents interruption before final receipt
    attempt = tmp_path / "attempts" / spec["id"] / "attempt.json"
    row = json.loads(attempt.read_text())
    row["status"] = "converged"
    attempt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="required sealed inventory"):
        study.run_campaign(tmp_path, bundle=bundle, calibrator=fake_calibrator, pricer=price, resume=True)


def test_sensitivity_phase_receipt_and_source_invalidation_are_persisted(tmp_path, monkeypatch):
    spec = {"id": "2025-08-07_bs_0", "dataset": study.DATES[0], "model": "bs", "start": 0, "max_nfev": 1}
    monkeypatch.setattr(study, "attempt_specs", lambda: [spec])
    monkeypatch.setattr(study, "_figures", lambda *args: None)
    bundle = {"ready": True, "panels": {study.DATES[0]: panel()}, "markets": {study.DATES[0]: market()}}
    source = study._source_identity(study.ROOT)
    calls = [0]
    def changing_identity(root):
        calls[0] += 1
        return source if calls[0] == 1 else {**source, "deliberate_changed_source.py": "changed"}
    monkeypatch.setattr(study, "_source_identity", changing_identity)
    def sensitivity(records, bundle, folder, pricer, budget):
        state = json.loads((folder / "active_phase.json").read_text())
        assert state["active"] and state["phase"] == "input_sensitivities"
    monkeypatch.setattr(study, "input_sensitivities", sensitivity)
    def price(params, *, strike, **kwargs):
        return np.full(len(strike), 2.)
    study.run_campaign(tmp_path, bundle=bundle, calibrator=fake_calibrator, pricer=price)
    row = json.loads((tmp_path / "attempts" / spec["id"] / "attempt.json").read_text())
    assert row["status"] == "source_identity_failure"
    assert not row["ready"] and not row["source_identity_valid"]
    study._verify_inventory(tmp_path / "attempts" / spec["id"], "attempt_inventory.json", required=True)
    assert not json.loads((tmp_path / "active_phase.json").read_text())["active"]
