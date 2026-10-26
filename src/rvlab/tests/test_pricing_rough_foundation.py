"""Honest numerical evidence and reproducible rough-foundation study helpers."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_foundation as study
from rvlab.pricing import PricingError


def test_price_gate_preserves_existing_tolerances():
    result = study.price_gate([2.0009, 2.002], [2., 2.])
    assert result["price_allowance"] == pytest.approx([.0012, .0012])
    assert result["accepted"].tolist() == [True, False]


def test_wide_sampling_interval_overlapping_reference_cannot_pass():
    result = study.price_gate(2., 2., standard_error=.01)
    assert result["ci_half_width"] == pytest.approx(.0196)
    assert not result["accepted"]


def test_conservative_gate_adds_sampling_error_to_discrepancy():
    result = study.price_gate([1.0005, 1.0005], 1., standard_error=[.0002, .0004])
    assert result["accepted"].tolist() == [True, False]
    assert result["conservative_error"] == pytest.approx([.000892, .001284])


@pytest.mark.parametrize("estimate,reference,se", [(np.nan, 1., 0.), (1., np.inf, 0.),
                                                    (1., 1., -.1), (1., 1., np.nan)])
def test_gate_rejects_invalid_numerical_evidence(estimate, reference, se):
    with pytest.raises(ValueError):
        study.price_gate(estimate, reference, standard_error=se)


def test_sample_summary_counts_independent_paths_not_contracts():
    samples = np.array([[1., 10.], [3., 14.], [5., 18.]])
    mean, se = study.sample_summary(samples)
    assert mean == pytest.approx([3., 14.])
    assert se == pytest.approx(np.std(samples, axis=0, ddof=1)/np.sqrt(3))
    with pytest.raises(ValueError):
        study.sample_summary([[1., 2.]])


def test_checkpoint_replay_retains_original_time_and_changed_inputs(tmp_path):
    value, original = study._checkpoint(tmp_path, "trial", {"seed": 42},
        lambda: {"status": "completed", "prices": [1.]})
    replay, cached = study._checkpoint(tmp_path, "trial", {"seed": 42},
        lambda: pytest.fail("matching source/input record should replay"))
    assert value == replay and cached["reused"]
    assert cached["original_elapsed_seconds"] == original["original_elapsed_seconds"]
    first = (tmp_path / ("trial_"+original["fingerprint"][:16]+".json")).read_bytes()
    study._checkpoint(tmp_path, "trial", {"seed": 43}, lambda: {"prices": [2.]})
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert first == (tmp_path / ("trial_"+original["fingerprint"][:16]+".json")).read_bytes()


def test_numerical_failure_is_retained_and_corrupt_cache_rejected(tmp_path):
    def failure():
        raise PricingError("variance transition failed")
    value, record = study._checkpoint(tmp_path, "failed", {}, failure)
    assert value == {"status": "numerical_failure", "error": "variance transition failed"}
    path = tmp_path/("failed_"+record["fingerprint"][:16]+".json")
    obj = json.loads(path.read_text()); obj["specification"]["changed"] = True
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Corrupt"):
        study._checkpoint(tmp_path, "failed", {}, failure)


def test_accurate_price_with_unstable_reference_cannot_be_accepted():
    contracts = study.foundation_contracts()
    rows = study._price_records(contracts, np.ones(9), np.ones(9),
        stage="fixed-lift simulation", model="flat", factors=8, steps=1024,
        reference_stable=False)
    frame = pd.DataFrame(rows)
    assert frame.absolute_error.eq(0).all()
    assert not frame.accepted.any()
    tables = {"prices": frame, "limits": pd.DataFrame()}
    summary = study._summary(tables)
    assert len(summary) == 1 and not summary.iloc[0].accepted


def test_contract_and_market_design_make_inception_assumptions_explicit():
    contracts = study.foundation_contracts()
    assert len(contracts) == 9
    assert contracts.days.tolist() == [14]*3+[60]*3+[365]*3
    assert contracts.strike.tolist() == [95., 100., 105.]*3
    assert not contracts.iloc[0].is_call and contracts.iloc[1].is_call
    no_cash, cash = study._market(), study._market(cash=True)
    assert no_cash.dividends.events == () and no_cash.dividends.coverage == 1.
    assert cash.dividends.events[0].time == .08
    assert cash.dividends.events[0].amount == 1.5
    assert study._truths()["curve"].curve.knot_times == (20/365, 45/365)
    assert study._truths()["flat"].vol_of_vol == .3


def test_no_cash_forward_is_spot_over_discount_not_cash_proxy():
    contracts = study.foundation_contracts()
    market = study._market()
    arguments = study._price_args(contracts, market)
    assert arguments["forward"]*arguments["discount"] == pytest.approx(np.full(9, 100.))


def test_kernel_budget_addendum_is_optional_and_cannot_override_acceptance(tmp_path):
    table, provenance = study._kernel_budget_addendum(tmp_path/"missing.json")
    assert table.empty and not provenance["available"]
    assert study.PRICE_ATOL == .001 and study.PRICE_RTOL == 1e-4


def test_kernel_budget_addendum_rejects_unverified_source(tmp_path):
    raw = tmp_path/"raw"; raw.mkdir()
    (raw/"trial.py").write_text("# changed after execution\n")
    record = {"runner_source_path":"/previous/checkout/trial.py",
        "engine_source_sha256":"wrong", "runner_source_sha256":"wrong"}
    path = tmp_path/"budgets.json"; path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source does not match"):
        study._kernel_budget_addendum(path)


def test_reference_checks_cannot_hide_axis_error_by_joint_cancellation(tmp_path, monkeypatch):
    def fractional(params, contracts, market, config):
        # Each extreme is close to the joint value, but not to each other.
        amount = 2.0009 if config.quadrature_nodes == 384 and config.fourier_cutoff == 320 else 2.0018 if config.fourier_cutoff == 320 else 2.
        return {"status":"completed", "prices":[amount]*len(contracts)}
    monkeypatch.setattr(study, "_reference_prices", fractional)
    monkeypatch.setattr(study, "_lift_prices", lambda params,contracts,market,kernel,config:
        {"status":"completed", "prices":[2.]*len(contracts)})
    monkeypatch.setattr(study, "_simulate_trial", lambda *a,**k:
        {"status":"completed", "prices":[2.]*9, "standard_errors":[0.]*9,
         "gains_mean":[100.]*3,"gains_standard_error":[.1]*3,
         "variance_mean":[.04]*3,"variance_standard_error":[.001]*3,
         "diagnostics":{}})
    context = SimpleNamespace(output_dir=tmp_path,progress=lambda message:None)
    result = study.rough_foundation_study(context,paths=2,compute_figures=False)
    references = result["tables"]["reference_refinement"]
    fractional_rows = references[references.stage.str.startswith("fractional")]
    assert not fractional_rows.reference_stable.any()
    assert not fractional_rows.accepted.any()
    kernels = result["tables"]["prices"].query("stage == 'kernel approximation'")
    assert not kernels.accepted.any()
    assert not result["manifest"]["american_pricing_ready"]
    assert not result["manifest"]["stock_delta_ready"]


def test_historical_addendum_uses_verified_source_snapshot_without_relabelling(tmp_path):
    import hashlib
    old = b"# prior validated simulation source\n"
    old_sha = hashlib.sha256(old).hexdigest()
    snapshot = tmp_path/"raw/pricing_sources"/old_sha/"rough_simulation.py"
    snapshot.parent.mkdir(parents=True); snapshot.write_bytes(old)
    runner = tmp_path/"raw/trial.py"; runner.write_text("# durable trial runner\n")
    record = {"source_path":"/previous/checkout/rough_simulation.py",
        "source_sha256":old_sha,"runner_source_path":"/previous/checkout/trial.py",
        "runner_source_sha256":hashlib.sha256(runner.read_bytes()).hexdigest()}
    path = tmp_path/"historical.json"; path.write_text(json.dumps(record))
    _, provenance = study._verified_extra(path)
    assert not provenance["matches_current_engine"]
    assert provenance["source_evidence"]["rough_simulation.py"] == str(snapshot)
    snapshot.write_text("# later edited source\n")
    with pytest.raises(ValueError, match="source has changed"):
        study._verified_extra(path)


def test_cash_jump_market_cannot_enter_no_cash_fourier_reference():
    with pytest.raises(ValueError, match="no-cash market"):
        study._price_args(study.foundation_contracts(),study._market(cash=True))


@pytest.mark.parametrize("enabled", [False, True])
def test_recorded_artifact_displays_need_no_parameter_cell_globals(enabled):
    """Execute the real display cells with the refresher's restricted globals."""
    from pathlib import Path
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(Path(study.__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb",as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"
        and cell.metadata.get("rvlab_presentation") and "rough_foundation_" in cell.source]
    assert len(cells) == 3
    summary = pd.DataFrame({"stage":["kernel approximation","fixed-lift simulation","limit"],
        "model":["flat","flat","control"],"factors":[8,8,1],"accepted":[True,False,True]})
    tables = {"rough_foundation_summary":summary,
        "rough_foundation_candidate_selection":pd.DataFrame(),
        "rough_foundation_failures":pd.DataFrame()}
    messages, figures, selected_tables = [], [], []
    results = {"manifest":{"rough_foundation_executed":enabled,
        "rough_foundation":{"source_elapsed_seconds":55.}},"tables":tables}
    namespace = {"results":results,"tables":tables,"np":np,"pd":pd,
        "Markdown":lambda text:text,"display":messages.append,
        "present_figure":lambda result,name,**kwargs:figures.append(name),
        "present_table":lambda table,**kwargs:selected_tables.append(table)}
    # Deliberately omit RUN_ROUGH_FOUNDATION and every parameter-cell variable.
    for index,cell in enumerate(cells):
        exec(compile(cell.source,f"recorded-foundation-display-{index}","exec"),namespace)
    if enabled:
        assert len(figures) == 2 and len(selected_tables) == 4
        assert any("readiness remains **false**" in message for message in messages)
    else:
        assert not figures and not selected_tables
        assert any("disabled" in message for message in messages)
