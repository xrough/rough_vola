"""American stopping evidence must not relabel old European controls."""
import hashlib
import json

import pytest

from rvlab.experiments import _pricing_rough_american as study


def _archive(tmp_path):
    archive = tmp_path / "snapshot"
    results = archive / "results"
    results.mkdir(parents=True)
    (results / "rough_precision_summary.csv").write_text("stage,accepted\npooled final,True\n")
    (results / "rough_precision_failures.csv").write_text("\n")
    (results / "rough_precision_final_acceptance.png").write_bytes(b"original pixels")
    manifest = {"evidence_version": "pricing_research_v1",
        "notebook": "01_pricing_and_exercise", "created_at": "original execution",
        "tables": {"rough_precision_summary": 1, "rough_precision_failures": 0},
        "figures": ["rough_precision_final_acceptance"],
        "rough_precision": {"backend": "rough_heston_simulation_precision",
            "source_elapsed_seconds": 123., "profile_fingerprint": "historical-profile",
            "sources": {"rough_simulation.py": "original-source"},
            "tested_european_prices_accepted": True}}
    (results / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for path in results.iterdir():
        payload = path.read_bytes()
        rows.append({"preserved": str(path.relative_to(archive)),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    (archive / "preservation_manifest.json").write_text(json.dumps({"files": rows}))
    return archive


def test_precision_replay_preserves_european_identity_and_original_cost(tmp_path):
    archive = _archive(tmp_path)
    result = study.load_historical_precision(archive=archive)
    manifest = result["manifest"]
    assert result["tables"]["summary"].iloc[0].accepted
    assert result["tables"]["failures"].empty
    assert result["figures"]["final_acceptance"].read_bytes() == b"original pixels"
    assert manifest["profile_fingerprint"] == "historical-profile"
    assert manifest["sources"] == {"rough_simulation.py": "original-source"}
    assert manifest["source_elapsed_seconds"] == 123.
    assert manifest["tested_european_prices_accepted"]
    assert manifest["historical_replay"] and not manifest["current_engine_execution"]
    assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
    assert study.load_historical_precision(archive=archive, compute_figures=False)["figures"] == {}


@pytest.mark.parametrize("target", ["rough_precision_summary.csv", "rough_precision_final_acceptance.png"])
def test_precision_replay_rejects_changed_recorded_prices_and_pixels(tmp_path, target):
    archive = _archive(tmp_path)
    (archive / "results" / target).write_bytes(b"modified")
    with pytest.raises(ValueError, match="Changed preserved"):
        study.load_historical_precision(archive=archive)


def test_precision_replay_checks_undisplayed_sources_too(tmp_path):
    archive = _archive(tmp_path)
    path = archive / "numerical_source.py"
    path.write_text("# saved numerical source\n")
    manifest_path = archive / "preservation_manifest.json"
    inventory = json.loads(manifest_path.read_text())
    inventory["files"].append({"preserved": path.name,
        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path.write_text(json.dumps(inventory))
    path.unlink()
    with pytest.raises(ValueError, match="Missing preserved"):
        study.load_historical_precision(archive=archive)


def test_precision_replay_rejects_a_verified_different_backend(tmp_path):
    archive = _archive(tmp_path)
    path = archive / "results/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rough_precision"]["backend"] = "american rough Heston"
    path.write_text(json.dumps(manifest))
    inventory_path = archive / "preservation_manifest.json"
    inventory = json.loads(inventory_path.read_text())
    row = next(row for row in inventory["files"] if row["preserved"] == "results/manifest.json")
    row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="not the European"):
        study.load_historical_precision(archive=archive)


def test_stopping_gate_adds_sampling_and_paired_gap_without_relaxing_allowance():
    result = study.stopping_accuracy(1., .0001, 1.0004, .0004, .00002)
    assert result["price_allowance"] == pytest.approx(.0011)
    assert result["discrete_price_error_budget"] == pytest.approx(.0004+1.96*(.0001+.00002))
    assert result["discrete_price_gate_passed"]
    # Passing both components separately need not pass their common budget.
    failed = study.stopping_accuracy(1., .0004, 1.0005, .0005, .0001)
    assert failed["sampling_precision_passed"]
    assert failed["discrete_stopping_gap_passed"]
    assert not failed["discrete_price_gate_passed"]


def test_wide_interval_or_inconsistent_dual_is_not_accepted():
    wide = study.stopping_accuracy(1., .001, 1., 0., .00001)
    assert not wide["sampling_precision_passed"]
    assert not wide["discrete_price_gate_passed"]
    inconsistent = study.stopping_accuracy(1., 0., .99, -.01, .0001)
    assert not inconsistent["dual_consistency_passed"]
    assert not inconsistent["discrete_price_gate_passed"]


@pytest.mark.parametrize("args", [
    (1., -.1, 1., 0., .1), (1., .1, 1., 0., -.1),
    (1., .1, 1.1, .2, .1), (float("nan"), .1, 1., 0., .1),
])
def test_stopping_diagnostics_reject_malformed_statistics(args):
    with pytest.raises(ValueError):
        study.stopping_accuracy(*args)


def test_delta_sampling_success_alone_does_not_certify_delta_accuracy():
    result = study.delta_accuracy(.5, .0001)
    assert result["delta_allowance"] == pytest.approx(.0045)
    assert result["delta_sampling_precision_passed"]
    assert not result["delta_reference_gate_passed"]
    checked = study.delta_accuracy(.5, .0001, reference=.5001, reference_stable=True)
    assert checked["delta_reference_gate_passed"]
    unstable = study.delta_accuracy(.5, .0001, reference=.5, reference_stable=False)
    assert not unstable["delta_reference_gate_passed"]


@pytest.mark.parametrize("delta,se", [(float("inf"), .1), (.5, -.1)])
def test_delta_diagnostics_reject_invalid_statistics(delta, se):
    with pytest.raises(ValueError, match="Delta estimates"):
        study.delta_accuracy(delta, se)


def test_synthetic_stopping_contracts_are_predeclared_and_do_not_use_market_data():
    contracts = study.american_contracts()
    assert len(contracts) == 15 and contracts.contract_id.is_unique
    assert sorted(contracts.days.unique()) == [14, 30, 60]
    assert set(contracts.query("model == 'flat'").strike) == {95., 100., 105.}
    assert contracts.is_call.sum() == 1
    assert contracts.loc[contracts.is_call, "scenario"].item() == "cash dividend"
    assert "spy" not in contracts.to_json().lower()


def test_independent_pde_refinement_rows_survive_json_checkpoint_serialization(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from rvlab import pricing
    from rvlab.pricing import FDConfig
    from rvlab.experiments._pricing_rough_foundation import _market, _jsonable
    monkeypatch.setattr(pricing, "evaluate_vanilla", lambda *a, **kw: SimpleNamespace(
        prices=1., deltas=-.5, price_stable=True, delta_stable=False,
        config=FDConfig(), diagnostics={"refinement": pd.DataFrame([
            {"axis": "spot", "price_change": .0001},
            {"axis": "time", "price_change": .0002}])}))
    contract = study.american_contracts().query("model == 'Hurst .5'").iloc[0].to_dict()
    record = study._fd_reference(study._american_models()["Hurst .5"], _market(), contract)
    encoded = json.dumps(_jsonable(record), allow_nan=False)
    recovered = json.loads(encoded)
    assert recovered["diagnostics"]["refinement"] == [
        {"axis": "spot", "price_change": .0001}, {"axis": "time", "price_change": .0002}]
    assert not recovered["delta_stable"]


def _mock_stopping_campaign(monkeypatch, tmp_path, *, failed=False):
    from dataclasses import asdict
    from types import SimpleNamespace
    from rvlab import pricing
    from rvlab.pricing import ExponentialKernel, PricingError
    monkeypatch.setattr(pricing, "approximate_fractional_kernel",
        lambda h, t, n, **kw: ExponentialKernel((0. if h == .5 else 1.,), (1.,), hurst=h, horizon=t))
    monkeypatch.setattr(study, "_fd_reference", lambda *args: {
        "status": "completed", "price": 1., "delta": -.5,
        "price_stable": True, "delta_stable": True})
    calls = []
    blocked = [False]
    def run(params, market, kernel, contract, config):
        assert not blocked[0], "A matching checkpoint should have been reused"
        calls.append((contract["contract_id"], config.training_seed, config.valuation_seed))
        if failed and contract["contract_id"] == "rough_put_60_100":
            raise PricingError("retained stopping regression failure")
        return {"status": "completed", "prices": [1.], "primal_standard_errors": [.00001],
            "dual_prices": [1.0001], "dual_standard_errors": [.00002],
            "paired_gaps": [.0001], "paired_gap_standard_errors": [.00001],
            "deltas": [-.5], "delta_standard_errors": [.0001],
            "raw_prices": [1.1], "raw_standard_errors": [.1], "exercise_probabilities": [.2],
            "bump_deltas": [-.5, -.50001], "bump_delta_standard_errors": [.0001, .0002],
            "one_sided_deltas": [-.51, -.49], "one_sided_delta_standard_errors": [.001, .001],
            "retrained_deltas": [None], "retrained_delta_standard_errors": [None],
            "settings": asdict(config), "diagnostics": {"independent_training_valuation": True}}
    monkeypatch.setattr(study, "_stopping_trial", run)
    context = SimpleNamespace(output_dir=tmp_path, progress=lambda _: None)
    options = dict(training_paths=4, valuation_paths=4, steps=4, exercise_steps=2,
        replications=2, run_refinements=False,
        contract_ids=("rough_put_60_100", "heston_limit_put", "constant_variance_put"),
        compute_figures=False)
    result = study.rough_american_study(context, **options)
    return result, calls, blocked, context, options


def test_campaign_has_disjoint_streams_and_never_promotes_conditional_precision_to_readiness(monkeypatch, tmp_path):
    result, calls, _, _, _ = _mock_stopping_campaign(monkeypatch, tmp_path)
    assert len(calls) == 6
    assert not set(seed for _, seed, _ in calls) & set(seed for _, _, seed in calls)
    prices = result["tables"]["prices"]
    assert prices.discrete_price_gate_passed.all()
    assert not prices.american_accuracy_certified.any() and not prices.stock_delta_ready.any()
    limits = prices[prices.model.ne("flat")]
    assert limits.price_reference_gate_passed.all() and limits.delta_reference_gate_passed.all()
    assert result["tables"]["refinement"].paired_between_trials.eq(False).all()
    assert not result["manifest"]["american_pricing_ready"]
    assert not result["manifest"]["stock_delta_ready"]


def test_campaign_retains_failed_policy_trials_instead_of_dropping_them(monkeypatch, tmp_path):
    result, _, _, _, _ = _mock_stopping_campaign(monkeypatch, tmp_path, failed=True)
    assert len(result["tables"]["trials"]) == 6
    assert result["tables"]["trials"].status.eq("numerical_failure").sum() == 2
    assert len(result["tables"]["prices"]) == 4
    assert result["tables"]["failures"].error.str.contains("retained stopping").all()


def test_campaign_replays_only_fingerprint_matched_trials_with_original_runtimes(monkeypatch, tmp_path):
    first, calls, blocked, context, options = _mock_stopping_campaign(monkeypatch, tmp_path)
    blocked[0] = True
    second = study.rough_american_study(context, **options)
    assert len(calls) == 6
    assert second["manifest"]["checkpoint_replay"]
    assert second["manifest"]["profile_fingerprint"] == first["manifest"]["profile_fingerprint"]
    assert second["manifest"]["source_elapsed_seconds"] == first["manifest"]["source_elapsed_seconds"]


def test_explicit_reexecution_keeps_previous_raw_trial_bytes(monkeypatch, tmp_path):
    from pathlib import Path
    first, _, _, context, options = _mock_stopping_campaign(monkeypatch, tmp_path)
    previous = {str(path): path.read_bytes()
        for path in Path(first["manifest"]["checkpoint_dir"]).glob("*.json")}
    second = study.rough_american_study(context, reuse_checkpoints=False, **options)
    assert second["manifest"]["checkpoint_dir"] != first["manifest"]["checkpoint_dir"]
    assert not second["manifest"]["checkpoint_replay"]
    assert all(Path(path).read_bytes() == payload for path, payload in previous.items())


def test_source_edit_before_a_trial_becomes_an_identifiable_retained_failure(monkeypatch, tmp_path):
    first, _, _, context, options = _mock_stopping_campaign(monkeypatch, tmp_path)
    monkeypatch.setattr(study, "_core_source_hashes", lambda: {"unexpected_source.py": "changed"})
    result = study.rough_american_study(context, reuse_checkpoints=False, **options)
    assert result["tables"]["prices"].empty
    assert len(result["tables"]["trials"]) == 6
    assert result["tables"]["trials"].status.eq("numerical_failure").all()
    assert result["tables"]["failures"].error.str.contains("Numerical source changed").all()
    assert not result["manifest"]["american_pricing_ready"]


def test_disjoint_prefills_share_full_profile_and_keep_partial_status_explicit(monkeypatch, tmp_path):
    from types import SimpleNamespace
    first, calls, blocked, _, options = _mock_stopping_campaign(monkeypatch, tmp_path)
    context = SimpleNamespace(output_dir=tmp_path/"partitioned", progress=lambda _: None)
    preflight = study.rough_american_study(context, trial_indices=(), **options)
    assert preflight["manifest"]["partial_campaign"]
    assert preflight["manifest"]["completed_selected_trials"] == 0
    assert preflight["tables"]["prices"].empty
    one = study.rough_american_study(context, trial_indices=(0, 2), **options)
    two = study.rough_american_study(context, trial_indices=(1, 3, 4, 5), **options)
    assert one["manifest"]["completed_selected_trials"] == 2
    assert two["manifest"]["completed_selected_trials"] == 4
    assert one["manifest"]["profile_fingerprint"] == two["manifest"]["profile_fingerprint"]
    assert one["manifest"]["profile_fingerprint"] == first["manifest"]["profile_fingerprint"]
    blocked[0] = True
    complete = study.rough_american_study(context, **options)
    assert len(calls) == 12
    assert not complete["manifest"]["partial_campaign"]
    assert complete["manifest"]["completed_selected_trials"] == 6
    assert complete["manifest"]["checkpoint_replay"]


@pytest.mark.parametrize("options", [
    {"training_paths": True}, {"valuation_paths": 1}, {"steps": 0},
    {"replications": 0}, {"run_refinements": 1}, {"contract_ids": ["SPY"]},
    {"contract_ids": ["cash_call", "cash_call"]}, {"contract_ids": []},
])
def test_campaign_rejects_invalid_predeclared_design_before_running_models(tmp_path, options):
    from types import SimpleNamespace
    context = SimpleNamespace(output_dir=tmp_path, progress=lambda _: None)
    with pytest.raises((ValueError, TypeError)):
        study.rough_american_study(context, compute_figures=False, **options)


@pytest.mark.parametrize("options,error", [
    ({"run_rough_american": 1}, ValueError),
    ({"rough_american_kwargs": []}, TypeError),
    ({"rough_american_kwargs": {"seed": 12}}, ValueError),
    ({"rough_american_kwargs": {"reuse_checkpoints": False}}, ValueError),
    ({"run_rough_american": True, "rough_precision_kwargs": {"paths": 4}}, ValueError),
    ({"run_rough_american": True, "rough_study_kwargs": {"paths": 4}}, ValueError),
])
def test_workflow_rejects_ambiguous_american_or_historical_execution_options(options, error):
    from types import SimpleNamespace
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):
        pricing_research(SimpleNamespace(), **options)


def test_american_workflow_replays_previous_european_controls_without_rerunning_them(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from rvlab.experiments import _pricing_research as research
    from rvlab.experiments import _pricing_rough_foundation as foundation
    from rvlab.experiments import _pricing_rough_precision as precision
    from rvlab.pricing import PricingConfig, FDConfig
    plt = pytest.importorskip("matplotlib.pyplot")
    foundation_bundle = {"tables": {"summary": pd.DataFrame([{"accepted": False}])},
        "figures": {}, "manifest": {"backend": "rough_heston_lifted_simulation_foundation", "historical_replay": True}}
    precision_bundle = {"tables": {"summary": pd.DataFrame([{"accepted": True}])},
        "figures": {}, "manifest": {"backend": "rough_heston_simulation_precision", "historical_replay": True}}
    american_bundle = {"tables": {"prices": pd.DataFrame([{"american_accuracy_certified": False}])},
        "figures": {}, "manifest": {"backend": "rough_heston_lifted_primal_dual", "american_pricing_ready": False}}
    monkeypatch.setattr(foundation, "rough_foundation_study", lambda *a, **k: pytest.fail("Must preserve old foundation identity"))
    monkeypatch.setattr(precision, "rough_precision_study", lambda *a, **k: pytest.fail("Must not rerun the European campaign"))
    monkeypatch.setattr(study, "load_historical_foundation", lambda context: foundation_bundle)
    monkeypatch.setattr(study, "load_historical_precision", lambda context: precision_bundle)
    calls = []
    monkeypatch.setattr(study, "rough_american_study", lambda context, **options: calls.append(options) or american_bundle)
    try:
        result = research.pricing_research(SimpleNamespace(progress=lambda _: None),
            fourier_config=PricingConfig(32, 40., 64), fd_config=FDConfig(24, 8, 20),
            run_rough_foundation=True, run_rough_precision=True, run_rough_american=True,
            rough_reuse_checkpoints=False, rough_american_kwargs={"training_paths": 4})
        assert calls == [{"seed": 42, "reuse_checkpoints": False, "training_paths": 4}]
        manifest = result["manifest"]
        assert manifest["rough_foundation_included"] and manifest["rough_precision_included"]
        assert not manifest["rough_foundation_executed"] and not manifest["rough_precision_executed"]
        assert manifest["rough_american_executed"]
        assert manifest["rough_foundation"] is foundation_bundle["manifest"]
        assert manifest["rough_precision"] is precision_bundle["manifest"]
        assert manifest["rough_american"] is american_bundle["manifest"]
        assert result["tables"]["rough_american_prices"] is american_bundle["tables"]["prices"]
    finally:
        plt.close("all")


def test_actual_notebook_all_failed_stopping_display_is_an_audit_not_an_exception():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    nbformat = pytest.importorskip("nbformat")
    from rvlab.experiments.pricing_presentation import _table_html
    notebook = nbformat.read(Path(study.__file__).resolve().parents[3] /
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb", as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and
        cell.source.startswith('if results["manifest"].get("rough_american_included"')]
    assert len(cells) == 4
    shown = []
    namespace = {"pd": pd, "np": np, "Markdown": str, "display": shown.append,
        "results": {"manifest": {"rough_american_executed": True,
            "rough_american": {"source_elapsed_seconds": 1.}}, "figures": {}},
        "tables": {"rough_american_prices": pd.DataFrame(),
            "rough_american_refinement": pd.DataFrame(),
            "rough_american_failures": pd.DataFrame([{"trial": "all failed", "error": "retained"}])},
        "present_table": lambda frame, **kwargs: shown.append(_table_html(frame, **kwargs)),
        "present_figure": lambda *args, **kwargs: pytest.fail("No failed price should produce a figure")}
    for index, source in enumerate(cells):
        exec(compile(source, f"rough-stopping-display-{index}", "exec"), namespace)
    text = "\n".join(str(value) for value in shown)
    assert "No stopping trial produced a usable estimate" in text
    assert "0 completed trial-contract rows" in text
    assert "1 numerical failures" in text
    assert "remain **false**" in text


def test_arithmetic_repair_comparison_preserves_matched_raw_path_values_and_new_failures(monkeypatch, tmp_path):
    result, _, _, _, _ = _mock_stopping_campaign(monkeypatch, tmp_path)
    original = result["tables"]["prices"]
    repaired = original.copy()
    repaired["price"] += 1e-12
    repaired["delta"] -= 2e-12
    details, summary = study.arithmetic_repair_comparison(repaired, original.iloc[:-1])
    assert len(details) == 5
    assert details.raw_payoff_bit_identical.all()
    assert summary.iloc[0].same_settings and summary.iloc[0].raw_payoff_bit_identical
    assert summary.iloc[0].newly_finite_trials == 1
    assert summary.iloc[0].max_price_change == pytest.approx(1e-12, abs=1e-15)
    changed = repaired.copy()
    changed.loc[changed.index[0], "ridge"] *= 2
    _, flagged = study.arithmetic_repair_comparison(changed, original)
    assert not flagged.iloc[0].same_settings
