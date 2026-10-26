"""Paired stopping attribution preserves targets, covariance and prior evidence."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_stopping_attribution as study


def _samples():
    shock = np.array([-.03, .02, .01, -.01, .025, -.015])
    raw = np.column_stack([2.+shock, 2.002+shock])
    primal = raw[..., None]-shock[:, None, None]*np.array([1., 2.])
    upper = np.broadcast_to([2.1, 2.05], (len(shock), 2)).copy()
    return raw, primal, upper


def test_cross_matrix_keeps_policy_and_control_counterfactuals_separate():
    raw, primal, upper = _samples()
    tables = study.paired_attribution_tables(raw, primal, upper,
        policy_names=["base", "candidate"], control_names=["base", "optimized"])
    matrix = tables["matrix"]
    assert len(matrix) == 4
    assert matrix.groupby("control").dual_price.nunique().eq(1).all()
    assert matrix.groupby("policy").raw_price.nunique().eq(1).all()
    policies = tables["policy_changes"]
    assert policies.same_martingale.all()
    assert policies.dual_change.eq(0).all()
    np.testing.assert_allclose(policies.controlled_lower_change, .002)
    assert policies.controlled_change_standard_error.max() < 1e-15
    controls = tables["control_changes"]
    assert controls.same_exercise_policy.all() and controls.raw_payoff_change.eq(0).all()
    np.testing.assert_allclose(controls.dual_change, -.05)
    assert not matrix.american_pricing_ready.any() and not matrix.stock_delta_ready.any()


def test_cross_matrix_standard_errors_use_actual_paired_samples():
    raw, primal, upper = _samples()
    tables = study.paired_attribution_tables(raw, primal, upper,
        policy_names=["base", "candidate"], control_names=["base", "optimized"])
    row = tables["control_changes"].iloc[0]
    difference = (upper[:, 1]-primal[:, 0, 1])-(upper[:, 0]-primal[:, 0, 0])
    assert row.gap_change == pytest.approx(difference.mean())
    assert row.gap_change_standard_error == pytest.approx(difference.std(ddof=1)/np.sqrt(len(difference)))
    # Independent endpoint errors would be much larger here and discard the
    # deliberately shared valuation paths.
    assert row.dual_change_standard_error < 1e-15


@pytest.mark.parametrize("which", ["shape", "nonfinite", "domination", "names", "empty"])
def test_counterfactual_matrix_rejects_invalid_or_unshared_inputs(which):
    raw, primal, upper = _samples()
    policy_names, control_names = ["base", "candidate"], ["base", "optimized"]
    if which == "shape": upper = upper[:2]
    elif which == "nonfinite": primal[0, 0, 0] = np.nan
    elif which == "domination": upper[:] = 0.
    elif which == "names": policy_names = ["duplicate", "duplicate"]
    elif which == "empty": raw, primal, upper = raw[:, :0], primal[:, :0], upper
    with pytest.raises(ValueError):
        study.paired_attribution_tables(raw, primal, upper,
            policy_names=policy_names, control_names=control_names)


def _archive(tmp_path):
    archive = tmp_path/"snapshot"
    results = archive/"results"
    results.mkdir(parents=True)
    (results/"rough_american_accuracy_prices.csv").write_text("price,discrete_price_gate_passed\n2.859477144706678,False\n")
    (results/"rough_american_accuracy_failures.csv").write_text("\n")
    (results/"rough_american_accuracy_accuracy_budgets.png").write_bytes(b"original pixels")
    (archive/"source.py").write_text("# original pricing source\n")
    manifest = {"evidence_version": "pricing_research_v1", "notebook": "01_pricing_and_exercise",
        "created_at": "original execution", "tables": {"rough_american_accuracy_prices": 1,
            "rough_american_accuracy_failures": 0}, "figures": ["rough_american_accuracy_accuracy_budgets"],
        "rough_american_accuracy": {"backend": "rough_heston_lifted_primal_dual",
            "profile_fingerprint": study.HISTORICAL_ACCURACY_PROFILE,
            "source_elapsed_seconds": 4567.869738458656, "sources": {"original.py": "old-hash"}}}
    (results/"manifest.json").write_text(json.dumps(manifest))
    rows = []
    for path in sorted(archive.rglob('*')):
        if path.is_file(): rows.append({"preserved": str(path.relative_to(archive)),
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (archive/"preservation_manifest.json").write_text(json.dumps({"files": rows}))
    return archive


def test_historical_replay_keeps_original_identity_and_execution_cost(tmp_path):
    archive = _archive(tmp_path)
    result = study.load_historical_accuracy(archive=archive)
    manifest = result["manifest"]
    assert manifest["profile_fingerprint"] == study.HISTORICAL_ACCURACY_PROFILE
    assert manifest["source_elapsed_seconds"] == 4567.869738458656
    assert manifest["sources"] == {"original.py": "old-hash"}
    assert manifest["historical_replay"] and not manifest["current_engine_execution"]
    assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
    assert result["tables"]["failures"].empty
    assert result["tables"]["prices"].price.item() == 2.859477144706678
    assert result["figures"]["accuracy_budgets"].read_bytes() == b"original pixels"


@pytest.mark.parametrize("relative", ["source.py", "results/rough_american_accuracy_prices.csv",
    "results/rough_american_accuracy_accuracy_budgets.png"])
def test_historical_replay_refuses_changed_source_prices_and_figures(tmp_path, relative):
    archive = _archive(tmp_path)
    (archive/relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed preserved"):
        study.load_historical_accuracy(archive=archive)


def test_verified_different_profile_is_not_relabelled_the_prior_campaign(tmp_path):
    archive = _archive(tmp_path)
    path = archive/"results/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rough_american_accuracy"]["profile_fingerprint"] = "different-profile"
    path.write_text(json.dumps(manifest))
    inventory_path = archive/"preservation_manifest.json"
    inventory = json.loads(inventory_path.read_text())
    row = next(row for row in inventory["files"] if row["preserved"] == "results/manifest.json")
    row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="declared 14-trial"):
        study.load_historical_accuracy(archive=archive)


@pytest.mark.parametrize("options,error", [
    ({"run_rough_stopping_attribution": 1}, ValueError),
    ({"rough_stopping_attribution_kwargs": []}, TypeError),
    ({"rough_stopping_attribution_kwargs": {"seed": 7}}, ValueError),
    ({"run_rough_stopping_attribution": True, "rough_american_accuracy_kwargs": {"trial_specs": []}}, ValueError),
])
def test_workflow_rejects_ambiguous_attribution_and_historical_options(options, error):
    from types import SimpleNamespace
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):
        pricing_research(SimpleNamespace(), **options)


def test_new_attribution_workflow_never_reprices_or_relabels_previous_accuracy(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from rvlab.experiments import _pricing_research as research
    from rvlab.experiments import _pricing_rough_american as american
    from rvlab.experiments import _pricing_rough_american_accuracy as accuracy
    from rvlab.pricing import PricingConfig, FDConfig
    import matplotlib.pyplot as plt
    historical_american = {"tables": {}, "figures": {},
        "manifest": {"backend": "rough_heston_lifted_primal_dual", "historical_replay": True}}
    historical_accuracy = {"tables": {"prices": pd.DataFrame([{"price": 2.859477}])},
        "figures": {}, "manifest": {"backend": "rough_heston_lifted_primal_dual",
            "historical_replay": True, "profile_fingerprint": study.HISTORICAL_ACCURACY_PROFILE,
            "source_elapsed_seconds": 4567.869738458656}}
    current = {"tables": {"matrix": pd.DataFrame([{"policy": "base", "control": "optimized"}])},
        "figures": {}, "manifest": {"backend": "rough_heston_stopping_attribution"}}
    monkeypatch.setattr(american, "load_historical_american", lambda *a, **k: historical_american)
    monkeypatch.setattr(study, "load_historical_accuracy", lambda *a, **k: historical_accuracy)
    monkeypatch.setattr(accuracy, "rough_american_accuracy_study", lambda *a, **k: pytest.fail("Must not rerun the 76-minute historical campaign"))
    calls = []
    monkeypatch.setattr(study, "rough_stopping_attribution_study",
        lambda context, **options: calls.append(options) or current, raising=False)
    monkeypatch.setattr(study, "add_attribution_extras", lambda value: value)
    monkeypatch.setattr(study, "add_attribution_execution", lambda value, **kwargs: value)
    try:
        result = research.pricing_research(SimpleNamespace(progress=lambda _: None, output_dir=tmp_path),
            fourier_config=PricingConfig(32, 40., 64), fd_config=FDConfig(24, 8, 20),
            run_rough_american_accuracy=True, run_rough_stopping_attribution=True,
            rough_stopping_attribution_kwargs={"trial_specs": []}, rough_reuse_checkpoints=False)
        manifest = result["manifest"]
        assert manifest["rough_american_accuracy_included"]
        assert not manifest["rough_american_accuracy_executed"]
        assert manifest["rough_american_accuracy_execution_mode"] == "preserved_historical_records"
        assert manifest["rough_american_accuracy"] is historical_accuracy["manifest"]
        assert manifest["rough_stopping_attribution_executed"]
        assert result["tables"]["rough_american_accuracy_prices"] is historical_accuracy["tables"]["prices"]
        assert calls == [{"seed": 42, "reuse_checkpoints": False, "trial_specs": []}]
    finally:
        plt.close("all")


def test_same_calendar_reference_split_retains_negative_components_and_reference_change():
    raw, primal, upper = _samples()
    matrix = study.paired_attribution_tables(raw, primal, upper,
        policy_names=["base", "candidate"], control_names=["base", "optimized"])["matrix"]
    result = study.finite_reference_partition(matrix, reference_price=1.99,
        reference_change=.00003, reference_stable=True, reference_scope="same declared finite GBM calendar")
    np.testing.assert_allclose(result.reconstructed_gap, matrix.paired_gap)
    assert result.estimated_policy_shortfall.lt(0).all()
    assert result.policy_point_share.lt(0).all()
    assert result.upper_point_share.gt(1).all()
    first = result.iloc[0]
    assert first.policy_shortfall_interval_high-first.estimated_policy_shortfall == pytest.approx(.00003)
    assert not result.american_pricing_ready.any()


def test_attribution_figures_keep_failed_gates_and_negative_components_visible():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    raw, primal, upper = _samples()
    tables = study.paired_attribution_tables(raw, primal, upper,
        policy_names=["base", "candidate"], control_names=["base", "optimized"])
    tables["reference_partition"] = study.finite_reference_partition(tables["matrix"],
        reference_price=1.999, reference_change=.0001,
        reference_stable=True, reference_scope="same finite calendar")
    assert not tables["matrix"].discrete_price_gate_passed.any()
    figures = study._attribution_figures(tables)
    assert set(figures) == {"fixed_policy_bounds", "paired_attribution", "known_limit_partition"}
    for figure in figures.values():
        figure.canvas.draw()
        plt.close(figure)
    assert study._attribution_figures({"matrix": pd.DataFrame()}) == {}


def test_paired_checkpoint_samples_are_independently_byte_verified(tmp_path):
    path = tmp_path/"paired_samples.npz"
    raw, primal, upper = _samples()
    item = study._save_sample_archive(path, raw=raw, primal=primal, upper=upper)
    record = {"sample_archives": [item]}
    assert study._verify_sample_archives(record) is record
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="Changed paired-sample"):
        study._verify_sample_archives(record)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing paired-sample"):
        study._verify_sample_archives(record)


def _case(**updates):
    return {"case": "rough", "contract_id": "rough_put_60_100",
        "config": {"training_method": "fourier_premium", "premium_cap": .7,
            "training_paths": 8, "valuation_paths": 8, "steps": 4, "exercise_steps": 2},
        "dual_configs": {"static": {"training_paths": 8, "candidate_bins": (1,)}},
        "policy_variants": {"positive": {"premium_floor": 0., "premium_cap": .7}}, **updates}


def test_attribution_design_assigns_four_disjoint_streams_before_simulation():
    rows = study._validated_attribution_design([_case(), _case(case="repeat")], 42)
    seeds = [value for row in rows for value in (row["config"].training_seed,
        row["config"].valuation_seed, row["dual_seed"], row["improvement_seed"])]
    assert len(set(seeds)) == 8
    assert rows[0]["dual_configs"]["static"].training_seed == rows[0]["dual_seed"]
    assert rows[0]["config"].valuation_seed not in seeds[::4]


@pytest.mark.parametrize("updates", [
    {"contract_id": "cash_call"}, {"case": ""}, {"factors": 1},
    {"config": {"training_seed": 3}}, {"dual_configs": {"bad": {"training_seed": 4}}},
    {"dual_configs": {"a": {"training_paths": 8}, "b": {"training_paths": 16}}},
    {"policy_variants": {"bad": {"premium_floor": .8, "premium_cap": .7}}},
    {"improvement_paths": 1}, {"reference_nodes": 5},
])
def test_attribution_design_rejects_malformed_or_confounded_cases(updates):
    with pytest.raises((ValueError, TypeError)):
        study._validated_attribution_design([_case(**updates)], 42)


def test_control_representation_selection_uses_training_score_only():
    from types import SimpleNamespace
    def fit(score):
        return SimpleNamespace(diagnostics={"selection_objective": "error_budget",
            "candidates": [{"heldout_error_budget": .006}], "selected_score": score,
            "selected_candidate": "4 bins", "selection_source": "held-out training"})
    selected, rows = study._select_control_from_training({"static": fit(.0059), "state": fit(.0042)})
    assert selected == "state"
    assert [row["control"] for row in rows if row["selected_before_valuation"]] == ["state"]
    assert not any(row["valuation_used_for_selection"] for row in rows)
    selected, _ = study._select_control_from_training({"worse": fit(.007)})
    assert selected == "unit baseline"


def test_real_tiny_attribution_crosses_policies_without_valuation_selection(tmp_path):
    from rvlab.pricing import approximate_fractional_kernel, RoughEuropeanControlConfig
    from rvlab.experiments._pricing_rough_american import _american_models
    from rvlab.experiments._pricing_rough_foundation import _market
    spec = _case()
    spec["config"]["premium_control"] = RoughEuropeanControlConfig(8, 35., 64)
    case = study._validated_attribution_design([spec], 42)[0]
    kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003, max_nfev=4000)
    result = study._attribution_trial(case, _american_models()["flat"], _market(), kernel, tmp_path)
    assert result["status"] == "completed"
    assert result["same_raw_payoffs_across_controls"] and result["same_upper_samples_across_policies"]
    assert not result["policy_variants_selected_on_valuation"]
    assert study._verify_sample_archives(result) is result
    frozen = json.loads((tmp_path/"selection_before_valuation.json").read_text())
    assert frozen["reserved_valuation_seed"] == 5_000_042
    assert not frozen["valuation_used_for_selection"]
    matrix = pd.DataFrame(result["tables"]["matrix"])
    assert len(matrix) == 4 and not matrix.american_pricing_ready.any()


def test_study_replays_cases_and_preserves_failed_attempts_without_repricing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rvlab.pricing import PricingError
    from rvlab.experiments import _pricing_rough_american as american
    calls = []
    raw, primal, upper = _samples()
    paired = study.paired_attribution_tables(raw, primal, upper,
        policy_names=["base policy", "candidate"], control_names=["unit baseline", "optimized"])
    def trial(case, *args):
        calls.append(case["case"])
        if case["case"] == "failed":
            raise PricingError("preserved numerical failure")
        return {"status": "completed", "tables": {key: value.to_dict("records") for key, value in paired.items()},
            "selected_control": "unit baseline", "sample_archives": [], "reference": None}
    monkeypatch.setattr(study, "_attribution_trial", trial)
    monkeypatch.setattr(study, "_attribution_sources", lambda: {"study_numerics": "declared"})
    monkeypatch.setattr(american, "_core_source_hashes", lambda: {})
    monkeypatch.setattr(american, "_preserve_sources", lambda *args: [])
    context = SimpleNamespace(output_dir=tmp_path, progress=lambda value: None)
    specs = [_case(), _case(case="failed")]
    first = study.rough_stopping_attribution_study(context, trial_specs=specs, compute_figures=False)
    assert len(first["tables"]["matrix"]) == 4
    assert first["tables"]["trials"].status.tolist() == ["completed", "numerical_failure"]
    assert "preserved numerical failure" in first["tables"]["failures"].error.iloc[0]
    second = study.rough_stopping_attribution_study(context, trial_specs=specs, compute_figures=False)
    assert calls == ["rough", "failed"]
    assert second["manifest"]["checkpoint_replay"]
    assert first["manifest"]["profile_fingerprint"] == second["manifest"]["profile_fingerprint"]
    assert second["manifest"]["source_elapsed_seconds"] == first["manifest"]["source_elapsed_seconds"]
    assert not second["manifest"]["valuation_used_for_selection"]
    assert second["manifest"]["optimizer_counts_incomplete_on_failed_cases"]


def test_current_notebook_attribution_display_handles_all_failed_cases():
    import nbformat
    from pathlib import Path
    notebook = nbformat.read(Path(__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb", as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and
        cell.source.startswith('if results["manifest"].get("rough_stopping_attribution_included"')]
    assert len(cells) == 3
    messages = []
    environment = {"pd": pd, "results": {"manifest": {"rough_stopping_attribution_executed": False,
        "rough_stopping_attribution_included": True,
        "rough_stopping_attribution": {}}, "figures": {}},
        "tables": {"rough_stopping_attribution_failures": pd.DataFrame([{"error": "retained"}])},
        "display": lambda value: messages.append(value), "Markdown": str,
        "present_table": lambda *args, **kwargs: None}
    for cell in cells:
        exec(cell, environment)
    assert any("No completed attribution" in value for value in messages)


def test_execution_timing_verifies_runner_bytes_and_keeps_worker_span_distinct(tmp_path):
    profile = "new-profile"
    for label, started, ended, indices in (("a", "00:00:00", "00:02:00", [0, 2]),
                                           ("b", "00:00:01", "00:03:00", [1])):
        folder = tmp_path/"rough_stopping_attribution_v1/campaign_v1"/label
        folder.mkdir(parents=True)
        payload = b"# exact executed runner\n"
        (folder/"executed_runner.py").write_bytes(payload)
        (folder/"manifest.json").write_text(json.dumps({"profile_fingerprint": profile,
            "mode": "worker", "runner_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_replay": False, "started_at": "2026-10-18T"+started+"+00:00",
            "completed_at": "2026-10-18T"+ended+"+00:00", "elapsed_seconds": 120.,
            "selected_case_indices": indices}))
    bundle = {"tables": {}, "manifest": {"profile_fingerprint": profile, "source_elapsed_seconds": 290.}}
    result = study.add_attribution_execution(bundle, output_dir=tmp_path)
    timing = result["manifest"]["execution_timing"]
    assert timing["measured_worker_span_seconds"] == 180.
    assert timing["unique_checkpoint_wall_seconds"] == 290.
    assert timing["cold_case_indices"] == [0, 1, 2] and timing["overlapping_cold_case_indices"] == []
    (tmp_path/"rough_stopping_attribution_v1/campaign_v1/a/executed_runner.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed attribution runner"):
        study.add_attribution_execution(bundle, output_dir=tmp_path)
