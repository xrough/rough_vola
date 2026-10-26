"""Follow-up trials preserve the failed baseline and independent stream design."""
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_american_accuracy as accuracy
from rvlab.experiments import _pricing_rough_american as baseline


def _design():
    return [{"trial": "declared central", "contract_id": "rough_put_60_100",
        "config": {"training_paths": 4, "valuation_paths": 4, "steps": 4,
            "exercise_steps": 2, "batch_size": 4}},
        {"trial": "declared independent", "contract_id": "rough_put_60_100",
        "config": {"training_paths": 4, "valuation_paths": 4, "steps": 4,
            "exercise_steps": 2, "batch_size": 4}}]


def test_accuracy_design_freezes_contracts_configs_and_disjoint_streams():
    rows = accuracy._validated_design(_design(), 42)
    assert rows[0][0]["strike"] == 100.
    assert rows[0][4].training_seed == 42
    assert rows[1][4].training_seed == 1042
    assert not {row[4].training_seed for row in rows} & {row[4].valuation_seed for row in rows}
    assert rows[0][4].training_paths == 4


@pytest.mark.parametrize("specification", [None, [], [{"trial": "a", "contract_id": "SPY"}],
    [{"trial": "a", "contract_id": "rough_put_60_100", "config": {"training_seed": 2}}],
    [{"trial": "a", "contract_id": "rough_put_60_100", "factors": 1}],
    [{"trial": "a", "contract_id": "rough_put_60_100", "extra": True}],
    [{"trial": "a", "contract_id": "heston_limit_put", "factors": 4}],
])
def test_accuracy_design_rejects_ambiguous_or_unavailable_inputs(specification):
    with pytest.raises((ValueError, TypeError)):
        accuracy._validated_design(specification, 42)


def _mock_campaign(monkeypatch, tmp_path, *, failed=False):
    from rvlab import pricing
    # The mocked engine has a frozen identity independent of concurrent source
    # edits by the developer; source-change guards are exercised separately.
    frozen_sources = accuracy._core_source_hashes()
    monkeypatch.setattr(accuracy, "_core_source_hashes", lambda: frozen_sources)
    monkeypatch.setattr(accuracy, "_preserve_sources", lambda *args: [])
    monkeypatch.setattr(pricing, "approximate_fractional_kernel",
        lambda h, t, n, **kw: pricing.ExponentialKernel((1.,), (1.,), hurst=h, horizon=t))
    historical = {"tables": {"prices": pd.DataFrame([{"contract_id": "rough_put_60_100",
        "stage": "initial", "price": 1., "paired_gap": .5, "price_standard_error": .1,
        "delta_standard_error": .1}])}, "manifest": {"historical_archive_manifest_sha256": "baseline"}}
    monkeypatch.setattr(accuracy, "load_historical_american", lambda **kw: historical)
    calls = []
    def run(params, market, kernel, contract, config):
        calls.append((config.training_seed, config.valuation_seed))
        if failed and len(calls) == 2:
            raise pricing.PricingError("Retained independent trial failure")
        return {"status": "completed", "prices": [1.], "primal_standard_errors": [.00001],
            "dual_prices": [1.0001], "dual_standard_errors": [.00002],
            "paired_gaps": [.0001], "paired_gap_standard_errors": [.00001],
            "deltas": [-.5], "delta_standard_errors": [.0001],
            "raw_prices": [1.1], "raw_standard_errors": [.1], "exercise_probabilities": [.2],
            "bump_deltas": [-.5, -.50001], "bump_delta_standard_errors": [.0001, .0002],
            "one_sided_deltas": [-.51, -.49], "one_sided_delta_standard_errors": [.001, .001],
            "retrained_deltas": [None], "retrained_delta_standard_errors": [None],
            "settings": asdict(config), "diagnostics": {"independent_training_valuation": True}}
    monkeypatch.setattr(accuracy, "_stopping_trial", run)
    context = SimpleNamespace(output_dir=tmp_path, progress=lambda _: None)
    result = accuracy.rough_american_accuracy_study(context, trial_specs=_design(), compute_figures=False)
    return result, calls, context


def test_follow_up_keeps_trials_caches_and_unchanged_acceptance(monkeypatch, tmp_path):
    result, calls, context = _mock_campaign(monkeypatch, tmp_path)
    second = accuracy.rough_american_accuracy_study(context, trial_specs=_design(), compute_figures=False)
    assert len(calls) == 2
    assert second["manifest"]["checkpoint_replay"]
    assert result["tables"]["summary"].iloc[0].discrete_price_passes == 2
    assert result["manifest"]["acceptance"]["price_atol"] == .001
    assert result["manifest"]["acceptance"]["delta_atol"] == .002
    assert not result["manifest"]["american_pricing_ready"]
    assert not result["manifest"]["stock_delta_ready"]
    assert result["tables"]["comparison"].gap_reduction_factor.min() == pytest.approx(5000.)


def test_follow_up_preserves_failed_attempt_and_partial_status(monkeypatch, tmp_path):
    result, calls, context = _mock_campaign(monkeypatch, tmp_path, failed=True)
    assert len(calls) == 2 and len(result["tables"]["prices"]) == 1
    assert len(result["tables"]["trials"]) == 2
    assert result["tables"]["failures"].error.str.contains("Retained").all()
    partial = accuracy.rough_american_accuracy_study(context, trial_specs=_design(),
        trial_indices=(0,), compute_figures=False)
    assert partial["manifest"]["partial_campaign"]
    assert partial["manifest"]["profile_fingerprint"] == result["manifest"]["profile_fingerprint"]


def test_verified_american_baseline_is_not_relabelled_european(tmp_path):
    from rvlab.tests.test_pricing_rough_american_study import _archive
    import hashlib
    import json
    archive = _archive(tmp_path)
    manifest_path = archive/"results/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rough_american"] = {"backend": "rough_heston_lifted_primal_dual",
        "source_elapsed_seconds": 125., "profile_fingerprint": "failed-baseline"}
    manifest_path.write_text(json.dumps(manifest))
    inventory_path = archive/"preservation_manifest.json"
    inventory = json.loads(inventory_path.read_text())
    row = next(row for row in inventory["files"] if row["preserved"] == "results/manifest.json")
    row.update(bytes=manifest_path.stat().st_size, sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    inventory_path.write_text(json.dumps(inventory))
    result = baseline.load_historical_american(archive=archive, compute_figures=False)
    assert result["manifest"]["backend"] == "rough_heston_lifted_primal_dual"
    assert result["manifest"]["profile_fingerprint"] == "failed-baseline"
    assert result["manifest"]["historical_replay"] and not result["manifest"]["current_engine_execution"]
    assert result["manifest"]["source_elapsed_seconds"] == 125.


@pytest.mark.parametrize("options,error", [
    ({"run_rough_american_accuracy": 1}, ValueError),
    ({"rough_american_accuracy_kwargs": []}, TypeError),
    ({"rough_american_accuracy_kwargs": {"seed": 12}}, ValueError),
    ({"run_rough_american_accuracy": True, "rough_american_kwargs": {"steps": 4}}, ValueError),
])
def test_accuracy_workflow_rejects_ambiguous_execution_options(options, error):
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):
        pricing_research(SimpleNamespace(), **options)


def test_accuracy_workflow_keeps_original_american_source_identity(monkeypatch):
    from rvlab.experiments import _pricing_research as research
    from rvlab.pricing import PricingConfig, FDConfig
    import matplotlib.pyplot as plt
    baseline_bundle = {"tables": {"prices": pd.DataFrame()}, "figures": {},
        "manifest": {"backend": "rough_heston_lifted_primal_dual", "historical_replay": True}}
    current_bundle = {"tables": {"prices": pd.DataFrame()}, "figures": {},
        "manifest": {"backend": "rough_heston_lifted_primal_dual", "current_engine_execution": True}}
    monkeypatch.setattr(baseline, "rough_american_study", lambda *a, **k: pytest.fail("Must not rerun unsuccessful baseline"))
    monkeypatch.setattr(baseline, "load_historical_american", lambda *a, **k: baseline_bundle)
    calls = []
    monkeypatch.setattr(accuracy, "rough_american_accuracy_study", lambda context, **options: calls.append(options) or current_bundle)
    try:
        result = research.pricing_research(SimpleNamespace(progress=lambda _: None),
            fourier_config=PricingConfig(32, 40., 64), fd_config=FDConfig(24, 8, 20),
            run_rough_american_accuracy=True, rough_reuse_checkpoints=False,
            rough_american_accuracy_kwargs={"trial_specs": _design()})
        manifest = result["manifest"]
        assert manifest["rough_american_included"] and not manifest["rough_american_executed"]
        assert manifest["rough_american_accuracy_executed"]
        assert manifest["rough_american"] is baseline_bundle["manifest"]
        assert manifest["rough_american_accuracy"] is current_bundle["manifest"]
        assert calls == [{"seed": 42, "reuse_checkpoints": False, "trial_specs": _design()}]
    finally:
        plt.close("all")


def test_exploratory_probes_keep_their_identity_and_verify_undisplayed_source(tmp_path):
    import hashlib
    import json
    folder = tmp_path/"probe"
    folder.mkdir()
    (folder/"trial.csv").write_text("mode,price,se,dual_gap\nfailed_basis,2.8,.02,.1\n")
    (folder/"trial.py").write_text("# exploratory source\n")
    rows = [{"preserved": path.name, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in folder.iterdir()]
    (folder/"preservation_manifest.json").write_text(json.dumps({"files": rows}))
    table, records = accuracy.load_accuracy_probes([folder])
    assert table.iloc[0]["mode"] == "failed_basis"
    assert table.iloc[0].historical_replay
    assert records[0]["files"] == 2
    (folder/"trial.py").write_text("# changed exploratory source\n")
    with pytest.raises(ValueError, match="Changed preserved"):
        accuracy.load_accuracy_probes([folder])


def test_all_failed_follow_up_displays_an_audit_without_fabricated_prices():
    from pathlib import Path
    import nbformat
    from rvlab.experiments.pricing_presentation import _table_html
    from rvlab.experiments._pricing_rough_stopping_attribution import ATTRIBUTION_ARCHIVE
    notebook = nbformat.read(ATTRIBUTION_ARCHIVE/"01_pricing_and_exercise.ipynb", as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and
        cell.source.startswith('if results["manifest"].get("rough_american_accuracy_executed"')]
    assert len(cells) == 3
    shown = []
    namespace = {"pd": pd, "Markdown": str, "display": shown.append,
        "results": {"manifest": {"rough_american_accuracy_executed": True,
            "rough_american_accuracy": {"source_elapsed_seconds": 1.}}, "figures": {}},
        "tables": {"rough_american_accuracy_prices": pd.DataFrame(),
            "rough_american_accuracy_summary": pd.DataFrame([{"completed_trials": 0,
                "planned_trials": 2, "discrete_price_passes": 0,
                "delta_sampling_passes": 0, "failures": 2}]),
            "rough_american_accuracy_failures": pd.DataFrame([{"error": "retained"}])},
        "present_table": lambda frame, **kw: shown.append(_table_html(frame, **kw)),
        "present_figure": lambda *a, **k: pytest.fail("Failed trials cannot have a price figure")}
    for cell in cells:
        exec(cell, namespace)
    text = " ".join(map(str, shown))
    assert "No follow-up trial produced a usable estimate" in text
    assert "0/2 trial-contract rows complete" in text
    assert "2 numerical/unsupported failures" in text
    assert "readiness remain **false**" in text


def test_refinement_diagnostics_keep_independent_and_shared_path_covariance_distinct():
    import numpy as np
    from rvlab.experiments._pricing_rough_american_accuracy_diagnostics import add_accuracy_refinements
    base = {"trial": "rough base", "contract_id": "rough_put_60_100", "price": 2.,
        "delta": -.4, "price_standard_error": .0001, "delta_standard_error": .0002,
        "discrete_price_gate_passed": False, "retrained_deltas": np.nan,
        "training_method": "fourier_premium", "surrogate_interpolation": "value_bridge"}
    fine = {**base, "trial": "joint and retrained bumps", "price": 2.0001,
        "delta": -.4001, "retrained_deltas": -.4002,
        "retrained_delta_standard_errors": .0003}
    result = {"tables": {"prices": pd.DataFrame([base, fine])}, "manifest": {}}
    result = add_accuracy_refinements(result)
    assert result["tables"]["prices"].effective_surrogate_interpolation.str.contains("N/A").all()
    row = result["tables"]["refinement"].iloc[0]
    assert row.price_change_ci_half_width == pytest.approx(1.96*np.hypot(.0001, .0001))
    assert not row.paired_between_trials and not row.both_discrete_price_gates_passed
    retraining = result["tables"]["retraining"].iloc[0]
    assert retraining.change_ci_half_width_upper_bound == pytest.approx(1.96*(.0002+.0003))
    assert "Cauchy" in retraining.covariance_treatment
    assert not retraining.stock_delta_ready


def test_later_display_probe_does_not_invalidate_frozen_price_profile(tmp_path):
    import hashlib
    import json
    from rvlab.experiments._pricing_rough_american_accuracy_diagnostics import add_accuracy_extras
    folder = tmp_path/"later_probe"
    folder.mkdir()
    payload = b"mode,price\nlater_nodal,2.85\n"
    (folder/"results.csv").write_bytes(payload)
    (folder/"preservation_manifest.json").write_text(json.dumps({"files": [{
        "preserved": "results.csv", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}))
    result = {"tables": {}, "manifest": {"profile_fingerprint": "frozen-pricing",
        "preserved_probes": [{"inventory_sha256": "original-four-inputs"}]}}
    add_accuracy_extras(result, archives=(folder,))
    assert result["manifest"]["profile_fingerprint"] == "frozen-pricing"
    assert result["manifest"]["preserved_probes"] == [{"inventory_sha256": "original-four-inputs"}]
    assert len(result["manifest"]["experimental_extras"]["archives"]) == 1
    assert result["tables"]["additional_probes"].iloc[0]["mode"] == "later_nodal"


def test_execution_timing_verifies_saved_runners_and_keeps_serial_cost_unmeasured(tmp_path):
    import hashlib
    import json
    from rvlab.experiments._pricing_rough_american_accuracy_diagnostics import add_accuracy_execution
    for index, letter in enumerate("abcdefg"):
        folder = tmp_path/"rough_american_accuracy_v1/campaign_v1"/f"worker_{letter}"
        folder.mkdir(parents=True)
        runner = f"# executed worker {letter}\n".encode()
        (folder/"executed_runner.py").write_bytes(runner)
        (folder/"manifest.json").write_text(json.dumps({"profile_fingerprint": "frozen",
            "runner_sha256": hashlib.sha256(runner).hexdigest(), "mode": "worker",
            "started_at": f"2026-10-18T00:00:{index:02d}+00:00",
            "completed_at": f"2026-10-18T00:00:{index+10:02d}+00:00",
            "elapsed_seconds": 10., "checkpoint_replay": False, "selected_trial_indices": [index]}))
    result = {"tables": {}, "manifest": {"study_version": "rough_american_accuracy_v1",
        "profile_fingerprint": "frozen", "source_elapsed_seconds": 17.}}
    add_accuracy_execution(result, output_dir=tmp_path)
    timing = result["manifest"]["execution_timing"]
    assert timing["all_seven_cold_workers_recorded"]
    assert timing["cold_parallel_wall_seconds_excluding_reference_audit"] == 16.
    assert timing["unique_checkpoint_wall_seconds"] == 17.
    assert timing["cold_serial_wall_seconds"] is None
    (tmp_path/"rough_american_accuracy_v1/campaign_v1/worker_a/executed_runner.py").write_text("# changed")
    with pytest.raises(ValueError, match="Changed executed"):
        add_accuracy_execution(result, output_dir=tmp_path)
