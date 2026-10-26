"""Presentation consumes every sealed repeat without selecting outcomes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest


def test_centering_figures_preserve_sealed_tables_and_keep_trigger_separate():
    from rvlab.experiments._pricing_rough_centering_reporting import add_centering_figures
    repeats = pd.DataFrame([{"quantity": "contrast", "seed": seed,
        "estimate": estimate, "standard_error": .01}
        for seed, estimate in ((410005042, .02), (411005042, -.01))])
    pooled = pd.DataFrame([{"quantity": "contrast", "estimate": .005,
        "standard_error": .007}])
    trigger = pd.DataFrame([{"estimate": .02, "standard_error": .0084}])
    conditional = pd.DataFrame([{"state_id": state, "component": component,
        "conditional_increment_mean": residual, "quadrature_error_estimate": 1e-12}
        for state in (0, 1) for component, residual in (("european", -1e-13), ("premium", 0.))])
    tables = {"per_seed": repeats, "pooled": pooled,
        "exploratory_trigger": trigger, "conditional_components": conditional}
    originals = {name: frame.copy(deep=True) for name, frame in tables.items()}
    result = add_centering_figures({"tables": tables, "figures": {}, "manifest": {}})
    assert set(result["figures"]) == {"independent_repeats", "conditional_residuals"}
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, originals[name])
    figure = result["figures"]["independent_repeats"]
    assert [t.get_text() for t in figure.axes[0].get_yticklabels()] == ["410,005,042", "411,005,042"]
    assert "excluded from pool" in figure.axes[1].get_yticklabels()[1].get_text()
    for figure in result["figures"].values():
        plt.close(figure)


def test_failed_centering_study_keeps_empty_diagnostics_without_fabricating_figures():
    from rvlab.experiments._pricing_rough_centering_reporting import add_centering_figures
    source = {"tables": {"per_seed": pd.DataFrame(), "pooled": pd.DataFrame()},
        "figures": {}, "manifest": {"status": "failed"}}
    result = add_centering_figures(source)
    assert result["figures"] == {}
    assert result["manifest"]["status"] == "failed"


def test_centering_execution_separates_worker_span_from_summed_wall_times(tmp_path):
    from rvlab.experiments._pricing_rough_centering_reporting import add_centering_execution
    import hashlib
    import json
    for name, mode, start, end, elapsed in (
            ("worker_a", "worker", "2026-10-18T11:00:00+00:00", "2026-10-18T11:02:00+00:00", 120.),
            ("worker_b", "worker", "2026-10-18T11:00:30+00:00", "2026-10-18T11:02:30+00:00", 120.),
            ("prepare", "prepare", "2026-10-18T10:59:00+00:00", "2026-10-18T10:59:10+00:00", 10.)):
        folder = tmp_path/"rough_control_centering_v1/campaign_v1"/name
        folder.mkdir(parents=True)
        source = b"# frozen runner\n"
        (folder/"executed_runner.py").write_bytes(source)
        (folder/"manifest.json").write_text(json.dumps({"profile_fingerprint": "fixed",
            "runner_sha256": hashlib.sha256(source).hexdigest(), "mode": mode,
            "started_at": start, "completed_at": end, "elapsed_seconds": elapsed}))
    checks = [{"policy": policy, "quantity": "raw", "bit_identical": True,
        "max_absolute_change": 0.} for policy in ("original", "selected")]
    source = {"tables": {"reconstruction": pd.DataFrame([checks])},
        "figures": {}, "manifest": {"profile_fingerprint": "fixed"}}
    result = add_centering_execution(source, output_dir=tmp_path)
    assert source["manifest"] == {"profile_fingerprint": "fixed"}
    assert result["manifest"]["campaign_worker_span_seconds"] == 150.
    assert result["manifest"]["worker_wall_seconds_sum"] == 240.
    assert result["manifest"]["worker_manifest_count"] == 2
    assert len(result["tables"]["execution_stages"]) == 3
    assert len(result["tables"]["reconstruction"]) == 2
    assert len(source["tables"]["reconstruction"]) == 1
    (tmp_path/"rough_control_centering_v1/campaign_v1/worker_b/executed_runner.py").write_text("changed")
    with pytest.raises(ValueError, match="Changed executed centering runner"):
        add_centering_execution(source, output_dir=tmp_path)


def test_late_centering_supplement_is_verified_and_cannot_enter_primary_pool(tmp_path, monkeypatch):
    import hashlib
    import json
    from rvlab.experiments import _pricing_rough_centering_reporting as reporting

    profile = "0123456789abcdef"+"0"*48
    directory = tmp_path/"rough_control_centering_v1"/profile[:16]/"supplemental_disagreement_states_v1"
    directory.mkdir(parents=True)
    trace = tmp_path/"existing_independent_trace.npz"
    trace.write_bytes(b"existing source trace")
    seal = {"primary_profile": profile[:16], "selection_uses_conditional_residuals": False,
        "selection_uses_frozen_decisions_only": True, "policy_or_control_selection": False,
        "selection_rule": "sealed disagreement-state rule", "source_trace": {
            "path": str(trace), "bytes": trace.stat().st_size,
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}}
    seal_path = directory/"selection_before_conditional_residuals.json"
    seal_path.write_text(json.dumps(seal))
    checked = {"status": "completed", "late_exploratory": True,
        "primary_design_input": False, "primary_pool_input": False,
        "new_paths_generated": False, "tolerances_changed": False,
        "elapsed_seconds": 1., "selection_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
        "tables": {name: [{"state_id": 0}] for name in ("states", "components", "branches")}}
    result_path = directory/"result.json"

    def reseal():
        result_path.write_text(json.dumps(checked))
        files = [{"path": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (result_path, seal_path)]
        digest = hashlib.sha256(json.dumps(files, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        (directory/"preservation_inventory.json").write_text(json.dumps({
            "files": files, "inventory_sha256": digest}))
        monkeypatch.setattr(reporting, "SUPPLEMENT_INVENTORY_SHA256", digest)

    reseal()
    pool = pd.DataFrame([{"quantity": "contrast", "estimate": 0.}])
    source = {"tables": {"pooled": pool}, "figures": {},
        "manifest": {"profile_fingerprint": profile}}
    result = reporting.add_centering_supplement(source, output_dir=tmp_path)
    assert result["tables"]["pooled"] is pool
    assert set(source["tables"]) == {"pooled"}
    assert result["manifest"]["supplementary_disagreement_audit"]["primary_pool_input"] is False
    assert result["tables"]["supplementary_conditional_components"].acceptance_input.tolist() == [False]
    trace.write_bytes(b"changed external source")
    with pytest.raises(ValueError, match="Changed existing independent trace"):
        reporting.add_centering_supplement(source, output_dir=tmp_path)
    trace.write_bytes(b"existing source trace")
    checked["primary_pool_input"] = True
    reseal()
    with pytest.raises(ValueError, match="exploratory scope"):
        reporting.add_centering_supplement(source, output_dir=tmp_path)
