"""Precision reporting preserves primary estimates and excluded pilot scopes."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_centering_precision_reporting as reporting


def _result():
    seeds = (510005042, 511005042)
    pooled = pd.DataFrame([{"quantity": quantity, "estimate": estimate,
        "standard_error": se} for quantity, estimate, se in
        (("raw_contrast", .003, .006), ("projected_fine", 2e-13, 2e-14))])
    per_seed = pd.DataFrame([{"quantity": "projected_fine", "seed": seed,
        "estimate": value, "standard_error": 3e-14} for seed, value in zip(seeds, (1e-13, 3e-13))])
    common = {"price_allowance": .001286, "sampling_half_width": 4e-14,
        "absolute_refinement_change": 1e-12, "quadrature_error": 1e-11,
        "tail_error": 1e-30, "arithmetic_error": 1e-10,
        "uncertainty_endpoint_low": -1.1e-10, "uncertainty_endpoint_high": 1.1e-10}
    budget = pd.DataFrame([{"scope": "stream", "seed": seed, **common} for seed in seeds] +
        [{"scope": "pooled", "seed": None, **common}])
    return {"tables": {"pooled": pooled, "per_seed": per_seed, "numerical_budget": budget,
        "previous_comparison": pd.DataFrame([{"estimate": -.001, "standard_error": .003}])},
        "figures": {}, "manifest": {"primary_complete": True}}


def test_precision_figures_keep_all_seeds_and_separate_historical_raw_pool(tmp_path):
    source = _result()
    originals = {name: frame.copy(deep=True) for name, frame in source["tables"].items()}
    result = reporting.add_precision_reporting(source, output_dir=tmp_path)
    assert set(result["figures"]) == {"tolerance_scale_comparison", "integration_and_uncertainty"}
    for name, frame in originals.items():
        pd.testing.assert_frame_equal(source["tables"][name], frame)
    assert source["manifest"] == {"primary_complete": True}
    axes = result["figures"]["tolerance_scale_comparison"].axes
    assert axes[0].get_yticklabels()[0].get_text() == "Historical raw pool\nkept separate"
    assert [x.get_text() for x in axes[1].get_yticklabels()] == ["510,005,042", "511,005,042", "Fresh-only conditional pool"]
    for figure in result["figures"].values():
        plt.close(figure)


def test_failed_precision_does_not_invent_estimates(tmp_path):
    result = reporting.add_precision_reporting({"tables": {"pooled": pd.DataFrame(),
        "numerical_budget": pd.DataFrame()}, "figures": {}, "manifest": {"status": "failed"}}, output_dir=tmp_path)
    assert result["figures"] == {} and result["manifest"]["status"] == "failed"


def test_budget_pilot_is_verified_and_excluded_from_primary(tmp_path, monkeypatch):
    directory = tmp_path / "rough_centering_precision_v1/pilots/quadrature_budget"
    directory.mkdir(parents=True)
    pd.DataFrame([{"value": 17}]).to_csv(directory/"budget_trials.csv", index=False)
    pd.DataFrame([{"value": 18}]).to_csv(directory/"tail_robustness.csv", index=False)
    files = {}
    for path in directory.glob("*.csv"):
        raw = path.read_bytes(); files[path.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {"formal_new_precision_campaign_input": False, "thresholds_changed": False,
        "no_new_paths": True, "trace_seed": 430005042, "files": files}
    path = directory / "manifest.json"; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reporting, "_BUDGET_MANIFEST_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    tables = reporting._pilot_tables(tmp_path)
    assert tables["pilot_budget_trials"].value.tolist() == [17]
    assert not tables["excluded_trials"].primary_pool_input.any()
    assert not tables["excluded_trials"].acceptance_input.any()
    (directory/"budget_trials.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed precision-study"):
        reporting._pilot_tables(tmp_path)


def test_smoke_is_verified_and_remains_outside_primary_pool(tmp_path, monkeypatch):
    directory = tmp_path / "rough_centering_precision_v1/pilots/microstate_verification"
    directory.mkdir(parents=True)
    (directory/"manifest.json").write_text(json.dumps({"seed": 890105042,
        "excluded_from_primary_inference": True, "no_new_exercise_dates": True}))
    pd.DataFrame([{"gate": 1}]).to_csv(directory/"gated_states.csv", index=False)
    files = {path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in directory.iterdir()}
    monkeypatch.setattr(reporting, "_SMOKE_FILES", files)
    tables = reporting._pilot_tables(tmp_path)
    assert tables["pilot_microstate_states"].gate.tolist() == [1]
    assert not tables["excluded_trials"].primary_pool_input.any()
    (directory/"manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="Changed precision-study"):
        reporting._pilot_tables(tmp_path)


def test_pilot_file_paths_cannot_escape_directory(tmp_path, monkeypatch):
    directory = tmp_path / "rough_centering_precision_v1/pilots/quadrature_budget"
    directory.mkdir(parents=True)
    manifest = {"formal_new_precision_campaign_input": False, "thresholds_changed": False,
        "no_new_paths": True, "trace_seed": 1, "files": {"../escape": {"bytes": 1, "sha256": "x"}}}
    path = directory/"manifest.json";path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reporting, "_BUDGET_MANIFEST_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="escapes"):
        reporting._pilot_tables(tmp_path)


def test_worker_timing_keeps_profile_identity_and_concurrent_costs_separate(tmp_path):
    directory = tmp_path / "rough_centering_precision_v1/campaign_v1"
    runner=b"# executed runner\n"
    for name, profile, start, end, seconds in (
            ("worker_a","fixed","2026-10-18T12:00:00+00:00","2026-10-18T12:02:00+00:00",120.),
            ("worker_b","fixed","2026-10-18T12:00:30+00:00","2026-10-18T12:02:30+00:00",120.),
            ("superseded_prepare","old","2026-10-18T11:00:00+00:00","2026-10-18T11:01:00+00:00",60.)):
        path=directory/name;path.mkdir(parents=True)
        (path/"executed_runner.py").write_bytes(runner)
        (path/"manifest.json").write_text(json.dumps({"profile_fingerprint":profile,"mode":"worker",
            "runner_sha256":hashlib.sha256(runner).hexdigest(),"started_at":start,"completed_at":end,
            "elapsed_seconds":seconds,"checkpoint_replay":False}))
    tables, timing=reporting._execution_tables(tmp_path,"fixed")
    assert len(tables["execution_stages"])==2
    assert timing["campaign_worker_span_seconds"]==150.
    assert timing["worker_wall_seconds_sum"]==240.
    assert timing["worker_manifest_count"]==2
    (directory/"worker_a/executed_runner.py").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Changed precision-study"):
        reporting._execution_tables(tmp_path,"fixed")
