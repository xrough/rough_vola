"""Provenance checks fail closed before numerical results reach the notebook."""
import hashlib
import json

import pytest
import pandas as pd
from types import SimpleNamespace

from rvlab.experiments import _pricing_rough_shared_delta_collection as c


def _inventory(folder):
    entries = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size,
        "sha256": c.sha(p)} for p in sorted(folder.rglob("*")) if p.is_file()]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    (folder / "preservation_inventory.json").write_text(json.dumps({"files": entries,
        "inventory_sha256": digest}))


def test_streaming_fingerprint_and_complete_inventory(tmp_path):
    payload = tmp_path / "samples.npz"
    value = b"z" * (2 * 1024 * 1024 + 3)
    payload.write_bytes(value)
    assert c.sha(payload) == hashlib.sha256(value).hexdigest()
    _inventory(tmp_path)
    inventory, entries = c.preserved(tmp_path)
    assert len(inventory["files"]) == 1 and len(entries) == 2
    (tmp_path / "unrecorded.txt").write_text("omitted outcome")
    with pytest.raises(ValueError, match="omits"):
        c.preserved(tmp_path)


def test_changed_and_duplicate_artifacts_are_rejected(tmp_path):
    path = tmp_path / "outcome.json"
    path.write_text("failure")
    _inventory(tmp_path)
    path.write_text("success")
    with pytest.raises(ValueError, match="Changed"):
        c.preserved(tmp_path)


def test_conflicting_provenance_and_naive_chronology_are_rejected(tmp_path):
    path = tmp_path / "source.py"
    rows = [{"path": str(path), "bytes": 1, "sha256": "x"}]
    assert c._unique_entries(rows + rows) == rows
    with pytest.raises(ValueError, match="Conflicting"):
        c._unique_entries(rows + [{**rows[0], "sha256": "y"}])
    with pytest.raises(ValueError, match="timezones"):
        c._timestamp("2026-10-20T00:00:00")


@pytest.fixture
def design(monkeypatch):
    streams = [{"seed": 123, "paths": 4096, "steps": 128,
        "representations": ["original", "micro"]}]
    monkeypatch.setattr(c, "STREAMS", streams)
    return {"version": "rough_shared_delta_v1", "parent_campaign": c.PARENT,
        "streams": streams, "total_paths": 4096, "frozen_policy": "GH16 blend1",
        "policy_operator_steps": 128, "exercise_intervals": 64, "kernel_factors": 4,
        "common_control_features": 193, "bump_relatives": [.005, .0025],
        "relative_spots": [-.005, -.0025, 0., .0025, .005],
        "physical_random_batch_size": 512, "work_batch_size": 32,
        "calendar_european_function_grid_points": 129,
        "physical_european_function_grid_points_by_steps": {"128": 129, "256": 257},
        "original_and_micro_are_distinct_centered_controls": True,
        "microstep_control_not_a_telescoping_block_bridge": True,
        "independent_primitive_panel": {"absolute_path_indices": [0, 1, 511, 512],
            "calendar_indices": [16, 32, 48, 63], "bump_indices": [0, 2, 4]},
        "no_fresh_selection_or_fitting": True,
        "no_empirical_valuation_recentering": True,
        "no_regression_intercept_at_valuation": True,
        "all_bumps_recompute_continuations_and_stops": True,
        "all_paths_retained": True, "prior_paths_used_for_acceptance": False,
        "price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002, "delta_rtol": .005,
        "confidence_multiplier": 1.96, "spread_fraction": .1,
        "tolerances_changed": False, "optimal_american_delta_ready": False,
        "new_hedging_readiness": False,
        "forbidden_prior_seeds": [1, 2], "created_at": "2026-10-20T00:00:00+00:00"}


def test_frozen_design_accepts_only_the_predeclared_protocol(design):
    c.validate_design(design)


@pytest.mark.parametrize("key,value", [
    ("delta_atol", .003), ("delta_rtol", .01), ("confidence_multiplier", 1.),
    ("all_paths_retained", 1), ("prior_paths_used_for_acceptance", True),
    ("no_empirical_valuation_recentering", False), ("policy_operator_steps", 256),
    ("exercise_intervals", 128), ("bump_relatives", [.01, .005]),
    ("streams", []), ("forbidden_prior_seeds", [123]),
    ("forbidden_prior_seeds", [True, 2]), ("forbidden_prior_seeds", [1, 1]),
])
def test_changed_budgets_tolerances_or_seeds_fail_closed(design, key, value):
    design[key] = value
    with pytest.raises(ValueError):
        c.validate_design(design)


def _gates():
    return {name: pd.DataFrame({"estimator": ["shared"] * n,
        gate: [True] * n}) for name, gate, n in (
        ("delta_sampling", "sampling_pass", 6), ("bump_refinement", "bump_check_pass", 3),
        ("representation_change", "tolerance_bias_resolved", 2),
        ("time_refinement", "time_check_pass", 2),
        ("combined_refinement", "combined_check_pass", 2))}


def test_all_prescribed_checks_are_required_and_failures_remain_visible():
    tables = _gates()
    tables["combined_refinement"].loc[1, "combined_check_pass"] = False
    summary = c.delta_gate_summary(tables)
    assert summary.checks.sum() == 15
    assert summary.passed.sum() == 14
    assert not summary.all_pass.all()
    assert not summary.optimal_delta_ready.any()
    tables["delta_sampling"] = tables["delta_sampling"].iloc[:-1]
    with pytest.raises(ValueError, match="omitted"):
        c.delta_gate_summary(tables)


def test_textual_false_cannot_pass_an_acceptance_check():
    tables = _gates()
    tables["bump_refinement"]["bump_check_pass"] = "False"
    with pytest.raises(ValueError, match="changed"):
        c.delta_gate_summary(tables)


@pytest.fixture
def record(design):
    design["source_sha256"] = {"frozen.py": "digest"}
    return {"index": 0, **design["streams"][0], "design_sha256": "design-digest",
        "source_sha256": design["source_sha256"], "sources_unchanged": True,
        "all_paths_retained": True, "new_fitting_or_selection": False,
        "started_at": "2026-10-20T00:01:00+00:00",
        "completed_at": "2026-10-20T00:02:00+00:00", "elapsed_seconds": 60.}


def test_independent_stream_record_matches_its_frozen_design(design, record):
    c.validate_record(record, design, index=0, design_sha256="design-digest")


@pytest.mark.parametrize("key,value", [
    ("seed", 124), ("paths", 8192), ("steps", 256),
    ("representations", ["micro"]), ("sources_unchanged", 1),
    ("all_paths_retained", False), ("new_fitting_or_selection", True),
    ("design_sha256", "replacement"), ("elapsed_seconds", float("nan")),
    ("started_at", "2026-10-19T00:00:00+00:00"),
    ("completed_at", "2026-10-20T00:00:30+00:00"),
])
def test_partial_reused_refitted_or_misdated_streams_are_rejected(design, record, key, value):
    record[key] = value
    with pytest.raises(ValueError):
        c.validate_record(record, design, index=0, design_sha256="design-digest")


def test_precise_finite_policy_checks_do_not_certify_raw_bias_or_broader_greeks(monkeypatch, tmp_path):
    tables = _gates()
    tables["raw_control"] = pd.DataFrame({"comparison": ["raw minus shared"] * 6 +
        ["stable minus shared"] * 6, "tolerance_bias_resolved": [False] * 6 + [True] * 6,
        "zero_in_interval": [True] * 12})
    monkeypatch.setattr(c, "collect_shared_delta_campaign", lambda _: {
        "tables": tables, "input_files": [], "manifest": {}})
    result = c.shared_delta_study(SimpleNamespace(output_dir=tmp_path, progress=lambda _: None),
        compute_figures=False)
    flags = result["manifest"]
    assert flags["fixed_policy_delta_checks_pass"]
    assert not flags["raw_control_tolerance_bias_resolved"]
    assert flags["stable_control_tolerance_bias_resolved"]
    assert flags["all_raw_control_marginal_intervals_include_zero"]
    assert not any(flags[k] for k in ("american_pricing_ready", "stock_delta_ready", "optimal_delta_ready"))


def test_aggregate_seal_must_follow_every_stream_completion(design, record):
    c.validate_completion_timing(design, {"completed_at": "2026-10-20T00:03:00+00:00"}, [record])
    with pytest.raises(ValueError, match="predates"):
        c.validate_completion_timing(design, {"completed_at": "2026-10-20T00:01:30+00:00"}, [record])
    with pytest.raises(ValueError, match="timezones"):
        c.validate_completion_timing(design, {"completed_at": "2026-10-20T00:03:00"}, [record])
