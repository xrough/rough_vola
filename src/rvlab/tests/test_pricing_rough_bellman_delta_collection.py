"""Incomplete or modified evidence cannot become a current-study result."""
from types import SimpleNamespace
from pathlib import Path
import hashlib
import sys

import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_bellman_delta_collection as collection


@pytest.mark.parametrize("options,error", [
    ({"seed": True}, ValueError), ({"seed": 43}, ValueError),
    ({"compute_figures": 1}, TypeError), ({"unknown_numerical_override": 1}, TypeError),
])
def test_invalid_study_options_fail_before_accessing_a_context(options, error):
    with pytest.raises(error):
        collection.bellman_delta_study(None, **options)


def test_missing_campaign_raises_instead_of_substituting_spent_or_synthetic_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="Complete the independent experiment"):
        collection.collect_delta_campaign(tmp_path/"validation"/"missing")


@pytest.mark.parametrize("mutation", ["omitted", "snapshot", "current", "extra"])
def test_source_dependency_or_snapshot_changes_are_rejected(tmp_path, monkeypatch, mutation):
    root, folder = tmp_path/"repo", tmp_path/"campaign"
    source = root/"src/model.py"
    saved = folder/"source/src/model.py"
    source.parent.mkdir(parents=True)
    saved.parent.mkdir(parents=True)
    source.write_text("frozen = 1\n")
    saved.write_bytes(source.read_bytes())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    declaration = {"source_sha256": {"src/model.py": sha}}
    monkeypatch.setattr(collection, "ROOT", root)
    if mutation == "omitted": declaration["source_sha256"].clear()
    if mutation == "snapshot": saved.write_text("frozen = 2\n")
    if mutation == "current": source.write_text("frozen = 2\n")
    if mutation == "extra": declaration["source_sha256"]["src/other.py"] = sha
    with pytest.raises(ValueError, match="source|dependency"):
        collection._verify_sources(folder, declaration, {"src/model.py"})


def test_a_complete_current_and_saved_source_identity_passes(tmp_path, monkeypatch):
    root, folder = tmp_path/"repo", tmp_path/"campaign"
    for prefix in (root, folder/"source"):
        (prefix/"src").mkdir(parents=True)
        (prefix/"src/model.py").write_text("frozen = 1\n")
    monkeypatch.setattr(collection, "ROOT", root)
    sha = hashlib.sha256((root/"src/model.py").read_bytes()).hexdigest()
    collection._verify_sources(folder, {"source_sha256": {"src/model.py": sha}}, {"src/model.py"})


def test_composite_evidence_retains_separate_estimator_scopes_and_readiness(monkeypatch, tmp_path):
    row = {"path": str(tmp_path/"evidence.npz"), "bytes": 12, "sha256": "pinned"}
    delta = {"tables": {}, "input_files": [row], "manifest": {"complete": True}}
    gap = {"tables": {"gap_trials": pd.DataFrame([{"status": "completed"}]*4)},
        "input_files": [row], "manifest": {"complete": True, "completed_final_paths": 8192}}
    monkeypatch.setattr(collection, "collect_delta_campaign", lambda *a: delta)
    monkeypatch.setitem(sys.modules, "rvlab.experiments._pricing_rough_bellman_control_collection",
        SimpleNamespace(collect_control_campaign=lambda *a: gap))
    result = collection.bellman_delta_study(SimpleNamespace(output_dir=tmp_path, progress=lambda _: None),
        compute_figures=False)
    manifest = result["manifest"]
    assert len(manifest["input_files"]) == 1
    assert manifest["primary_policy_fits"] == 0
    assert manifest["primary_control_optimizer_calls"] == 4
    assert not manifest["new_control_stock_delta_validated"]
    assert not manifest["stock_delta_ready"]
    assert not manifest["american_pricing_ready"]
    assert manifest["delta_campaign"] is delta["manifest"]
    assert manifest["gap_campaign"] is gap["manifest"]
    assert len(manifest["analysis_sources"]) == 6


def test_composite_rejects_conflicting_fingerprints_for_the_same_input(monkeypatch, tmp_path):
    row = {"path": str(tmp_path/"evidence.npz"), "bytes": 12, "sha256": "first"}
    delta = {"tables": {}, "input_files": [row], "manifest": {"complete": True}}
    gap = {"tables": {"gap_trials": pd.DataFrame()},
        "input_files": [{**row, "sha256": "changed"}], "manifest": {"complete": True}}
    monkeypatch.setattr(collection, "collect_delta_campaign", lambda *a: delta)
    monkeypatch.setitem(sys.modules, "rvlab.experiments._pricing_rough_bellman_control_collection",
        SimpleNamespace(collect_control_campaign=lambda *a: gap))
    with pytest.raises(ValueError, match="Conflicting"):
        collection.bellman_delta_study(SimpleNamespace(output_dir=tmp_path, progress=lambda _: None),
            compute_figures=False)
