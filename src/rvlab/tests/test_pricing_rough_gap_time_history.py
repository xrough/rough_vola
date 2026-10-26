"""The preceding selected-policy validation campaign remains verified historical evidence."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_gap_time_history import (
    POLICY_VALIDATION_PROFILE, load_historical_policy_validation)


def _archive(tmp_path, **overrides):
    directory = tmp_path / "archive"
    results = directory / "results"
    results.mkdir(parents=True)
    table = pd.DataFrame([{"quantity": "projected_fine", "estimate": 1.4e-14}])
    table.to_csv(results / "rough_policy_validation_pooled.csv", index=False)
    (results / "rough_policy_validation_test.png").write_bytes(b"recorded pixels")
    stage = {"backend": "rough_heston_policy_validation", "profile_fingerprint": POLICY_VALIDATION_PROFILE,
        "primary_complete": True, "original_elapsed_seconds": 123.4,
        "input_files": [{"path": f"frozen input {i}"} for i in range(119)],
        "source_sha256": {"frozen_source": "a" * 64}, **overrides}
    manifest = {"notebook": "01_pricing_and_exercise", "evidence_version": "pricing_research_v1",
        "created_at": "2026-10-18T12:16:00Z", "rough_policy_validation": stage,
        "tables": {"rough_policy_validation_pooled": 1},
        "figures": ["rough_policy_validation_test"]}
    (results / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for path in results.iterdir():
        data = path.read_bytes()
        rows.append({"preserved": str(path.relative_to(directory)),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (directory / "preservation_manifest.json").write_text(json.dumps({"files": rows}))
    return directory, stage


def test_historical_stopping_delta_preserves_sources_costs_and_frozen_pixels(tmp_path):
    archive, original = _archive(tmp_path)
    messages = []
    result = load_historical_policy_validation(SimpleNamespace(progress=messages.append), archive=archive)
    assert result["manifest"]["historical_replay"]
    assert not result["manifest"]["current_engine_execution"]
    assert not result["manifest"]["american_pricing_ready"]
    assert not result["manifest"]["stock_delta_ready"]
    assert result["manifest"]["source_sha256"] == original["source_sha256"]
    assert result["manifest"]["original_elapsed_seconds"] == 123.4
    assert result["tables"]["pooled"].estimate.item() == 1.4e-14
    assert result["figures"]["test"].read_bytes() == b"recorded pixels"
    assert len(messages) == 1
    assert not load_historical_policy_validation(archive=archive, compute_figures=False)["figures"]


@pytest.mark.parametrize("changes", [{"profile_fingerprint": "wrong"},
    {"primary_complete": False}, {"backend": "unverified backend"},
    {"input_files": []}, {"input_files": [{"path": "duplicate"}] * 119}])
def test_wrong_or_partial_campaign_cannot_be_relabelled_complete(tmp_path, changes):
    archive, _ = _archive(tmp_path, **changes)
    with pytest.raises(ValueError, match="declared complete"):
        load_historical_policy_validation(archive=archive)


def test_modified_preserved_bytes_are_rejected(tmp_path):
    archive, _ = _archive(tmp_path)
    (archive / "results/rough_policy_validation_pooled.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="Changed preserved"):
        load_historical_policy_validation(archive=archive)


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_nonboolean_figure_option_is_rejected_before_archive_access(value):
    with pytest.raises(TypeError, match="boolean"):
        load_historical_policy_validation(archive="missing", compute_figures=value)
