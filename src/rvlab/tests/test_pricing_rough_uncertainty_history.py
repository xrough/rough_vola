"""Spent campaigns are byte-verified replay, never repriced by a later stage."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_uncertainty_history import (
    CENTERING_PROFILE, load_historical_centering,
)


def _archive(tmp_path, **stage_updates):
    root = tmp_path / "archive"
    results = root / "results"
    results.mkdir(parents=True)
    stage = {"backend": "rough_heston_control_centering",
        "profile_fingerprint": CENTERING_PROFILE, "primary_complete": True,
        "american_pricing_ready": False, "stock_delta_ready": False,
        "spent_nested_value": {"original": 17}}
    stage.update(stage_updates)
    manifest = {"notebook": "01_pricing_and_exercise",
        "evidence_version": "pricing_research_v1", "created_at": "original-time",
        "rough_control_centering": stage,
        "tables": {"rough_control_centering_pooled": 1,
            "rough_control_centering_failures": 0, "other_stage": 1},
        "figures": ["rough_control_centering_independent_repeats"]}
    (results / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame([{"quantity": "contrast", "estimate": -.0010733824631053665}]).to_csv(
        results / "rough_control_centering_pooled.csv", index=False)
    (results / "rough_control_centering_failures.csv").write_text("\n")
    (results / "rough_control_centering_independent_repeats.png").write_bytes(b"original-png")
    _seal(root)
    return root


def _seal(root):
    rows = []
    for path in sorted((root / "results").iterdir()):
        data = path.read_bytes()
        rows.append({"preserved": str(path.relative_to(root)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()})
    (root / "preservation_manifest.json").write_text(json.dumps({"files": rows}))


def test_historical_centering_keeps_values_figures_and_original_scope(tmp_path):
    root = _archive(tmp_path)
    seen = []
    class Context:
        def progress(self, message): seen.append(message)
    result = load_historical_centering(Context(), archive=root)
    assert set(result["tables"]) == {"pooled", "failures"}
    assert result["tables"]["pooled"].iloc[0].estimate == -.0010733824631053665
    assert result["tables"]["failures"].empty
    assert result["figures"]["independent_repeats"].read_bytes() == b"original-png"
    manifest = result["manifest"]
    assert manifest["profile_fingerprint"] == CENTERING_PROFILE
    assert manifest["historical_replay"] and not manifest["current_engine_execution"]
    assert manifest["historical_execution_created_at"] == "original-time"
    assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
    assert manifest["spent_nested_value"] == {"original": 17}
    assert seen


def test_historical_centering_can_skip_figure_objects(tmp_path):
    assert load_historical_centering(archive=_archive(tmp_path), compute_figures=False)["figures"] == {}


@pytest.mark.parametrize("updates", [
    {"profile_fingerprint": "replacement"}, {"primary_complete": False},
    {"backend": "other"},
])
def test_historical_centering_rejects_a_different_campaign(tmp_path, updates):
    with pytest.raises(ValueError, match="declared complete"):
        load_historical_centering(archive=_archive(tmp_path, **updates))


def test_historical_centering_rejects_changed_preserved_bytes(tmp_path):
    root = _archive(tmp_path)
    (root / "results/rough_control_centering_pooled.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed preserved"):
        load_historical_centering(archive=root)


def test_historical_centering_rejects_changed_declared_count(tmp_path):
    root = _archive(tmp_path)
    manifest = root / "results/manifest.json"
    obj = json.loads(manifest.read_text())
    obj["tables"]["rough_control_centering_pooled"] = 2
    manifest.write_text(json.dumps(obj)); _seal(root)
    with pytest.raises(ValueError, match="table count"):
        load_historical_centering(archive=root)


def test_historical_centering_checks_boolean_before_reading_files():
    with pytest.raises(TypeError, match="boolean"):
        load_historical_centering(archive=Path("absent"), compute_figures=1)
