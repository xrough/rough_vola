"""Report evidence stays separate from numerical execution and market readiness."""
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import _pricing_first_stage as report
from rvlab.experiments import pricing


def archive(tmp_path, *, count=1, data_kind="synthetic", stem="02_synthetic_calibration"):
    folder = tmp_path / "archive"
    folder.mkdir()
    result = folder / "results" / stem
    result.mkdir(parents=True)
    name = "recorded_fit"
    (result / (name + ".csv")).write_text("price\n1.25\n")
    manifest = {"notebook": stem, "evidence_version": "pricing_research_v1",
        "data_kind": data_kind, "tables": {name: count}, "figures": []}
    (result / "manifest.json").write_text(json.dumps(manifest))
    rows = [{"source": str(p.relative_to(folder)), "preserved": str(p.relative_to(folder)),
             "bytes": p.stat().st_size, "sha256": report.sha(p)}
            for p in sorted(result.iterdir())]
    inventory = folder / "preservation_manifest.json"
    inventory.write_text(json.dumps({"files": rows}))
    return folder, report.sha(inventory)


def test_verified_historical_report_loads_without_numerical_execution(tmp_path):
    folder, digest = archive(tmp_path)
    result = report.load_preserved_report("02_synthetic_calibration", archive=folder,
                                         expected_sha256=digest)
    assert result["tables"]["recorded_fit"].price.tolist() == [1.25]
    assert len(result["input_files"]) == 3
    assert result["manifest"]["evidence_version"] == "pricing_research_v1"


def test_changed_historical_csv_fails_closed(tmp_path):
    folder, digest = archive(tmp_path)
    (folder / "results/02_synthetic_calibration/recorded_fit.csv").write_text("price\n2.50\n")
    with pytest.raises(ValueError, match="evidence changed"):
        report.load_preserved_report("02_synthetic_calibration", archive=folder,
                                     expected_sha256=digest)


def test_inventory_identity_is_required_before_reading_history(tmp_path):
    folder, digest = archive(tmp_path)
    with pytest.raises(ValueError, match="inventory changed"):
        report.load_preserved_report("02_synthetic_calibration", archive=folder,
                                     expected_sha256="different")


def test_historical_row_count_and_missing_required_prefix_are_rejected(tmp_path):
    folder, digest = archive(tmp_path, count=2)
    with pytest.raises(ValueError, match="row count"):
        report.load_preserved_report("02_synthetic_calibration", archive=folder,
                                     expected_sha256=digest)
    with pytest.raises(ValueError, match="Required historical evidence"):
        report.load_preserved_report("02_synthetic_calibration", prefixes=("absent_",),
                                     archive=folder, expected_sha256=digest)


@pytest.mark.parametrize("kind", ["market", "proxy"])
def test_synthetic_report_cannot_load_market_records(tmp_path, kind):
    folder, digest = archive(tmp_path, data_kind=kind)
    with pytest.raises(ValueError, match="cannot load market"):
        report.load_preserved_report("02_synthetic_calibration", archive=folder,
                                     expected_sha256=digest)


@pytest.mark.parametrize("name", ["../escape.csv", "results/../escape.csv"])
def test_inventory_paths_cannot_escape_archive(tmp_path, name):
    folder, digest = archive(tmp_path)
    p = folder / "preservation_manifest.json"
    data = json.loads(p.read_text())
    # The second path resolves inside the archive and is allowed; it still
    # cannot replace a required unrecorded table or cause a fallback.
    data["files"][0]["preserved"] = name
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.load_preserved_report("02_synthetic_calibration", archive=folder,
                                     expected_sha256=report.sha(p))


def test_first_stage_output_namespace_preserves_old_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "ROOT", tmp_path)
    for name in pricing.NOTEBOOKS:
        stem = name.removesuffix(".ipynb")
        assert pricing.setup_first_stage(name).output_dir == tmp_path / "notebooks/_output/pricing/first_stage_v1" / stem
        assert pricing.setup_demo(name).output_dir == tmp_path / "notebooks/_output/pricing/research_v1" / stem
    with pytest.raises(ValueError):
        pricing.setup_first_stage("another_notebook.ipynb")


@pytest.mark.parametrize("seed", [True, 43, -1, 42.0])
def test_recorded_first_stage_design_requires_explicit_supported_seed(seed):
    with pytest.raises(ValueError, match="seed 42"):
        report.first_stage_calibration(SimpleNamespace(progress=lambda _: None), seed=seed)


def test_calibration_report_retains_original_execution_and_reprices_without_refitting(monkeypatch):
    original = {"evidence_version": "pricing_research_v1", "data_kind": "synthetic",
                "american_joint_executed": True, "source_sha256": {"old.py": "old"}}
    historical = {"tables": {}, "figures": {}, "manifest": original, "input_files": []}
    monkeypatch.setattr(report, "load_preserved_report", lambda *_a, **_k: historical)
    fake = SimpleNamespace(reprice_selected_calibration=lambda _t, _m:
                           pd.DataFrame({"passed": [True] * 4}))
    monkeypatch.setitem(sys.modules, "rvlab.experiments._pricing_first_stage_reprice", fake)
    monkeypatch.setitem(sys.modules, "rvlab.experiments._pricing_first_stage_reporting",
                        SimpleNamespace(add_first_stage_reporting=lambda x: x))
    result = report.first_stage_calibration(SimpleNamespace(progress=lambda _: None))
    m = result["manifest"]
    assert m["historical_numerical_manifest"] == original
    assert original["american_joint_executed"] is True
    assert m["american_joint_executed"] is False and m["american_joint_included"] is True
    assert m["optimizer_calls_this_report"] == 0
    assert m["current_source_repricing_pass"] and not m["current_calibration_recomputed"]
    assert m["evidence_version"] == "pricing_first_stage_v1"


def test_new_facades_delegate_without_changing_the_scientific_result(monkeypatch):
    calls = []
    for name in ("pricing", "calibration", "spy"):
        monkeypatch.setattr(report, "first_stage_" + name,
            lambda context, _name=name, **kwargs: calls.append((_name, context, kwargs)) or {"kind": _name})
    monkeypatch.setattr(pricing, "_finish", lambda context, result: result)
    for name in ("pricing", "calibration", "spy"):
        assert getattr(pricing, "run_first_stage_" + name)("context", seed=42) == {"kind": name}
    assert calls == [(name, "context", {"seed": 42}) for name in ("pricing", "calibration", "spy")]


def test_report_stamp_does_not_promote_numerical_readiness():
    result = {"manifest": {"american_pricing_ready": False, "stock_delta_ready": False}}
    report._stamp(result, role="pricing")
    assert result["manifest"]["artifact_family"] == "first_stage_v1"
    assert not result["manifest"]["american_pricing_ready"]
    assert not result["manifest"]["stock_delta_ready"]


def test_first_stage_manifest_binds_saved_table_bytes(tmp_path):
    context = pricing.DemoContext("02_synthetic_calibration", tmp_path, time.perf_counter())
    result = {"tables": {"prices": pd.DataFrame({"price": [1.25, 2.5]})},
              "figures": {}, "manifest": {"artifact_family": "first_stage_v1"}}
    saved = pricing._finish(context, result)["manifest"]
    path = tmp_path / "prices.csv"
    assert saved["artifact_sha256"] == {"prices.csv": report.sha(path)}
    # Equal row counts and file lengths cannot conceal a post-execution edit.
    before = path.stat().st_size
    path.write_text(path.read_text().replace("1.25", "9.25"))
    assert path.stat().st_size == before
    assert report.sha(path) != saved["artifact_sha256"]["prices.csv"]
