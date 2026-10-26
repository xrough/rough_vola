"""The matched-policy report cannot borrow the prior report's identity."""
import json
import sys
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import pricing
from rvlab.experiments.pricing_presentation import load_recorded_results, present_archive


def test_new_family_is_separate_from_both_preserved_namespaces(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "ROOT", tmp_path)
    for name in pricing.NOTEBOOKS:
        stem = name.removesuffix(".ipynb")
        assert pricing.setup_policy_comparison(name).output_dir == tmp_path / "notebooks/_output/pricing/first_stage_v2" / stem
        assert pricing.setup_first_stage(name).output_dir == tmp_path / "notebooks/_output/pricing/first_stage_v1" / stem
        assert pricing.setup_demo(name).output_dir == tmp_path / "notebooks/_output/pricing/research_v1" / stem
    with pytest.raises(ValueError):
        pricing.setup_policy_comparison("../escape.ipynb")


def test_new_facades_delegate_to_comparison_report_without_mutating_results(monkeypatch):
    calls = []
    module = SimpleNamespace(**{"policy_comparison_" + name:
        (lambda context, _name=name, **kwargs:
            calls.append((_name, context, kwargs)) or {"kind": _name})
        for name in ("pricing", "calibration", "spy")})
    monkeypatch.setitem(sys.modules, "rvlab.experiments._pricing_policy_comparison_report", module)
    monkeypatch.setattr(pricing, "_finish", lambda context, result: result)
    for name in ("pricing", "calibration", "spy"):
        assert getattr(pricing, "run_policy_comparison_" + name)("context", seed=42) == {"kind": name}
    assert calls == [(name, "context", {"seed": 42}) for name in ("pricing", "calibration", "spy")]


def test_new_saved_artifacts_bind_bytes_and_reject_prior_family(tmp_path):
    stem = "01_pricing_and_exercise"
    folder = tmp_path / stem
    folder.mkdir()
    result = {"tables": {"comparison": pd.DataFrame({"policy": ["lsm"], "price": [1.25]})},
        "figures": {}, "manifest": {"artifact_family": "first_stage_v2",
            "evidence_version": "pricing_first_stage_v2", "data_kind": "synthetic"}}
    context = pricing.DemoContext(stem, folder, time.perf_counter())
    pricing._finish(context, result)
    assert load_recorded_results(stem, artifact_root=tmp_path, family="first_stage_v2")["tables"]["comparison"].price.tolist() == [1.25]
    with pytest.raises(ValueError, match="matching first_stage_v1"):
        load_recorded_results(stem, artifact_root=tmp_path, family="first_stage_v1")
    csv = folder / "comparison.csv"
    csv.write_text(csv.read_text().replace("1.25", "9.25"))
    with pytest.raises(ValueError, match="changed after execution"):
        load_recorded_results(stem, artifact_root=tmp_path, family="first_stage_v2")


def test_second_report_never_falls_back_to_first_report(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "ROOT", tmp_path)
    old = tmp_path / "notebooks/_output/pricing/first_stage_v1/01_pricing_and_exercise"
    old.mkdir(parents=True)
    (old / "manifest.json").write_text(json.dumps({"artifact_family": "first_stage_v1"}))
    with pytest.raises(FileNotFoundError):
        load_recorded_results("01_pricing_and_exercise", family="first_stage_v2")


def test_new_archive_links_explicitly_identify_second_report(monkeypatch):
    from IPython import display as module
    shown = []
    monkeypatch.setattr(module, "display", shown.append)
    present_archive({"manifest": {"notebook": "01_pricing_and_exercise",
        "artifact_family": "first_stage_v2", "elapsed_seconds": 1.},
        "tables": {"failed_trials": pd.DataFrame()}, "figures": {"historical_figure": "recorded.png"}})
    html = shown[0].data
    assert "../_output/pricing/first_stage_v2/01_pricing_and_exercise/failed_trials.csv" in html
    assert "Report artifacts collected" in html and "cold computation" in html
    assert "Saved figures · 1" in html
    assert "../_output/pricing/first_stage_v2/01_pricing_and_exercise/historical_figure.png" in html
    assert "does not identify it as a new computation" in html


def test_saved_figure_links_reject_path_traversal(monkeypatch):
    from IPython import display as module
    monkeypatch.setattr(module, "display", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="figure stem"):
        present_archive({"manifest": {"notebook": "01_pricing_and_exercise",
            "artifact_family": "first_stage_v2"}, "tables": {},
            "figures": {"../old_result": "preserved.png"}})
