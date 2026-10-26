"""Actual display cells distinguish fixed-policy sampling from optimal Greeks."""
from pathlib import Path

import pandas as pd
import pytest


def _cells():
    nbformat = pytest.importorskip("nbformat")
    path = Path(__file__).resolve().parents[3]/"notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb"
    notebook = nbformat.read(path, as_version=4)
    cells = [c.source for c in notebook.cells if c.cell_type == "code" and
        c.source.startswith('if results["manifest"].get("rough_stopping_delta_included"')]
    assert len(cells) == 3
    return cells


def test_disabled_followup_does_not_need_results():
    seen = []
    env = {"results": {"manifest": {}, "figures": {}}, "tables": {}, "pd": pd,
        "display": seen.append, "Markdown": str,
        "present_table": lambda *a, **k: pytest.fail("Disabled stage has no table"),
        "present_figure": lambda *a, **k: pytest.fail("Disabled stage has no figure")}
    for cell in _cells():
        exec(cell, env)
    assert seen == ["The stopping-gap and stock-delta follow-up is disabled in this execution."]


def test_failed_followup_preserves_error_without_fabricated_prices_or_deltas():
    seen = []
    env = {"results": {"manifest": {"rough_stopping_delta_executed": True}, "figures": {}},
        "tables": {"rough_stopping_delta_failures": pd.DataFrame([{"error": "failed independently"}])},
        "pd": pd, "display": seen.append, "Markdown": str,
        "present_table": lambda f, **k: seen.append(f.to_dict("records")),
        "present_figure": lambda *a, **k: pytest.fail("Failed stage has no figure")}
    for cell in _cells():
        exec(cell, env)
    assert any("No completed stopping comparison" in str(x) for x in seen)
    assert any("No completed independent stock-delta pool" in str(x) for x in seen)
    assert any("failed independently" in str(x) for x in seen)


def test_readable_summary_preserves_complete_raw_and_controlled_rows_in_appendix():
    frames = []
    data = pd.DataFrame([{"policy": p, "estimator": e, "bump_relative": b,
        "paths": 4096, "mean": -.4, "ci_half_width": .002,
        "delta_allowance": .004, "sampling_passed": e != "raw"}
        for p in ("original_policy", "selected_policy")
        for e in ("raw", "stopped martingale") for b in (.01, .005, .0025)])
    env = {"results": {"manifest": {"rough_stopping_delta_executed": True}, "figures": {}},
        "tables": {"rough_stopping_delta_delta_estimates": data}, "pd": pd,
        "display": lambda x: None, "Markdown": str,
        "present_table": lambda f, **k: frames.append((f.copy(), k)),
        "present_figure": lambda *a, **k: pytest.fail("Fixture has no figure")}
    for cell in _cells():
        exec(cell, env)
    summary = next(frame for frame, options in frames if options["title"] == "The stopped control addresses sampling uncertainty")
    assert len(summary) == 6
    complete = next(frame for frame, options in frames if options["title"] == "All primary raw and controlled stock-bump estimates")
    pd.testing.assert_frame_equal(complete, data)
