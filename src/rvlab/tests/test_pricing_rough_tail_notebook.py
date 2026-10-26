"""Tail-attribution displays retain all paths and distinguish missing evidence."""
from pathlib import Path

import pandas as pd
import pytest


def _cells():
    nbformat = pytest.importorskip("nbformat")
    path = Path(__file__).resolve().parents[3] / "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb"
    notebook = nbformat.read(path, as_version=4)
    cells = [c.source for c in notebook.cells if c.cell_type == "code" and
        c.source.startswith('if results["manifest"].get("rough_tail_attribution_included"')]
    assert len(cells) == 3
    return cells


def _environment(manifest, tables, frames, text):
    return {"results": {"manifest": manifest, "figures": {}}, "tables": tables,
        "pd": pd, "display": text.append, "Markdown": str,
        "present_table": lambda f, **k: frames.append((f.copy(), k)),
        "present_figure": lambda *a, **k: pytest.fail("Fixture has no figures")}


def test_disabled_tail_study_needs_no_campaign_results():
    frames, text = [], []
    environment = _environment({}, {}, frames, text)
    for cell in _cells():
        exec(cell, environment)
    assert not frames
    assert text == ["The large-tail policy/dual attribution study is disabled in this execution."]


def test_failed_tail_study_shows_failure_without_fabricated_attribution():
    frames, text = [], []
    failures = pd.DataFrame([{"error": "independent numerical failure"}])
    environment = _environment({"rough_tail_attribution_executed": True},
        {"rough_tail_attribution_failures": failures}, frames, text)
    for cell in _cells():
        exec(cell, environment)
    assert any("No completed independent tail-attribution evidence" in item for item in text)
    assert len(frames) == 1
    pd.testing.assert_frame_equal(frames[0][0], failures)
    assert frames[0][1]["collapsed"] is True


def test_full_cross_matrix_and_state_paths_survive_compact_presentation():
    policies = ["frozen selected policy"] + [f"rule {i}" for i in range(6)]
    controls = ["current baseline"] + [f"control {i}" for i in range(16)]
    comparison = pd.DataFrame([{"policy": p, "control": c,
        "selected_policy": p == "rule 0", "selected_control": c == "control 0",
        "price": 3., "upper": 3.002, "gap": .002,
        "price_error_budget": .003, "price_allowance": .0013,
        "price_budget_pass": False} for c in controls for p in policies])
    spent = pd.DataFrame({"path_index": range(8192), "gap": [.002] * 8192,
        "acceptance_input": False, "used_for_selection": False})
    tables = {"rough_tail_attribution_comparison": comparison,
        "rough_tail_attribution_spent_tail_states": spent}
    frames, text = [], []
    environment = _environment({"rough_tail_attribution_executed": True,
        "rough_tail_attribution": {"selected_policy": "rule 0", "selected_control": "control 0"}},
        tables, frames, text)
    for cell in _cells():
        exec(cell, environment)
    curated = next(f for f, k in frames if k["title"] == "Training-selected comparisons on untouched final paths")
    assert len(curated) == 4
    matrix, matrix_options = next((f, k) for f, k in frames if k["title"].endswith("· comparison"))
    pd.testing.assert_frame_equal(matrix, comparison)
    assert len(matrix) == 119 and matrix_options["max_rows"] == 119
    states, state_options = next((f, k) for f, k in frames if k["title"].endswith("· spent tail states"))
    pd.testing.assert_frame_equal(states, spent)
    assert len(states) == 8192 and state_options["max_rows"] == 30
    assert "30 of 8192" in state_options["caption"]
    assert "not an acceptance subset" in state_options["caption"]
