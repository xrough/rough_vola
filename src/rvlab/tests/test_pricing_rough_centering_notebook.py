"""The actual centering notebook cells retain absent/failed fresh evidence."""
from pathlib import Path

import pandas as pd
import pytest


def _centering_cells():
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(Path(__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb", as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and
        cell.source.startswith('if results["manifest"].get("rough_control_centering_included"')]
    assert len(cells) == 3
    return cells


def test_disabled_centering_notebook_needs_no_new_artifacts():
    displayed = []
    environment = {"results": {"manifest": {}, "figures": {}}, "tables": {},
        "display": displayed.append, "Markdown": str,
        "present_table": lambda *a, **k: pytest.fail("Disabled study has no tables"),
        "present_figure": lambda *a, **k: pytest.fail("Disabled study has no figures")}
    for source in _centering_cells():
        exec(source, environment)
    assert displayed == ["The raw/control centering investigation is disabled in this execution."]


def test_failed_centering_notebook_keeps_audit_without_fabricating_fresh_pool():
    from rvlab.experiments.pricing_presentation import _table_html
    displayed = []
    environment = {"pd": pd, "results": {"manifest": {
        "rough_control_centering_executed": True}, "figures": {}},
        "tables": {"rough_control_centering_failures": pd.DataFrame([
            {"error": "independent stream failure retained"}])},
        "display": displayed.append, "Markdown": str,
        "present_table": lambda frame, **kwargs: displayed.append(_table_html(frame, **kwargs)),
        "present_figure": lambda *a, **k: pytest.fail("A failed study has no fresh figures")}
    for source in _centering_cells():
        exec(source, environment)
    text = "\n".join(map(str, displayed))
    assert "No completed independent-repeat pool" in text
    assert "independent stream failure retained" in text


def test_centering_notebook_displays_every_repeat_without_sign_filtering():
    frames = []
    repeat_rows = [{"quantity": "contrast", "seed": seed, "paths": 8192,
        "estimate": value, "standard_error": .008, "confidence_low": value-.016,
        "confidence_high": value+.016, "z_score": value/.008,
        "zero_consistent": abs(value) <= .016}
        for seed, value in ((410005042, -.03), (411005042, .03))]
    pool = pd.DataFrame([{**repeat_rows[0], "streams": 2, "paths": 16384,
        "estimate": 0., "z_score": 0., "zero_consistent": True}])
    environment = {"pd": pd, "results": {"manifest": {
        "rough_control_centering_executed": True}, "figures": {}},
        "tables": {"rough_control_centering_per_seed": pd.DataFrame(repeat_rows),
            "rough_control_centering_pooled": pool},
        "display": lambda value: None, "Markdown": str,
        "present_table": lambda frame, **kwargs: frames.append((frame.copy(), kwargs)),
        "present_figure": lambda *a, **k: pytest.fail("No figure recorded in this fixture")}
    for source in _centering_cells():
        exec(source, environment)
    repeats = next(frame for frame, kwargs in frames if
        kwargs["title"] == "All eight predeclared independent repeats")
    assert repeats.seed.tolist() == [410005042, 411005042]
    assert repeats.estimate.tolist() == [-.03, .03]
