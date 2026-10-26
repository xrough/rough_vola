"""The actual reader-facing notebook must preserve failed numerical evidence."""
from pathlib import Path

import pandas as pd
import pytest


def _policy_cells():
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(Path(__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb", as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and
        cell.source.startswith('if results["manifest"].get("rough_policy_improvement_included"')]
    assert len(cells) == 4
    return cells


@pytest.mark.parametrize("summary_rows", [[], [{"case": "unresolved", "status": "numerical_failure"}]])
def test_policy_notebook_all_failed_study_displays_audit_without_requiring_fit_fields(summary_rows):
    from rvlab.experiments.pricing_presentation import _table_html
    displayed = []
    environment = {"pd": pd, "results": {"manifest": {
        "rough_policy_improvement_executed": True, "rough_policy_improvement": {}},
        "figures": {}},
        "tables": {"rough_policy_improvement_summary": pd.DataFrame(summary_rows),
            "rough_policy_improvement_failures": pd.DataFrame([{"error": "failure retained"}])},
        "display": displayed.append, "Markdown": str,
        "present_table": lambda frame, **kwargs: displayed.append(_table_html(frame, **kwargs)),
        "present_figure": lambda *a, **k: pytest.fail("Failed pricing must not fabricate figures")}
    for source in _policy_cells():
        exec(source, environment)
    text = "\n".join(map(str, displayed))
    assert "No completed policy-improvement cases" in text
    assert "failure retained" in text
    assert "tolerances are unchanged" in text


def test_disabled_policy_study_presentation_does_not_require_new_artifacts():
    displayed = []
    environment = {"results": {"manifest": {}, "figures": {}}, "tables": {},
        "display": displayed.append, "Markdown": str,
        "present_table": lambda *a, **k: pytest.fail("Disabled study has no tables"),
        "present_figure": lambda *a, **k: pytest.fail("Disabled study has no figures")}
    for source in _policy_cells():
        exec(source, environment)
    assert displayed == ["The exercise-policy investigation is disabled in this execution."]


def test_historical_policy_study_stays_visible_without_current_engine_execution():
    displayed = []
    environment = {"pd": pd, "results": {"manifest": {
        "rough_policy_improvement_executed": False,
        "rough_policy_improvement_included": True,
        "rough_policy_improvement": {"historical_replay": True}}, "figures": {}},
        "tables": {}, "display": displayed.append, "Markdown": str,
        "present_table": lambda *a, **k: pytest.fail("No recorded tables in this fixture"),
        "present_figure": lambda *a, **k: pytest.fail("No recorded figures in this fixture")}
    for source in _policy_cells():
        exec(source, environment)
    text = "\n".join(map(str, displayed))
    assert "No completed policy-improvement cases" in text
    assert "tolerances are unchanged" in text
    assert "disabled" not in text


def test_enabled_policy_study_shows_the_observed_raw_control_discrepancy_in_main_text():
    frames = []
    rows = [{"case": label, "raw_minus_controlled_gain": gain,
        "raw_minus_controlled_standard_error": .0084, "confidence_low": gain-.0164,
        "confidence_high": gain+.0164, "z_score": gain/.0084, "zero_consistent": consistent}
        for label, gain, consistent in (("ordinary case", 0., True), ("joint refinement", .020149, False))]
    environment = {"pd": pd, "results": {"manifest": {
        "rough_policy_improvement_executed": True, "rough_policy_improvement": {}}, "figures": {}},
        "tables": {"rough_policy_improvement_control_neutrality": pd.DataFrame(rows)},
        "display": lambda value: None, "Markdown": str,
        "present_table": lambda frame, **kwargs: frames.append((frame.copy(), kwargs)),
        "present_figure": lambda *a, **k: pytest.fail("No recorded figures in this fixture")}
    for source in _policy_cells():
        exec(source, environment)
    assert len(frames) == 1
    assert frames[0][0].case.tolist() == ["joint refinement"]
    assert frames[0][1]["title"] == "Retained pointwise raw/control discrepancy"
    assert "cause remains unresolved" in frames[0][1]["caption"]
