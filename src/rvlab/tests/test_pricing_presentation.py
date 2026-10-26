"""Presentation must preserve unsuccessful outcomes and recorded provenance."""

import json

import pandas as pd
import pytest

from rvlab.experiments.pricing_presentation import _table_html, load_recorded_results


def test_compact_table_keeps_failure_statuses_and_does_not_mutate_data():
    table = pd.DataFrame({"status": ["converged", "exhausted", "unstable", "failed"],
                          "test_rmse": [.01234, .03, .04, float("nan")],
                          "internal": ["unneeded"] * 4})
    before = table.copy(deep=True)
    html = _table_html(table, columns=["status", "test_rmse"],
                       formats={"test_rmse": "${:.3f}"}, title="Fit <audit>")
    assert all(status in html for status in ("Converged", "Exhausted", "Unstable", "Failed"))
    assert "$0.012" in html and "—" in html
    assert "unneeded" not in html and "Fit &lt;audit&gt;" in html
    pd.testing.assert_frame_equal(table, before)


def test_table_escapes_labels_and_content_and_declares_truncation():
    table = pd.DataFrame({"text": ["<script>bad()</script>", "second"]})
    html = _table_html(table, labels={"text": "<unsafe>"}, max_rows=1)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;unsafe&gt;" in html and "Showing 1 of 2 rows" in html


def test_recorded_loader_handles_empty_failure_table_and_preserves_manifest(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    manifest = {"notebook": "demo", "tables": {"failures": 0, "fits": 1},
                "figures": ["smile"], "created_at": "original execution"}
    source = json.dumps(manifest)
    (folder / "manifest.json").write_text(source)
    (folder / "failures.csv").write_text("\n")
    (folder / "fits.csv").write_text("status\nexhausted\n")
    (folder / "smile.png").write_bytes(b"recorded pixels")
    results = load_recorded_results("demo.ipynb", artifact_root=tmp_path)
    assert results["tables"]["failures"].empty
    assert results["tables"]["fits"].status.tolist() == ["exhausted"]
    assert results["manifest"] == manifest
    assert (folder / "manifest.json").read_text() == source
    (folder / "fits.csv").write_text("status\n")
    with pytest.raises(ValueError, match="expected 1 recorded rows"):
        load_recorded_results("demo.ipynb", artifact_root=tmp_path)


def test_missing_recorded_results_raise_without_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recorded_results("absent", artifact_root=tmp_path)


def test_display_refresh_never_executes_numerical_cells_or_relabels_provenance(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    from rvlab.experiments import pricing
    from rvlab.experiments.pricing_presentation import refresh_presentations
    path = tmp_path / "demo.ipynb"
    numerical = nbformat.v4.new_code_cell("raise RuntimeError('must not recalibrate')",
                                         execution_count=8)
    display_cell = nbformat.v4.new_code_cell("present_table(results['tables']['fits'])",
        metadata={"rvlab_presentation": True})
    provenance = {"executed_at": "historical execution", "elapsed_seconds": 900}
    nbformat.write(nbformat.v4.new_notebook(cells=[numerical, display_cell],
        metadata={"rvlab_execution": provenance}), path)

    def render_only(temporary_path, **kwargs):
        notebook = nbformat.read(temporary_path, 4)
        assert len(notebook.cells) == 2  # Artifact bootstrap and one display.
        assert all("must not recalibrate" not in c.source for c in notebook.cells)
        notebook.cells[1].outputs = [nbformat.v4.new_output("display_data",
            data={"text/html": "<table>Recorded fit</table>"})]
        nbformat.write(notebook, temporary_path)

    monkeypatch.setattr(pricing, "execute_notebook", render_only)
    refresh_presentations([path])
    refreshed = nbformat.read(path, 4)
    assert refreshed.cells[0] == numerical
    assert refreshed.cells[1].outputs[0].data["text/html"] == "<table>Recorded fit</table>"
    assert refreshed.cells[1].execution_count is None
    assert refreshed.metadata.rvlab_execution == provenance
    assert refreshed.metadata.rvlab_presentation.numerical_execution_preserved is True


def test_collapsed_comparison_embeds_unsuccessful_and_negative_rows_without_sign_filter(monkeypatch):
    from IPython import display as display_module
    from rvlab.experiments.pricing_presentation import present_table
    table = pd.DataFrame({"policy": ["rejected", "selected", "failed"],
        "raw_gain": [-.01, -.002, float("nan")], "controlled_gain": [.0002, .0005, float("nan")]})
    before = table.copy(deep=True)
    shown = []
    monkeypatch.setattr(display_module, "display", shown.append)
    present_table(table, title="All <attempts>", collapsed=True, max_rows=len(table))
    html = shown[0].data
    assert '<details class="rvlab-audit">' in html
    assert '<summary>All &lt;attempts&gt;</summary>' in html
    assert all(value in html for value in ("rejected", "selected", "Failed", "-0.01", "-0.002", "0.0005"))
    pd.testing.assert_frame_equal(table, before)
    with pytest.raises(TypeError, match="collapsed must be boolean"):
        present_table(table, collapsed="yes")
