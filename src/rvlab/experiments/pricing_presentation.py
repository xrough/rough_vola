"""Reader-facing displays for pricing notebooks; no pricing or fitting logic.

``python -m rvlab.experiments.pricing_presentation --refresh all`` rebuilds
only presentation cells from recorded local artifacts. It never recalibrates
and preserves the notebooks' original numerical execution provenance.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

import numpy as np
import pandas as pd


_CSS = """
<style>
.rvlab-table {margin:1.25em 0 1.8em; overflow-x:auto; color:inherit;}
.rvlab-table table {border-collapse:collapse; font-size:13px; width:auto;
  min-width:55%; line-height:1.6; text-align:left;}
.rvlab-table th {border-bottom:2px solid #78909c; text-align:left;
  padding:9px 12px; font-weight:600; white-space:normal;}
.rvlab-table td {border-bottom:1px solid #dde3e8; padding:7px 12px;
  vertical-align:top; max-width:260px; overflow-wrap:anywhere;}
.rvlab-table td.numeric {text-align:right; font-variant-numeric:tabular-nums;}
.rvlab-caption {font-size:13px; line-height:1.6; color:#52616b; margin:.6em 0;}
.rvlab-table-title {font-weight:600; margin-bottom:.5em;}
.rvlab-figure {margin:1.25em 0 2em; max-width:1100px;}
.rvlab-figure img {max-width:100%; height:auto;}
.rvlab-audit {margin:1.5em 0; padding:12px 16px; border:1px solid #cdd8df;
  border-radius:5px; line-height:1.7;}
.rvlab-audit summary {cursor:pointer; font-weight:600;}
.rvlab-audit pre {font-size:12px; white-space:pre-wrap; overflow-wrap:anywhere;
  max-height:450px; overflow:auto;}
</style>
"""

_LABELS = {
    "label": "Fit", "model": "Model", "input_scenario": "Inputs",
    "train_rmse": "Train RMSE ($)", "test_rmse": "Held-out RMSE ($)",
    "refined_test_rmse": "Refined held-out RMSE ($)",
    "train_scaled_rmse": "Train spread units", "test_scaled_rmse": "Held-out spread units",
    "train_clean_rmse": "Train vs clean ($)", "test_clean_rmse": "Held-out vs clean ($)",
    "test_refined_clean_rmse": "Held-out after refinement ($)",
    "test_delta_rmse": "Held-out delta RMSE", "stage": "Fit stage", "dataset": "Quotes",
    "max_price_change": "Largest refinement change ($)",
    "active_bounds": "Active bounds", "nfev": "Evaluations",
    "numerical_stable": "Refinement passed", "optimizer_success": "Optimizer converged",
    "status": "Outcome", "elapsed_seconds": "Runtime (s)",
    "free_parameter_count": "Free parameters", "max_change_vs_finest": "Change vs finest ($)",
    "expiry_date": "Expiry", "snapshot_date": "Date", "box": "Search box",
}

_VALUES = {
    "raw_european_proxy": "Raw parity proxy",
    "american_bs_sensitivity": "Exercise sensitivity",
    "forward_only_control": "Forward-only control",
    "heston": "Heston", "rough_heston": "Rough Heston", "bs": "Black–Scholes",
    "heston_curve": "Heston curve", "exploratory": "Initial fit", "continuation": "Budget continuation",
    "converged": "Converged", "exhausted": "Exhausted", "unstable": "Unstable",
    "converged_unstable": "Converged; refinement failed", "optimizer_failed": "Optimizer failed",
    "failed": "Failed", "numerical_failure": "Numerical failure",
    "original": "Original", "expanded": "Expanded",
}


def presentation_style():
    """Install scoped, theme-compatible table and figure styling."""
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6,
        "lines.linewidth": 1.8, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.prop_cycle": cycler(color=["#2878A5", "#D17B32", "#268577", "#9A6599", "#737B83"]),
    })
    display(HTML(_CSS))


def _format(value, spec=None):
    if value is None or value is pd.NA or value is pd.NaT:
        return "—"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "—"
    if spec is not None:
        return str(spec(value) if callable(spec) else
                   (spec.format(value) if "{" in spec else format(value, spec)))
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items()) or "None"
    if isinstance(value, str):
        return _VALUES.get(value, value)
    return str(value)


def _table_html(dataframe, *, columns=None, labels=None, formats=None,
                title=None, caption=None, max_rows=20):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("present_table expects a pandas DataFrame")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    labels, formats = labels or {}, formats or {}
    selected = dataframe if columns is None else dataframe.loc[:, list(columns)]
    parts = ['<div class="rvlab-table">']
    if title:
        parts.append(f'<div class="rvlab-table-title">{escape(title)}</div>')
    if selected.empty:
        parts.append("<p>No recorded rows.</p>")
    else:
        parts.append("<table><thead><tr>")
        for col in selected:
            parts.append(f"<th>{escape(str(labels.get(col, _LABELS.get(col, str(col).replace('_', ' ').capitalize()))))}</th>")
        parts.append("</tr></thead><tbody>")
        for row in selected.head(max_rows).itertuples(index=False, name=None):
            parts.append("<tr>")
            for col, value in zip(selected.columns, row):
                numeric = isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
                parts.append(f'<td class="{"numeric" if numeric else "text"}">{escape(_format(value, formats.get(col)))}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table>")
    if len(selected) > max_rows:
        parts.append(f'<p class="rvlab-caption">Showing {max_rows} of {len(selected):,} rows; the appendix links the complete CSV.</p>')
    if caption:
        parts.append(f'<p class="rvlab-caption">{escape(caption)}</p>')
    parts.append("</div>")
    return "".join(parts)


def present_table(dataframe, *, collapsed=False, **kwargs):
    """Display a compact selected table, escaping content and preserving inputs.

    ``labels`` and ``formats`` are mappings keyed by original column names;
    format values may be Python format specifications or callables. Use
    ``collapsed=True`` for complete comparisons in a notebook appendix.
    """
    from IPython.display import HTML, display
    if not isinstance(collapsed, bool):
        raise TypeError("collapsed must be boolean")
    body = _table_html(dataframe, **kwargs)
    if collapsed:
        summary = escape(str(kwargs.get("title") or "Complete comparison"))
        body = f'<details class="rvlab-audit"><summary>{summary}</summary>{body}</details>'
    display(HTML(body))


def present_figure(results, name, *, caption=None):
    """Embed a computed Matplotlib figure or its recorded PNG, with a caption."""
    from IPython.display import HTML, display
    figure = results["figures"][name]
    if isinstance(figure, (str, Path)):
        pixels = Path(figure).read_bytes()
    else:
        import matplotlib.pyplot as plt
        buffer = BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        pixels = buffer.getvalue()
        plt.close(figure)
    encoded = base64.b64encode(pixels).decode("ascii")
    description = caption or name.replace("_", " ").capitalize()
    body = f'<figure class="rvlab-figure"><img alt="{escape(description, quote=True)}" src="data:image/png;base64,{encoded}">'
    if caption:
        body += f'<figcaption class="rvlab-caption">{escape(caption)}</figcaption>'
    display(HTML(body + "</figure>"), metadata={"rvlab_figure": name})


def present_archive(results, *, title="Technical appendix"):
    """Offer complete artifacts and provenance without flooding the main text."""
    from IPython.display import HTML, display
    manifest = results["manifest"]
    stem = manifest["notebook"]
    if Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError("invalid recorded notebook stem")
    from .pricing import artifact_family
    family = manifest.get("artifact_family", artifact_family(stem))
    if family not in {artifact_family(stem), "first_stage_v1", "first_stage_v2"}:
        raise ValueError("Unsupported recorded artifact family")
    prefix = f"../_output/pricing/{family + '/' if family else ''}{quote(stem)}/"
    rows = []
    for name, table in results["tables"].items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid recorded artifact stem")
        rows.append(f'<tr><td><a href="{prefix}{quote(name)}.csv">{escape(name.replace("_", " "))}</a></td><td>{len(table):,}</td></tr>')
    figure_links = []
    for name in results.get("figures", {}):
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid recorded figure stem")
        figure_links.append(f'<li><a href="{prefix}{quote(name)}.png">{escape(name.replace("_", " "))}</a></li>')
    saved_figures = (f'<details><summary>Saved figures · {len(figure_links)}</summary>'
        '<p>Current comparisons, reference checks and preserved historical figures retain the roles '
        'described in the narrative and manifest. Listing a figure does not identify it as a new computation.</p>'
        f'<ul>{"".join(figure_links)}</ul></details>') if figure_links else ""
    created = escape(str(manifest.get("created_at", "not recorded")))
    elapsed = _format(manifest.get("elapsed_seconds"), ".1f")
    raw = escape(json.dumps(manifest, indent=2, default=str, allow_nan=False))
    timing = (f'Report artifacts collected {created} in {elapsed} seconds. '
        'Underlying numerical runtimes and original experiment identities are retained in the evidence records.'
        if family in {"first_stage_v1", "first_stage_v2"} else
        f'Numerical results recorded {created}; original experiment runtime {elapsed} seconds.')
    display(HTML(f'<details class="rvlab-audit"><summary>{escape(title)} · full tables and reproducibility</summary>'
        f'<p>{timing} '
        'The saved notebook embeds its figures and selected tables. CSV/JSON links require the local artifact directory.</p>'
        '<p>Presentation-only refreshes reuse these results without recalibration; numerical execution dates and manifests are preserved. '
        + ('Report cells collect verified numerical evidence; the guide separately documents cold computation.</p>'
           if family in {"first_stage_v1", "first_stage_v2"} else 'Run all cells to recompute the experiment from its declared inputs.</p>')
        + f'<div class="rvlab-table"><table><thead><tr><th>Complete diagnostic CSV</th><th>Rows</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        + saved_figures
        + f'<p><a href="{prefix}manifest.json">Original numerical manifest (JSON)</a></p>'
        f'<details><summary>Recorded parameters, controls and fingerprints</summary><pre>{raw}</pre></details></details>'))


def load_recorded_results(notebook_name, *, artifact_root=None, family=None):
    """Load recorded display inputs; missing files raise, with no data fallback."""
    from .pricing import ROOT, RESEARCH_STEMS, RESEARCH_EVIDENCE_VERSION, _simple_stem, artifact_family
    from ._pricing_american import AMERICAN_BACKEND, AMERICAN_EVIDENCE_VERSION, SPY_NOTEBOOKS
    stem = _simple_stem(Path(notebook_name).stem)
    if family not in (None, "first_stage_v1", "first_stage_v2"):
        raise ValueError("Unsupported recorded artifact family")
    explicit_first_stage = family in {"first_stage_v1", "first_stage_v2"}
    if explicit_first_stage and stem not in RESEARCH_STEMS:
        raise ValueError("First-stage reports use the three active notebook names")
    destination_family = family if explicit_first_stage else artifact_family(stem)
    destination = Path(artifact_root) / stem if artifact_root else ROOT / "notebooks/_output/pricing" / destination_family / stem
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("notebook") != stem:
        raise ValueError("Recorded manifest belongs to a different notebook")
    if stem in RESEARCH_STEMS:
        if explicit_first_stage and (manifest.get("artifact_family") != family or
                manifest.get("evidence_version") != "pricing_" + family):
            raise ValueError(f"First-stage reports require matching {family} evidence")
        if not explicit_first_stage and manifest.get("evidence_version") != RESEARCH_EVIDENCE_VERSION:
            raise ValueError("Consolidated research requires current pricing_research_v1 evidence; rerun this notebook")
        if stem == "03_spy_calibration":
            if (manifest.get("backend") != AMERICAN_BACKEND or
                    manifest.get("exercise_style") != "american"):
                raise ValueError("SPY research requires the American backend and exercise style")
        elif manifest.get("data_kind") != "synthetic" or "spy_inputs" in manifest:
            raise ValueError("Synthetic research artifacts must contain synthetic observations only")
    if stem in SPY_NOTEBOOKS or stem == "08_synthetic_american_pricing":
        if (manifest.get("backend") != AMERICAN_BACKEND or
                manifest.get("evidence_version") != AMERICAN_EVIDENCE_VERSION or
                manifest.get("exercise_style") != "american"):
            raise ValueError("Superseded European SPY artifacts cannot be displayed as American evidence; rerun the American notebook")
    if stem == "03_calibration_identification" and (
            manifest.get("evidence_version") != "synthetic_identification_v2" or
            manifest.get("data_kind") != "synthetic" or "spy_inputs" in manifest):
        raise ValueError("Identification artifacts must be synthetic only; rerun notebook 03")
    if explicit_first_stage:
        expected = {f"{_simple_stem(name)}.csv" for name in manifest["tables"]} | {
            f"{_simple_stem(name)}.png" for name in manifest["figures"]}
        recorded = manifest.get("artifact_sha256")
        if not isinstance(recorded, dict) or set(recorded) != expected:
            raise ValueError("First-stage display requires the exact executed CSV/PNG digest map")
        for name, expected_digest in recorded.items():
            path = destination / name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_digest:
                raise ValueError(f"First-stage artifact changed after execution: {name}")
    tables = {}
    for name, expected_rows in manifest["tables"].items():
        _simple_stem(name)
        try:
            table = pd.read_csv(destination / f"{name}.csv")
        except pd.errors.EmptyDataError:
            if expected_rows != 0:
                raise
            table = pd.DataFrame()
        if len(table) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} recorded rows, found {len(table)}")
        tables[name] = table
    figures = {name: destination / f"{_simple_stem(name)}.png" for name in manifest["figures"]}
    for path in figures.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"tables": tables, "figures": figures, "manifest": manifest}


def refresh_presentations(paths):
    """Execute display cells only in fresh kernels; preserve computation cells."""
    import nbformat
    from .pricing import ROOT, _simple_stem, execute_notebook
    for path in map(Path, paths):
        if path.resolve().is_relative_to((ROOT / "notebooks/legacy/pricing").resolve()):
            raise ValueError("Archived pricing notebooks and their outputs are preserved unchanged")
        notebook = nbformat.read(path, as_version=4)
        family = notebook.metadata.get("rvlab_artifact_family")
        if family not in (None, "first_stage_v1", "first_stage_v2"):
            raise ValueError("Unsupported recorded artifact family")
        indexes = [i for i, cell in enumerate(notebook.cells)
                   if cell.cell_type == "code" and cell.metadata.get("rvlab_presentation")]
        if not indexes:
            raise ValueError(f"{path.name} has no marked presentation cells")
        bootstrap = nbformat.v4.new_code_cell(
            f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "import numpy as np\nimport pandas as pd\n"
            "from IPython.display import display, Markdown\n"
            "from rvlab.experiments.pricing_presentation import (load_recorded_results, "
            "presentation_style, present_table, present_figure, present_archive)\n"
            f"results = load_recorded_results({path.stem!r}, family={family!r})\n"
            "tables = results['tables']\n")
        temporary_notebook = nbformat.v4.new_notebook(cells=[bootstrap, *[notebook.cells[i] for i in indexes]])
        with tempfile.TemporaryDirectory(prefix="rvlab-presentation-") as temporary:
            source = Path(temporary) / path.name
            nbformat.write(temporary_notebook, source)
            execute_notebook(source, timeout=120)
            rendered = nbformat.read(source, as_version=4)
        figure_exports = []
        for index, cell in zip(indexes, rendered.cells[1:]):
            notebook.cells[index].outputs = cell.outputs
            # Display-only replay is not a new execution of the numerical cells.
            notebook.cells[index].execution_count = None
            for output in cell.outputs:
                name = output.get("metadata", {}).get("rvlab_figure")
                if name is None:
                    continue
                pixels = re.search(r'src="data:image/png;base64,([^"]+)"',
                                   output.get("data", {}).get("text/html", ""))
                if pixels is None:
                    continue
                export_root = ROOT / "notebooks/_output/pricing"
                if family:
                    export_root = export_root / family
                destination = export_root / "presentation" / path.stem / f"{_simple_stem(name)}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(base64.b64decode(pixels.group(1), validate=True))
                figure_exports.append(str(destination.relative_to(ROOT)))
        notebook.metadata["rvlab_presentation"] = {
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "mode": "recorded-artifact display only; no pricing or calibration rerun",
            "numerical_execution_preserved": True,
            "display_cells": len(indexes),
            "figure_exports": figure_exports,
            "artifact_family": family,
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, path)


def main(argv=None):
    from .pricing import NOTEBOOKS, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", nargs="+", required=True,
                        help="Notebook filenames, or all; reuse recorded results only")
    args = parser.parse_args(argv)
    names = NOTEBOOKS if args.refresh == ["all"] else args.refresh
    if any(name not in NOTEBOOKS for name in names):
        parser.error("refresh selects active notebook filenames only; archived outputs are preserved unchanged")
    refresh_presentations([ROOT / "notebooks/pricing" / name for name in names])


if __name__ == "__main__":
    main()
