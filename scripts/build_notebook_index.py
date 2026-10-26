#!/usr/bin/env python3
"""
build_notebook_index.py — generate the archived recipe index in notebooks/legacy/README.md.

The index is parsed out of the notebooks themselves, so it can never drift from
what is actually there, and a new recipe appears in it for free.

    python3 scripts/build_notebook_index.py           # rewrite the index
    python3 scripts/build_notebook_index.py --check   # fail if it is stale

The index sits between the marker comments in notebooks/legacy/README.md; the rest
of that file is hand-written and left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks/legacy"
README = NOTEBOOKS / "README.md"

BEGIN = "<!-- BEGIN RECIPE INDEX -->"
END = "<!-- END RECIPE INDEX -->"

RECIPE_HEADING = re.compile(r"^##\s+(R\d+\.\d+)\s+·\s+(.+?)\s*$", re.M)
USE_WHEN = re.compile(r"\*\*Use when\*\*\s*(.+?)(?:\n\n|\Z)", re.S)
# "Use when" is written for the notebook, where a bare "#r12-2" resolves. Copied
# into the README it would not, and truncating to 110 chars can cut a link in
# half — so unwrap links to their text before the snippet is taken.
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
TOPIC = re.compile(r"^\*\*Topic\*\*\s*(.+?)\s*$", re.M)

# Recipe topics are deliberately specific, which makes 200+ of them hard to browse.
# Each belongs to exactly one family, and the families are named for the stage of a
# research programme they serve rather than for a data-processing step — so the index
# reads as the method it came from. The specific topic stays on the row, so both
# scanning and Ctrl-F still work. A topic missing from this map is an error, not a
# default — see assign_family().
FAMILIES: dict[str, tuple[str, ...]] = {
    "Sources and provenance": (
        "Archives", "Arrays", "Caching", "Data catalog", "Data discovery", "File I/O",
        "File discovery", "Large files", "Many small files", "Memory", "Performance"),
    "Measurement and data quality": (
        "Categorical data", "Cross-source validation", "Data contracts", "Data quality",
        "Datetime", "Drift", "Health check", "Missing data", "Outliers", "Target quality",
        "Timezones"),
    "Combining sources": (
        "As-of joins", "Cross-source joins", "Interval joins", "Joins", "Resampling",
        "Reshaping"),
    "Model inputs and targets": (
        "Cross-sectional features", "Encoding", "Feature engineering", "Feature selection",
        "Preprocessing", "Target construction", "Vectorisation"),
    "Exploration and evidence": (
        "Categorical", "Conditional analysis", "Correlation", "Craft", "Dimensionality",
        "Distributions", "Exploratory analysis", "Exploratory summary", "Relationships",
        "Time series", "Unsupervised", "Visualisation"),
    "Volatility models and estimators": (
        "Conditional volatility", "Estimation", "Estimator bias", "Forecast evaluation",
        "Implied volatility", "Model implementation", "Parameter sweeps",
        "Realized volatility", "Regimes", "Simulation"),
    "Predictive models": (
        "Baselines", "Calibration", "Class imbalance", "Classification", "Ensembling",
        "Hyperparameters", "Model selection", "Pipelines", "Regression", "Thresholds",
        "Uncertainty"),
    "Validation and inference": (
        "Cross-validation", "Experiment design", "Inference", "Leakage", "Multiplicity",
        "Robustness", "Significance", "Validation", "Validation design"),
    "Diagnostics and interpretation": (
        "Classification metrics", "Diagnostics", "Interpretability", "Metrics",
        "Model comparison", "Model diagnostics", "Model evaluation", "Ranking metrics"),
    "From model to position": (
        "Attribution", "Backtesting", "Costs", "Evaluation", "Position construction",
        "Risk metrics", "Risk scaling"),
    "Research practice": (
        "Environment", "Experiment infrastructure", "Navigation", "Provenance",
        "Reporting", "Reproducibility"),
}

TOPIC_FAMILY = {t: fam for fam, topics in FAMILIES.items() for t in topics}


def assign_family(topic: str) -> str:
    """Fail loudly rather than silently dumping a new topic into an 'Other' bucket."""
    try:
        return TOPIC_FAMILY[topic]
    except KeyError:
        raise SystemExit(
            f"topic {topic!r} has no family — add it to FAMILIES in {Path(__file__).name}")


def slug(text: str) -> str:
    """GitHub's heading-anchor rule, for the contents line."""
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).replace(" ", "-")


def source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def collect() -> list[dict]:
    """Recipes in the archived reading order, as index rows."""
    rows = []
    order = NOTEBOOKS / ".order"
    paths = ([NOTEBOOKS / line.strip() for line in order.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
             if order.exists() else sorted(NOTEBOOKS.glob("*.ipynb")))
    for path in paths:
        nb = json.loads(path.read_text())
        for cell in nb["cells"]:
            if cell.get("cell_type") != "markdown":
                continue
            text = source(cell)
            found = RECIPE_HEADING.findall(text)
            if not found:
                continue
            rid, title = found[0]
            use = USE_WHEN.search(text)
            topic = TOPIC.search(text)
            rows.append({
                "topic": topic.group(1).strip() if topic else title.split("—")[0].strip(),
                "recipe": rid,
                "title": title.strip(),
                "notebook": path.name,
                "anchor": rid.lower().replace(".", "-"),
                "use_when": (" ".join(MD_LINK.sub(r"\1", use.group(1)).split())[:110]
                             if use else ""),
            })
    return rows


def render(rows: list[dict]) -> str:
    """Grouped by family, then by topic, so a task name is easy to scan for."""
    by_family: dict[str, list[dict]] = {fam: [] for fam in FAMILIES}
    for r in rows:
        by_family[assign_family(r["topic"])].append(r)

    counts = " · ".join(f"[{fam}](#{slug(fam)}) {len(rs)}"
                        for fam, rs in by_family.items() if rs)
    out = [BEGIN, "",
           f"_{len(rows)} recipes across {len({r['notebook'] for r in rows})} notebooks. "
           "Generated by `scripts/build_notebook_index.py` — do not edit by hand._", "",
           counts, ""]

    for fam, rs in by_family.items():
        if not rs:
            continue
        out += [f"### {fam}", "",
                "| Topic | Recipe | Use when | Go |", "|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: (r["topic"].lower(), r["recipe"])):
            link = f"[{r['recipe']}]({r['notebook']}#{r['anchor']})"
            out.append(f"| {r['topic']} | {r['title']} | {r['use_when']} | {link} |")
        out.append("")
    out += [END]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if README.md is out of date")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print("no recipes found — are the notebooks written yet?", file=sys.stderr)
        return 1

    table = render(rows)
    if not README.exists():
        print(f"{README} does not exist", file=sys.stderr)
        return 1

    text = README.read_text()
    if BEGIN not in text or END not in text:
        print(f"markers {BEGIN} / {END} not found in {README}", file=sys.stderr)
        return 1

    start, stop = text.index(BEGIN), text.index(END) + len(END)
    updated = text[:start] + table + text[stop:]

    if args.check:
        if updated != text:
            print("notebooks/legacy/README.md recipe index is STALE — "
                  "run: python3 scripts/build_notebook_index.py", file=sys.stderr)
            return 1
        print(f"index is current ({len(rows)} recipes)")
        return 0

    README.write_text(updated)
    print(f"wrote {len(rows)} recipes into {README.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
