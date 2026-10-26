"""Publish the executed first-stage report without copying numerical payloads.

Build only after the final three fresh-kernel executions. Verification checks
current execution sources independently of notebook metadata, the published
closed inventory and the original small artifact inputs. It never prices,
calibrates, simulates, repairs an artifact or changes a readiness flag.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
from html.parser import HTMLParser
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
FAMILY = "first_stage_v1"
VERSION = "pricing_first_stage_v1"
REPORT_FAMILIES = {"first_stage_v1": "pricing_first_stage_v1",
                   "first_stage_v2": "pricing_first_stage_v2"}
STEMS = ("01_pricing_and_exercise", "02_synthetic_calibration", "03_spy_calibration")
TITLES = ("American pricing and policy deltas", "Synthetic calibration and identification", "SPY input handoff · blocked")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the project: {path}")
    return path


def _record(path, *, role):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "role": role}


def _same(record):
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise ValueError(f"Release input changed or is missing: {path}")


def _evidence_inputs(manifest, *, root, records):
    """Bind sealed scientific inputs without duplicating numerical payloads."""
    for row in manifest.get("historical_input_files", []):
        path = _inside(row["path"], root)
        _same({**row, "path": str(path)})
        records[str(path)] = {**row, "path": str(path), "role": "preserved_historical_input"}
    comparison = manifest.get("policy_comparison")
    if not comparison:
        return
    folder = _inside(comparison["folder"], root)
    pins = comparison.get("input_files", [])
    expected = {folder / name for name in
        ("design_before_training.json", "manifest.json", "preservation_inventory.json")}
    if len(pins) != 3 or {_inside(row["path"], root) for row in pins} != expected:
        raise ValueError("Policy-comparison report must pin its exact design, manifest and closed inventory")
    for row in pins:
        path = _inside(row["path"], root)
        _same({**row, "path": str(path)})
        records[str(path)] = {**row, "path": str(path), "role": "sealed_policy_campaign_identity"}
    if comparison.get("design_sha256") != sha(folder / "design_before_training.json"):
        raise ValueError("Policy-comparison design differs from its report identity")
    inventory = json.loads((folder / "preservation_inventory.json").read_text())
    entries = inventory["files"]
    names = [row["path"] for row in entries]
    # The frozen campaign sealer omits this nested inventory by basename,
    # while sealing every pilot payload. Verify and pin its own closed tree
    # explicitly; do not permit arbitrary unlisted inventory-shaped files.
    nested = (["timing_pilot/preservation_inventory.json"]
              if any(name.startswith("timing_pilot/") for name in names) else [])
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(names) or
            inventory.get("entry_count") != len(names) or inventory.get("entries_sha256") != digest or
            actual != set(names) | {"preservation_inventory.json"} | set(nested)):
        raise ValueError("Policy-comparison evidence inventory is not closed")
    for row in entries:
        path = _inside(folder / row["path"], folder)
        _same({**row, "path": str(path)})
        records[str(path)] = {**row, "path": str(path), "role": "sealed_policy_campaign_input"}
    for relative in nested:
        path = folder / relative
        pilot = json.loads(path.read_text())
        rows = pilot["files"]
        names = [row["path"] for row in rows]
        actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob("*") if p.is_file()}
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(names) or
                pilot.get("entry_count") != len(names) or pilot.get("entries_sha256") != digest or
                actual != set(names) | {"preservation_inventory.json"}):
            raise ValueError("The nested timing-pilot inventory is not closed")
        for row in rows:
            source = _inside(path.parent / row["path"], path.parent)
            _same({**row, "path": str(source)})
        records[str(path)] = _record(path, role="sealed_nested_timing_pilot_inventory")


def validate_reports(root, *, family=FAMILY):
    """Validate current code, complete executions and every flat report artifact."""
    import nbformat
    if family not in REPORT_FAMILIES:
        raise ValueError("Unsupported first-stage report family")
    root = Path(root).resolve()
    folder = root / "notebooks/_output/pricing" / family
    records, reports = {}, []
    expected_interpreter = (root / ".venv/bin/python").absolute()
    for stem in STEMS:
        notebook_path = root / "notebooks/pricing" / f"{stem}.ipynb"
        notebook = nbformat.read(notebook_path, 4)
        manifest_path = folder / stem / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (notebook.metadata.get("rvlab_artifact_family") != family or
                manifest.get("artifact_family") != family or manifest.get("evidence_version") != REPORT_FAMILIES[family] or
                manifest.get("notebook") != stem or manifest.get("first_stage_report_complete") is not True):
            raise ValueError(f"Incomplete or incorrectly identified first-stage report: {stem}")
        execution = notebook.metadata.get("rvlab_execution", {})
        if (not execution.get("executed_at") or not execution.get("elapsed_seconds", 0) > 0 or
                Path(execution.get("interpreter", "")).absolute() != expected_interpreter or
                Path(manifest.get("interpreter", "")).absolute() != expected_interpreter):
            raise ValueError(f"A fresh project-interpreter execution is required: {stem}")
        if execution.get("artifact_manifest_sha256") != sha(manifest_path):
            raise ValueError(f"The fresh-kernel notebook does not bind the current artifact manifest: {stem}")
        for cell in notebook.cells:
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            if cell.execution_count is None or any(o.output_type == "error" for o in cell.get("outputs", [])):
                raise ValueError(f"Unexecuted or failed notebook cell: {stem}")
        sources = manifest.get("source_sha256", {})
        expected_sources = {str(p.relative_to(root)) for folder in
            (root / "src/rvlab/experiments", root / "src/rvlab/pricing") for p in folder.glob("*.py")}
        if not sources or set(sources) != expected_sources:
            raise ValueError(f"Incomplete execution source map: {stem}; a current module was added or omitted")
        for relative, digest in sources.items():
            source = _inside(root / relative, root)
            if not source.is_file() or sha(source) != digest:
                raise ValueError(f"Execution source is stale: {relative}; rerun all reports after source stability")
            records[str(source)] = _record(source, role="execution_source")
        if stem == STEMS[0] and manifest.get("rough_benchmark", {}).get("complete") is not True:
            raise ValueError("The bounded generic benchmark must be complete, including explicit recovery records")
        if (family == "first_stage_v2" and stem == STEMS[0] and
                manifest.get("policy_comparison", {}).get("complete") is not True):
            raise ValueError("The matched policy comparison must retain every declared trial")
        if stem == STEMS[1] and (manifest.get("optimizer_calls_this_report") != 0 or
                manifest.get("current_source_repricing_pass") is not True or
                manifest.get("current_calibration_recomputed") is not False):
            raise ValueError("Calibration replay and current-source repricing identities are inconsistent")
        if stem == STEMS[2] and (manifest.get("exercise_style") != "american" or
                manifest.get("backend") != "american_vanilla_fd" or
                manifest.get("calibration_ready") is not False or manifest.get("hedging_ready") is not False):
            raise ValueError("The published SPY bridge must retain its explicit American blocked-input state")
        if family == "first_stage_v2":
            _evidence_inputs(manifest, root=root, records=records)
        artifact_dir = folder / stem
        artifacts = manifest.get("artifact_sha256", {})
        expected_artifacts = {f"{name}.csv" for name in manifest.get("tables", {})} | {
            f"{name}.png" for name in manifest.get("figures", [])}
        if not artifacts or set(artifacts) != expected_artifacts:
            raise ValueError(f"An exact CSV/PNG execution digest map is required: {stem}")
        for name, digest in artifacts.items():
            if Path(name).name != name or sha(artifact_dir / name) != digest:
                raise ValueError(f"A saved report artifact changed after execution: {stem}/{name}")
        for name, rows in manifest.get("tables", {}).items():
            if Path(name).name != name or not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
                raise ValueError("Malformed declared report table")
            path = artifact_dir / f"{name}.csv"
            with path.open(newline="") as stream:
                count = max(sum(1 for _ in csv.reader(stream)) - 1, 0)
            if count != rows:
                raise ValueError(f"Declared table row count changed: {path}")
            records[str(path.resolve())] = _record(path, role="table")
        for name in manifest.get("figures", []):
            if Path(name).name != name:
                raise ValueError("Malformed declared report figure")
            path = artifact_dir / f"{name}.png"
            with path.open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"Invalid report PNG: {path}")
            records[str(path.resolve())] = _record(path, role="figure")
        for path, role in ((notebook_path, "executed_notebook"), (manifest_path, "report_manifest")):
            records[str(path.resolve())] = _record(path, role=role)
        reports.append({"stem": stem, "manifest": manifest, "execution": dict(execution),
                        "notebook": notebook, "notebook_path": notebook_path})
    return reports, records


def rewrite_link(url, *, notebook_dir, release_dir, root):
    """Resolve original notebook links before moving HTML into the release folder."""
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return url
    target = _inside(notebook_dir / unquote(parsed.path), root)
    if target.suffix == ".ipynb" and target.stem in STEMS and target.parent == notebook_dir:
        path = target.with_suffix(".html").name
    else:
        if not target.exists():
            raise ValueError(f"Broken local report link: {url}")
        path = quote(Path(os.path.relpath(target, release_dir)).as_posix(), safe="/-._~")
    return urlunsplit(("", "", path, parsed.query, parsed.fragment))


class _Links(HTMLParser):
    def __init__(self, **kwargs):
        super().__init__(convert_charrefs=False)
        self.kwargs, self.parts = kwargs, []

    def _tag(self):
        text = self.get_starttag_text()
        def replace(match):
            # Unescape entities for URL parsing; escape the rewritten attribute again.
            from html import unescape
            value = rewrite_link(unescape(match.group(3)), **self.kwargs)
            return match.group(1) + match.group(2) + escape(value, quote=True) + match.group(2)
        self.parts.append(re.sub(r"((?:href|src)\s*=\s*)([\"'])(.*?)\2", replace, text, flags=re.I | re.S))

    def handle_starttag(self, tag, attrs): self._tag()
    def handle_startendtag(self, tag, attrs): self._tag()
    def handle_endtag(self, tag): self.parts.append(f"</{tag}>")
    def handle_data(self, data): self.parts.append(data)
    def handle_entityref(self, name): self.parts.append(f"&{name};")
    def handle_charref(self, name): self.parts.append(f"&#{name};")
    def handle_comment(self, data): self.parts.append(f"<!--{data}-->")
    def handle_decl(self, decl): self.parts.append(f"<!{decl}>")
    def handle_pi(self, data): self.parts.append(f"<?{data}>")


def _html(report, *, root, destination):
    from nbconvert import HTMLExporter
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(report["notebook"])
    parser = _Links(notebook_dir=root / "notebooks/pricing", release_dir=destination, root=root)
    parser.feed(body)
    body = "".join(parser.parts)
    stem = report["stem"]
    note = ('<nav style="padding:1em 3em;line-height:1.6;border-bottom:1px solid #ccd6df">'
        '<a href="index.html">First-stage research report</a> · '
        f'<a href="snapshot/notebooks/pricing/{stem}.ipynb">Executed notebook</a> · '
        f'<a href="../{stem}/manifest.json">Execution and evidence manifest</a><br>'
        'Synthetic finite-law numerical research; historical evidence retains its original identity. '
        'Real SPY calibration and hedging remain blocked by independent stock/dividend inputs.</nav>')
    return re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + note, body, count=1)


def _snapshot_paths(root):
    paths = {root / relative for relative in ("CLAUDE.md", "Readme.md", "pyproject.toml",
        "notebooks/README.md", "notebooks/pricing/README.md", "notebooks/hedging/README.md",
        "notebooks/legacy/pricing/README.md") if (root / relative).is_file()}
    for folder, pattern in (("src/rvlab", "*.py"), ("scripts", "*.py"),
                            ("scripts", "*.sh"), ("examples", "*.py")):
        paths.update((root / folder).rglob(pattern))
    paths.update(root / "notebooks/pricing" / f"{stem}.ipynb" for stem in STEMS)
    return sorted(p for p in paths if p.is_file())


def _code_identity(notebook):
    return {"cells": [{"source": c.source, "execution_count": c.execution_count}
        for c in notebook.cells if c.cell_type == "code"],
        "execution": dict(notebook.metadata.get("rvlab_execution", {})),
        "family": notebook.metadata.get("rvlab_artifact_family")}


def build_release(root=ROOT, *, output_dir=None, test_reports=(), executed_dir=None, family=FAMILY):
    root = Path(root).resolve()
    if family not in REPORT_FAMILIES:
        raise ValueError("Unsupported first-stage report family")
    parent = root / "notebooks/_output/pricing" / family
    destination = Path(output_dir or parent / "release").resolve()
    if destination != parent / "release":
        raise ValueError(f"The release destination must be the {family}/release directory")
    if destination.exists():
        raise FileExistsError("A release is already present; verify it or preserve it before a new release")
    if not test_reports:
        raise ValueError("At least one recorded test report is required")
    if executed_dir is None:
        raise ValueError("The final fresh-kernel notebook directory is required")
    if Path(executed_dir).resolve() == root / "notebooks/pricing":
        raise ValueError("Fresh-kernel reference copies must be independent of the active notebook paths")
    passing_test_reports = []
    for path in map(Path, test_reports):
        text = path.read_text(errors="replace")
        if re.search(r"\b[1-9][0-9]* passed\b", text) and not re.search(r"\b[1-9][0-9]* failed\b", text):
            passing_test_reports.append(str(path.resolve()))
    if not passing_test_reports:
        raise ValueError("A recorded passing pytest report is required; a file alone is not a test result")
    reports, records = validate_reports(root, family=family)
    import nbformat
    execution_references = []
    for report in reports:
        path = Path(executed_dir) / f"{report['stem']}.ipynb"
        reference = nbformat.read(path, 4)
        if _code_identity(reference) != _code_identity(report["notebook"]):
            raise ValueError(f"Active code or execution identity differs from the fresh-kernel notebook: {report['stem']}")
        execution_references.append({"stem": report["stem"], "source": str(path.resolve()),
            "sha256": sha(path), "code_identity": _code_identity(reference)})
    destination.mkdir(parents=True)
    for report in reports:
        (destination / f"{report['stem']}.html").write_text(_html(report, root=root, destination=destination))
    for source in _snapshot_paths(root):
        target = destination / "snapshot" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = _record(source, role="source_docs_runner_snapshot")
        shutil.copy2(source, target)
        if sha(target) != before["sha256"]:
            raise ValueError(f"Source snapshot copy differs: {source}")
        records.setdefault(str(source.resolve()), before)
    report_records = []
    for row in execution_references:
        target = destination / "validation/fresh_kernel" / f"{row['stem']}.ipynb"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], target)
        if sha(target) != row["sha256"]:
            raise ValueError("Fresh-kernel reference changed while being copied")
        row["copy"] = str(target.relative_to(destination))
    for i, value in enumerate(test_reports):
        path = Path(value).resolve()
        record = _record(path, role="test_report")
        target = destination / "validation" / f"{i:02d}_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records[str(path)] = record
        report_records.append({**record, "copy": str(target.relative_to(destination))})
    environment = {"python": sys.version, "interpreter": sys.executable,
        "platform": platform.platform(), "thread_environment": {k: os.environ.get(k) for k in
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MPLBACKEND")},
        "versions": {name: metadata.version(name) for name in
        ("numpy", "scipy", "pandas", "matplotlib", "nbformat", "nbconvert")}}
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    links = "".join(f'<li><a href="{stem}.html">{escape(title)}</a></li>' for stem, title in zip(STEMS, TITLES))
    (destination / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>American pricing · first-stage research</title><body style="max-width:900px;margin:3em auto;'
        'font:17px/1.65 system-ui;padding:0 1em"><h1>American pricing, policy deltas and synthetic inference</h1>'
        '<p>A numerical-methods study of finite-factor rough-Heston pricing and frozen-policy stock sensitivities. '
        'The reports distinguish the model, discrete law, exercise rule, estimator and measured errors.</p>'
        f'<ol>{links}</ol><p><strong>Scope:</strong> completed experiments need not pass their numerical gates. '
        'Preserved focused and calibration records retain their original source, path and timing identities. '
        'Current-source calibration repricing performs no new optimization.</p>'
        '<p><strong>SPY is blocked:</strong> independent stock observations and historically available cash-dividend '
        'expectations are missing. These reports do not certify optimal continuous-American prices, Greeks or a portfolio hedge replay.</p>'
        '<p>Figures are embedded; CSVs, manifests and archived trial logs link to their original local locations. '
        'Keep this release inside its project tree. Large Monte Carlo payloads are not duplicated. '
        'MathJax rendering may require the standard notebook CDN.</p>'
        '<p><a href="snapshot/notebooks/pricing/README.md">Guide snapshot</a> · '
        '<a href="environment.json">Environment</a> · <a href="release_inventory.json">Hashes and verification scope</a></p></body></html>')
    for record in records.values():
        _same(record)
    files = [{"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(destination.rglob("*")) if p.is_file()]
    inventory = {"version": "pricing_first_stage_release_v1", "artifact_family": family,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "scope": "HTML, source/docs/runner snapshot, environment and test reports; large numerical payloads remain at their sealed original locations",
        "scientific_scope": "Finite-law empirical checks; no optimal-American or market/portfolio readiness promotion",
        "notebooks": [{"stem": r["stem"], "execution": r["execution"],
            "manifest_sha256": sha(parent / r["stem"] / "manifest.json")} for r in reports],
        "execution_references": execution_references, "passing_test_reports": passing_test_reports, "test_reports": report_records,
        "input_files": sorted(records.values(), key=lambda r: r["path"]), "files": files}
    (destination / "release_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    verify_release(destination)
    return inventory


def verify_release(destination):
    """Check a closed release and its live source/artifact inputs without execution."""
    destination = Path(destination).resolve()
    inventory = json.loads((destination / "release_inventory.json").read_text())
    if inventory.get("version") != "pricing_first_stage_release_v1":
        raise ValueError("Unsupported release inventory")
    declared = {r["path"] for r in inventory["files"]}
    actual = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
    if actual != declared | {"release_inventory.json"}:
        raise ValueError("Closed release inventory has missing or undeclared files")
    for row in inventory["files"]:
        path = _inside(destination / row["path"], destination)
        _same({**row, "path": str(path)})
    for row in inventory["input_files"]:
        _same(row)
    # Recheck current source maps and notebook execution validity, not only hashes
    # copied into the release's own inventory.
    family = inventory.get("artifact_family", FAMILY)
    reports, _ = validate_reports(Path(inventory["root"]), family=family)
    references = {r["stem"]: r for r in inventory["execution_references"]}
    if set(references) != set(STEMS):
        raise ValueError("Every final fresh-kernel execution reference is required")
    for report in reports:
        row = references[report["stem"]]
        if _code_identity(report["notebook"]) != row["code_identity"]:
            raise ValueError("Current notebook code differs from its final fresh-kernel reference")
    return {"passed": True, "files": len(declared), "inputs": len(inventory["input_files"]),
            "release_inventory_sha256": sha(destination / "release_inventory.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--test-report", type=Path, action="append", default=[])
    parser.add_argument("--executed-dir", type=Path, help="Final fresh-kernel notebook copies; required when building")
    parser.add_argument("--verify", action="store_true", help="Read-only verification of the existing closed release")
    parser.add_argument("--family", choices=tuple(REPORT_FAMILIES), default=FAMILY,
                        help="Explicit report generation; historical first_stage_v1 is never overwritten")
    args = parser.parse_args()
    destination = args.root.resolve() / "notebooks/_output/pricing" / args.family / "release"
    if args.verify:
        result = verify_release(destination)
    else:
        inventory = build_release(args.root, test_reports=args.test_report, executed_dir=args.executed_dir, family=args.family)
        result = {"built": str(destination), "files": len(inventory["files"]),
                  "release_inventory_sha256": sha(destination / "release_inventory.json")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
