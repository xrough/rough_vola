"""Closed HTML releases must reject stale executions and changed live evidence."""
import base64
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import nbformat
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/build_pricing_first_stage_release.py"
SPEC = importlib.util.spec_from_file_location("first_stage_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")


def _fixture(root, family=release.FAMILY):
    files = {"src/rvlab/experiments/example.py": "# report facade\n",
        "src/rvlab/pricing/example.py": "# tested numerical source\n",
        "notebooks/pricing/README.md": "# Guide\n", "CLAUDE.md": "Keep the historical source.\n",
        "scripts/runner.py": "# selected interpreter\n", "examples/demo.py": "# example\n"}
    for name, contents in files.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(contents)
    archived = root / "notebooks/legacy/pricing/history.md"
    archived.parent.mkdir(parents=True); archived.write_text("# Preserved failures\n")
    source_map = {name: release.sha(root / name) for name in files if name.startswith("src/")}
    comparison = {"complete": True}
    if family == "first_stage_v2":
        campaign = root / "notebooks/_output/pricing/first_stage_v2/policy_comparison_v1"
        campaign.mkdir(parents=True)
        (campaign / "design_before_training.json").write_text('{"predeclared":true}\n')
        (campaign / "manifest.json").write_text('{"complete":true}\n')
        (campaign / "paths.npz").write_bytes(b"immutable numerical payload fixture")
        pilot = campaign / "timing_pilot"
        pilot.mkdir()
        (pilot / "timing.json").write_text('{"runtime_only":true}\n')
        pilot_rows = [{"path": "timing.json", "bytes": (pilot / "timing.json").stat().st_size,
            "sha256": release.sha(pilot / "timing.json")}]
        pilot_canonical = json.dumps(pilot_rows, sort_keys=True, separators=(",", ":"))
        (pilot / "preservation_inventory.json").write_text(json.dumps({"files": pilot_rows,
            "entry_count": 1, "entries_sha256": hashlib.sha256(pilot_canonical.encode()).hexdigest()}))
        rows = [{"path": str(p.relative_to(campaign)), "bytes": p.stat().st_size, "sha256": release.sha(p)}
            for p in sorted(campaign.rglob("*")) if p.is_file() and p.name != "preservation_inventory.json"]
        canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        (campaign / "preservation_inventory.json").write_text(json.dumps({"files": rows,
            "entry_count": len(rows), "entries_sha256": hashlib.sha256(canonical.encode()).hexdigest()}))
        comparison.update(folder=str(campaign), design_sha256=release.sha(campaign / "design_before_training.json"),
            input_files=[{"path": str(campaign / name), "bytes": (campaign / name).stat().st_size,
                "sha256": release.sha(campaign / name)} for name in
                ("design_before_training.json", "manifest.json", "preservation_inventory.json")])
    for stem in release.STEMS:
        notebook = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell("# Research report\n[Guide](README.md) · "
                "[Other](02_synthetic_calibration.ipynb#inference) · [History](../legacy/pricing/history.md)"),
            nbformat.v4.new_code_cell("print('complete')", execution_count=1, outputs=[
                nbformat.v4.new_output("display_data", data={"text/html":
                    f'<a href="../_output/pricing/{family}/{stem}/table.csv">CSV</a>'
                    '<img alt="Fixture figure" src="data:image/png;base64,'+base64.b64encode(PNG).decode()+'">'})])])
        notebook.metadata.update(rvlab_artifact_family=family,
            rvlab_execution={"executed_at": "2026-10-20T09:00:00Z", "elapsed_seconds": 2.,
                "interpreter": str(root / ".venv/bin/python")})
        nbformat.write(notebook, root / "notebooks/pricing" / f"{stem}.ipynb")
        folder = root / "notebooks/_output/pricing" / family / stem
        folder.mkdir(parents=True)
        (folder / "table.csv").write_text("value\n1\n")
        (folder / "figure.png").write_bytes(PNG)
        manifest = {"artifact_family": family, "evidence_version": release.REPORT_FAMILIES[family],
            "notebook": stem, "first_stage_report_complete": True, "source_sha256": source_map,
            "tables": {"table": 1}, "figures": ["figure"],
            "interpreter": str(root / ".venv/bin/python"), "rough_benchmark": {"complete": True},
            "policy_comparison": comparison,
            "optimizer_calls_this_report": 0, "current_source_repricing_pass": True,
            "current_calibration_recomputed": False, "exercise_style": "american",
            "backend": "american_vanilla_fd", "calibration_ready": False, "hedging_ready": False}
        manifest['artifact_sha256'] = {name: release.sha(folder / name) for name in ('table.csv', 'figure.png')}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        notebook.metadata.rvlab_execution.artifact_manifest_sha256 = release.sha(folder / "manifest.json")
        nbformat.write(notebook, root / "notebooks/pricing" / f"{stem}.ipynb")
    fresh = root / "fresh_kernel"; fresh.mkdir()
    for path in (root / "notebooks/pricing").glob("*.ipynb"): shutil.copy2(path, fresh / path.name)
    # Numerical payloads stay external, regardless of extension or content size.
    (folder / "uncopied_paths.npz").write_bytes(b"large numerical payload stand-in")
    report = root / "test_result.log"; report.write_text("3125 passed, 1 skipped\n")
    return report


def test_build_and_readonly_verify_preserve_inputs_embed_figures_and_rewrite_links(tmp_path):
    report = _fixture(tmp_path)
    before = {str(p): release.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    inventory = release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel")
    destination = tmp_path / "notebooks/_output/pricing/first_stage_v1/release"
    result = release.verify_release(destination)
    assert result["passed"] and len(inventory["notebooks"]) == 3
    assert not list(destination.rglob("*.npz"))
    assert (destination / "snapshot/src/rvlab/pricing/example.py").is_file()
    for name, digest in before.items(): assert release.sha(Path(name)) == digest
    html = (destination / "01_pricing_and_exercise.html").read_text()
    assert 'href="../01_pricing_and_exercise/table.csv"' in html
    assert 'href="02_synthetic_calibration.html#inference"' in html
    assert 'href="../../../../legacy/pricing/history.md"' in html
    assert "data:image/png;base64," in html and "print('complete')" not in html
    assert "SPY is blocked" in (destination / "index.html").read_text()
    with pytest.raises(FileExistsError): release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel")


@pytest.mark.parametrize("change", ["source", "new_module", "family", "error", "unexecuted", "interpreter", "rows", "png", "spy_fit"])
def test_release_rejects_stale_sources_incomplete_cells_and_changed_artifact_contracts(tmp_path, change):
    report = _fixture(tmp_path)
    folder = tmp_path / "notebooks/_output/pricing/first_stage_v1"
    if change == "source": (tmp_path / "src/rvlab/pricing/example.py").write_text("# changed\n")
    elif change == "new_module": (tmp_path / "src/rvlab/experiments/new.py").write_text("# added\n")
    elif change in {"error", "unexecuted", "interpreter"}:
        path = tmp_path / "notebooks/pricing/01_pricing_and_exercise.ipynb"
        notebook = nbformat.read(path, 4)
        if change == "error": notebook.cells[1].outputs = [nbformat.v4.new_output("error", ename="Failure", evalue="trial", traceback=[])]
        if change == "unexecuted": notebook.cells[1].execution_count = None
        if change == "interpreter": notebook.metadata.rvlab_execution.interpreter = "/other/python"
        nbformat.write(notebook, path)
    elif change == "rows": (folder / release.STEMS[0] / "table.csv").write_text("value\n1\n2\n")
    elif change == "png": (folder / release.STEMS[0] / "figure.png").write_bytes(b"not a PNG")
    else:
        path = folder / (release.STEMS[2] if change == "spy_fit" else release.STEMS[0]) / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["calibration_ready" if change == "spy_fit" else "artifact_family"] = True if change == "spy_fit" else "research_v1"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel")


@pytest.mark.parametrize("where", ["published", "input", "extra"])
def test_verify_detects_changed_published_files_live_inputs_and_uninventoried_additions(tmp_path, where):
    report = _fixture(tmp_path); release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel")
    destination = tmp_path / "notebooks/_output/pricing/first_stage_v1/release"
    path = (destination / "index.html" if where == "published" else
        tmp_path / "notebooks/_output/pricing/first_stage_v1/01_pricing_and_exercise/table.csv" if where == "input"
        else destination / "unrecorded.txt")
    path.write_text("changed")
    with pytest.raises(ValueError): release.verify_release(destination)


def test_local_links_cannot_escape_or_silently_point_to_missing_evidence(tmp_path):
    _fixture(tmp_path)
    kwargs = {"notebook_dir": tmp_path / "notebooks/pricing", "release_dir": tmp_path / "release", "root": tmp_path}
    assert release.rewrite_link("https://arxiv.org/abs/1609.02108", **kwargs) == "https://arxiv.org/abs/1609.02108"
    assert release.rewrite_link("#claim", **kwargs) == "#claim"
    for target in ("missing.csv", "../../../outside.txt"):
        with pytest.raises(ValueError): release.rewrite_link(target, **kwargs)


def test_post_execution_notebook_code_edits_are_rejected_without_relying_on_output_counts(tmp_path):
    report = _fixture(tmp_path)
    path = tmp_path / "notebooks/pricing/01_pricing_and_exercise.ipynb"
    notebook = nbformat.read(path, 4); notebook.cells[1].source = "print('edited after execution')"
    nbformat.write(notebook, path)
    with pytest.raises(ValueError, match="differs from the fresh-kernel"):
        release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel")


def test_empty_or_failed_test_log_cannot_be_presented_as_passing_validation(tmp_path):
    report = _fixture(tmp_path); report.write_text("1 failed, 3125 passed\n")
    with pytest.raises(ValueError, match="passing pytest"):
        release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel")


def test_same_row_count_and_file_size_artifact_change_cannot_be_blessed_at_build_time(tmp_path):
    report = _fixture(tmp_path)
    path = tmp_path / 'notebooks/_output/pricing/first_stage_v1/01_pricing_and_exercise/table.csv'
    path.write_text('value\n9\n')
    with pytest.raises(ValueError, match='changed after execution'):
        release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / 'fresh_kernel')


def test_second_report_build_and_verification_use_its_explicit_family(tmp_path):
    report = _fixture(tmp_path, family="first_stage_v2")
    with pytest.raises((ValueError, FileNotFoundError)):
        release.validate_reports(tmp_path)
    inventory = release.build_release(tmp_path, test_reports=[report],
        executed_dir=tmp_path / "fresh_kernel", family="first_stage_v2")
    assert inventory["artifact_family"] == "first_stage_v2"
    destination = tmp_path / "notebooks/_output/pricing/first_stage_v2/release"
    assert release.verify_release(destination)["passed"]
    assert 'href="../01_pricing_and_exercise/table.csv"' in (destination / "01_pricing_and_exercise.html").read_text()
    assert not (tmp_path / "notebooks/_output/pricing/first_stage_v1").exists()


def test_second_report_cannot_publish_an_incomplete_policy_comparison(tmp_path):
    _fixture(tmp_path, family="first_stage_v2")
    path = tmp_path / "notebooks/_output/pricing/first_stage_v2/01_pricing_and_exercise/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["policy_comparison"]["complete"] = False
    path.write_text(json.dumps(manifest))
    notebook = tmp_path / "notebooks/pricing/01_pricing_and_exercise.ipynb"
    source = nbformat.read(notebook, 4)
    source.metadata.rvlab_execution.artifact_manifest_sha256 = release.sha(path)
    nbformat.write(source, notebook)
    with pytest.raises(ValueError, match="retain every declared trial"):
        release.validate_reports(tmp_path, family="first_stage_v2")


@pytest.mark.parametrize("change", ["payload", "manifest", "inventory", "extra", "nested_inventory", "extra_inventory"])
def test_second_release_pins_scientific_campaign_even_if_flat_report_bytes_are_unchanged(tmp_path, change):
    report = _fixture(tmp_path, family="first_stage_v2")
    release.build_release(tmp_path, test_reports=[report], executed_dir=tmp_path / "fresh_kernel", family="first_stage_v2")
    campaign = tmp_path / "notebooks/_output/pricing/first_stage_v2/policy_comparison_v1"
    name = {"payload": "paths.npz", "manifest": "manifest.json",
        "inventory": "preservation_inventory.json", "extra": "unrecorded.txt",
        "nested_inventory": "timing_pilot/preservation_inventory.json",
        "extra_inventory": "unlisted/preservation_inventory.json"}[change]
    (campaign / name).parent.mkdir(parents=True, exist_ok=True)
    (campaign / name).write_bytes(b"changed scientific evidence")
    destination = tmp_path / "notebooks/_output/pricing/first_stage_v2/release"
    with pytest.raises(ValueError):
        release.verify_release(destination)
