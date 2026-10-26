"""Check display-only refresh on copies while leaving published reports intact."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nbformat
from rvlab.experiments.pricing import NOTEBOOKS
from rvlab.experiments.pricing_presentation import refresh_presentations


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numerical_identity(notebook):
    return {"cells": [dict(cell) for cell in notebook.cells
        if cell.cell_type == "code" and not cell.metadata.get("rvlab_presentation")],
        "execution": dict(notebook.metadata.get("rvlab_execution", {}))}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("first_stage_v1", "first_stage_v2"), default="first_stage_v1")
    args = parser.parse_args()
    family = ROOT / "notebooks/_output/pricing" / args.family
    active = [ROOT / "notebooks/pricing" / name for name in NOTEBOOKS]
    watched = set(active)
    for name in NOTEBOOKS:
        folder = family / Path(name).stem
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("artifact_family") != args.family:
            raise ValueError("Refresh guard family differs from its report manifest")
        watched.add(manifest_path)
        watched.update(folder / name for name in manifest["artifact_sha256"])
    before = {str(path): sha(path) for path in watched}
    rows = []
    with tempfile.TemporaryDirectory(prefix="rvlab-first-stage-refresh-guard-") as temporary:
        copies = []
        for source in active:
            target = Path(temporary) / source.name
            shutil.copyfile(source, target)
            copies.append(target)
        identities = {p.name: numerical_identity(nbformat.read(p, 4)) for p in copies}
        refresh_presentations(copies)
        for path in copies:
            refreshed = nbformat.read(path, 4)
            if numerical_identity(refreshed) != identities[path.name]:
                raise ValueError("Display refresh changed a numerical cell or execution identity")
            if any(o.output_type == "error" for c in refreshed.cells for o in c.get("outputs", [])):
                raise ValueError("Display refresh produced a notebook error")
            rows.append({"notebook": path.name, "numerical_cells_and_metadata_identical": True,
                         "presentation": dict(refreshed.metadata["rvlab_presentation"])})
    if before != {str(path): sha(path) for path in watched}:
        raise ValueError("Copy-only refresh changed a published notebook or result artifact")
    result = {"verified_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "script_sha256": sha(Path(__file__)), "source": "independent temporary notebook copies",
        "published_notebooks_and_result_artifacts_unchanged": True,
        "watched_files": len(watched), "records": rows}
    (family / "validation/presentation_refresh_guard.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COPY-ONLY REFRESH GUARD PASSED", len(rows), "reports", flush=True)


if __name__ == "__main__":
    main()
