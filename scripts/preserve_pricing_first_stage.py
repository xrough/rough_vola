"""Preserve the executed pre-report research without duplicating old archives.

APFS clones have independent inodes; hash checks use bounded memory.  The
source set comes from the executed manifest rather than concurrently developed
new benchmark modules. Existing preservation files are never overwritten.
"""
from datetime import datetime, timezone
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20"
RESULTS = ROOT / "notebooks/_output/pricing/research_v1"
EXECUTION_SOURCE_DIFFERENCES = []
INDEX_NAME = "ARCHIVE_INDEX.md"


def validate_destinations(pairs):
    """Reject source/index collisions even on case-insensitive filesystems."""
    names = {"preservation_manifest.json".casefold(), INDEX_NAME.casefold()}
    for _source, relative in pairs:
        name = relative.as_posix().casefold()
        if name in names:
            raise ValueError(f"Case-insensitive preservation destination collision: {relative}")
        names.add(name)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources():
    pairs = {}
    for name in ("01_pricing_and_exercise", "02_synthetic_calibration", "03_spy_calibration"):
        pairs[ROOT / "notebooks/pricing" / (name + ".ipynb")] = Path(name + ".ipynb")
        folder = RESULTS / name
        pairs.update({p: Path("results") / name / p.relative_to(folder)
                      for p in folder.rglob("*") if p.is_file()})
    pairs.update({ROOT / src: Path(target) for src, target in (
        ("notebooks/pricing/README.md", "ORIGINAL_GUIDE.md"),
        ("notebooks/README.md", "ORIGINAL_NOTEBOOK_GUIDE.md"),
        ("CLAUDE.md", "CLAUDE.md"), ("Readme.md", "Readme.md"),
        ("pyproject.toml", "pyproject.toml"), (".gitignore", ".gitignore"),
        ("scripts/run_notebooks.sh", "scripts/run_notebooks.sh"),
        ("scripts/preserve_pricing_first_stage.py", "scripts/preserve_pricing_first_stage.py"))})
    for name in ("01_pricing_and_exercise", "02_synthetic_calibration", "03_spy_calibration"):
        manifest = json.loads((RESULTS / name / "manifest.json").read_text())
        for relative, digest in manifest["source_sha256"].items():
            path = ROOT / relative
            actual = sha(path) if path.is_file() else None
            if actual != digest:
                if name == "01_pricing_and_exercise":
                    raise ValueError(f"Current pricing execution source changed before preservation: {relative}")
                EXECUTION_SOURCE_DIFFERENCES.append({"notebook": name, "source": relative,
                    "recorded_execution_sha256": digest, "current_snapshot_sha256": actual,
                    "note": "The older notebook execution retains its original manifest; the current source snapshot is not relabelled as its executed version."})
            if actual is not None:
                pairs[path] = Path(relative)
    for path in (ROOT / "src/rvlab/tests").glob("test_pricing*.py"):
        if "benchmark" not in path.name and "first_stage" not in path.name:
            pairs[path] = path.relative_to(ROOT)
    for path in (ROOT / "examples").glob("rough*.py"):
        if "benchmark" not in path.name:
            pairs[path] = path.relative_to(ROOT)
    return sorted(pairs.items(), key=lambda item: str(item[0]))


def clone(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    fn = library.clonefile
    fn.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    fn.restype = ctypes.c_int
    if fn(os.fsencode(source), os.fsencode(destination), 0) == 0:
        return "apfs_clone"
    error = ctypes.get_errno()
    if error not in (errno.ENOTSUP, errno.EXDEV, errno.ENOSYS):
        raise OSError(error, os.strerror(error), str(destination))
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 512 * 1024**2:
        raise OSError("Not enough free space for guarded ordinary-copy fallback")
    shutil.copy2(source, destination)
    return "ordinary_copy"


def main():
    manifest_path = ARCHIVE / "preservation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("The pre-report preservation is already sealed")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rows, mechanisms = [], {"apfs_clone": 0, "ordinary_copy": 0, "verified_existing": 0}
    pairs = sources()
    validate_destinations(pairs)
    for index, (source, relative) in enumerate(pairs):
        before = sha(source)
        destination = ARCHIVE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            mechanism = "verified_existing"
        else:
            mechanism = clone(source, destination)
        if (source.stat().st_ino == destination.stat().st_ino or
                sha(destination) != before or sha(source) != before):
            raise ValueError(f"Preservation mismatch: {source}")
        rows.append({"source": str(source.relative_to(ROOT)), "preserved": str(relative),
                     "bytes": source.stat().st_size, "sha256": before})
        mechanisms[mechanism] += 1
        if index % 250 == 0:
            print(f"Preserved {index + 1} files", flush=True)
    for row in rows:
        if sha(ROOT / row["source"]) != row["sha256"]:
            raise ValueError(f"Source changed during preservation: {row['source']}")
    manifest = {"preserved_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Three executed research notebooks, complete current result trees and nested seals, executed pricing/experiment sources, pricing tests, examples, guides and runners before the first-stage report consolidation. Earlier archives remain separate and unchanged.",
        "copy_mechanism": mechanisms,
        "older_execution_source_differences": EXECUTION_SOURCE_DIFFERENCES, "files": rows}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (ARCHIVE / INDEX_NAME).write_text(
        "# Research before the first-stage report\n\n"
        "These three executed notebooks preserve the complete chronological numerical and calibration studies before presentation consolidation. "
        "Their results, failed trials, source identities and nested validation seals retain their original meaning. "
        "The active report is in `notebooks/pricing/`.\n\n"
        "`preservation_manifest.json` records every original and preserved path, byte count and SHA-256. "
        "Copies have independent inodes; source-before/copy/source-after hashes and a final source recheck agree. "
        "No old archive is copied into this archive or modified.\n")
    for row in rows:
        if sha(ARCHIVE / row["preserved"]) != row["sha256"]:
            raise ValueError(f"A generated preservation file replaced evidence: {row['preserved']}")
    print(json.dumps({"files": len(rows), "bytes": sum(row["bytes"] for row in rows),
        "mechanisms": mechanisms, "manifest_sha256": sha(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
