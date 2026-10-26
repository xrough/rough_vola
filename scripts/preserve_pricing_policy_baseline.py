"""Seal the executed first report before the matched exercise-policy study.

The complete previous release and all small report/campaign records are copied.
Large NPZ payloads remain at their immutable original locations and are listed
with streamed hashes.  No hard links or case-insensitive destination collisions
are permitted.  Historical execution manifests retain their original identity.
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
RESULTS = ROOT / "notebooks/_output/pricing/first_stage_v1"
ARCHIVE = ROOT / "notebooks/legacy/pricing/first_stage_policy_baseline_2026-10-21"
STEMS = ("01_pricing_and_exercise", "02_synthetic_calibration", "03_spy_calibration")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clone(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    function = library.clonefile
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 0) == 0:
        return "apfs_clone"
    error = ctypes.get_errno()
    if error not in (errno.ENOTSUP, errno.EXDEV, errno.ENOSYS):
        raise OSError(error, os.strerror(error), str(destination))
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 512 * 1024**2:
        raise OSError("Insufficient free space for ordinary-copy fallback")
    shutil.copy2(source, destination)
    return "ordinary_copy"


def files():
    pairs, external, execution_maps = {}, [], {}
    for stem in STEMS:
        notebook = ROOT / "notebooks/pricing" / f"{stem}.ipynb"
        pairs[notebook] = notebook.relative_to(ROOT)
        manifest = json.loads((RESULTS / stem / "manifest.json").read_text())
        execution_maps[stem] = manifest["source_sha256"]
        for relative, expected in manifest["source_sha256"].items():
            source = ROOT / relative
            if sha(source) != expected:
                raise ValueError(f"Previous execution source already changed: {relative}")
            pairs[source] = Path(relative)
    for source in RESULTS.rglob("*"):
        if not source.is_file():
            continue
        if source.suffix == ".npz":
            external.append({"path": str(source.resolve()), "bytes": source.stat().st_size,
                             "sha256": sha(source), "copied": False,
                             "role": "immutable numerical payload; retained at original location"})
        else:
            pairs[source] = Path("results/first_stage_v1") / source.relative_to(RESULTS)
    for relative in ("CLAUDE.md", "Readme.md", "pyproject.toml", ".gitignore",
                     "notebooks/README.md", "notebooks/pricing/README.md",
                     "notebooks/legacy/pricing/README.md", "notebooks/hedging/README.md"):
        source = ROOT / relative
        if source.is_file():
            pairs[source] = Path(relative)
    for directory, pattern in (("scripts", "*.py"), ("scripts", "*.sh"),
                               ("examples", "*.py"), ("src/rvlab/tests", "test_pricing*.py")):
        for source in (ROOT / directory).glob(pattern):
            pairs[source] = source.relative_to(ROOT)
    reserved = {"archive_index.md", "preservation_manifest.json"}
    names = set(reserved)
    for relative in pairs.values():
        name = relative.as_posix().casefold()
        if name in names:
            raise ValueError(f"Case-insensitive archive destination collision: {relative}")
        names.add(name)
    return sorted(pairs.items()), external, execution_maps


def verify():
    """Verify preserved bytes and immutable external payloads, not current code."""
    seal = ARCHIVE / "preservation_manifest.json"
    manifest = json.loads(seal.read_text())
    if manifest.get("version") != "pricing_policy_baseline_archive_v1":
        raise ValueError("Unsupported policy baseline archive")
    rows = manifest["files"]
    declared = {r["preserved"] for r in rows}
    if len(declared) != len(rows) or len({p.casefold() for p in declared}) != len(rows):
        raise ValueError("Archive inventory contains duplicate or case-folded paths")
    actual = {str(p.relative_to(ARCHIVE)) for p in ARCHIVE.rglob("*") if p.is_file()}
    if actual != declared | {"preservation_manifest.json", "ARCHIVE_INDEX.md"}:
        raise ValueError("Archive has missing or undeclared files")
    for row in rows:
        path = (ARCHIVE / row["preserved"]).resolve()
        if not path.is_relative_to(ARCHIVE.resolve()):
            raise ValueError("Archived path escapes preservation directory")
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Archived evidence changed: {path}")
    for row in manifest["external_numerical_payloads"]:
        path = Path(row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Immutable external payload changed: {path}")
    return {"passed": True, "files": len(rows),
            "external_payloads": len(manifest["external_numerical_payloads"]),
            "manifest_sha256": sha(seal)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify preserved historical evidence without comparing it with new current code")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), indent=2))
        return
    seal = ARCHIVE / "preservation_manifest.json"
    if seal.exists():
        raise FileExistsError("The previous report archive is already sealed")
    pairs, external, execution_maps = files()
    release = json.loads((RESULTS / "release/release_inventory.json").read_text())
    declared = {r["path"] for r in release["files"]}
    actual = {str(p.relative_to(RESULTS / "release")) for p in (RESULTS / "release").rglob("*") if p.is_file()}
    if actual != declared | {"release_inventory.json"}:
        raise ValueError("Prior release is not a closed inventory")
    for row in release["files"]:
        source = RESULTS / "release" / row["path"]
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"Prior sealed release changed: {source}")
    rows, mechanisms = [], {}
    for index, (source, relative) in enumerate(pairs):
        before = sha(source)
        destination = ARCHIVE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        mechanism = "verified_existing" if destination.exists() else clone(source, destination)
        if (source.stat().st_ino == destination.stat().st_ino or
                sha(destination) != before or sha(source) != before):
            raise ValueError(f"Preservation mismatch: {source}")
        rows.append({"source": str(source.relative_to(ROOT)), "preserved": str(relative),
                     "bytes": source.stat().st_size, "sha256": before})
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
        if index % 200 == 0:
            print(f"Preserved {index + 1}/{len(pairs)} files", flush=True)
    for row in rows:
        if sha(ROOT / row["source"]) != row["sha256"]:
            raise ValueError(f"Source changed during preservation: {row['source']}")
    for row in external:
        source = Path(row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError(f"External numerical payload changed during preservation: {source}")
    data = {"version": "pricing_policy_baseline_archive_v1",
        "preserved_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Three executed first-stage reports, all non-NPZ first_stage_v1 records, entire closed prior release, original execution source maps, guides, runners, examples and pricing tests before the matched exercise-policy comparison.",
        "historical_execution_identity": "The copied execution manifests retain their original source maps. The historical release remains byte-preserved; its live-current-source verifier will intentionally become stale when new sources/notebooks are introduced. Its closed snapshot remains verifiable independently.",
        "copy_mechanism": mechanisms, "execution_source_maps": execution_maps,
        "prior_release_inventory_sha256": sha(RESULTS / "release/release_inventory.json"),
        "prior_release_closed_inventory_pass": True, "external_numerical_payloads": external,
        "files": rows}
    seal.write_text(json.dumps(data, indent=2) + "\n")
    (ARCHIVE / "ARCHIVE_INDEX.md").write_text(
        "# First-stage report before the matched policy comparison\n\n"
        "This archive preserves the three executed reports and their complete flat outputs, the closed presentation release, execution sources and supporting records before the new comparison. Original numerical NPZ payloads remain at their immutable locations; their byte counts and hashes are recorded in the manifest.\n\n"
        "The release snapshot records its historical execution identity. Its live-current-source verification becomes stale after new development; this is distinct from the byte-integrity of its preserved closed snapshot. No old results are relabelled as a new experiment.\n\n"
        "`preservation_manifest.json` records every independent-inode copy and external payload, with source-before/copy/source-after and final-source streamed SHA-256 checks.\n")
    print(json.dumps({"files": len(rows), "copied_bytes": sum(r["bytes"] for r in rows),
        "external_payloads": len(external), "external_bytes": sum(r["bytes"] for r in external),
        "mechanisms": mechanisms, "manifest_sha256": sha(seal)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
