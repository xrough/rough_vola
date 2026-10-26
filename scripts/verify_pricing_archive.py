#!/usr/bin/env python3
"""Verify the frozen notebook and local-result inventory after consolidation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("notebooks/legacy/pricing/preservation_manifest.json")


def verify_inventory(root, manifest):
    """Return explicit missing/changed records; additional new files are allowed."""
    root = Path(root).resolve()
    issues = []
    records = [(r["archived_path"], r) for r in manifest["archived_files"]]
    records += [(r["path"], r) for r in manifest["result_files"]]
    for relative, record in records:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            issues.append(f"Outside repository: {relative}")
        elif not path.is_file():
            issues.append(f"Missing: {relative}")
        elif path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            issues.append(f"Changed: {relative}")
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    issues = verify_inventory(args.root, manifest)
    for issue in issues:
        print(issue)
    if issues:
        print(f"Preservation check failed: {len(issues)} missing or changed files.")
        return 1
    print(f"Preserved: {len(manifest['archived_files'])} archived files and "
          f"{len(manifest['result_files'])} result files; all hashes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
