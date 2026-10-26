"""The preservation check must detect changed and missing historical evidence."""
import hashlib
import importlib.util
from pathlib import Path


def test_archive_verifier_reports_missing_and_changed_files(tmp_path):
    script = Path(__file__).resolve().parents[3] / "scripts/verify_pricing_archive.py"
    spec = importlib.util.spec_from_file_location("verify_pricing_archive", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = b"recorded evidence"
    digest = hashlib.sha256(original).hexdigest()
    archived = tmp_path / "old.ipynb"
    result = tmp_path / "prices.csv"
    archived.write_bytes(original)
    result.write_bytes(original)
    manifest = {"archived_files": [{"archived_path": archived.name,
        "bytes": len(original), "sha256": digest}], "result_files": [
        {"path": result.name, "bytes": len(original), "sha256": digest}]}
    assert module.verify_inventory(tmp_path, manifest) == []
    # Same-sized corruption must be detected by the digest, not only size.
    archived.write_bytes(b"x"*len(original))
    result.unlink()
    assert module.verify_inventory(tmp_path, manifest) == ["Changed: old.ipynb", "Missing: prices.csv"]
    manifest["result_files"][0]["path"] = "../outside.csv"
    assert module.verify_inventory(tmp_path, manifest)[1] == "Outside repository: ../outside.csv"
