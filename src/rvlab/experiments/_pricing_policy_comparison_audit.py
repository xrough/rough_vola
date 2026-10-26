"""Cold provenance and phase-closure audit for the new policy comparison.

These guards inspect saved files only. They never generate paths, select a
candidate, change an estimator, or turn successful execution into accuracy.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from ._pricing_policy_comparison_numeric import POLICY_NAMES
from ._pricing_rough_benchmark import benchmark_cases


ROOT = Path(__file__).resolve().parents[3]
ARMS = ("matched_paths", "matched_allowance")
NEW_SOURCES = (
    "src/rvlab/experiments/_pricing_policy_comparison_numeric.py",
    "src/rvlab/experiments/_pricing_policy_comparison_run.py",
    "src/rvlab/experiments/_pricing_policy_comparison_stats.py",
    "src/rvlab/experiments/_pricing_policy_comparison_audit.py",
    "examples/rough_policy_comparison.py",
)
HISTORICAL_DESIGN = Path("notebooks/_output/pricing/first_stage_v1/01_pricing_and_exercise/"
    "rough_generic_benchmark_v1/serialization_repair_v1/design_before_replay.json")


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(folder, name):
    value = (Path(folder)/name).resolve()
    if not value.is_relative_to(Path(folder).resolve()):
        raise ValueError("audit path escapes its sealed root: "+name)
    return value


def _time(value):
    if not isinstance(value, str):
        raise ValueError("an explicit UTC timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("an explicit UTC timestamp is required") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("an explicit UTC timestamp is required")
    return parsed


def _same_record(actual, expected, label):
    if actual != expected:
        raise ValueError("phase record differs from its per-attempt manifest: "+label)


def _status(record):
    if record.get("status") not in ("completed", "failed"):
        raise ValueError("every attempt needs an identifiable completed/failed status")
    if record["status"] == "failed" and (
            not isinstance(record.get("error_type"), str) or not record["error_type"]
            or not isinstance(record.get("message"), str)
            or not isinstance(record.get("traceback"), str)):
        raise ValueError("failed numerical attempts must retain type/message/traceback")


def _design_cases(design):
    cases = design.get("cases", [])
    if ([c.get("case_id") for c in cases] != [c["case_id"] for c in benchmark_cases()]
            or design.get("training_replicas") != 2 or design.get("valuation_streams") != 2
            or tuple(design.get("arms", ())) != ARMS
            or tuple(design.get("policies", ())) != POLICY_NAMES):
        raise ValueError("the fixed six-case/two-replica experiment design changed")
    return cases


def _training_specs(design):
    result = {}
    for case in _design_cases(design):
        for arm in ARMS:
            for replica in range(2):
                key = f"{case['case_id']}_{arm}_r{replica}"
                counts = ({name: design["training_paths"] for name in POLICY_NAMES}
                    if arm == ARMS[0] else case["clock_training_paths"])
                result[key] = {"case_id": case["case_id"], "arm": arm, "replica": replica,
                    "training_seed": case["training_seeds"][replica], "training_paths": counts,
                    "kernel_refinement": False}
    case = design["cases"][4]
    for replica in range(2):
        result[f"refinement_kernel_r{replica}"] = {"case_id": case["case_id"],
            "arm": ARMS[0], "replica": replica,
            "training_seed": design["refinement"]["training_seeds"][replica],
            "training_paths": {name: design["training_paths"] for name in POLICY_NAMES},
            "kernel_refinement": True}
    return result


def _file_map(folder, subtree):
    folder = Path(folder)
    return {str(p.relative_to(folder)): _sha(p) for p in sorted((folder/subtree).rglob("*"))
        if p.is_file()}


def _manifest_map(folder, subtree):
    folder = Path(folder)
    return {str(p.relative_to(folder)): _sha(p) for p in sorted((folder/subtree).rglob("manifest.json"))
        if p.is_file()}


def audit_sources(folder, design):
    """Verify the old immutable 63 plus exactly five new numerical sources."""
    folder = Path(folder)
    historical = _read(ROOT/HISTORICAL_DESIGN)["source_sha256"]
    pins = design.get("source_sha256", {})
    if len(historical) != 63 or set(pins) != set(historical)|set(NEW_SOURCES):
        raise ValueError("the declared 68-source numerical closure changed")
    if any(pins[name] != digest for name, digest in historical.items()):
        raise ValueError("historical numerical source hashes changed")
    actual = {str(p.relative_to(folder/"source")) for p in (folder/"source").rglob("*")
        if p.is_file()}
    if actual != set(pins):
        raise ValueError("the numerical source snapshot is not closed")
    for name, digest in pins.items():
        if _sha(_inside(ROOT, name)) != digest or _sha(_inside(folder/"source", name)) != digest:
            raise ValueError("live or snapshotted numerical source changed: "+name)
    return {"source_count": len(pins), "all_live_and_snapshot_hashes_match": True}


def audit_training(folder, design):
    """Verify all 26 attempts, their identities, timestamps and complete files."""
    folder = Path(folder)
    specs = _training_specs(design)
    record = _read(folder/"training_complete.json")
    design_hash = _sha(folder/"design_before_training.json")
    if (record.get("all_attempts_complete") is not True or len(record.get("records", [])) != 26
            or record.get("design_sha256") != design_hash):
        raise ValueError("every declared training attempt must close before final valuation")
    disk_map = _file_map(folder, "training")
    if record.get("file_sha256") != disk_map:
        raise ValueError("training file inventory is not closed or its bytes changed")
    if record.get("manifest_sha256") != _manifest_map(folder, "training"):
        raise ValueError("training per-attempt manifest hashes changed")
    keys = {p.name for p in (folder/"training").iterdir() if p.is_dir()}
    if keys != set(specs):
        raise ValueError("training attempt identities are incomplete or duplicated")
    seen = set()
    stored = record["records"]
    for item in stored:
        _status(item)
        if item.get("kernel_refinement"):
            key = f"refinement_kernel_r{item.get('replica')}"
        else:
            key = f"{item.get('case_id')}_{item.get('arm')}_r{item.get('replica')}"
        if key not in specs or key in seen:
            raise ValueError("training attempt identities are incomplete or duplicated")
        seen.add(key)
        for name, expected in specs[key].items():
            if item.get(name) != expected:
                raise ValueError("training identity/seed/path budget changed: "+key+" / "+name)
        target = folder/"training"/key
        _same_record(_read(target/"manifest.json"), item, key)
        files = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
        allowed = {"manifest.json", "policies/policies.json", "policies/coefficients.npz"}
        if not files <= allowed or (item["status"] == "completed" and files != allowed):
            raise ValueError("unexpected or missing fitted-policy artifact: "+key)
        if item.get("design_sha256") != design_hash:
            raise ValueError("training attempt identifies another experiment design")
        if not (_time(design["created_at"]) <= _time(item["started_at"])
                <= _time(item["completed_at"]) <= _time(record["created_at"])):
            raise ValueError("training chronology changed")
        if item["status"] == "completed":
            saved = _read(target/"policies/policies.json")
            if (item.get("policies_json_sha256") != _sha(target/"policies/policies.json")
                    or item.get("policy_set_fingerprint") != saved.get("policy_set_fingerprint")
                    or saved.get("coefficients_sha256") != _sha(target/"policies/coefficients.npz")):
                raise ValueError("saved fitted-policy identity changed: "+key)
    if seen != set(specs):
        raise ValueError("not every declared training attempt is present")
    if record.get("all_successful") != all(r["status"] == "completed" for r in stored):
        raise ValueError("training success flag hides a failure")
    return record


def audit_attempts(folder, design, phase):
    """Verify the exact 12 main or six refinement attempts and parent links."""
    if phase not in ("valuation", "refinement"):
        raise ValueError("phase must be valuation or refinement")
    folder = Path(folder)
    training = audit_training(folder, design)
    record = _read(folder/(phase+"_complete.json"))
    specs = {}
    if phase == "valuation":
        for case in _design_cases(design):
            for stream in range(2):
                specs[f"{case['case_id']}_s{stream}"] = {"case_id": case["case_id"],
                    "stream": stream, "seed": case["valuation_seeds"][stream]}
    else:
        if set(design["refinement"]["valuation_seeds"]) != {
                "lookahead_order", "physical_steps", "kernel_factors"}:
            raise ValueError("the three declared refinement axes changed")
        for name, seeds in design["refinement"]["valuation_seeds"].items():
            if len(seeds) != 2:
                raise ValueError("each refinement needs exactly two independent streams")
            for stream in range(2):
                specs[f"{name}_s{stream}"] = {"name": name, "stream": stream, "seed": seeds[stream]}
    if (record.get("all_attempts_complete") is not True
            or len(record.get("records", [])) != len(specs)):
        raise ValueError("every declared "+phase+" attempt must be retained")
    if record.get("manifest_sha256") != _manifest_map(folder, phase):
        raise ValueError(phase+" per-attempt manifest hashes changed")
    keys = {p.name for p in (folder/phase).iterdir() if p.is_dir()}
    if keys != set(specs):
        raise ValueError(phase+" attempt identities are incomplete or duplicated")
    design_hash = _sha(folder/"design_before_training.json")
    training_hash = _sha(folder/"training_complete.json")
    seen = set()
    for item in record["records"]:
        _status(item)
        key = (f"{item.get('case_id')}_s{item.get('stream')}" if phase == "valuation"
            else f"{item.get('name')}_s{item.get('stream')}")
        if key not in specs or key in seen:
            raise ValueError(phase+" attempt identities are incomplete or duplicated")
        seen.add(key)
        if any(item.get(k) != value for k, value in specs[key].items()):
            raise ValueError(phase+" identity or independent stream seed changed")
        if item.get("design_sha256") != design_hash or item.get("training_complete_sha256") != training_hash:
            raise ValueError(phase+" attempt parent design/training closure changed")
        target = folder/phase/key
        _same_record(_read(target/"manifest.json"), item, key)
        trace_names = ({"trace.npz", "trace_metadata.json"} if phase == "valuation" else
            {"base_trace.npz", "base_trace_metadata.json"} | (
                set() if item["name"] == "lookahead_order" else
                {"variant_trace.npz", "variant_trace_metadata.json"}))
        files = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
        expected_files = {"manifest.json", "samples.npz"}|trace_names
        if not files <= expected_files or (item["status"] == "completed" and files != expected_files):
            raise ValueError("unexpected or missing saved path artifact: "+key)
        if not (_time(training["created_at"]) <= _time(item["started_at"])
                <= _time(item["completed_at"]) <= _time(record["created_at"])):
            raise ValueError("final paths were drawn before all training closed, or chronology changed")
        if item["status"] == "completed":
            if item.get("sample_sha256") != _sha(target/"samples.npz"):
                raise ValueError(phase+" observations changed")
            if item.get("paths") != (design["valuation_paths_per_stream"] if phase == "valuation"
                    else design["refinement"]["paths_per_stream"]):
                raise ValueError(phase+" path budget changed")
            if phase == "valuation" and item.get("trace_sha256") != _sha(target/"trace.npz"):
                raise ValueError("saved valuation trace changed")
            if phase == "valuation" and item.get("trace_metadata_sha256") != _sha(target/"trace_metadata.json"):
                raise ValueError("saved valuation trace metadata changed")
            if phase == "refinement" and item.get("trace_sha256") != {
                    name: _sha(target/name) for name in sorted(trace_names)}:
                raise ValueError("saved refinement traces or metadata changed")
    if seen != set(specs):
        raise ValueError("not every declared "+phase+" attempt is present")
    if record.get("all_successful") != all(r["status"] == "completed" for r in record["records"]):
        raise ValueError(phase+" success flag hides a failure")
    return record


__all__ = ["audit_sources", "audit_training", "audit_attempts"]
