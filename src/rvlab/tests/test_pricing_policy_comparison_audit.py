"""Cold audit rejects coherent-but-wrong experiment records as well as tampering."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rvlab.experiments import _pricing_policy_comparison_audit as audit
from rvlab.experiments._pricing_policy_comparison_numeric import POLICY_NAMES
from rvlab.experiments._pricing_rough_benchmark import benchmark_cases


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def read(path):
    return json.loads(Path(path).read_text())


def reseal_training(folder, record):
    record["file_sha256"] = audit._file_map(folder, "training")
    record["manifest_sha256"] = audit._manifest_map(folder, "training")
    write(folder/"training_complete.json", record)


def reseal_attempts(folder, phase, record):
    record["manifest_sha256"] = audit._manifest_map(folder, phase)
    write(folder/(phase+"_complete.json"), record)


@pytest.fixture
def campaign(tmp_path):
    cases = []
    for i, case in enumerate(benchmark_cases()):
        cases.append({**case, "training_seeds": [100+10*i, 101+10*i],
            "valuation_seeds": [1000+10*i, 1001+10*i],
            "clock_training_paths": dict(zip(POLICY_NAMES, (8192, 4096, 512)))})
    design = {"created_at": "2026-10-21T00:00:00+00:00", "cases": cases,
        "training_paths": 2048, "valuation_paths_per_stream": 2048,
        "training_replicas": 2, "valuation_streams": 2,
        "arms": list(audit.ARMS), "policies": list(POLICY_NAMES),
        "refinement": {"training_seeds": [200, 201], "paths_per_stream": 512,
            "valuation_seeds": {"lookahead_order": [2000, 2001],
                "physical_steps": [2010, 2011], "kernel_factors": [2020, 2021]}}}
    write(tmp_path/"design_before_training.json", design)
    design_hash = audit._sha(tmp_path/"design_before_training.json")
    rows = []
    for key, spec in audit._training_specs(design).items():
        target = tmp_path/"training"/key
        (target/"policies").mkdir(parents=True)
        (target/"policies/coefficients.npz").write_bytes(b"outcome-free provenance fixture")
        saved = {"policy_set_fingerprint": key,
            "coefficients_sha256": audit._sha(target/"policies/coefficients.npz")}
        write(target/"policies/policies.json", saved)
        row = {**spec, "status": "completed", "design_sha256": design_hash,
            "started_at": "2026-10-21T00:01:00+00:00",
            "completed_at": "2026-10-21T00:02:00+00:00",
            "policy_set_fingerprint": key,
            "policies_json_sha256": audit._sha(target/"policies/policies.json")}
        write(target/"manifest.json", row)
        rows.append(row)
    training = {"created_at": "2026-10-21T00:03:00+00:00", "records": rows,
        "all_attempts_complete": True, "all_successful": True, "design_sha256": design_hash}
    reseal_training(tmp_path, training)
    training_hash = audit._sha(tmp_path/"training_complete.json")
    for phase in ("valuation", "refinement"):
        rows = []
        definitions = ([(f"{c['case_id']}_s{s}", {"case_id": c["case_id"], "stream": s,
            "seed": c["valuation_seeds"][s]}) for c in cases for s in range(2)] if phase == "valuation"
            else [(f"{name}_s{s}", {"name": name, "stream": s, "seed": seeds[s]})
                for name, seeds in design["refinement"]["valuation_seeds"].items() for s in range(2)])
        for key, spec in definitions:
            target = tmp_path/phase/key
            target.mkdir(parents=True)
            (target/"samples.npz").write_bytes(b"outcome-free sample hash fixture")
            row = {**spec, "status": "completed", "design_sha256": design_hash,
                "training_complete_sha256": training_hash,
                "started_at": "2026-10-21T00:04:00+00:00",
                "completed_at": "2026-10-21T00:05:00+00:00",
                "paths": 2048 if phase == "valuation" else 512,
                "sample_sha256": audit._sha(target/"samples.npz")}
            if phase == "valuation":
                (target/"trace.npz").write_bytes(b"outcome-free trace hash fixture")
                row["trace_sha256"] = audit._sha(target/"trace.npz")
                write(target/"trace_metadata.json", {"outcome_free_fixture": True})
                row["trace_metadata_sha256"] = audit._sha(target/"trace_metadata.json")
            else:
                names = ["base"] if spec["name"] == "lookahead_order" else ["base", "variant"]
                for prefix in names:
                    (target/(prefix+"_trace.npz")).write_bytes(b"outcome-free refinement trace")
                    write(target/(prefix+"_trace_metadata.json"), {"outcome_free_fixture": True})
                row["trace_sha256"] = {p.name: audit._sha(p) for p in target.glob("*trace*")}
            write(target/"manifest.json", row)
            rows.append(row)
        reseal_attempts(tmp_path, phase, {"created_at": "2026-10-21T00:06:00+00:00",
            "records": rows, "all_attempts_complete": True, "all_successful": True})
    return tmp_path, design


def test_exact_26_training_and_12_main_6_refinement_attempts(campaign):
    folder, design = campaign
    assert len(audit.audit_training(folder, design)["records"]) == 26
    assert len(audit.audit_attempts(folder, design, "valuation")["records"]) == 12
    assert len(audit.audit_attempts(folder, design, "refinement")["records"]) == 6


@pytest.mark.parametrize("defect", ["duplicate", "seed", "budget", "design", "chronology", "timezone", "snapshot_record"])
def test_coherently_resealed_training_errors_still_fail(campaign, defect):
    folder, design = campaign
    record = read(folder/"training_complete.json")
    if defect == "duplicate": record["records"][-1] = deepcopy(record["records"][0])
    else:
        row = record["records"][0]
        if defect == "seed": row["training_seed"] += 1
        if defect == "budget": row["training_paths"]["cashflow_lsm"] += 1
        if defect == "design": row["design_sha256"] = "wrong"
        if defect == "chronology": row["started_at"] = "2026-10-20T00:00:00+00:00"
        if defect == "timezone": row["started_at"] = "2026-10-21T00:01:00"
        if defect != "snapshot_record":
            key = next(iter(audit._training_specs(design)))
            write(folder/"training"/key/"manifest.json", row)
        else: row["unsealed_change"] = True
    reseal_training(folder, record)
    with pytest.raises(ValueError): audit.audit_training(folder, design)


@pytest.mark.parametrize("defect", ["changed_bytes", "extra_file", "extra_attempt", "missing_manifest_pin"])
def test_training_file_closure(campaign, defect):
    folder, design = campaign
    key = next(iter(audit._training_specs(design)))
    if defect == "changed_bytes": (folder/"training"/key/"policies/coefficients.npz").write_bytes(b"changed")
    elif defect == "extra_file": (folder/"training"/key/"unreported.txt").write_text("extra")
    elif defect == "extra_attempt":
        (folder/"training/extra_attempt").mkdir(); (folder/"training/extra_attempt/manifest.json").write_text("{}")
    else:
        record = read(folder/"training_complete.json"); record["manifest_sha256"].pop(next(iter(record["manifest_sha256"])))
        write(folder/"training_complete.json", record)
    with pytest.raises(ValueError): audit.audit_training(folder, design)


@pytest.mark.parametrize("phase", ["valuation", "refinement"])
@pytest.mark.parametrize("defect", ["duplicate", "seed", "design", "training_parent", "premature_paths", "path_count", "unclosed", "per_manifest_pin"])
def test_attempt_identity_independence_and_chronology(campaign, phase, defect):
    folder, design = campaign
    record = read(folder/(phase+"_complete.json"))
    row = record["records"][0]
    key = f"{row.get('case_id', row.get('name'))}_s{row['stream']}"
    if defect == "duplicate": record["records"][-1] = deepcopy(row)
    elif defect == "unclosed": record["all_attempts_complete"] = False
    elif defect == "per_manifest_pin":
        record["manifest_sha256"].pop(next(iter(record["manifest_sha256"])))
        write(folder/(phase+"_complete.json"), record)
        with pytest.raises(ValueError): audit.audit_attempts(folder, design, phase)
        return
    else:
        if defect == "seed": row["seed"] = design["cases"][0]["training_seeds"][0]
        elif defect == "design": row["design_sha256"] = "another design"
        elif defect == "training_parent": row["training_complete_sha256"] = "wrong"
        elif defect == "premature_paths": row["started_at"] = "2026-10-21T00:02:59+00:00"
        elif defect == "path_count": row["paths"] *= 2
        write(folder/phase/key/"manifest.json", row)
    reseal_attempts(folder, phase, record)
    with pytest.raises(ValueError): audit.audit_attempts(folder, design, phase)


def test_numerical_failure_with_empty_message_is_retained_not_silently_dropped(campaign):
    folder, design = campaign
    record = read(folder/"valuation_complete.json")
    row = record["records"][0]
    row.update(status="failed", error_type="FloatingPointError", message="", traceback="fixture traceback")
    record["all_successful"] = False
    key = f"{row['case_id']}_s{row['stream']}"
    write(folder/"valuation"/key/"manifest.json", row)
    reseal_attempts(folder, "valuation", record)
    checked = audit.audit_attempts(folder, design, "valuation")
    assert checked["records"][0]["status"] == "failed"
    assert checked["all_attempts_complete"] and not checked["all_successful"]


def test_success_flag_cannot_hide_failed_attempt(campaign):
    folder, design = campaign
    record = read(folder/"refinement_complete.json")
    row = record["records"][0]
    row.update(status="failed", error_type="ValueError", message="failed", traceback="fixture")
    write(folder/"refinement"/f"{row['name']}_s{row['stream']}"/"manifest.json", row)
    reseal_attempts(folder, "refinement", record)
    with pytest.raises(ValueError, match="hides a failure"):
        audit.audit_attempts(folder, design, "refinement")


@pytest.mark.parametrize("phase", ["valuation", "refinement"])
@pytest.mark.parametrize("defect", ["metadata", "extra_file", "missing_trace"])
def test_saved_trace_closure_and_metadata_hashes(campaign, phase, defect):
    folder, design = campaign
    record = read(folder/(phase+"_complete.json"))
    row = record["records"][0]
    key = f"{row.get('case_id', row.get('name'))}_s{row['stream']}"
    target = folder/phase/key
    metadata = "trace_metadata.json" if phase == "valuation" else "base_trace_metadata.json"
    trace = "trace.npz" if phase == "valuation" else "base_trace.npz"
    if defect == "metadata": write(target/metadata, {"changed": True})
    elif defect == "extra_file": (target/"unreported_trials.txt").write_text("extra")
    else: (target/trace).unlink()
    with pytest.raises(ValueError): audit.audit_attempts(folder, design, phase)


def test_source_closure_checks_old_hashes_live_bytes_and_snapshot(tmp_path, monkeypatch):
    live = tmp_path/"live"; saved = tmp_path/"campaign"
    historical = {}
    for i in range(63):
        name = f"old/numerical_{i}.py"
        path = live/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(i))
        historical[name] = audit._sha(path)
    write(live/audit.HISTORICAL_DESIGN, {"source_sha256": historical})
    pins = dict(historical)
    for name in audit.NEW_SOURCES:
        path = live/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        pins[name] = audit._sha(path)
    for name in pins:
        target = saved/"source"/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((live/name).read_bytes())
    monkeypatch.setattr(audit, "ROOT", live)
    design = {"source_sha256": pins}
    assert audit.audit_sources(saved, design)["source_count"] == 68
    changed = {**design, "source_sha256": dict(pins)}
    changed["source_sha256"][next(iter(historical))] = "changed historical hash"
    with pytest.raises(ValueError, match="historical"):
        audit.audit_sources(saved, changed)
    (saved/"source/extra.py").write_text("unsealed")
    with pytest.raises(ValueError, match="not closed"):
        audit.audit_sources(saved, design)
    (saved/"source/extra.py").unlink()
    (live/audit.NEW_SOURCES[0]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        audit.audit_sources(saved, design)
