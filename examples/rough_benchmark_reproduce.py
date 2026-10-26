"""Cold, sequential reproduction of the predefined generic rough benchmark.

Use a NEW first_stage_v1/cold_* destination.  The approved design JSON alone
supplies configurations and kernel coefficients; hand-generated guard folders
are not required.  The same seeds reproduce numerical evidence and do not
provide new independent validation.  No original or repaired output is changed.

    ... prepare --output-dir notebooks/_output/pricing/first_stage_v1/cold_demo
    ... train --all --output-dir notebooks/_output/pricing/first_stage_v1/cold_demo
    ... value --all --output-dir notebooks/_output/pricing/first_stage_v1/cold_demo
    ... collect --output-dir notebooks/_output/pricing/first_stage_v1/cold_demo

One exclusive phase lock enforces sequential execution. A crash leaves its lock
visible; this script never silently unlocks, retries or overwrites an attempt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np

from rvlab.experiments import _pricing_rough_benchmark as b
from rvlab.experiments._pricing_rough_benchmark_serialization import save_benchmark_policy

ALLOWED_ROOT = ROOT/"notebooks/_output/pricing/first_stage_v1"
APPROVED_DESIGN = ALLOWED_ROOT/"01_pricing_and_exercise/rough_generic_benchmark_v1/independent_v1/design_before_training.json"
APPROVED_SHA = "9903db3a1a93b46fb043fe174af270509ad128c575d88979a2ee9ca80e14f1c2"
VERSION = "rough_generic_benchmark_cold_reproduction_v1"
NEW_SOURCES = ("src/rvlab/experiments/_pricing_rough_benchmark_serialization.py",
    "examples/rough_benchmark_reproduce.py")
ROLE = {"same_seed_numerical_reproduction": True, "new_independent_validation": False,
    "eligible_for_report_acceptance": False, "acceptance_evidence": False}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def destination(value):
    path = Path(value)
    path = (ROOT/path if not path.is_absolute() else path).resolve()
    if path.parent != ALLOWED_ROOT.resolve() or not re.fullmatch(r"cold_[A-Za-z0-9][A-Za-z0-9_-]*", path.name):
        raise ValueError("choose a direct first_stage_v1/cold_<label> destination")
    return path


def original_design(path):
    path = Path(path)
    if b.sha256(path) != APPROVED_SHA:
        raise ValueError("cold reproduction requires the exact approved design JSON")
    design = json.loads(path.read_text())
    definitions = b.benchmark_trials(include_joint=True)
    if (len(design["trials"]) != 10 or len(design["source_sha256"]) != 61
            or design["parameters"] != b.PARAMETERS or design["trial_count"] != 10
            or design["total_valuation_paths"] != 40960):
        raise ValueError("the original six cases, four refinements or fixed budget changed")
    training = {r["policy_id"]: 1146005042+3000000*i for i, r in enumerate(definitions)
        if r["policy_id"] == r["trial_id"]}
    for i, (row, definition) in enumerate(zip(design["trials"], definitions)):
        expected = {**definition, "index": i, "training_seed": training[definition["policy_id"]],
            "training_paths": 2048, "valuation_paths": 2048,
            "valuation_seeds": [1147005042+3000000*i, 1148005042+3000000*i]}
        if row != expected:
            raise ValueError("a reproduction case, policy, seed or numerical budget changed")
    return design


def source_pins(original):
    pins = dict(original["source_sha256"])
    for name, digest in pins.items():
        if b.sha256(ROOT/name) != digest:
            raise ValueError("a historical numerical source changed: "+name)
    for name in NEW_SOURCES:
        pins[name] = b.sha256(ROOT/name)
    return pins


def environment():
    return {"python": sys.version, "interpreter": sys.executable, "platform": sys.platform,
        "packages": {name: metadata.version(name) for name in ("numpy", "scipy", "pandas")},
        "thread_and_plot_environment": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "MPLBACKEND")}}


def prepare(folder, approved_design):
    original = original_design(approved_design)
    pins = source_pins(original)
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(approved_design, folder/"approved_design.json")
    declaration = {"version": VERSION, "created_at": utc(), "source_sha256": pins,
        "approved_design_sha256": APPROVED_SHA, "input_sha256": {"approved_design.json": APPROVED_SHA},
        "trials": original["trials"], "kernels": original["kernels"],
        "source_configuration_only": True, "hand_generated_guard_folders_required": False,
        "sequential_execution": True, "maximum_workers": 1, "safe_boolean_serializer": True,
        "all_training_before_any_valuation": True, "all_failures_retained": True,
        "no_silent_retry_or_overwrite": True, "valuation_paths": 40960,
        "software_environment": environment(),
        "no_empirical_valuation_recentering": True, **ROLE, **b.TOLERANCES}
    write(folder/"design_before_reproduction.json", declaration)
    for name in pins:
        target = folder/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    print("COLD PREPARED", folder, "same seeds; not new independent validation", flush=True)


def load(folder):
    design = json.loads((folder/"design_before_reproduction.json").read_text())
    if design["version"] != VERSION or any(design.get(k) is not v for k, v in ROLE.items()):
        raise ValueError("cold reproduction role or version changed")
    for key, val in b.TOLERANCES.items():
        if design.get(key) != val or isinstance(val, bool) and design.get(key) is not val:
            raise ValueError("cold numerical tolerance declarations changed")
    if (design.get("valuation_paths") != 40960 or design.get("maximum_workers") != 1
            or design.get("safe_boolean_serializer") is not True
            or design.get("all_training_before_any_valuation") is not True
            or design.get("sequential_execution") is not True):
        raise ValueError("cold fixed budget, serializer or phase declarations changed")
    if design.get("software_environment") != environment():
        raise ValueError("cold software or configured thread environment changed between phases")
    original = original_design(folder/"approved_design.json")
    if source_pins(original) != design["source_sha256"]:
        raise ValueError("source code changed after cold predeclaration")
    for name, digest in design["source_sha256"].items():
        if b.sha256(folder/"source"/name) != digest:
            raise ValueError("cold numerical source snapshot changed")
    if design["trials"] != original["trials"] or design["kernels"] != original["kernels"]:
        raise ValueError("cold numerical configurations or kernels changed")
    return design


@contextmanager
def phase_lock(folder):
    if (folder/"manifest.json").exists():
        raise FileExistsError("this cold reproduction is already closed")
    lock = folder/"reproduction.lock"
    with lock.open("x") as handle:
        json.dump({"created_at": utc(), "role": "exclusive sequential phase; no automatic stale-lock deletion"}, handle)
    try:
        yield
    finally:
        lock.unlink()


def begin():
    return utc(), time.perf_counter(), time.process_time()


def finish(record, started, wall, cpu, design, folder):
    return {**record, "started_at": started, "completed_at": utc(),
        "elapsed_seconds": time.perf_counter()-wall, "process_cpu_seconds": time.process_time()-cpu,
        "source_sha256": design["source_sha256"],
        "design_sha256": b.sha256(folder/"design_before_reproduction.json"), **ROLE}


def failed(exc):
    return {"status": "failed", "error_type": type(exc).__name__,
        "message": str(exc), "traceback": traceback.format_exc()}


def train(folder, index):
    design = load(folder)
    trial = design["trials"][index]
    if trial["trial_id"] != trial["policy_id"]:
        raise ValueError("physical refinement reuses its original trained rule")
    target = folder/f"training_{index}"
    target.mkdir(exist_ok=False)
    stamp, wall, cpu = begin()
    try:
        config = b.benchmark_config(training_seed=trial["training_seed"],
            valuation_seed=trial["valuation_seeds"][0], steps=trial["steps"],
            exercise_steps=trial["exercise_steps"], training_paths=2048, valuation_paths=2048)
        policy = b.train_benchmark_policy(trial,
            b.ExponentialKernel(**design["kernels"][str(trial["factors"])]) , config)
        record = {"status": "completed", **save_benchmark_policy(policy, target/"policy")}
    except Exception as exc:
        record = failed(exc)
    load(folder)
    write(target/"manifest.json", finish({**record, "index": index,
        "trial_id": trial["trial_id"], "training_seed": trial["training_seed"],
        "training_paths": 2048}, stamp, wall, cpu, design, folder))
    print("COLD TRAIN ATTEMPT", index, record["status"], flush=True)


def training_records(folder, design):
    records = {}
    for index, trial in enumerate(design["trials"]):
        if trial["trial_id"] != trial["policy_id"]:
            continue
        path = folder/f"training_{index}/manifest.json"
        if not path.is_file():
            raise ValueError("all nine training attempts must finish before any valuation")
        record = json.loads(path.read_text())
        if (record["index"] != index or record["training_seed"] != trial["training_seed"]
                or record["training_paths"] != 2048 or record["source_sha256"] != design["source_sha256"]
                or record["design_sha256"] != b.sha256(folder/"design_before_reproduction.json")
                or record["status"] not in ("completed", "failed")):
            raise ValueError("cold training identity or source changed")
        if record["status"] == "completed":
            if (b.sha256(path.parent/"policy/policy.json") != record["policy_sha256"]
                    or b.sha256(path.parent/"policy/coefficients.npz") != record["coefficient_sha256"]):
                raise ValueError("cold training policy seal changed")
        records[trial["policy_id"]] = (index, record)
    return records


def value(folder, index, stream):
    design = load(folder)
    trained = training_records(folder, design)
    trial = design["trials"][index]
    policy_index, training = trained[trial["policy_id"]]
    target = folder/f"trial_{index}_stream_{stream}"
    target.mkdir(exist_ok=False)
    stamp, wall, cpu = begin()
    if training["status"] == "failed":
        record = {"status": "blocked_training_failure", "training_failure": training}
    else:
        try:
            policy = b.load_policy(folder/f"training_{policy_index}/policy")
            config = replace(policy.config, valuation_paths=2048,
                valuation_seed=trial["valuation_seeds"][stream], steps=trial["steps"])
            arrays, diagnostics = b.value_benchmark_policy(policy, config)
            np.savez_compressed(target/"samples.npz", **arrays)
            record = {"status": "completed", "sample_sha256": b.sha256(target/"samples.npz"),
                "array_schema": {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()},
                "diagnostics": diagnostics, "policy_fingerprint": training["policy_fingerprint"],
                "all_paths_retained": True}
        except Exception as exc:
            record = failed(exc)
    load(folder)
    write(target/"manifest.json", finish({**record, "index": index, "stream_index": stream,
        "trial_id": trial["trial_id"], "seed": trial["valuation_seeds"][stream], "paths": 2048,
        "training_manifest_sha256": b.sha256(folder/f"training_{policy_index}/manifest.json")},
        stamp, wall, cpu, design, folder))
    print("COLD VALUATION ATTEMPT", index, stream, record["status"], flush=True)


def collect(folder):
    from rvlab.experiments._pricing_rough_benchmark_audit import summarize_benchmark, validate_benchmark_samples
    design = load(folder)
    if list(folder.glob("*.csv")):
        raise FileExistsError("cold collection outputs already exist; no partial overwrite is permitted")
    trained = training_records(folder, design)
    streams, specs, records, probes = [], [], [], []
    last_training = max(row["completed_at"] for _, row in trained.values())
    for index, trial in enumerate(design["trials"]):
        policy_index, training = trained[trial["policy_id"]]
        for stream, seed in enumerate(trial["valuation_seeds"]):
            target = folder/f"trial_{index}_stream_{stream}"
            path = target/"manifest.json"
            if not path.is_file():
                raise ValueError("all twenty valuation outcomes are required before cold collection")
            record = json.loads(path.read_text())
            if (record["index"] != index or record["stream_index"] != stream or record["seed"] != seed
                    or record["paths"] != 2048 or record["source_sha256"] != design["source_sha256"]
                    or record["design_sha256"] != b.sha256(folder/"design_before_reproduction.json")
                    or record["training_manifest_sha256"] != b.sha256(folder/f"training_{policy_index}/manifest.json")
                    or not last_training < record["started_at"] <= record["completed_at"]):
                raise ValueError("cold stream identities, chronology or immutable settings changed")
            spec = {"trial_id": trial["trial_id"], "seed": seed, "training_seed": trial["training_seed"],
                "paths": 2048, "steps": trial["steps"], "exercise_steps": trial["exercise_steps"],
                "factors": trial["factors"], "strike": trial["case"]["strike"], "maturity": trial["case"]["maturity"]}
            if record["status"] == "completed":
                if (training["status"] != "completed"
                        or record["policy_fingerprint"] != training["policy_fingerprint"]
                        or record.get("all_paths_retained") is not True
                        or b.sha256(target/"samples.npz") != record["sample_sha256"]):
                    raise ValueError("cold sample archive or training status changed")
                with np.load(target/"samples.npz", allow_pickle=False) as saved:
                    arrays = {k: saved[k] for k in saved.files}
                if {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()} != record["array_schema"]:
                    raise ValueError("cold sample schema changed")
                checked = validate_benchmark_samples(arrays,
                    b.load_policy(folder/f"training_{policy_index}/policy"), spec=spec)
                streams.append(checked["stream"])
                probes.extend({"trial_id": trial["trial_id"], "seed": seed, **p} for p in checked["probes"])
                del arrays, checked
            elif record["status"] in ("failed", "blocked_training_failure"):
                if record["status"] == "blocked_training_failure" and training["status"] != "failed":
                    raise ValueError("cold valuation falsely claims training failure")
                streams.append({**spec, "failure": {"status": record["status"],
                    "message": record.get("message") or record["status"]}, "controls": {}})
            else:
                raise ValueError("unknown cold valuation outcome")
            specs.append(spec)
            records.append(record)
    pairs = [{"coarse_trial": r["reference_trial"], "fine_trial": r["trial_id"],
        "same_full_policy": r["stage"] == "physical"} for r in design["trials"] if r["reference_trial"]]
    tables = summarize_benchmark(streams, expected_streams=specs, refinement_pairs=pairs)
    tables["conditional_probes"] = __import__("pandas").DataFrame(probes)
    for name, table in tables.items():
        for key, val in ROLE.items(): table[key] = val
        table.to_csv(folder/(name+".csv"), index=False)
    write(folder/"manifest.json", {**design, "complete": True, "completed_at": utc(),
        "records": records, "all_twenty_attempts_retained": True,
        "completed_paths": sum(r["paths"] for r in records if r["status"] == "completed"),
        "tables": {name: len(table) for name, table in tables.items()}, **ROLE})


def preserve(folder):
    rows = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size, "sha256": b.sha256(p)}
        for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "preservation_inventory.json"]
    write(folder/"preservation_inventory.json", {"created_at": utc(), "files": rows,
        "entry_count": len(rows), "entries_sha256": hashlib.sha256(b.canonical(rows).encode()).hexdigest()})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "train", "value", "collect"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--design", type=Path, default=APPROVED_DESIGN,
        help="exact approved design JSON; prepare only")
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="all fixed attempts, executed sequentially")
    choice.add_argument("--index", type=int, choices=range(10))
    p.add_argument("--stream", type=int, choices=(0, 1))
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    folder = destination(args.output_dir)
    if args.stage in ("train", "value") and not args.all and args.index is None:
        p.error("train/value require --all or a fixed --index")
    if args.stage == "value" and not args.all and args.stream is None:
        p.error("single valuation requires --stream")
    if args.stage in ("prepare", "collect") and (args.all or args.index is not None or args.stream is not None):
        p.error("prepare/collect do not accept attempt selections")
    if args.all and args.stream is not None:
        p.error("--all includes both original streams")
    if args.stage != "value" and args.stream is not None:
        p.error("--stream is only meaningful for a single valuation")
    if args.stage == "prepare":
        prepare(folder, args.design)
        return
    load(folder)
    with phase_lock(folder):
        if args.stage == "train":
            for i in ([0, 1, 2, 3, 4, 5, 7, 8, 9] if args.all else [args.index]): train(folder, i)
        elif args.stage == "value":
            for i, s in ([(i, s) for i in range(10) for s in (0, 1)] if args.all else [(args.index, args.stream)]):
                value(folder, i, s)
        else:
            collect(folder)
    if args.stage == "collect":
        preserve(folder)
        print("COLD CLOSED: same-seed reproduction, not new independent validation", flush=True)


if __name__ == "__main__":
    main()
