"""Seal, train and independently value the fixed six-case generic benchmark.

Run preflight, then prepare (only after review), train each unique policy,
value the two declared streams per trial, and finally collect.  Every failure
is a durable result; neither collection nor the notebooks rerun failed fits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from rvlab.experiments import _pricing_rough_benchmark as b
from rvlab.pricing.rough_lift import approximate_fractional_kernel


BASE = ROOT/"notebooks/_output/pricing/first_stage_v1/01_pricing_and_exercise/rough_generic_benchmark_v1"
PREFLIGHT = BASE/"spent_preflight_v1"
IMPLEMENTATION_GUARD = BASE/"implementation_guard_v1"
PILOT_AUDIT = BASE/"pilot_independent_audit_v1"
FOLDER = BASE/"independent_v1"
OLD_DESIGN = ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise/rough_shared_delta_v1/independent_v1/design_before_valuation.json"
TRAIN_PATHS = VALUE_PATHS = 2048
FIRST_TRAINING_SEED = 1_146_005_042
PILOT_TRAIN_SEED, PILOT_VALUE_SEED = 1_094_005_042, 1_095_005_042


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sources():
    old = json.loads(OLD_DESIGN.read_text())["source_sha256"]
    if len(old) != 58:
        raise ValueError("expected the original 58 sealed numerical sources")
    result = dict(old)
    for name in ("src/rvlab/experiments/_pricing_rough_benchmark.py", "examples/rough_benchmark.py",
            "src/rvlab/experiments/_pricing_rough_benchmark_audit.py"):
        result[name] = b.sha256(ROOT/name)
    for name, expected in old.items():
        if b.sha256(ROOT/name) != expected:
            raise ValueError("a frozen historical numerical source changed: "+name)
    return result


def snapshot(folder, pins):
    for name, expected in pins.items():
        source = ROOT/name
        if b.sha256(source) != expected:
            raise ValueError("source changed before snapshot: "+name)
        target = folder/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def verify_sources(design):
    if sources() != design["source_sha256"]:
        raise ValueError("benchmark source changed after predeclaration")


def inventory(folder):
    files = [{"path": str(path.relative_to(folder)), "bytes": path.stat().st_size,
        "sha256": b.sha256(path)} for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "preservation_inventory.json"]
    write_json(folder/"preservation_inventory.json", {"created_at": utc(),
        "files": files, "entry_count": len(files),
        "entries_sha256": __import__("hashlib").sha256(b.canonical(files).encode()).hexdigest()})


def verify_preserved(folder):
    """Closed historical files are checked against their own frozen inventory."""
    record = json.loads((folder/"preservation_inventory.json").read_text())
    files = record["files"]
    actual = sorted(str(path.relative_to(folder)) for path in folder.rglob("*")
        if path.is_file() and path.name != "preservation_inventory.json")
    if (actual != sorted(row["path"] for row in files)
            or len(files) != record["entry_count"]
            or __import__("hashlib").sha256(b.canonical(files).encode()).hexdigest()
                != record["entries_sha256"]):
        raise ValueError("preflight inventory is not closed or changed")
    for row in files:
        path = folder/row["path"]
        if path.stat().st_size != row["bytes"] or b.sha256(path) != row["sha256"]:
            raise ValueError("preserved preflight file changed: "+row["path"])


def kernels():
    return {str(factors): asdict(approximate_fractional_kernel(.15, 90/365, factors))
        for factors in (4, 8)}


def stream_specs(trials):
    policy_train = {row["policy_id"]: FIRST_TRAINING_SEED+3_000_000*i
        for i, row in enumerate(trials) if row["policy_id"] == row["trial_id"]}
    records = []
    for i, trial in enumerate(trials):
        records.append({**trial, "index": i,
            "training_seed": policy_train[trial["policy_id"]],
            "training_paths": TRAIN_PATHS, "valuation_paths": VALUE_PATHS,
            "valuation_seeds": [FIRST_TRAINING_SEED+3_000_000*i+1_000_000,
                FIRST_TRAINING_SEED+3_000_000*i+2_000_000]})
    train = {row["training_seed"] for row in records}
    values = [seed for row in records for seed in row["valuation_seeds"]]
    if len(values) != len(set(values)) or train.intersection(values):
        raise ValueError("benchmark streams must be independent")
    return records


def failure(exc):
    return {"status": "failed", "error_type": type(exc).__name__,
        "message": str(exc), "traceback": traceback.format_exc()}


def preflight():
    PREFLIGHT.mkdir(parents=True, exist_ok=False)
    pins = sources()
    ks = kernels()
    declarations = [{"label": "base4", "factors": 4, "steps": 128, "exercise_steps": 64},
        {"label": "joint8", "factors": 8, "steps": 256, "exercise_steps": 128}]
    design = {"version": b.VERSION, "role": "implementation and runtime planning only",
        "created_at": utc(), "source_sha256": pins, "kernels": ks,
        "training_seed": PILOT_TRAIN_SEED, "valuation_seed": PILOT_VALUE_SEED,
        "training_paths": 256, "valuation_paths": 256,
        "seed_labels_previously_spent": True,
        "changed_N_draw_mapping": True,
        "historical_path_replay_claimed": False, "acceptance_evidence": False,
        "no_final_case_or_coefficient_selection": True, "fixtures": declarations,
        "planned_training_paths": TRAIN_PATHS, "planned_valuation_paths": VALUE_PATHS,
        "proposed_trials": b.benchmark_trials(include_joint=True),
        "parameters": b.PARAMETERS, **b.TOLERANCES}
    write_json(PREFLIGHT/"design_before_implementation.json", design)
    snapshot(PREFLIGHT, pins)
    results = []
    for declaration in declarations:
        folder = PREFLIGHT/declaration["label"]
        folder.mkdir()
        start = time.perf_counter()
        started = utc()
        try:
            trial = b.benchmark_trials()[-1 if declaration["label"] == "joint8" else 4]
            config = b.benchmark_config(training_seed=PILOT_TRAIN_SEED,
                valuation_seed=PILOT_VALUE_SEED, training_paths=256, valuation_paths=256,
                steps=declaration["steps"], exercise_steps=declaration["exercise_steps"])
            kernel = b.ExponentialKernel(**ks[str(declaration["factors"])])
            print("PILOT TRAIN", declaration["label"], flush=True)
            training_start = time.perf_counter()
            policy = b.train_benchmark_policy(trial, kernel, config)
            training_seconds = time.perf_counter()-training_start
            b.save_policy(policy, folder/"policy")
            print("PILOT VALUE", declaration["label"], flush=True)
            value_start = time.perf_counter()
            arrays, diagnostics = b.value_benchmark_policy(policy, config)
            valuation_seconds = time.perf_counter()-value_start
            np.savez_compressed(folder/"samples.npz", **arrays)
            record = {"status": "completed", "training_seconds": training_seconds,
                "valuation_seconds": valuation_seconds, "diagnostics": diagnostics,
                "sample_sha256": b.sha256(folder/"samples.npz"),
                "policy_sha256": b.sha256(folder/"policy/policy.json"),
                "projected_full_training_seconds_linear": 8*training_seconds,
                "projected_full_valuation_seconds_linear": 8*valuation_seconds,
                "projection_is_guarantee": False,
                "sample_compressed_bytes": (folder/"samples.npz").stat().st_size}
        except Exception as exc:
            record = failure(exc)
        record.update({**declaration, "started_at": started, "completed_at": utc(),
            "elapsed_seconds": time.perf_counter()-start,
            "acceptance_evidence": False, "all_failures_retained": True})
        write_json(folder/"record.json", record)
        results.append(record)
        print("PILOT COMPLETE", declaration["label"], record["status"], flush=True)
    verify_sources(design)
    write_json(PREFLIGHT/"manifest.json", {**design, "completed_at": utc(),
        "complete": True, "records": results, "sources_unchanged": True})
    inventory(PREFLIGHT)


def prepare():
    verify_preserved(PREFLIGHT)
    verify_preserved(IMPLEMENTATION_GUARD)
    verify_preserved(PILOT_AUDIT)
    preflight_manifest = json.loads((PREFLIGHT/"manifest.json").read_text())
    implementation = json.loads((IMPLEMENTATION_GUARD/"manifest.json").read_text())
    audit = json.loads((PILOT_AUDIT/"manifest.json").read_text())
    if not preflight_manifest["complete"]:
        raise ValueError("preflight must be complete before formal prepare")
    if (not implementation["complete"] or any(row["status"] != "completed"
            for row in implementation["records"])):
        raise ValueError("both stochastic implementation roundtrips must pass before prepare")
    if not audit["complete"] or not audit["all_checks_passed"] or audit["new_paths"] != 0:
        raise ValueError("the independent cached pilot audit must pass without new paths")
    FOLDER.mkdir(parents=True, exist_ok=False)
    pins = sources()
    trials = stream_specs(b.benchmark_trials(include_joint=True))
    design = {"version": b.VERSION, "created_at": utc(), "source_sha256": pins,
        "kernels": kernels(), "parameters": b.PARAMETERS,
        "trials": trials, "trial_count": len(trials), "stream_count": 2*len(trials),
        "policy_training_count": len({row["policy_id"] for row in trials}),
        "total_valuation_paths": 2*VALUE_PATHS*len(trials),
        "relative_spots": list(b.RELATIVE_SPOTS), "bump_relatives": [.005, .0025],
        "estimators": list(b.ESTIMATORS),
        "preflight_manifest_sha256": b.sha256(PREFLIGHT/"manifest.json"),
        "preflight_folder": str(PREFLIGHT.relative_to(ROOT)),
        "implementation_guard_folder": str(IMPLEMENTATION_GUARD.relative_to(ROOT)),
        "pilot_audit_folder": str(PILOT_AUDIT.relative_to(ROOT)),
        "input_sha256": {str(path.relative_to(ROOT)): b.sha256(path) for path in (
            OLD_DESIGN, PREFLIGHT/"manifest.json", PREFLIGHT/"preservation_inventory.json",
            IMPLEMENTATION_GUARD/"manifest.json", IMPLEMENTATION_GUARD/"preservation_inventory.json",
            PILOT_AUDIT/"manifest.json", PILOT_AUDIT/"preservation_inventory.json")},
        "old_design_sha256": b.sha256(OLD_DESIGN),
        "old_source_count": len(json.loads(OLD_DESIGN.read_text())["source_sha256"]),
        "no_fresh_selection": True, "no_fitting_on_valuation_paths": True,
        "no_empirical_valuation_recentering": True, "all_paths_retained": True,
        "all_failures_retained": True, "acceptance_requires_all_streams_reported": True,
        "historical_or_pilot_paths_used_for_acceptance": False,
        "independent_training_valuation": True,
        "maximum_workers": 2, "joint_refinement_included": True,
        "decision_function": "original training-grid European function plus frozen clipped premium",
        "control_center": "actual physical-grid discrete branch/order/Gaussian law",
        "uncertainty_scope": "valuation Monte Carlo conditional on the independently trained policy; no training/kernel/calendar error coverage",
        "regression_ridge": 1e-5, "premium_cap": .7,
        "stock_integral": "exact_drift", "batch_size": 512,
        "premium_control": {"quadrature_nodes": 24, "fourier_cutoff": 100., "riccati_steps": 512},
        "all_policy_training_attempts_complete_before_any_valuation": True,
        "focused_GH16_shared193_reused": False,
        "price_gate": "G+1.96*SE(G)+1.96*SE(L)<=.001+.0001*abs(L)",
        "price_and_delta_accuracy_claimed": False,
        "optimal_american_delta_ready": False, "new_hedging_readiness": False,
        **b.TOLERANCES}
    write_json(FOLDER/"design_before_training.json", design)
    snapshot(FOLDER, pins)
    print("PREPARED", b.sha256(FOLDER/"design_before_training.json"), len(pins), flush=True)


def get_design():
    design = json.loads((FOLDER/"design_before_training.json").read_text())
    verify_sources(design)
    for name, digest in design["input_sha256"].items():
        if b.sha256(ROOT/name) != digest:
            raise ValueError("a declared benchmark implementation input changed: "+name)
    return design


def train(index):
    design = get_design()
    trial = design["trials"][index]
    if trial["policy_id"] != trial["trial_id"]:
        raise ValueError("physical refinement reuses the sealed base policy; do not retrain")
    folder = FOLDER/("policy_"+trial["policy_id"])
    if folder.exists():
        raise FileExistsError(folder)
    record_path = FOLDER/(f"training_{index}_manifest.json")
    if record_path.exists():
        raise FileExistsError(record_path)
    start, started = time.perf_counter(), utc()
    print("TRAIN START", index, trial["trial_id"], flush=True)
    try:
        config = b.benchmark_config(training_seed=trial["training_seed"],
            valuation_seed=trial["valuation_seeds"][0], training_paths=TRAIN_PATHS,
            valuation_paths=VALUE_PATHS, steps=trial["steps"], exercise_steps=trial["exercise_steps"])
        policy = b.train_benchmark_policy(trial,
            b.ExponentialKernel(**design["kernels"][str(trial["factors"])]) , config)
        payload = b.save_policy(policy, folder)
        record = {"status": "completed", "policy_fingerprint": payload["policy_fingerprint"],
            "policy_sha256": b.sha256(folder/"policy.json"),
            "coefficient_sha256": b.sha256(folder/"coefficients.npz")}
    except Exception as exc:
        record = failure(exc)
    verify_sources(design)
    record.update({"index": index, "trial_id": trial["trial_id"],
        "policy_id": trial["policy_id"], "training_seed": trial["training_seed"],
        "training_paths": TRAIN_PATHS, "started_at": started, "completed_at": utc(),
        "elapsed_seconds": time.perf_counter()-start, "source_sha256": design["source_sha256"],
        "design_sha256": b.sha256(FOLDER/"design_before_training.json"),
        "all_failures_retained": True})
    write_json(record_path, record)
    print("TRAIN COMPLETE", index, record["status"], flush=True)


def value(index, stream):
    if stream not in (0, 1):
        raise ValueError("stream must be zero or one")
    design = get_design()
    # Every policy construction is completed and visible before any valuation.
    for i, row in enumerate(design["trials"]):
        if row["policy_id"] == row["trial_id"] and not (FOLDER/f"training_{i}_manifest.json").is_file():
            raise ValueError("all declared training attempts must complete before valuation")
    trial = design["trials"][index]
    policy_index = next(i for i, row in enumerate(design["trials"])
        if row["trial_id"] == trial["policy_id"])
    training = json.loads((FOLDER/f"training_{policy_index}_manifest.json").read_text())
    folder = FOLDER/f"trial_{index}_stream_{stream}"
    folder.mkdir(exist_ok=False)
    start, started = time.perf_counter(), utc()
    print("VALUE START", index, stream, trial["trial_id"], flush=True)
    if training["status"] != "completed":
        record = {"status": "blocked_training_failure", "training_failure": training}
    else:
        try:
            policy_folder = FOLDER/("policy_"+trial["policy_id"])
            if b.sha256(policy_folder/"policy.json") != training["policy_sha256"]:
                raise ValueError("training policy changed before independent valuation")
            policy = b.load_policy(policy_folder)
            config = replace(policy.config, valuation_paths=VALUE_PATHS,
                valuation_seed=trial["valuation_seeds"][stream], steps=trial["steps"])
            arrays, diagnostics = b.value_benchmark_policy(policy, config)
            print("VALUE ARCHIVING", index, stream, flush=True)
            np.savez_compressed(folder/"samples.npz", **arrays)
            record = {"status": "completed", "sample_sha256": b.sha256(folder/"samples.npz"),
                "array_schema": {key: {"shape": list(a.shape), "dtype": str(a.dtype)} for key, a in arrays.items()},
                "diagnostics": diagnostics, "policy_fingerprint": training["policy_fingerprint"],
                "policy_sha256": training["policy_sha256"], "all_paths_retained": True}
        except Exception as exc:
            record = failure(exc)
    verify_sources(design)
    record.update({"index": index, "stream_index": stream, "trial_id": trial["trial_id"],
        "policy_id": trial["policy_id"], "seed": trial["valuation_seeds"][stream],
        "paths": VALUE_PATHS, "training_seed": trial["training_seed"],
        "steps": trial["steps"], "exercise_steps": trial["exercise_steps"],
        "factors": trial["factors"], "strike": trial["case"]["strike"],
        "maturity": trial["case"]["maturity"], "started_at": started, "completed_at": utc(),
        "elapsed_seconds": time.perf_counter()-start, "source_sha256": design["source_sha256"],
        "design_sha256": b.sha256(FOLDER/"design_before_training.json"),
        "training_manifest_sha256": b.sha256(FOLDER/f"training_{policy_index}_manifest.json"),
        "new_fitting_or_selection": False, "all_failures_retained": True})
    write_json(folder/"manifest.json", record)
    print("VALUE COMPLETE", index, stream, record["status"], flush=True)


def collect():
    design = get_design()
    records, training = [], []
    design_hash = b.sha256(FOLDER/"design_before_training.json")
    training_by_id = {}
    for i, trial in enumerate(design["trials"]):
        if trial["policy_id"] == trial["trial_id"]:
            item = json.loads((FOLDER/f"training_{i}_manifest.json").read_text())
            if (item["index"] != i or item["trial_id"] != trial["trial_id"]
                    or item["training_seed"] != trial["training_seed"]
                    or item["training_paths"] != TRAIN_PATHS
                    or item["design_sha256"] != design_hash
                    or item["source_sha256"] != design["source_sha256"]):
                raise ValueError("training artifact differs from declared policy")
            if item["status"] == "completed":
                policy_folder = FOLDER/("policy_"+trial["policy_id"])
                if (b.sha256(policy_folder/"policy.json") != item["policy_sha256"]
                        or b.sha256(policy_folder/"coefficients.npz") != item["coefficient_sha256"]):
                    raise ValueError("training coefficient seal changed")
                b.load_policy(policy_folder)
            elif item["status"] != "failed":
                raise ValueError("unknown training outcome")
            training.append(item)
            training_by_id[trial["policy_id"]] = (i, item)
    for i, trial in enumerate(design["trials"]):
        policy_index, trained = training_by_id[trial["policy_id"]]
        for stream, seed in enumerate(trial["valuation_seeds"]):
            folder = FOLDER/f"trial_{i}_stream_{stream}"
            record = json.loads((folder/"manifest.json").read_text())
            if (record["index"] != i or record["stream_index"] != stream or record["seed"] != seed
                    or record["paths"] != VALUE_PATHS or record["trial_id"] != trial["trial_id"]
                    or record["training_seed"] != trial["training_seed"]
                    or record["steps"] != trial["steps"] or record["exercise_steps"] != trial["exercise_steps"]
                    or record["factors"] != trial["factors"]
                    or record["training_manifest_sha256"] != b.sha256(FOLDER/f"training_{policy_index}_manifest.json")
                    or record["source_sha256"] != design["source_sha256"]
                    or record["design_sha256"] != design_hash):
                raise ValueError("valuation artifact differs from declared stream")
            if record["status"] == "completed":
                if (trained["status"] != "completed" or not record["all_paths_retained"]
                        or record["policy_fingerprint"] != trained["policy_fingerprint"]
                        or record["policy_sha256"] != trained["policy_sha256"]
                        or b.sha256(folder/"samples.npz") != record["sample_sha256"]):
                    raise ValueError("valuation sample or policy seal changed")
                with np.load(folder/"samples.npz", allow_pickle=False) as arrays:
                    schema = {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
                        for key in arrays.files}
                if schema != record["array_schema"]:
                    raise ValueError("valuation sample schema changed")
            elif record["status"] not in ("failed", "blocked_training_failure"):
                raise ValueError("unknown valuation outcome")
            elif record["status"] == "blocked_training_failure" and trained["status"] != "failed":
                raise ValueError("valuation falsely claims training was blocked")
            records.append(record)
    if max(row["completed_at"] for row in training) >= min(row["started_at"] for row in records):
        raise ValueError("training must be sealed before any independent valuation")
    write_json(FOLDER/"manifest.json", {**design, "complete": True,
        "completed_at": utc(), "records": records, "training_records": training,
        "design_sha256": b.sha256(FOLDER/"design_before_training.json"),
        "all_declared_attempts_retained": True, "sources_unchanged": True,
        "all_completed_paths_retained": all(row.get("all_paths_retained", True) for row in records),
        "completed_valuation_paths": sum(row["paths"] for row in records if row["status"] == "completed"),
        "all_streams_reported": len(records) == design["stream_count"]})
    inventory(FOLDER)
    print("COLLECTION COMPLETE", len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "prepare", "train", "value", "collect"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--stream", type=int)
    args = parser.parse_args()
    if args.stage in ("train", "value") and (args.index is None or not 0 <= args.index < 10):
        parser.error("train/value require an index from zero through nine")
    if args.stage == "preflight": preflight()
    elif args.stage == "prepare": prepare()
    elif args.stage == "train": train(args.index)
    elif args.stage == "value": value(args.index, args.stream)
    else: collect()


if __name__ == "__main__":
    main()
