"""Recover only the two predefined NumPy-Boolean serialization failures.

The original 61 sources and complete campaign remain immutable.  Policy fits
are deterministically replayed on their original training streams; ALL saved
coefficient bytes must match.  Only the scalar flag's JSON representation is
normalized.  The four previously blocked valuation streams use their original
seeds and budgets, with no outcome selection or additional accepted paths.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from rvlab.experiments import _pricing_rough_benchmark as b
from rvlab.experiments import _pricing_rough_benchmark_serialization as safe
from rvlab.experiments._pricing_rough_benchmark_audit import validate_benchmark_samples

spec = importlib.util.spec_from_file_location("frozen_generic_benchmark_runner", ROOT/"examples/rough_benchmark.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
FOLDER = old.BASE/"serialization_repair_v1"
INDICES = (2, 5)
IMPLEMENTATION_SEED, IMPLEMENTATION_PATHS = 1_095_005_042, 4
VERSION = "rough_generic_benchmark_serialization_repair_v1"


def original_protocol():
    design = json.loads((old.FOLDER/"design_before_training.json").read_text())
    old.verify_sources(design)
    return design


def sources():
    pins = dict(original_protocol()["source_sha256"])
    for name in ("src/rvlab/experiments/_pricing_rough_benchmark_serialization.py",
            "examples/rough_benchmark_serialization_repair.py"):
        pins[name] = b.sha256(ROOT/name)
    return pins


def verify_original_failures(design):
    for index in INDICES:
        trial = design["trials"][index]
        failed = json.loads((old.FOLDER/f"training_{index}_manifest.json").read_text())
        if (failed["status"] != "failed" or failed["error_type"] != "TypeError"
                or "JSON serializable" not in failed["message"] or "bool" not in failed["message"]
                or failed["trial_id"] != trial["trial_id"]
                or failed["training_seed"] != trial["training_seed"]):
            raise ValueError("repair scope is restricted to the original two Boolean serialization failures")
        policy_folder = old.FOLDER/("policy_"+trial["policy_id"])
        if not (policy_folder/"coefficients.npz").is_file() or (policy_folder/"policy.json").exists():
            raise ValueError("original partial coefficient evidence or JSON failure changed")
        for stream, seed in enumerate(trial["valuation_seeds"]):
            folder = old.FOLDER/f"trial_{index}_stream_{stream}"
            record = json.loads((folder/"manifest.json").read_text())
            if (record["status"] != "blocked_training_failure" or record["seed"] != seed
                    or record["paths"] != 2048 or (folder/"samples.npz").exists()):
                raise ValueError("recovery valuation seeds must have drawn zero original acceptance paths")


def prepare():
    old.verify_preserved(old.FOLDER)
    parent = json.loads((old.FOLDER/"manifest.json").read_text())
    design = original_protocol()
    if not parent["complete"] or len(parent["records"]) != 20:
        raise ValueError("the original full twenty-attempt campaign must close before repair")
    verify_original_failures(design)
    files = [old.FOLDER/"design_before_training.json", old.FOLDER/"manifest.json",
        old.FOLDER/"preservation_inventory.json"]
    for index in INDICES:
        trial = design["trials"][index]
        files += [old.FOLDER/f"training_{index}_manifest.json",
            old.FOLDER/("policy_"+trial["policy_id"])/"coefficients.npz"]
        files += [old.FOLDER/f"trial_{index}_stream_{stream}/manifest.json" for stream in (0, 1)]
    inputs = dict(design["input_sha256"])
    inputs.update({str(path.relative_to(ROOT)): b.sha256(path) for path in files})
    pins = sources()
    FOLDER.mkdir(parents=True, exist_ok=False)
    declaration = {"version": VERSION, "created_at": old.utc(), "source_sha256": pins,
        "input_sha256": inputs, "parent_campaign": str(old.FOLDER.relative_to(ROOT)),
        "parent_design_sha256": b.sha256(old.FOLDER/"design_before_training.json"),
        "parent_manifest_sha256": b.sha256(old.FOLDER/"manifest.json"),
        "indices": list(INDICES), "trials": [design["trials"][i] for i in INDICES],
        "role": "serialization recovery of two predetermined failed attempts; no new numerical method",
        "permitted_change": "exercise_at_inception NumPy scalar to native Boolean with identical truth value",
        "every_partial_coefficient_key_shape_dtype_and_byte_must_match": True,
        "all_other_policy_fields_must_be_identical": True,
        "original_training_streams_replayed": True, "training_path_budget_changed": False,
        "original_four_never_drawn_valuation_seeds": [seed for i in INDICES
            for seed in design["trials"][i]["valuation_seeds"]],
        "valuation_paths_per_stream": 2048, "total_recovered_valuation_paths": 8192,
        "original_predeclared_total_paths_after_composition": 40960,
        "new_or_adaptive_acceptance_seeds": False, "no_new_valuation_selection": True,
        "no_empirical_valuation_recentering": True, "original_failures_remain_immutable": True,
        "both_identity_replays_required_before_any_recovered_valuation": True,
        "implementation_fixture": {"seed": IMPLEMENTATION_SEED, "paths_per_case": IMPLEMENTATION_PATHS,
            "same_paths_replayed_before_and_after_normalization": True,
            "changed_N_old_seed_label": True, "acceptance_evidence": False},
        "maximum_workers": 2, "all_failures_retained": True,
        "price_or_optimal_delta_readiness_promoted": False, **b.TOLERANCES}
    old.write_json(FOLDER/"design_before_replay.json", declaration)
    old.snapshot(FOLDER, pins)
    print("REPAIR PREPARED", b.sha256(FOLDER/"design_before_replay.json"), len(pins), flush=True)


def get_design():
    design = json.loads((FOLDER/"design_before_replay.json").read_text())
    if sources() != design["source_sha256"]:
        raise ValueError("repair sources changed after predeclaration")
    for name, digest in design["input_sha256"].items():
        if b.sha256(ROOT/name) != digest:
            raise ValueError("a sealed original input changed: "+name)
    return design


def replay(index):
    declaration = get_design()
    parent = original_protocol()
    if index not in INDICES:
        raise ValueError("only the original two serialization failures may be replayed")
    target = FOLDER/f"replay_{index}"
    target.mkdir(exist_ok=False)
    trial = parent["trials"][index]
    original_partial = old.FOLDER/("policy_"+trial["policy_id"])/"coefficients.npz"
    started, clock = old.utc(), time.perf_counter()
    print("REPLAY START", index, flush=True)
    try:
        config = b.benchmark_config(training_seed=trial["training_seed"],
            valuation_seed=trial["valuation_seeds"][0], training_paths=2048,
            valuation_paths=2048, steps=trial["steps"], exercise_steps=trial["exercise_steps"])
        policy = b.train_benchmark_policy(trial,
            b.ExponentialKernel(**parent["kernels"][str(trial["factors"])]) , config)
        coefficient_proof = safe.verify_partial_coefficients(policy, original_partial)
        normalized, identity = safe.normalize_policy_metadata(policy)
        if identity["original_flag_type"] == "builtins.bool":
            raise ValueError("the original NumPy Boolean serialization failure was not reproduced")
        saved = safe.save_benchmark_policy(policy, target/"policy")
        restored = b.load_policy(target/"policy")
        saved_coefficient_proof = safe.verify_partial_coefficients(restored, original_partial)
        fixture_config = replace(config, valuation_paths=IMPLEMENTATION_PATHS,
            valuation_seed=IMPLEMENTATION_SEED)
        before, before_diag = b.value_benchmark_policy(policy, fixture_config)
        after, after_diag = b.value_benchmark_policy(restored, fixture_config)
        invariance = safe.assert_valuation_arrays_identical(before, after)
        np.savez_compressed(target/"implementation_before.npz", **before)
        np.savez_compressed(target/"implementation_after.npz", **after)
        fixture_spec = {"trial_id": "serialization_implementation_"+str(index),
            "seed": IMPLEMENTATION_SEED, "training_seed": trial["training_seed"],
            "paths": IMPLEMENTATION_PATHS, "steps": trial["steps"],
            "exercise_steps": trial["exercise_steps"], "factors": trial["factors"],
            "strike": trial["case"]["strike"], "maturity": trial["case"]["maturity"]}
        audited = validate_benchmark_samples(after, restored, spec=fixture_spec)
        record = {"status": "completed", "coefficient_proof_before_normalization": coefficient_proof,
            "coefficient_proof_after_normalized_save_load": saved_coefficient_proof,
            "metadata_normalization": identity, "saved_policy": saved,
            "implementation_invariance": invariance,
            "implementation_spec": fixture_spec,
            "implementation_diagnostics_before": before_diag, "implementation_diagnostics_after": after_diag,
            "implementation_checks": audited["checks"], "implementation_probes": audited["probes"],
            "implementation_sample_sha256": {"before": b.sha256(target/"implementation_before.npz"),
                "after": b.sha256(target/"implementation_after.npz")},
            "implementation_acceptance_evidence": False}
    except Exception as exc:
        record = old.failure(exc)
    get_design()
    record.update({"index": index, "trial_id": trial["trial_id"],
        "training_seed": trial["training_seed"], "training_paths": 2048,
        "started_at": started, "completed_at": old.utc(), "elapsed_seconds": time.perf_counter()-clock,
        "source_sha256": declaration["source_sha256"],
        "design_sha256": b.sha256(FOLDER/"design_before_replay.json"),
        "original_partial_coefficient_sha256": b.sha256(original_partial),
        "numerical_settings_changed": False, "valuation_selection_performed": False})
    old.write_json(target/"manifest.json", record)
    print("REPLAY COMPLETE", index, record["status"], flush=True)


def verified_replays():
    records = {}
    for index in INDICES:
        folder = FOLDER/f"replay_{index}"
        record = json.loads((folder/"manifest.json").read_text())
        if (record["status"] != "completed" or record["index"] != index
                or record["design_sha256"] != b.sha256(FOLDER/"design_before_replay.json")
                or record["source_sha256"] != get_design()["source_sha256"]
                or not record["implementation_invariance"]["all_valuation_arrays_byte_identical"]
                or not all(record["metadata_normalization"]["all_other_policy_fields_identical"].values())):
            raise ValueError("both complete byte-identity/type-only replays are mandatory before recovered valuation")
        saved = record["saved_policy"]
        if (b.sha256(folder/"policy/policy.json") != saved["policy_sha256"]
                or b.sha256(folder/"policy/coefficients.npz") != saved["coefficient_sha256"]):
            raise ValueError("normalized prevaluation policy seal changed")
        for label in ("before", "after"):
            if b.sha256(folder/f"implementation_{label}.npz") != record["implementation_sample_sha256"][label]:
                raise ValueError("implementation identity evidence changed")
        records[index] = record
    return records


def value(index, stream):
    declaration = get_design()
    replays = verified_replays()
    if index not in INDICES or stream not in (0, 1):
        raise ValueError("only the four original blocked valuation streams may run")
    trial = original_protocol()["trials"][index]
    verify_original_failures(original_protocol())
    folder = FOLDER/f"trial_{index}_stream_{stream}"
    folder.mkdir(exist_ok=False)
    started, clock = old.utc(), time.perf_counter()
    print("RECOVERED VALUE START", index, stream, flush=True)
    try:
        policy = b.load_policy(FOLDER/f"replay_{index}/policy")
        config = replace(policy.config, valuation_seed=trial["valuation_seeds"][stream], valuation_paths=2048)
        arrays, diagnostics = b.value_benchmark_policy(policy, config)
        np.savez_compressed(folder/"samples.npz", **arrays)
        record = {"status": "completed", "sample_sha256": b.sha256(folder/"samples.npz"),
            "array_schema": {key: {"shape": list(a.shape), "dtype": str(a.dtype)} for key, a in arrays.items()},
            "diagnostics": diagnostics,
            "policy_fingerprint": replays[index]["metadata_normalization"]["normalized_fingerprint"],
            "all_paths_retained": True}
    except Exception as exc:
        record = old.failure(exc)
    get_design()
    record.update({"index": index, "stream_index": stream, "trial_id": trial["trial_id"],
        "policy_id": trial["policy_id"], "seed": trial["valuation_seeds"][stream],
        "paths": 2048, "training_seed": trial["training_seed"], "steps": trial["steps"],
        "exercise_steps": trial["exercise_steps"], "factors": trial["factors"],
        "strike": trial["case"]["strike"], "maturity": trial["case"]["maturity"],
        "started_at": started, "completed_at": old.utc(), "elapsed_seconds": time.perf_counter()-clock,
        "source_sha256": declaration["source_sha256"],
        "design_sha256": b.sha256(FOLDER/"design_before_replay.json"),
        "replay_manifest_sha256": b.sha256(FOLDER/f"replay_{index}/manifest.json"),
        "original_blocked_manifest_sha256": b.sha256(old.FOLDER/f"trial_{index}_stream_{stream}/manifest.json"),
        "original_never_drawn_seed_and_budget_unchanged": True,
        "new_fitting_or_selection": False, "metadata_only_recovery": True,
        "all_failures_retained": True})
    old.write_json(folder/"manifest.json", record)
    print("RECOVERED VALUE COMPLETE", index, stream, record["status"], flush=True)


def collect():
    design = get_design()
    replays = verified_replays()
    records = []
    for index in INDICES:
        trial = original_protocol()["trials"][index]
        for stream, seed in enumerate(trial["valuation_seeds"]):
            folder = FOLDER/f"trial_{index}_stream_{stream}"
            record = json.loads((folder/"manifest.json").read_text())
            if (record["index"] != index or record["stream_index"] != stream or record["seed"] != seed
                    or record["paths"] != 2048 or record["source_sha256"] != design["source_sha256"]
                    or record["design_sha256"] != b.sha256(FOLDER/"design_before_replay.json")
                    or record["replay_manifest_sha256"] != b.sha256(FOLDER/f"replay_{index}/manifest.json")):
                raise ValueError("recovery stream identity or source seal changed")
            if record["status"] == "completed":
                if (b.sha256(folder/"samples.npz") != record["sample_sha256"]
                        or record["policy_fingerprint"] != replays[index]["metadata_normalization"]["normalized_fingerprint"]):
                    raise ValueError("recovery sample archive or policy changed")
                with np.load(folder/"samples.npz", allow_pickle=False) as arrays:
                    actual = {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
                        for key in arrays.files}
                if actual != record["array_schema"]:
                    raise ValueError("recovery sample schema changed")
            elif record["status"] != "failed":
                raise ValueError("unknown recovery outcome")
            records.append(record)
    if max(row["completed_at"] for row in replays.values()) >= min(row["started_at"] for row in records):
        raise ValueError("all deterministic replay/normalization proofs must precede every recovered valuation")
    old.write_json(FOLDER/"manifest.json", {**design, "complete": True, "completed_at": old.utc(),
        "design_sha256": b.sha256(FOLDER/"design_before_replay.json"), "records": records,
        "replay_records": list(replays.values()), "all_four_attempts_retained": True,
        "completed_recovered_paths": sum(row["paths"] for row in records if row["status"] == "completed"),
        "sources_unchanged": True, "no_additional_accepted_paths_beyond_original_design": True})
    old.inventory(FOLDER)
    print("REPAIR COLLECTION COMPLETE", len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "replay", "value", "collect"))
    parser.add_argument("--index", type=int, choices=INDICES)
    parser.add_argument("--stream", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.stage in ("replay", "value") and args.index is None:
        parser.error("replay/value require one of the two fixed case indices")
    if args.stage == "value" and args.stream is None:
        parser.error("value requires one of the two fixed stream indices")
    if args.stage == "prepare": prepare()
    elif args.stage == "replay": replay(args.index)
    elif args.stage == "value": value(args.index, args.stream)
    else: collect()


if __name__ == "__main__":
    main()
