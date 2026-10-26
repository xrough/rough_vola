"""Audit a type-only serialization recovery without selecting new observations."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import _pricing_rough_benchmark as b
from . import _pricing_rough_benchmark_serialization as safe
from ._pricing_rough_benchmark_audit import validate_benchmark_samples, summarize_benchmark
from ._pricing_rough_benchmark_collection import (
    ROOT, ARRAY_KEYS, read, sha, file_entry, timestamp, closed_inventory, validate_policy,
)


EXTRA_SOURCES = {"src/rvlab/experiments/_pricing_rough_benchmark_serialization.py",
                 "examples/rough_benchmark_serialization_repair.py"}
INDICES = (2, 5)
DESIGN_SHA256 = "dbf190398826cbd61d389a3cc6e4a1994be35b577478f5776bfd2a65669f6760"


def validate_recovery_protocol(design, parent_design):
    """Restrict recovery to the four original undrawn streams, without selection."""
    if (design.get("version") != "rough_generic_benchmark_serialization_repair_v1" or
            design.get("indices") != list(INDICES) or
            design.get("trials") != [parent_design["trials"][i] for i in INDICES] or
            design.get("original_four_never_drawn_valuation_seeds") != [
                seed for i in INDICES for seed in parent_design["trials"][i]["valuation_seeds"]]):
        raise ValueError("Serialization recovery changed its original failed cases or seed identities")
    expected_settings = {"valuation_paths_per_stream": 2048,
        "total_recovered_valuation_paths": 8192,
        "original_predeclared_total_paths_after_composition": 40960,
        "maximum_workers": 2, "new_or_adaptive_acceptance_seeds": False,
        "no_new_valuation_selection": True, "no_empirical_valuation_recentering": True,
        "original_failures_remain_immutable": True,
        "both_identity_replays_required_before_any_recovered_valuation": True,
        "every_partial_coefficient_key_shape_dtype_and_byte_must_match": True,
        "all_other_policy_fields_must_be_identical": True,
        "original_training_streams_replayed": True,
        "training_path_budget_changed": False, "all_failures_retained": True,
        "price_or_optimal_delta_readiness_promoted": False, **b.TOLERANCES}
    for key, expected in expected_settings.items():
        if design.get(key) != expected or isinstance(expected, bool) and design.get(key) is not expected:
            raise ValueError("Serialization recovery changed a declared scope or tolerance")
    if design.get("implementation_fixture") != {
            "seed": 1095005042, "paths_per_case": 4,
            "same_paths_replayed_before_and_after_normalization": True,
            "changed_N_old_seed_label": True, "acceptance_evidence": False}:
        raise ValueError("Implementation-only identity paths cannot enter acceptance evidence")


def load_arrays(path, schema):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != ARRAY_KEYS:
            raise ValueError("Recovery omitted or changed saved numerical fields")
        arrays = {key: saved[key] for key in saved.files}
    if {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()} != schema:
        raise ValueError("Recovery archive changed shape or dtype")
    return arrays


def validate_implementation_spec(spec, trial, index):
    expected = {"trial_id": "serialization_implementation_" + str(index),
        "seed": 1095005042, "training_seed": trial["training_seed"], "paths": 4,
        "steps": trial["steps"], "exercise_steps": trial["exercise_steps"],
        "factors": trial["factors"], "strike": trial["case"]["strike"],
        "maturity": trial["case"]["maturity"]}
    if spec != expected:
        raise ValueError("Metadata identity fixture changed its predeclared seed, N or transition settings")


def collect_recovery(original, folder):
    folder = Path(folder).resolve()
    parent = folder.parent / "independent_v1"
    parent_manifest = original["campaign"]
    parent_design = read(parent / "design_before_training.json")
    design_path = folder / "design_before_replay.json"
    design_hash, design = sha(design_path), read(design_path)
    if design_hash != DESIGN_SHA256:
        raise ValueError("The approved metadata-only recovery design changed")
    validate_recovery_protocol(design, parent_design)
    manifest = read(folder / "manifest.json")
    if (design.get("parent_design_sha256") != sha(parent / "design_before_training.json") or
            design.get("parent_manifest_sha256") != sha(parent / "manifest.json") or
            not timestamp(parent_manifest["completed_at"]) < timestamp(design["created_at"])):
        raise ValueError("Serialization recovery changed its original failed cases or parent identity")
    if (any(manifest.get(k) != v for k, v in design.items()) or manifest.get("complete") is not True or
            manifest.get("all_four_attempts_retained") is not True or manifest.get("design_sha256") != design_hash):
        raise ValueError("All four recovery attempts must complete before report collection")
    expected_sources = {**parent_design["source_sha256"],
        **{name: sha(ROOT / name) for name in EXTRA_SOURCES}}
    if design["source_sha256"] != expected_sources:
        raise ValueError("Serialization recovery changed the frozen numerical algorithm")
    entries = closed_inventory(folder)
    for name, digest in design["source_sha256"].items():
        for path in (ROOT / name, folder / "source" / name):
            if sha(path) != digest:
                raise ValueError("A serialization recovery source changed")
            entries.append(file_entry(path))
    required = set(parent_design["input_sha256"])
    required.update(str((parent / name).relative_to(ROOT)) for name in (
        "design_before_training.json", "manifest.json", "preservation_inventory.json"))
    for i in INDICES:
        trial = parent_design["trials"][i]
        required.add(str((parent / f"training_{i}_manifest.json").relative_to(ROOT)))
        required.add(str((parent / ("policy_" + trial["policy_id"]) / "coefficients.npz").relative_to(ROOT)))
        required.update(str((parent / f"trial_{i}_stream_{j}/manifest.json").relative_to(ROOT)) for j in (0, 1))
    if set(design["input_sha256"]) != required:
        raise ValueError("Serialization recovery omitted an original failure or coefficient input")
    for name, digest in design["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("An original serialization-failure input changed")
        entries.append(file_entry(ROOT / name))
    policies, replays, proof_rows, checks, probes = {}, [], [], [], []
    for index in INDICES:
        trial = parent_design["trials"][index]
        failed = read(parent / f"training_{index}_manifest.json")
        original_policy_folder = parent / ("policy_" + trial["policy_id"])
        if (failed["status"] != "failed" or failed.get("error_type") != "TypeError" or
                "bool is not JSON serializable" not in failed.get("message", "") or
                (original_policy_folder / "policy.json").exists()):
            raise ValueError("Recovery is restricted to the recorded Boolean writer failures")
        replay_folder = folder / f"replay_{index}"
        replay = read(replay_folder / "manifest.json")
        if (replay.get("status") != "completed" or replay.get("index") != index or
                replay.get("trial_id") != trial["trial_id"] or
                replay.get("training_seed") != trial["training_seed"] or replay.get("training_paths") != 2048 or
                replay.get("source_sha256") != design["source_sha256"] or
                replay.get("design_sha256") != design_hash or
                replay.get("numerical_settings_changed") is not False or
                replay.get("valuation_selection_performed") is not False or
                not timestamp(design["created_at"]) < timestamp(replay["started_at"]) <= timestamp(replay["completed_at"])):
            raise ValueError("A deterministic metadata replay changed the original training construction")
        saved = replay["saved_policy"]
        if (sha(replay_folder / "policy/policy.json") != saved["policy_sha256"] or
                sha(replay_folder / "policy/coefficients.npz") != saved["coefficient_sha256"]):
            raise ValueError("A normalized policy coefficient seal changed")
        policy = b.load_policy(replay_folder / "policy")
        validate_policy(policy, trial, parent_design, saved)
        partial = original_policy_folder / "coefficients.npz"
        proof = safe.verify_partial_coefficients(policy, partial)
        if replay["original_partial_coefficient_sha256"] != sha(partial):
            raise ValueError("The partial coefficient archive used in recovery changed")
        # Reconstruct the representation-sensitive original identity without
        # retraining or changing the normalized rule's coefficients.
        typed = replace(policy, exercise_at_inception=np.bool_(policy.exercise_at_inception))
        _, identity = safe.normalize_policy_metadata(typed)
        if identity != replay["metadata_normalization"] or identity != saved["metadata_normalization"]:
            raise ValueError("Recovery's typed/native identity map changed")
        for key in ("coefficient_proof_before_normalization", "coefficient_proof_after_normalized_save_load"):
            if replay[key] != proof:
                raise ValueError("Recovery did not preserve every coefficient/grid byte")
        before_path, after_path = replay_folder / "implementation_before.npz", replay_folder / "implementation_after.npz"
        for label, path in (("before", before_path), ("after", after_path)):
            if sha(path) != replay["implementation_sample_sha256"][label]:
                raise ValueError("A paired implementation proof changed")
        with np.load(before_path, allow_pickle=False) as source:
            before = {key: source[key] for key in source.files}
        with np.load(after_path, allow_pickle=False) as source:
            after = {key: source[key] for key in source.files}
        invariance = safe.assert_valuation_arrays_identical(before, after)
        if invariance != replay["implementation_invariance"]:
            raise ValueError("Recovery's stock-bump/stopping identity proof changed")
        validate_implementation_spec(replay["implementation_spec"], trial, index)
        if replay.get("implementation_acceptance_evidence") is not False:
            raise ValueError("The paired metadata fixture cannot supply acceptance evidence")
        audited = validate_benchmark_samples(after, policy, spec=replay["implementation_spec"])
        checks.append({"role": "implementation only", **audited["checks"]})
        del before, after, audited
        policies[index] = policy
        replays.append(replay)
        proof_rows.append({"trial_id": trial["trial_id"], "original_training_status": "failed serialization",
            "original_blocked_streams": 2, "original_flag_type": identity["original_flag_type"],
            "normalized_flag_type": identity["normalized_flag_type"],
            "truth_value": identity["normalized_truth_value"], "coefficient_arrays_checked": proof["array_count"],
            "all_coefficient_bytes_identical": True, "all_own_stops_and_output_arrays_identical": True,
            "fitted_rule_changed": False, "numerical_method_changed": False,
            "old_typed_fingerprint": identity["original_typed_fingerprint"],
            "normalized_fingerprint": identity["normalized_fingerprint"],
            "training_seed": trial["training_seed"], "replay_seconds": replay["elapsed_seconds"]})
    if manifest["replay_records"] != replays:
        raise ValueError("A failed or complete metadata replay was omitted")
    last_replay = max(timestamp(r["completed_at"]) for r in replays)
    streams = list(original["streams"])
    records, execution = [], []
    for index in INDICES:
        trial = parent_design["trials"][index]
        for stream_index in (0, 1):
            position = index * 2 + stream_index
            expected = original["expected_streams"][position]
            blocked_folder = parent / f"trial_{index}_stream_{stream_index}"
            blocked = read(blocked_folder / "manifest.json")
            if (blocked["status"] != "blocked_training_failure" or (blocked_folder / "samples.npz").exists() or
                    not streams[position].get("failure")):
                raise ValueError("Recovery would duplicate previously drawn acceptance paths")
            value_folder = folder / f"trial_{index}_stream_{stream_index}"
            record = read(value_folder / "manifest.json")
            if (any(record.get(k) != v for k, v in expected.items()) or record.get("index") != index or
                    record.get("stream_index") != stream_index or
                    record.get("source_sha256") != design["source_sha256"] or
                    record.get("design_sha256") != design_hash or
                    record.get("replay_manifest_sha256") != sha(folder / f"replay_{index}/manifest.json") or
                    record.get("original_blocked_manifest_sha256") != sha(blocked_folder / "manifest.json") or
                    record.get("original_never_drawn_seed_and_budget_unchanged") is not True or
                    record.get("new_fitting_or_selection") is not False or
                    record.get("metadata_only_recovery") is not True or
                    not last_replay < timestamp(record["started_at"]) <= timestamp(record["completed_at"]) <= timestamp(manifest["completed_at"])):
                raise ValueError("A recovered stream changed its original seed, budget, policy proof or chronology")
            if record["status"] == "completed":
                if (record.get("all_paths_retained") is not True or
                        record["policy_fingerprint"] != replays[INDICES.index(index)]["metadata_normalization"]["normalized_fingerprint"] or
                        sha(value_folder / "samples.npz") != record["sample_sha256"]):
                    raise ValueError("A recovered valuation changed its policy or retained sample")
                arrays = load_arrays(value_folder / "samples.npz", record["array_schema"])
                audited = validate_benchmark_samples(arrays, policies[index], spec=expected)
                streams[position] = audited["stream"]
                checks.append({"role": "recovered original stream", **audited["checks"]})
                probes.extend({"trial_id": trial["trial_id"], "seed": expected["seed"], **p}
                              for p in audited["probes"])
                del arrays, audited
            elif record["status"] == "failed":
                streams[position] = {**expected, "failure": {"status": "recovery failed", "message": record.get("message") or "failed"}, "controls": {}}
            else:
                raise ValueError("Unknown serialization recovery outcome")
            records.append(record)
            execution.append({**expected, "status": record["status"], "elapsed_seconds": record["elapsed_seconds"],
                "started_at": record["started_at"], "completed_at": record["completed_at"],
                "timing_scope": "Recovered original valuation stream; metadata repair is not a pricing speedup"})
    if manifest["records"] != records:
        raise ValueError("One of the four recovered attempts was omitted")
    recovered_paths = sum(r["paths"] for r in records if r["status"] == "completed")
    if recovered_paths != manifest["completed_recovered_paths"]:
        raise ValueError("Recovered path counts changed")
    tables = summarize_benchmark(streams, expected_streams=original["expected_streams"],
                                 refinement_pairs=original["refinement_pairs"])
    tables["original_failures"] = original["tables"]["failures"].copy()
    tables["original_execution"] = original["tables"]["execution"].copy()
    tables["serialization_recovery"] = pd.DataFrame(proof_rows)
    tables["recovery_execution"] = pd.DataFrame(execution)
    composed_execution = tables["original_execution"].copy()
    composed_execution["execution_source"] = "original campaign"
    for row in execution:
        mask = ((composed_execution.trial_id == row["trial_id"]) &
                (composed_execution.seed == row["seed"]))
        if mask.sum() != 1:
            raise ValueError("Recovery timings must replace exactly one original blocked stream")
        for key, value in row.items():
            composed_execution.loc[mask, key] = value
        composed_execution.loc[mask, "execution_source"] = "metadata-only recovery of original undrawn stream"
    tables["execution"] = composed_execution
    tables["recovery_path_identity_checks"] = pd.DataFrame(checks)
    tables["recovery_conditional_probes"] = pd.DataFrame(probes)
    # Keep the original operational/source tables alongside recomputed pooled
    # statistics. The original failure table remains explicit, never erased.
    tables.update({k: v for k, v in original["tables"].items()
                   if k not in tables and k not in ("failures",)})
    return {"tables": tables, "streams": streams, "campaign": original["campaign"],
            "recovery": manifest, "input_files": original["input_files"] + entries,
            "refinement_pairs": original["refinement_pairs"],
            "completed_paths": original["campaign"]["completed_valuation_paths"] + recovered_paths}
