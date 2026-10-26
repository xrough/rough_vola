"""Sealed independent delta tests for the newly trained American rough rule."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_new_policy_delta import (
    LEVELS, TOLERANCES, flatten_samples, independent_time_comparison,
    new_policy_delta_samples, summarize_arrays, validate_arrays)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def identity(path): return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=int, default=[])
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    output = ROOT / "notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder = output / "rough_new_policy_delta_v1/new_selected_2x4096"
    folder.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "rough_larger_policy_v1/32768_paths_two_starts"
    source_paths = [Path(__file__).resolve()] + [ROOT / "src/rvlab/experiments" / name for name in (
        "_pricing_rough_large_policy.py", "_pricing_rough_new_policy_delta.py",
        "_pricing_rough_stock_delta.py", "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py")]
    source_paths += sorted((ROOT / "src/rvlab/pricing").glob("*.py"))
    sources = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    sealed = json.loads((bundle_path / "frozen_policy_before_final.json").read_text())
    design = {"version": "rough_new_policy_delta_v1", "seeds": [810005042, 811005042, 812005042],
        "paths": [4096, 4096, 2048], "steps": [128, 128, 256], "exercise_steps": 64,
        "bump_relatives": list(LEVELS), "source_sha256": sources,
        "selected_policy_name": sealed["selected_name"],
        "policy_labels": {"original_policy": "previous dense tensor", "selected_policy": sealed["selected_name"]},
        "new_policy_seal_sha256": sha(bundle_path / "frozen_policy_before_final.json"),
        "new_policy_arrays_sha256": sha(bundle_path / "frozen_policy_coefficients.npz"),
        "historical_bundle_path": sealed["historical_bundle_path"],
        "historical_bundle_sha256": sealed["historical_bundle_sha256"],
        "all_selection_training_seeds": sealed["all_selection_training_seeds"],
        "fixed_parameters_initial_factors_discount_dividends": True,
        "new_and_previous_rules_reapplied_at_each_bump": True,
        "no_valuation_selection_or_fitting": True, "all_paths_retained": True,
        "same_common_stopped_martingale": True, "optimal_delta_ready": False,
        "tolerances_changed": False, **TOLERANCES}
    design_path = folder / "design_before_valuation.json"
    if design_path.exists():
        previous = json.loads(design_path.read_text())
        if {key: value for key, value in previous.items() if key != "sealed_at_utc"} != design:
            raise ValueError("Changed new-policy delta design or numerical sources")
    else:
        design_path.write_text(json.dumps({**design, "sealed_at_utc": now()}, indent=2, sort_keys=True))
    bundle = load_larger_policy_bundle(bundle_path)
    for index in args.indices:
        if index not in (0, 1, 2): raise ValueError("indices must be 0, 1 or 2")
        record_path = folder / f"stream_{index}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["design_sha256"] != sha(design_path): raise ValueError("Changed stream design")
            for item in record["files"]:
                if sha(item["path"]) != item["sha256"]: raise ValueError("Changed stream artifact")
            print(json.dumps({"index": index, "cached": True}), flush=True)
            continue
        started = now(); clock = perf_counter()
        result = new_policy_delta_samples(bundle, seed=design["seeds"][index],
            paths=design["paths"][index], steps=design["steps"][index])
        elapsed = perf_counter() - clock
        arrays = flatten_samples(result)
        validate_arrays(arrays, paths=design["paths"][index])
        estimates, comparisons = summarize_arrays(arrays)
        estimates["seed"] = design["seeds"][index]; estimates["steps"] = design["steps"][index]
        estimates.to_csv(folder / f"stream_{index}_estimates.csv", index=False)
        comparisons.to_csv(folder / f"stream_{index}_comparisons.csv", index=False)
        np.savez_compressed(folder / f"stream_{index}_samples.npz", **arrays)
        if {str(path.relative_to(ROOT)): sha(path) for path in source_paths} != sources:
            raise ValueError("Numerical source changed during new-policy delta measurement")
        files = [folder / f"stream_{index}_{suffix}" for suffix in ("estimates.csv", "comparisons.csv", "samples.npz")]
        record = {"index": index, "seed": design["seeds"][index], "paths": design["paths"][index],
            "steps": design["steps"][index], "started_at_utc": started, "finished_at_utc": now(),
            "elapsed_seconds": elapsed, "design_sha256": sha(design_path),
            "diagnostics": result["diagnostics"], "source_sha256": sources,
            "sources_unchanged": True, "files": [identity(path) for path in files]}
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(json.dumps({"index": index, "elapsed_seconds": elapsed}), flush=True)
        print(estimates.to_string(index=False), flush=True)
    if args.collect:
        records, archives = [], []
        for index in (0, 1, 2):
            record = json.loads((folder / f"stream_{index}.json").read_text())
            if record["design_sha256"] != sha(design_path): raise ValueError("Changed stream design")
            for item in record["files"]:
                if sha(item["path"]) != item["sha256"]: raise ValueError("Changed stream artifact")
            with np.load(folder / f"stream_{index}_samples.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            validate_arrays(arrays, paths=design["paths"][index])
            archives.append(arrays); records.append(record)
        joined = {key: np.concatenate([archives[i][key] for i in (0, 1)], axis=0)
            for key in archives[0] if key != "relative_spots"}
        joined["relative_spots"] = archives[0]["relative_spots"]
        validate_arrays(joined, paths=8192)
        pooled, paired = summarize_arrays(joined)
        refined, refined_paired = summarize_arrays(archives[2])
        time = independent_time_comparison(pooled, refined)
        per_seed = pd.concat([pd.read_csv(folder / f"stream_{i}_estimates.csv") for i in (0, 1, 2)], ignore_index=True)
        tables = {"pooled_estimates.csv": pooled, "pooled_comparisons.csv": paired,
            "time_estimates.csv": refined, "time_comparisons.csv": refined_paired,
            "independent_time_comparison.csv": time, "per_seed.csv": per_seed}
        for name, frame in tables.items(): frame.to_csv(folder / name, index=False)
        files = [folder / name for name in tables]
        files += [folder / f"stream_{i}.json" for i in (0, 1, 2)]
        manifest = {"version": design["version"], "complete": True, "design_sha256": sha(design_path),
            "source_sha256": sources, "records": records, "primary_paths": 8192,
            "primary_seed_count": 2, "time_paths": 2048, "time_seed_count": 1,
            "selected_policy_name": design["selected_policy_name"],
            "policy_labels": design["policy_labels"],
            "independent_paths": True, "all_paths_retained": True,
            "new_rule_delta_measured": True, "optimal_delta_ready": False,
            "tolerances_changed": False, "files": [identity(path) for path in files],
            "limitations": ["Finite-bump sensitivity of a frozen lower policy, not an optimal American Greek.",
                "Sampling, bump and weak-time checks are individual correlated diagnostics, not a total error bound.",
                "Fixed finite kernel, exercise calendar, explicit no-cash synthetic inception case.",
                "Prior fixed-model reference measurements are historical and not rerun on this new rule."]}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(pooled.to_string(index=False), flush=True)
        print(time.to_string(index=False), flush=True)


if __name__ == "__main__": main()
