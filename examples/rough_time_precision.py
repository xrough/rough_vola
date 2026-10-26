"""Fresh 256/512-step stock-delta precision study; no adaptive path pooling."""
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
from rvlab.experiments._pricing_rough_time_precision import (
    LEVELS, TOLERANCES, flatten_samples, independent_resolution_comparison,
    new_policy_time_samples, summarize_time_arrays, validate_time_arrays)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def identity(path): return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True))


def historical_baseline(folder, design):
    """Read the old 128-step baseline; exclude the old 256-step pilot."""
    if sha(folder / "manifest.json") != design["historical_baseline_manifest_sha256"]:
        raise ValueError("Changed historical time baseline manifest")
    for name, digest in design["historical_baseline_files_sha256"].items():
        if sha(folder / name) != digest: raise ValueError("Changed historical time baseline artifact")
    frame = pd.read_csv(folder / "pooled_estimates.csv", float_precision="round_trip")
    frame = frame[frame.bump_relative.isin(LEVELS)].copy()
    if len(frame) != 8 or not (frame.paths == 8192).all():
        raise ValueError("Historical baseline is not the sealed 8192-path 128-step study")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=int, default=[])
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    output = ROOT / "notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder = output / "rough_time_precision_v1/fresh256_and512"
    folder.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "rough_larger_policy_v1/32768_paths_two_starts"
    baseline = output / "rough_new_policy_delta_v1/new_selected_2x4096"
    source_paths = [Path(__file__).resolve()] + [ROOT / "src/rvlab/experiments" / name for name in (
        "_pricing_rough_large_policy.py", "_pricing_rough_time_precision.py", "_pricing_rough_stock_delta.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py")]
    source_paths += sorted((ROOT / "src/rvlab/pricing").glob("*.py"))
    sources = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    sealed = json.loads((bundle_path / "frozen_policy_before_final.json").read_text())
    baseline_names = ["pooled_estimates.csv", "preservation_inventory.json",
        "stream_0_samples.npz", "stream_1_samples.npz", "stream_0.json", "stream_1.json"]
    design = {"version": "rough_time_precision_v1", "seeds": [910005042, 911005042, 912005042],
        "paths": [4096, 4096, 2048], "steps": [256, 256, 512], "exercise_steps": 64,
        "bump_relatives": list(LEVELS), "stock_count": 5, "source_sha256": sources,
        "selected_policy_name": sealed["selected_name"],
        "policy_labels": {"original_policy": "previous dense tensor", "selected_policy": sealed["selected_name"]},
        "new_policy_seal_sha256": sha(bundle_path / "frozen_policy_before_final.json"),
        "new_policy_arrays_sha256": sha(bundle_path / "frozen_policy_coefficients.npz"),
        "historical_bundle_path": sealed["historical_bundle_path"],
        "historical_bundle_sha256": sealed["historical_bundle_sha256"],
        "all_selection_training_seeds": sealed["all_selection_training_seeds"],
        "historical_baseline_path": str(baseline), "historical_baseline_steps": 128,
        "historical_baseline_paths": 8192, "historical_baseline_seeds": [810005042, 811005042],
        "historical_baseline_manifest_sha256": sha(baseline / "manifest.json"),
        "historical_baseline_files_sha256": {name: sha(baseline / name) for name in baseline_names},
        "historical_baseline_role": "sealed independent 128-step reference; no new paths or refitting",
        "old_256_pilot_excluded_from_fresh_estimates": True,
        "all_new_streams_declared_before_valuation": True, "no_optional_path_extension": True,
        "fixed_parameters_initial_factors_discount_dividends": True,
        "new_and_previous_rules_reapplied_at_each_bump": True,
        "no_valuation_selection_or_fitting": True, "all_paths_retained": True,
        "same_common_stopped_martingale": True, "independent_physical_resolutions": True,
        "optimal_delta_ready": False, "tolerances_changed": False, **TOLERANCES}
    design_path = folder / "design_before_valuation.json"
    if design_path.exists():
        previous = json.loads(design_path.read_text())
        if {key: value for key, value in previous.items() if key != "sealed_at_utc"} != design:
            raise ValueError("Changed time-precision design or numerical sources")
    else:
        write(design_path, {**design, "sealed_at_utc": now()})
    source_inventory = []
    for relative, digest in sources.items():
        target = folder / "source" / relative
        if target.exists():
            if sha(target) != digest: raise ValueError("Changed numerical source snapshot")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        source_inventory.append(identity(target))
    if not (folder / "source_inventory.json").exists():
        write(folder / "source_inventory.json", {"files": source_inventory})
    baseline_frame = historical_baseline(baseline, design)
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
        result = new_policy_time_samples(bundle, seed=design["seeds"][index],
            paths=design["paths"][index], steps=design["steps"][index])
        elapsed = perf_counter() - clock
        arrays = flatten_samples(result)
        validate_time_arrays(arrays, paths=design["paths"][index])
        estimates, comparisons = summarize_time_arrays(arrays)
        estimates["seed"] = design["seeds"][index]; estimates["steps"] = design["steps"][index]
        estimates.to_csv(folder / f"stream_{index}_estimates.csv", index=False)
        comparisons.to_csv(folder / f"stream_{index}_comparisons.csv", index=False)
        np.savez_compressed(folder / f"stream_{index}_samples.npz", **arrays)
        if {str(path.relative_to(ROOT)): sha(path) for path in source_paths} != sources:
            raise ValueError("Numerical source changed during time-precision measurement")
        files = [folder / f"stream_{index}_{suffix}" for suffix in ("estimates.csv", "comparisons.csv", "samples.npz")]
        record = {"index": index, "seed": design["seeds"][index], "paths": design["paths"][index],
            "steps": design["steps"][index], "started_at_utc": started, "finished_at_utc": now(),
            "elapsed_seconds": elapsed, "design_sha256": sha(design_path),
            "diagnostics": result["diagnostics"], "source_sha256": sources,
            "sources_unchanged": True, "files": [identity(path) for path in files]}
        write(record_path, record)
        print(json.dumps({"index": index, "elapsed_seconds": elapsed}), flush=True)
        print(estimates.to_string(index=False), flush=True)
    if args.collect:
        records, archives, per_seed = [], [], []
        for index in (0, 1, 2):
            record = json.loads((folder / f"stream_{index}.json").read_text())
            if record["design_sha256"] != sha(design_path): raise ValueError("Changed stream design")
            for item in record["files"]:
                if sha(item["path"]) != item["sha256"]: raise ValueError("Changed stream artifact")
            with np.load(folder / f"stream_{index}_samples.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            validate_time_arrays(arrays, paths=design["paths"][index])
            archives.append(arrays); records.append(record)
            per_seed.append(pd.read_csv(folder / f"stream_{index}_estimates.csv"))
        joined = {key: np.concatenate([archives[i][key] for i in (0, 1)], axis=0)
            for key in archives[0] if key != "relative_spots"}
        joined["relative_spots"] = archives[0]["relative_spots"]
        validate_time_arrays(joined, paths=8192)
        primary, paired = summarize_time_arrays(joined)
        confirmation, confirmation_paired = summarize_time_arrays(archives[2])
        tables = {"pooled_256_estimates.csv": primary, "pooled_256_comparisons.csv": paired,
            "confirmation_512_estimates.csv": confirmation,
            "confirmation_512_comparisons.csv": confirmation_paired,
            "historical_128_baseline.csv": baseline_frame,
            "independent_128_256_comparison.csv": independent_resolution_comparison(
                baseline_frame, primary, coarse_steps=128, fine_steps=256),
            "independent_256_512_comparison.csv": independent_resolution_comparison(
                primary, confirmation, coarse_steps=256, fine_steps=512),
            "per_seed.csv": pd.concat(per_seed, ignore_index=True)}
        for name, frame in tables.items(): frame.to_csv(folder / name, index=False)
        files = [folder / name for name in tables] + [folder / f"stream_{i}.json" for i in range(3)]
        manifest = {"version": design["version"], "complete": True,
            "design_sha256": sha(design_path), "source_sha256": sources, "records": records,
            "primary_paths": 8192, "primary_steps": 256, "primary_seed_count": 2,
            "confirmation_paths": 2048, "confirmation_steps": 512,
            "confirmation_seed_count": 1, "historical_baseline_paths": 8192,
            "historical_baseline_steps": 128, "stock_count": 5,
            "selected_policy_name": design["selected_policy_name"],
            "policy_labels": design["policy_labels"], "independent_paths": True,
            "all_paths_retained": True, "old_256_pilot_pooled": False,
            "new_rule_delta_measured": True, "optimal_delta_ready": False,
            "tolerances_changed": False, "files": [identity(path) for path in files],
            "limitations": ["Finite-bump sensitivity of a fixed lower policy, not optimal American delta.",
                "The old 256-step pilot is historical and excluded from all fresh estimates.",
                "Independent resolutions use the sum of variances, without falsely paired path indices.",
                "Individual sampling/bump/time diagnostics do not constitute a combined error certificate.",
                "Finite kernel, fixed exercise calendar and explicit no-cash synthetic inception case."]}
        write(folder / "manifest.json", manifest)
        for name in ("pooled_256_estimates.csv", "independent_128_256_comparison.csv", "independent_256_512_comparison.csv"):
            print(name, flush=True); print(tables[name].to_string(index=False), flush=True)


if __name__ == "__main__": main()
