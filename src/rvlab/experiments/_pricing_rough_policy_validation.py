"""Read-only validation of the newly selected rough stopping policy.

Numerical campaigns run explicitly through the bounded example runners.  This
collector checks their declared designs, model identities, complete path arrays,
statistics and unchanged acceptance rules before a notebook can display them.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import (
    _fixed_tolerances, _manifest, _sha, _stat, _table, _verify_file)


POLICIES = ("original_policy", "selected_policy")
ESTIMATORS = ("raw", "stopped martingale")
BUMPS = (.01, .005, .0025)
RELATIVE_SPOTS = np.array([-.01, -.005, -.0025, 0., .0025, .005, .01])
ROOT = Path(__file__).resolve().parents[3]


def _entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _compare_table(actual, expected, keys):
    """Compare complete row sets, including every failed estimator and gate."""
    if expected.empty:
        if not actual.empty:
            raise ValueError("Unexpected rows in a declared empty policy-validation table")
        return
    if (len(actual) != len(expected) or actual.duplicated(keys).any()
            or not set(expected).issubset(actual)):
        raise ValueError("Incomplete or duplicated policy-validation table")
    left = actual.sort_values(keys, na_position="last").reset_index(drop=True)
    right = expected.sort_values(keys, na_position="last").reset_index(drop=True)
    for name in right:
        if pd.api.types.is_numeric_dtype(right[name]):
            equal = np.allclose(left[name], right[name], rtol=1e-11, atol=2e-14, equal_nan=True)
        else:
            equal = left[name].fillna("<missing>").equals(right[name].fillna("<missing>"))
        if not equal:
            raise ValueError("Changed policy-validation statistic or identity: " + name)


def _validate_delta_arrays(arrays, paths):
    expected = {"relative_spots"}
    if not np.array_equal(arrays.get("relative_spots"), RELATIVE_SPOTS):
        raise ValueError("Changed new-policy stock bumps")
    for policy in POLICIES:
        expected.update(policy + "__" + name for name in ("raw_prices", "controlled_prices", "stops"))
        stops = arrays.get(policy + "__stops", np.array([]))
        if (stops.shape != (paths, 7) or not np.issubdtype(stops.dtype, np.integer)
                or np.any(stops < 0) or np.any(stops > 64)):
            raise ValueError("Invalid new-policy exercise indices")
        for estimator, name in zip(ESTIMATORS, ("raw_prices", "controlled_prices")):
            prices = arrays.get(policy + "__" + name, np.array([]))
            if prices.shape != (paths, 7) or not np.isfinite(prices).all():
                raise ValueError("Missing or nonfinite new-policy price paths")
            if estimator == "raw" and np.any(prices < 0):
                raise ValueError("Raw vanilla stopping payoffs must be nonnegative")
            for bump in BUMPS:
                key = policy + "__" + f"{estimator}:{bump:g}"
                expected.add(key)
                values = arrays.get(key, np.array([]))
                lo = np.flatnonzero(RELATIVE_SPOTS == -bump)[0]
                hi = np.flatnonzero(RELATIVE_SPOTS == bump)[0]
                if (values.shape != (paths,) or not np.isfinite(values).all()
                        or not np.allclose(values, (prices[:, hi] - prices[:, lo]) / (200*bump),
                            rtol=0., atol=2e-14)):
                    raise ValueError("Delta differs from its complete paired stock-price paths")
    if set(arrays) != expected:
        raise ValueError("Missing or unexpected new-policy sample arrays")


def _delta_statistics(arrays):
    estimates, comparisons = [], []
    for policy in POLICIES:
        for estimator in ESTIMATORS:
            for bump in BUMPS:
                sample = arrays[policy + "__" + f"{estimator}:{bump:g}"]
                stat = _stat(sample)
                allowance = .002 + .005*abs(stat["mean"])
                estimates.append({"policy": policy, "estimator": estimator,
                    "bump_relative": bump, "paths": len(sample), **stat,
                    "delta_allowance": allowance, "sampling_passed": stat["ci_half_width"] <= allowance,
                    "optimal_delta_ready": False})
            for coarse, fine in zip(BUMPS[:-1], BUMPS[1:]):
                fine_values = arrays[policy + "__" + f"{estimator}:{fine:g}"]
                stat = _stat(fine_values - arrays[policy + "__" + f"{estimator}:{coarse:g}"])
                allowance = .002 + .005*abs(float(fine_values.mean()))
                budget = abs(stat["mean"]) + stat["ci_half_width"]
                comparisons.append({"comparison": "bump refinement", "policy": policy,
                    "estimator": estimator, "coarse_bump": coarse, "fine_bump": fine, **stat,
                    "delta_allowance": allowance, "change_plus_uncertainty": budget,
                    "comparison_passed": budget <= allowance})
    for estimator in ESTIMATORS:
        for bump in BUMPS:
            stat = _stat(arrays["selected_policy__" + f"{estimator}:{bump:g}"]
                - arrays["original_policy__" + f"{estimator}:{bump:g}"])
            comparisons.append({"comparison": "selected minus original", "policy": "paired policies",
                "estimator": estimator, "coarse_bump": np.nan, "fine_bump": bump, **stat})
    return pd.DataFrame(estimates), pd.DataFrame(comparisons)


def _time_comparison(base, refined):
    rows = []
    for fine in refined.itertuples():
        coarse = base.loc[(base.policy == fine.policy) & (base.estimator == fine.estimator)
            & (base.bump_relative == fine.bump_relative)].iloc[0]
        change = fine.mean - coarse["mean"]
        se = float(np.hypot(fine.standard_error, coarse.standard_error))
        allowance = .002 + .005*abs(fine.mean)
        budget = abs(change) + 1.96*se
        rows.append({"policy": fine.policy, "estimator": fine.estimator,
            "bump_relative": fine.bump_relative, "coarse_steps": 128, "fine_steps": 256,
            "mean": change, "standard_error": se, "ci_half_width": 1.96*se,
            "delta_allowance": allowance, "change_plus_uncertainty": budget,
            "comparison_passed": budget <= allowance, "independent_streams": True,
            "same_exercise_calendar": True, "optimal_delta_ready": False})
    return pd.DataFrame(rows)


def _delta_campaign(folder):
    folder = Path(folder).resolve()
    required = {"pooled_estimates.csv", "pooled_comparisons.csv", "time_estimates.csv",
        "time_comparisons.csv", "independent_time_comparison.csv", "per_seed.csv"}
    required.update(f"stream_{i}.json" for i in range(3))
    manifest, entries = _manifest(folder, required_files=required)
    design_path = folder / "design_before_valuation.json"
    design = json.loads(design_path.read_text())
    if (design.get("version") != "rough_new_policy_delta_v1"
            or design.get("seeds") != [810005042, 811005042, 812005042]
            or design.get("paths") != [4096, 4096, 2048]
            or design.get("steps") != [128, 128, 256]
            or design.get("exercise_steps") != 64 or design.get("bump_relatives") != list(BUMPS)
            or manifest.get("complete") is not True or manifest.get("primary_paths") != 8192
            or manifest.get("primary_seed_count") != 2 or manifest.get("time_paths") != 2048
            or manifest.get("time_seed_count") != 1 or manifest.get("optimal_delta_ready") is not False
            or manifest.get("independent_paths") is not True or manifest.get("all_paths_retained") is not True
            or manifest.get("new_rule_delta_measured") is not True):
        raise ValueError("Changed or incomplete independent new-policy delta design")
    _fixed_tolerances(design, required=True)
    for name in ("fixed_parameters_initial_factors_discount_dividends",
            "new_and_previous_rules_reapplied_at_each_bump", "no_valuation_selection_or_fitting",
            "all_paths_retained", "same_common_stopped_martingale"):
        if design.get(name) is not True:
            raise ValueError("New-policy delta violates the frozen-policy valuation contract")
    source_paths = {str(path.relative_to(ROOT)) for path in (ROOT / "src/rvlab/pricing").glob("*.py")}
    source_paths.update("src/rvlab/experiments/" + name for name in (
        "_pricing_rough_large_policy.py", "_pricing_rough_new_policy_delta.py",
        "_pricing_rough_stock_delta.py", "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py"))
    source_paths.add("examples/rough_new_policy_delta.py")
    if set(design.get("source_sha256", {})) != source_paths:
        raise ValueError("New-policy design omits a numerical source")
    for name, digest in design["source_sha256"].items():
        snapshot = folder / "source" / name
        if _sha(snapshot) != digest:
            raise ValueError("Changed new-policy numerical source snapshot")
        entries.append(_entry(snapshot))
    bundle = folder.parents[1] / "rough_larger_policy_v1/32768_paths_two_starts"
    seal_path, arrays_path = bundle / "frozen_policy_before_final.json", bundle / "frozen_policy_coefficients.npz"
    seal = json.loads(seal_path.read_text())
    labels = {"original_policy": "previous dense tensor", "selected_policy": "dense tensor ridge1e-5"}
    if (design.get("selected_policy_name") != "dense tensor ridge1e-5"
            or design.get("policy_labels") != labels or manifest.get("policy_labels") != labels
            or manifest.get("selected_policy_name") != design["selected_policy_name"]
            or design.get("new_policy_seal_sha256") != _sha(seal_path)
            or design.get("new_policy_arrays_sha256") != _sha(arrays_path)
            or seal.get("selected_name") != design["selected_policy_name"]
            or seal.get("all_selection_training_seeds") != design.get("all_selection_training_seeds")
            or set(design["seeds"]) & set(design["all_selection_training_seeds"])):
        raise ValueError("Changed newly selected delta policy or training/valuation separation")
    entries.extend((_entry(seal_path), _entry(arrays_path)))
    histories = Path(design["historical_bundle_path"])
    if _sha(histories / "manifest.json") != design["historical_bundle_sha256"]:
        raise ValueError("Changed historical model/control bundle")
    entries.append(_entry(histories / "manifest.json"))
    historical = json.loads((histories / "manifest.json").read_text())
    if _sha(histories / "coefficients.npz") != historical["coefficient_archive_sha256"]:
        raise ValueError("Changed historical model/control coefficients")
    entries.append(_entry(histories / "coefficients.npz"))
    archives, stream_tables = [], []
    records = []
    for i, (seed, paths, steps) in enumerate(zip(design["seeds"], design["paths"], design["steps"])):
        path = folder / f"stream_{i}.json"
        record = json.loads(path.read_text())
        if (record.get("index") != i or record.get("seed") != seed or record.get("paths") != paths
                or record.get("steps") != steps or record.get("design_sha256") != _sha(design_path)
                or record.get("source_sha256") != design["source_sha256"]
                or record.get("sources_unchanged") is not True):
            raise ValueError("Changed new-policy valuation stream")
        diagnostics = record.get("diagnostics", {})
        expected_diagnostics = {"seed": seed, "paths": paths, "steps": steps, "actual_time_steps": steps,
            "independent_training_valuation": True, "valuation_used_for_fitting": False,
            "fixed_parameters_factors_discount_dividends": True,
            "decisions_reapplied_at_each_bump": True, "stopping_times_held_fixed": False,
            "same_common_martingale_for_policies": True,
            "selected_policy_name": design["selected_policy_name"],
            "policy_selection_used_delta_paths": False, "optimal_value_delta_certified": False,
            "optimal_delta_ready": False, "tolerances_changed": False}
        if any(diagnostics.get(key) != value for key, value in expected_diagnostics.items()):
            raise ValueError("Changed new-policy decision, control or fixed-input diagnostics")
        filenames = {f"stream_{i}_{suffix}" for suffix in ("samples.npz", "estimates.csv", "comparisons.csv")}
        if len(record.get("files", [])) != 3 or {Path(r["path"]).name for r in record["files"]} != filenames:
            raise ValueError("New-policy stream omits a required artifact")
        for item in record["files"]:
            p = folder / Path(item["path"]).name
            _verify_file(p, item)
            entries.append(_entry(p))
        with np.load(folder / f"stream_{i}_samples.npz", allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        _validate_delta_arrays(arrays, paths)
        estimates, comparisons = _delta_statistics(arrays)
        estimates["seed"], estimates["steps"] = seed, steps
        _compare_table(_table(folder / f"stream_{i}_estimates.csv", entries), estimates,
            ["policy", "estimator", "bump_relative"])
        _compare_table(_table(folder / f"stream_{i}_comparisons.csv", entries), comparisons,
            ["comparison", "policy", "estimator", "coarse_bump", "fine_bump"])
        archives.append(arrays)
        stream_tables.append(estimates)
        records.append(record)
    if manifest.get("records") != records:
        raise ValueError("Manifest differs from the measured new-policy streams")
    joined = {key: np.concatenate([a[key] for a in archives[:2]]) for key in archives[0] if key != "relative_spots"}
    joined["relative_spots"] = RELATIVE_SPOTS
    pooled, paired = _delta_statistics(joined)
    time, time_paired = _delta_statistics(archives[2])
    changes = _time_comparison(pooled, time)
    tables = {"delta_estimates": pooled, "delta_comparisons": paired,
        "delta_time_estimates": time, "delta_time_bump_comparisons": time_paired,
        "delta_time_comparisons": changes, "delta_stream_estimates": pd.concat(stream_tables, ignore_index=True)}
    names = {"delta_estimates": "pooled_estimates.csv", "delta_comparisons": "pooled_comparisons.csv",
        "delta_time_estimates": "time_estimates.csv", "delta_time_bump_comparisons": "time_comparisons.csv",
        "delta_time_comparisons": "independent_time_comparison.csv", "delta_stream_estimates": "per_seed.csv"}
    for key, value in tables.items():
        columns = (["comparison", "policy", "estimator", "coarse_bump", "fine_bump"]
            if "comparison" in value else ["policy", "estimator", "bump_relative"])
        if "seed" in value:
            columns.append("seed")
        _compare_table(_table(folder / names[key], entries), value, columns)
    one_sided = []
    for policy in POLICIES:
        stops = joined[policy + "__stops"]
        for estimator, price_key in zip(ESTIMATORS, ("raw_prices", "controlled_prices")):
            prices = joined[policy + "__" + price_key]
            for bump in BUMPS:
                lo, hi = np.flatnonzero(RELATIVE_SPOTS == -bump)[0], np.flatnonzero(RELATIVE_SPOTS == bump)[0]
                for side, values in (("left", (prices[:, 3] - prices[:, lo])/(100*bump)),
                        ("right", (prices[:, hi] - prices[:, 3])/(100*bump))):
                    one_sided.append({"policy": policy, "estimator": estimator,
                        "bump_relative": bump, "side": side, "paths": len(prices), **_stat(values),
                        "bumped_stopping_change_fraction": float(np.mean(stops[:, lo] != stops[:, hi])),
                        "stopping_times_reapplied": True, "optimal_delta_ready": False})
    tables["delta_one_sided_bumps"] = pd.DataFrame(one_sided)
    return tables, entries, manifest


def policy_validation_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The declared research seed is 42")
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    root = Path(context.output_dir).resolve()
    context.progress("Verifying newly selected policy deltas and independent stopping-gap trials")
    tables, entries, delta_manifest = _delta_campaign(root / "rough_new_policy_delta_v1/new_selected_2x4096")
    from ._pricing_rough_policy_validation_stopping import load_stopping_campaign
    stopping, files, stopping_manifest = load_stopping_campaign(root / "rough_policy_validation_v1/local_control_v1")
    tables.update(stopping)
    entries.extend(files)
    checked = []
    for name, gate in (("delta_estimates", "sampling_passed"),
            ("delta_time_estimates", "sampling_passed"),
            ("delta_comparisons", "comparison_passed"),
            ("delta_time_bump_comparisons", "comparison_passed"),
            ("delta_time_comparisons", "comparison_passed")):
        frame = tables[name]
        rows = frame.loc[(frame.policy == "selected_policy") & (frame.estimator == "stopped martingale")]
        checked.extend(bool(value) for value in rows[gate])
    tables["configuration"] = pd.DataFrame([
        {"item": "Exercise policy", "value": "sealed dense tensor ridge1e-5; no new policy fitting or selection"},
        {"item": "Stock delta", "value": "8,192 fresh primary paths at 128 physical steps; 2,048 at 256; same 64 exercise intervals"},
        {"item": "Sensitivity target", "value": "frozen lower-policy finite stock differences, own stopping decisions at each bump"},
        {"item": "Stopping correction", "value": "predictable LEFT gates; RIGHT factor reliability integrated under exact composed weak law"},
        {"item": "Price allowance", "value": ".001+1e-4*abs(V); gap+1.96gapSE+1.96lowerSE"},
        {"item": "Delta allowance", "value": ".002+.005*abs(delta); correlated individual sampling/bump/time checks"},
        {"item": "Spread allowance", "value": "10% of effective half-spread unchanged; no market spread in this synthetic case"},
        {"item": "Scope", "value": "60-day ATM put, synthetic inception, declared empty cash schedule, fixed finite kernel/law/calendar"},
        {"item": "Real SPY", "value": "independent spot, historical dividend expectations and rough-factor states remain unresolved"}])
    tables["execution_costs"] = pd.DataFrame([
        {"campaign": "new policy delta", "stage": "primary" if i < 2 else "independent time refinement",
            "seed": record["seed"], "paths": record["paths"], "steps": record["steps"],
            "elapsed_seconds": record["elapsed_seconds"], "started_at_utc": record["started_at_utc"],
            "finished_at_utc": record["finished_at_utc"], "overlapping_workers": True,
            "runtime_scope": "individual task elapsed counter; not aggregate serial wall-clock"}
        for i, record in enumerate(delta_manifest["records"])])
    tables["execution_costs"] = pd.concat([tables["execution_costs"], pd.DataFrame([{
        "campaign": "local control", "stage": "pilot+fitting+selection+two final streams",
        "elapsed_seconds": stopping_manifest["elapsed_seconds"],
        "started_at_utc": stopping_manifest["created_at"], "finished_at_utc": stopping_manifest["completed_at"],
        "overlapping_workers": True, "runtime_scope": "campaign elapsed counter; concurrent delta tasks separate"}])], ignore_index=True)
    inventory = {}
    for item in entries:
        path = str(Path(item["path"]).resolve())
        normalized = {**item, "path": path}
        if path in inventory and inventory[path] != normalized:
            raise ValueError("Conflicting identities for the same numerical input file")
        inventory[path] = normalized
    entries = [inventory[name] for name in sorted(inventory)]
    tables["provenance"] = pd.DataFrame(entries)
    from ._pricing_rough_foundation import _fingerprint
    analysis_sources = {name: _sha(Path(__file__).with_name(name)) for name in (
        "_pricing_rough_policy_validation.py", "_pricing_rough_policy_validation_stopping.py",
        "_pricing_rough_stopping_delta.py")}
    profile = _fingerprint({"version": "rough_policy_validation_v1", "files": entries,
        "analysis_sources": analysis_sources})
    manifest = {"backend": "rough_heston_policy_validation", "study_version": "rough_policy_validation_v1",
        "profile_fingerprint": profile, "input_files": entries, "analysis_sources": analysis_sources,
        "primary_complete": True, "historical_replay": False, "current_engine_execution": False,
        "execution_mode": "verified collection of current independent numerical campaigns",
        "new_policy_name": delta_manifest["selected_policy_name"], "new_policy_delta_measured": True,
        "new_policy_individual_delta_checks_passed": bool(checked and all(checked)),
        "individual_delta_check_count": len(checked), "delta_primary_paths": 8192, "delta_time_paths": 2048,
        "stopping_primary_paths": 8192, "stopping_selected_control": stopping_manifest["selected"],
        "primary_policy_fits": 0, "primary_dual_optimizer_calls": len(stopping["stopping_trials"]),
        "exploratory_dual_optimizer_calls": len(stopping["stopping_pilot_trials"]),
        "tolerances_changed": False, "american_pricing_ready": False, "stock_delta_ready": False,
        "new_stopping_control_delta_validated": False,
        "limitations": ["Stock delta targets one sealed lower exercise policy under a fixed finite numerical law.",
            "Individual sampling, stock-bump and time checks are correlated and do not form a total error certificate.",
            "A new martingale changes lower sample variance, not the exercise policy or its expected payoff.",
            "The new stopping estimator has no inherited stock-delta uncertainty check.",
            "Stopping-value intervals do not bound the derivative of optimal American value."]}
    result = {"tables": tables, "figures": {}, "manifest": manifest}
    if compute_figures:
        from ._pricing_rough_policy_validation_reporting import add_policy_validation_reporting
        result = add_policy_validation_reporting(result)
    return result
