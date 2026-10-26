"""Collect sealed stock-grid and weak-time experiments without simulation."""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import _fixed_tolerances, _manifest, _sha, _stat, _table, _verify_file
from ._pricing_rough_policy_validation import _compare_table, _entry, _delta_campaign


ROOT = Path(__file__).resolve().parents[3]
POLICIES = ("original_policy", "selected_policy")
ESTIMATORS = ("raw", "stopped martingale")
BUMPS = (.005, .0025)
SPOTS = np.array([-.005, -.0025, 0., .0025, .005])


def _time_arrays(arrays, paths):
    names = {"relative_spots"}
    if not np.array_equal(arrays.get("relative_spots"), SPOTS):
        raise ValueError("Changed the five declared time-study stock levels")
    for policy in POLICIES:
        names.update(policy+"__"+key for key in ("raw_prices", "controlled_prices", "stops"))
        stops = arrays.get(policy+"__stops", np.array([]))
        if (stops.shape != (paths, 5) or not np.issubdtype(stops.dtype, np.integer)
                or np.any(stops < 0) or np.any(stops > 64)):
            raise ValueError("Missing paths or invalid time-study exercise decisions")
        for estimator, field in zip(ESTIMATORS, ("raw_prices", "controlled_prices")):
            prices = arrays.get(policy+"__"+field, np.array([]))
            if (prices.shape != (paths, 5) or not np.isfinite(prices).all()
                    or (estimator == "raw" and np.any(prices < 0))):
                raise ValueError("Invalid five-stock time-study price paths")
            for bump in BUMPS:
                key = policy+"__"+f"{estimator}:{bump:g}"
                names.add(key)
                lo, hi = np.flatnonzero(SPOTS == -bump)[0], np.flatnonzero(SPOTS == bump)[0]
                delta = arrays.get(key, np.array([]))
                if (delta.shape != (paths,) or not np.isfinite(delta).all()
                        or not np.allclose(delta, (prices[:, hi]-prices[:, lo])/(200*bump), rtol=0., atol=2e-14)):
                    raise ValueError("Time-study delta does not equal its own paired stock-price differences")
    if set(arrays) != names:
        raise ValueError("Time-study sample arrays are incomplete or contain unexpected estimators")


def _time_statistics(arrays):
    rows, changes = [], []
    for policy in POLICIES:
        for estimator in ESTIMATORS:
            for bump in BUMPS:
                values = arrays[policy+"__"+f"{estimator}:{bump:g}"]
                stat = _stat(values)
                allowance = .002+.005*abs(stat["mean"])
                rows.append({"policy": policy, "estimator": estimator, "bump_relative": bump,
                    "paths": len(values), **stat, "delta_allowance": allowance,
                    "sampling_passed": stat["ci_half_width"] <= allowance, "optimal_delta_ready": False})
            fine = arrays[policy+"__"+f"{estimator}:0.0025"]
            coarse = arrays[policy+"__"+f"{estimator}:0.005"]
            stat = _stat(fine-coarse)
            allowance = .002+.005*abs(float(fine.mean()))
            budget = abs(stat["mean"])+stat["ci_half_width"]
            changes.append({"comparison": "bump refinement", "policy": policy,
                "estimator": estimator, "coarse_bump": .005, "fine_bump": .0025, **stat,
                "delta_allowance": allowance, "change_plus_uncertainty": budget,
                "comparison_passed": budget <= allowance})
    for estimator in ESTIMATORS:
        for bump in BUMPS:
            key = f"{estimator}:{bump:g}"
            stat = _stat(arrays["selected_policy__"+key]-arrays["original_policy__"+key])
            changes.append({"comparison": "selected minus original", "policy": "paired policies",
                "estimator": estimator, "coarse_bump": np.nan, "fine_bump": bump, **stat})
    return pd.DataFrame(rows), pd.DataFrame(changes)


def _resolution_comparison(coarse, fine, coarse_steps, fine_steps):
    rows = []
    for f in fine.itertuples():
        match = coarse.loc[(coarse.policy == f.policy) & (coarse.estimator == f.estimator)
            & (coarse.bump_relative == f.bump_relative)]
        if len(match) != 1:
            raise ValueError("Missing or duplicated independent-resolution reference")
        c = match.iloc[0]
        change = f.mean-c["mean"]
        se = float(np.hypot(f.standard_error, c.standard_error))
        allowance = .002+.005*abs(f.mean)
        budget = abs(change)+1.96*se
        rows.append({"policy": f.policy, "estimator": f.estimator, "bump_relative": f.bump_relative,
            "coarse_steps": coarse_steps, "fine_steps": fine_steps, "coarse_paths": int(c.paths),
            "fine_paths": f.paths, "mean": change, "standard_error": se, "ci_half_width": 1.96*se,
            "delta_allowance": allowance, "change_plus_uncertainty": budget,
            "comparison_passed": budget <= allowance, "independent_streams": True,
            "same_exercise_calendar": True, "optimal_delta_ready": False})
    return pd.DataFrame(rows)


def _collect_time(folder):
    folder = Path(folder).resolve()
    csv_names = {"pooled_256_estimates.csv", "pooled_256_comparisons.csv",
        "confirmation_512_estimates.csv", "confirmation_512_comparisons.csv", "historical_128_baseline.csv",
        "independent_128_256_comparison.csv", "independent_256_512_comparison.csv", "per_seed.csv"}
    required = csv_names | {f"stream_{i}.json" for i in range(3)}
    manifest, entries = _manifest(folder, required_files=required)
    design_path = folder/"design_before_valuation.json"
    design = json.loads(design_path.read_text())
    expected = {"version": "rough_time_precision_v1", "seeds": [910005042, 911005042, 912005042],
        "paths": [4096, 4096, 2048], "steps": [256, 256, 512], "exercise_steps": 64,
        "bump_relatives": list(BUMPS), "stock_count": 5, "selected_policy_name": "dense tensor ridge1e-5",
        "historical_baseline_steps": 128, "historical_baseline_paths": 8192,
        "historical_baseline_seeds": [810005042, 811005042],
        "old_256_pilot_excluded_from_fresh_estimates": True, "all_new_streams_declared_before_valuation": True,
        "no_optional_path_extension": True, "fixed_parameters_initial_factors_discount_dividends": True,
        "new_and_previous_rules_reapplied_at_each_bump": True, "no_valuation_selection_or_fitting": True,
        "all_paths_retained": True, "same_common_stopped_martingale": True,
        "independent_physical_resolutions": True, "optimal_delta_ready": False, "tolerances_changed": False}
    if any(design.get(k) != v for k, v in expected.items()):
        raise ValueError("Changed or incomplete predeclared time-precision study")
    _fixed_tolerances(design, required=True)
    expected_manifest = {"version": "rough_time_precision_v1", "selected_policy_name": "dense tensor ridge1e-5",
        "complete": True, "primary_paths": 8192, "primary_steps": 256,
        "primary_seed_count": 2, "confirmation_paths": 2048, "confirmation_steps": 512,
        "confirmation_seed_count": 1, "historical_baseline_paths": 8192, "historical_baseline_steps": 128,
        "stock_count": 5, "independent_paths": True, "all_paths_retained": True,
        "old_256_pilot_pooled": False, "new_rule_delta_measured": True, "optimal_delta_ready": False}
    if any(manifest.get(k) != v for k, v in expected_manifest.items()):
        raise ValueError("The time-precision manifest omits or changes declared paths or acceptance scope")
    sources = {str(p.relative_to(ROOT)) for p in (ROOT/"src/rvlab/pricing").glob("*.py")}
    sources.update("src/rvlab/experiments/"+name for name in ("_pricing_rough_large_policy.py",
        "_pricing_rough_time_precision.py", "_pricing_rough_stock_delta.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py"))
    sources.add("examples/rough_time_precision.py")
    if set(design.get("source_sha256", {})) != sources:
        raise ValueError("The time-precision design omits a numerical source")
    for name, digest in design["source_sha256"].items():
        p = folder/"source"/name
        if _sha(p) != digest:
            raise ValueError("Changed time-precision source snapshot")
        entries.append(_entry(p))
    source_inventory = folder/"source_inventory.json"
    source_rows = json.loads(source_inventory.read_text()).get("files", [])
    expected_source_rows = [_entry(folder/"source"/name) for name in design["source_sha256"]]
    if (len(source_rows) != len(expected_source_rows)
            or sorted(source_rows, key=lambda r: r["path"]) != sorted(expected_source_rows, key=lambda r: r["path"])):
        raise ValueError("The time-study source inventory differs from the executed source snapshots")
    environment_path = folder/"execution_environment.json"
    environment = json.loads(environment_path.read_text())
    if (environment.get("greek_worker_count") != 2
            or any(environment.get("environment", {}).get(name) != "1"
                for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            or any(not environment.get(name) for name in ("interpreter", "python", "platform", "numpy", "scipy", "pandas"))):
        raise ValueError("Missing or changed declared time-study execution environment")
    entries.extend((_entry(source_inventory), _entry(environment_path)))
    output = folder.parents[1]
    bundle = output/"rough_larger_policy_v1/32768_paths_two_starts"
    model_path, coefficient_path = bundle/"frozen_policy_before_final.json", bundle/"frozen_policy_coefficients.npz"
    model = json.loads(model_path.read_text())
    labels = {"original_policy": "previous dense tensor", "selected_policy": "dense tensor ridge1e-5"}
    if (design["new_policy_seal_sha256"] != _sha(model_path)
            or design["new_policy_arrays_sha256"] != _sha(coefficient_path)
            or design.get("policy_labels") != labels or manifest.get("policy_labels") != labels
            or model["selected_name"] != design["selected_policy_name"]
            or design.get("historical_bundle_path") != model.get("historical_bundle_path")
            or design.get("historical_bundle_sha256") != model.get("historical_bundle_sha256")
            or design["all_selection_training_seeds"] != model["all_selection_training_seeds"]
            or set(design["seeds"]) & set(model["all_selection_training_seeds"])):
        raise ValueError("Changed time-study model identity or reused fitting/selection paths")
    entries.extend((_entry(model_path), _entry(coefficient_path)))
    baseline_folder = output/"rough_new_policy_delta_v1/new_selected_2x4096"
    if Path(design["historical_baseline_path"]).resolve() != baseline_folder:
        raise ValueError("The declared historical reference belongs to a different experiment")
    if _sha(baseline_folder/"manifest.json") != design["historical_baseline_manifest_sha256"]:
        raise ValueError("Historical reference manifest changed after the new design was sealed")
    baseline_names = {"pooled_estimates.csv", "preservation_inventory.json", "stream_0_samples.npz",
        "stream_1_samples.npz", "stream_0.json", "stream_1.json"}
    if set(design["historical_baseline_files_sha256"]) != baseline_names:
        raise ValueError("Incomplete historical time-reference fingerprint")
    for name, digest in design["historical_baseline_files_sha256"].items():
        if _sha(baseline_folder/name) != digest:
            raise ValueError("Changed sealed historical time-reference artifact")
        entries.append(_entry(baseline_folder/name))
    historical, old_files, _ = _delta_campaign(baseline_folder)
    entries.extend(old_files)
    baseline = historical["delta_estimates"].loc[lambda t: t.bump_relative.isin(BUMPS)].copy()
    old256 = historical["delta_time_estimates"].loc[lambda t: t.bump_relative.isin(BUMPS)].copy()
    records, archives, per_seed = [], [], []
    for i, (seed, paths, steps) in enumerate(zip(design["seeds"], design["paths"], design["steps"])):
        record = json.loads((folder/f"stream_{i}.json").read_text())
        if (record.get("index") != i or record.get("seed") != seed or record.get("paths") != paths
                or record.get("steps") != steps or record.get("design_sha256") != _sha(design_path)
                or record.get("source_sha256") != design["source_sha256"]
                or record.get("sources_unchanged") is not True):
            raise ValueError("Time-precision stream differs from its sealed design")
        flags = {"seed": seed, "paths": paths, "steps": steps, "actual_time_steps": steps,
            "independent_training_valuation": True, "valuation_used_for_fitting": False,
            "fixed_parameters_factors_discount_dividends": True, "decisions_reapplied_at_each_bump": True,
            "stopping_times_held_fixed": False, "same_common_martingale_for_policies": True,
            "selected_policy_name": design["selected_policy_name"], "policy_selection_used_delta_paths": False,
            "optimal_value_delta_certified": False, "old_256_pilot_pooled": False,
            "five_stock_bumps": True, "fresh_resolutions_independent": True,
            "optimal_delta_ready": False, "tolerances_changed": False}
        if any(record.get("diagnostics", {}).get(k) != v for k, v in flags.items()):
            raise ValueError("Changed time-study exercise, grid or fixed-input diagnostics")
        names = {f"stream_{i}_{suffix}" for suffix in ("samples.npz", "estimates.csv", "comparisons.csv")}
        if len(record.get("files", [])) != 3 or {Path(r["path"]).name for r in record["files"]} != names:
            raise ValueError("Time-study stream omits required path artifacts")
        for row in record["files"]:
            p = folder/Path(row["path"]).name
            _verify_file(p, row)
            entries.append(_entry(p))
        with np.load(folder/f"stream_{i}_samples.npz", allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        _time_arrays(arrays, paths)
        estimates, changes = _time_statistics(arrays)
        estimates["seed"], estimates["steps"] = seed, steps
        _compare_table(_table(folder/f"stream_{i}_estimates.csv", entries), estimates,
            ["policy", "estimator", "bump_relative"])
        _compare_table(_table(folder/f"stream_{i}_comparisons.csv", entries), changes,
            ["comparison", "policy", "estimator", "coarse_bump", "fine_bump"])
        records.append(record)
        archives.append(arrays)
        per_seed.append(estimates)
    if records != manifest.get("records"):
        raise ValueError("Time-study manifest does not match its measured streams")
    joined = {key: np.concatenate([a[key] for a in archives[:2]]) for key in archives[0] if key != "relative_spots"}
    primary, paired = _time_statistics(joined)
    fine, fine_paired = _time_statistics(archives[2])
    changes = _resolution_comparison(baseline, primary, 128, 256)
    finer = _resolution_comparison(primary, fine, 256, 512)
    tables = {"delta_estimates": primary, "delta_512_estimates": fine,
        "delta_bump_comparisons": paired, "delta_512_bump_comparisons": fine_paired,
        "delta_time_comparisons": changes, "delta_finer_time_comparisons": finer,
        "delta_historical_128": baseline, "delta_stream_estimates": pd.concat(per_seed, ignore_index=True)}
    csv_map = dict(zip(tables, ("pooled_256_estimates.csv", "confirmation_512_estimates.csv",
        "pooled_256_comparisons.csv", "confirmation_512_comparisons.csv", "independent_128_256_comparison.csv",
        "independent_256_512_comparison.csv", "historical_128_baseline.csv", "per_seed.csv")))
    for name, table in tables.items():
        keys = (["comparison", "policy", "estimator", "coarse_bump", "fine_bump"] if "comparison" in table
            else ["policy", "estimator", "bump_relative"])
        if "seed" in table:
            keys.append("seed")
        _compare_table(_table(folder/csv_map[name], entries), table, keys)
    grids = []
    for frame, steps, role, is_new, acceptance in ((baseline, 128, "historical independent baseline", False, "reference with propagated uncertainty"),
            (old256, 256, "historical 2048-path precision comparator", False, "precision comparison only; excluded from new estimates"),
            (primary, 256, "new 8192-path primary", True, "new primary estimate"),
            (fine, 512, "new 2048-path confirmation", True, "new resolution confirmation")):
        grids.append(frame.assign(steps=steps, grid_role=role, new_paths=is_new, acceptance_role=acceptance))
    tables["delta_grid_estimates"] = pd.concat(grids, ignore_index=True, sort=False)
    precision = []
    for row in primary.itertuples():
        old = old256.loc[(old256.policy == row.policy) & (old256.estimator == row.estimator)
            & (old256.bump_relative == row.bump_relative)].iloc[0]
        precision.append({"policy": row.policy, "estimator": row.estimator, "bump_relative": row.bump_relative,
            "old_paths": 2048, "new_paths": 8192, "old_ci_half_width": old.ci_half_width,
            "new_ci_half_width": row.ci_half_width, "standard_error_ratio": row.standard_error/old.standard_error,
            "old_256_pooled": False, "acceptance_input": False})
    tables["delta_precision_comparison"] = pd.DataFrame(precision)
    return tables, entries, manifest


def gap_time_refinement_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The declared research seed is 42")
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    root = Path(context.output_dir).resolve()
    context.progress("Verifying independent stock-grid corrections and fresh 256/512-step delta paths")
    tables, entries, time_manifest = _collect_time(root/"rough_time_precision_v1/fresh256_and512")
    from ._pricing_rough_gap_time_stopping import collect_grid_campaign
    stopping, files, stopping_manifest = collect_grid_campaign(root/"rough_gap_time_refinement_v1/stock_grid_control_v2")
    tables.update(stopping)
    entries.extend(files)
    return _finish_collection(tables, entries, time_manifest, stopping_manifest, compute_figures=compute_figures)


def _finish_collection(tables, entries, time_manifest, stopping_manifest, *, compute_figures):
    from ._pricing_rough_foundation import _fingerprint
    checks = []
    for name, gate in (("delta_estimates", "sampling_passed"), ("delta_512_estimates", "sampling_passed"),
            ("delta_bump_comparisons", "comparison_passed"), ("delta_512_bump_comparisons", "comparison_passed"),
            ("delta_time_comparisons", "comparison_passed"), ("delta_finer_time_comparisons", "comparison_passed")):
        rows = tables[name]
        rows = rows.loc[(rows.policy == "selected_policy") & (rows.estimator == "stopped martingale")]
        checks.extend(bool(value) for value in rows[gate])
    if len(checks) != 10:
        raise ValueError("The new frozen-policy diagnostics omit a declared sampling, bump or time check")
    tables["configuration"] = pd.DataFrame([
        {"item": "Policy and inputs", "value": "fixed ridge1e-5 rule; same synthetic inception factors, cash schedule and discount curve"},
        {"item": "New delta paths", "value": "two fresh 4096-path streams at 256 steps; independent 2048-path confirmation at 512"},
        {"item": "Historical reference", "value": "sealed 128-step / 8192-path baseline with its uncertainty propagated"},
        {"item": "Historical 256-step paths", "value": "2048-path precision comparator only; excluded from new estimates and selection"},
        {"item": "Exercise and stock bumps", "value": "same 64 exercise intervals; 0.5% and 0.25% symmetric bumps; own decisions at every stock"},
        {"item": "Stopping controls", "value": "prior selected local martingale held fixed; additional centered refinements of its stock-value grid"},
        {"item": "Price allowance", "value": ".001+1e-4*abs(V); gap+1.96gapSE+1.96lowerSE"},
        {"item": "Delta allowance", "value": ".002+.005*abs(delta); independent physical resolutions, paired stock bumps"},
        {"item": "Spread allowance", "value": "10% effective half-spread unchanged; no bid/ask data in this synthetic case"},
        {"item": "Scope", "value": "finite kernel, unchanged exercise calendar and weak laws; no optimal-value delta certificate"},
        {"item": "Real SPY", "value": "independent spot, historical dividend expectations and conditional rough factor states remain separate blockers"}])
    costs = [{"campaign": "time precision", "stage": "fresh 256-step primary" if i < 2 else "fresh 512-step confirmation",
        "seed": record["seed"], "paths": record["paths"], "steps": record["steps"],
        "elapsed_seconds": record["elapsed_seconds"], "started_at_utc": record["started_at_utc"],
        "finished_at_utc": record["finished_at_utc"], "overlapping_workers": True,
        "runtime_scope": "individual task elapsed counter; not serial aggregate wall-clock"}
        for i, record in enumerate(time_manifest["records"])]
    costs.append({"campaign": "stock-grid correction", "stage": "pilot, fitting/selection and two independent finals",
        "elapsed_seconds": stopping_manifest["elapsed_seconds"], "started_at_utc": stopping_manifest["created_at"],
        "finished_at_utc": stopping_manifest["completed_at"], "overlapping_workers": True,
        "runtime_scope": "campaign elapsed counter; concurrent delta work separate"})
    tables["execution_costs"] = pd.DataFrame(costs)
    inventory = {}
    for row in entries:
        path = str(Path(row["path"]).resolve())
        entry = {**row, "path": path}
        if path in inventory and inventory[path] != entry:
            raise ValueError("Conflicting fingerprints for the same numerical input file")
        inventory[path] = entry
    entries = [inventory[path] for path in sorted(inventory)]
    tables["provenance"] = pd.DataFrame(entries)
    sources = {name: _sha(Path(__file__).with_name(name)) for name in (
        "_pricing_rough_gap_time_refinement.py", "_pricing_rough_gap_time_stopping.py",
        "_pricing_rough_policy_validation.py", "_pricing_rough_policy_validation_stopping.py",
        "_pricing_rough_stopping_delta.py")}
    profile = _fingerprint({"version": "rough_gap_time_refinement_v1", "files": entries, "analysis_sources": sources})
    manifest = {"backend": "rough_heston_gap_time_refinement", "study_version": "rough_gap_time_refinement_v1",
        "profile_fingerprint": profile, "input_files": entries, "analysis_sources": sources,
        "primary_complete": True, "historical_replay": False, "current_engine_execution": False,
        "execution_mode": "verified collection of current independent numerical campaigns",
        "new_time_delta_individual_checks_passed": all(checks), "individual_delta_check_count": len(checks),
        "fresh_256_paths": 8192, "fresh_512_paths": 2048, "historical_128_paths": 8192,
        "historical_256_pooled": False, "stopping_primary_paths": 8192,
        "stopping_selected_control": stopping_manifest["selected"], "primary_policy_fits": 0,
        "primary_dual_optimizer_calls": len(tables["stopping_trials"]),
        "exploratory_dual_optimizer_calls": len(tables["stopping_pilot_trials"]),
        "tolerances_changed": False, "american_pricing_ready": False, "stock_delta_ready": False,
        "new_stopping_control_delta_validated": False,
        "limitations": ["Individual stock-bump, sampling and weak-time diagnostics are correlated marginal checks, not a total error certificate.",
            "Physical-time refinement holds the learned exercise rule and its eligible dates fixed.",
            "The historical 128-step reference has sampling error; the historical 256-step comparator supplies no new valuation paths.",
            "Stopping controls change sample variance, not the expected payoff of the fixed exercise rule.",
            "The refined stopping estimator does not inherit stock-bump uncertainty checks from the stable Greek martingale.",
            "Finite-calendar value intervals do not bound optimal American-value derivatives."]}
    result = {"tables": tables, "figures": {}, "manifest": manifest}
    if compute_figures:
        from ._pricing_rough_gap_time_reporting import add_gap_time_reporting
        result = add_gap_time_reporting(result)
    return result
