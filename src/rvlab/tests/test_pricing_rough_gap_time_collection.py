"""Fresh-law collection preserves independent errors and rejects altered evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_gap_time_refinement as study
from rvlab.experiments import _pricing_rough_stopping_delta as old


def _write(path, record):
    path.write_text(json.dumps(record, indent=2))


def _arrays(paths=4):
    output = {"relative_spots": study.SPOTS.copy()}
    for i, policy in enumerate(study.POLICIES):
        output[policy+"__stops"] = np.full((paths, 5), 64, dtype=int)
        slope = -.4-.001*i + np.linspace(-.01, .01, paths)[:, None]
        for estimator, field, scale in (("raw", "raw_prices", 1.), ("stopped martingale", "controlled_prices", .1)):
            prices = 3.+100*study.SPOTS*(slope*scale)
            output[policy+"__"+field] = prices
            for bump in study.BUMPS:
                lo, hi = np.flatnonzero(study.SPOTS == -bump)[0], np.flatnonzero(study.SPOTS == bump)[0]
                output[policy+"__"+f"{estimator}:{bump:g}"] = (prices[:, hi]-prices[:, lo])/(200*bump)
    return output


def test_sample_statistic_reconstruction_keeps_pathwise_covariance():
    arrays = _arrays()
    study._time_arrays(arrays, 4)
    estimates, changes = study._time_statistics(arrays)
    assert len(estimates) == 8 and len(changes) == 8 and len(arrays) == 15
    row = estimates.iloc[0]
    values = arrays[row.policy+"__"+f"{row.estimator}:{row.bump_relative:g}"]
    assert row.standard_error == pytest.approx(values.std(ddof=1)/2)
    assert row.ci_half_width == pytest.approx(1.96*row.standard_error)
    assert not estimates.optimal_delta_ready.any()
    paired = changes.query("comparison == 'selected minus original'").iloc[0]
    key = f"{paired.estimator}:{paired.fine_bump:g}"
    expected = arrays["selected_policy__"+key]-arrays["original_policy__"+key]
    assert paired.standard_error == pytest.approx(expected.std(ddof=1)/2)


@pytest.mark.parametrize("alteration", ["omit", "extra", "nonfinite", "identity", "stops", "spot_order", "negative_price"])
def test_corrupted_stock_paths_are_rejected(alteration):
    arrays = _arrays()
    key = "selected_policy__stopped martingale:0.0025"
    if alteration == "omit": arrays.pop(key)
    if alteration == "extra": arrays["hidden"] = np.zeros(4)
    if alteration == "nonfinite": arrays[key][0] = np.nan
    if alteration == "identity": arrays[key][0] += .001
    if alteration == "stops": arrays["selected_policy__stops"][0, 0] = 65
    if alteration == "spot_order": arrays["relative_spots"] = study.SPOTS[::-1]
    if alteration == "negative_price": arrays["selected_policy__raw_prices"][0, 0] = -1.
    with pytest.raises(ValueError): study._time_arrays(arrays, 4)


def test_independent_resolutions_propagate_both_sampling_errors():
    coarse, _ = study._time_statistics(_arrays())
    fine = coarse.copy()
    fine["mean"] += .0008
    fine["standard_error"] *= 2
    result = study._resolution_comparison(coarse, fine, 256, 512)
    np.testing.assert_allclose(result.standard_error, np.sqrt(5)*coarse.standard_error)
    np.testing.assert_allclose(result.change_plus_uncertainty, .0008+1.96*result.standard_error)
    assert result.independent_streams.all() and result.same_exercise_calendar.all()
    assert not result.optimal_delta_ready.any()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    project = tmp_path/"project"
    fake = project/"src/rvlab/experiments/_pricing_rough_stopping_delta.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("# collection fixture\n")
    monkeypatch.setattr(study, "ROOT", project)
    monkeypatch.setattr(old, "__file__", str(fake))
    folder = project/"output/rough_time_precision_v1/fresh256_and512"
    folder.mkdir(parents=True)
    sources = ["examples/rough_time_precision.py", "src/rvlab/pricing/source.py"]
    sources += ["src/rvlab/experiments/"+name for name in ("_pricing_rough_large_policy.py",
        "_pricing_rough_time_precision.py", "_pricing_rough_stock_delta.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py")]
    hashes = {}
    for name in sources:
        path = project/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# "+name)
        snapshot = folder/"source"/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[name] = study._sha(path)
    _write(folder/"source_inventory.json", {"files": [study._entry(folder/"source"/n) for n in hashes]})
    _write(folder/"execution_environment.json", {"greek_worker_count": 2,
        "environment": {k: "1" for k in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        **{k: "test-fixture" for k in ("interpreter", "python", "platform", "numpy", "scipy", "pandas")}})
    output = folder.parents[1]
    bundle = output/"rough_larger_policy_v1/32768_paths_two_starts"
    bundle.mkdir(parents=True)
    model = {"selected_name": "dense tensor ridge1e-5", "all_selection_training_seeds": [1, 2],
        "historical_bundle_path": str(project/"historical"), "historical_bundle_sha256": "oldhash"}
    _write(bundle/"frozen_policy_before_final.json", model)
    np.savez_compressed(bundle/"frozen_policy_coefficients.npz", values=np.arange(3.))
    baseline_folder = output/"rough_new_policy_delta_v1/new_selected_2x4096"
    baseline_folder.mkdir(parents=True)
    baseline_names = ["pooled_estimates.csv", "preservation_inventory.json", "stream_0_samples.npz",
        "stream_1_samples.npz", "stream_0.json", "stream_1.json", "manifest.json"]
    for name in baseline_names: (baseline_folder/name).write_text(name)
    baseline, _ = study._time_statistics(_arrays(8192))
    old256, _ = study._time_statistics(_arrays(2048))
    seen = []
    def historical(path):
        seen.append(path)
        return {"delta_estimates": baseline, "delta_time_estimates": old256}, [], {}
    monkeypatch.setattr(study, "_delta_campaign", historical)
    labels = {"original_policy": "previous dense tensor", "selected_policy": "dense tensor ridge1e-5"}
    design = {"version": "rough_time_precision_v1", "seeds": [910005042, 911005042, 912005042],
        "paths": [4096, 4096, 2048], "steps": [256, 256, 512], "exercise_steps": 64,
        "bump_relatives": list(study.BUMPS), "stock_count": 5, "selected_policy_name": model["selected_name"],
        "historical_baseline_steps": 128, "historical_baseline_paths": 8192,
        "historical_baseline_seeds": [810005042, 811005042],
        "old_256_pilot_excluded_from_fresh_estimates": True, "all_new_streams_declared_before_valuation": True,
        "no_optional_path_extension": True, "fixed_parameters_initial_factors_discount_dividends": True,
        "new_and_previous_rules_reapplied_at_each_bump": True, "no_valuation_selection_or_fitting": True,
        "all_paths_retained": True, "same_common_stopped_martingale": True,
        "independent_physical_resolutions": True, "optimal_delta_ready": False, "tolerances_changed": False,
        "price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002, "delta_rtol": .005,
        "confidence_multiplier": 1.96, "spread_fraction": .1, "source_sha256": hashes,
        "new_policy_seal_sha256": study._sha(bundle/"frozen_policy_before_final.json"),
        "new_policy_arrays_sha256": study._sha(bundle/"frozen_policy_coefficients.npz"),
        "policy_labels": labels, "all_selection_training_seeds": [1, 2],
        "historical_bundle_path": model["historical_bundle_path"], "historical_bundle_sha256": "oldhash",
        "historical_baseline_path": str(baseline_folder),
        "historical_baseline_manifest_sha256": study._sha(baseline_folder/"manifest.json"),
        "historical_baseline_files_sha256": {n: study._sha(baseline_folder/n) for n in baseline_names[:-1]}}
    design_path = folder/"design_before_valuation.json"
    _write(design_path, design)
    records, archives, frames = [], [], []
    for i, (seed, paths, steps) in enumerate(zip(design["seeds"], design["paths"], design["steps"])):
        arrays = _arrays(paths)
        # Distinct declared streams make accidental pooling testable.
        for policy in study.POLICIES:
            for estimator, field in (("raw", "raw_prices"), ("stopped martingale", "controlled_prices")):
                arrays[policy+"__"+field] += 100*study.SPOTS*(i+1)*1e-4
                for bump in study.BUMPS:
                    p = arrays[policy+"__"+field]
                    lo, hi = np.flatnonzero(study.SPOTS == -bump)[0], np.flatnonzero(study.SPOTS == bump)[0]
                    arrays[policy+"__"+f"{estimator}:{bump:g}"] = (p[:, hi]-p[:, lo])/(200*bump)
        estimates, changes = study._time_statistics(arrays)
        estimates["seed"], estimates["steps"] = seed, steps
        estimates.to_csv(folder/f"stream_{i}_estimates.csv", index=False)
        changes.to_csv(folder/f"stream_{i}_comparisons.csv", index=False)
        np.savez_compressed(folder/f"stream_{i}_samples.npz", **arrays)
        flags = {"seed": seed, "paths": paths, "steps": steps, "actual_time_steps": steps,
            "independent_training_valuation": True, "valuation_used_for_fitting": False,
            "fixed_parameters_factors_discount_dividends": True, "decisions_reapplied_at_each_bump": True,
            "stopping_times_held_fixed": False, "same_common_martingale_for_policies": True,
            "selected_policy_name": model["selected_name"], "policy_selection_used_delta_paths": False,
            "optimal_value_delta_certified": False, "old_256_pilot_pooled": False,
            "five_stock_bumps": True, "fresh_resolutions_independent": True,
            "optimal_delta_ready": False, "tolerances_changed": False}
        record = {"index": i, "seed": seed, "paths": paths, "steps": steps, "diagnostics": flags,
            "source_sha256": hashes, "design_sha256": study._sha(design_path), "sources_unchanged": True,
            "elapsed_seconds": 1., "started_at_utc": "fixture-start", "finished_at_utc": "fixture-end",
            "files": [study._entry(folder/f"stream_{i}_{s}") for s in ("samples.npz", "estimates.csv", "comparisons.csv")]}
        _write(folder/f"stream_{i}.json", record)
        records.append(record); archives.append(arrays); frames.append(estimates)
    joined = {k: np.concatenate([a[k] for a in archives[:2]]) for k in archives[0] if k != "relative_spots"}
    primary, paired = study._time_statistics(joined)
    fine, fine_paired = study._time_statistics(archives[2])
    tables = {"pooled_256_estimates.csv": primary, "pooled_256_comparisons.csv": paired,
        "confirmation_512_estimates.csv": fine, "confirmation_512_comparisons.csv": fine_paired,
        "historical_128_baseline.csv": baseline, "per_seed.csv": pd.concat(frames, ignore_index=True),
        "independent_128_256_comparison.csv": study._resolution_comparison(baseline, primary, 128, 256),
        "independent_256_512_comparison.csv": study._resolution_comparison(primary, fine, 256, 512)}
    for name, table in tables.items(): table.to_csv(folder/name, index=False)
    manifest = {"version": "rough_time_precision_v1", "selected_policy_name": model["selected_name"],
        "complete": True, "primary_paths": 8192, "primary_steps": 256, "primary_seed_count": 2,
        "confirmation_paths": 2048, "confirmation_steps": 512, "confirmation_seed_count": 1,
        "historical_baseline_paths": 8192, "historical_baseline_steps": 128, "stock_count": 5,
        "independent_paths": True, "all_paths_retained": True, "old_256_pilot_pooled": False,
        "new_rule_delta_measured": True, "optimal_delta_ready": False, "policy_labels": labels,
        "source_sha256": hashes, "design_sha256": study._sha(design_path), "records": records,
        "files": [study._entry(folder/n) for n in [*tables, *(f"stream_{i}.json" for i in range(3))]]}
    _write(folder/"manifest.json", manifest)
    return folder, seen


def test_complete_campaign_has_no_historical_256_pooling_and_canonical_paths(campaign):
    folder, seen = campaign
    tables, entries, _ = study._collect_time(folder)
    assert seen == [folder.parents[1]/"rough_new_policy_delta_v1/new_selected_2x4096"]
    assert len(tables["delta_estimates"]) == 8 and (tables["delta_estimates"].paths == 8192).all()
    assert (tables["delta_512_estimates"].paths == 2048).all()
    assert not tables["delta_precision_comparison"].old_256_pooled.any()
    assert not tables["delta_precision_comparison"].acceptance_input.any()
    assert len(tables["delta_grid_estimates"]) == 32
    assert all(Path(r["path"]).is_absolute() for r in entries)


@pytest.mark.parametrize("alteration", ["source", "snapshot", "design", "model", "samples", "table", "record", "omit", "manifest", "baseline", "source_inventory", "environment"])
def test_sealed_campaign_rejects_changed_provenance_or_results(campaign, alteration):
    folder, _ = campaign
    design_path = folder/"design_before_valuation.json"
    design = json.loads(design_path.read_text())
    if alteration == "source": (study.ROOT/"examples/rough_time_precision.py").write_text("changed")
    if alteration == "snapshot": (folder/"source/examples/rough_time_precision.py").write_text("changed")
    if alteration == "design":
        design["old_256_pilot_excluded_from_fresh_estimates"] = False; _write(design_path, design)
    if alteration == "model": (folder.parents[1]/"rough_larger_policy_v1/32768_paths_two_starts/frozen_policy_coefficients.npz").write_bytes(b"changed")
    if alteration == "samples": (folder/"stream_0_samples.npz").write_bytes(b"changed")
    if alteration == "table": (folder/"pooled_256_estimates.csv").write_text("changed")
    if alteration == "record": (folder/"stream_0.json").write_text("{}")
    if alteration == "omit": (folder/"stream_2.json").unlink()
    if alteration == "manifest":
        p = folder/"manifest.json"; m = json.loads(p.read_text()); m["old_256_pilot_pooled"] = True; _write(p, m)
    if alteration == "baseline": Path(design["historical_baseline_path"], "pooled_estimates.csv").write_text("changed")
    if alteration == "source_inventory": _write(folder/"source_inventory.json", {"files": []})
    if alteration == "environment": _write(folder/"execution_environment.json", {"greek_worker_count": 3})
    with pytest.raises((ValueError, FileNotFoundError)): study._collect_time(folder)


def test_missing_campaign_and_undeclared_options_do_not_trigger_simulation(tmp_path):
    with pytest.raises(FileNotFoundError, match="Complete the independent experiment"):
        study.gap_time_refinement_study(SimpleNamespace(output_dir=tmp_path, progress=lambda _: None))
    for value in (True, 43):
        with pytest.raises(ValueError): study.gap_time_refinement_study(None, seed=value)
    with pytest.raises(TypeError): study.gap_time_refinement_study(None, compute_figures=1)


def test_profile_deduplicates_canonical_paths_before_fingerprinting(campaign):
    folder, _ = campaign
    tables, entries, manifest = study._collect_time(folder)
    tables["stopping_trials"] = pd.DataFrame([{"status": "completed"}])
    tables["stopping_pilot_trials"] = pd.DataFrame([{"status": "completed"}])
    stopping = {"selected": "stable baseline", "elapsed_seconds": 1., "created_at": "a", "completed_at": "b"}
    first = study._finish_collection(dict(tables), entries, manifest, stopping, compute_figures=False)
    alias = {**entries[0], "path": str(Path(entries[0]["path"]).parent/"."/Path(entries[0]["path"]).name)}
    second = study._finish_collection(dict(tables), entries+[alias], manifest, stopping, compute_figures=False)
    assert first["manifest"]["profile_fingerprint"] == second["manifest"]["profile_fingerprint"]
    assert first["manifest"]["individual_delta_check_count"] == 10
    assert not first["manifest"]["new_stopping_control_delta_validated"]
    assert not first["manifest"]["american_pricing_ready"]
