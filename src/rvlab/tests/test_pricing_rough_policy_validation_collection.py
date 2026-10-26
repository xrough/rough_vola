"""The new-policy collector rejects incomplete paths and altered diagnostics."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_policy_validation as study


@pytest.fixture
def arrays():
    relative = study.RELATIVE_SPOTS.copy()
    slopes = np.array([-.38, -.42, -.36, -.4])
    output = {"relative_spots": relative}
    for index, policy in enumerate(study.POLICIES):
        output[policy + "__stops"] = np.full((4, 7), 64, dtype=int)
        for estimator, name, scale in (("raw", "raw_prices", 1.),
                ("stopped martingale", "controlled_prices", .1)):
            values = 3. + 100*relative[None, :]*(slopes[:, None]*scale - .002*index)
            output[policy + "__" + name] = values
            for bump in study.BUMPS:
                lo = np.flatnonzero(relative == -bump)[0]
                hi = np.flatnonzero(relative == bump)[0]
                output[policy + "__" + f"{estimator}:{bump:g}"] = (values[:, hi] - values[:, lo])/(200*bump)
    return output


def test_all_bump_paths_and_whole_path_uncertainty_are_verified(arrays):
    study._validate_delta_arrays(arrays, 4)
    estimates, comparisons = study._delta_statistics(arrays)
    assert len(estimates) == 12 and len(comparisons) == 14
    row = estimates.query("policy == 'original_policy' and estimator == 'raw' and bump_relative == .0025").iloc[0]
    assert row["mean"] == pytest.approx(-.39)
    assert row.standard_error == pytest.approx(np.std([-.38, -.42, -.36, -.4], ddof=1)/2)
    assert row.ci_half_width == pytest.approx(1.96*row.standard_error)
    assert not row.sampling_passed
    assert not row.optimal_delta_ready


@pytest.mark.parametrize("alteration", ["omit", "extra", "finite", "price_identity", "path_count", "bump_order", "noninteger_stop", "stop_range", "negative_raw"])
def test_changed_or_incomplete_sample_archives_are_rejected(arrays, alteration):
    key = "selected_policy__stopped martingale:0.0025"
    if alteration == "omit":
        arrays.pop(key)
    elif alteration == "extra":
        arrays["unreported_paths"] = np.ones(4)
    elif alteration == "finite":
        arrays[key][0] = np.nan
    elif alteration == "price_identity":
        arrays[key][0] += .001
    elif alteration == "path_count":
        arrays[key] = arrays[key][:-1]
    elif alteration == "bump_order":
        arrays["relative_spots"] = arrays["relative_spots"][::-1]
    elif alteration == "noninteger_stop":
        arrays["selected_policy__stops"] = arrays["selected_policy__stops"].astype(float)
    elif alteration == "stop_range":
        arrays["selected_policy__stops"][0, 0] = 65
    else:
        arrays["selected_policy__raw_prices"] -= 100
    with pytest.raises(ValueError):
        study._validate_delta_arrays(arrays, 4)


@pytest.mark.parametrize("alteration", ["omit_failed_row", "duplicate", "changed_mean", "changed_gate", "changed_label"])
def test_collected_tables_cannot_hide_failed_rows_or_change_statistics(arrays, alteration):
    expected, _ = study._delta_statistics(arrays)
    actual = expected.copy()
    if alteration == "omit_failed_row":
        actual = actual.iloc[1:]
    elif alteration == "duplicate":
        actual.iloc[0] = actual.iloc[1]
    elif alteration == "changed_mean":
        actual.loc[0, "mean"] += .00001
    elif alteration == "changed_gate":
        actual.loc[0, "sampling_passed"] = not actual.loc[0, "sampling_passed"]
    else:
        actual.loc[0, "policy"] = "different rule"
    with pytest.raises(ValueError):
        study._compare_table(actual, expected, ["policy", "estimator", "bump_relative"])


def test_independent_time_refinement_uses_sum_of_variances_not_a_paired_difference(arrays):
    base, _ = study._delta_statistics(arrays)
    refined = base.copy()
    refined["mean"] += .0005
    refined["standard_error"] *= 2
    time = study._time_comparison(base, refined)
    row = time.iloc[0]
    assert row["mean"] == pytest.approx(.0005)
    assert row.standard_error == pytest.approx(np.sqrt(5)*base.iloc[0].standard_error)
    assert row.change_plus_uncertainty == pytest.approx(.0005 + 1.96*row.standard_error)
    assert row.independent_streams and row.same_exercise_calendar
    assert not row.optimal_delta_ready


def test_missing_campaign_has_no_numerical_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="Complete the independent experiment"):
        study.policy_validation_study(SimpleNamespace(output_dir=tmp_path, progress=lambda _: None))


def test_no_comparison_is_invented_when_every_optimizer_attempt_failed():
    study._compare_table(pd.DataFrame(), pd.DataFrame(), ["control"])
    with pytest.raises(ValueError, match="Unexpected rows"):
        study._compare_table(pd.DataFrame([{"control": "invented fit"}]), pd.DataFrame(), ["control"])


@pytest.mark.parametrize("options,error", [({"seed": True}, ValueError), ({"seed": 43}, ValueError),
    ({"compute_figures": 1}, TypeError), ({"reprice_missing": True}, TypeError)])
def test_undeclared_workflow_options_fail_before_collecting(options, error):
    with pytest.raises(error):
        study.policy_validation_study(None, **options)


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


@pytest.fixture
def campaign(tmp_path, monkeypatch, arrays):
    """A complete sealed three-stream campaign, without market or MC access."""
    from rvlab.experiments import _pricing_rough_stopping_delta as old
    project = tmp_path / "project"
    fake_file = project / "src/rvlab/experiments/_pricing_rough_stopping_delta.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# collector fixture\n")
    monkeypatch.setattr(study, "ROOT", project)
    monkeypatch.setattr(old, "__file__", str(fake_file))
    folder = project / "output/rough_new_policy_delta_v1/new_selected_2x4096"
    folder.mkdir(parents=True)
    sources = ["examples/rough_new_policy_delta.py", "src/rvlab/pricing/test_source.py"]
    sources += ["src/rvlab/experiments/" + name for name in ("_pricing_rough_large_policy.py",
        "_pricing_rough_new_policy_delta.py", "_pricing_rough_stock_delta.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py")]
    hashes = {}
    for name in sources:
        source = project / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# fixed numeric source " + name + "\n")
        snapshot = folder / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        hashes[name] = study._sha(source)
    bundle = folder.parents[1] / "rough_larger_policy_v1/32768_paths_two_starts"
    bundle.mkdir(parents=True)
    np.savez_compressed(bundle / "frozen_policy_coefficients.npz", model=np.arange(3.))
    _write(bundle / "frozen_policy_before_final.json", {
        "selected_name": "dense tensor ridge1e-5", "all_selection_training_seeds": [610005042, 620005042]})
    historical = project / "historical"
    historical.mkdir()
    np.savez_compressed(historical / "coefficients.npz", model=np.arange(2.))
    _write(historical / "manifest.json", {"coefficient_archive_sha256": study._sha(historical / "coefficients.npz")})
    labels = {"original_policy": "previous dense tensor", "selected_policy": "dense tensor ridge1e-5"}
    design = {"version": "rough_new_policy_delta_v1", "seeds": [810005042, 811005042, 812005042],
        "paths": [4096, 4096, 2048], "steps": [128, 128, 256], "exercise_steps": 64,
        "bump_relatives": list(study.BUMPS), "source_sha256": hashes,
        "selected_policy_name": "dense tensor ridge1e-5", "policy_labels": labels,
        "all_selection_training_seeds": [610005042, 620005042],
        "new_policy_seal_sha256": study._sha(bundle / "frozen_policy_before_final.json"),
        "new_policy_arrays_sha256": study._sha(bundle / "frozen_policy_coefficients.npz"),
        "historical_bundle_path": str(historical), "historical_bundle_sha256": study._sha(historical / "manifest.json"),
        "fixed_parameters_initial_factors_discount_dividends": True,
        "new_and_previous_rules_reapplied_at_each_bump": True, "no_valuation_selection_or_fitting": True,
        "all_paths_retained": True, "same_common_stopped_martingale": True,
        "price_atol": .001, "price_rtol": .0001, "delta_atol": .002, "delta_rtol": .005,
        "confidence_multiplier": 1.96, "spread_fraction": .1}
    design_path = folder / "design_before_valuation.json"
    _write(design_path, design)
    records, archives, frames = [], [], []
    for i, (seed, paths, steps) in enumerate(zip(design["seeds"], design["paths"], design["steps"])):
        a = {key: (value.copy() if key == "relative_spots" else np.tile(value, (paths//4,) + (1,)*(value.ndim-1)))
            for key, value in arrays.items()}
        estimates, comparisons = study._delta_statistics(a)
        estimates["seed"], estimates["steps"] = seed, steps
        estimates.to_csv(folder / f"stream_{i}_estimates.csv", index=False)
        comparisons.to_csv(folder / f"stream_{i}_comparisons.csv", index=False)
        np.savez_compressed(folder / f"stream_{i}_samples.npz", **a)
        diagnostics = {"seed": seed, "paths": paths, "steps": steps, "actual_time_steps": steps,
            "independent_training_valuation": True, "valuation_used_for_fitting": False,
            "fixed_parameters_factors_discount_dividends": True,
            "decisions_reapplied_at_each_bump": True, "stopping_times_held_fixed": False,
            "same_common_martingale_for_policies": True, "selected_policy_name": design["selected_policy_name"],
            "policy_selection_used_delta_paths": False, "optimal_value_delta_certified": False,
            "optimal_delta_ready": False, "tolerances_changed": False}
        record = {"index": i, "seed": seed, "paths": paths, "steps": steps,
            "design_sha256": study._sha(design_path), "source_sha256": hashes,
            "sources_unchanged": True, "diagnostics": diagnostics,
            "files": [study._entry(folder / f"stream_{i}_{suffix}")
                for suffix in ("estimates.csv", "comparisons.csv", "samples.npz")]}
        _write(folder / f"stream_{i}.json", record)
        records.append(record)
        archives.append(a)
        frames.append(estimates)
    joined = {key: np.concatenate([a[key] for a in archives[:2]]) for key in archives[0] if key != "relative_spots"}
    estimates, paired = study._delta_statistics(joined)
    refined, fine_paired = study._delta_statistics(archives[2])
    tables = {"pooled_estimates.csv": estimates, "pooled_comparisons.csv": paired,
        "time_estimates.csv": refined, "time_comparisons.csv": fine_paired,
        "independent_time_comparison.csv": study._time_comparison(estimates, refined),
        "per_seed.csv": pd.concat(frames, ignore_index=True)}
    for name, frame in tables.items():
        frame.to_csv(folder / name, index=False)
    manifest = {"version": design["version"], "complete": True, "design_sha256": study._sha(design_path),
        "source_sha256": hashes, "records": records, "primary_paths": 8192, "primary_seed_count": 2,
        "time_paths": 2048, "time_seed_count": 1, "selected_policy_name": design["selected_policy_name"],
        "policy_labels": labels, "independent_paths": True, "all_paths_retained": True,
        "new_rule_delta_measured": True, "optimal_delta_ready": False, "tolerances_changed": False,
        "files": [study._entry(folder / name) for name in tables] +
            [study._entry(folder / f"stream_{i}.json") for i in range(3)]}
    _write(folder / "manifest.json", manifest)
    return folder


def _reseal_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["files"] = [study._entry(folder / Path(r["path"]).name) for r in manifest["files"]]
    _write(folder / "manifest.json", manifest)


def test_complete_campaign_checks_sources_models_streams_statistics_and_failed_cases(campaign):
    tables, entries, manifest = study._delta_campaign(campaign)
    assert len(tables["delta_estimates"]) == 12
    assert len(tables["delta_comparisons"]) == 14
    assert len(tables["delta_time_comparisons"]) == 12
    assert len(tables["delta_stream_estimates"]) == 36
    assert len(entries) > 25
    assert manifest["new_rule_delta_measured"]
    assert not manifest["optimal_delta_ready"]


def test_relative_and_absolute_campaign_paths_have_identical_provenance(campaign, monkeypatch):
    absolute = study._delta_campaign(campaign)
    monkeypatch.chdir(campaign.parent)
    relative = study._delta_campaign(Path(campaign.name))
    assert absolute[1] == relative[1]
    assert absolute[2] == relative[2]
    assert all(Path(row["path"]).is_absolute() for row in relative[1])


@pytest.mark.parametrize("alteration", ["source", "snapshot", "model", "historical_coefficients",
    "design_seed", "design_tolerance", "manifest_record", "stream_decision", "table_gate", "table_rows"])
def test_full_campaign_rejects_tampered_sources_models_gates_and_self_consistent_csvs(campaign, alteration):
    if alteration in ("source", "snapshot"):
        path = ((study.ROOT if alteration == "source" else campaign / "source") /
            "src/rvlab/experiments/_pricing_rough_stock_delta.py")
        path.write_text("# changed calculation\n")
    elif alteration == "model":
        (campaign.parents[1] / "rough_larger_policy_v1/32768_paths_two_starts/frozen_policy_coefficients.npz").write_bytes(b"changed")
    elif alteration == "historical_coefficients":
        (study.ROOT / "historical/coefficients.npz").write_bytes(b"changed")
    elif alteration in ("design_seed", "design_tolerance"):
        path = campaign / "design_before_valuation.json"
        data = json.loads(path.read_text())
        if alteration == "design_seed": data["seeds"][0] = data["all_selection_training_seeds"][0]
        else: data["delta_atol"] *= 2
        _write(path, data)
    elif alteration == "manifest_record":
        path = campaign / "manifest.json"
        data = json.loads(path.read_text())
        data["records"][0]["paths"] = 10
        _write(path, data)
    elif alteration == "stream_decision":
        path = campaign / "stream_0.json"
        data = json.loads(path.read_text())
        data["diagnostics"]["stopping_times_held_fixed"] = True
        _write(path, data)
        _reseal_manifest(campaign)
    else:
        path = campaign / "pooled_estimates.csv"
        data = pd.read_csv(path)
        if alteration == "table_gate": data.loc[0, "sampling_passed"] = not data.loc[0, "sampling_passed"]
        else: data = data.iloc[1:]
        data.to_csv(path, index=False)
        _reseal_manifest(campaign)
    with pytest.raises(ValueError):
        study._delta_campaign(campaign)
