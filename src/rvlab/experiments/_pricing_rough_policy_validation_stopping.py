"""Independent reconstruction of the sealed local-martingale experiment."""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import _manifest, _sha, _table
from ._pricing_rough_policy_validation import _compare_table, _entry


def _basis(corrections, bins):
    increments = np.diff(corrections, axis=1)
    indices = np.linspace(0, corrections.shape[1]-1, bins+1, dtype=int)
    output = []
    for left, right in zip(indices[:-1], indices[1:]):
        masked = np.zeros_like(increments)
        masked[:, left:right] = increments[:, left:right]
        output.append(np.concatenate((np.zeros_like(corrections[:, :1]), np.cumsum(masked, axis=1)), axis=1))
    return np.concatenate(output, axis=2)


def _check_samples(arrays, controls, paths):
    names = {"raw", "stops", "payoff", "baseline_martingale", "local_corrections"}
    names.update(f"control{i}_{suffix}" for i in range(len(controls)) for suffix in ("lower", "upper"))
    if set(arrays) != names or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Local-control samples omit declared arrays or contain nonfinite values")
    stops, payoff, base = (arrays[name] for name in ("stops", "payoff", "baseline_martingale"))
    corrections, raw = arrays["local_corrections"], arrays["raw"]
    if (stops.shape != (paths,) or not np.issubdtype(stops.dtype, np.integer)
            or np.any(stops < 0) or np.any(stops > 64) or payoff.shape != (paths, 65)
            or base.shape != payoff.shape or corrections.shape != (paths, 65, 3)
            or raw.shape != (paths,) or np.any(payoff < 0)
            or np.any(base[:, 0] != 0) or np.any(corrections[:, 0] != 0)):
        raise ValueError("Local-control samples change the path count, calendar or martingale origin")
    path = np.arange(paths)
    if not np.array_equal(raw, payoff[path, stops]):
        raise ValueError("Raw stopping payoff differs from its recorded stopping decision")
    results = {}
    for i, (name, control) in enumerate(controls.items()):
        bins, weights = control["bins"], np.asarray(control["weights"], float)
        if (isinstance(bins, bool) or not isinstance(bins, Integral) or bins not in (1, 4) or weights.shape != (bins*3,)
                or not np.isfinite(weights).all() or np.any(weights < -1e-12)
                or np.any(weights > 1+1e-12) or np.any(weights.reshape(bins, 3).sum(axis=1) > 1+1e-12)):
            raise ValueError("Local control violates its predeclared coefficient budget")
        martingale = base + np.einsum("ntd,d->nt", _basis(corrections, bins), weights)
        lower = raw - martingale[path, stops]
        upper = np.max(payoff - martingale, axis=1)
        if np.any(upper < lower-2e-11):
            raise ValueError("Local-control endpoints violate their pathwise ordering")
        for key, expected in ((f"control{i}_lower", lower), (f"control{i}_upper", upper)):
            if arrays[key].shape != (paths,) or not np.allclose(arrays[key], expected, rtol=0., atol=2e-13):
                raise ValueError("Saved endpoints differ from the sealed martingale and own stopping dates")
        results[name] = (lower, upper)
    baseline = controls.get("stable baseline", {})
    if baseline.get("bins") != 1 or not np.array_equal(baseline.get("weights"), np.zeros(3)):
        raise ValueError("The stable zero-correction fallback was changed")
    return results


def _strat_se(chunks):
    return float(np.sqrt(sum(len(a)*np.var(a, ddof=1) for a in chunks))/sum(map(len, chunks)))


def _mean(chunks):
    return float(np.concatenate(chunks).mean())


def _summary(chunks):
    mean, se = _mean(chunks), _strat_se(chunks)
    return mean, se, mean-1.96*se, mean+1.96*se


def _final_tables(streams, selected, policy):
    comparison, changes, per_seed = [], [], []
    for name in streams[0]["values"]:
        lows = [s["values"][name][0] for s in streams]
        uppers = [s["values"][name][1] for s in streams]
        gaps = [u-l for u, l in zip(uppers, lows)]
        price, pse, _, _ = _summary(lows)
        gap, gse, _, _ = _summary(gaps)
        allowance = .001 + 1e-4*abs(price)
        budget = gap + 1.96*(gse+pse)
        comparison.append({"control": name, "policy": policy, "paths": 8192, "streams": 2,
            "selected_before_final": name == selected, "price": price, "price_se": pse,
            "upper": _mean(uppers), "gap": gap, "gap_se": gse, "price_error_budget": budget,
            "price_allowance": allowance, "price_budget_pass": budget <= allowance,
            "american_pricing_ready": False, "stock_delta_ready": False})
        for stream, lower, upper in zip(streams, lows, uppers):
            gain = lower - stream["values"]["stable baseline"][0]
            lo, lse, _, _ = _summary([lower])
            ga, se, _, _ = _summary([upper-lower])
            change, cse, _, _ = _summary([gain])
            allowed = .001 + 1e-4*abs(lo)
            error = ga + 1.96*(lse+se)
            per_seed.append({"seed": stream["seed"], "control": name, "policy": policy,
                "paths": len(lower), "selected_before_final": name == selected,
                "price": lo, "price_se": lse, "upper": float(upper.mean()), "gap": ga, "gap_se": se,
                "paired_lower_change": change, "paired_lower_change_se": cse,
                "price_error_budget": error, "price_allowance": allowed, "price_budget_pass": error <= allowed,
                "american_pricing_ready": False, "stock_delta_ready": False})
        if name != "stable baseline":
            row = {"control": name, "selected_before_final": name == selected,
                "policy": policy, "all_paths_retained": True}
            baseline = [s["values"]["stable baseline"] for s in streams]
            sequences = {"lower_change": [lo-b[0] for lo, b in zip(lows, baseline)],
                "upper_change": [up-b[1] for up, b in zip(uppers, baseline)],
                "gap_change": [gap-(b[1]-b[0]) for gap, b in zip(gaps, baseline)]}
            for key, seq in sequences.items():
                mean, se, low, high = _summary(seq)
                row.update({key: mean, key+"_se": se, key+"_low": low, key+"_high": high})
            changes.append(row)
    return pd.DataFrame(comparison), pd.DataFrame(changes), pd.DataFrame(per_seed)


def load_stopping_campaign(folder):
    from . import _pricing_rough_policy_validation as validation
    folder = Path(folder).resolve()
    manifest, entries = _manifest(folder, inventory=True)
    design_path = folder / "design_before_pilot.json"
    design = json.loads(design_path.read_text())
    expected = {"version": "rough_local_control_v1", "selected_policy_name": "dense tensor ridge1e-5",
        "pilot": {"seed": 350005042, "paths": 512, "source": "first512 sealed historical16a final paths, spent"},
        "training": {"seed": 842005042, "paths": 4096, "fit_paths": 2048, "heldout_paths": 2048},
        "final_streams": [{"seed": 852005042, "paths": 4096}, {"seed": 853005042, "paths": 4096}],
        "left_gate_widths": [None, .1, .3], "right_factor_reliability_scale": 3.,
        "bin_candidates": [1, 4], "penalty_candidates": [0., 1e-4],
        "nonnegative_coefficient_sum_per_bin_max": 1., "stable_old_martingale_zero_correction_fallback": True,
        "no_policy_retraining": True, "no_path_clipping_drops_or_reweighting": True,
        "physical_time_and_exercise_calendars_frozen": True, "target_final_paths": 8192,
        "unchanged_tolerances": True, "price_allowance": ".001+1e-4*abs(V)",
        "delta_allowance": ".002+.005*abs(delta)", "confidence_multiplier": 1.96,
        "american_pricing_ready": False, "stock_delta_ready": False}
    if (any(design.get(key) != value for key, value in expected.items())
            or any(manifest.get(key) != value for key, value in design.items())
            or manifest.get("completed_final_streams") != 2 or manifest.get("completed_final_paths") != 8192):
        raise ValueError("Changed or incomplete independently designed local-control experiment")
    root = validation.ROOT
    expected_sources = {str(path.relative_to(root)) for path in (root / "src/rvlab/pricing").glob("*.py")}
    expected_sources.update("src/rvlab/experiments/" + name for name in (
        "_pricing_rough_local_control.py", "_pricing_rough_exercise_block.py", "_pricing_rough_large_policy.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py"))
    expected_sources.add("examples/rough_local_control.py")
    if set(design.get("source_sha256", {})) != expected_sources:
        raise ValueError("Local-control design omits a numerical source")
    for name, digest in design["source_sha256"].items():
        if _sha(folder / "source" / name) != digest:
            raise ValueError("Local-control source snapshot differs from the declared implementation")
    seal_path = folder / "controls_before_final.json"
    seal = json.loads(seal_path.read_text())
    if (manifest.get("control_seal_sha256") != _sha(seal_path)
            or seal.get("design_sha256") != _sha(design_path)
            or seal.get("selected") != manifest.get("selected")
            or seal.get("final_paths_generated_before_seal") is not False):
        raise ValueError("Changed control selection or pre-valuation seal")
    bundle = folder.parents[1] / "rough_larger_policy_v1/32768_paths_two_starts"
    policy_seal = bundle / "frozen_policy_before_final.json"
    if (design.get("selected_policy_seal_sha256") != _sha(policy_seal)
            or seal.get("selected_policy_seal_sha256") != _sha(policy_seal)
            or design.get("selected_policy_coefficients_sha256") != _sha(bundle / "frozen_policy_coefficients.npz")):
        raise ValueError("Local control changed the newly validated exercise policy")
    policy = json.loads(policy_seal.read_text())
    if ({842005042, 852005042, 853005042} & set(policy["all_selection_training_seeds"])
            or policy["selected_name"] != design["selected_policy_name"]):
        raise ValueError("Local-control fitting or valuation reused exercise-policy training paths")
    entries.extend((_entry(policy_seal), _entry(bundle / "frozen_policy_coefficients.npz")))
    # The archive inventory covers all actual experiment bytes, including the
    # spent pilot, every attempted optimization and all final path arrays.
    selection = _table(folder / "stopping_selection.csv", entries)
    trials = _table(folder / "stopping_trials.csv", entries)
    controls = seal.get("controls", {})
    attempted = {f"local {bins} bins penalty{penalty:g}" for bins in (1, 4) for penalty in (0., 1e-4)}
    if (len(trials) != 4 or set(trials.control) != attempted or trials.control.duplicated().any()
            or not set(trials.status).issubset({"completed", "optimizer_failure"})):
        raise ValueError("A declared local-control trial or optimizer failure was omitted")
    completed = set(trials.loc[trials.status == "completed", "control"])
    if (set(controls) != {"stable baseline"} | completed or len(selection) != len(controls)
            or selection.control.duplicated().any() or set(selection.control) != set(controls)):
        raise ValueError("Missing local control or failed fallback candidate")
    for bins in (1, 4):
        for penalty in (0., 1e-4):
            name = f"local {bins} bins penalty{penalty:g}"
            if name in controls and (controls[name].get("bins") != bins or controls[name].get("penalty") != penalty):
                raise ValueError("A frozen local control differs from its declared candidate")
    if (not (selection.heldout_paths == 2048).all() or not (selection.target_final_paths == 8192).all()
            or selection.final_valuation_used_for_selection.any()
            or not selection.selection_source.eq("second half of independent control-training stream only").all()):
        raise ValueError("Local-control selection used the wrong paths")
    expected_score = selection.heldout_gap + 1.96*np.sqrt(2048/8192)*(selection.heldout_gap_se+selection.heldout_price_se)
    if (not np.isfinite(expected_score).all() or not np.allclose(selection.selection_score, expected_score, rtol=1e-12, atol=2e-14)
            or selection.loc[selection.selection_score.idxmin(), "control"] != seal["selected"]
            or not np.array_equal(selection.selected_before_final, selection.control == seal["selected"])):
        raise ValueError("Changed training-only selection rule or selected control")
    streams = []
    for spec in design["final_streams"]:
        seed = spec["seed"]
        path = folder / f"final_{seed}_samples.npz"
        record = json.loads((folder / f"final_{seed}_manifest.json").read_text())
        if (record.get("seed") != seed or record.get("paths") != 4096
                or record.get("controls") != list(controls) or record.get("sample_sha256") != _sha(path)
                or record.get("control_seal_sha256") != _sha(seal_path)):
            raise ValueError("Changed independent final local-control stream")
        diagnostics = record.get("diagnostics", {})
        if (diagnostics.get("right_reliability_integrated_inside_branch_expectation") is not True
                or diagnostics.get("left_gates_predictable") is not True
                or diagnostics.get("all_paths_retained") is not True
                or diagnostics.get("no_simulation_law_or_exercise_calendar_changes") is not True
                or diagnostics.get("max_probability_error", 1.) > 2e-12
                or diagnostics.get("widths") != [None, .1, .3] or diagnostics.get("reliability_scale") != 3.):
            raise ValueError("Local-control conditional-centering diagnostics changed")
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        streams.append({"seed": seed, "values": _check_samples(arrays, controls, spec["paths"])})
    comparison, changes, per_seed = _final_tables(streams, seal["selected"], design["selected_policy_name"])
    for name, table, keys in (("stopping_comparison", comparison, ["control"]),
            ("stopping_paired_changes", changes, ["control"]),
            ("stopping_per_seed", per_seed, ["seed", "control"])):
        try:
            recorded = _table(folder / (name+".csv"), entries)
        except pd.errors.EmptyDataError:
            if not table.empty:
                raise
            recorded = pd.DataFrame()
        _compare_table(recorded, table, keys)
    stream_table = _table(folder / "stopping_streams.csv", entries)
    expected_streams = pd.DataFrame([
        {"stage": "spent pilot", "seed": 350005042, "paths": 512, "acceptance_input": False},
        {"stage": "control fit/holdout", "seed": 842005042, "paths": 4096, "acceptance_input": False},
        *[{"stage": "independent final", **spec, "acceptance_input": True} for spec in design["final_streams"]]])
    _compare_table(stream_table, expected_streams, ["seed"])
    pilot = json.loads((folder / "pilot_manifest.json").read_text())
    if pilot.get("new_acceptance_evidence") is not False or pilot.get("formal_design_changed_after_pilot") is not False:
        raise ValueError("Spent exploratory paths entered the formal control design or acceptance")
    tables = {"stopping_comparison": comparison, "stopping_paired_changes": changes,
        "stopping_selection": selection, "stopping_trials": trials, "stopping_streams": stream_table,
        "stopping_per_seed": per_seed,
        "stopping_pilot_selection": _table(folder / "pilot_selection.csv", entries),
        "stopping_pilot_trials": _table(folder / "pilot_trials.csv", entries)}
    return tables, entries, manifest
