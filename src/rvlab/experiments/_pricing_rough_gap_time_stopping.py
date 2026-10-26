"""Reconstruct the sealed nested-stock-grid campaign from whole-path samples."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import _manifest, _sha, _table, _verify_file
from ._pricing_rough_policy_validation import _compare_table, _entry
from ._pricing_rough_policy_validation_stopping import _basis, _final_tables


def _preserved(folder):
    inventory = folder/"preservation_inventory.json"
    record = json.loads(inventory.read_text())
    digest = hashlib.sha256(json.dumps(record["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != record["inventory_sha256"]:
        raise ValueError("Changed computational-trial preservation inventory")
    entries = [_entry(inventory)]
    seen = set()
    for row in record["files"]:
        path = (folder/row["path"]).resolve()
        if path in seen or not path.is_relative_to(folder.resolve()):
            raise ValueError("Duplicate or escaping computational-trial artifact")
        seen.add(path)
        _verify_file(path, row)
        entries.append(_entry(path))
    return record, entries


def _weights(control, features):
    bins, weights = control.get("bins"), np.asarray(control.get("weights"), float)
    if (isinstance(bins, bool) or not isinstance(bins, Integral) or bins not in (1, 4)
            or weights.shape != (bins*features,) or not np.isfinite(weights).all()
            or np.any(weights < -1e-12) or np.any(weights > 1+1e-12)
            or np.any(weights.reshape(bins, features).sum(axis=1) > 1+1e-12)):
        raise ValueError("Stock-grid control changes its declared per-bin coefficient budget")
    return bins, weights


def _grid_samples(arrays, controls, prior_control, paths):
    required = {"raw", "stops", "payoff", "stable_old_martingale", "prior_local_corrections",
        "baseline_martingale", "grid_corrections"}
    required.update(f"control{i}_{suffix}" for i in range(len(controls)) for suffix in ("lower", "upper"))
    if set(arrays) != required or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Stock-grid arrays are incomplete, unexpected or nonfinite")
    raw, stops, payoff, stable, prior, baseline, grid = (arrays[k] for k in (
        "raw", "stops", "payoff", "stable_old_martingale", "prior_local_corrections",
        "baseline_martingale", "grid_corrections"))
    if (stops.shape != (paths,) or not np.issubdtype(stops.dtype, np.integer)
            or np.any(stops < 0) or np.any(stops > 64) or raw.shape != (paths,)
            or payoff.shape != (paths, 65) or np.any(payoff < 0)
            or stable.shape != payoff.shape or baseline.shape != payoff.shape
            or prior.shape != (paths, 65, 3) or grid.shape != (paths, 65, 6)
            or any(np.any(a[:, 0] != 0) for a in (stable, baseline, prior, grid))):
        raise ValueError("Stock-grid arrays change the path count, calendar or martingale origin")
    path = np.arange(paths)
    if not np.array_equal(raw, payoff[path, stops]):
        raise ValueError("Raw payoff differs from the frozen policy's own stopping decisions")
    bins, weights = _weights(prior_control, 3)
    expected = stable + np.einsum("ntd,d->nt", _basis(prior, bins), weights)
    if not np.allclose(baseline, expected, rtol=0., atol=2e-13):
        raise ValueError("The prior selected local martingale was replaced or reconstructed incorrectly")
    fallback = controls.get("stable baseline", {})
    if (fallback.get("bins") != 1 or fallback.get("penalty") != 0.
            or not np.array_equal(fallback.get("weights"), np.zeros(6))):
        raise ValueError("The prior selected local control must remain the zero-new-correction fallback")
    result = {}
    for i, (name, control) in enumerate(controls.items()):
        bins, weights = _weights(control, 6)
        martingale = baseline + np.einsum("ntd,d->nt", _basis(grid, bins), weights)
        lower, upper = raw-martingale[path, stops], np.max(payoff-martingale, axis=1)
        if np.any(upper < lower-2e-11):
            raise ValueError("The dual endpoints violate pathwise lower/upper ordering")
        for suffix, values in (("lower", lower), ("upper", upper)):
            recorded = arrays[f"control{i}_{suffix}"]
            if recorded.shape != (paths,) or not np.allclose(recorded, values, rtol=0., atol=2e-13):
                raise ValueError("Stock-grid endpoints differ from their sealed control and stopping rule")
        result[name] = lower, upper
    return result


def _selection_tables(folder, entries, seal):
    selection = _table(folder/"stopping_selection.csv", entries)
    trials = _table(folder/"stopping_trials.csv", entries)
    columns = {"control", "heldout_paths", "target_final_paths", "heldout_gap", "heldout_gap_se",
        "heldout_price", "heldout_price_se", "final_valuation_used_for_selection", "selection_source",
        "selection_score", "selected_before_final"}
    if not columns.issubset(selection) or not {"control", "status"}.issubset(trials):
        raise ValueError("Missing stock-grid selection statistics or trial status")
    attempted = {f"grid {b} bins penalty{p:g}" for b in (1, 4) for p in (0., 1e-4)}
    if (len(trials) != 4 or set(trials.control) != attempted or trials.control.duplicated().any()
            or not set(trials.status).issubset({"completed", "optimizer_failure"})):
        raise ValueError("A predeclared stock-grid fit or failed attempt was omitted")
    controls = seal.get("controls", {})
    completed = set(trials.loc[trials.status == "completed", "control"])
    if (set(controls) != {"stable baseline"} | completed or len(selection) != len(controls)
            or selection.control.duplicated().any() or set(selection.control) != set(controls)):
        raise ValueError("A fitted control or the stable fallback was omitted")
    fallback = controls["stable baseline"]
    if (fallback.get("bins") != 1 or fallback.get("penalty") != 0.
            or not np.array_equal(fallback.get("weights"), np.zeros(6))):
        raise ValueError("Changed zero-correction fallback specification")
    for b in (1, 4):
        for p in (0., 1e-4):
            name = f"grid {b} bins penalty{p:g}"
            if name in controls and (controls[name].get("bins") != b or controls[name].get("penalty") != p):
                raise ValueError("A sealed stock-grid control changed its candidate specification")
    if (not (selection.heldout_paths == 2048).all() or not (selection.target_final_paths == 8192).all()
            or selection.final_valuation_used_for_selection.any()
            or not selection.selection_source.eq("second half of independent control-training stream only").all()):
        raise ValueError("Stock-grid selection used undeclared or final valuation paths")
    fields = ["heldout_gap", "heldout_gap_se", "heldout_price", "heldout_price_se"]
    if (not np.isfinite(selection[fields].to_numpy()).all()
            or (selection[["heldout_gap", "heldout_gap_se", "heldout_price_se"]] < 0).any().any()):
        raise ValueError("Stock-grid holdout gaps and standard errors must be finite and nonnegative")
    score = selection.heldout_gap + 1.96*np.sqrt(2048/8192)*(selection.heldout_gap_se+selection.heldout_price_se)
    if (not np.isfinite(score).all() or not np.allclose(selection.selection_score, score, rtol=1e-12, atol=2e-14)
            or selection.loc[selection.selection_score.idxmin(), "control"] != seal.get("selected")
            or not np.array_equal(selection.selected_before_final, selection.control == seal["selected"])):
        raise ValueError("The frozen training-only selection score or selected control changed")
    for control in controls.values():
        _weights(control, 6)
    return selection, trials


def _interpolation_audit(folder, root, model_path, arrays_path):
    manifest, entries = _manifest(folder, inventory=True)
    if (manifest.get("version") != "rough_grid_interpolation_validation_v1"
            or manifest.get("no_new_paths_or_fit") is not True
            or manifest.get("new_acceptance_evidence") is not False
            or manifest.get("used_for_final_control_selection") is not False
            or manifest.get("model_seal_sha256") != _sha(model_path)
            or manifest.get("coefficients_sha256") != _sha(arrays_path)):
        raise ValueError("The deterministic interpolation audit changed its inputs or acceptance role")
    expected_sources = {"examples/rough_grid_control_interpolation.py",
        "src/rvlab/experiments/_pricing_rough_node_refinement_fast.py",
        "src/rvlab/experiments/_pricing_rough_node_refinement.py",
        "src/rvlab/pricing/rough_stopping_fourier.py", "src/rvlab/pricing/rough_tensor_premium.py"}
    if set(manifest.get("source_sha256", {})) != expected_sources:
        raise ValueError("The interpolation audit omits a numerical source")
    for name, digest in manifest["source_sha256"].items():
        if _sha(folder/"source"/name) != digest or _sha(root/name) != digest:
            raise ValueError("Changed interpolation-audit source snapshot")
    table = _table(folder/"interpolation.csv", entries)
    pairs = {(j, s) for j in (16, 32, 48, 60) for s in (1, 4, 8)}
    if (len(table) != 12 or set(zip(table.date_index, table.subdivisions)) != pairs
            or table.duplicated(["date_index", "subdivisions"]).any()
            or not (table.spent_states == 8).all() or not (table.stock_lower == 85.).all()
            or not (table.stock_upper == 110.).all() or not (table.direct_evaluation_nodes == 2001).all()
            or table.new_acceptance_evidence.any()
            or not np.array_equal(table.positive_nodes, table.subdivisions.map({1: 20, 4: 77, 8: 153}))
            or not np.isfinite(table[["rms_interpolation_error", "max_absolute_interpolation_error"]]).all().all()
            or (table.rms_interpolation_error < 0).any()
            or (table.max_absolute_interpolation_error < table.rms_interpolation_error).any()):
        raise ValueError("Malformed or incomplete deterministic interpolation diagnostic")
    decreasing = all(np.all(np.diff(group.sort_values("subdivisions").rms_interpolation_error) < 0)
        for _, group in table.groupby("date_index"))
    if manifest.get("all_date_rms_strictly_decreases") != decreasing:
        raise ValueError("Interpolation conclusion differs from its recorded diagnostic")
    model = json.loads(model_path.read_text())
    historical_path = Path(model["historical_bundle_path"])/"manifest.json"
    if _sha(historical_path) != model["historical_bundle_sha256"]:
        raise ValueError("The spent interpolation states belong to a changed historical bundle")
    historical = json.loads(historical_path.read_text())
    checkpoint = Path(historical["source_checkpoint"]["path"])
    if _sha(checkpoint) != manifest.get("source_checkpoint_sha256"):
        raise ValueError("The spent interpolation-state checkpoint changed")
    entries.extend((_entry(historical_path), _entry(checkpoint)))
    return table, entries


def _gap_distribution(streams, selected):
    """Post-valuation descriptive tails; retain every path and frozen selection."""
    rows = []
    for name in streams[0]["values"]:
        gaps = np.concatenate([s["values"][name][1]-s["values"][name][0] for s in streams])
        ordered, total = np.sort(gaps), float(gaps.sum())
        row = {"control": name, "selected_before_final": name == selected,
            "paths": len(gaps), "zero_gap_fraction_at_1e_12": float(np.mean(gaps <= 1e-12)),
            "gap_median": float(np.median(gaps)), "gap_p90": float(np.quantile(gaps, .9)),
            "gap_p99": float(np.quantile(gaps, .99)), "gap_maximum": float(gaps.max()),
            "acceptance_input": False, "used_for_selection": False, "all_paths_retained": True}
        for fraction, label in ((.1, "largest_decile_gap_share"), (.01, "largest_percentile_gap_share")):
            count = int(np.ceil(len(gaps)*fraction))
            row[label] = float(ordered[-count:].sum()/total) if total else 0.
        rows.append(row)
    return pd.DataFrame(rows)


def collect_grid_campaign(folder):
    from ._pricing_rough_gap_time_refinement import ROOT
    folder = Path(folder).resolve()
    manifest, entries = _manifest(folder, inventory=True)
    design_path = folder/"design_before_pilot.json"
    design = json.loads(design_path.read_text())
    expected = {"version": "rough_stock_grid_control_v2", "selected_policy_name": "dense tensor ridge1e-5",
        "pilot": {"seed": 350005042, "paths": 512, "source": "first512 sealed historical16a final paths, spent"},
        "training": {"seed": 943005042, "paths": 4096, "fit_paths": 2048, "heldout_paths": 2048},
        "final_streams": [{"seed": 954005042, "paths": 4096}, {"seed": 955005042, "paths": 4096}],
        "subdivision_candidates": [4, 8], "positive_stock_node_counts": {"4": 77, "8": 153},
        "shared_european_node_count": 153, "left_gate_widths": [None, .1, .3],
        "right_factor_reliability_scale": 3., "reliability_scope": "RIGHT full factor state included inside conditional expectation",
        "bin_candidates": [1, 4], "penalty_candidates": [0., 1e-4], "nonnegative_coefficient_sum_per_bin_max": 1.,
        "prior_selected_local_martingale_zero_new_correction_fallback": True, "no_policy_retraining": True,
        "no_path_clipping_drops_or_reweighting": True, "physical_time_and_exercise_calendars_frozen": True,
        "target_final_paths": 8192, "unchanged_tolerances": True,
        "price_allowance": ".001+1e-4*abs(V)", "delta_allowance": ".002+.005*abs(delta)",
        "confidence_multiplier": 1.96, "american_pricing_ready": False, "stock_delta_ready": False}
    if (any(design.get(k) != v for k, v in expected.items())
            or any(manifest.get(k) != v for k, v in design.items())
            or manifest.get("completed_final_streams") != 2 or manifest.get("completed_final_paths") != 8192):
        raise ValueError("Changed or incomplete predeclared nested-stock-grid study")
    sources = {str(p.relative_to(ROOT)) for p in (ROOT/"src/rvlab/pricing").glob("*.py")}
    sources.update("src/rvlab/experiments/"+n for n in ("_pricing_rough_node_refinement.py", "_pricing_rough_node_refinement_fast.py",
        "_pricing_rough_local_control.py", "_pricing_rough_exercise_block.py", "_pricing_rough_large_policy.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py"))
    sources.add("examples/rough_grid_control_fast.py")
    sources.add("examples/rough_grid_control_benchmark.py")
    if set(design.get("source_sha256", {})) != sources:
        raise ValueError("The stock-grid design omits a numerical source")
    for name, digest in design["source_sha256"].items():
        if _sha(folder/"source"/name) != digest:
            raise ValueError("Changed frozen stock-grid numerical source")
    coarse = np.asarray(design.get("coarse_positive_stock_knots"), float)
    if coarse.shape != (20,) or not np.isfinite(coarse).all() or np.any(coarse <= 0) or np.any(np.diff(coarse) <= 0):
        raise ValueError("Changed coarse stock interpolation grid")
    for multiple in (4, 8):
        refined = np.r_[coarse[0], np.concatenate([np.linspace(a, b, multiple+1)[1:]
            for a, b in zip(coarse[:-1], coarse[1:])])]
        if not np.array_equal(refined, design["fine_positive_stock_knots"][str(multiple)]):
            raise ValueError("The declared stock refinements are not exact nested subdivisions")
    output = folder.parents[1]
    bundle = output/"rough_larger_policy_v1/32768_paths_two_starts"
    model_path, arrays_path = bundle/"frozen_policy_before_final.json", bundle/"frozen_policy_coefficients.npz"
    model = json.loads(model_path.read_text())
    prior_path = output/"rough_policy_validation_v1/local_control_v1/controls_before_final.json"
    prior_seal = json.loads(prior_path.read_text())
    prior_control = prior_seal["controls"][prior_seal["selected"]]
    if (design.get("selected_policy_seal_sha256") != _sha(model_path)
            or design.get("selected_policy_coefficients_sha256") != _sha(arrays_path)
            or Path(design.get("prior_control_seal_path", "")).resolve() != prior_path
            or design.get("prior_control_seal_sha256") != _sha(prior_path)
            or design.get("prior_selected_control") != prior_seal["selected"]
            or design.get("prior_control") != prior_control or model["selected_name"] != design["selected_policy_name"]):
        raise ValueError("Changed exercise policy, prior local control or coefficient identity")
    independent = {943005042, 954005042, 955005042}
    if independent & (set(model["all_selection_training_seeds"]) | {350005042, 842005042, 852005042, 853005042, 942005042}):
        raise ValueError("New stock-grid streams reuse earlier policy/control fitting or valuation paths")
    entries.extend(_entry(p) for p in (model_path, arrays_path, prior_path))
    # The performance failure remains part of the current study's evidence.
    # It supplies no fresh final valuation paths and cannot select a control.
    performance = design.get("performance_provenance", {})
    initial = folder.parent/"stock_grid_control_v1"
    benchmark = folder.parent/"coefficient_gather_validation_v1"
    initial_inventory, old_files = _preserved(initial)
    benchmark_inventory, benchmark_files = _preserved(benchmark)
    status_path, benchmark_path = initial/"partial_status.json", benchmark/"manifest.json"
    status, speed = json.loads(status_path.read_text()), json.loads(benchmark_path.read_text())
    if (performance.get("initial_profile") != "stock_grid_control_v1"
            or performance.get("initial_profile_inventory_sha256") != _sha(initial/"preservation_inventory.json")
            or performance.get("initial_profile_status_sha256") != _sha(status_path)
            or performance.get("initial_training_seed_spent") != 942005042
            or performance.get("initial_final_seeds_not_generated") != [952005042, 953005042]
            or performance.get("benchmark_manifest_sha256") != _sha(benchmark_path)
            or performance.get("benchmark_inventory_sha256") != _sha(benchmark/"preservation_inventory.json")
            or performance.get("new_unseen_training_and_final_paths") is not True
            or status.get("status") != "interrupted_for_execution_performance" or status.get("exit_code") != 130
            or status.get("training942_treated_as_spent") is not True
            or status.get("old_declared_final952_953_not_generated") is not True
            or status.get("replacement_changes_model_or_acceptance_tolerances") is not False
            or speed.get("all_equivalence_checks_pass") is not True
            or speed.get("max_absolute_difference", 1.) > 1e-11
            or speed.get("model_seal_sha256") != _sha(model_path)
            or speed.get("coefficients_sha256") != _sha(arrays_path)):
        raise ValueError("The interrupted computational trial or equivalent replacement lost its provenance")
    entries.extend(old_files+benchmark_files)
    seal_path = folder/"controls_before_final.json"
    seal = json.loads(seal_path.read_text())
    if (manifest.get("control_seal_sha256") != _sha(seal_path)
            or seal.get("design_sha256") != _sha(design_path) or seal.get("selected") != manifest.get("selected")
            or seal.get("selected_policy_seal_sha256") != _sha(model_path)
            or seal.get("prior_control_seal_sha256") != _sha(prior_path)
            or seal.get("final_paths_generated_before_seal") is not False):
        raise ValueError("Changed stock-grid selection or the pre-final control seal")
    selection, trials = _selection_tables(folder, entries, seal)
    streams = []
    for spec in design["final_streams"]:
        seed = spec["seed"]
        path = folder/f"final_{seed}_samples.npz"
        record = json.loads((folder/f"final_{seed}_manifest.json").read_text())
        if (record.get("seed") != seed or record.get("paths") != 4096
                or record.get("controls") != list(seal["controls"]) or record.get("sample_sha256") != _sha(path)
                or record.get("control_seal_sha256") != _sha(seal_path)):
            raise ValueError("Changed independent final stock-grid stream")
        diag = record.get("diagnostics", {})
        flags = {"widths": [None, .1, .3], "right_reliability_scale": 3., "subdivisions": [4, 8],
            "positive_nodes": [77, 153], "coarse_nodes_exact_subsets": True, "shared_european_values": True,
            "all_paths_retained": True, "right_reliability_inside_conditional_expectation": True,
            "left_gates_predictable": True, "policy_dynamics_exercise_and_physical_steps_unchanged": True}
        if any(diag.get(k) != v for k, v in flags.items()) or diag.get("max_probability_error", 1.) > 2e-12:
            raise ValueError("Changed stock-grid conditional-centering diagnostics")
        if diag.get("coefficient_gather_execution_only") is not True:
            raise ValueError("The stock-grid stream changed its declared equivalent execution method")
        prior_diag = diag.get("prior_local", {})
        if (prior_diag.get("right_reliability_integrated_inside_branch_expectation") is not True
                or prior_diag.get("left_gates_predictable") is not True
                or prior_diag.get("all_paths_retained") is not True
                or prior_diag.get("no_simulation_law_or_exercise_calendar_changes") is not True
                or prior_diag.get("max_probability_error", 1.) > 2e-12):
            raise ValueError("Changed prior local-control centering diagnostics")
        with np.load(path, allow_pickle=False) as z:
            samples = {k: z[k].copy() for k in z.files}
        streams.append({"seed": seed, "values": _grid_samples(samples, seal["controls"], prior_control, 4096)})
    comparison, changes, per_seed = _final_tables(streams, seal["selected"], design["selected_policy_name"])
    for name, frame, keys in (("stopping_comparison", comparison, ["control"]),
            ("stopping_paired_changes", changes, ["control"]), ("stopping_per_seed", per_seed, ["seed", "control"])):
        recorded = _table(folder/(name+".csv"), entries)
        _compare_table(recorded, frame, keys)
    stream_table = _table(folder/"stopping_streams.csv", entries)
    expected_streams = pd.DataFrame([
        {"stage": "spent pilot", "seed": 350005042, "paths": 512, "acceptance_input": False},
        {"stage": "control fit/holdout", "seed": 943005042, "paths": 4096, "acceptance_input": False},
        *[{"stage": "independent final", **s, "acceptance_input": True} for s in design["final_streams"]]])
    _compare_table(stream_table, expected_streams, ["seed"])
    pilot = json.loads((folder/"pilot_manifest.json").read_text())
    if pilot.get("new_acceptance_evidence") is not False or pilot.get("formal_design_changed_after_pilot") is not False:
        raise ValueError("Spent pilot paths entered selection or acceptance")
    tables = {"stopping_comparison": comparison, "stopping_paired_changes": changes,
        "stopping_selection": selection, "stopping_trials": trials,
        "stopping_streams": stream_table, "stopping_per_seed": per_seed,
        "stopping_pilot_selection": _table(folder/"pilot_selection.csv", entries),
        "stopping_pilot_trials": _table(folder/"pilot_trials.csv", entries),
        "stopping_interrupted_pilot_selection": _table(initial/"pilot_selection.csv", entries),
        "stopping_interrupted_pilot_trials": _table(initial/"pilot_trials.csv", entries),
        "stopping_implementation_equivalence": _table(benchmark/"equivalence.csv", entries),
        "stopping_implementation_timing": _table(benchmark/"timing.csv", entries)}
    interpolation, interpolation_files = _interpolation_audit(
        folder.parent/"interpolation_validation_v1", ROOT, model_path, arrays_path)
    tables["stopping_interpolation_audit"] = interpolation
    tables["stopping_gap_distribution"] = _gap_distribution(streams, seal["selected"])
    entries.extend(interpolation_files)
    return tables, entries, manifest
