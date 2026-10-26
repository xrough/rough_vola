"""Audited collection of independent policy-versus-dual tail experiments."""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_policy_validation_stopping import _summary
from ._pricing_rough_stopping_delta import _manifest, _sha, _table
from ._pricing_rough_policy_validation import _entry, _compare_table
from ._pricing_rough_gap_time_stopping import _preserved


ROOT = Path(__file__).resolve().parents[3]


def _dominance_statistics(streams, selected_policy, selected_control, *, baseline_policy, baseline_control):
    """Paired majority contrasts preserve uncertainty in the baseline gap.

    Upper decreases and lower increases identify removable dual and policy
    error respectively. Neither a tail cohort nor the timing of an ex-post
    maximum is an unbiased decomposition of the optimal stopping error.
    """
    sequences = {"policy lower gain": [], "dual upper decrease": [],
        "policy gain minus half baseline gap": [], "dual decrease minus half baseline gap": []}
    for stream in streams:
        base = stream["values"][baseline_control]
        chosen = stream["values"][selected_control]
        lower = base["lower"][:, baseline_policy]
        gap = base["upper"]-lower
        policy_gain = base["lower"][:, selected_policy]-lower
        upper_decrease = base["upper"]-chosen["upper"]
        sequences["policy lower gain"].append(policy_gain)
        sequences["dual upper decrease"].append(upper_decrease)
        sequences["policy gain minus half baseline gap"].append(policy_gain-.5*gap)
        sequences["dual decrease minus half baseline gap"].append(upper_decrease-.5*gap)
    rows = []
    for name, chunks in sequences.items():
        mean, se, low, high = _summary(chunks)
        rows.append({"contrast": name, "mean": mean, "standard_error": se,
            "interval_low": low, "interval_high": high, "paths": sum(map(len, chunks)),
            "confidence_multiplier": 1.96, "paired_whole_paths": True,
            "majority_contrast": "half baseline gap" in name,
            "resolved_positive": low > 0., "optimal_policy_error_identified": False})
    return pd.DataFrame(rows)


def tail_attribution_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The declared research seed is42")
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    folder = Path(context.output_dir).resolve()/"rough_tail_attribution_v1/joint_switch_v1"
    context.progress("Verifying independently selected policy and dual tail challenges")
    return _collect_tail_study(folder, compute_figures=compute_figures)


def _collect_tail_study(folder, *, compute_figures):
    from ._pricing_rough_tail_samples import validate_control_specs, validate_tail_samples
    from ._pricing_rough_tail_tables import selection_statistics, final_statistics
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    folder = Path(folder).resolve()
    manifest, entries = _manifest(folder, inventory=True)
    design_path = folder/"design_before_training.json"
    design = json.loads(design_path.read_text())
    policy_specs = {
        "frozen selected policy": {"offset": 0., "multiplier": 1.},
        "premium offset -0.020": {"offset": -.02, "multiplier": 1.},
        "premium offset -0.005": {"offset": -.005, "multiplier": 1.},
        "premium offset +0.005": {"offset": .005, "multiplier": 1.},
        "premium offset +0.020": {"offset": .02, "multiplier": 1.},
        "premium multiplier 0.9": {"offset": 0., "multiplier": .9},
        "premium multiplier 1.1": {"offset": 0., "multiplier": 1.1}}
    expected = {"version": "rough_tail_attribution_joint_switch_v1",
        "training": {"seed": 1043005042, "paths": 4096, "fit_paths": 2048, "heldout_paths": 2048},
        "final_streams": [{"seed": 1054005042, "paths": 4096}, {"seed": 1055005042, "paths": 4096}],
        "features": ["selected poststop intrinsic", "selected poststop intrinsic soft .1", "selected poststop intrinsic soft .3",
            "boundary intrinsic soft .03", "boundary intrinsic soft .1", "boundary intrinsic soft .3", "OTM European"],
        "families": {"poststop": [0, 1, 2], "extended": list(range(7))}, "bins": [1, 4], "penalties": [0., 1e-4],
        "nonnegative_coefficient_sum_per_bin_max": 1., "extended_weight_scales": [1., .25, .5],
        "pilot_motivated_weight_shrinkage": True, "shrinkage_declared_before_any_fresh_paths": True,
        "declared_control_count_if_all_fits_complete": 17, "selected_policy_name": "dense tensor ridge1e-5",
        "policy_specs": policy_specs, "shared_holdout_for_separate_objectives": True,
        "all_cross_combinations_reported": True, "target_final_paths": 8192, "physical_steps": 128,
        "exercise_intervals": 64, "all_paths_retained": True, "confidence_multiplier": 1.96,
        "price_allowance": ".001+1e-4*abs(V)", "delta_allowance": ".002+.005*abs(delta)",
        "tolerances_changed": False, "american_pricing_ready": False, "new_rule_stock_delta_validated": False,
        "no_new_paths_generated_before_design": True}
    if (any(design.get(k) != v for k, v in expected.items())
            or any(manifest.get(k) != v for k, v in design.items())
            or manifest.get("complete") is not True or manifest.get("completed_final_paths") != 8192):
        raise ValueError("Changed or incomplete predeclared independent tail-attribution design")
    source_names = {str(p.relative_to(ROOT)) for p in (ROOT/"src/rvlab/pricing").glob("*.py")}
    source_names.update("src/rvlab/experiments/"+n for n in ("_pricing_rough_tail_switch.py",
        "_pricing_rough_tail_switch_guarded.py", "_pricing_rough_tail_policy.py", "_pricing_rough_local_control.py",
        "_pricing_rough_node_refinement_fast.py", "_pricing_rough_exercise_block.py", "_pricing_rough_large_policy.py",
        "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py"))
    source_names.add("examples/rough_tail_attribution.py")
    if set(design.get("source_sha256", {})) != source_names:
        raise ValueError("A tail-campaign numerical source is omitted")
    for name, digest in design["source_sha256"].items():
        if _sha(folder/"source"/name) != digest:
            raise ValueError("Changed tail-campaign numerical source snapshot")
    output = folder.parents[1]
    bundle_folder = output/"rough_larger_policy_v1/32768_paths_two_starts"
    prior_path = output/"rough_policy_validation_v1/local_control_v1/controls_before_final.json"
    grid_path = output/"rough_gap_time_refinement_v1/stock_grid_control_v2/controls_before_final.json"
    required_pins = {str(p.relative_to(output)) for p in (bundle_folder/"frozen_policy_before_final.json",
        bundle_folder/"frozen_policy_coefficients.npz", prior_path, grid_path)}
    if set(design.get("input_sha256", {})) != required_pins:
        raise ValueError("A frozen baseline model, policy or martingale input is missing")
    for name, digest in design["input_sha256"].items():
        if _sha(output/name) != digest:
            raise ValueError("Changed tail-study baseline identity")
        entries.append(_entry(output/name))
    bundle = load_larger_policy_bundle(bundle_folder)
    prior, grid = json.loads(prior_path.read_text()), json.loads(grid_path.read_text())
    prior_control, grid_control = prior["controls"][prior["selected"]], grid["controls"][grid["selected"]]
    if design.get("prior_control") != prior_control or design.get("grid_control") != grid_control:
        raise ValueError("The preceding selected martingale coefficients were changed")
    new_seeds = {1043005042, 1054005042, 1055005042}
    forbidden = set(bundle["manifest"]["all_selection_training_seeds"]) | set(design.get("forbidden_baseline_seeds", []))
    if new_seeds & forbidden or not {350005042, 842005042, 943005042, 954005042, 955005042}.issubset(forbidden):
        raise ValueError("Tail fitting or valuation reused baseline or spent diagnostic paths")
    seal_path = folder/"selection_before_final.json"
    seal = json.loads(seal_path.read_text())
    policies = list(policy_specs)
    if (seal.get("design_sha256") != _sha(design_path) or seal.get("selection_seed") != 1043005042
            or seal.get("policy_specs") != policy_specs or seal.get("policy_order") != policies
            or seal.get("final_paths_generated_before_seal") is not False
            or manifest.get("selection_sha256") != _sha(seal_path)
            or manifest.get("selected_control") != seal.get("selected_control")
            or manifest.get("selected_policy") != seal.get("selected_policy")
            or manifest.get("control_order") != list(seal.get("controls", {}))
            or manifest.get("policy_order") != policies):
        raise ValueError("Changed pre-final policy/control selection or contract ordering")
    controls = seal["controls"]
    trials = _table(folder/"control_trials.csv", entries)
    attempted = {f"{family} {bins} bins penalty{penalty:g}" for family in ("poststop", "extended")
        for bins in (1, 4) for penalty in (0., 1e-4)}
    if (len(trials) != 8 or set(trials.control) != attempted or trials.control.duplicated().any()
            or not set(trials.status).issubset({"completed", "optimizer_failure"})):
        raise ValueError("A declared tail-control fit or optimizer failure was omitted")
    completed = set(trials.loc[trials.status.eq("completed"), "control"])
    validate_control_specs(controls, completed)
    training_path, training_record_path = folder/"training_samples.npz", folder/"training_manifest.json"
    training_record = json.loads(training_record_path.read_text())
    if (training_record.get("seed") != 1043005042 or training_record.get("paths") != 4096
            or training_record.get("sample_sha256") != _sha(training_path)
            or training_record.get("selection_sha256") != _sha(seal_path)
            or training_record.get("design_sha256") != _sha(design_path)
            or training_record.get("source_sha256") != design["source_sha256"]
            or training_record.get("sources_unchanged") is not True
            or manifest.get("training_manifest_sha256") != _sha(training_record_path)):
        raise ValueError("Changed independent control fitting/selection proof")
    with np.load(training_path, allow_pickle=False) as z:
        training = validate_tail_samples({k: z[k].copy() for k in z.files}, controls, bundle,
            prior_control, grid_control, policy_specs, paths=4096)
    control_selection, policy_selection, sc, sp, families = selection_statistics(training, controls, policies, training_seed=1043005042)
    if (seal.get("selected_control") != sc or seal.get("selected_policy") != sp
            or seal.get("family_selected_controls") != families or manifest.get("family_selected_controls") != families):
        raise ValueError("Final outcomes changed a training-only selection")
    _compare_table(_table(folder/"control_selection.csv", entries), control_selection, ["control"])
    _compare_table(_table(folder/"policy_selection.csv", entries), policy_selection, ["policy"])
    streams, records = [], []
    for i, spec in enumerate(expected["final_streams"]):
        path = folder/f"final_{i}_samples.npz"
        record = json.loads((folder/f"final_{i}_manifest.json").read_text())
        flags = {"index": i, **spec, "sample_sha256": _sha(path), "selection_sha256": _sha(seal_path),
            "design_sha256": _sha(design_path), "source_sha256": design["source_sha256"],
            "sources_unchanged": True, "policy_order": policies, "control_order": list(controls)}
        if any(record.get(k) != v for k, v in flags.items()):
            raise ValueError("Changed independent final tail stream, source identity or ordering")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        stream = validate_tail_samples(arrays, controls, bundle, prior_control, grid_control, policy_specs, paths=4096)
        stream["seed"] = spec["seed"]
        streams.append(stream)
        records.append(record)
    if records != manifest.get("records"):
        raise ValueError("Final tail-stream records differ from the completed campaign")
    tables = final_statistics(streams, controls, policies, sc, sp)
    from ._pricing_rough_tail_capacity import poststop_capacity, summarize_poststop_capacity, check_poststop_upper
    capacity_chunks = []
    for stream in streams:
        capacity = poststop_capacity(stream["payoff"], stream["current_martingale"], stream["stops"][:, 0])
        for name, control in controls.items():
            if control["family"] == "poststop":
                check_poststop_upper(capacity, stream["values"][name]["upper"])
        capacity_chunks.append(capacity)
    tables["poststop_capacity"] = summarize_poststop_capacity(capacity_chunks)
    for name in ("comparison", "paired_changes", "per_seed", "dominance", "gap_distribution"):
        keys = ["challenge"] if name == "dominance" else ["control", "policy"]
        if name == "per_seed":
            keys.append("seed")
        _compare_table(_table(folder/(name+".csv"), entries), tables[name], keys)
    tables.update(control_selection=control_selection, policy_selection=policy_selection, control_trials=trials)
    _check_diagnostics(training_record.get("diagnostics", {}))
    for record in records:
        _check_diagnostics(record.get("diagnostics", {}))
    spent, files = _collect_spent(folder.parent, design)
    tables.update(spent)
    entries.extend(files)
    return _finish_tail(tables, entries, manifest, training_record, compute_figures=compute_figures)


def _check_diagnostics(record):
    local, grid, tail = (record.get(k, {}) for k in ("prior_local", "prior_grid", "tail_switch"))
    if (local.get("left_gates_predictable") is not True or local.get("all_paths_retained") is not True
            or local.get("right_reliability_integrated_inside_branch_expectation") is not True
            or grid.get("left_gates_predictable") is not True or grid.get("all_paths_retained") is not True
            or grid.get("right_reliability_inside_conditional_expectation") is not True
            or tail.get("left_gates_predictable") is not True or tail.get("all_paths_retained") is not True
            or tail.get("first_three_corrections_zero_at_selected_stop") is not True
            or tail.get("no_new_conditional_pricing_approximation") is not True
            or tail.get("policy_dynamics_and_calendars_unchanged") is not True
            or max(local.get("max_probability_error", 1.), grid.get("max_probability_error", 1.)) > 2e-12):
        raise ValueError("Changed conditional-centering or predictable tail-control diagnostics")


def _collect_spent(folder, design):
    entries, tables = [], {}
    pilot_folder, replay_folder = folder/"spent_switch_pilot_v1", folder/"spent_state_replay_v1"
    if (_sha(pilot_folder/"preservation_inventory.json") != design.get("pilot_inventory_sha256")
            or _sha(replay_folder/"preservation_inventory.json") != design.get("old_tail_replay_inventory_sha256")):
        raise ValueError("The exploratory pilot or replay changed after formal design sealing")
    pilot, files = _manifest(pilot_folder, inventory=True)
    entries.extend(files)
    if (pilot.get("version") != "rough_tail_switch_spent_pilot_v1" or pilot.get("spent_seed") != 350005042
            or pilot.get("paths") != 512 or pilot.get("fit_paths") != 256 or pilot.get("heldout_paths") != 256
            or pilot.get("new_acceptance_evidence") is not False
            or pilot.get("used_for_formal_control_selection") is not False or pilot.get("new_paths_generated") is not False):
        raise ValueError("The spent pilot changed its role or entered final acceptance")
    pilot_pins = {"selected_policy_seal_sha256": "rough_larger_policy_v1/32768_paths_two_starts/frozen_policy_before_final.json",
        "prior_control_seal_sha256": "rough_policy_validation_v1/local_control_v1/controls_before_final.json",
        "grid_control_seal_sha256": "rough_gap_time_refinement_v1/stock_grid_control_v2/controls_before_final.json"}
    if any(pilot.get(k) != design["input_sha256"][p] for k, p in pilot_pins.items()):
        raise ValueError("The naive pilot used a different exercise policy or preceding martingale")
    tables["spent_pilot_comparison"] = _table(pilot_folder/"pilot_comparison.csv", entries)
    tables["spent_pilot_trials"] = _table(pilot_folder/"pilot_trials.csv", entries)
    if (len(tables["spent_pilot_trials"]) != 8
            or tables["spent_pilot_comparison"].new_acceptance_evidence.any()):
        raise ValueError("A naive pilot attempt or failed trial was omitted")
    replay, files = _manifest(replay_folder, inventory=True)
    entries.extend(files)
    if (replay.get("version") != "rough_spent_state_replay_v1" or replay.get("seeds") != [954005042, 955005042]
            or replay.get("paths_per_seed") != 4096 or replay.get("rows") != 8192
            or replay.get("all_payoff_streams_bit_exact") is not True or replay.get("all_paths_retained") is not True
            or replay.get("used_for_acceptance") is not False or replay.get("used_for_formal_selection") is not False):
        raise ValueError("The replayed spent states changed their law or acceptance role")
    for path, digest in replay.get("input_sha256", {}).items():
        if _sha(path) != digest:
            raise ValueError("A sealed input to spent-state replay changed")
        entries.append(_entry(path))
    tables["spent_state_summary"] = _table(replay_folder/"state_summary.csv", entries)
    tables["spent_tail_states"] = _table(replay_folder/"tail_states.csv", entries)
    states = tables["spent_tail_states"]
    if (len(states) != 8192 or states.duplicated(["seed", "path_index"]).any()
            or set(states.seed) != {954005042, 955005042} or states.acceptance_input.any() or states.used_for_selection.any()):
        raise ValueError("The retrospective state table omitted paths or became acceptance evidence")
    timing_folder = folder/"spent_policy_timing"
    timing, files = _manifest(timing_folder, inventory=True)
    entries.extend(files)
    if (timing.get("version") != "rough_tail_spent_policy_v1" or timing.get("complete") is not True
            or timing.get("acceptance_input") is not False or timing.get("used_for_selection") is not False
            or timing.get("new_paths") != 0 or timing.get("new_fits") != 0
            or timing.get("causal_policy_or_dual_error_identified") is not False):
        raise ValueError("Ex-post tail timing was promoted to causal policy/dual evidence")
    if timing.get("source_control_seal_sha256") != design["input_sha256"][pilot_pins["grid_control_seal_sha256"]]:
        raise ValueError("The timing diagnostic uses a different preceding selected control")
    for key, name in (("spent_timing_summary", "tail_cohort_summary.csv"),
            ("spent_control_tail_sensitivity", "control_tail_sensitivity.csv")):
        tables[key] = _table(timing_folder/name, entries)
    capacity_folder = folder/"spent_poststop_capacity_v1"
    capacity, files = _manifest(capacity_folder, inventory=True)
    entries.extend(files)
    if (capacity.get("version") != "rough_tail_spent_capacity_v1" or capacity.get("complete") is not True
            or capacity.get("acceptance_input") is not False or capacity.get("used_for_selection") is not False
            or capacity.get("new_paths") != 0 or capacity.get("new_fits") != 0
            or capacity.get("spent_seeds") != [954005042, 955005042]
            or capacity.get("causal_policy_or_dual_error_identified") is not False
            or capacity.get("source_control_seal_sha256") != timing.get("source_control_seal_sha256")):
        raise ValueError("The post-stop capacity diagnostic changed its baseline or scope")
    from ._pricing_rough_tail_capacity import summarize_poststop_capacity
    keys = ("baseline_lower", "baseline_upper", "baseline_gap", "prefix_upper", "capacity", "remaining_prefix_gap")
    reconstructed = []
    with np.load(capacity_folder/"capacity_samples.npz", allow_pickle=False) as z:
        expected_arrays = {f"seed{s}_control{i}_{key}" for s in (954005042, 955005042)
            for i in range(5) for key in keys}
        if set(z.files) != expected_arrays:
            raise ValueError("An original path or control was omitted from capacity diagnostics")
        for i, name in enumerate(capacity["control_order"]):
            chunks = [{key: z[f"seed{s}_control{i}_{key}"].copy() for key in keys} for s in (954005042, 955005042)]
            if any(a.shape != (4096,) for chunk in chunks for a in chunk.values()):
                raise ValueError("Spent capacity diagnostics change original path counts")
            table = summarize_poststop_capacity(chunks)
            table.insert(0, "control", name)
            table.insert(1, "selected_before_final", name == capacity["selected_control"])
            reconstructed.append(table)
    table = pd.concat(reconstructed, ignore_index=True)
    _compare_table(_table(capacity_folder/"capacity_comparison.csv", entries), table, ["control"])
    tables["spent_poststop_capacity"] = table
    return tables, entries


def _finish_tail(tables, entries, manifest, training_record, *, compute_figures):
    from ._pricing_rough_foundation import _fingerprint
    majority = tables["dominance"].set_index("challenge").majority_dominance_proved
    if bool(majority["dual"]) and bool(majority["policy"]):
        dominant = "conflicting marginal majority intervals; unresolved"
    elif bool(majority["dual"]):
        dominant = "dual estimator"
    elif bool(majority["policy"]):
        dominant = "exercise policy"
    else:
        dominant = "unresolved by the independent majority contrasts"
    tables["configuration"] = pd.DataFrame([
        {"item": "Question", "value": "Common-M policy challenges and fixed-policy dual replacements on all independent paths"},
        {"item": "Contract", "value": "synthetic inception ATM put; spot=strike100;60/365years;4%continuousrate;explicitly no cash dividends"},
        {"item": "Fixed numerical law", "value": "four-factor kernel;128physicalsteps;64exerciseintervals; same vanilla payoff and full-state law"},
        {"item": "Policy rules", "value": "frozen baseline, signed premium offsets±.005/±.02 tapered(T-t)/T, and premium multipliers.9/1.1; same floor0/cap.7"},
        {"item": "Dual controls", "value": "strictly post-stop intrinsic switches, broader predictable boundary/OTM switches; nonnegative unit per-bin coefficient budget"},
        {"item": "Preserved failure", "value": "broad switches overfit spent256-path fitting; deterministic.25/.5 weight shrinkage declared before all fresh paths"},
        {"item": "Fitting and selection", "value": "fresh4096paths split2048fit/2048heldout; separate policy/control scores share heldout; both fixed before finals"},
        {"item": "Final evidence", "value": "two independent4096-path streams; all declared policy/control combinations retained"},
        {"item": "Dominance proof", "value": "paired95%lower bound of identified improvement minus half uncertain baselinegap must exceed0"},
        {"item": "Price tolerance", "value": ".001+1e-4*abs(V); budgetgap+1.96gapSE+1.96lowerSE"},
        {"item": "Delta/spread tolerances", "value": ".002+.005*abs(delta),10%effectivehalfspread unchanged; no new Greek certificate or quoted spreads"},
        {"item": "Readiness", "value": "no optimal-value, kernel/continuous-exercise, delta or SPY-input readiness promotion"}])
    tables["execution_costs"] = pd.DataFrame([
        {"stage": "fitting and heldout selection", "seed": 1043005042, "paths": 4096,
            "elapsed_seconds": training_record["elapsed_seconds"], "completed_at_utc": training_record["completed_at"],
            "scope": "one physical trace shared across every candidate; excludes archived work and display replay"},
        *[{"stage": "independent final", "seed": r["seed"], "paths": r["paths"],
            "elapsed_seconds": r["elapsed_seconds"], "completed_at_utc": r["completed_at"],
            "scope": "individual task duration; final workers overlap"} for r in manifest["records"]]])
    unique = {}
    for row in entries:
        path = str(Path(row["path"]).resolve())
        entry = {**row, "path": path}
        if path in unique and unique[path] != entry:
            raise ValueError("Conflicting fingerprints for a canonical tail-study input")
        unique[path] = entry
    entries = [unique[p] for p in sorted(unique)]
    tables["provenance"] = pd.DataFrame(entries)
    sources = {name: _sha(Path(__file__).with_name(name)) for name in (
        "_pricing_rough_tail_attribution.py", "_pricing_rough_tail_samples.py", "_pricing_rough_tail_tables.py",
        "_pricing_rough_tail_capacity.py",
        "_pricing_rough_policy_validation_stopping.py", "_pricing_rough_gap_time_stopping.py")}
    fingerprint = _fingerprint({"version": "rough_tail_attribution_v1", "files": entries, "analysis_sources": sources})
    result_manifest = {"backend": "rough_heston_tail_attribution", "study_version": "rough_tail_attribution_v1",
        "profile_fingerprint": fingerprint, "input_files": entries, "analysis_sources": sources,
        "primary_complete": True, "historical_replay": False, "current_engine_execution": False,
        "execution_mode": "independently verified collection of sealed current path campaigns",
        "selected_policy": manifest["selected_policy"], "selected_control": manifest["selected_control"],
        "family_selected_controls": manifest["family_selected_controls"], "dominant_source": dominant,
        "final_paths": 8192, "policy_candidate_count": 7, "control_candidate_count": len(manifest["control_order"]),
        "final_cross_combinations": len(tables["comparison"]), "primary_policy_fits": 0,
        "primary_control_optimizer_calls": len(tables["control_trials"]),
        "spent_pilot_control_optimizer_calls": len(tables["spent_pilot_trials"]),
        "all_paths_retained": True, "tolerances_changed": False,
        "american_pricing_ready": False, "stock_delta_ready": False, "new_rule_stock_delta_validated": False,
        "limitations": [
            "Separate policy/control selections share heldout paths; final valuation is independent of all selection.",
            "Identified bound improvements need not explain the whole stopping gap; a failure to prove majority does not establish equality.",
            "The margin family does not exhaust admissible stopping rules, and the switch family does not exhaust martingales.",
            "Ex-post tail cohorts, peak times and cancelling payoff/martingale components are descriptive rather than causal decompositions.",
            "Strictly post-stop controls leave the baseline lower sample unchanged; broader controls change its sampling variance.",
            "No final-error reselection, clipping, dropped paths or optional path extension occurs.",
            "Old finite-rule delta checks do not certify new controls or an altered policy's optimal-value delta."]}
    result = {"tables": tables, "figures": {}, "manifest": result_manifest}
    if compute_figures:
        from ._pricing_rough_tail_reporting import add_tail_reporting
        result = add_tail_reporting(result)
    return result
