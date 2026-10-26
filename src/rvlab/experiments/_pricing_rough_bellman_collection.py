"""Verify a sealed Bellman policy campaign without simulation or selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import _manifest, _sha, _table
from ._pricing_rough_policy_validation import _entry, _compare_table
from ._pricing_rough_gap_time_stopping import _preserved
from ._pricing_rough_bellman_samples import validate_bellman_samples
from ._pricing_rough_bellman_tables import selection_statistics, final_statistics, continuation_diagnostics


ROOT = Path(__file__).resolve().parents[3]
TRAIN_SEED = 1066005042
FINAL_SEEDS = (1077005042,1078005042)
ORDERS = (8,16,32)


def _read(path):
    return json.loads(Path(path).read_text())


def _expected_specs():
    result = {"frozen selected policy": {"order": None, "blend": 0.}}
    for order in ORDERS:
        for blend in (.5,1.):
            result[f"GH{order} blend{blend:g}"] = {"order": order, "blend": blend}
    return result


def _frozen_inputs(folder, design, entries):
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    output = folder.parents[1]
    model = output/"rough_larger_policy_v1/32768_paths_two_starts"
    prior_path = output/"rough_policy_validation_v1/local_control_v1/controls_before_final.json"
    grid_path = output/"rough_gap_time_refinement_v1/stock_grid_control_v2/controls_before_final.json"
    tail_path = output/"rough_tail_attribution_v1/joint_switch_v1/selection_before_final.json"
    required = {str(p.relative_to(output)) for p in (model/"frozen_policy_before_final.json",
        model/"frozen_policy_coefficients.npz",prior_path,grid_path,tail_path)}
    pins = design.get("input_sha256",{})
    if not required.issubset(pins):
        raise ValueError("Bellman design omits a frozen model, baseline policy or common-control input")
    for name,digest in pins.items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or _sha(path) != digest:
            raise ValueError("A frozen Bellman campaign input changed")
        entries.append(_entry(path))
    bundle = load_larger_policy_bundle(model)
    p,g,tail = map(_read,(prior_path,grid_path,tail_path))
    prior,grid = p["controls"][p["selected"]],g["controls"][g["selected"]]
    if tail.get("selected_control") != "extended 1 bins penalty0":
        raise ValueError("The Bellman study must retain the previously selected tail control")
    fresh = {TRAIN_SEED,*FINAL_SEEDS}
    forbidden = set(bundle["manifest"]["all_selection_training_seeds"])|{1043005042,1054005042,1055005042}
    if fresh & forbidden or not forbidden.issubset(set(design.get("forbidden_prior_seeds",[]))):
        raise ValueError("Bellman selection or final streams overlap prior policy/control fitting or spent paths")
    if (design.get("common_control_name") != tail["selected_control"]
            or design.get("common_control") != tail["controls"][tail["selected_control"]]
            or design.get("frozen_regression_policy_name") != bundle["selected_name"]):
        raise ValueError("The Bellman common martingale or frozen regression policy was replaced")
    return bundle,prior,grid,tail


def _sources(folder,design):
    old = {str(p.relative_to(ROOT)) for p in (ROOT/"src/rvlab/pricing").glob("*.py")}
    old.update("src/rvlab/experiments/"+n for n in ("_pricing_rough_tail_switch.py",
        "_pricing_rough_tail_switch_guarded.py","_pricing_rough_tail_policy.py",
        "_pricing_rough_local_control.py","_pricing_rough_node_refinement_fast.py",
        "_pricing_rough_exercise_block.py","_pricing_rough_large_policy.py",
        "_pricing_rough_frozen_policy.py","_pricing_rough_policy_improvement.py",
        "_pricing_rough_bellman_policy.py"))
    old.update(("examples/rough_tail_attribution.py","examples/rough_bellman_policy.py"))
    if set(design.get("source_sha256",{})) != old:
        raise ValueError("A Bellman numerical source is omitted or replaced")
    for name,digest in design["source_sha256"].items():
        if _sha(folder/"source"/name) != digest or _sha(ROOT/name) != digest:
            raise ValueError("A frozen Bellman numerical source or snapshot changed")


def _audit(folder,design,entries):
    audit_folder = folder.parent/"independent_adaptive_audit_v1"
    inventory,files = _preserved(audit_folder)
    audit_pin = "rough_bellman_policy_v1/independent_adaptive_audit_v1/preservation_inventory.json"
    if _sha(audit_folder/"preservation_inventory.json") != design.get("input_sha256",{}).get(audit_pin):
        raise ValueError("The independent spent-state audit changed after fresh design sealing")
    entries.extend(files)
    manifest = _read(audit_folder/"manifest.json")
    audit_design = _read(audit_folder/"design_before_evaluation.json")
    required = {"version":"rough_bellman_independent_adaptive_audit_v1","states":32,
        "orders":[8,16,32],"dates":[16,32,48,63],"spent_seeds":[1054005042,1055005042],
        "new_fits":0,"new_outer_paths":0,"used_for_formal_policy_selection":False,
        "only_premium_capped":True,"tolerances_changed":False,"allowance":".001+1e-4*abs(C)"}
    if any(audit_design.get(k) != v for k,v in required.items()):
        raise ValueError("Changed independent spent-state Bellman audit design or role")
    for name,digest in audit_design.get("source_sha256",{}).items():
        if _sha(audit_folder/"source"/name) != digest or _sha(ROOT/name) != digest:
            raise ValueError("Independent adaptive audit numerical sources changed")
        if name in design["source_sha256"] and design["source_sha256"][name] != digest:
            raise ValueError("Fresh Bellman policy differs from the independently audited implementation")
    output = folder.parents[1]
    for name,digest in audit_design.get("input_sha256",{}).items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or _sha(path) != digest:
            raise ValueError("An independent Bellman reference state or input changed")
        entries.append(_entry(path))
    panel = pd.read_csv(audit_folder/"panel.csv",float_precision="round_trip")
    checks = pd.read_csv(audit_folder/"checks.csv",float_precision="round_trip")
    diagnostics = pd.read_csv(audit_folder/"state_diagnostics.csv",float_precision="round_trip")
    if len(panel) != 32 or len(checks) != 96:
        raise ValueError("The independent Bellman panel omitted a declared state or resolution")
    expected_panel = pd.DataFrame(audit_design["state_panel"])
    columns = [c for c in expected_panel if c != "factors"]
    _compare_table(panel[columns],expected_panel[columns],["state_index"])
    if any(json.loads(actual) != expected for actual,expected in zip(panel.factors,expected_panel.factors)):
        raise ValueError("The independent scalar panel changed its full factor states")
    if manifest.get("new_paths_generated",False) or manifest.get("used_for_selection",False):
        raise ValueError("Spent numerical audit was promoted to fresh policy acceptance or selection")
    from ._pricing_rough_bellman_audit_collection import validate_adaptive_audit
    validate_adaptive_audit(audit_folder,audit_design,manifest,panel,checks,diagnostics)
    # Preserve the verified CSV representation of list-valued provenance
    # columns (factor coordinates and warning lists) for exact round trips.
    return {"spent_scalar_checks":checks,"spent_scalar_states":panel,
        "spent_scalar_diagnostics":diagnostics}


def verify_bellman_diagnostics(record,stream):
    flags = {"common_upper_exact":True,"no_future_realized_state":True,
        "approximate_Snell_not_policy_improvement_guarantee":True}
    mask = stream["computed_mask"]
    if (any(record.get(k) != v for k,v in flags.items())
            or record.get("masked_states") != int(mask.sum())):
        raise ValueError("Bellman diagnostics changed their state coverage or interpretation")
    dates = record.get("bellman_dates",[])
    expected = list(np.flatnonzero(mask.any(axis=0)))
    if [r.get("exercise_index") for r in dates] != expected:
        raise ValueError("Bellman diagnostics omit or reorder a computed exercise date")
    for row,j in zip(dates,expected):
        fixed = {"states":int(mask[:,j].sum()),"orders":[8,16,32],
            "physical_increments":2,"terminal_exact_payoff":j == 63,
            "only_premium_capped":True,"actual_future_state_used":False,
            "all_factor_branches_and_stock_orderings":True}
        if any(row.get(k) != v for k,v in fixed.items()) or row.get("max_probability_error",1.) > 2e-12:
            raise ValueError("Changed two-step conditional law, Snell envelope or nonanticipation diagnostics")
        maximum = np.max(abs(stream["continuation"][mask[:,j],j]
            -stream["direct_continuation"][mask[:,j],j]),axis=0)
        actual = np.asarray(row.get("max_cv_direct_difference"),float)
        if actual.shape != (3,) or not np.isfinite(actual).all() or not np.allclose(actual,maximum,rtol=0.,atol=2e-12):
            raise ValueError("Conditional quadrature diagnostics differ from retained values")


def collect_bellman_campaign(folder, *, compute_figures=True):
    folder = Path(folder).resolve()
    manifest,entries = _manifest(folder,inventory=True)
    design_path,seal_path = folder/"design_before_training.json",folder/"selection_before_final.json"
    design,seal = _read(design_path),_read(seal_path)
    specs = _expected_specs()
    expected = {"version":"rough_bellman_one_step_v1",
        "training":{"seed":TRAIN_SEED,"paths":2048,"role":"selection only"},
        "final_streams":[{"seed":seed,"paths":4096} for seed in FINAL_SEEDS],
        "physical_steps":128,"exercise_intervals":64,"all_paths_retained":True,
        "candidate_specs":specs,"orders":list(ORDERS),"batch_size":32,
        "confidence_multiplier":1.96,"tolerances_changed":False,
        "price_allowance":".001+1e-4*abs(V)","delta_allowance":".002+.005*abs(delta)"}
    if (any(design.get(k) != v for k,v in expected.items())
            or any(manifest.get(k) != v for k,v in design.items())
            or manifest.get("complete") is not True or manifest.get("completed_final_paths") != 8192):
        raise ValueError("Bellman campaign changed its predeclared law, candidates, independent streams or tolerances")
    _sources(folder,design)
    bundle,prior,grid,tail = _frozen_inputs(folder,design,entries)
    if (seal.get("design_sha256") != _sha(design_path) or seal.get("selection_seed") != TRAIN_SEED
            or seal.get("candidate_specs") != specs or seal.get("policy_order") != list(specs)
            or seal.get("final_paths_generated_before_seal") is not False
            or manifest.get("selection_sha256") != _sha(seal_path)
            or manifest.get("policy_order") != list(specs)):
        raise ValueError("Changed Bellman candidate ordering or pre-final selection seal")
    records,streams = [],[]
    for label,seed,paths,index in [("training",TRAIN_SEED,2048,None)]+[
            (f"final_{i}",seed,4096,i) for i,seed in enumerate(FINAL_SEEDS)]:
        sample_path,record_path = folder/(label+"_samples.npz"),folder/(label+"_manifest.json")
        record = _read(record_path)
        flags = {"seed":seed,"paths":paths,"sample_sha256":_sha(sample_path),
            "design_sha256":_sha(design_path),"selection_sha256":_sha(seal_path),
            "source_sha256":design["source_sha256"],"sources_unchanged":True}
        if index is not None:
            flags.update(index=index,policy_order=list(specs))
        if any(record.get(k) != v for k,v in flags.items()):
            raise ValueError("A Bellman stream changed its seed, sample, policy order or frozen numerical source")
        with np.load(sample_path,allow_pickle=False) as arrays:
            stream = validate_bellman_samples({k:arrays[k].copy() for k in arrays.files},
                bundle,prior,grid,tail,specs,paths=paths,orders=ORDERS)
        verify_bellman_diagnostics(record.get("diagnostics",{}),stream)
        stream["seed"] = seed
        if index is None:
            training,training_record = stream,record
        else:
            streams.append(stream);records.append(record)
    if records != manifest.get("records") or manifest.get("training_manifest_sha256") != _sha(folder/"training_manifest.json"):
        raise ValueError("Campaign completion does not identify its full selection and final records")
    selection,selected,challenger = selection_statistics(training,specs,seed=TRAIN_SEED)
    for record in (seal,manifest):
        if record.get("selected_policy") != selected or record.get("selected_challenger") != challenger:
            raise ValueError("Final outcomes changed a selection-only Bellman policy choice")
    saved_selection = _table(folder/"policy_selection.csv",entries)
    common = [c for c in selection if c in saved_selection]
    for old,new in (("selected_before_final","selected_policy"),
            ("selected_challenger_before_final","selected_challenger")):
        if old in saved_selection:
            saved_selection[new] = saved_selection.pop(old)
            if new not in common: common.append(new)
    required = {"policy","paired_lower_gain","paired_lower_gain_se","selection_score",
        "exercise_change_fraction","selection_seed","final_paths_used_for_selection",
        "selected_policy","selected_challenger"}
    if not required.issubset(common):
        raise ValueError("Bellman policy selection diagnostics are incomplete")
    _compare_table(saved_selection[common],selection[common],["policy"])
    tables = final_statistics(streams,specs,selected_policy=selected,selected_challenger=challenger)
    for name in tables:
        if name == "exercise_times":
            rows = []
            for p,(policy,spec) in enumerate(specs.items()):
                for stream in streams:
                    indices,counts = np.unique(stream["stops"][:,p],return_counts=True)
                    rows.extend({"policy":policy,**spec,"selected_policy":policy == selected,
                        "selected_challenger":policy == challenger,"seed":stream["seed"],
                        "exercise_index":int(index),"paths":int(count)} for index,count in zip(indices,counts))
            saved = _table(folder/"exercise_times.csv",entries)
            _compare_table(saved,pd.DataFrame(rows),["policy","seed","exercise_index"])
            continue
        saved = _table(folder/(name+".csv"),entries)
        keys = (["coarse_order","fine_order","blend"] if name == "quadrature_policy_refinement"
            else ["policy"]+(["seed"] if name == "per_seed" else []))
        columns = list(saved.columns)
        omitted = set(tables[name])-set(columns)
        permitted = {"path_gaps_worsened_fraction","path_gaps_improved_fraction","maximum_path_gap_increase"} if name == "tail_distribution" else set()
        if set(columns)-set(tables[name]) or not omitted.issubset(permitted):
            raise ValueError("A saved Bellman final table omits required diagnostics")
        _compare_table(saved,tables[name][columns],keys)
    tables["policy_selection"] = selection
    tables["continuation_refinement"] = continuation_diagnostics(streams,ORDERS)
    tables.update(_audit(folder,design,entries))
    tables["execution_costs"] = pd.DataFrame([
        {"stage":"selection only","seed":TRAIN_SEED,"paths":2048,
            "elapsed_seconds":training_record["elapsed_seconds"],"completed_at_utc":training_record["completed_at"],
            "scope":"fixed candidates; no regression or control fitting"},
        *[{"stage":"independent final","seed":r["seed"],"paths":4096,
            "elapsed_seconds":r["elapsed_seconds"],"completed_at_utc":r["completed_at"],
            "scope":"workers overlap; shared trace for every candidate"} for r in records]])
    return _finish(tables,entries,manifest,compute_figures=compute_figures)


def _finish(tables,entries,manifest,*,compute_figures):
    rows = {}
    for row in entries:
        key = str(Path(row["path"]).resolve())
        normalized = {**row,"path":key}
        if key in rows and rows[key] != normalized:
            raise ValueError("Conflicting canonical Bellman evidence fingerprints")
        rows[key] = normalized
    entries = [rows[p] for p in sorted(rows)]
    sources = {name:_sha(Path(__file__).with_name(name)) for name in (
        "_pricing_rough_bellman_collection.py","_pricing_rough_bellman_samples.py",
        "_pricing_rough_bellman_tables.py","_pricing_rough_bellman_audit_collection.py","_pricing_rough_tail_samples.py",
        "_pricing_rough_policy_validation_stopping.py")}
    fingerprint = hashlib.sha256(json.dumps({"version":"rough_bellman_policy_v1",
        "files":entries,"analysis_sources":sources},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    challenger = tables["policy_error"].loc[tables["policy_error"].selected_challenger].iloc[0]
    conclusion = ("designated challenger resolves policy improvement exceeding half the baseline gap" if challenger.policy_majority_proved
        else "positive removable policy error resolved; majority of the baseline gap remains unresolved" if challenger.improvement_resolved
        else "no positive policy improvement resolved for the designated challenger")
    tables["provenance"] = pd.DataFrame(entries)
    tables["configuration"] = pd.DataFrame([
        {"item":"Contract","value":"synthetic inception ATM put;S=K=100;T=60/365;4%continuousrate;explicitly no cash dividends"},
        {"item":"Numerical law","value":"same four-factor lift,128physicalsteps,64exerciseintervals; full two-step mixture"},
        {"item":"Lookahead","value":"E[Dnext max(payoff,Euro+clip(q,0,.7))|current stock and all factors]/Dleft; Snell envelope uncapped"},
        {"item":"Quadrature","value":"analytic European and clipped quadratic premium;GH8/16/32 exercise excess; terminal payoff analytic"},
        {"item":"Policies","value":"baseline plus half/full Bellman continuations per order; no model refitting"},
        {"item":"Mask","value":"all ITM dates1…63 on all paths, including after stopping; irrelevant entries explicitly NaN"},
        {"item":"Selection","value":"2048separatepaths; max positive paired-gain95%LCB with fallback; bestnonbaselinechallenger also locked"},
        {"item":"Final evidence","value":"two independent4096-path streams; identical frozen latesttail martingale for every policy"},
        {"item":"Accuracy gates","value":"price .001+1e-4|V|;budgetgap+1.96gapSE+1.96lowerSE; delta .002+.005|delta|; spread fraction.1 unchanged"},
        {"item":"Interpretation","value":"approximate next-date envelope does not guarantee policy improvement; statewise refinement is diagnostic"}])
    result_manifest = {"backend":"rough_heston_bellman_policy","study_version":"rough_bellman_policy_v1",
        "profile_fingerprint":fingerprint,"input_files":entries,"analysis_sources":sources,
        "baseline_policy":"frozen selected policy","selected_policy":manifest["selected_policy"],
        "selected_challenger":manifest["selected_challenger"],"common_control":"extended 1 bins penalty0",
        "primary_inference_policy":manifest["selected_challenger"],
        "policy_candidate_count":7,"selection_paths":2048,"final_paths":8192,"primary_policy_fits":0,
        "primary_control_optimizer_calls":0,"all_paths_retained":True,"tolerances_changed":False,
        "primary_complete":True,"historical_replay":False,"current_engine_execution":False,
        "physical_steps":128,"exercise_intervals":64,"quadrature_orders":[8,16,32],
        "price_atol":.001,"price_rtol":1e-4,"delta_atol":.002,"delta_rtol":.005,
        "spread_fraction":.1,"confidence_multiplier":1.96,
        "execution_mode":"verified read-only collection of independent sealed policy challenges",
        "policy_error_conclusion":conclusion,"american_pricing_ready":False,"stock_delta_ready":False,
        "new_rule_stock_delta_validated":False,"limitations":[
            "The approximate Snell envelope is not the exact continuation value of the frozen policy; lookahead can worsen it.",
            "The same upper estimator is retained, so differences isolate admissible-policy performance rather than total dual error.",
            "Quadrature differences and spent scalar checks are diagnostics, not rigorous bounds on every future state.",
            "All final candidates remain visible; accepted rule and designated challenger use selection paths only.",
            "Primary marginal95percent inference concerns the predesignated challenger; other candidate intervals are descriptive rather than simultaneous familywise claims.",
            "This study does not certify kernel, continuous-exercise, optimal-value delta or real-SPY readiness."]}
    result = {"tables":tables,"figures":{},"manifest":result_manifest}
    if compute_figures:
        from ._pricing_rough_bellman_reporting import add_bellman_reporting
        result = add_bellman_reporting(result)
    return result
