"""Optimized, independently refitted stock-grid controls; V1 remains preserved."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_frozen_policy import _load_trace
from rvlab.experiments._pricing_rough_local_control import (
    local_correction_martingales,correction_basis,fit_correction_weights,correction_endpoints)
from rvlab.experiments._pricing_rough_node_refinement_fast import node_refinement_martingales,refined_stock_knots
from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy,fourier_policy_fingerprint
from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
from rvlab.pricing.rough_dual import apply_rough_dual_weights


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _now():return datetime.now(timezone.utc).isoformat()
def _write(path,obj):path.write_text(json.dumps(_jsonable(obj),indent=2,allow_nan=False)+"\n")
def _stats(a):return float(np.mean(a)),float(np.std(a,ddof=1)/np.sqrt(len(a)))
def _subset(trace,stop):
    pathkeys={"discounted_payoffs","european_martingale","premium_martingale","intrinsic_martingale",
        "micro_boundary_martingale","micro_poststop_martingale","stocks","european_proxy","continuation_proxy","factors","stop_indices"}
    return {k:(v[:stop].copy() if k in pathkeys else v) for k,v in trace.items()}


def _workspace(proxy,policy,weights,prior_control,trace):
    baseline=evaluate_trace_policy(trace,policy=policy,martingale_weights=weights)
    m=apply_rough_dual_weights(trace,weights)["martingale"][:,:,0]
    prior,prior_diagnostics=local_correction_martingales(proxy,policy,trace)
    prior_basis=correction_basis(prior,trace["times"],prior_control["bins"])
    baseline_m=m+np.einsum("ntd,d->nt",prior_basis,np.asarray(prior_control["weights"]))
    correction,diagnostics=node_refinement_martingales(proxy,policy,trace)
    return {"payoff":trace["discounted_payoffs"][:,:,0],"stable_old_martingale":m,
        "prior_local_corrections":prior,"baseline_martingale":baseline_m,
        "stops":baseline.stop_indices[:,0],"raw":baseline.raw_samples[:,0],"times":trace["times"],
        "corrections":correction,"diagnostics":{**diagnostics,"prior_local":prior_diagnostics,"coefficient_gather_execution_only":True}}


def _end(w,control):
    basis=correction_basis(w["corrections"],w["times"],control["bins"])
    return correction_endpoints(w["payoff"],w["baseline_martingale"],w["stops"],basis,np.asarray(control["weights"]))


def _selection(workspace,fit_paths,target_paths):
    # This split is declared before any outcomes are read. All fits share the
    # first subset; only the second subset selects among declared candidates.
    candidates={"stable baseline":{"bins":1,"penalty":0.,"weights":np.zeros(workspace["corrections"].shape[-1])}}
    records=[]
    for bins in (1,4):
        basis=correction_basis(workspace["corrections"],workspace["times"],bins)
        for penalty in (0.,1e-4):
            name=f"grid {bins} bins penalty{penalty:g}"
            coefficient,record=fit_correction_weights(workspace["payoff"][:fit_paths],
                workspace["baseline_martingale"][:fit_paths],workspace["stops"][:fit_paths],basis[:fit_paths],bins=bins,penalty=penalty)
            records.append({"control":name,**record})
            if coefficient is not None:candidates[name]={"bins":bins,"penalty":penalty,"weights":coefficient}
    rows=[];selected="stable baseline";best=float("inf")
    for name,control in candidates.items():
        lower,upper=_end(workspace,control);l=lower[fit_paths:];gap=(upper-lower)[fit_paths:]
        mean,se=_stats(gap);_,lse=_stats(l)
        # Sampling terms are projected to the predeclared pooled final size.
        targetse=se*np.sqrt(len(l)/target_paths);targetlse=lse*np.sqrt(len(l)/target_paths)
        score=mean+1.96*(targetse+targetlse)
        rows.append({"control":name,"heldout_paths":len(l),"heldout_gap":mean,"heldout_gap_se":se,
            "heldout_price":float(l.mean()),"heldout_price_se":lse,"target_final_paths":target_paths,
            "selection_score":score,"selection_source":"second half of independent control-training stream only",
            "final_valuation_used_for_selection":False})
        if score<best:selected,best=name,score
    for row in rows:row["selected_before_final"]=row["control"]==selected
    return candidates,selected,pd.DataFrame(rows),pd.DataFrame(records)


def main():
    started=time.perf_counter();output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_gap_time_refinement_v1/stock_grid_control_v2"
    if folder.exists():raise FileExistsError(f"Preserved experiment exists: {folder}")
    folder.mkdir(parents=True)
    bundle_folder=output/"rough_larger_policy_v1/32768_paths_two_starts"
    frozen=load_larger_policy_bundle(bundle_folder);proxy,policy,oldweights=[frozen[k] for k in ("proxy","selected_policy","weights")]
    prior_seal_path=output/"rough_policy_validation_v1/local_control_v1/controls_before_final.json"
    prior_seal=json.loads(prior_seal_path.read_text())
    prior_control=prior_seal['controls'][prior_seal['selected']]
    sources=[Path(__file__),ROOT/"examples/rough_grid_control_benchmark.py"]+[ROOT/"src/rvlab/experiments"/name for name in (
        "_pricing_rough_node_refinement.py","_pricing_rough_node_refinement_fast.py","_pricing_rough_local_control.py","_pricing_rough_exercise_block.py","_pricing_rough_large_policy.py",
        "_pricing_rough_frozen_policy.py","_pricing_rough_policy_improvement.py")]+sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    hashes={str(p.relative_to(ROOT)):_sha(p) for p in sources}
    for path in sources:
        target=folder/"source"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    design={"version":"rough_stock_grid_control_v2","created_at":_now(),"source_sha256":hashes,
        "performance_provenance":{
            "initial_profile":"stock_grid_control_v1",
            "initial_profile_inventory_sha256":_sha(output/"rough_gap_time_refinement_v1/stock_grid_control_v1/preservation_inventory.json"),
            "initial_profile_status_sha256":_sha(output/"rough_gap_time_refinement_v1/stock_grid_control_v1/partial_status.json"),
            "initial_training_seed_spent":942005042,"initial_final_seeds_not_generated":[952005042,953005042],
            "benchmark_manifest_sha256":_sha(output/"rough_gap_time_refinement_v1/coefficient_gather_validation_v1/manifest.json"),
            "benchmark_inventory_sha256":_sha(output/"rough_gap_time_refinement_v1/coefficient_gather_validation_v1/preservation_inventory.json"),
            "change":"TensorPremium coefficient gather only; numerical function, settings and acceptance tolerances unchanged",
            "new_unseen_training_and_final_paths":True},
        "selected_policy_name":frozen["selected_name"],"selected_policy_seal_sha256":_sha(bundle_folder/"frozen_policy_before_final.json"),
        "selected_policy_coefficients_sha256":_sha(bundle_folder/"frozen_policy_coefficients.npz"),
        "prior_selected_control":prior_seal['selected'],"prior_control_seal_sha256":_sha(prior_seal_path),
        "prior_control_seal_path":str(prior_seal_path.resolve()),"prior_control":prior_control,
        "proxy_fingerprint":fourier_policy_fingerprint(proxy),"pilot":{"seed":350005042,"paths":512,"source":"first512 sealed historical16a final paths, spent"},
        "training":{"seed":943005042,"paths":4096,"fit_paths":2048,"heldout_paths":2048},
        "final_streams":[{"seed":954005042,"paths":4096},{"seed":955005042,"paths":4096}],
        "coarse_positive_stock_knots":proxy.config.stock_knots,"subdivision_candidates":[4,8],
        "fine_positive_stock_knots":{str(m):refined_stock_knots(proxy.config.stock_knots,m) for m in (4,8)},
        "positive_stock_node_counts":{str(m):len(refined_stock_knots(proxy.config.stock_knots,m)) for m in (4,8)},
        "right_function":"R3(U)*(selected refined Snell premium minus selected coarse Snell premium)",
        "shared_european_node_count":len(refined_stock_knots(proxy.config.stock_knots,8)),
        "left_gate_widths":[None,.1,.3],"right_factor_reliability_scale":3.,"reliability_definition":"exp(-(max(abs((U-center)/scale))/3)^8)",
        "reliability_scope":"RIGHT full factor state included inside conditional expectation",
        "bin_candidates":[1,4],"penalty_candidates":[0.,1e-4],"nonnegative_coefficient_sum_per_bin_max":1.,
        "prior_selected_local_martingale_zero_new_correction_fallback":True,"no_policy_retraining":True,
        "no_path_clipping_drops_or_reweighting":True,"physical_time_and_exercise_calendars_frozen":True,
        "target_final_paths":8192,"selection_rule":"heldout gap plus projected1.96 gapSE and priceSE at8192paths",
        "unchanged_tolerances":True,"price_allowance":".001+1e-4*abs(V)","delta_allowance":".002+.005*abs(delta)",
        "confidence_multiplier":1.96,"american_pricing_ready":False,"stock_delta_ready":False}
    _write(folder/"design_before_pilot.json",design)
    # The spent pilot is preserved but cannot select the formal coefficient,
    # bin, penalty, gate width, training split or final seed.
    print("PILOT spent512",flush=True)
    historical=json.loads((Path(frozen["manifest"]["historical_bundle_path"])/"manifest.json").read_text())
    checkpoint=json.loads(Path(historical["source_checkpoint"]["path"]).read_text())["payload"]
    trace=_subset(_load_trace(checkpoint,"final_valuation"),512)
    pilot=_workspace(proxy,policy,oldweights,prior_control,trace)
    pc,ps,pselection,ptrials=_selection(pilot,256,8192)
    pselection.insert(0,"purpose","spent exploratory only");pselection.to_csv(folder/"pilot_selection.csv",index=False)
    ptrials.to_csv(folder/"pilot_trials.csv",index=False)
    _write(folder/"pilot_manifest.json",{"diagnostics":pilot["diagnostics"],"selected_exploratory":ps,
        "new_acceptance_evidence":False,"formal_design_changed_after_pilot":False,"completed_at":_now()})
    del trace,pilot
    print("TRAINING fresh4096",flush=True)
    trace=value_fourier_stopping_policy(proxy,config=replace(proxy.config,valuation_paths=4096,valuation_seed=943005042),trace=True)["trace"]
    training=_workspace(proxy,policy,oldweights,prior_control,trace)
    controls,selected,selection,trials=_selection(training,2048,8192)
    selection.to_csv(folder/"stopping_selection.csv",index=False);trials.to_csv(folder/"stopping_trials.csv",index=False)
    sealed={"sealed_at":_now(),"selected":selected,"controls":controls,
        "final_paths_generated_before_seal":False,"design_sha256":_sha(folder/"design_before_pilot.json"),
        "selected_policy_seal_sha256":design["selected_policy_seal_sha256"],"prior_control_seal_sha256":design["prior_control_seal_sha256"]}
    _write(folder/"controls_before_final.json",sealed)
    print("SELECTED",selected,flush=True)
    del trace,training
    samples_by_seed=[];perseed=[]
    for spec in design["final_streams"]:
        seed=spec["seed"];print("FINAL",seed,flush=True)
        trace=value_fourier_stopping_policy(proxy,config=replace(proxy.config,valuation_paths=spec["paths"],valuation_seed=seed),trace=True)["trace"]
        workspace=_workspace(proxy,policy,oldweights,prior_control,trace)
        values={name:_end(workspace,c) for name,c in controls.items()};baseline=values["stable baseline"]
        sample={"raw":workspace["raw"],"stops":workspace["stops"],"payoff":workspace["payoff"],
            "stable_old_martingale":workspace["stable_old_martingale"],"prior_local_corrections":workspace["prior_local_corrections"],
            "baseline_martingale":workspace["baseline_martingale"],"grid_corrections":workspace["corrections"]}
        for index,(name,(lower,upper)) in enumerate(values.items()):
            price,pse=_stats(lower);gap,gse=_stats(upper-lower);gain,gainse=_stats(lower-baseline[0])
            allowance=.001+1e-4*abs(price);budget=gap+1.96*(gse+pse)
            perseed.append({"seed":seed,"control":name,"policy":frozen["selected_name"],"paths":len(lower),
                "selected_before_final":name==selected,"price":price,"price_se":pse,"upper":float(upper.mean()),
                "gap":gap,"gap_se":gse,"paired_lower_change":gain,"paired_lower_change_se":gainse,
                "price_error_budget":budget,"price_allowance":allowance,"price_budget_pass":budget<=allowance,
                "american_pricing_ready":False,"stock_delta_ready":False})
            sample[f"control{index}_lower"]=lower;sample[f"control{index}_upper"]=upper
        np.savez_compressed(folder/f"final_{seed}_samples.npz",**sample)
        _write(folder/f"final_{seed}_manifest.json",{"seed":seed,"paths":spec["paths"],"controls":list(controls),
            "sample_sha256":_sha(folder/f"final_{seed}_samples.npz"),"control_seal_sha256":_sha(folder/"controls_before_final.json"),
            "diagnostics":workspace["diagnostics"],"completed_at":_now()})
        samples_by_seed.append(values);pd.DataFrame(perseed).to_csv(folder/"stopping_per_seed.csv",index=False)
        del trace,workspace,sample
    comparison=[];changes=[]
    strat=lambda seq:float(np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in seq))/sum(map(len,seq)))
    for name in controls:
        lows=[s[name][0] for s in samples_by_seed];ups=[s[name][1] for s in samples_by_seed];gaps=[u-l for u,l in zip(ups,lows)]
        price=float(np.concatenate(lows).mean());pse=strat(lows);gap=float(np.concatenate(gaps).mean());gse=strat(gaps)
        allowance=.001+1e-4*abs(price);budget=gap+1.96*(pse+gse)
        comparison.append({"control":name,"policy":frozen["selected_name"],"paths":8192,"streams":2,
            "selected_before_final":name==selected,"price":price,"price_se":pse,"upper":float(np.concatenate(ups).mean()),
            "gap":gap,"gap_se":gse,"price_error_budget":budget,"price_allowance":allowance,"price_budget_pass":budget<=allowance,
            "american_pricing_ready":False,"stock_delta_ready":False})
        if name!="stable baseline":
            gain=[s[name][0]-s["stable baseline"][0] for s in samples_by_seed]
            upperchange=[s[name][1]-s["stable baseline"][1] for s in samples_by_seed]
            gapchange=[(s[name][1]-s[name][0])-(s["stable baseline"][1]-s["stable baseline"][0]) for s in samples_by_seed]
            row={"control":name,"selected_before_final":name==selected,"policy":frozen["selected_name"],"all_paths_retained":True}
            for key,seq in (("lower_change",gain),("upper_change",upperchange),("gap_change",gapchange)):
                estimate=float(np.concatenate(seq).mean());se=strat(seq)
                row.update({key:estimate,key+"_se":se,key+"_low":estimate-1.96*se,key+"_high":estimate+1.96*se})
            changes.append(row)
    pd.DataFrame(comparison).to_csv(folder/"stopping_comparison.csv",index=False)
    pd.DataFrame(changes).to_csv(folder/"stopping_paired_changes.csv",index=False)
    streams=[{"stage":"spent pilot","seed":350005042,"paths":512,"acceptance_input":False},
        {"stage":"control fit/holdout","seed":943005042,"paths":4096,"acceptance_input":False}]
    streams +=[{"stage":"independent final","seed":s["seed"],"paths":s["paths"],"acceptance_input":True} for s in design["final_streams"]]
    pd.DataFrame(streams).to_csv(folder/"stopping_streams.csv",index=False)
    changesource=[key for key,digest in hashes.items() if _sha(ROOT/key)!=digest]
    manifest={**design,"completed_at":_now(),"elapsed_seconds":time.perf_counter()-started,
        "selected":selected,"completed_final_streams":2,"completed_final_paths":8192,"source_changes":changesource,
        "sources_unchanged":not changesource,"control_seal_sha256":_sha(folder/"controls_before_final.json"),
        "tables":{"stopping_comparison":len(comparison),"stopping_paired_changes":len(changes),
            "stopping_selection":len(selection),"stopping_trials":len(trials),"stopping_streams":len(streams)}}
    _write(folder/"manifest.json",manifest)
    entries=[{"path":str(p.relative_to(folder)),"bytes":p.stat().st_size,"sha256":_sha(p)} for p in sorted(folder.rglob("*")) if p.is_file()]
    canonical=json.dumps(entries,sort_keys=True,separators=(",",":")).encode()
    _write(folder/"preservation_inventory.json",{"files":entries,"inventory_sha256":hashlib.sha256(canonical).hexdigest()})
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    print(pd.DataFrame(changes).to_string(index=False),flush=True)
    print("SECONDS",manifest["elapsed_seconds"],"SOURCECHANGES",changesource,flush=True)


if __name__=="__main__":main()
