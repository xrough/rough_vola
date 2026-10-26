"""Independent research trial of a tensor-policy obstacle martingale refresh."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_frozen_policy import reconstruct_joint_frozen_policy
from rvlab.experiments._pricing_rough_stopping_refresh import (
    refresh_tensor_obstacle_control,fit_microdate_value_control)
from rvlab.pricing.rough_dual import train_rough_dual_weights, _weighted_martingale
from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy, fourier_policy_fingerprint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(samples):
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1)/np.sqrt(len(samples)))
    return {"mean":mean,"standard_error":se,"low":mean-1.96*se,"high":mean+1.96*se}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",required=True)
    parser.add_argument("--dual-paths",type=int,default=2048)
    parser.add_argument("--valuation-paths",type=int,default=1024)
    parser.add_argument("--dual-seed",type=int,default=510000042)
    parser.add_argument("--valuation-seed",type=int,default=512000042)
    parser.add_argument("--dense",action="store_true")
    parser.add_argument("--microdate-value",action="store_true")
    parser.add_argument("--value-paths",type=int,default=4096)
    parser.add_argument("--value-seed",type=int,default=514000042)
    args = parser.parse_args()
    if not args.label or not all(c.isalnum() or c in "-_" for c in args.label):
        parser.error("label must be a simple identifier")
    if args.dual_seed == args.valuation_seed:
        parser.error("training and valuation seeds must be distinct")
    started=time.perf_counter()
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_stopping_refresh_v1"/args.label
    if folder.exists():
        raise FileExistsError(f"Preserved trial exists: {folder}")
    folder.mkdir(parents=True)
    sources=[Path(__file__),ROOT/"src/rvlab/experiments/_pricing_rough_stopping_refresh.py"]+sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    hashes={str(path.relative_to(ROOT)):_sha(path) for path in sources}
    for path in sources:
        target=folder/"source"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    frozen=reconstruct_joint_frozen_policy(output)
    old,original,selected,oldweights=(frozen[key] for key in ("proxy","original_policy","selected_policy","weights"))
    grid=old.config.stock_knots
    if args.dense:
        grid=tuple(sorted(set(grid)|{(a+b)/2 for a,b in zip(grid[:-1],grid[1:])}))
    initial={"created_at":datetime.now(timezone.utc).isoformat(),"settings":vars(args),
        "source_sha256":hashes,"original_proxy_fingerprint":fourier_policy_fingerprint(old),
        "value_fit_degree":2,"value_fit_ridge":1e-4,
        "value_stock_knots":frozen["manifest"]["base_config"]["stock_knots"],
        "control_stock_knots":grid,"tolerances_unchanged":True,
        "value_training_used_for_policy_reselection":False,"final_valuation_used_for_training":False}
    (folder/"design_before_value_training.json").write_text(json.dumps(_jsonable(initial),indent=2,allow_nan=False)+"\n")
    if args.microdate_value:
        print("VALUE TRAINING",args.label,flush=True)
        fresh=fit_microdate_value_control(old,selected,training_paths=args.value_paths,
            training_seed=args.value_seed,value_stock_knots=frozen["manifest"]["base_config"]["stock_knots"],
            control_stock_knots=grid)
    else:
        fresh=refresh_tensor_obstacle_control(old,selected,stock_knots=grid)
    dual=replace(oldweights.config,training_paths=args.dual_paths,training_seed=args.dual_seed,
        target_valuation_paths=args.valuation_paths,candidate_bins=(1,4),dispersion_penalties=(0.,))
    design={"study_version":"rough_stopping_refresh_v1","label":args.label,
        "created_at":datetime.now(timezone.utc).isoformat(),"settings":vars(args),
        "source_sha256":hashes,"frozen_original_bundle_sha256":_sha(Path(frozen["folder"])/"manifest.json"),
        "original_proxy_fingerprint":fourier_policy_fingerprint(old),
        "new_proxy_fingerprint":fourier_policy_fingerprint(fresh),"dual_config":asdict(dual),
        "exercise_policy_frozen":True,"no_new_policy_fit":True,"valuation_used_for_selection":False,
        "tolerances_unchanged":True,"price_allowance":".001+1e-4*abs(price)",
        "delta_allowance":".002+.005*abs(delta)","confidence_multiplier":1.96,
        "bound_scope":"fixed finite kernel, actual weak law and exercise dates",
        "american_pricing_ready":False,"stock_delta_ready":False}
    (folder/"design_before_training.json").write_text(json.dumps(_jsonable(design),indent=2,allow_nan=False)+"\n")
    print("TRAINING",args.label,flush=True)
    training=value_fourier_stopping_policy(fresh,config=replace(fresh.config,
        valuation_seed=args.dual_seed,valuation_paths=args.dual_paths),trace=True)["trace"]
    weights=train_rough_dual_weights(fresh,config=dual,training_trace=training)
    (folder/"dual_before_valuation.json").write_text(json.dumps(_jsonable({
        "time_edges":weights.time_edges,"weights":weights.weights,"diagnostics":dict(weights.diagnostics)}),indent=2,allow_nan=False)+"\n")
    pd.DataFrame(weights.diagnostics["candidates"]).to_csv(folder/"training_selection.csv",index=False)
    print("SELECTED",weights.diagnostics["selected_candidate"],weights.diagnostics["selected_score"],flush=True)
    results={}
    traces={}
    for name,proxy in (("historical_control",old),("refreshed_tensor_obstacle",fresh)):
        print("VALUATION",name,flush=True)
        trace=value_fourier_stopping_policy(proxy,config=replace(proxy.config,
            valuation_seed=args.valuation_seed,valuation_paths=args.valuation_paths),trace=True)["trace"]
        traces[name]=trace
        fit=oldweights if name=="historical_control" else weights
        for rule,policy in (("original",original),("dense_tensor",selected)):
            result=evaluate_trace_policy(trace,policy=policy,martingale_weights=fit)
            results[(name,rule)]=result
    for key in ("stocks","factors","discounted_payoffs","european_martingale"):
        np.testing.assert_array_equal(traces["historical_control"][key],traces["refreshed_tensor_obstacle"][key])
    rows,samples=[],{}
    for (name,rule),value in results.items():
        lower=value.controlled_samples[:,0];upper=value.dual_samples[:,0];gap=upper-lower
        lowstat,gapstat=_summary(lower),_summary(gap)
        allowance=.001+1e-4*abs(lowstat["mean"])
        budget=gapstat["high"]+1.96*lowstat["standard_error"]
        reference=results[("historical_control",rule)]
        change=lower-reference.controlled_samples[:,0]
        upperchange=upper-reference.dual_samples[:,0]
        rows.append({"control":name,"policy":rule,"paths":len(lower),"price":lowstat["mean"],
            "price_se":lowstat["standard_error"],"upper":float(upper.mean()),"gap":gapstat["mean"],
            "gap_se":gapstat["standard_error"],"price_allowance":allowance,"price_error_budget":budget,
            "price_budget_pass":budget<=allowance,"paired_lower_change":_summary(change)["mean"],
            "paired_lower_change_se":_summary(change)["standard_error"],
            "paired_upper_change":_summary(upperchange)["mean"],
            "paired_upper_change_se":_summary(upperchange)["standard_error"],
            "raw_identical_to_original_control":bool(np.array_equal(value.raw_samples,reference.raw_samples)),
            "stops_identical_to_original_control":bool(np.array_equal(value.stop_indices,reference.stop_indices)),
            "american_pricing_ready":False,"stock_delta_ready":False})
        prefix=name+"__"+rule
        samples[prefix+"__lower"]=lower;samples[prefix+"__upper"]=upper
        samples[prefix+"__raw"]=value.raw_samples[:,0];samples[prefix+"__stops"]=value.stop_indices[:,0]
    for name in ("historical_control","refreshed_tensor_obstacle"):
        np.testing.assert_array_equal(results[(name,"original")].dual_samples,results[(name,"dense_tensor")].dual_samples)
    table=pd.DataFrame(rows)
    table.to_csv(folder/"comparison.csv",index=False)
    np.savez_compressed(folder/"paired_samples.npz",**samples)
    changes=[key for key,digest in hashes.items() if _sha(ROOT/key)!=digest]
    manifest={**design,"completed_at":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.perf_counter()-started,
        "sources_unchanged":not changes,"source_changes":changes,"samples_sha256":_sha(folder/"paired_samples.npz"),
        "same_paths":True,"same_upper_across_policies":True}
    (folder/"manifest.json").write_text(json.dumps(_jsonable(manifest),indent=2,allow_nan=False)+"\n")
    print(table.to_string(index=False),flush=True)
    print("SECONDS",manifest["elapsed_seconds"],"SOURCECHANGES",changes,flush=True)


if __name__=="__main__":
    main()
