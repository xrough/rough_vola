"""Predeclared larger controlled-policy fits with independent final streams."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_frozen_policy import reconstruct_joint_frozen_policy,_encode_models
from rvlab.experiments._pricing_rough_policy_improvement import _policy_model_identity
from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy,fourier_policy_fingerprint
from rvlab.pricing.rough_policy_diagnostics import fit_controlled_policy_iteration,select_trace_policy,evaluate_trace_policy
from rvlab.pricing.rough_tensor_premium import tensor_premium_fitter


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _now():return datetime.now(timezone.utc).isoformat()
def _write(path,value):path.write_text(json.dumps(_jsonable(value),indent=2,allow_nan=False)+"\n")
def _stat(value):
    mean=float(value.mean());se=float(value.std(ddof=1)/np.sqrt(len(value)))
    return {"mean":mean,"se":se,"low":mean-1.96*se,"high":mean+1.96*se}


def main():
    started=time.perf_counter()
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_larger_policy_v1/32768_paths_two_starts"
    if folder.exists():raise FileExistsError(f"Preserved study exists: {folder}")
    folder.mkdir(parents=True)
    sources=[Path(__file__)]+[ROOT/"src/rvlab/experiments"/name for name in (
        "_pricing_rough_frozen_policy.py","_pricing_rough_policy_improvement.py","_pricing_rough_large_policy.py")]+sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    hashes={str(path.relative_to(ROOT)):_sha(path) for path in sources}
    for path in sources:
        target=folder/"source"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    frozen=reconstruct_joint_frozen_policy(output)
    proxy,weights=frozen["proxy"],frozen["weights"]
    previous=frozen["selected_policy"]
    design={"version":"rough_larger_policy_v1","created_at":_now(),
        "historical_bundle_path":frozen["folder"],"historical_bundle_sha256":_sha(Path(frozen["folder"])/"manifest.json"),
        "source_sha256":hashes,"proxy_fingerprint":fourier_policy_fingerprint(proxy),
        "training_paths":32768,"training_seed":610005042,
        "selection_paths":8192,"selection_seed":620005042,
        "final_streams":[{"seed":630005042,"paths":8192},{"seed":631005042,"paths":8192}],
        "candidates":{"previous dense tensor":{"type":"frozen fallback"},
            "dense tensor ridge1e-4":{"degree":2,"ridge":1e-4,"stock_knots":proxy.config.stock_knots},
            "dense tensor ridge1e-5":{"degree":2,"ridge":1e-5,"stock_knots":proxy.config.stock_knots}},
        "selection_rule":"paired lower gain minus1.96SE versus frozen fallback on independent selection only",
        "training_paths_shared_between_predeclared_fits":True,"kernel_transition_exercise_calendar_control_frozen":True,
        "no_new_dual_training":True,"final_valuation_used_for_selection":False,
        "unchanged_tolerances":True,"price_allowance":".001+1e-4*abs(V)",
        "delta_allowance":".002+.005*abs(delta)","confidence_multiplier":1.96,
        "american_pricing_ready":False,"stock_delta_ready":False}
    _write(folder/"design_before_training.json",design)
    print("TRAINING32768",flush=True)
    training=value_fourier_stopping_policy(proxy,config=replace(proxy.config,
        valuation_paths=32768,valuation_seed=610005042),trace=True)["trace"]
    candidates={"previous dense tensor":previous}
    for name,ridge in (("dense tensor ridge1e-4",1e-4),("dense tensor ridge1e-5",1e-5)):
        print("FITTING",name,flush=True)
        candidates[name]=fit_controlled_policy_iteration(proxy,training,martingale_weights=weights,
            fit_continuation=tensor_premium_fitter(degree=2,ridge=ridge),
            stock_knots=proxy.config.stock_knots,ridge=ridge,premium_floor=0.,premium_cap=proxy.premium_cap)
    del training
    print("SELECTION8192",flush=True)
    selection=value_fourier_stopping_policy(proxy,config=replace(proxy.config,
        valuation_paths=8192,valuation_seed=620005042),trace=True)["trace"]
    chosen=select_trace_policy(selection,candidates,baseline_name="previous dense tensor",martingale_weights=weights)
    all_seeds=tuple(sorted({seed for candidate in chosen["candidates"].values() for seed in candidate.training_seeds}))
    candidates={name:replace(policy,training_seeds=all_seeds) for name,policy in chosen["candidates"].items()}
    pd.DataFrame(chosen["rows"]).to_csv(folder/"training_selection.csv",index=False)
    arrays={"times":previous.times};metadata={}
    for index,(name,policy) in enumerate(candidates.items()):
        metadata[name]={"models":_encode_models(policy.continuations,"policy"+str(index),arrays),
            "exercise_at_inception":policy.exercise_at_inception,"training_seeds":policy.training_seeds,
            "premium_floor":policy.premium_floor,"premium_cap":policy.premium_cap,"diagnostics":dict(policy.diagnostics),
            "identity":_policy_model_identity(policy)}
    np.savez_compressed(folder/"frozen_policy_coefficients.npz",**arrays)
    sealed={"version":"rough_larger_policy_bundle_v1","sealed_at":_now(),
        "selected_name":chosen["selected_name"],"selection_diagnostics":chosen["diagnostics"],
        "historical_bundle_path":frozen["folder"],"historical_bundle_sha256":design["historical_bundle_sha256"],
        "coefficient_archive":"frozen_policy_coefficients.npz","coefficient_archive_sha256":_sha(folder/"frozen_policy_coefficients.npz"),
        "all_selection_training_seeds":all_seeds,"policies":metadata,
        "no_final_paths_generated_before_policy_seal":True}
    _write(folder/"frozen_policy_before_final.json",sealed)
    print("SELECTED",chosen["selected_name"],flush=True)
    del selection
    rows=[];pooled={name:[] for name in candidates};pooled_upper=[]
    for spec in design["final_streams"]:
        seed=spec["seed"];print("FINAL",seed,flush=True)
        trace=value_fourier_stopping_policy(proxy,config=replace(proxy.config,
            valuation_paths=spec["paths"],valuation_seed=seed),trace=True)["trace"]
        values={name:evaluate_trace_policy(trace,policy=policy,martingale_weights=weights) for name,policy in candidates.items()}
        base=values["previous dense tensor"]
        samples={}
        for index,(name,value) in enumerate(values.items()):
            lower=value.controlled_samples[:,0];upper=value.dual_samples[:,0];gap=upper-lower
            gain=lower-base.controlled_samples[:,0]
            if not np.array_equal(value.dual_samples,base.dual_samples):raise AssertionError("Changed common upper")
            lo,ga,gi=_stat(lower),_stat(gap),_stat(gain)
            allowance=.001+1e-4*abs(lo["mean"]);budget=ga["high"]+1.96*lo["se"]
            rows.append({"seed":seed,"policy":name,"paths":len(lower),"selected_before_final":name==chosen["selected_name"],
                "price":lo["mean"],"price_se":lo["se"],"upper":float(upper.mean()),"gap":ga["mean"],"gap_se":ga["se"],
                "paired_lower_gain":gi["mean"],"paired_lower_gain_se":gi["se"],"gain_low":gi["low"],"gain_high":gi["high"],
                "exercise_change_fraction":float(np.mean(value.stop_indices!=base.stop_indices)),
                "price_allowance":allowance,"price_error_budget":budget,"price_budget_pass":budget<=allowance,
                "same_upper_samples":True,"american_pricing_ready":False,"stock_delta_ready":False})
            samples["policy"+str(index)+"_lower"]=lower;samples["policy"+str(index)+"_raw"]=value.raw_samples[:,0]
            samples["policy"+str(index)+"_stops"]=value.stop_indices[:,0]
            pooled[name].append(lower)
        samples["common_upper"]=base.dual_samples[:,0]
        np.savez_compressed(folder/("final_"+str(seed)+"_samples.npz"),**samples)
        _write(folder/("final_"+str(seed)+"_manifest.json"),{"seed":seed,"paths":spec["paths"],
            "sample_sha256":_sha(folder/("final_"+str(seed)+"_samples.npz")),"policies":list(values),
            "policy_seal_sha256":_sha(folder/"frozen_policy_before_final.json"),"completed_at":_now()})
        pd.DataFrame(rows).to_csv(folder/"per_seed.csv",index=False)
        pooled_upper.append(base.dual_samples[:,0]);del trace,values,base
    upper=np.concatenate(pooled_upper);baseline=np.concatenate(pooled["previous dense tensor"])
    pooledrows=[]
    for name,arrays in pooled.items():
        lower=np.concatenate(arrays);lo,ga,gi=_stat(lower),_stat(upper-lower),_stat(lower-baseline)
        # Equal independent streams: the pooled i.i.d. SE is reported alongside
        # a stratified SE which retains each predeclared stream as a stratum.
        strat=lambda chunks:float(np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in chunks))/sum(map(len,chunks)))
        lse=strat(arrays);gse=strat([u-l for u,l in zip(pooled_upper,arrays)])
        gainse=strat([l-b for l,b in zip(arrays,pooled["previous dense tensor"])])
        allowance=.001+1e-4*abs(lo["mean"]);budget=ga["mean"]+1.96*gse+1.96*lse
        pooledrows.append({"policy":name,"paths":len(lower),"streams":2,"selected_before_final":name==chosen["selected_name"],
            "price":lo["mean"],"price_se":lse,"price_iid_se":lo["se"],"upper":float(upper.mean()),
            "gap":ga["mean"],"gap_se":gse,"paired_lower_gain":gi["mean"],"paired_lower_gain_se":gainse,
            "gain_low":gi["mean"]-1.96*gainse,"gain_high":gi["mean"]+1.96*gainse,
            "price_allowance":allowance,"price_error_budget":budget,"price_budget_pass":budget<=allowance,
            "same_upper_samples":True,"american_pricing_ready":False,"stock_delta_ready":False})
    pd.DataFrame(pooledrows).to_csv(folder/"pooled.csv",index=False)
    changes=[key for key,digest in hashes.items() if _sha(ROOT/key)!=digest]
    manifest={**design,"completed_at":_now(),"elapsed_seconds":time.perf_counter()-started,
        "selected_name":chosen["selected_name"],"policy_seal_sha256":_sha(folder/"frozen_policy_before_final.json"),
        "source_changes":changes,"sources_unchanged":not changes,"same_upper_across_policies":True,
        "completed_final_streams":2,"completed_final_paths":16384}
    _write(folder/"manifest.json",manifest)
    entries=[{"path":str(p.relative_to(folder)),"bytes":p.stat().st_size,"sha256":_sha(p)} for p in sorted(folder.rglob("*")) if p.is_file()]
    canonical=json.dumps(entries,sort_keys=True,separators=(",",":")).encode()
    _write(folder/"preservation_inventory.json",{"files":entries,"inventory_sha256":hashlib.sha256(canonical).hexdigest()})
    print(pd.DataFrame(pooledrows).to_string(index=False),flush=True)
    print("SECONDS",manifest["elapsed_seconds"],"SOURCECHANGES",changes,flush=True)


if __name__=="__main__":main()
