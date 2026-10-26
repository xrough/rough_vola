"""Bounded fresh Heston follow-up after the terminal-control precision failure."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import sys,hashlib,json
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from rvlab.experiments._pricing_rough_stock_delta import frozen_policy_martingale_delta_samples
from rvlab.pricing import (DiscountCurve,DividendSchedule,RoughHestonParams,HestonParams,
    SpotMarket,approximate_fractional_kernel,heston_price,PricingConfig)
from rvlab.pricing.rough_dual import RoughDualWeights,RoughDualConfig
from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy,fourier_policy_fingerprint

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main():
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_stock_delta_v1/heston_stopped_control_followup";folder.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__).resolve(),ROOT/"src/rvlab/experiments/_pricing_rough_stock_delta.py"]
    paths+=sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    design={"version":"heston_stopped_control_delta_followup_v1","seeds":[716005042,717005042],
        "paths_per_stream":2048,"steps":[128,256],"bump_relatives":[.01,.005,.0025],
        "training_seed":716105042,"weights":[1.,0.,0.],"source_sha256":sources,
        "design_context":"Bounded follow-up after terminal conditional-control Heston256 reference gate failed; all prior outcomes retained",
        "delta_atol":.002,"delta_rtol":.005,"confidence_multiplier":1.96,"optimal_rough_delta_ready":False}
    dp=folder/"design_before_valuation.json"
    if dp.exists():
        old=json.loads(dp.read_text())
        if {k:v for k,v in old.items() if k!="sealed_at_utc"}!=design:raise ValueError("Changed follow-up design")
    else:dp.write_text(json.dumps({**design,"sealed_at_utc":now()},indent=2,sort_keys=True))
    mp=folder/"manifest.json"
    if mp.exists():
        manifest=json.loads(mp.read_text())
        if manifest["design_sha256"]!=sha(dp):raise ValueError("Changed followupdesign")
        for item in manifest["files"]:
            if sha(item["path"])!=item["sha256"]:raise ValueError("Changed savedfollowup")
        print("cached Heston stopped-control follow-up",flush=True);return
    maturity=60/365;market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.5);hp=HestonParams(.04,.045,1.5,.3,-.6)
    config=RoughAmericanConfig(training_paths=256,valuation_paths=64,steps=64,exercise_steps=16,
        training_seed=716105042,valuation_seed=716105043,batch_size=512)
    proxy=train_fourier_stopping_policy(params,market,approximate_fractional_kernel(.5,1.,1),
        strike=100.,maturity=maturity,config=config,is_call=False,exercise_style="european",premium_cap=.7)
    rule=ControlledExercisePolicy(proxy,proxy.exercise_times,proxy.continuations,False,(config.training_seed,),0.,.7,{})
    weights=RoughDualWeights(np.array([0.,maturity]),np.array([[1.,0.,0.]]),fourier_policy_fingerprint(proxy),
        config.training_seed,716105044,config.valuation_seed,
        RoughDualConfig(components=("european","premium","micro_boundary")),{"selected_candidate":0})
    frozen={"proxy":proxy,"original_policy":rule,"selected_policy":rule,"weights":weights}
    discount=market.discount_curve.discount(maturity);reference_rows=[]
    for nodes,cutoff,stock_bump in ((256,240.,.005),(512,480.,.005),(512,480.,.0025)):
        cfg=PricingConfig(nodes,cutoff,1024)
        delta=float((heston_price(hp,forward=(100+stock_bump)/discount,discount=discount,strike=100.,
            maturity=maturity,is_call=False,config=cfg)-heston_price(hp,forward=(100-stock_bump)/discount,
            discount=discount,strike=100.,maturity=maturity,is_call=False,config=cfg))/(2*stock_bump))
        reference_rows.append({"nodes":nodes,"cutoff":cutoff,"stock_bump":stock_bump,"delta":delta})
    reference_delta=reference_rows[-1]["delta"]
    reference_change=max(abs(r["delta"]-reference_delta) for r in reference_rows)
    rows=[];timings=[];arrays={}
    for seed,steps in zip(design["seeds"],design["steps"]):
        clock=perf_counter();result=frozen_policy_martingale_delta_samples(frozen,seed=seed,paths=2048,steps=steps)
        timings.append({"seed":seed,"steps":steps,"elapsed_seconds":perf_counter()-clock})
        for row in result["estimates"].query("policy=='selected_policy'").to_dict("records"):
            allowance=.002+.005*abs(reference_delta)
            rows.append({**row,"reference_delta":reference_delta,"absolute_error":abs(row["mean"]-reference_delta),
                "reference_change_diagnostic":reference_change,"reference_allowance":allowance,
                "reference_gate_passed":abs(row["mean"]-reference_delta)+row["ci_half_width"]+reference_change<=allowance})
        for key,value in result["samples"]["selected_policy"].items():arrays[f"steps{steps}__"+key]=value
        print(json.dumps(timings[-1]),flush=True)
    if {str(p.relative_to(ROOT)):sha(p) for p in paths}!=sources:raise ValueError("Sourceschanged duringfollowup")
    pd.DataFrame(rows).to_csv(folder/"reference_estimates.csv",index=False)
    pd.DataFrame(timings).to_csv(folder/"timings.csv",index=False)
    pd.DataFrame(reference_rows).to_csv(folder/"fourier_reference_refinement.csv",index=False)
    np.savez_compressed(folder/"samples.npz",**arrays)
    files=[folder/name for name in ("reference_estimates.csv","timings.csv","fourier_reference_refinement.csv","samples.npz")]
    manifest={"version":design["version"],"design_sha256":sha(dp),"source_sha256":sources,
        "sources_unchanged":True,"reference_delta":reference_delta,"reference_change_diagnostic":reference_change,
        "tolerances_changed":False,"optimal_rough_delta_ready":False,
        "files":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True));print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=="__main__":main()
