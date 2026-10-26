"""Independent BS stopping and European Heston controls for delta diagnostics."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from rvlab.experiments._pricing_rough_stock_delta import (
    frozen_policy_delta_samples,frozen_policy_martingale_delta_samples)
from rvlab.pricing import (DiscountCurve,DividendSchedule,RoughHestonParams,
    HestonParams,SpotMarket,approximate_fractional_kernel,heston_price,PricingConfig)
from rvlab.pricing.rough_dual import RoughDualWeights,RoughDualConfig
from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
from rvlab.pricing.rough_policy_reference import build_black_scholes_reference_policy
from rvlab.pricing.rough_stopping_reference import black_scholes_stopping_reference
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy,fourier_policy_fingerprint

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main():
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_stock_delta_v1/independent_reference_checks"
    folder.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),ROOT/"src/rvlab/experiments/_pricing_rough_stock_delta.py"]
    sources+=sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    identities={str(p.relative_to(ROOT)):sha(p) for p in sources}
    design={"version":"rough_stock_delta_references_v1",
        "bs_seed":713005042,"bs_paths":4096,"bs_training_seed":713105042,
        "heston_seeds":[714005042,715005042],"heston_paths":32768,
        "heston_training_seed":714105042,"heston_time_steps":[128,256],
        "bump_relatives":[.01,.005,.0025],"source_sha256":identities,
        "delta_atol":.002,"delta_rtol":.005,"ci_multiplier":1.96,
        "reference_scope":"American BS identical finite calendar; European Heston continuous affine model",
        "optimal_rough_delta_ready":False}
    design_path=folder/"design_before_valuation.json"
    if design_path.exists():
        old=json.loads(design_path.read_text())
        if {k:v for k,v in old.items() if k!="sealed_at_utc"}!=design:raise ValueError("Changed reference design")
    else:design_path.write_text(json.dumps({**design,"sealed_at_utc":now()},indent=2,sort_keys=True))
    manifest_path=folder/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest["design_sha256"]!=sha(design_path):raise ValueError("Changed design")
        for item in manifest["files"]:
            if sha(item["path"])!=item["sha256"]:raise ValueError("Changed reference output")
        print("cached independent reference checks",flush=True);return
    maturity=60/365; market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    rows=[]; timings=[]; arrays={}
    def build(params,training_seed,style):
        kernel=approximate_fractional_kernel(params.hurst,1.,4 if params.hurst<.5 else 1,min_time_ratio=.003)
        config=RoughAmericanConfig(training_paths=256,valuation_paths=64,steps=64,
            exercise_steps=16,training_seed=training_seed,valuation_seed=training_seed+1,
            batch_size=512,stock_knots=(.7,.85,.95,1.,1.05,1.15,1.3))
        proxy=train_fourier_stopping_policy(params,market,kernel,strike=100.,maturity=maturity,
            config=config,is_call=False,exercise_style=style,premium_cap=.7)
        rule=ControlledExercisePolicy(proxy,proxy.exercise_times,proxy.continuations,
            proxy.exercise_at_inception,(training_seed,),0.,.7,{})
        weights=RoughDualWeights(np.array([0.,maturity]),np.array([[1.,1.,0.]]),
            fourier_policy_fingerprint(proxy),training_seed,training_seed+2,training_seed+1,
            RoughDualConfig(components=("european","premium","micro_boundary")),{"selected_candidate":0})
        return {"proxy":proxy,"original_policy":rule,"selected_policy":rule,"weights":weights}
    # Independent Gaussian DP supplies the actions and independently resolved
    # same-calendar reference sensitivity; valuation paths supply no fitting.
    bs=build(RoughHestonParams(.04,.04,1.5,0.,-.6,.15),713105042,"american")
    reference_policy=build_black_scholes_reference_policy(bs["proxy"],spot_nodes=16385)
    bs["original_policy"]=bs["selected_policy"]=reference_policy
    start=perf_counter();result=frozen_policy_martingale_delta_samples(bs,seed=713005042,paths=4096,steps=64)
    timings.append({"case":"American BS finite calendar","elapsed_seconds":perf_counter()-start})
    reference=black_scholes_stopping_reference(market,volatility=.2,strike=100.,
        exercise_times=bs["proxy"].exercise_times,spot_nodes=16385,check_refinement=True)
    for row in result["estimates"].query("policy=='selected_policy'").to_dict("records"):
        allowance=.002+.005*abs(reference.delta)
        rows.append({"case":"American BS finite calendar",**row,"reference_delta":reference.delta,
            "reference_spatial_change":reference.diagnostics["spatial_delta_change"],
            "reference_domain_change":reference.diagnostics["domain_delta_change"],
            "reference_stable":reference.diagnostics["delta_stable"],
            "reference_allowance":allowance,"absolute_error":abs(row["mean"]-reference.delta),
            "reference_gate_passed":abs(row["mean"]-reference.delta)+row["ci_half_width"]<=allowance})
    for key,value in result["samples"]["selected_policy"].items():arrays["bs__"+key]=value
    # European Heston validates actual weak-law finite-bump sensitivities
    # against a separate continuous-model Fourier derivative. Its remaining
    # weak-time difference is observed, never identified as a policy error.
    heston=build(RoughHestonParams(.04,.045,1.5,.3,-.6,.5),714105042,"european")
    hp=HestonParams(.04,.045,1.5,.3,-.6);discount=market.discount_curve.discount(maturity)
    bump=.005
    reference_delta=float((heston_price(hp,forward=(100.+bump)/discount,discount=discount,
        strike=100.,maturity=maturity,is_call=False,config=PricingConfig(256,240.,1024))
        -heston_price(hp,forward=(100.-bump)/discount,discount=discount,strike=100.,
        maturity=maturity,is_call=False,config=PricingConfig(256,240.,1024)))/(2*bump))
    for seed,steps in zip(design["heston_seeds"],design["heston_time_steps"]):
        start=perf_counter();result=frozen_policy_delta_samples(heston,seed=seed,paths=32768,steps=steps)
        timings.append({"case":f"European Heston {steps}steps","elapsed_seconds":perf_counter()-start})
        for row in result["estimates"].query("policy=='selected_policy'").to_dict("records"):
            allowance=.002+.005*abs(reference_delta)
            rows.append({"case":f"European Heston {steps}steps",**row,"reference_delta":reference_delta,
                "reference_spatial_change":0.,"reference_domain_change":0.,"reference_stable":True,
                "reference_allowance":allowance,"absolute_error":abs(row["mean"]-reference_delta),
                "reference_gate_passed":abs(row["mean"]-reference_delta)+row["ci_half_width"]<=allowance})
        for key,value in result["samples"]["selected_policy"].items():arrays[f"heston{steps}__"+key]=value
    if {str(p.relative_to(ROOT)):sha(p) for p in sources}!=identities:
        raise ValueError("Numerical sources changed during reference validation")
    pd.DataFrame(rows).to_csv(folder/"reference_estimates.csv",index=False)
    pd.DataFrame(timings).to_csv(folder/"timings.csv",index=False)
    np.savez_compressed(folder/"samples.npz",**arrays)
    files=[folder/name for name in ("reference_estimates.csv","timings.csv","samples.npz")]
    manifest={"version":"rough_stock_delta_references_v1","design_sha256":sha(design_path),
        "source_sha256":identities,"sources_unchanged":True,
        "bs_reference":{"price":reference.price,"delta":reference.delta,"diagnostics":dict(reference.diagnostics)},
        "heston_reference_delta":reference_delta,"tolerances_changed":False,"optimal_rough_delta_ready":False,
        "files":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print(pd.DataFrame(timings).to_string(index=False),flush=True)

if __name__=="__main__":main()
