"""Bounded independent-path delta campaign; no new policy or control fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import numpy as np
import pandas as pd

from rvlab.experiments._pricing_rough_frozen_policy import load_frozen_policy_bundle
from rvlab.experiments._pricing_rough_stock_delta import (
    _summary, frozen_policy_martingale_delta_samples)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices",nargs="+",type=int,default=[])
    parser.add_argument("--collect",action="store_true")
    args=parser.parse_args()
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_stock_delta_v1"/"frozen_joint_2x4096"
    folder.mkdir(parents=True,exist_ok=True)
    bundle=output/"rough_martingale_centering_v1/frozen_joint_policy"
    source_paths=[Path(__file__).resolve(),ROOT/"src/rvlab/experiments/_pricing_rough_stock_delta.py",
        ROOT/"src/rvlab/experiments/_pricing_rough_frozen_policy.py"]
    source_paths+=sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    design={"version":"rough_stock_delta_v1","seeds":[710005042,711005042],
        "paths_per_stream":4096,"bump_relatives":[.01,.005,.0025],
        "steps":128,"exercise_steps":64,"source_sha256":sources,
        "frozen_bundle_manifest_sha256":sha(bundle/"manifest.json"),
        "frozen_bundle_arrays_sha256":sha(bundle/"coefficients.npz"),
        "pilot_seed":790005042,"pilot_paths":512,
        "pilot_excluded":True,"terminal_cv_spent_pilot_seed":510005042,
        "price_atol":.001,"price_rtol":.0001,"delta_atol":.002,"delta_rtol":.005,
        "confidence_multiplier":1.96,"spread_fraction":.1,
        "no_valuation_selection_or_fitting":True,"optimal_delta_ready":False}
    design_path=folder/"design_before_primary_valuation.json"
    if design_path.exists():
        previous=json.loads(design_path.read_text())
        if {k:v for k,v in previous.items() if k!="sealed_at_utc"}!=design:
            raise ValueError("Changed predeclared stock-delta design or numerical source")
    else:
        design_path.write_text(json.dumps({**design,"sealed_at_utc":now()},indent=2,sort_keys=True))
    frozen=load_frozen_policy_bundle(bundle)
    for index in args.indices:
        if index not in (0,1):raise ValueError("primary indices must be0 or1")
        record=folder/f"stream_{index}.json"
        if record.exists():
            previous=json.loads(record.read_text())
            if previous["design_sha256"]!=sha(design_path):raise ValueError("Changed stream design")
            for item in previous["files"]:
                if sha(item["path"])!=item["sha256"]:raise ValueError("Changed stream artifact")
            print(json.dumps({"index":index,"cached":True}),flush=True);continue
        seed=design["seeds"][index]; start=now(); clock=perf_counter()
        result=frozen_policy_martingale_delta_samples(frozen,seed=seed,paths=4096,
            bump_relatives=tuple(design["bump_relatives"]),steps=128)
        elapsed=perf_counter()-clock
        result["estimates"].to_csv(folder/f"stream_{index}_estimates.csv",index=False)
        result["comparisons"].to_csv(folder/f"stream_{index}_comparisons.csv",index=False)
        arrays={name+"__"+key:values for name,d in result["samples"].items() for key,values in d.items()}
        arrays.update({name+"__stops":values for name,values in result["stops"].items()})
        arrays["relative_spots"]=result["relative_spots"]
        np.savez_compressed(folder/f"stream_{index}_samples.npz",**arrays)
        files=[folder/f"stream_{index}_{suffix}" for suffix in ("estimates.csv","comparisons.csv","samples.npz")]
        declaration={"index":index,"seed":seed,"paths":4096,"started_at_utc":start,
            "finished_at_utc":now(),"elapsed_seconds":elapsed,"design_sha256":sha(design_path),
            "diagnostics":result["diagnostics"],"source_sha256":sources,
            "files":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}
        record.write_text(json.dumps(declaration,indent=2,sort_keys=True))
        print(json.dumps({"index":index,"seed":seed,"elapsed_seconds":elapsed}),flush=True)
    if args.collect:
        records=[]; all_arrays=[]
        for index in (0,1):
            record=json.loads((folder/f"stream_{index}.json").read_text())
            if record["design_sha256"]!=sha(design_path):raise ValueError("Changed primary design")
            for item in record["files"]:
                if sha(item["path"])!=item["sha256"]:raise ValueError("Changed primary artifact")
            with np.load(folder/f"stream_{index}_samples.npz",allow_pickle=False) as archive:
                arrays={key:archive[key].copy() for key in archive.files}
            records.append(record); all_arrays.append(arrays)
        joined={key:np.concatenate([a[key] for a in all_arrays],axis=0)
            for key in all_arrays[0] if key!="relative_spots"}
        rows=[];comparisons=[]
        for name in ("original_policy","selected_policy"):
            for estimator in ("raw","stopped martingale"):
                for level in design["bump_relatives"]:
                    key=name+"__"+f"{estimator}:{level:g}"
                    summary=_summary(joined[key]);allowance=.002+.005*abs(summary["mean"])
                    rows.append({"policy":name,"estimator":estimator,"bump_relative":level,
                        "paths":8192,**summary,"delta_allowance":allowance,
                        "sampling_passed":summary["ci_half_width"]<=allowance,
                        "optimal_delta_ready":False})
                for coarse,fine in zip(design["bump_relatives"][:-1],design["bump_relatives"][1:]):
                    values=joined[name+"__"+f"{estimator}:{fine:g}"]-joined[name+"__"+f"{estimator}:{coarse:g}"]
                    comparisons.append({"comparison":"bump refinement","policy":name,"estimator":estimator,
                        "coarse_bump":coarse,"fine_bump":fine,**_summary(values)})
        for estimator in ("raw","stopped martingale"):
            for level in design["bump_relatives"]:
                values=joined["selected_policy__"+f"{estimator}:{level:g}"]-joined["original_policy__"+f"{estimator}:{level:g}"]
                comparisons.append({"comparison":"selected minus original","policy":"paired policies",
                    "estimator":estimator,"coarse_bump":np.nan,"fine_bump":level,**_summary(values)})
        pd.DataFrame(rows).to_csv(folder/"pooled_estimates.csv",index=False)
        pd.DataFrame(comparisons).to_csv(folder/"pooled_comparisons.csv",index=False)
        manifest={"version":"rough_stock_delta_v1","complete":True,"paths":8192,
            "seed_count":2,"design_sha256":sha(design_path),"records":records,
            "independent_paths":True,"all_streams_retained":True,
            "tolerances_changed":False,"optimal_delta_ready":False,
            "limitations":["Frozen lower-policy finite-bump sensitivity; price bounds do not bound optimal delta.",
                "No new time/exercise/kernel derivative validation is inferred from this fixed-law campaign."]}
        (folder/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
        print(pd.DataFrame(rows).to_string(index=False),flush=True)
        print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=="__main__":main()
