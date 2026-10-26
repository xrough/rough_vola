"""Predeclared independent weak-time delta probe on the frozen exercise dates."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rvlab.experiments._pricing_rough_frozen_policy import load_frozen_policy_bundle
from rvlab.experiments._pricing_rough_stock_delta import frozen_policy_martingale_delta_samples

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main():
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_stock_delta_v1/frozen_joint_time_refinement"
    folder.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__).resolve(),ROOT/"src/rvlab/experiments/_pricing_rough_stock_delta.py",
        ROOT/"src/rvlab/experiments/_pricing_rough_frozen_policy.py"]
    source_paths+=sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    bundle=output/"rough_martingale_centering_v1/frozen_joint_policy"
    declaration={"version":"rough_stock_delta_time_refinement_v1","seed":712005042,
        "paths":2048,"steps":256,"exercise_steps":64,"bump_relatives":[.01,.005,.0025],
        "source_sha256":sources,"frozen_bundle_manifest_sha256":sha(bundle/"manifest.json"),
        "price_atol":.001,"price_rtol":.0001,"delta_atol":.002,"delta_rtol":.005,
        "confidence_multiplier":1.96,"spread_fraction":.1,
        "target":"Independent256-step weak-law sensitivity, same frozen policies and65exercise dates",
        "valuation_not_used_to_choose_design":True,"optimal_delta_ready":False}
    path=folder/"design_before_valuation.json"
    if path.exists():
        old=json.loads(path.read_text())
        if {k:v for k,v in old.items() if k!="sealed_at_utc"}!=declaration:
            raise ValueError("Changed predeclared time-delta design")
    else:path.write_text(json.dumps({**declaration,"sealed_at_utc":now()},indent=2,sort_keys=True))
    manifest_path=folder/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest["design_sha256"]!=sha(path):raise ValueError("Changed design")
        for item in manifest["files"]:
            if sha(item["path"])!=item["sha256"]:raise ValueError("Changed saved time probe")
        print("cached independent time-refinement probe",flush=True);return
    frozen=load_frozen_policy_bundle(bundle); start=now();clock=perf_counter()
    result=frozen_policy_martingale_delta_samples(frozen,seed=712005042,paths=2048,steps=256)
    elapsed=perf_counter()-clock
    if {str(p.relative_to(ROOT)):sha(p) for p in source_paths}!=sources:
        raise ValueError("Numerical sources changed during independent time-delta valuation")
    result["estimates"].to_csv(folder/"estimates.csv",index=False)
    result["comparisons"].to_csv(folder/"comparisons.csv",index=False)
    arrays={name+"__"+key:values for name,d in result["samples"].items() for key,values in d.items()}
    arrays.update({name+"__stops":values for name,values in result["stops"].items()})
    arrays["relative_spots"]=result["relative_spots"]
    np.savez_compressed(folder/"samples.npz",**arrays)
    files=[folder/name for name in ("estimates.csv","comparisons.csv","samples.npz")]
    manifest={**result["diagnostics"],"design_sha256":sha(path),"started_at_utc":start,
        "finished_at_utc":now(),"elapsed_seconds":elapsed,"source_sha256":sources,
        "sources_unchanged":True,"files":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(json.dumps({"elapsed_seconds":elapsed,"seed":712005042,"paths":2048,"steps":256}),flush=True)
    print(result["estimates"].to_string(index=False),flush=True)

if __name__=="__main__":main()
