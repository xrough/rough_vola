"""Spent-path exploratory attribution of exercise-interval centering.

No stream in this probe is fresh acceptance evidence. Its deterministic subset
is used only to decide whether a separately sealed independent study is useful.
"""
from dataclasses import replace
from datetime import datetime, timezone
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
from rvlab.experiments._pricing_rough_frozen_policy import reconstruct_joint_frozen_policy,_load_trace
from rvlab.experiments._pricing_rough_stopping_refresh import refresh_tensor_obstacle_control
from rvlab.experiments._pricing_rough_exercise_block import block_control_trace
from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    output=ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder=output/"rough_stopping_refresh_v1/exploratory_spent512_block"
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    sources=[Path(__file__)]+[ROOT/"src/rvlab/experiments"/name for name in (
        "_pricing_rough_frozen_policy.py","_pricing_rough_stopping_refresh.py","_pricing_rough_exercise_block.py")]+sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    hashes={str(p.relative_to(ROOT)):_sha(p) for p in sources}
    for path in sources:
        target=folder/"source"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    frozen=reconstruct_joint_frozen_policy(output)
    old,selected,original,weights=[frozen[k] for k in ("proxy","selected_policy","original_policy","weights")]
    record=json.loads(Path(frozen["manifest"]["source_checkpoint"]["path"]).read_text())["payload"]
    full=_load_trace(record,"final_valuation")
    n=len(full["discounted_payoffs"])
    pathkeys={"discounted_payoffs","european_martingale","premium_martingale","intrinsic_martingale",
        "micro_boundary_martingale","micro_poststop_martingale","stocks","european_proxy",
        "continuation_proxy","factors","stop_indices"}
    trace={k:(v[:512].copy() if k in pathkeys else v) for k,v in full.items()}
    fresh=refresh_tensor_obstacle_control(old,selected)
    cloned=replace(weights,policy_fingerprint=fourier_policy_fingerprint(fresh))
    declaration={"created_at":datetime.now(timezone.utc).isoformat(),"source_sha256":hashes,
        "purpose":"exploratory spent-path hypothesis probe, excluded from fresh acceptance",
        "new_acceptance_evidence":False,"spent_seed":trace["seed"],"subset":"first512 in sealed historical path order",
        "original_paths":n,"new_training":False,"new_selection":False,
        "old_weight_coefficients_applied_without_refit":True,"unchanged_tolerances":True,
        "frozen_bundle_sha256":_sha(Path(frozen["folder"])/"manifest.json"),
        "original_policy_fingerprint":fourier_policy_fingerprint(old),
        "tensor_proxy_fingerprint":fourier_policy_fingerprint(fresh)}
    (folder/"design_before_probe.json").write_text(json.dumps(_jsonable(declaration),indent=2,allow_nan=False)+"\n")
    oldblock=block_control_trace(old,trace,decision_policy=selected)
    newblock=block_control_trace(fresh,trace,decision_policy=selected)
    candidates={"old micro / old weights":(trace,weights),
        "old obstacle block / old weights":(oldblock,weights),
        "tensor block / unit":(newblock,(1.,1.)),
        "tensor block / old weights":(newblock,cloned)}
    rows,samples=[],{}
    baseline={rule:evaluate_trace_policy(trace,policy=policy,martingale_weights=weights)
        for rule,policy in (("original",original),("dense tensor",selected))}
    for name,(data,fit) in candidates.items():
        for rule,policy in (("original",original),("dense tensor",selected)):
            value=evaluate_trace_policy(data,policy=policy,martingale_weights=fit)
            reference=baseline[rule]
            lower=value.controlled_samples[:,0];upper=value.dual_samples[:,0];gap=upper-lower
            change=gap-(reference.dual_samples[:,0]-reference.controlled_samples[:,0])
            se=lambda v:float(np.std(v,ddof=1)/np.sqrt(len(v)))
            rows.append({"control":name,"policy":rule,"paths":len(lower),"lower":float(lower.mean()),
                "lower_se":se(lower),"upper":float(upper.mean()),"gap":float(gap.mean()),"gap_se":se(gap),
                "paired_gap_change":float(change.mean()),"paired_gap_change_se":se(change),
                "raw_unchanged":bool(np.array_equal(value.raw_samples,reference.raw_samples)),
                "stops_unchanged":bool(np.array_equal(value.stop_indices,reference.stop_indices)),
                "new_acceptance_evidence":False})
            key=name.replace(" / ","__").replace(" ","_")+"__"+rule.replace(" ","_")
            samples[key+"__lower"]=lower;samples[key+"__upper"]=upper;samples[key+"__gap"]=gap
    table=pd.DataFrame(rows);table.to_csv(folder/"comparison.csv",index=False)
    np.savez_compressed(folder/"paired_samples.npz",**samples)
    changes=[k for k,v in hashes.items() if _sha(ROOT/k)!=v]
    manifest={**declaration,"completed_at":datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":time.perf_counter()-started,"sources_unchanged":not changes,"source_changes":changes,
        "samples_sha256":_sha(folder/"paired_samples.npz"),"max_probability_error":newblock["block_max_probability_error"]}
    (folder/"manifest.json").write_text(json.dumps(_jsonable(manifest),indent=2,allow_nan=False)+"\n")
    entries=[{"path":str(p.relative_to(folder)),"bytes":p.stat().st_size,"sha256":_sha(p)} for p in sorted(folder.rglob("*")) if p.is_file()]
    canonical=json.dumps(entries,sort_keys=True,separators=(",",":")).encode()
    (folder/"preservation_inventory.json").write_text(json.dumps({"files":entries,"inventory_sha256":hashlib.sha256(canonical).hexdigest()},indent=2)+"\n")
    print(table.to_string(index=False),flush=True)
    print("SECONDS",manifest["elapsed_seconds"],"SOURCECHANGES",changes,flush=True)


if __name__=="__main__":main()
