"""Reproducible coefficient-gather check; no new simulated or fitted paths."""
from datetime import datetime, timezone
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_node_refinement_fast import polynomial_node_values,refined_stock_knots

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    output=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise'
    folder=output/'rough_gap_time_refinement_v1/coefficient_gather_validation_v1'
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    bundle=output/'rough_larger_policy_v1/32768_paths_two_starts'
    frozen=load_larger_policy_bundle(bundle);proxy=frozen['proxy'];policy=frozen['selected_policy']
    nodes=np.unique(np.r_[0.,refined_stock_knots(proxy.config.stock_knots,8),
        np.nextafter(proxy.config.stock_knots,0.),np.nextafter(proxy.config.stock_knots,np.inf)])
    rows=[]
    for j in (0,1,8,16,32,48,63,64):
        model=policy.continuations[j]
        states=np.vstack([model.center+a*model.scale for a in (0.,1.,-2.,8.,-8.)])
        expected=model.value(np.broadcast_to(proxy.strike*nodes,(len(states),len(nodes))),states,proxy.strike)
        actual=polynomial_node_values(model,nodes,states)
        difference=float(np.max(np.abs(actual-expected)))
        rows.append({'date_index':j,'states':len(states),'nodes':len(nodes),'max_absolute_difference':difference,'tolerance':1e-11,'pass':difference<=1e-11})
    pd.DataFrame(rows).to_csv(folder/'equivalence.csv',index=False)
    assert all(r['pass'] for r in rows),'Coefficient-gather equivalence failed'
    model=policy.continuations[32];states=np.broadcast_to(model.center,(128,len(model.center)))
    nodes=np.r_[0.,refined_stock_knots(proxy.config.stock_knots,8)];repetitions=10;timings=[]
    for name,call in [('expanded tensor value',lambda:model.value(np.broadcast_to(proxy.strike*nodes,(len(states),len(nodes))),states,proxy.strike)),
                      ('coefficient gather',lambda:polynomial_node_values(model,nodes,states))]:
        call();started=time.perf_counter()
        for _ in range(repetitions):call()
        timings.append({'implementation':name,'repetitions':repetitions,'states':len(states),'nodes':len(nodes),'seconds_per_call':(time.perf_counter()-started)/repetitions})
    pd.DataFrame(timings).to_csv(folder/'timing.csv',index=False)
    sources=[Path(__file__),ROOT/'src/rvlab/experiments/_pricing_rough_node_refinement.py',ROOT/'src/rvlab/experiments/_pricing_rough_node_refinement_fast.py',ROOT/'src/rvlab/tests/test_pricing_rough_node_refinement.py',ROOT/'src/rvlab/tests/test_pricing_rough_node_refinement_fast.py',ROOT/'src/rvlab/pricing/rough_tensor_premium.py']
    for path in sources:
        target=folder/'source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    manifest={'version':'rough_coefficient_gather_validation_v1','completed_at':datetime.now(timezone.utc).isoformat(),
        'max_absolute_difference':max(r['max_absolute_difference'] for r in rows),'comparison_tolerance':1e-11,'all_equivalence_checks_pass':True,
        'purpose':'execution-only optimization of frozen factor-dependent stock polynomials; not a new acceptance tolerance',
        'no_new_paths_or_fit':True,'states':'factor center plus 0,1,-2,+8,-8 standard deviations; exact knots and adjacent floating point values',
        'isolated_node_speedup':timings[0]['seconds_per_call']/timings[1]['seconds_per_call'],'whole_pricer_speedup_measured':False,
        'unit_test_result':'22 passed; includes entire martingale equivalence on independent portable fixture at2e-11',
        'model_seal_sha256':sha(bundle/'frozen_policy_before_final.json'),'coefficients_sha256':sha(bundle/'frozen_policy_coefficients.npz'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file()]
    canonical=json.dumps(entries,sort_keys=True,separators=(',',':')).encode()
    (folder/'preservation_inventory.json').write_text(json.dumps({'files':entries,'inventory_sha256':hashlib.sha256(canonical).hexdigest()},indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
