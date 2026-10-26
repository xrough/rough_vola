"""Deterministic interpolation audit on spent historical factor states only."""
from pathlib import Path
from dataclasses import replace
import sys,hashlib,json,time
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_frozen_policy import _load_trace
from rvlab.experiments._pricing_rough_node_refinement_fast import polynomial_node_values,refined_stock_knots,interpolate_nodes
output=root/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise';bundle=output/'rough_larger_policy_v1/32768_paths_two_starts'
frozen=load_larger_policy_bundle(bundle);proxy=frozen['proxy'];selected=frozen['selected_policy']
historical=json.loads((Path(frozen['manifest']['historical_bundle_path'])/'manifest.json').read_text());checkpoint=json.loads(Path(historical['source_checkpoint']['path']).read_text())['payload']
trace=_load_trace(checkpoint,'final_valuation');rows=[]
for j in [16,32,48,60]:
    factors=trace['factors'][:8,j];direct_nodes=np.r_[0.,np.linspace(.85,1.1,2001)]
    def value(nodes):
        euro=replace(proxy.value_premiums[j],stock_nodes=nodes).european_values(factors)
        q=np.clip(polynomial_node_values(selected.continuations[j],nodes,factors),0.,selected.premium_cap)
        return np.clip(np.maximum(q,np.maximum(proxy.strike*(1-nodes),0.)-euro),0.,selected.premium_cap)
    truth=value(direct_nodes)
    for subdivisions in [1,4,8]:
        nodes=np.r_[0.,proxy.config.stock_knots if subdivisions==1 else refined_stock_knots(proxy.config.stock_knots,subdivisions)]
        approx=interpolate_nodes(nodes,value(nodes),direct_nodes)
        error=approx-truth
        rows.append({'date_index':j,'date_years':float(proxy.continuation_times[j]),'spent_states':8,'subdivisions':subdivisions,'positive_nodes':len(nodes)-1,'stock_lower':85.,'stock_upper':110.,'direct_evaluation_nodes':2001,'rms_interpolation_error':float(np.sqrt(np.mean(error[:,1:]**2))),'max_absolute_interpolation_error':float(np.max(abs(error[:,1:]))),'new_acceptance_evidence':False})
print(pd.DataFrame(rows).to_string(index=False))

from datetime import datetime,timezone
folder=output/'rough_gap_time_refinement_v1/interpolation_validation_v1'
if folder.exists():raise FileExistsError(folder)
folder.mkdir(parents=True)
pd.DataFrame(rows).to_csv(folder/'interpolation.csv',index=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[Path(__file__),root/'src/rvlab/experiments/_pricing_rough_node_refinement_fast.py',root/'src/rvlab/experiments/_pricing_rough_node_refinement.py',root/'src/rvlab/pricing/rough_stopping_fourier.py',root/'src/rvlab/pricing/rough_tensor_premium.py']
for path in sources:
    target=folder/'source'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
manifest={'version':'rough_grid_interpolation_validation_v1','completed_at':datetime.now(timezone.utc).isoformat(),
    'no_new_paths_or_fit':True,'new_acceptance_evidence':False,'used_for_final_control_selection':False,
    'states_source':'first8 historical16a final_valuation states at exercise dates16,32,48,60; spent350005042 stream',
    'objective':'compare linear stock-node interpolation against direct selected frozen Snell premium on a dense deterministic stock mesh',
    'stock_mesh':'2001 equally spaced spots85 through110; zero-stock node used to preserve existing lower interpolation tail',
    'interpretation':'error in frozen control approximation, not an option-price bias bound or certification of the exercise policy',
    'all_date_rms_strictly_decreases':all(rows[i+2]['rms_interpolation_error']<rows[i+1]['rms_interpolation_error']<rows[i]['rms_interpolation_error'] for i in range(0,len(rows),3)),
    'source_checkpoint_sha256':sha(Path(historical['source_checkpoint']['path'])),
    'model_seal_sha256':sha(bundle/'frozen_policy_before_final.json'),'coefficients_sha256':sha(bundle/'frozen_policy_coefficients.npz'),
    'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources}}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file()]
canonical=json.dumps(entries,sort_keys=True,separators=(',',':')).encode()
(folder/'preservation_inventory.json').write_text(json.dumps({'files':entries,'inventory_sha256':hashlib.sha256(canonical).hexdigest()},indent=2)+'\n')
