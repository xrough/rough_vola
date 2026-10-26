"""Excluded spent-path pilot of predictable tail switches; no new MC paths."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_frozen_policy import _load_trace
from rvlab.experiments._pricing_rough_tail_switch import current_control_workspace,FEATURES,FAMILIES
from rvlab.experiments._pricing_rough_local_control import correction_basis,fit_correction_weights,correction_endpoints

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(_jsonable(obj),indent=2,allow_nan=False)+'\n')
def stats(a):return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a)))

def main():
    start=time.perf_counter();output=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise'
    folder=output/'rough_tail_attribution_v1/spent_switch_pilot_v1'
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    source=[Path(__file__)]+[ROOT/'src/rvlab/experiments'/n for n in ('_pricing_rough_tail_switch.py','_pricing_rough_local_control.py','_pricing_rough_node_refinement_fast.py','_pricing_rough_exercise_block.py','_pricing_rough_large_policy.py','_pricing_rough_frozen_policy.py','_pricing_rough_policy_improvement.py')]+sorted((ROOT/'src/rvlab/pricing').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source}
    for p in source:
        target=folder/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    model=output/'rough_larger_policy_v1/32768_paths_two_starts';frozen=load_larger_policy_bundle(model)
    priorpath=output/'rough_policy_validation_v1/local_control_v1/controls_before_final.json';prior=json.loads(priorpath.read_text())
    gridpath=output/'rough_gap_time_refinement_v1/stock_grid_control_v2/controls_before_final.json';grid=json.loads(gridpath.read_text())
    design={'version':'rough_tail_switch_spent_pilot_v1','created_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':hashes,'spent_seed':350005042,'paths':512,'fit_paths':256,'heldout_paths':256,
        'features':FEATURES,'families':FAMILIES,'bins':[1,4],'penalties':[0.,1e-4],
        'selected_policy_seal_sha256':sha(model/'frozen_policy_before_final.json'),
        'prior_control_seal_sha256':sha(priorpath),'grid_control_seal_sha256':sha(gridpath),
        'new_acceptance_evidence':False,'used_for_formal_control_selection':False,'new_paths_generated':False}
    write(folder/'design_before_pilot.json',design)
    historical=json.loads((Path(frozen['manifest']['historical_bundle_path'])/'manifest.json').read_text())
    checkpoint=json.loads(Path(historical['source_checkpoint']['path']).read_text())['payload']
    trace=_load_trace(checkpoint,'final_valuation');pathkeys={'discounted_payoffs','european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale','micro_poststop_martingale','stocks','european_proxy','continuation_proxy','factors','stop_indices'}
    trace={k:(v[:512].copy() if k in pathkeys else v) for k,v in trace.items()}
    print('RECONSTRUCT CURRENT M ON SPENT512',flush=True)
    w=current_control_workspace(frozen['proxy'],frozen['selected_policy'],frozen['weights'],prior['controls'][prior['selected']],grid['controls'][grid['selected']],trace)
    controls={'current baseline':{'family':'fallback','bins':1,'penalty':0.,'weights':np.zeros(len(FEATURES))}};trials=[];rows=[]
    for family,features in FAMILIES.items():
        for bins in (1,4):
            basis=correction_basis(w['tail_corrections'][:,:,features],w['times'],bins)
            for penalty in (0.,1e-4):
                name=f'{family} {bins} bins penalty{penalty:g}'
                weight,record=fit_correction_weights(w['payoff'][:256],w['current_martingale'][:256],w['stops'][:256],basis[:256],bins=bins,penalty=penalty)
                trials.append({'control':name,'family':family,**record})
                if weight is not None:
                    full=np.zeros((bins,len(FEATURES)));full[:,features]=weight.reshape(bins,-1)
                    controls[name]={'family':family,'bins':bins,'penalty':penalty,'weights':full.ravel()}
    base=None
    for name,control in controls.items():
        basis=correction_basis(w['tail_corrections'],w['times'],control['bins'])
        lo,up=correction_endpoints(w['payoff'],w['current_martingale'],w['stops'],basis,np.asarray(control['weights']))
        if base is None:base=(lo,up)
        gap,gse=stats((up-lo)[256:]);price,pse=stats(lo[256:]);change,cse=stats((up-base[1])[256:])
        rows.append({'control':name,'family':control['family'],'heldout_paths':256,'gap':gap,'gap_se':gse,'price':price,'price_se':pse,
            'upper_change':change,'upper_change_se':cse,'max_absolute_lower_change':float(np.max(abs(lo-base[0]))),
            'selection_score':gap+1.96*np.sqrt(256/8192)*(gse+pse),'new_acceptance_evidence':False})
    pd.DataFrame(rows).to_csv(folder/'pilot_comparison.csv',index=False);pd.DataFrame(trials).to_csv(folder/'pilot_trials.csv',index=False)
    write(folder/'pilot_controls.json',controls)
    manifest={**design,'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
        'diagnostics':w['diagnostics'],'sources_unchanged':all(sha(ROOT/n)==h for n,h in hashes.items())}
    write(folder/'manifest.json',manifest)
    files=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file()]
    write(folder/'preservation_inventory.json',{'files':files,'inventory_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
