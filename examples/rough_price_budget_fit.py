"""Freeze shared-control coefficients using only spent1094/1095 paths."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_price_budget_controls import predictable_basis,stopped_basis,fit_weights
OUTPUT=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise'
PRIOR=OUTPUT/'rough_bellman_validation_v1/bellman_control_v1'
ANCHOR=OUTPUT/'rough_price_budget_v1/spent_terminal_anchor_v1'
FOLDER=OUTPUT/'rough_price_budget_v1/spent_shared_fit_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(_jsonable(v),indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def main():
 if FOLDER.exists():raise FileExistsError('Preserve previous coefficient studies')
 FOLDER.mkdir(parents=True);sources=[ROOT/n for n in read(PRIOR/'design_before_training.json')['source_sha256']]
 sources += [ROOT/'src/rvlab/experiments'/n for n in ('_pricing_rough_price_budget.py','_pricing_rough_price_budget_controls.py')]+[Path(__file__)]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
 inputs=[PRIOR/f'final_{i}_{suffix}' for i in range(2) for suffix in ('manifest.json','samples.npz')]
 inputs += [ANCHOR/n for n in ('manifest.json','preservation_inventory.json','stream_0_samples.npz','stream_1_samples.npz')]
 pins={str(p.relative_to(OUTPUT)):sha(p) for p in inputs}
 for p in sources:
  target=FOLDER/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 specs=[]
 for family in ('terminal','poststop','targets','targets_stable','full_stopped'):
  for ridge in (1e-6,1e-4,.01):specs.append({'family':family,'ridge':ridge,'eligible_for_selection':False})
 for family in ('gated','gated_anchor'):
  for ridge in (1e-4,.001,.01,.1,1.):specs.append({'family':family,'ridge':ridge,'eligible_for_selection':True})
 design={'version':'rough_price_budget_spent_fit_v1','created_at':now(),'source_sha256':hashes,'input_sha256':pins,
  'fit_seeds':[1094005042,1095005042],'crossfit':'both directions; all4096rows each; choose minimum mean heldout variance on eligible gated family',
  'specifications':specs,'coefficient_bound':2.,'ridge_fit_then_raw_coefficient_projection':True,
  'final_refit':'all8192 spent rows after specification chosen','new_outer_paths':0,'used_for_fresh_acceptance':False,
  'gates':'LEFT one,clip((S/K-1)/.05,+/-3),clip(factors normalized by frozen perdate selected-model center/scale,+/-3)',
  'bins':4,'feature_order':'bin,process,gate; optional exactly centered pairedEuropean anchor last',
  'no_regression_intercept_at_valuation':True,'no_empirical_valuation_recentering':True,'tolerances_changed':False}
 write(FOLDER/'design_before_fitting.json',design);bundle=load_larger_policy_bundle(OUTPUT/'rough_larger_policy_v1/32768_paths_two_starts')
 data=[];basis_names=None
 for i in range(2):
  with np.load(PRIOR/f'final_{i}_samples.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  with np.load(ANCHOR/f'stream_{i}_samples.npz',allow_pickle=False) as z:anchor=z['shared_control'].copy()
  t=a['gh16_stops'];base=a['latest_martingale'];target=a['target_martingales']
  f={'terminal':a['european_martingale'][:,-1,None],
   'poststop':(a['european_martingale'][:,-1]-a['european_martingale'][np.arange(len(t)),t])[:,None],
   'targets':stopped_basis(target-base[:,:,None],t),
   'stable':np.column_stack([stopped_basis(a[k][:,:,None],t) for k in ('european_martingale','premium_martingale','micro_boundary_martingale')]),
   'prior':stopped_basis(a['prior_local_corrections'],t),'grid':stopped_basis(a['grid_corrections'],t),'tail':stopped_basis(a['tail_corrections'],t)}
  f['targets_stable']=np.column_stack((f['targets'],f['stable']))
  f['full_stopped']=np.column_stack([f[k] for k in ('targets','stable','prior','grid','tail')])
  f['gated'],names=predictable_basis(a,bundle);f['gated_anchor']=np.column_stack((f['gated'],anchor));basis_names=names
  np.savez_compressed(FOLDER/f'spent_{i}_basis.npz',lower=a['candidate3_lower'],upper=a['candidate3_upper'],anchor=anchor,**f)
  data.append((a['candidate3_lower'],f))
 rows=[];archive={}
 for k,spec in enumerate(specs):
  for fold in (0,1):
   y,f=data[fold];v,vf=data[1-fold];x=f[spec['family']];vx=vf[spec['family']]
   weights,record=fit_weights(x,y,ridge=spec['ridge']);archive[f'weights_{k}_{fold}']=weights
   residual=v-vx@weights
   rows.append({'spec_index':k,**spec,'fit_seed':(1094005042,1095005042)[fold],
    'diagnostic_seed':(1094005042,1095005042)[1-fold],'features':len(weights),'rows_fit':len(y),'rows_diagnostic':len(v),
    'training_sd':record['training_residual_standard_deviation'],'diagnostic_sd':float(np.std(residual,ddof=1)),
    'diagnostic_mean_shift':float(np.mean(residual-v)),'max_abs_unprojected_coefficient':record['max_abs_unprojected_coefficient'],
    'active_projection_count':record['active_projection_count'],'used_for_fresh_acceptance':False})
 frame=pd.DataFrame(rows);frame.to_csv(FOLDER/'spent_trials.csv',index=False)
 scores=frame.loc[frame.eligible_for_selection].groupby('spec_index',sort=True).diagnostic_sd.apply(lambda x:float(np.mean(x*x)))
 chosen=int(scores.idxmin());spec=specs[chosen];x=np.concatenate([f[spec['family']] for y,f in data]);y=np.concatenate([y for y,f in data])
 weights,fit=fit_weights(x,y,ridge=spec['ridge']);archive['selected_weights']=weights
 np.savez_compressed(FOLDER/'all_fit_coefficients.npz',**archive)
 names=basis_names+(['paired-European-anchor'] if spec['family']=='gated_anchor' else [])
 seal={'version':'rough_price_budget_shared_weights_v1','sealed_at':now(),'design_sha256':sha(FOLDER/'design_before_fitting.json'),
  'selected_spec_index':chosen,'selected_specification':spec,'weights':weights,'basis_columns':names,'fit':fit,
  'fit_seeds':[1094005042,1095005042],'all_rows_refitted':8192,'fresh_paths_generated_before_seal':False,
  'selection_used_only_spent_split_diagnostics':True,'coefficient_archive_sha256':sha(FOLDER/'all_fit_coefficients.npz')}
 write(FOLDER/'weights_before_final.json',seal)
 if any(sha(ROOT/n)!=v for n,v in hashes.items()) or any(sha(OUTPUT/n)!=v for n,v in pins.items()):raise ValueError('Spent fit source/input changed')
 write(FOLDER/'manifest.json',{**design,'complete':True,'selected_specification':spec,'selected_spec_index':chosen,'completed_at':now(),
  'weights_sha256':sha(FOLDER/'weights_before_final.json'),'trials':len(rows)})
 entries=[{'path':str(p.relative_to(FOLDER)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(FOLDER.rglob('*')) if p.is_file()]
 write(FOLDER/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
 print(frame.loc[frame.eligible_for_selection].to_string(index=False));print('SELECTED',spec,'features',len(weights),'score_sd',np.sqrt(scores.loc[chosen]))

if __name__=='__main__':main()
