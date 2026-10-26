"""Independent fixed-budget shared-control validation of frozen GH16 bounds."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,os,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import numpy as np
import rough_bellman_control as old
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_price_budget import conditional_european_replay,shared_european_control
from rvlab.experiments._pricing_rough_price_budget_controls import predictable_basis,common_shift

OUTPUT=old.OUTPUT;PRIOR=old.FOLDER
FIT=OUTPUT/'rough_price_budget_v1/spent_shared_fit_v1'
FOLDER=OUTPUT/'rough_price_budget_v1/shared_control_v1'
STREAMS=[{'seed':seed,'paths':16384} for seed in (1106005042,1107005042,1108005042)]
TOLERANCES={'price_atol':.001,'price_rtol':1e-4,'delta_atol':.002,'delta_rtol':.005,'confidence_multiplier':1.96,'spread_fraction':.1,'tolerances_changed':False}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(_jsonable(v),indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def source_paths():return [ROOT/n for n in read(PRIOR/'design_before_training.json')['source_sha256']]+[
 ROOT/'src/rvlab/experiments'/n for n in ('_pricing_rough_price_budget.py','_pricing_rough_price_budget_controls.py')]+[Path(__file__)]
def verified():
 d=read(FOLDER/'design_before_valuation.json')
 if any(sha(ROOT/n)!=h or sha(FOLDER/'source'/n)!=h for n,h in d['source_sha256'].items()):raise ValueError('Frozen shared-control numerical source changed')
 if any(sha(OUTPUT/n)!=h for n,h in d['input_sha256'].items()):raise ValueError('Frozen shared-control input changed')
 if sha(FOLDER/'weights_before_final.json')!=d['coefficient_sha256']:raise ValueError('Frozen shared-control coefficients changed')
 return d

def prepare():
 if FOLDER.exists():raise FileExistsError('Preserve the previous independent study')
 prior=read(PRIOR/'design_before_training.json');selection=read(PRIOR/'selection_before_final.json')
 seal=read(FIT/'weights_before_final.json');fitmanifest=read(FIT/'manifest.json')
 if (selection['selected_control']!='target 4 bins penalty0' or seal['selected_specification']!={'family':'gated_anchor','ridge':.01,'eligible_for_selection':True}
     or len(seal['weights'])!=193 or seal['fresh_paths_generated_before_seal'] is not False
     or fitmanifest['complete'] is not True or fitmanifest['weights_sha256']!=sha(FIT/'weights_before_final.json')):
  raise ValueError('Frozen GH16 dual, spent specification or coefficient identity changed')
 forbidden=set(prior['forbidden_prior_seeds'])|{1092005042,1093005042,1094005042,1095005042}
 if len({s['seed'] for s in STREAMS})!=3 or forbidden&{s['seed'] for s in STREAMS}:raise ValueError('Fresh streams overlap prior fitting/valuation')
 FOLDER.mkdir(parents=True);hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
 for p in source_paths():
  target=FOLDER/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 pins=dict(prior['input_sha256']);inputs=[PRIOR/n for n in ('manifest.json','design_before_training.json','selection_before_final.json')]
 inputs += [FIT/n for n in ('manifest.json','design_before_fitting.json','weights_before_final.json','all_fit_coefficients.npz','spent_trials.csv','preservation_inventory.json')]
 anchor=OUTPUT/'rough_price_budget_v1/spent_terminal_anchor_v1'
 inputs += [anchor/n for n in ('manifest.json','preservation_inventory.json')]
 pins.update({str(p.relative_to(OUTPUT)):sha(p) for p in inputs})
 (FOLDER/'weights_before_final.json').write_bytes((FIT/'weights_before_final.json').read_bytes())
 design={'version':'rough_price_budget_shared_control_v1','created_at':now(),'source_sha256':hashes,'input_sha256':pins,
  'streams':STREAMS,'total_paths':49152,'estimator_order':['unchanged control','shared control'],
  'frozen_policy':'GH16 blend1','frozen_dual_control':'target 4 bins penalty0','physical_steps':128,'exercise_intervals':64,
  'coefficient_sha256':sha(FOLDER/'weights_before_final.json'),'coefficient_fit_seeds':[1094005042,1095005042],
  'selected_specification':seal['selected_specification'],'features':193,'bins':4,'coefficient_bound':2.,
  'left_stock_feature_scale':.05,'left_feature_clip':3.,'factor_normalization':'frozen selected-policy perdate center/scale',
  'column_order':'bin,process,gate; pairedEuropean anchor last',
  'selection_rule':'minimum mean heldout variance over both spent split directions; finalrefitall8192spentrows',
  'no_fresh_selection_or_fitting':True,'no_empirical_valuation_recentering':True,'no_regression_intercept_at_valuation':True,
  'both_endpoints_same_shared_shift':True,'gap_pointwise_unchanged':True,'gap_improvement_claimed':False,
  'shared_shift_expectation_zero':True,'upper_is_Rogers_max_after_shift':False,
  'continuous_European_price_used_to_center':False,'independent_reference_mean_used':False,
  'all_paths_retained':True,'no_new_paths_generated_before_design':True,'forbidden_prior_seeds':sorted(forbidden),
  'price_error_budget':'gap +1.96*(lower_priceSE+gapSE)','price_allowance':'.001+1e-4*abs(V)',
  'new_policy_fits':0,'new_dual_control_fits':0,'optimal_value_ready':False,'stock_delta_ready':False,
  'projected_heldout_lower_sd':.01571427894276138,'projected_price_se_49152':.01571427894276138/np.sqrt(49152),
  'runtime_projection_only':True,**TOLERANCES}
 write(FOLDER/'design_before_valuation.json',design)
 write(FOLDER/'execution_environment.json',{'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),
  'numpy':np.__version__,'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}})
 print('PREPARED',now(),len(hashes),flush=True)

def final(index):
 design=verified();spec=design['streams'][index];seal=read(FOLDER/'weights_before_final.json')
 if any(FOLDER.glob(f'stream_{index}_*')):raise FileExistsError('Preserve previous final attempt')
 started=now();before=perf_counter();print('FINAL',index,spec,started,flush=True)
 prior=read(PRIOR/'design_before_training.json');prior['forbidden_prior_seeds']=design['forbidden_prior_seeds']
 trace,w,data,diagnostics=old.generate(spec['seed'],spec['paths'],prior)
 controls=read(PRIOR/'selection_before_final.json')['controls']
 data.update(old.evaluate(data,w['payoff'],trace['times'],controls))
 old.tail.save_samples(FOLDER/f'stream_{index}_price_samples.npz',trace,w,data)
 reference={k:trace[k] for k in ('stocks','factors','actual_time_grid')}
 conditional,conditional_diagnostic=conditional_european_replay(old.inputs()[0]['proxy'],seed=spec['seed'],paths=spec['paths'],reference=reference)
 minimal={'payoff':w['payoff'],'european_martingale':trace['european_martingale'][:,:,0]}
 anchor=shared_european_control(minimal,conditional)
 np.savez_compressed(FOLDER/f'stream_{index}_conditional_samples.npz',**conditional,shared_control=anchor)
 arrays={**data,**reference,'stocks':trace['stocks'],'factors':trace['factors'],
  **{k:trace[k][:,:,0] for k in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale')}}
 basis,names=predictable_basis(arrays,old.inputs()[0],include_anchor=True,anchor=anchor)
 if names!=seal['basis_columns']:raise ValueError('Shared-control feature order changed')
 lower,upper=data['candidate3_lower'],data['candidate3_upper'];newlower,newupper,shift=common_shift(lower,upper,basis,seal['weights'])
 if not np.allclose(newupper-newlower,upper-lower,rtol=0.,atol=2e-13):raise FloatingPointError('Common shift changed pathwise gap')
 np.savez_compressed(FOLDER/f'stream_{index}_shift_samples.npz',basis=basis,unchanged_lower=lower,unchanged_upper=upper,
  shared_lower=newlower,shared_upper=newupper,shift=shift)
 files={name:sha(FOLDER/f'stream_{index}_{name}_samples.npz') for name in ('price','conditional','shift')}
 schema={}
 for name in files:
  with np.load(FOLDER/f'stream_{index}_{name}_samples.npz',allow_pickle=False) as z:schema[name]={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
 write(FOLDER/f'stream_{index}_manifest.json',{'index':index,**spec,'started_at':started,'completed_at':now(),'elapsed_seconds':perf_counter()-before,
  'design_sha256':sha(FOLDER/'design_before_valuation.json'),'coefficient_sha256':sha(FOLDER/'weights_before_final.json'),
  'source_sha256':design['source_sha256'],'sources_unchanged':True,'sample_sha256':files,'array_schema':schema,
  'frozen_policy':'GH16 blend1','frozen_dual_control':'target 4 bins penalty0','old_diagnostics':diagnostics,
  'conditional_diagnostics':conditional_diagnostic,'new_fitting_or_selection':False,'all_paths_retained':True,
  'gap_pointwise_unchanged':True,'empirical_valuation_recentering':False})
 verified();print('FINAL COMPLETE',index,now(),flush=True)

def collect():
 design=verified();records=[];dsha=sha(FOLDER/'design_before_valuation.json');seal=read(FOLDER/'weights_before_final.json')
 if (FOLDER/'manifest.json').exists():raise FileExistsError('Preserve completed campaign')
 for i,spec in enumerate(design['streams']):
  r=read(FOLDER/f'stream_{i}_manifest.json')
  if (any(r.get(k)!=v for k,v in {'index':i,**spec,'design_sha256':dsha,'coefficient_sha256':design['coefficient_sha256'],
      'source_sha256':design['source_sha256'],'sources_unchanged':True,'new_fitting_or_selection':False,
      'gap_pointwise_unchanged':True,'empirical_valuation_recentering':False,'all_paths_retained':True}.items())
      or set(r.get('sample_sha256',{}))!={'price','conditional','shift'}
      or any(sha(FOLDER/f'stream_{i}_{name}_samples.npz')!=h for name,h in r['sample_sha256'].items())
      or not seal['sealed_at']<design['created_at']<r['started_at']<r['completed_at']):raise ValueError('Incomplete/changed final stream or chronology')
  records.append(r)
 write(FOLDER/'manifest.json',{**design,'complete':True,'completed_at':now(),'records':records,'design_sha256':dsha,
  'all_streams_retained':True,'completed_paths':sum(s['paths'] for s in STREAMS)})
 entries=[{'path':str(p.relative_to(FOLDER)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(FOLDER.rglob('*')) if p.is_file()]
 write(FOLDER/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
 print('COLLECTED',now(),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','final','collect'));parser.add_argument('--index',type=int,choices=(0,1,2));args=parser.parse_args()
 if args.stage=='final' and args.index is None:parser.error('final requires --index')
 {'prepare':prepare,'final':lambda:final(args.index),'collect':collect}[args.stage]()
