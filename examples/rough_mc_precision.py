"""Equivalent selected-only execution and fixed196608-path MC validation."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,os,platform,sys
from time import perf_counter,process_time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import numpy as np
import rough_price_budget as previous
import rough_bellman_control as old
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_mc_precision import selected_samples
OUTPUT=old.OUTPUT;PARENT=previous.FOLDER;FIT=previous.FIT
FOLDER=OUTPUT/'rough_mc_precision_v1/independent_v1'
PREFLIGHT=OUTPUT/'rough_mc_precision_v1/spent_equivalence_v3'
PREFLIGHT_V2=OUTPUT/'rough_mc_precision_v1/spent_equivalence_v2'
PREFLIGHT_V1=OUTPUT/'rough_mc_precision_v1/spent_equivalence_v1'
STREAMS=[{'seed':s,'paths':65536} for s in (1119005042,1120005042,1121005042)]
PREFLIGHT_INDICES=np.r_[np.arange(251),511,512,1023,4095]
TOLERANCES=previous.TOLERANCES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(_jsonable(v),indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def source_paths():return [ROOT/n for n in read(PARENT/'design_before_valuation.json')['source_sha256']]+[ROOT/'src/rvlab/experiments/_pricing_rough_mc_precision.py',Path(__file__)]
def source_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
def snapshot(folder):
 hashes=source_hashes()
 for p in source_paths():
  target=folder/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
 return hashes

def inventory(folder):
 entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='preservation_inventory.json']
 write(folder/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})

def controls():
 bundle,prior,grid,_=old.inputs()
 tail=read(old.TAIL/'selection_before_final.json');target=read(old.FOLDER/'selection_before_final.json')
 seal=read(PARENT/'weights_before_final.json')
 return bundle,prior,grid,tail['controls'][tail['selected_control']],target['controls'][target['selected_control']],old.load_proxy(),seal

def generate(seed,paths,*,indices=None,forbidden=(),workers=2,workspace=None):
 b,p,g,t,c,m,seal=controls()
 def progress(phase,done,total):print('PROGRESS',seed,phase,done,total,now(),flush=True)
 a,q,d=selected_samples(b,p,g,t,c,m,np.asarray(seal['weights']),seed=seed,paths=paths,indices=indices,forbidden_seeds=forbidden,progress=progress,workers=workers,workspace=workspace,pure_chunk_paths=4096)
 if d['basis_names']!=seal['basis_columns']:raise ValueError('Frozen193-column identity changed')
 return a,q,d

def preflight():
 if PREFLIGHT.exists():raise FileExistsError('Preserve the previous spent equivalence trial')
 PREFLIGHT.mkdir(parents=True);sources=snapshot(PREFLIGHT);pins={}
 for i in range(2):
  for p in (old.FOLDER/f'final_{i}_samples.npz',FIT/f'spent_{i}_basis.npz',OUTPUT/f'rough_price_budget_v1/spent_terminal_anchor_v1/stream_{i}_samples.npz'):
   pins[str(p.relative_to(OUTPUT))]=sha(p)
 design={'version':'rough_mc_precision_spent_equivalence_v1','created_at':now(),'source_sha256':sources,'input_sha256':pins,
  'seeds':[1094005042,1095005042],'full_stream_paths':4096,'indices':PREFLIGHT_INDICES,'equivalence_atol':2e-13,
  'all_65_dates_and_129_physical_dates':True,'fresh_paths':0,'role':'spent implementation-equivalence and runtime only',
  'reference_old52_source_sha256':read(PARENT/'design_before_valuation.json')['source_sha256'],
  'work_batch_size':64,'physical_random_batch_size':512,'pure_state_workers':2,'pure_partition_rule':'contiguous4096chunks; preflight255 split128/127','pure_chunk_paths':4096,'file_backed_workspace':True,'no_random_or_fitting_in_children':True,'european65_and129_kept_separate':True}
 for p in (PREFLIGHT_V1/'manifest.json',PREFLIGHT_V1/'preservation_inventory.json',PREFLIGHT_V2/'manifest.json',PREFLIGHT_V2/'preservation_inventory.json'):pins[str(p.relative_to(OUTPUT))]=sha(p)
 write(PREFLIGHT/'design_before_evaluation.json',design);records=[];checks=[]
 preliminary={'role':'unsealed exploratory timings preserved for provenance, not acceptance','old_local_grid_target_seconds_batch64':[6.588320750044659,12.921840999973938,13.738263041945174],'old_local_grid_target_seconds_batch256':[7.915708417072892,13.064527374925092,14.489401458995417],'combined_seconds_batch64':21.166975250001997,'combined_seconds_batch256':24.569013750180602,'all_three_process_families_bit_exact':True,'selected_work_batch_size':64,'european65_vs129_exponent_max_difference':6.404745667978754e-15,'european65_vs129_factor_max_difference':1.0822664515137473e-13,'european65_vs129_sharing_rejected':True,'physical_first_probe_seconds':2.4156248751096427}
 write(PREFLIGHT/'preliminary_trials.json',preliminary)
 for i,seed in enumerate(design['seeds']):
  scratch=PREFLIGHT.parent/f'.{PREFLIGHT.name}_stream_{i}_scratch'
  if scratch.exists():raise FileExistsError('Preserve previous scratch attempt')
  scratch.mkdir()
  start=perf_counter();cpu=process_time();started=now();a,q,d=generate(seed,4096,indices=PREFLIGHT_INDICES,workspace=scratch)
  parallel_seconds=perf_counter()-start;serial_start=perf_counter()
  serial,serialq,serialdiag=generate(seed,4096,indices=PREFLIGHT_INDICES,workers=1)
  serial_seconds=perf_counter()-serial_start
  for k in a:
   error=float(np.nanmax(abs(a[k]-serial[k]))) if a[k].dtype.kind!='b' else float(np.any(a[k]!=serial[k]))
   same_nan=np.array_equal(np.isnan(a[k]),np.isnan(serial[k])) if a[k].dtype.kind=='f' else True
   checks.append({'stream':i,'field':'parallel_vs_serial|'+k,'max_abs_error':error,'same_nan_mask':same_nan,'passed':same_nan and error<=2e-13})
  np.savez_compressed(PREFLIGHT/f'stream_{i}_selected_samples.npz',**a)
  np.savez_compressed(PREFLIGHT/f'stream_{i}_conditional_samples.npz',**q)
  with np.load(old.FOLDER/f'final_{i}_samples.npz',allow_pickle=False) as z:
   for k in a:
    if k not in z.files or k in ('times','discounts','eligible','actual_time_grid'):continue
    expected=z[k][PREFLIGHT_INDICES];error=float(np.nanmax(abs(a[k]-expected))) if a[k].dtype.kind!='b' else float(np.any(a[k]!=expected))
    same_nan=np.array_equal(np.isnan(a[k]),np.isnan(expected)) if a[k].dtype.kind=='f' else True
    checks.append({'stream':i,'field':k,'max_abs_error':error,'same_nan_mask':same_nan,'passed':same_nan and error<=2e-13})
  with np.load(FIT/f'spent_{i}_basis.npz',allow_pickle=False) as z:
   for k,ref in [('basis','gated_anchor'),('unchanged_lower','lower'),('unchanged_upper','upper')]:
    error=float(np.max(abs(a[k]-z[ref][PREFLIGHT_INDICES])));checks.append({'stream':i,'field':k,'max_abs_error':error,'same_nan_mask':True,'passed':error<=2e-13})
  with np.load(OUTPUT/f'rough_price_budget_v1/spent_terminal_anchor_v1/stream_{i}_samples.npz',allow_pickle=False) as z:
   for k in q:
    if k in ('absolute_path_indices','physical_time_grid'):continue
    error=float(np.max(abs(q[k]-z[k][PREFLIGHT_INDICES])));checks.append({'stream':i,'field':k,'max_abs_error':error,'same_nan_mask':True,'passed':error<=2e-13})
  records.append({'seed':seed,'paths':4096,'evaluated':len(PREFLIGHT_INDICES),'started_at':started,'completed_at':now(),'elapsed_seconds':perf_counter()-start,
   'scratch_raw_bytes':sum(p.stat().st_size for p in scratch.rglob('*.npy')),'cpu_seconds_parent_only':process_time()-cpu,'parallel_seconds':parallel_seconds,'serial_seconds':serial_seconds,'diagnostics':d,'serial_diagnostics':serialdiag,'selected_sha256':sha(PREFLIGHT/f'stream_{i}_selected_samples.npz'),'conditional_sha256':sha(PREFLIGHT/f'stream_{i}_conditional_samples.npz')})
  import shutil
  shutil.rmtree(scratch)
  print('SPENT COMPLETE',i,records[-1]['elapsed_seconds'],flush=True)
 passed=all(c['passed'] for c in checks)
 estimated_bytes=sum((PREFLIGHT/f'stream_0_{part}_samples.npz').stat().st_size for part in ('selected','conditional'))*196608/len(PREFLIGHT_INDICES)
 write(PREFLIGHT/'equivalence_checks.json',checks)
 write(PREFLIGHT/'manifest.json',{**design,'complete':True,'completed_at':now(),'all_checks_passed':passed,'records':records,
  'checks_sha256':sha(PREFLIGHT/'equivalence_checks.json'),'estimated_total_final_compressed_bytes':estimated_bytes,
  'estimated_total_final_scratch_raw_bytes':max(r['scratch_raw_bytes'] for r in records)*196608/len(PREFLIGHT_INDICES),'performance_projection_only':True,'projected_uncontended_parallel_wall_seconds_per65536':max(r['parallel_seconds'] for r in records)*65536/len(PREFLIGHT_INDICES),
  'sources_unchanged':source_hashes()==sources})
 inventory(PREFLIGHT)
 if not passed:raise ValueError('Spent equivalence failed; retain trial and do not launch fresh paths')
 print('EQUIVALENCE PASSED',now(),estimated_bytes,flush=True)

def prepare():
 if FOLDER.exists():raise FileExistsError('Preserve the previous independent campaign')
 parent=read(PARENT/'design_before_valuation.json');manifest=read(PARENT/'manifest.json');bench=read(PREFLIGHT/'manifest.json');seal=read(PARENT/'weights_before_final.json')
 if not manifest['complete'] or not bench['complete'] or not bench['all_checks_passed'] or bench['source_sha256']!=source_hashes():raise ValueError('Complete unchanged spent-equivalence and parent evidence required')
 if seal['selected_specification']!={'family':'gated_anchor','ridge':.01,'eligible_for_selection':True} or len(seal['weights'])!=193:raise ValueError('Frozen193 coefficients changed')
 forbidden=set(parent['forbidden_prior_seeds'])|{s['seed'] for s in parent['streams']}
 b,_,_,_,_,_,_=controls();forbidden|=set(b['manifest']['all_selection_training_seeds'])
 if len({s['seed'] for s in STREAMS})!=3 or forbidden&{s['seed'] for s in STREAMS}:raise ValueError('Independent seeds reused')
 import shutil
 needed=bench['estimated_total_final_compressed_bytes']+bench['estimated_total_final_scratch_raw_bytes']+2*1024**3
 if shutil.disk_usage(OUTPUT).free<needed:raise OSError('Insufficient disk for declared memmaps, compressed archives and2GiB reserve')
 FOLDER.mkdir(parents=True);sources=snapshot(FOLDER);pins=dict(parent['input_sha256'])
 for p in [PARENT/n for n in ('design_before_valuation.json','manifest.json','weights_before_final.json','preservation_inventory.json')]+[PREFLIGHT/n for n in ('design_before_evaluation.json','manifest.json','equivalence_checks.json','preservation_inventory.json','preliminary_trials.json')]+[PREFLIGHT_V1/n for n in ('manifest.json','preservation_inventory.json')]+[PREFLIGHT_V2/n for n in ('manifest.json','preservation_inventory.json')]:pins[str(p.relative_to(OUTPUT))]=sha(p)
 (FOLDER/'weights_before_final.json').write_bytes((PARENT/'weights_before_final.json').read_bytes())
 design={**{k:v for k,v in parent.items() if k not in ('created_at','source_sha256','input_sha256','streams','total_paths','forbidden_prior_seeds')},
  'version':'rough_mc_precision_v1','created_at':now(),'source_sha256':sources,'input_sha256':pins,'streams':STREAMS,'total_paths':196608,
  'parent_campaign':'rough_price_budget_v1/shared_control_v1','parent_design_sha256':sha(PARENT/'design_before_valuation.json'),
  'coefficient_sha256':sha(FOLDER/'weights_before_final.json'),'preflight_folder':str(PREFLIGHT.relative_to(OUTPUT)),'preflight_manifest_sha256':sha(PREFLIGHT/'manifest.json'),'work_batch_size':64,'physical_random_batch_size':512,
  'selected_estimator_only':True,'pure_state_workers':2,'pure_partition_rule':'contiguous4096chunks','pure_chunk_paths':4096,'file_backed_workspace':True,'no_random_or_fitting_in_children':True,'old_paths_used_for_acceptance':False,'forbidden_prior_seeds':sorted(forbidden),
  'fixed_budget_replicas':{'replicates':4,'paths_per_seed_per_replicate':16384,'paths_per_replicate':49152,
   'index_rule':'replicate k uses [k*16384:(k+1)*16384] from each fresh seed','all_replicates_retained':True,'acceptance_or_selection_role':False},
  'acceptance_uses_all_streams':True,'fixed_paths_before_fresh_evaluation':True,'preflight_source_sha256':bench['source_sha256'],
  'no_fresh_selection_or_fitting':True,'no_empirical_valuation_recentering':True,'no_regression_intercept_at_valuation':True,
  'all_paths_retained':True,'file_backed_physical_and_conditional':True,'selected_memmaps_written_per_chunk':True,'pure_result_chunks_buffered_max':2,'scratch_removed_only_after_archives_hashed':True,'estimated_peak_raw_plus_compressed_bytes':needed-2*1024**3,'disk_reserve_bytes':2*1024**3,'tolerances_changed':False,**TOLERANCES}
 write(FOLDER/'design_before_valuation.json',design)
 write(FOLDER/'execution_environment.json',{'interpreter':sys.executable,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
  'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','MPLBACKEND')}})
 print('PREPARED',now(),len(sources),flush=True)

def verified():
 d=read(FOLDER/'design_before_valuation.json')
 if source_hashes()!=d['source_sha256'] or any(sha(FOLDER/'source'/n)!=h for n,h in d['source_sha256'].items()):raise ValueError('Frozen numerical sources changed')
 if any(sha(OUTPUT/n)!=h for n,h in d['input_sha256'].items()):raise ValueError('Frozen input changed')
 if sha(FOLDER/'weights_before_final.json')!=d['coefficient_sha256']:raise ValueError('Frozen193coefficients changed')
 return d

def final(index):
 d=verified();spec=d['streams'][index]
 if any(FOLDER.glob(f'stream_{index}_*')):raise FileExistsError('Preserve previous stream attempt')
 start=perf_counter();cpu=process_time();started=now();print('FINAL',index,spec,started,flush=True)
 scratch=FOLDER.parent/f'.independent_v1_stream_{index}_scratch'
 if scratch.exists():raise FileExistsError('Preserve previous scratch attempt')
 scratch.mkdir()
 a,q,diagnostics=generate(spec['seed'],spec['paths'],forbidden=tuple(d['forbidden_prior_seeds']),workspace=scratch)
 print('ARCHIVING',index,now(),flush=True)
 np.savez_compressed(FOLDER/f'stream_{index}_samples.npz',**a);np.savez_compressed(FOLDER/f'stream_{index}_conditional_samples.npz',**q)
 files={'selected':sha(FOLDER/f'stream_{index}_samples.npz'),'conditional':sha(FOLDER/f'stream_{index}_conditional_samples.npz')}
 schema={name:{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()} for name,arrays in [('selected',a),('conditional',q)]}
 write(FOLDER/f'stream_{index}_manifest.json',{'index':index,**spec,'started_at':started,'completed_at':now(),'elapsed_seconds':perf_counter()-start,'cpu_seconds_parent_only':process_time()-cpu,'cpu_seconds':process_time()-cpu+sum(c['cpu_seconds'] for c in diagnostics['chunks']),
  'design_sha256':sha(FOLDER/'design_before_valuation.json'),'coefficient_sha256':sha(FOLDER/'weights_before_final.json'),'source_sha256':d['source_sha256'],
  'sources_unchanged':True,'sample_sha256':files,'array_schema':schema,'diagnostics':diagnostics,'all_paths_retained':True,
  'frozen_policy':'GH16 blend1','frozen_dual_control':'target 4 bins penalty0','new_fitting_or_selection':False,'gap_pointwise_unchanged':True,'empirical_valuation_recentering':False,'scratch_workspace':str(scratch.relative_to(OUTPUT)),'scratch_cleanup_after_verified_archives':True})
 import shutil
 shutil.rmtree(scratch)
 verified();print('FINAL COMPLETE',index,now(),flush=True)

def collect():
 d=verified();records=[]
 if (FOLDER/'manifest.json').exists():raise FileExistsError('Preserve completed campaign')
 for i,spec in enumerate(d['streams']):
  r=read(FOLDER/f'stream_{i}_manifest.json')
  if (r['index']!=i or any(r[k]!=v for k,v in spec.items()) or r['source_sha256']!=d['source_sha256'] or not r['sources_unchanged']
   or r['design_sha256']!=sha(FOLDER/'design_before_valuation.json') or r['coefficient_sha256']!=d['coefficient_sha256']
   or not d['created_at']<r['started_at']<r['completed_at'] or not r['all_paths_retained']):raise ValueError('Stream identity or chronology failed')
  for name,suffix in [('selected',''),('conditional','_conditional')]:
   if r['sample_sha256'][name]!=sha(FOLDER/f'stream_{i}{suffix}_samples.npz'):raise ValueError('Saved stream changed')
  records.append(r)
 write(FOLDER/'manifest.json',{**d,'complete':True,'completed_at':now(),'records':records,'completed_paths':196608,'all_streams_retained':True,'design_sha256':sha(FOLDER/'design_before_valuation.json')})
 inventory(FOLDER);print('COLLECTED',now(),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('preflight','prepare','final','collect'));parser.add_argument('--index',type=int,choices=range(3));args=parser.parse_args()
 if args.stage=='final':
  if args.index is None:parser.error('--index required')
  final(args.index)
 else:globals()[args.stage]()
