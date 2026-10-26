"""Independent bump/time checks of the frozen shared rough-Heston estimator."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
import argparse, hashlib, json, os, platform, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import numpy as np
import rough_mc_precision as previous
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_shared_delta import shared_delta_samples, RELATIVE_SPOTS, PROCESS_NAMES
OUTPUT=previous.OUTPUT
PARENT=previous.FOLDER
FOLDER=OUTPUT/'rough_shared_delta_v1/independent_v1'
PREFLIGHT=OUTPUT/'rough_shared_delta_v1/spent_v1'
TOLERANCES=previous.TOLERANCES
STREAMS=[{'seed':seed,'paths':4096,'steps':steps,'representations':['original','micro'] if steps==128 else ['micro']}
 for seed,steps in ((1132005042,128),(1133005042,128),(1134005042,256),(1135005042,256))]
SPENT=[{'seed':1094005042,'paths':4096,'steps':128,'reference':'old price control'},
 {'seed':1089005042,'paths':2048,'steps':128,'reference':'old five-stock valuer'},
 {'seed':1091005042,'paths':2048,'steps':256,'reference':'old five-stock valuer'}]
ANALYSIS_NAMES=['src/rvlab/experiments/_pricing_rough_shared_delta_statistics.py','src/rvlab/experiments/_pricing_rough_shared_delta_samples.py']

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(_jsonable(value),indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def source_paths():
 names=list(read(PARENT/'design_before_valuation.json')['source_sha256'])
 names+=['src/rvlab/experiments/_pricing_rough_shared_delta.py','examples/rough_shared_delta.py']+ANALYSIS_NAMES
 return [ROOT/n for n in names]
def hashes():return {str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
def snapshot(folder):
 result=hashes()
 for p in source_paths():
  q=folder/'source'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
 return result

def inventory(folder):
 files=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)}
  for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='preservation_inventory.json']
 write(folder/'preservation_inventory.json',{'files':files,'inventory_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()})

def controls():return previous.controls()
def generate(spec,*,indices=None,forbidden=(),quiet=False):
 b,p,g,t,c,m,seal=controls()
 def progress(phase,done,total):
  if not quiet and (phase in ('physical_increment','completed_pure_paths')):
   print('PROGRESS',spec['seed'],phase,done,total,now(),flush=True)
 a,audit,d=shared_delta_samples(b,p,g,t,c,m,np.asarray(seal['weights']),seed=spec['seed'],paths=spec['paths'],steps=spec['steps'],
  indices=indices,forbidden_seeds=forbidden,progress=progress,work_batch=32)
 if d['basis_names']!=seal['basis_columns']:raise ValueError('Frozen193-column identity changed')
 return a,audit,d

def _compare(checks,stream,name,value,reference,atol=2e-13):
 if value.shape!=reference.shape:raise ValueError(f'Spent field shape differs: {name}')
 mask=np.array_equal(np.isnan(value),np.isnan(reference)) if value.dtype.kind=='f' else True
 error=float(np.nanmax(abs(value-reference))) if value.dtype.kind!='b' else float(np.any(value!=reference))
 checks.append({'stream':stream,'field':name,'max_abs_error':error,'same_nan_mask':mask,'allowance':atol,'passed':mask and error<=atol})

def preflight():
 if PREFLIGHT.exists():raise FileExistsError('Preserve the previous spent trial')
 PREFLIGHT.mkdir(parents=True);sources=snapshot(PREFLIGHT)
 # Preserve every exploratory cost trial, including its exact historical helper.
 trial=PREFLIGHT/'trials';trial.mkdir()
 for name in ('rough_shared_delta_fourstep_benchmark.py','rough_shared_delta_fourstep_benchmark.json','rough_shared_delta_fourstep_benchmark.log',
  'rough_shared_delta_spent_probe.py','rough_shared_delta_spent_probe.json','rough_shared_delta_spent_probe.log'):
  p=Path('/private/tmp')/name
  if p.exists():shutil.copyfile(p,trial/name)
 oldprice=previous.old.FOLDER
 olddelta=OUTPUT/'rough_bellman_validation_v1/frozen_rule_delta_v1'
 refs=[oldprice/'final_0_samples.npz',previous.FIT/'spent_0_basis.npz',
  olddelta/'stream_0_samples.npz',olddelta/'stream_2_samples.npz']
 design={'version':'rough_shared_delta_spent_v1','created_at':now(),'source_sha256':sources,
  'input_sha256':{str(p.relative_to(OUTPUT)):sha(p) for p in refs},'streams':SPENT,
  'path_index_rule':'0..250 plus511,512,1023,last_full_stream_index','evaluated_paths_each':255,
  'fresh_paths':0,'role':'spent equivalence, runtime and budget planning only','equivalence_atol':2e-13,
  'physical_random_batch_size':512,'work_batch_size':32,'all_full_stream_random_draws_consumed':True,
  'original_and_micro_are_distinct_centered_controls':True,'no_fresh_selection_or_fitting':True,
  'analysis_sources_may_receive_guard_tests_before_formal_freeze':ANALYSIS_NAMES}
 write(PREFLIGHT/'design_before_evaluation.json',design)
 records=[];checks=[];descriptive=[]
 for i,spec in enumerate(SPENT):
  indices=np.r_[np.arange(251),511,512,1023,spec['paths']-1]
  start=perf_counter();cpu=process_time();started=now();a,audit,d=generate(spec,indices=indices)
  elapsed=perf_counter()-start
  np.savez_compressed(PREFLIGHT/f'stream_{i}_samples.npz',**a)
  write(PREFLIGHT/f'stream_{i}_primitive_audit.json',audit)
  if i==0:
   with np.load(oldprice/'final_0_samples.npz') as z:
    for key,rkey in [('stocks','stocks'),('factors','factors'),('payoff','payoff'),('calendar_european','european_proxy'),
     ('gh16_stops','gh16_stops'),('original_latest_martingale','latest_martingale'),('original_unchanged_lower','candidate3_lower')]:
     value=a[key];value=value[:,:,2] if value.ndim==3 and key!='factors' else (value[:,2] if value.ndim==2 else value)
     ref=z[rkey][indices];ref=ref[:,:,0] if ref.ndim==3 and rkey!='factors' else ref
     _compare(checks,i,key,value,ref)
   with np.load(previous.FIT/'spent_0_basis.npz') as z:
    _compare(checks,i,'original_basis',a['original_basis'][:,2],z['gated_anchor'][indices])
    _compare(checks,i,'original_shared_lower',a['original_shared_lower'][:,2],z['lower'][indices]-z['gated_anchor'][indices]@np.asarray(controls()[-1]['weights']))
  else:
   with np.load(olddelta/f'stream_{0 if i==1 else 2}_samples.npz') as z:
    for key,rkey in [('stocks','stocks'),('factors','factors'),('payoff','payoff'),('calendar_european','baseline_european_proxy'),
      ('gh16_stops','bellman_stops'),('gh16_continuation','bellman_continuation'),('gh16_computed_mask','bellman_computed_mask')]:
     _compare(checks,i,key,a[key],z[rkey][indices])
  for rep in d['representations']:
   for est in ('raw','stable','unchanged_lower','shared_lower'):
    y=a[rep+'_'+est]
    deltas=np.column_stack(((y[:,4]-y[:,0])/(2*100*.005),(y[:,3]-y[:,1])/(2*100*.0025)))
    for k,bump in enumerate((.005,.0025)):
     descriptive.append({'stream':i,'seed':spec['seed'],'steps':spec['steps'],'representation':rep,'estimator':est,
      'bump_relative':bump,'paths':len(y),'mean':float(deltas[:,k].mean()),'standard_deviation':float(deltas[:,k].std(ddof=1)),
      'projected_sampling_halfwidth_N4096':float(1.96*deltas[:,k].std(ddof=1)/np.sqrt(4096)),
      'planning_only':True,'acceptance_role':False})
  records.append({'index':i,**spec,'started_at':started,'completed_at':now(),'elapsed_seconds':elapsed,
   'cpu_seconds':process_time()-cpu,'sample_sha256':sha(PREFLIGHT/f'stream_{i}_samples.npz'),
   'evaluated_paths':len(indices),'diagnostics':d,'projected_full4096_seconds':elapsed*4096/255,
   'sample_compressed_bytes':(PREFLIGHT/f'stream_{i}_samples.npz').stat().st_size})
  print('SPENT COMPLETE',i,elapsed,flush=True)
 write(PREFLIGHT/'equivalence_checks.json',checks);write(PREFLIGHT/'planning_diagnostics.json',descriptive)
 numeric_unchanged=all(hashes()[k]==v for k,v in sources.items() if k not in ANALYSIS_NAMES)
 write(PREFLIGHT/'manifest.json',{**design,'complete':True,'completed_at':now(),'records':records,
  'all_checks_passed':all(c['passed'] for c in checks),'numerical_sources_unchanged':numeric_unchanged,
  'checks_sha256':sha(PREFLIGHT/'equivalence_checks.json'),'planning_sha256':sha(PREFLIGHT/'planning_diagnostics.json')})
 inventory(PREFLIGHT)
 if not all(c['passed'] for c in checks) or not numeric_unchanged:raise ValueError('Spent equivalence failed; retain trial')
 print('SPENT PASSED',now(),flush=True)

def prepare():
 if FOLDER.exists():raise FileExistsError('Preserve the previous independent campaign')
 bench=read(PREFLIGHT/'manifest.json');parent=read(PARENT/'manifest.json')
 if not bench['complete'] or not bench['all_checks_passed'] or not parent['complete']:raise ValueError('Complete parent and spent evidence required')
 sources=hashes()
 if any(sources[k]!=v for k,v in bench['source_sha256'].items() if k not in ANALYSIS_NAMES):raise ValueError('Benchmarked numerical sources changed')
 forbidden=set(parent['forbidden_prior_seeds'])|{s['seed'] for s in parent['streams']}
 b,*_=controls();forbidden|=set(b['manifest']['all_selection_training_seeds'])
 if len({s['seed'] for s in STREAMS})!=len(STREAMS) or forbidden&{s['seed'] for s in STREAMS}:raise ValueError('Independent seeds reused')
 FOLDER.mkdir(parents=True);sources=snapshot(FOLDER)
 shutil.copyfile(PARENT/'weights_before_final.json',FOLDER/'weights_before_final.json')
 pins=dict(parent['input_sha256'])
 for p in [PARENT/n for n in ('manifest.json','design_before_valuation.json','weights_before_final.json','preservation_inventory.json')]+[PREFLIGHT/n for n in ('design_before_evaluation.json','manifest.json','equivalence_checks.json','planning_diagnostics.json','preservation_inventory.json')]:pins[str(p.relative_to(OUTPUT))]=sha(p)
 design={'version':'rough_shared_delta_v1','created_at':now(),'source_sha256':sources,'input_sha256':pins,
  'parent_campaign':'rough_mc_precision_v1/independent_v1','parent_manifest_sha256':sha(PARENT/'manifest.json'),
  'parent_design_sha256':sha(PARENT/'design_before_valuation.json'),'coefficient_sha256':sha(FOLDER/'weights_before_final.json'),
  'preflight_folder':str(PREFLIGHT.relative_to(OUTPUT)),'preflight_manifest_sha256':sha(PREFLIGHT/'manifest.json'),
  'streams':STREAMS,'total_paths':sum(s['paths'] for s in STREAMS),'frozen_policy':'GH16 blend1',
  'policy_operator_steps':128,'exercise_intervals':64,'kernel_factors':4,'common_control_features':193,
  'bump_relatives':[.005,.0025],'relative_spots':RELATIVE_SPOTS,'no_fresh_selection_or_fitting':True,
  'no_empirical_valuation_recentering':True,'no_regression_intercept_at_valuation':True,
  'all_bumps_recompute_continuations_and_stops':True,'all_paths_retained':True,'prior_paths_used_for_acceptance':False,
  'forbidden_prior_seeds':sorted(forbidden),'physical_random_batch_size':512,'work_batch_size':32,
  'process_names':PROCESS_NAMES,'calendar_european_function_grid_points':129,
  'physical_european_function_grid_points_by_steps':{'128':129,'256':257},
  'original_and_micro_are_distinct_centered_controls':True,'microstep_control_not_a_telescoping_block_bridge':True,
  'microstep_right_function':'fixed next calendar date original129 European/local65/grid65/target129 state functions',
  'microstep_gates':'fixed calendar LEFT state; all increments and own stopping evaluated separately for each bumped stock',
  'independent_primitive_panel':{'absolute_path_indices':[0,1,511,512],'calendar_indices':[16,32,48,63],'bump_indices':[0,2,4]},
  'optimal_american_delta_ready':False,'new_hedging_readiness':False,'tolerances_changed':False,**TOLERANCES}
 write(FOLDER/'design_before_valuation.json',design)
 write(FOLDER/'execution_environment.json',{'interpreter':sys.executable,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
  'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','MPLBACKEND')}})
 print('PREPARED',now(),len(sources),flush=True)

def verified():
 d=read(FOLDER/'design_before_valuation.json')
 if hashes()!=d['source_sha256'] or any(sha(FOLDER/'source'/n)!=h for n,h in d['source_sha256'].items()):raise ValueError('Frozen source changed')
 if any(sha(OUTPUT/n)!=h for n,h in d['input_sha256'].items()):raise ValueError('Frozen input changed')
 if sha(FOLDER/'weights_before_final.json')!=d['coefficient_sha256']:raise ValueError('Frozen193 coefficients changed')
 return d

def final(index):
 d=verified();spec=d['streams'][index]
 if any(FOLDER.glob(f'stream_{index}_*')):raise FileExistsError('Preserve previous stream attempt')
 start=perf_counter();cpu=process_time();started=now();print('FINAL',index,spec,started,flush=True)
 arrays,audit,diagnostics=generate(spec,forbidden=tuple(d['forbidden_prior_seeds']))
 np.savez_compressed(FOLDER/f'stream_{index}_samples.npz',**arrays)
 write(FOLDER/f'stream_{index}_primitive_audit.json',audit)
 write(FOLDER/f'stream_{index}_manifest.json',{'index':index,**spec,'started_at':started,'completed_at':now(),
  'elapsed_seconds':perf_counter()-start,'cpu_seconds':process_time()-cpu,'design_sha256':sha(FOLDER/'design_before_valuation.json'),
  'coefficient_sha256':d['coefficient_sha256'],'source_sha256':d['source_sha256'],'sources_unchanged':True,
  'sample_sha256':sha(FOLDER/f'stream_{index}_samples.npz'),'array_schema':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
  'all_paths_retained':True,'new_fitting_or_selection':False,'diagnostics':diagnostics})
 verified();print('FINAL COMPLETE',index,now(),flush=True)

def collect():
 d=verified();records=[]
 if (FOLDER/'manifest.json').exists():raise FileExistsError('Preserve completed campaign')
 for i,spec in enumerate(d['streams']):
  r=read(FOLDER/f'stream_{i}_manifest.json')
  if (r['index']!=i or any(r[k]!=v for k,v in spec.items()) or r['source_sha256']!=d['source_sha256']
   or not r['sources_unchanged'] or not r['all_paths_retained'] or r['new_fitting_or_selection']
   or r['design_sha256']!=sha(FOLDER/'design_before_valuation.json')
   or r['coefficient_sha256']!=d['coefficient_sha256'] or not d['created_at']<r['started_at']<r['completed_at']):raise ValueError('Final stream identity failed')
  if r['sample_sha256']!=sha(FOLDER/f'stream_{i}_samples.npz'):raise ValueError('Final stream archive changed')
  records.append(r)
 write(FOLDER/'manifest.json',{**d,'complete':True,'completed_at':now(),'records':records,
  'design_sha256':sha(FOLDER/'design_before_valuation.json'),'all_streams_retained':True,
  'completed_paths':sum(r['paths'] for r in records)})
 inventory(FOLDER);print('COLLECTED',now(),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','prepare','final','collect'));p.add_argument('--index',type=int,choices=range(4));a=p.parse_args()
 if a.stage=='final':
  if a.index is None:p.error('--index required')
  final(a.index)
 else:globals()[a.stage]()
