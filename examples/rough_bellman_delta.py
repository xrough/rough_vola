"""Sealed finite stock differences of the frozen GH16 full Bellman policy."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time,os,platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import rough_bellman_policy as previous
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_bellman_delta import (
    LEVELS,POLICIES,TOLERANCES,bellman_policy_stops,bellman_delta_samples,summarize_bellman_delta)
from rvlab.experiments._pricing_rough_bellman_policy import BellmanEvaluator
from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint

OUTPUT=previous.OUTPUT;BASE=OUTPUT/'rough_bellman_validation_v1';FOLDER=BASE/'frozen_rule_delta_v1'
PROBE=BASE/'delta_runtime_probe_v1';PRIOR=previous.FOLDER
STREAMS=({'seed':1089005042,'paths':2048,'steps':128},
    {'seed':1090005042,'paths':2048,'steps':128},{'seed':1091005042,'paths':2048,'steps':256})
STATIC=('relative_spots','times','discounts','eligible','actual_time_grid','operator_time_grid',
    'initial_spots','strike','is_call','stable_weights','stable_time_edges','stable_components')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(_jsonable(obj),indent=2,allow_nan=False)+'\n')
def sources():return list(dict.fromkeys(previous.sources()+[Path(__file__),
    ROOT/'src/rvlab/experiments/_pricing_rough_bellman_delta.py',ROOT/'src/rvlab/experiments/_pricing_rough_stock_delta.py']))
def inventory(folder):
    entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(folder.rglob('*')) if p.is_file()]
    write(folder/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(
        json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
def snapshot(folder):
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources()}
    for p in sources():
        dst=folder/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    return hashes
def benchmark():
    if PROBE.exists():raise FileExistsError(PROBE)
    PROBE.mkdir(parents=True);hashes=snapshot(PROBE);bundle,*_=previous.tail.inputs();proxy=bundle['proxy']
    evaluator=BellmanEvaluator(proxy,bundle['selected_policy'],orders=(16,));rows=[];saved={}
    relative=np.array([-.005,-.0025,0.,.0025,.005])
    for i,seed in enumerate((1077005042,1078005042)):
        with np.load(PRIOR/f'final_{i}_samples.npz',allow_pickle=False) as z:
            n=128;trace={'seed':seed,'stocks':z['stocks'][:n]*(1+relative)[None,None,:],
                'factors':z['factors'][:n],'times':z['times'],'discounts':z['discounts'],
                'eligible':z['eligible'],'actual_time_grid':z['actual_time_grid'],
                'initial_spots':proxy.market.spot*(1+relative),'strike':proxy.strike,
                'policy_fingerprint':fourier_policy_fingerprint(proxy),'kernel_fingerprint':proxy.kernel.fingerprint}
            start=time.perf_counter();result=bellman_policy_stops(trace,evaluator,outer_steps=128)
            elapsed=time.perf_counter()-start;matches=np.array_equal(result['stops'][:,2],z['new_stops'][:n,4])
            rows.append({'spent_seed':seed,'paths':n,'spots':5,'elapsed_seconds':elapsed,
                'computed_states':result['diagnostics']['computed_states'],'center_stops_match_frozen':matches,
                'projected2048_lookahead_seconds':elapsed*2048/n,'new_acceptance_evidence':False})
            saved[f'seed{seed}_stops']=result['stops'];saved[f'seed{seed}_computed_mask']=result['computed_mask']
            if not matches:raise ValueError('Single-order alive-state implementation changed the frozen central policy')
    pd.DataFrame(rows).to_csv(PROBE/'timings.csv',index=False);np.savez_compressed(PROBE/'samples.npz',**saved)
    write(PROBE/'manifest.json',{'version':'rough_bellman_delta_runtime_probe_v1','complete':True,
        'completed_at':now(),'source_sha256':hashes,'spent_prior_final_paths':True,
        'no_new_paths_or_fitting':True,'new_acceptance_evidence':False,'paths_per_spent_stream':128,
        'frozen_center_stops_exact':True,'all_bumps_recompute':True})
    inventory(PROBE);print(pd.DataFrame(rows).to_string(index=False),flush=True)
def prepare():
    if FOLDER.exists():raise FileExistsError(FOLDER)
    old=read(PRIOR/'manifest.json');seal=read(PRIOR/'selection_before_final.json');probe=read(PROBE/'manifest.json')
    if not old['complete'] or seal['selected_policy']!='GH16 blend1' or not probe['complete']:
        raise ValueError('Require the completed frozen GH16 policy and bounded runtime probe')
    bundle,*_,pins=previous.tail.inputs()
    for p in (PRIOR/'manifest.json',PRIOR/'selection_before_final.json',PRIOR/'design_before_training.json',
            PROBE/'manifest.json',PROBE/'preservation_inventory.json'):
        pins[str(p.relative_to(OUTPUT))]=sha(p)
    forbidden=sorted(set(old['forbidden_prior_seeds'])|set(bundle['manifest']['all_selection_training_seeds'])|
        {1066005042,1077005042,1078005042,1043005042,1054005042,1055005042})
    seeds=[s['seed'] for s in STREAMS]
    if len(set(seeds))!=3 or set(seeds)&set(forbidden):raise ValueError('Delta streams overlap prior evidence')
    FOLDER.mkdir(parents=True);hashes=snapshot(FOLDER)
    write(FOLDER/'design_before_valuation.json',{'version':'rough_bellman_frozen_delta_v1','created_at':now(),
        'source_sha256':hashes,'input_sha256':pins,'streams':STREAMS,'forbidden_prior_seeds':forbidden,
        'policy_order':POLICIES,'frozen_policy':'GH16 blend1','bump_relatives':LEVELS,
        'relative_spots':[-.005,-.0025,0.,.0025,.005],'operator_steps':128,'exercise_steps':64,
        'operator_fixed_across_outer_time_refinement':True,'common_control':'original audited stable stopped martingale',
        'stable_components':bundle['weights'].config.components,'stable_weights':bundle['weights'].weights,
        'stable_time_edges':bundle['weights'].time_edges,'all_bumps_recompute_continuations_and_stops':True,
        'sampling_and_bump_and_time_gates_separate':True,'no_model_or_control_fitting':True,
        'all_paths_retained':True,'tolerances_changed':False,'optimal_delta_ready':False,
        'no_new_paths_generated_before_design':True,**TOLERANCES})
    write(FOLDER/'execution_environment.json',{'interpreter':sys.executable,'python':sys.version,
        'numpy':np.__version__,'platform':platform.platform(),
        'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}})
    print('PREPARED',sha(FOLDER/'design_before_valuation.json'),'SOURCES',len(hashes),now(),flush=True)
def verified():
    d=read(FOLDER/'design_before_valuation.json')
    if any(sha(ROOT/n)!=h for n,h in d['source_sha256'].items()):raise ValueError('Frozen delta numerical source changed')
    if any(sha(OUTPUT/n)!=h for n,h in d['input_sha256'].items()):raise ValueError('Frozen delta input changed')
    return d
def stream(index):
    d=verified();spec=d['streams'][index];path=FOLDER/f'stream_{index}_samples.npz'
    if path.exists() or (FOLDER/f'stream_{index}_manifest.json').exists():raise FileExistsError('Stream already attempted')
    start=time.perf_counter();print('STREAM',index,spec,now(),flush=True);bundle,*_=previous.tail.inputs()
    arrays,diagnostics=bellman_delta_samples(bundle,**spec,forbidden_seeds=tuple(d['forbidden_prior_seeds']))
    estimates,comparisons=summarize_bellman_delta(arrays)
    np.savez_compressed(path,**arrays);estimates.to_csv(FOLDER/f'stream_{index}_estimates.csv',index=False)
    comparisons.to_csv(FOLDER/f'stream_{index}_comparisons.csv',index=False);verified()
    write(FOLDER/f'stream_{index}_manifest.json',{'index':index,**spec,'completed_at':now(),
        'elapsed_seconds':time.perf_counter()-start,'design_sha256':sha(FOLDER/'design_before_valuation.json'),
        'sample_sha256':sha(path),'source_sha256':d['source_sha256'],'sources_unchanged':True,
        'diagnostics':diagnostics,'array_schema':{k:{'shape':v.shape,'dtype':str(v.dtype)} for k,v in arrays.items()}})
    print('COMPLETE',index,now(),flush=True);print(estimates.to_string(index=False),flush=True)
def collect():
    d=verified()
    if (FOLDER/'manifest.json').exists():raise FileExistsError('Delta campaign already collected')
    archives=[];records=[]
    for index,spec in enumerate(d['streams']):
        p=FOLDER/f'stream_{index}_samples.npz';r=read(FOLDER/f'stream_{index}_manifest.json')
        if r['sample_sha256']!=sha(p) or r['design_sha256']!=sha(FOLDER/'design_before_valuation.json'):
            raise ValueError('Changed delta stream')
        with np.load(p,allow_pickle=False) as z:archives.append({k:z[k].copy() for k in z.files})
        records.append(r)
    keys=[p+'_'+e+'_deltas' for p in ('baseline','bellman') for e in ('raw','controlled')]
    joined={k:np.concatenate([archives[i][k] for i in (0,1)]) for k in keys}
    pooled,paired=summarize_bellman_delta(joined);fine,fp=summarize_bellman_delta(archives[2]);changes=[]
    for row in fine.itertuples():
        b=pooled[(pooled.policy==row.policy)&(pooled.estimator==row.estimator)&(pooled.bump_relative==row.bump_relative)].iloc[0]
        change=row.mean-b['mean'];se=float(np.hypot(row.standard_error,b.standard_error));allowance=.002+.005*abs(row.mean)
        changes.append({'policy':row.policy,'estimator':row.estimator,'bump_relative':row.bump_relative,
            'coarse_steps':128,'fine_steps':256,'mean':change,'standard_error':se,'ci_half_width':1.96*se,
            'interval_low':change-1.96*se,'interval_high':change+1.96*se,'delta_allowance':allowance,
            'change_plus_uncertainty':abs(change)+1.96*se,'comparison_passed':abs(change)+1.96*se<=allowance,
            'independent_streams':True,'frozen_operator_steps':128,'same_exercise_calendar':True,'optimal_delta_ready':False})
    perseed=[]
    for i,spec in enumerate(d['streams']):
        e,_=summarize_bellman_delta(archives[i]);e['seed']=spec['seed'];e['steps']=spec['steps'];perseed.append(e)
    tables={'pooled_estimates':pooled,'pooled_comparisons':paired,'time_estimates':fine,
        'time_comparisons':fp,'independent_time_comparison':pd.DataFrame(changes),'per_seed':pd.concat(perseed,ignore_index=True)}
    for n,t in tables.items():t.to_csv(FOLDER/(n+'.csv'),index=False)
    write(FOLDER/'manifest.json',{**d,'completed_at':now(),'complete':True,'records':records,
        'primary_paths':4096,'time_paths':2048,'sources_unchanged':True,'all_paths_retained':True,
        'new_rule_delta_measured':True,'optimal_delta_ready':False,'table_rows':{n:len(t) for n,t in tables.items()}})
    inventory(FOLDER);print(pooled.to_string(index=False),flush=True);print(pd.DataFrame(changes).to_string(index=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('benchmark','prepare','stream','collect'))
    parser.add_argument('--index',type=int,choices=(0,1,2));args=parser.parse_args()
    if args.stage=='stream' and args.index is None:parser.error('stream requires --index')
    {'benchmark':benchmark,'prepare':prepare,'stream':lambda:stream(args.index),'collect':collect}[args.stage]()
