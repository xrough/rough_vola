"""Sealed selection and independent valuation of one-step Bellman challenges."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time,os,platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import rough_tail_attribution as tail
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_local_control import correction_basis
from rvlab.experiments._pricing_rough_bellman_policy import (
    BellmanEvaluator,bellman_candidate_specs,evaluate_bellman_candidates,select_bellman_policy,ORDERS)

OUTPUT=tail.OUTPUT;BASE=OUTPUT/'rough_bellman_policy_v1';FOLDER=BASE/'one_step_v1'
TAIL_FOLDER=OUTPUT/'rough_tail_attribution_v1/joint_switch_v1'
SELECTION_SEED=1066005042;FINAL_SEEDS=(1077005042,1078005042)
RESERVED=tuple(tail.BASELINE_RESERVED)+(1043005042,1054005042,1055005042)
AUDIT=BASE/'independent_adaptive_audit_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(_jsonable(obj),indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def stat(chunks):
    n=sum(map(len,chunks));mean=float(np.concatenate(chunks).mean());se=float(np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in chunks))/n)
    return mean,se,mean-1.96*se,mean+1.96*se
def sources():
    return list(dict.fromkeys(tail.source_paths()+[Path(__file__),ROOT/'src/rvlab/experiments/_pricing_rough_bellman_policy.py']))
def verified():
    d=read(FOLDER/'design_before_training.json')
    if any(sha(ROOT/n)!=h for n,h in d['source_sha256'].items()):raise ValueError('Frozen numerical source changed')
    if any(sha(OUTPUT/n)!=h for n,h in d['input_sha256'].items()):raise ValueError('Frozen prior input changed')
    return d
def latest_control():
    seal=read(TAIL_FOLDER/'selection_before_final.json')
    if seal['selected_policy']!=tail.BASELINE_NAME:raise ValueError('The baseline exercise policy changed since the Bellman design')
    return seal,seal['controls'][seal['selected_control']]
def benchmark():
    folder=BASE/'runtime_probe_v1'
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);frozen,_,_,_=tail.inputs();e=BellmanEvaluator(frozen['proxy'],frozen['selected_policy']);rows=[];counts=[]
    for i,seed in enumerate((1054005042,1055005042)):
        path=TAIL_FOLDER/f'final_{i}_samples.npz'
        with np.load(path,allow_pickle=False) as z:
            counts.append({'seed':seed,'paths':len(z['payoff']),'itm_decision_states':int(np.sum(z['payoff'][:,1:-1]>0))})
            for j in (16,32,48,62):
                ids=np.flatnonzero(z['payoff'][:,j]>0)[:32];start=time.perf_counter()
                c=e.conditional_expectations(j,z['stocks'][ids,j,0],z['factors'][ids,j]);elapsed=time.perf_counter()-start
                rows.append({'spent_seed':seed,'exercise_index':j,'states':len(ids),'elapsed_seconds':elapsed,
                    'seconds_per_state':elapsed/len(ids),'max_GH8_GH32_change':float(np.max(abs(c['continuation'][:,0]-c['continuation'][:,2]))),
                    'max_GH16_GH32_change':float(np.max(abs(c['continuation'][:,1]-c['continuation'][:,2]))),
                    'max_CV_direct_GH32_change':float(np.max(abs(c['continuation'][:,2]-c['direct_continuation'][:,2]))),'new_acceptance_evidence':False})
    pd.DataFrame(rows).to_csv(folder/'timings.csv',index=False);pd.DataFrame(counts).to_csv(folder/'state_counts.csv',index=False)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources()}
    for p in sources():
        dst=folder/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    write(folder/'manifest.json',{'version':'rough_bellman_runtime_probe_v1','completed_at':now(),'source_sha256':hashes,
        'no_new_paths_or_fitting':True,'new_acceptance_evidence':False,'orders':ORDERS,
        'states_source':'first32 ITM states per declared seed/date; spent selected-tail final paths',
        'projected_4096_lookahead_seconds':float(np.mean([r['seconds_per_state'] for r in rows])*np.mean([c['itm_decision_states'] for c in counts]))})
    inventory(folder);print(pd.DataFrame(rows).to_string(index=False),flush=True)
def inventory(folder):
    entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file()]
    write(folder/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
def prepare():
    if FOLDER.exists():raise FileExistsError(FOLDER)
    audit=read(AUDIT/'manifest.json')
    if audit.get('complete') is not True:raise ValueError('Independent adaptive audit is incomplete')
    frozen,_,_,pins=tail.inputs();old,control=latest_control()
    for p in (TAIL_FOLDER/'manifest.json',TAIL_FOLDER/'selection_before_final.json',AUDIT/'manifest.json',AUDIT/'preservation_inventory.json',BASE/'runtime_probe_v1/manifest.json',BASE/'runtime_probe_v1/preservation_inventory.json'):
        pins[str(p.relative_to(OUTPUT))]=sha(p)
    forbidden=sorted(set(RESERVED)|set(frozen['manifest']['all_selection_training_seeds']))
    if set((SELECTION_SEED,)+FINAL_SEEDS)&set(forbidden):raise ValueError('Bellman streams reuse prior fitting or valuation')
    FOLDER.mkdir(parents=True);hashes={str(p.relative_to(ROOT)):sha(p) for p in sources()}
    for p in sources():
        target=FOLDER/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    design={'version':'rough_bellman_one_step_v1','created_at':now(),'source_sha256':hashes,'input_sha256':pins,
        'training':{'seed':SELECTION_SEED,'paths':2048,'role':'selection only'},'final_streams':[{'seed':s,'paths':4096} for s in FINAL_SEEDS],
        'candidate_specs':bellman_candidate_specs(),'orders':ORDERS,'baseline_policy_name':tail.BASELINE_NAME,
        'frozen_regression_policy_name':frozen['selected_name'],'common_control_name':old['selected_control'],'common_control':control,
        'forbidden_prior_seeds':forbidden,'physical_steps':128,'exercise_intervals':64,'batch_size':32,
        'bellman_envelope':'max(payoff,Euro+clip(q,0,.7)); cap only q','integration':'analytic Euro/q expectation plus GH positive exercise excess',
        'computed_mask':'all current ITM states on exercise indices1 through63; irrelevant entries explicitNaN; all outer paths retained',
        'terminal_interval':'exact Gaussian conditional payoff','full_and_half_blends':True,
        'selection_rule':'maximum positive paired lower-gain95percentLCB; fallback0; independently seal highest nonbaselineLCB challenger',
        'all_candidates_reported':True,'new_model_or_control_fitting':False,'all_paths_retained':True,
        'price_allowance':'.001+1e-4*abs(V)','delta_allowance':'.002+.005*abs(delta)','confidence_multiplier':1.96,
        'tolerances_changed':False,'american_pricing_ready':False,'new_rule_stock_delta_validated':False,
        'no_new_paths_generated_before_design':True}
    write(FOLDER/'design_before_training.json',design)
    write(FOLDER/'execution_environment.json',{'interpreter':sys.executable,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}})
    print('PREPARED',now(),'SOURCES',len(hashes),flush=True)
def generate(seed,paths):
    start=time.perf_counter();frozen,trace,w=tail.generate(seed,paths);control_seal,control=latest_control()
    policies=tail.build_margin_policies(frozen['selected_policy']);_,data=tail.evaluate(w,trace,policies,control_seal['controls'])
    basis=correction_basis(w['tail_corrections'],w['times'],control['bins'])
    common=w['current_martingale']+np.einsum('ntd,d->nt',basis,np.asarray(control['weights']))
    base_elapsed=time.perf_counter()-start;look_start=time.perf_counter()
    evaluator=BellmanEvaluator(frozen['proxy'],frozen['selected_policy'])
    forbidden=RESERVED+((SELECTION_SEED,) if seed in FINAL_SEEDS else ())
    look=evaluator.trace_continuations(trace,forbidden_seeds=forbidden)
    result=evaluate_bellman_candidates(trace,look,common,is_call=frozen['proxy'].is_call)
    index=list(control_seal['controls']).index(control_seal['selected_control'])
    if (not np.array_equal(result['new_stops'][:,0],data['stops'][:,0]) or not np.array_equal(result['new_raw'][:,0],data['raw'][:,0])
            or not np.array_equal(result['new_lower'][:,0],data[f'control{index}_lower'][:,0])
            or not np.array_equal(result['common_upper'],data[f'control{index}_upper'])):raise ValueError('Latest selected control or baseline changed')
    data.update({k:look[k] for k in ('bellman_continuations','bellman_direct_continuations','bellman_computed_mask','baseline_continuation')})
    data.update({k:result[k] for k in ('new_raw','new_stops','new_lower','common_upper')})
    diagnostics={'common_control_seconds':base_elapsed,'lookahead_seconds':time.perf_counter()-look_start,
        'bellman_dates':look['diagnostics'],'masked_states':int(look['bellman_computed_mask'].sum()),
        'common_upper_exact':True,'no_future_realized_state':True,'approximate_Snell_not_policy_improvement_guarantee':True}
    return trace,w,data,result,diagnostics
def training():
    d=verified()
    if (FOLDER/'selection_before_final.json').exists() or (FOLDER/'training_samples.npz').exists():raise FileExistsError('Selection attempt already exists')
    start=time.perf_counter();print('SELECTION',SELECTION_SEED,flush=True);trace,w,data,result,diagnostics=generate(SELECTION_SEED,2048)
    selected,challenger,table=select_bellman_policy(result,selection_seed=SELECTION_SEED);table.to_csv(FOLDER/'policy_selection.csv',index=False)
    tail.save_samples(FOLDER/'training_samples.npz',trace,w,data)
    seal={'sealed_at':now(),'selected_policy':selected,'selected_challenger':challenger,'policy_order':result['policy_order'],
        'candidate_specs':bellman_candidate_specs(),'common_control_name':d['common_control_name'],'selection_seed':SELECTION_SEED,
        'design_sha256':sha(FOLDER/'design_before_training.json'),'final_paths_generated_before_seal':False}
    write(FOLDER/'selection_before_final.json',seal)
    write(FOLDER/'training_manifest.json',{'seed':SELECTION_SEED,'paths':2048,'role':'selection only','completed_at':now(),
        'elapsed_seconds':time.perf_counter()-start,'source_sha256':d['source_sha256'],'sources_unchanged':True,
        'sample_sha256':sha(FOLDER/'training_samples.npz'),'selection_sha256':sha(FOLDER/'selection_before_final.json'),
        'design_sha256':sha(FOLDER/'design_before_training.json'),'diagnostics':diagnostics})
    verified();print('SEALED',selected,'CHALLENGER',challenger,now(),flush=True)
def final(index):
    d=verified();seal=read(FOLDER/'selection_before_final.json');spec=d['final_streams'][index]
    if (FOLDER/f'final_{index}_samples.npz').exists():raise FileExistsError('Final stream already exists')
    start=time.perf_counter();print('FINAL',index,spec['seed'],flush=True);trace,w,data,_,diagnostics=generate(spec['seed'],spec['paths'])
    path=FOLDER/f'final_{index}_samples.npz';tail.save_samples(path,trace,w,data)
    write(FOLDER/f'final_{index}_manifest.json',{'index':index,'seed':spec['seed'],'paths':spec['paths'],'completed_at':now(),
        'elapsed_seconds':time.perf_counter()-start,'policy_order':list(bellman_candidate_specs()),'common_control_name':seal['common_control_name'],
        'sample_sha256':sha(path),'selection_sha256':sha(FOLDER/'selection_before_final.json'),'design_sha256':sha(FOLDER/'design_before_training.json'),
        'source_sha256':d['source_sha256'],'sources_unchanged':True,'diagnostics':diagnostics})
    verified();print('COMPLETE',index,now(),flush=True)
def collect():
    d=verified();seal=read(FOLDER/'selection_before_final.json')
    if (FOLDER/'manifest.json').exists():raise FileExistsError('Campaign is already collected')
    streams=[];records=[]
    for i,spec in enumerate(d['final_streams']):
        record=read(FOLDER/f'final_{i}_manifest.json');path=FOLDER/f'final_{i}_samples.npz'
        if record['sample_sha256']!=sha(path) or record['selection_sha256']!=sha(FOLDER/'selection_before_final.json'):raise ValueError('Changed final stream')
        with np.load(path,allow_pickle=False) as z:stream={k:z[k].copy() for k in ('new_raw','new_stops','new_lower','common_upper')}
        stream['seed']=spec['seed'];streams.append(stream);records.append(record)
    comparison=[];paired=[];perseed=[];errors=[];tails=[];exercise=[]
    for p,(name,spec) in enumerate(bellman_candidate_specs().items()):
        flags={'policy':name,**spec,'selected_policy':name==seal['selected_policy'],'selected_challenger':name==seal['selected_challenger']}
        lower=[s['new_lower'][:,p] for s in streams];upper=[s['common_upper'] for s in streams];gaps=[u-l for u,l in zip(upper,lower)]
        price,pse,_,_=stat(lower);u,use,_,_=stat(upper);gap,gse,_,_=stat(gaps);allowance=.001+1e-4*abs(price);budget=gap+1.96*(pse+gse)
        comparison.append({**flags,'paths':8192,'streams':2,'price':price,'price_se':pse,'upper':u,'upper_se':use,'gap':gap,'gap_se':gse,
            'price_error_budget':budget,'price_allowance':allowance,'price_budget_pass':budget<=allowance,'common_upper_identical':True,
            'american_pricing_ready':False,'new_rule_stock_delta_validated':False})
        gains=[s['new_lower'][:,p]-s['new_lower'][:,0] for s in streams];rawg=[s['new_raw'][:,p]-s['new_raw'][:,0] for s in streams]
        row=flags.copy()
        for label,chunks in (('lower_gain',gains),('raw_gain',rawg),('gap_change',[-x for x in gains])):
            mean,se,low,high=stat(chunks);row.update({label:mean,label+'_se':se,label+'_low':low,label+'_high':high})
        row['exercise_change_fraction']=float(np.mean(np.concatenate([s['new_stops'][:,p]!=s['new_stops'][:,0] for s in streams])));paired.append(row)
        gain,se,lo,hi=stat(gains);margin,mse,ml,mh=stat([g-.5*(s['common_upper']-s['new_lower'][:,0]) for s,g in zip(streams,gains)])
        errors.append({**flags,'identified_policy_gain':gain,'gain_standard_error':se,'gain_interval_low':lo,'gain_interval_high':hi,
            'improvement_resolved':lo>0,'majority_margin':margin,'majority_margin_se':mse,'majority_margin_low':ml,'majority_margin_high':mh,
            'policy_majority_proved':ml>0,'confidence_multiplier':1.96,'full_policy_error_identified':False,'same_martingale':True})
        for s,l,g,gainseq,raw in zip(streams,lower,gaps,gains,rawg):
            lm,ls,_,_=stat([l]);gm,gs,_,_=stat([g]);gn,gse,gl,gh=stat([gainseq]);rm,rs,rl,rh=stat([raw])
            perseed.append({**flags,'seed':s['seed'],'paths':len(l),'price':lm,'price_se':ls,'upper':float(s['common_upper'].mean()),'gap':gm,'gap_se':gs,
                'lower_gain':gn,'lower_gain_se':gse,'lower_gain_low':gl,'lower_gain_high':gh,'raw_gain':rm,'raw_gain_se':rs,'raw_gain_low':rl,'raw_gain_high':rh})
            for index,count in zip(*np.unique(s['new_stops'][:,p],return_counts=True)):exercise.append({**flags,'seed':s['seed'],'exercise_index':int(index),'paths':int(count)})
        allg=np.concatenate(gaps);ordered=np.sort(allg);total=allg.sum()
        tails.append({**flags,'paths':len(allg),'zero_gap_fraction':float(np.mean(allg<=1e-12)),'gap_p90':float(np.quantile(allg,.9)),
            'gap_p99':float(np.quantile(allg,.99)),'gap_maximum':float(allg.max()),'largest_decile_gap_share':float(ordered[-820:].sum()/total),
            'largest_percentile_gap_share':float(ordered[-82:].sum()/total),'used_for_selection':False,'acceptance_input':False})
    refinements=[];names=list(bellman_candidate_specs())
    for blend in (.5,1.):
        for coarse,fine in ((8,16),(16,32)):
            c=names.index(f'GH{coarse} blend{blend:g}');f=names.index(f'GH{fine} blend{blend:g}')
            changes=[s['new_lower'][:,f]-s['new_lower'][:,c] for s in streams];mean,se,low,high=stat(changes)
            value=next(r['price'] for r in comparison if r['policy']==names[f]);allowance=.001+1e-4*abs(value)
            refinements.append({'coarse_order':coarse,'fine_order':fine,'blend':blend,'price_change':mean,'price_change_se':se,
                'change_low':low,'change_high':high,'change_plus_uncertainty':abs(mean)+1.96*se,'price_allowance':allowance,
                'finite_policy_price_comparison_pass':abs(mean)+1.96*se<=allowance,'Bellman_quadrature_certified':False})
    tables={'comparison':comparison,'paired_changes':paired,'per_seed':perseed,'policy_error':errors,'tail_distribution':tails,'exercise_times':exercise,'quadrature_policy_refinement':refinements}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(FOLDER/(name+'.csv'),index=False)
    write(FOLDER/'manifest.json',{**d,'completed_at':now(),'complete':True,'selected_policy':seal['selected_policy'],
        'selected_challenger':seal['selected_challenger'],'policy_order':names,'common_control_name':seal['common_control_name'],
        'records':records,'selection_sha256':sha(FOLDER/'selection_before_final.json'),'training_manifest_sha256':sha(FOLDER/'training_manifest.json'),
        'sources_unchanged':True,'completed_final_paths':8192,'table_rows':{n:len(v) for n,v in tables.items()}})
    inventory(FOLDER);print(pd.DataFrame(comparison).to_string(index=False),flush=True);print(pd.DataFrame(errors).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('benchmark','prepare','train','final','collect'));parser.add_argument('--index',type=int,choices=(0,1));args=parser.parse_args()
    if args.stage=='final' and args.index is None:parser.error('final requires --index')
    {'benchmark':benchmark,'prepare':prepare,'train':training,'final':lambda:final(args.index),'collect':collect}[args.stage]()
