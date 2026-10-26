"""Staged independent policy/dual challenges of the frozen tail-gap baseline."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,os,platform,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rvlab.experiments.pricing import _jsonable
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_tail_switch_guarded import FEATURES,FAMILIES,current_control_workspace,tail_switch_corrections
from rvlab.experiments._pricing_rough_tail_policy import BASELINE_NAME,margin_policy_specs,build_margin_policies,evaluate_policy_candidates,select_margin_policy
from rvlab.experiments._pricing_rough_local_control import correction_basis,fit_correction_weights
from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy

OUTPUT=ROOT/'notebooks/_output/pricing/research_v1/01_pricing_and_exercise'
FOLDER=OUTPUT/'rough_tail_attribution_v1/joint_switch_v1'
TRAIN_SEED=1043005042
FINAL_SEEDS=(1054005042,1055005042)
BASE_CONTROL='current baseline'
BASELINE_RESERVED=(320005042,350005042,610005042,620005042,630005042,631005042,
    842005042,852005042,853005042,942005042,943005042,954005042,955005042)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,obj):p.write_text(json.dumps(_jsonable(obj),indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())
def record(p):return {'path':str(p.relative_to(FOLDER)),'bytes':p.stat().st_size,'sha256':sha(p)}
def stat(seq):
    n=sum(map(len,seq));return float(np.concatenate(seq).mean()),float(np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in seq))/n)
def inputs():
    model=OUTPUT/'rough_larger_policy_v1/32768_paths_two_starts'
    prior=OUTPUT/'rough_policy_validation_v1/local_control_v1/controls_before_final.json'
    grid=OUTPUT/'rough_gap_time_refinement_v1/stock_grid_control_v2/controls_before_final.json'
    frozen=load_larger_policy_bundle(model);p=read(prior);g=read(grid)
    pins={str(x.relative_to(OUTPUT)):sha(x) for x in (model/'frozen_policy_before_final.json',model/'frozen_policy_coefficients.npz',prior,grid)}
    return frozen,p['controls'][p['selected']],g['controls'][g['selected']],pins
def source_paths():
    names=('_pricing_rough_tail_switch.py','_pricing_rough_tail_switch_guarded.py','_pricing_rough_tail_policy.py','_pricing_rough_local_control.py',
        '_pricing_rough_node_refinement_fast.py','_pricing_rough_exercise_block.py','_pricing_rough_large_policy.py',
        '_pricing_rough_frozen_policy.py','_pricing_rough_policy_improvement.py')
    return [Path(__file__)]+[ROOT/'src/rvlab/experiments'/n for n in names]+sorted((ROOT/'src/rvlab/pricing').glob('*.py'))
def verified():
    d=read(FOLDER/'design_before_training.json')
    if any(sha(ROOT/n)!=h for n,h in d['source_sha256'].items()):raise ValueError('Frozen numerical source changed')
    if any(sha(OUTPUT/n)!=h for n,h in d['input_sha256'].items()):raise ValueError('Frozen baseline input changed')
    return d
def prepare():
    if FOLDER.exists():raise FileExistsError(FOLDER)
    frozen,prior,grid,pins=inputs()
    forbidden=set(BASELINE_RESERVED)|set(frozen['manifest']['all_selection_training_seeds'])
    if {TRAIN_SEED,*FINAL_SEEDS}&forbidden:raise ValueError('Fresh declared seeds overlap baseline fitting/valuation')
    FOLDER.mkdir(parents=True)
    sources=source_paths();hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        target=FOLDER/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    design={'version':'rough_tail_attribution_joint_switch_v1','created_at':now(),'source_sha256':hashes,'input_sha256':pins,
        'training':{'seed':TRAIN_SEED,'paths':4096,'fit_paths':2048,'heldout_paths':2048},
        'final_streams':[{'seed':s,'paths':4096} for s in FINAL_SEEDS],'features':FEATURES,'families':FAMILIES,
        'bins':[1,4],'penalties':[0.,1e-4],'nonnegative_coefficient_sum_per_bin_max':1.,
        'extended_weight_scales':[1.,.25,.5],'pilot_motivated_weight_shrinkage':True,
        'shrinkage_declared_before_any_fresh_paths':True,'declared_control_count_if_all_fits_complete':17,
        'selected_policy_name':frozen['selected_name'],'policy_specs':margin_policy_specs(),'prior_control':prior,'grid_control':grid,
        'policy_selection':'max positive paired-lower-gain95percentLCB on heldout half using current commonM',
        'control_selection':'minimum heldoutgap plus projected1.96(gapSE+priceSE) at8192paths using frozenbaselinepolicy',
        'shared_holdout_for_separate_objectives':True,'all_cross_combinations_reported':True,
        'dominance_rule':'paired95percentLCB of bound improvement minus half baselinegap greater than zero',
        'target_final_paths':8192,'physical_steps':128,'exercise_intervals':64,'all_paths_retained':True,
        'price_allowance':'.001+1e-4*abs(V)','delta_allowance':'.002+.005*abs(delta)','confidence_multiplier':1.96,
        'tolerances_changed':False,'american_pricing_ready':False,'new_rule_stock_delta_validated':False,
        'pilot_inventory_sha256':sha(OUTPUT/'rough_tail_attribution_v1/spent_switch_pilot_v1/preservation_inventory.json'),
        'old_tail_replay_inventory_sha256':sha(OUTPUT/'rough_tail_attribution_v1/spent_state_replay_v1/preservation_inventory.json'),
        'no_new_paths_generated_before_design':True}
    design['forbidden_baseline_seeds']=sorted(forbidden)
    write(FOLDER/'design_before_training.json',design)
    write(FOLDER/'execution_environment.json',{'interpreter':sys.executable,'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'pandas':pd.__version__,'environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}})
    print('PREPARED',now(),'SOURCES',len(hashes),flush=True)
def generate(seed,paths):
    frozen,prior,grid,_=inputs();proxy=frozen['proxy'];policy=frozen['selected_policy']
    trace=value_fourier_stopping_policy(proxy,config=replace(proxy.config,valuation_paths=paths,valuation_seed=seed),trace=True)['trace']
    forbidden=BASELINE_RESERVED+((TRAIN_SEED,) if seed in FINAL_SEEDS else ())
    w=current_control_workspace(proxy,policy,frozen['weights'],prior,grid,trace,forbidden_seeds=forbidden)
    details=tail_switch_corrections(proxy,policy,trace,w['current_martingale'],forbidden_seeds=forbidden)
    w.update(tail_gates=details['gates'],tail_margin=details['margin'])
    return frozen,trace,w
def fit_controls(w):
    controls={BASE_CONTROL:{'family':'fallback','bins':1,'penalty':0.,'shrink_scale':1.,'weights':np.zeros(len(FEATURES))}};trials=[]
    for family,features in FAMILIES.items():
        for bins in (1,4):
            basis=correction_basis(w['tail_corrections'][:,:,features],w['times'],bins)
            for penalty in (0.,1e-4):
                name=f'{family} {bins} bins penalty{penalty:g}'
                coefficient,trial=fit_correction_weights(w['payoff'][:2048],w['current_martingale'][:2048],w['stops'][:2048],basis[:2048],bins=bins,penalty=penalty)
                trials.append({'control':name,'family':family,**trial})
                if coefficient is not None:
                    full=np.zeros((bins,len(FEATURES)));full[:,features]=coefficient.reshape(bins,-1)
                    controls[name]={'family':family,'bins':bins,'penalty':penalty,'shrink_scale':1.,'weights':full.ravel()}
    for name,parent in tuple(controls.items()):
        if parent['family']=='extended':
            for scale in (.25,.5):
                controls[f'{name} shrink{scale:g}']={**parent,'parent_control':name,'shrink_scale':scale,'weights':np.asarray(parent['weights'])*scale}
    return controls,pd.DataFrame(trials)
def evaluate(w,trace,policies,controls):
    candidates=evaluate_policy_candidates(trace,policies,w['current_martingale'])
    raw=np.column_stack([r['raw'] for r in candidates.values()]);stops=np.column_stack([r['stops'] for r in candidates.values()])
    if not np.array_equal(stops[:,0],w['stops']):raise ValueError('Frozen baseline stopping rule changed')
    data={'raw':raw,'stops':stops};bases={};path=np.arange(len(raw))[:,None]
    for index,(name,control) in enumerate(controls.items()):
        bins=control['bins']
        if bins not in bases:bases[bins]=correction_basis(w['tail_corrections'],w['times'],bins)
        m=w['current_martingale']+np.einsum('ntd,d->nt',bases[bins],np.asarray(control['weights']))
        lower=raw-m[path,stops];upper=np.max(np.where(trace['eligible'],w['payoff']-m,-np.inf),axis=1)
        upper=np.maximum(upper,w['payoff'][:,0])
        if np.any(upper[:,None]<lower-2e-11):raise FloatingPointError('Joint lower/upper ordering failed')
        if control['family']=='poststop' and not np.array_equal(lower[:,0],data['control0_lower'][:,0]):raise FloatingPointError('Poststop control changed baseline lower')
        data[f'control{index}_lower']=lower;data[f'control{index}_upper']=upper
    return candidates,data
def save_samples(path,trace,w,data):
    arrays={k:w[k] for k in ('stable_old_martingale','prior_local_corrections','prior_martingale','grid_corrections','current_martingale','tail_corrections','tail_gates','tail_margin')}
    arrays.update({k:trace[k] for k in ('times','discounts','eligible','actual_time_grid','stocks','factors','european_proxy')})
    arrays['payoff']=w['payoff']
    arrays.update({k:trace[k][:,:,0] for k in ('european_martingale','intrinsic_martingale','premium_martingale','micro_boundary_martingale')})
    np.savez_compressed(path,**arrays,**data)
def train():
    d=verified();target=FOLDER/'selection_before_final.json'
    if target.exists() or any(FOLDER.glob('final_*')):raise FileExistsError('A selection/final stage already exists')
    start=time.perf_counter();print('TRAIN',TRAIN_SEED,flush=True);frozen,trace,w=generate(TRAIN_SEED,4096)
    controls,trials=fit_controls(w);policies=build_margin_policies(frozen['selected_policy'])
    evaluated,data=evaluate(w,trace,policies,controls)
    held={n:{k:v[2048:] for k,v in r.items()} for n,r in evaluated.items()}
    selected_policy,_,policy_selection=select_margin_policy(held,policies,selection_seed=TRAIN_SEED)
    rows=[]
    for i,(name,control) in enumerate(controls.items()):
        lower=data[f'control{i}_lower'][2048:,0];gap=data[f'control{i}_upper'][2048:]-lower
        mean,se=stat([gap]);price,pse=stat([lower])
        rows.append({'control':name,'family':control['family'],'heldout_paths':2048,'price':price,'price_se':pse,'gap':mean,'gap_se':se,
            'target_final_paths':8192,'selection_score':mean+1.96*.5*(se+pse),'final_paths_used_for_selection':False})
    selection=pd.DataFrame(rows);selected_control=selection.loc[selection.selection_score.idxmin(),'control']
    family_selection={}
    for family in FAMILIES:
        allowed=selection.loc[selection.family.isin(('fallback',family))]
        family_selection[family]=allowed.loc[allowed.selection_score.idxmin(),'control']
    selection['selected_before_final']=selection.control==selected_control
    trials.to_csv(FOLDER/'control_trials.csv',index=False);selection.to_csv(FOLDER/'control_selection.csv',index=False)
    policy_selection.to_csv(FOLDER/'policy_selection.csv',index=False)
    save_samples(FOLDER/'training_samples.npz',trace,w,data)
    seal={'sealed_at':now(),'selected_control':selected_control,'selected_policy':selected_policy,'family_selected_controls':family_selection,
        'controls':controls,'policy_order':list(policies),'policy_specs':margin_policy_specs(),'selection_seed':TRAIN_SEED,
        'design_sha256':sha(FOLDER/'design_before_training.json'),'final_paths_generated_before_seal':False}
    write(target,seal)
    write(FOLDER/'training_manifest.json',{'seed':TRAIN_SEED,'paths':4096,'elapsed_seconds':time.perf_counter()-start,'completed_at':now(),
        'source_sha256':d['source_sha256'],'design_sha256':sha(FOLDER/'design_before_training.json'),'selection_sha256':sha(target),
        'sample_sha256':sha(FOLDER/'training_samples.npz'),'diagnostics':w['diagnostics'],'sources_unchanged':not any(sha(ROOT/n)!=h for n,h in d['source_sha256'].items())})
    print('SEALED',selected_policy,selected_control,now(),flush=True)
def final(index):
    d=verified();seal=read(FOLDER/'selection_before_final.json');spec=d['final_streams'][index];seed=spec['seed']
    if (FOLDER/f'final_{index}_manifest.json').exists() or (FOLDER/f'final_{index}_samples.npz').exists():raise FileExistsError('Final stream already exists')
    start=time.perf_counter();print('FINAL',index,seed,flush=True);frozen,trace,w=generate(seed,spec['paths'])
    policies=build_margin_policies(frozen['selected_policy'])
    policies={n:replace(p,training_seeds=tuple(dict.fromkeys(p.training_seeds+(TRAIN_SEED,)))) for n,p in policies.items()}
    _,data=evaluate(w,trace,policies,seal['controls']);path=FOLDER/f'final_{index}_samples.npz';save_samples(path,trace,w,data)
    write(FOLDER/f'final_{index}_manifest.json',{'index':index,'seed':seed,'paths':spec['paths'],'completed_at':now(),
        'elapsed_seconds':time.perf_counter()-start,'policy_order':list(policies),'control_order':list(seal['controls']),
        'sample_sha256':sha(path),'selection_sha256':sha(FOLDER/'selection_before_final.json'),'design_sha256':sha(FOLDER/'design_before_training.json'),
        'source_sha256':d['source_sha256'],'diagnostics':w['diagnostics'],'sources_unchanged':not any(sha(ROOT/n)!=h for n,h in d['source_sha256'].items())})
    print('COMPLETE FINAL',index,now(),flush=True)
def collect():
    d=verified();seal=read(FOLDER/'selection_before_final.json')
    if (FOLDER/'manifest.json').exists():raise FileExistsError('Campaign already collected')
    streams=[];records=[]
    for i,spec in enumerate(d['final_streams']):
        r=read(FOLDER/f'final_{i}_manifest.json');path=FOLDER/f'final_{i}_samples.npz'
        if r['sample_sha256']!=sha(path) or r['selection_sha256']!=sha(FOLDER/'selection_before_final.json'):raise ValueError('Changed final stream')
        with np.load(path,allow_pickle=False) as z:streams.append({k:z[k].copy() for k in z.files if k.startswith('control')})
        records.append(r)
    rows=[];paired=[];perseed=[];tails=[];names=list(seal['controls']);policies=seal['policy_order']
    def summarize(seq):
        value,se=stat(seq);return {'mean':value,'standard_error':se,'low':value-1.96*se,'high':value+1.96*se}
    for c,name in enumerate(names):
        for p,policy in enumerate(policies):
            lo=[s[f'control{c}_lower'][:,p] for s in streams];up=[s[f'control{c}_upper'] for s in streams];g=[u-l for u,l in zip(up,lo)]
            price,pse=stat(lo);upper,use=stat(up);gap,gse=stat(g);allowance=.001+1e-4*abs(price)
            flags={'control':name,'family':seal['controls'][name]['family'],'policy':policy,'selected_control':name==seal['selected_control'],'selected_policy':policy==seal['selected_policy']}
            rows.append({**flags,'paths':8192,'streams':2,'price':price,'price_se':pse,'upper':upper,'upper_se':use,'gap':gap,'gap_se':gse,
                'price_error_budget':gap+1.96*(pse+gse),'price_allowance':allowance,'price_budget_pass':gap+1.96*(pse+gse)<=allowance,
                'american_pricing_ready':False,'new_rule_stock_delta_validated':False})
            pairs={k:[] for k in ('lower_change','upper_change','gap_change')}
            for s,l,u in zip(streams,lo,up):
                bl=s['control0_lower'][:,0];bu=s['control0_upper'];pairs['lower_change'].append(l-bl);pairs['upper_change'].append(u-bu);pairs['gap_change'].append((u-l)-(bu-bl))
            row=flags.copy()
            for k,seq in pairs.items():
                st=summarize(seq);row.update({k:st['mean'],k+'_se':st['standard_error'],k+'_low':st['low'],k+'_high':st['high']})
            paired.append(row)
            for i,(l,u) in enumerate(zip(lo,up)):
                value,se=stat([l]);gm,gs=stat([u-l]);perseed.append({**flags,'seed':d['final_streams'][i]['seed'],'paths':len(l),'price':value,'price_se':se,'upper':float(u.mean()),'gap':gm,'gap_se':gs})
            if p==0 or (name==seal['selected_control'] and policy==seal['selected_policy']):
                a=np.concatenate(g);total=a.sum();ordered=np.sort(a)
                tails.append({**flags,'paths':len(a),'zero_gap_fraction':float(np.mean(a<=1e-12)),'gap_p90':float(np.quantile(a,.9)),
                    'gap_p99':float(np.quantile(a,.99)),'gap_maximum':float(a.max()),'largest_decile_gap_share':float(ordered[-int(np.ceil(.1*len(a))):].sum()/total),
                    'largest_percentile_gap_share':float(ordered[-int(np.ceil(.01*len(a))):].sum()/total),'used_for_selection':False,'acceptance_input':False})
    ci=names.index(seal['selected_control']);pi=policies.index(seal['selected_policy']);dominance=[]
    for challenge in ('dual','policy'):
        gain=[];margin=[]
        for s in streams:
            b=s['control0_upper']-s['control0_lower'][:,0]
            improvement=s['control0_upper']-s[f'control{ci}_upper'] if challenge=='dual' else s['control0_lower'][:,pi]-s['control0_lower'][:,0]
            gain.append(improvement);margin.append(improvement-.5*b)
        st=summarize(gain);ms=summarize(margin)
        dominance.append({'challenge':challenge,'selected_control':seal['selected_control'],'selected_policy':seal['selected_policy'],
            'bound_improvement':st['mean'],'bound_improvement_se':st['standard_error'],'improvement_low':st['low'],'improvement_high':st['high'],
            'majority_margin':ms['mean'],'majority_margin_se':ms['standard_error'],'majority_margin_low':ms['low'],'majority_margin_high':ms['high'],
            'majority_dominance_proved':ms['low']>0,'remaining_source_unresolved':ms['low']<=0,'confidence_multiplier':1.96})
    tables={'comparison':rows,'paired_changes':paired,'per_seed':perseed,'dominance':dominance,'gap_distribution':tails}
    for name,values in tables.items():pd.DataFrame(values).to_csv(FOLDER/(name+'.csv'),index=False)
    manifest={**d,'completed_at':now(),'complete':True,'selected_control':seal['selected_control'],'selected_policy':seal['selected_policy'],
        'family_selected_controls':seal['family_selected_controls'],'policy_order':policies,'control_order':names,'records':records,
        'selection_sha256':sha(FOLDER/'selection_before_final.json'),'training_manifest_sha256':sha(FOLDER/'training_manifest.json'),
        'sources_unchanged':True,'completed_final_paths':8192,'table_rows':{n:len(v) for n,v in tables.items()}}
    write(FOLDER/'manifest.json',manifest);entries=[record(p) for p in sorted(FOLDER.rglob('*')) if p.is_file()]
    write(FOLDER/'preservation_inventory.json',{'files':entries,'inventory_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    print(pd.DataFrame(rows).loc[lambda t:t.selected_policy&t.selected_control].to_string(index=False),flush=True);print(pd.DataFrame(dominance).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','train','final','collect'));parser.add_argument('--index',type=int,choices=(0,1));args=parser.parse_args()
    if args.stage=='final' and args.index is None:parser.error('final requires --index')
    {'prepare':prepare,'train':train,'final':lambda:final(args.index),'collect':collect}[args.stage]()
