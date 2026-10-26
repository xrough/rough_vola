"""Verified collection of independently executed stopping/delta experiments.

Notebooks analyze the sealed current campaign; the explicit example runners
perform simulation. Missing or changed experiments raise rather than yielding
fabricated numerical results or silently reusing an older campaign.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_file(path,record):
    path=Path(path);raw=path.read_bytes()
    if (len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']):
        raise ValueError('Changed stopping/delta evidence: '+str(path))
    return path


def _fixed_tolerances(record,*,required=False):
    constants={'price_atol':.001,'price_rtol':1e-4,'delta_atol':.002,'delta_rtol':.005,
        'confidence_multiplier':1.96,'ci_multiplier':1.96,'spread_fraction':.1}
    names={'price_atol','price_rtol','delta_atol','delta_rtol','confidence_multiplier','spread_fraction'} if required else set(record)&set(constants)
    if any(name not in record or isinstance(record[name],bool) or record[name]!=constants[name] for name in names):
        raise ValueError('The stopping/delta acceptance tolerances changed')
    if record.get('tolerances_changed',False) or record.get('unchanged_tolerances',True) is False:
        raise ValueError('The stopping/delta acceptance tolerances changed')


def _delta_gate(row,*,paths):
    allowance=.002+.005*abs(row.mean)
    if (row.paths!=paths or not np.isclose(row.delta_allowance,allowance,rtol=1e-12,atol=2e-14)
        or row.sampling_passed!=(row.ci_half_width<=allowance) or row.optimal_delta_ready!=False):
        raise ValueError('Changed delta allowance, sampling gate, path count or readiness')


def _manifest(folder,*,inventory=False,current_sources=True,required_files=None):
    folder=Path(folder);path=folder/'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('Complete the independent experiment before notebook collection: '+str(path))
    m=json.loads(path.read_text());entries=[]
    _fixed_tolerances(m)
    if inventory:
        inv=json.loads((folder/'preservation_inventory.json').read_text())
        digest=hashlib.sha256(json.dumps(inv['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=inv['inventory_sha256']:raise ValueError('Changed experiment preservation inventory')
        entries.append({'path':str(folder/'preservation_inventory.json'),'bytes':(folder/'preservation_inventory.json').stat().st_size,'sha256':_sha(folder/'preservation_inventory.json')})
        for row in inv['files']:
            p=(folder/row['path']).resolve()
            if not p.is_relative_to(folder.resolve()):raise ValueError('Artifact escapes its experiment folder')
            _verify_file(p,row);entries.append({'path':str(p),**{k:row[k] for k in ('bytes','sha256')}})
    else:
        seen=set()
        for row in m.get('files',[]):
            name=Path(row['path']).name
            if name in seen:raise ValueError('Duplicate experiment artifact')
            seen.add(name);p=folder/name
            _verify_file(p,row);entries.append({'path':str(p),**{k:row[k] for k in ('bytes','sha256')}})
    if required_files is not None:
        names=[Path(row['path']).name for row in m.get('files',[])]
        if len(names)!=len(required_files) or set(names)!=set(required_files):
            raise ValueError('Experiment omits or duplicates its required artifacts')
    if 'design_sha256' in m:
        candidates=[p for p in folder.glob('design_before*.json') if _sha(p)==m['design_sha256']]
        if len(candidates)!=1:raise ValueError('Missing or changed predeclared design')
        p=candidates[0];design=json.loads(p.read_text());_fixed_tolerances(design)
        if 'source_sha256' in m and m['source_sha256']!=design.get('source_sha256'):
            raise ValueError('Manifest numerical sources differ from its sealed design')
        entries.append({'path':str(p),'bytes':p.stat().st_size,'sha256':_sha(p)})
    if m.get('sources_unchanged') is False or m.get('source_changes'):
        raise ValueError('A numerical source changed during the stopping/delta experiment')
    if current_sources:
        root=Path(__file__).resolve().parents[3]
        source_map=design.get('source_sha256',{}) if 'design_sha256' in m else m.get('source_sha256',{})
        for name,digest in source_map.items():
            p=root/name
            if _sha(p)!=digest:raise ValueError('Changed current numerical source: '+str(p))
    entries.append({'path':str(path),'bytes':path.stat().st_size,'sha256':_sha(path)})
    return m,entries


def _table(path,entries):
    path=Path(path).resolve()
    rows=[row for row in entries if Path(row['path']).resolve()==path]
    if len(rows)!=1:raise ValueError('An unsealed numerical table was requested: '+str(path))
    _verify_file(path,rows[0])
    return pd.read_csv(path,float_precision='round_trip')


def _stat(a):
    a=np.asarray(a,float)
    if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all():raise ValueError('Finite whole-path samples required')
    mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    return {'mean':mean,'standard_error':se,'ci_half_width':1.96*se,'interval_low':mean-1.96*se,'interval_high':mean+1.96*se}


def _primary_deltas(folder):
    manifest,entries=_manifest(folder)
    design_path=folder/'design_before_primary_valuation.json'
    design=json.loads(design_path.read_text())
    if (design['seeds']!=[710005042,711005042] or design['paths_per_stream']!=4096
        or design['bump_relatives']!=[.01,.005,.0025] or design['steps']!=128
        or manifest.get('complete') is not True or not manifest['all_streams_retained']
        or manifest.get('paths')!=8192 or manifest.get('seed_count')!=2
        or manifest.get('independent_paths') is not True or manifest.get('design_sha256')!=_sha(design_path)):
        raise ValueError('Wrong or incomplete independent delta design')
    bundle=folder.parents[1]/'rough_martingale_centering_v1/frozen_joint_policy'
    if (_sha(bundle/'manifest.json')!=design['frozen_bundle_manifest_sha256']
        or _sha(bundle/'coefficients.npz')!=design['frozen_bundle_arrays_sha256']):
        raise ValueError('Frozen delta policy/control bundle changed')
    project=Path(__file__).resolve().parents[3]
    for name,digest in design['source_sha256'].items():
        if _sha(project/name)!=digest:raise ValueError('Changed primary delta source: '+name)
    _fixed_tolerances(design,required=True)
    all_arrays=[];streams=[]
    for i,seed in enumerate(design['seeds']):
        path=folder/f'stream_{i}.json';record=json.loads(path.read_text())
        if record['seed']!=seed or record['paths']!=4096 or record['design_sha256']!=_sha(design_path) or record['source_sha256']!=design['source_sha256']:
            raise ValueError('Delta stream differs from its predeclared design')
        expected_files={f'stream_{i}_{name}' for name in ('samples.npz','estimates.csv','comparisons.csv')}
        actual_files=[Path(row['path']).name for row in record['files']]
        if len(actual_files)!=3 or set(actual_files)!=expected_files:
            raise ValueError('Primary delta stream omits or duplicates its exact artifact inventory')
        for row in record['files']:
            p=folder/Path(row['path']).name
            _verify_file(p,row);entries.append({'path':str(p),**{k:row[k] for k in ('bytes','sha256')}})
        with np.load(folder/f'stream_{i}_samples.npz',allow_pickle=False) as z:
            arrays={key:z[key].copy() for key in z.files}
        relative=np.array([-.01,-.005,-.0025,0.,.0025,.005,.01])
        if not np.array_equal(arrays.get('relative_spots'),relative):
            raise ValueError('Changed declared stock bumps')
        for policy in ('original_policy','selected_policy'):
            for name in ('raw_prices','controlled_prices','stops'):
                if arrays[policy+'__'+name].shape!=(4096,7):
                    raise ValueError('Missing stock-bump paths or contracts')
        if any(v.shape[0]!=4096 or not np.isfinite(v).all() for key,v in arrays.items() if key!='relative_spots'):
            raise ValueError('Delta samples omit declared paths')
        for policy in ('original_policy','selected_policy'):
            for estimator,price_key in (('raw','raw_prices'),('stopped martingale','controlled_prices')):
                values=arrays[policy+'__'+price_key]
                for level in (.01,.005,.0025):
                    lo=int(np.flatnonzero(relative==-level)[0]);hi=int(np.flatnonzero(relative==level)[0])
                    delta=(values[:,hi]-values[:,lo])/(200.*level)
                    if not np.allclose(delta,arrays[policy+'__'+f'{estimator}:{level:g}'],rtol=0.,atol=2e-14):
                        raise ValueError('Delta does not equal its paired stock-price differences')
        all_arrays.append(arrays)
        frame=pd.read_csv(folder/f'stream_{i}_estimates.csv',float_precision='round_trip')
        expected={(p,e,b) for p in ('original_policy','selected_policy') for e in ('raw','stopped martingale') for b in (.01,.005,.0025)}
        if len(frame)!=12 or set(zip(frame.policy,frame.estimator,frame.bump_relative))!=expected:
            raise ValueError('Stream table omits or duplicates a declared delta')
        for row in frame.itertuples():
            if row.seed!=seed or row.paths!=4096 or row.steps!=128:raise ValueError('Changed stream metadata')
            _delta_gate(row,paths=4096)
            actual=_stat(arrays[row.policy+'__'+f'{row.estimator}:{row.bump_relative:g}'])
            if any(not np.isclose(getattr(row,key),value,rtol=1e-12,atol=2e-14) for key,value in actual.items()):
                raise ValueError('Changed stream delta statistics')
        streams.append(frame)
        entries.append({'path':str(path),'bytes':path.stat().st_size,'sha256':_sha(path)})
    pooled=pd.read_csv(folder/'pooled_estimates.csv',float_precision='round_trip')
    paired=pd.read_csv(folder/'pooled_comparisons.csv',float_precision='round_trip')
    expected={(p,e,b) for p in ('original_policy','selected_policy') for e in ('raw','stopped martingale') for b in (.01,.005,.0025)}
    if len(pooled)!=12 or set(zip(pooled.policy,pooled.estimator,pooled.bump_relative))!=expected or len(paired)!=14:
        raise ValueError('Delta tables omit or duplicate declared policies/estimators/bumps')
    pairs={("bump refinement",p,e,c,f) for p in ('original_policy','selected_policy') for e in ('raw','stopped martingale') for c,f in ((.01,.005),(.005,.0025))}
    pairs|={("selected minus original","paired policies",e,None,b) for e in ('raw','stopped martingale') for b in (.01,.005,.0025)}
    actual={(r.comparison,r.policy,r.estimator,None if pd.isna(r.coarse_bump) else r.coarse_bump,r.fine_bump) for r in paired.itertuples()}
    if actual!=pairs:raise ValueError('Paired delta table omits or duplicates declared comparisons')
    joined={k:np.concatenate([a[k] for a in all_arrays]) for k in all_arrays[0] if k!='relative_spots'}
    for row in pooled.itertuples():
        _delta_gate(row,paths=8192)
        values=joined[row.policy+'__'+f'{row.estimator}:{row.bump_relative:g}'];actual=_stat(values)
        for name,value in actual.items():
            if not np.isclose(getattr(row,name),value,rtol=1e-12,atol=2e-14):raise ValueError('Changed pooled delta statistics')
    for row in paired.itertuples():
        if row.comparison=='bump refinement':
            key=row.policy+'__'+row.estimator+':'
            values=joined[key+f'{row.fine_bump:g}']-joined[key+f'{row.coarse_bump:g}']
        elif row.comparison=='selected minus original':
            key='__'+f'{row.estimator}:{row.fine_bump:g}'
            values=joined['selected_policy'+key]-joined['original_policy'+key]
        else:raise ValueError('Unknown saved delta comparison')
        for name,value in _stat(values).items():
            if not np.isclose(getattr(row,name),value,rtol=1e-12,atol=2e-14):raise ValueError('Changed paired delta comparison')
    for file in (design_path,folder/'pooled_estimates.csv',folder/'pooled_comparisons.csv'):
        entries.append({'path':str(file),'bytes':file.stat().st_size,'sha256':_sha(file)})
    return pooled,paired,pd.concat(streams,ignore_index=True),entries,manifest


def _larger_policy(folder):
    manifest,entries=_manifest(folder,inventory=True)
    if (manifest['training_seed']!=610005042 or manifest['training_paths']!=32768
        or manifest['selection_seed']!=620005042 or manifest['selection_paths']!=8192
        or manifest['completed_final_paths']!=16384 or manifest['completed_final_streams']!=2
        or not manifest['same_upper_across_policies']):
        raise ValueError('Wrong or incomplete independently selected larger-policy experiment')
    design=json.loads((folder/'design_before_training.json').read_text())
    if manifest['source_sha256']!=design['source_sha256'] or design['final_streams']!=[{'seed':630005042,'paths':8192},{'seed':631005042,'paths':8192}]:
        raise ValueError('Changed larger-policy design')
    seal_path=folder/'frozen_policy_before_final.json';seal=json.loads(seal_path.read_text())
    if _sha(seal_path)!=manifest['policy_seal_sha256'] or seal['selected_name']!=manifest['selected_name']:
        raise ValueError('Changed policy selection after valuation')
    _verify_file(folder/seal['coefficient_archive'],{'bytes':(folder/seal['coefficient_archive']).stat().st_size,'sha256':seal['coefficient_archive_sha256']})
    names=list(design['candidates']);arrays=[]
    for spec in design['final_streams']:
        stem='final_'+str(spec['seed']);record=json.loads((folder/(stem+'_manifest.json')).read_text())
        path=folder/(stem+'_samples.npz')
        if record['seed']!=spec['seed'] or record['paths']!=8192 or record['policies']!=names or record['policy_seal_sha256']!=_sha(seal_path) or record['sample_sha256']!=_sha(path):
            raise ValueError('Changed independent final policy stream')
        with np.load(path,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
        if any(v.shape!=(8192,) or not np.isfinite(v).all() for v in a.values()):
            raise ValueError('Final policy samples omit declared paths')
        for i in range(len(names)):
            if np.min(a['common_upper']-a[f'policy{i}_lower']) < -1e-10:
                raise ValueError('Policy samples violate their shared-upper identity')
        arrays.append(a)
    pooled=_table(folder/'pooled.csv',entries)
    if len(pooled)!=len(names) or set(pooled.policy)!=set(names):raise ValueError('Missing final policy candidate')
    strat=lambda chunks:float(np.sqrt(sum(len(x)*np.var(x,ddof=1) for x in chunks))/sum(map(len,chunks)))
    for i,name in enumerate(names):
        row=pooled.loc[pooled.policy==name].iloc[0]
        lower=[a[f'policy{i}_lower'] for a in arrays]
        gap=[a['common_upper']-x for a,x in zip(arrays,lower)]
        gain=[x-a['policy0_lower'] for a,x in zip(arrays,lower)]
        mean=lambda chunks:float(np.concatenate(chunks).mean())
        allowance=.001+1e-4*abs(mean(lower));budget=mean(gap)+1.96*strat(gap)+1.96*strat(lower)
        expected={'price':mean(lower),'price_se':strat(lower),'gap':mean(gap),'gap_se':strat(gap),
            'paired_lower_gain':mean(gain),'paired_lower_gain_se':strat(gain),
            'gain_low':mean(gain)-1.96*strat(gain),'gain_high':mean(gain)+1.96*strat(gain),
            'price_allowance':allowance,'price_error_budget':budget}
        if any(not np.isclose(row[key],value,rtol=1e-11,atol=2e-13) for key,value in expected.items()):
            raise ValueError('Changed pooled stopping statistics')
        if bool(row.price_budget_pass)!=(budget<=allowance) or bool(row.selected_before_final)!=(name==manifest['selected_name']):
            raise ValueError('Changed stopping acceptance or final policy selection')
    return pooled,entries,manifest


def stopping_delta_study(context,*,seed=42,compute_figures=True):
    from numbers import Integral
    from ._pricing_rough_foundation import _fingerprint
    if isinstance(seed,bool) or not isinstance(seed,Integral) or seed!=42:raise ValueError('Declared research seed is42')
    if not isinstance(compute_figures,bool):raise TypeError('compute_figures must be boolean')
    root=Path(context.output_dir);entries=[];tables={};records=[]
    context.progress('Verifying independent stopping-policy and stock-delta experiments; earlier studies are historical')
    delta,paired,streams,files,dmeta=_primary_deltas(root/'rough_stock_delta_v1/frozen_joint_2x4096')
    entries+=files;tables.update(delta_estimates=delta,delta_comparisons=paired,delta_stream_estimates=streams)
    folder=root/'rough_stock_delta_v1/frozen_joint_time_refinement'
    from ._pricing_rough_stopping_delta_validation import _time_deltas
    time,files,tm=_time_deltas(folder);entries+=files
    tables['delta_time_estimates']=time
    changes=[]
    for row in time.itertuples():
        base=delta.loc[(delta.policy==row.policy)&(delta.estimator==row.estimator)&(delta.bump_relative==row.bump_relative)]
        if len(base)!=1:raise ValueError('Independent time estimates do not align with baseline')
        b=base.iloc[0];change=row.mean-b['mean'];se=float(np.hypot(row.standard_error,b.standard_error))
        allowance=.002+.005*abs(row.mean)
        changes.append({'policy':row.policy,'estimator':row.estimator,'bump_relative':row.bump_relative,
            'coarse_steps':128,'fine_steps':256,'mean':change,'standard_error':se,'ci_half_width':1.96*se,
            'interval_low':change-1.96*se,'interval_high':change+1.96*se,'delta_allowance':allowance,
            'independent_not_paired':True,'difference_plus_sampling_within_allowance':abs(change)+1.96*se<=allowance,
            'optimal_delta_ready':False})
    tables['delta_time_comparisons']=pd.DataFrame(changes)
    folder=root/'rough_larger_policy_v1/32768_paths_two_starts'
    stop,files,pm=_larger_policy(folder);entries+=files
    stop['control']='frozen common martingale'
    tables.update(stopping_comparison=stop,stopping_paired_changes=stop.copy(),
        stopping_streams=_table(folder/'per_seed.csv',files),
        stopping_selection=_table(folder/'training_selection.csv',files))
    pilots=[]
    for label in ('pilot_tensor21','pilot_microdate_value','exploratory_spent512_block'):
        directory=root/'rough_stopping_refresh_v1'/label
        m,files=_manifest(directory,inventory=True,current_sources=False);entries+=files
        f=directory/'comparison.csv'
        if not f.exists():f=directory/'comparisons.csv'
        if f.exists():
            d=_table(f,files);d['trial']=label;d['acceptance_evidence']=False;pilots.append(d)
        records.append({'trial':label,'scope':'exploratory; retained regardless of outcome','acceptance_evidence':False})
    tables['stopping_trials']=pd.concat(pilots,ignore_index=True,sort=False) if pilots else pd.DataFrame()
    refs=[]
    for label in ('independent_reference_checks','heston_stopped_control_followup'):
        directory=root/'rough_stock_delta_v1'/label
        required={'reference_estimates.csv','timings.csv','samples.npz'}
        if label=='heston_stopped_control_followup':required.add('fourier_reference_refinement.csv')
        m,files=_manifest(directory,required_files=required);entries+=files
        path=directory/'reference_estimates.csv'
        d=_table(path,files)
        if label=='independent_reference_checks':
            expected_seeds={713005042:4096,714005042:32768,715005042:32768}
        else:expected_seeds={716005042:2048,717005042:2048}
        keys={(seed,est,bump) for seed in expected_seeds for est in ('raw','stopped martingale') for bump in (.01,.005,.0025)}
        # The original cheap-control reference trial uses its own explicitly
        # labelled estimator; it remains a failed comparison when appropriate.
        if label=='independent_reference_checks':
            keys={(713005042,e,b) for e in ('raw','stopped martingale') for b in (.01,.005,.0025)}
            keys|={(seed,e,b) for seed in (714005042,715005042) for e in ('raw','terminal conditional control') for b in (.01,.005,.0025)}
        if len(d)!=len(keys) or set(zip(d.seed,d.estimator,d.bump_relative))!=keys:
            raise ValueError('Reference tables omit declared cases or estimators')
        for row in d.itertuples():
            _delta_gate(row,paths=expected_seeds[row.seed])
            allowance=.002+.005*abs(row.reference_delta)
            error=abs(row.mean-row.reference_delta)
            if (not np.isclose(row.reference_allowance,allowance,rtol=1e-12,atol=2e-14)
                or not np.isclose(row.absolute_error,error,rtol=1e-12,atol=2e-14)
                or row.reference_gate_passed!=(error+row.ci_half_width<=allowance)):
                raise ValueError('Changed independent reference gate')
        d['reference_trial']=label;refs.append(d)
    tables['delta_reference_estimates']=pd.concat(refs,ignore_index=True,sort=False)
    delta_pilots=[]
    for label in ('terminal_cv_2048_spent_stream','stopped_martingale_512'):
        directory=root/'rough_stock_delta_probe_v1'/label
        pilot,files=_manifest(directory,current_sources=False,required_files={'estimates.csv','comparisons.csv','samples.npz'});entries+=files
        d=_table(directory/'estimates.csv',files)
        d['trial']=label;d['acceptance_evidence']=False;delta_pilots.append(d)
        records.append({'trial':label,'scope':'exploratory estimator comparison; excluded from primary pool','acceptance_evidence':False})
    tables['delta_trials']=pd.concat(delta_pilots,ignore_index=True)
    tables['trials']=pd.DataFrame(records)
    tables['configuration']=pd.DataFrame([
        {'item':'Price tolerance','value':'.001+1e-4*abs(price); gap upper interval plus lower sampling'},
        {'item':'Delta tolerance','value':'.002+.005*abs(delta); paired stock bumps, independent time comparison'},
        {'item':'Policy identity','value':'Delta audit uses old original/dense rules; new larger-fit rule has no inherited Greek validation'},
        {'item':'Scope','value':'60-day ATM no-cash put at inception; finite kernel, actual weak law and exercise calendar'},
        {'item':'Real SPY','value':'Independent spot/dividend expectations and historical rough factor states remain blocked'}])
    # Every external byte used in the current analysis is sealed into its own
    # profile. Primary designs and snapshots are verified above; old pilots
    # are read from their original preserved source inventories.
    inventory={r['path']:r for r in entries};entries=[inventory[k] for k in sorted(inventory)]
    analysis_sources={Path(__file__).name:_sha(__file__),
        '_pricing_rough_stopping_delta_validation.py':_sha(Path(__file__).with_name('_pricing_rough_stopping_delta_validation.py'))}
    profile=_fingerprint({'version':'rough_stopping_delta_v1','files':entries,'analysis_sources':analysis_sources})
    tables['provenance']=pd.DataFrame(entries)
    manifest={'backend':'rough_heston_stopping_delta','study_version':'rough_stopping_delta_v1',
        'profile_fingerprint':profile,'input_files':entries,'analysis_sources':analysis_sources,'primary_complete':True,
        'delta_primary_paths':8192,'delta_time_paths':2048,'stopping_primary_paths':16384,
        'stopping_selected_policy':pm['selected_name'],'current_engine_execution':False,
        'execution_mode':'verified collection of current independent numerical campaigns',
        'historical_replay':False,'primary_policy_candidates_fitted':2,'primary_dual_optimizer_calls':0,
        'tolerances_changed':False,'american_pricing_ready':False,'stock_delta_ready':False,
        'new_stopping_policy_delta_validated':False,
        'limitations':['Optimal value derivatives are not bounded by primal-dual price intervals.',
            'Stock-bump sampling, bump error, policy error and physical time error remain separate.',
            'All failed or unresolved pilots and reference checks remain in the notebook.']}
    result={'tables':tables,'figures':{},'manifest':manifest}
    if compute_figures:
        from ._pricing_rough_stopping_delta_reporting import add_stopping_delta_reporting
        result=add_stopping_delta_reporting(result)
    return result
