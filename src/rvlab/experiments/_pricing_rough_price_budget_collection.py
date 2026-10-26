"""Read-only audit of price-budget attribution and independent precision trials."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import _sha, _manifest, _fixed_tolerances
from ._pricing_rough_gap_time_stopping import _preserved
from ._pricing_rough_policy_validation import _entry
from ._pricing_rough_bellman_control_samples import validate_control_samples
from ._pricing_rough_price_budget_basis import reconstruct_predictable_basis
from ._pricing_rough_price_budget_samples import validate_conditional_european, validate_common_shift
from ._pricing_rough_price_budget_tables import (
    price_budget, diagnose_spent_budget, summarize_price_streams)

ROOT=Path(__file__).resolve().parents[3]
STREAMS=[{'seed':s,'paths':16384} for s in (1106005042,1107005042,1108005042)]


def read(path):return json.loads(Path(path).read_text())
def _time(value):return datetime.fromisoformat(value)
def _npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify_sources(folder,design,expected=None):
    sources=design.get('source_sha256',{})
    if not sources or (expected is not None and set(sources)!=set(expected)):
        raise ValueError('A price-budget numerical dependency was omitted or replaced')
    for name,digest in sources.items():
        source=(ROOT/name).resolve();snapshot=(Path(folder)/'source'/name).resolve()
        if (not source.is_relative_to(ROOT) or not snapshot.is_relative_to(Path(folder).resolve())
                or _sha(source)!=digest or _sha(snapshot)!=digest):
            raise ValueError('A frozen price-budget source or snapshot changed')


def verify_inputs(design,output):
    entries=[]
    for name,digest in design.get('input_sha256',{}).items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or _sha(path)!=digest:
            raise ValueError('A frozen price-budget development or model input changed')
        entries.append(_entry(path))
    if not entries:raise ValueError('Missing frozen price-budget input lineage')
    return entries


def _constant_basis(processes,stops):
    p=np.asarray(processes)
    if p.ndim==2:p=p[:,:,None]
    result=np.zeros((len(p),4,p.shape[2]))
    for j in range(1,65):
        result[:,(j-1)//16]+=(p[:,j]-p[:,j-1])*(stops>=j)[:,None]
    return result.reshape(len(p),-1)


def reconstruct_spent_features(arrays,bundle,anchor):
    stops=arrays['gh16_stops'];path=np.arange(len(stops))
    features={'terminal':arrays['european_martingale'][:,-1,None],
        'poststop':(arrays['european_martingale'][:,-1]-arrays['european_martingale'][path,stops])[:,None],
        'targets':_constant_basis(arrays['target_martingales']-arrays['latest_martingale'][:,:,None],stops),
        'stable':np.column_stack([_constant_basis(arrays[k],stops) for k in
            ('european_martingale','premium_martingale','micro_boundary_martingale')]),
        'prior':_constant_basis(arrays['prior_local_corrections'],stops),
        'grid':_constant_basis(arrays['grid_corrections'],stops),
        'tail':_constant_basis(arrays['tail_corrections'],stops)}
    features['targets_stable']=np.column_stack((features['targets'],features['stable']))
    features['full_stopped']=np.column_stack([features[k] for k in ('targets','stable','prior','grid','tail')])
    features['gated'],_=reconstruct_predictable_basis(arrays,bundle)
    features['gated_anchor']=np.column_stack((features['gated'],anchor))
    return features


def _fit(x,y,ridge):
    center=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale<1e-12,1.,scale)
    xc=(x-center)/scale
    raw=np.linalg.solve(xc.T@xc/len(x)+ridge*np.eye(x.shape[1]),xc.T@(y-y.mean())/len(x))/scale
    return np.clip(raw,-2.,2.),center,scale,raw


def select_spent_specification(frame):
    """Use both spent diagnostic folds, never training or fresh prices."""
    required={'eligible_for_selection','spec_index','diagnostic_sd','fit_seed','diagnostic_seed'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Complete spent crossfit diagnostics required')
    eligible=frame.loc[frame.eligible_for_selection.eq(True)]
    if eligible.empty or not np.isfinite(eligible.diagnostic_sd).all() or (eligible.diagnostic_sd<0).any():
        raise ValueError('No valid predeclared spent control specification')
    for _,folds in eligible.groupby('spec_index',sort=True):
        if (len(folds)!=2 or set(zip(folds.fit_seed,folds.diagnostic_seed))!=
                {(1094005042,1095005042),(1095005042,1094005042)}):
            raise ValueError('Each eligible specification requires both independent spent split directions')
    scores=eligible.groupby('spec_index',sort=True).diagnostic_sd.apply(lambda x:float(np.mean(x*x)))
    return int(scores.idxmin()),scores


def validate_spent_fit(folder,output,bundle,old_arrays,*,prior_entries=None):
    """Reconstruct every failed/eligible trial, both folds and the pre-final seal."""
    folder=Path(folder);inventory,entries=_preserved(folder)
    design=read(folder/'design_before_fitting.json');manifest=read(folder/'manifest.json')
    seal=read(folder/'weights_before_final.json')
    prior_sources=set(read(output/'rough_bellman_validation_v1/bellman_control_v1/design_before_training.json')['source_sha256'])
    verify_sources(folder,design,prior_sources|{'examples/rough_price_budget_fit.py',
        'src/rvlab/experiments/_pricing_rough_price_budget.py',
        'src/rvlab/experiments/_pricing_rough_price_budget_controls.py'})
    entries+=verify_inputs(design,output)
    specs=[{'family':family,'ridge':ridge,'eligible_for_selection':False}
        for family in ('terminal','poststop','targets','targets_stable','full_stopped')
        for ridge in (1e-6,1e-4,.01)]
    specs += [{'family':family,'ridge':ridge,'eligible_for_selection':True}
        for family in ('gated','gated_anchor') for ridge in (1e-4,.001,.01,.1,1.)]
    expected={'version':'rough_price_budget_spent_fit_v1','fit_seeds':[1094005042,1095005042],
        'specifications':specs,'coefficient_bound':2.,'ridge_fit_then_raw_coefficient_projection':True,
        'new_outer_paths':0,'used_for_fresh_acceptance':False,'bins':4,
        'no_regression_intercept_at_valuation':True,'no_empirical_valuation_recentering':True,
        'tolerances_changed':False}
    if (any(design.get(k)!=v for k,v in expected.items()) or manifest.get('complete') is not True
            or any(manifest.get(k)!=v for k,v in design.items()) or manifest.get('trials')!=50
            or seal.get('design_sha256')!=_sha(folder/'design_before_fitting.json')
            or manifest.get('weights_sha256')!=_sha(folder/'weights_before_final.json')
            or seal.get('fresh_paths_generated_before_seal') is not False
            or seal.get('selection_used_only_spent_split_diagnostics') is not True
            or seal.get('all_rows_refitted')!=8192 or seal.get('fit_seeds')!=expected['fit_seeds']
            or seal.get('coefficient_archive_sha256')!=_sha(folder/'all_fit_coefficients.npz')
            or not _time(design['created_at'])<_time(seal['sealed_at'])<=_time(manifest['completed_at'])):
        raise ValueError('Changed spent-fit roles, candidate grid, bounds, completion or chronology')
    anchor_folder=folder.parent/'spent_terminal_anchor_v1'
    _,anchor_entries=_preserved(anchor_folder);entries+=anchor_entries
    anchor_design=read(anchor_folder/'design_before_replay.json')
    verify_sources(anchor_folder,anchor_design,prior_sources|{'examples/rough_price_budget_spent.py',
        'src/rvlab/experiments/_pricing_rough_price_budget.py'})
    entries+=verify_inputs(anchor_design,output)
    anchor_manifest=read(anchor_folder/'manifest.json')
    if (anchor_manifest.get('new_outer_paths')!=0 or anchor_manifest.get('used_for_fresh_acceptance') is not False
            or anchor_manifest.get('complete') is not True or anchor_manifest.get('full_stream_paths')!=4096):
        raise ValueError('Spent conditional replay was promoted to independent acceptance')
    data=[];probes=[];feature_records=[];conditional_replays=[]
    for i,(a,seed) in enumerate(zip(old_arrays,expected['fit_seeds'])):
        conditional=_npz(anchor_folder/f'stream_{i}_samples.npz')
        conditional_replays.append(conditional)
        anchor=validate_conditional_european({**a,**conditional},bundle,seed=seed,paths=4096)
        if not np.allclose(conditional['shared_control'],anchor['shared_european_control'],rtol=0.,atol=2e-12):
            raise ValueError('Spent European control changed its exact centering identity')
        probes.extend({'role':'spent conditional replay','seed':seed,**row} for row in anchor['probes'])
        saved=_npz(folder/f'spent_{i}_basis.npz')
        rebuilt=reconstruct_spent_features(a,bundle,anchor['shared_european_control'])
        if set(saved)!=(set(rebuilt)|{'lower','upper','anchor'}):
            raise ValueError('A spent control family or all-path endpoint was omitted')
        for key,value in {**rebuilt,'lower':a['candidate3_lower'],'upper':a['candidate3_upper'],
                'anchor':anchor['shared_european_control']}.items():
            if saved[key].shape!=value.shape or not np.allclose(saved[key],value,rtol=0.,atol=2e-12):
                raise ValueError('A spent control feature, state gate or stopping mask changed')
        data.append(saved)
        feature_records.append({'seed':seed,'rows':4096,'gated_columns':saved['gated'].shape[1],
            'all_features_reconstructed':True,'new_acceptance_evidence':False})
    coefficients=_npz(folder/'all_fit_coefficients.npz');rows=[]
    if set(coefficients)!=({f'weights_{k}_{fold}' for k in range(25) for fold in (0,1)}|{'selected_weights'}):
        raise ValueError('A predeclared spent coefficient trial was omitted or replaced')
    for k,spec in enumerate(specs):
        for fold in (0,1):
            train=data[fold];validation=data[1-fold];x=train[spec['family']]
            weights,center,scale,raw=_fit(x,train['lower'],spec['ridge'])
            if not np.allclose(coefficients[f'weights_{k}_{fold}'],weights,rtol=1e-11,atol=2e-12):
                raise ValueError('Spent crossfit coefficients differ from their training-only objective')
            residual=validation['lower']-validation[spec['family']]@weights
            rows.append({'spec_index':k,**spec,'fit_seed':expected['fit_seeds'][fold],
                'diagnostic_seed':expected['fit_seeds'][1-fold],'features':len(weights),
                'rows_fit':4096,'rows_diagnostic':4096,
                'training_sd':float(np.std(train['lower']-x@weights,ddof=1)),
                'diagnostic_sd':float(np.std(residual,ddof=1)),
                'diagnostic_mean_shift':float(np.mean(residual-validation['lower'])),
                'max_abs_unprojected_coefficient':float(np.max(abs(raw))),
                'active_projection_count':int(np.sum(abs(raw)>2)),
                'used_for_fresh_acceptance':False})
    frame=pd.DataFrame(rows);saved=pd.read_csv(folder/'spent_trials.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(saved,frame,check_exact=False,rtol=1e-10,atol=2e-12)
    chosen,score=select_spent_specification(frame);spec=specs[chosen]
    x=np.concatenate([d[spec['family']] for d in data]);y=np.concatenate([d['lower'] for d in data])
    weights,center,scale,raw=_fit(x,y,spec['ridge'])
    process_names=('target0-minus-latest','target1-minus-latest','target2-minus-latest','target3-minus-latest',
        'european','premium','intrinsic','micro_boundary')
    column_names=[f'bin{b}|{p}|{g}' for b in range(4) for p in process_names
        for g in ('one','stock',*[f'factor{i}' for i in range(len(bundle['proxy'].kernel.nodes))])]
    if spec['family']=='gated_anchor':column_names.append('paired-European-anchor')
    if (seal.get('selected_spec_index')!=chosen or seal.get('selected_specification')!=spec
            or manifest.get('selected_spec_index')!=chosen or manifest.get('selected_specification')!=spec
            or not np.allclose(seal['weights'],weights,rtol=1e-11,atol=2e-12)
            or not np.array_equal(coefficients['selected_weights'],np.asarray(seal['weights']))
            or not np.array_equal(seal['fit']['fit_center'],center)
            or not np.array_equal(seal['fit']['fit_scale'],scale)
            or seal.get('basis_columns')!=column_names
            or seal['fit'].get('fit_intercept_applied_at_valuation') is not False):
        raise ValueError('Specification selection, all-spent refit or no-intercept seal changed')
    from ._pricing_rough_price_budget_trials import naive_european_trials
    return {'manifest':manifest,'seal':seal,'tables':{'spent_trials':frame,
        'naive_european_trials':naive_european_trials(old_arrays,conditional_replays),
        'spent_feature_checks':pd.DataFrame(feature_records),'conditional_probes':pd.DataFrame(probes)},
        'input_files':entries,'weights':weights}


def validate_design(design):
    _fixed_tolerances(design,required=True)
    expected={'version':'rough_price_budget_shared_control_v1','streams':STREAMS,
        'total_paths':49152,'estimator_order':['unchanged control','shared control'],
        'frozen_policy':'GH16 blend1','frozen_dual_control':'target 4 bins penalty0',
        'physical_steps':128,'exercise_intervals':64,'coefficient_fit_seeds':[1094005042,1095005042],
        'selected_specification':{'family':'gated_anchor','ridge':.01,'eligible_for_selection':True},
        'features':193,'bins':4,'coefficient_bound':2.,'left_stock_feature_scale':.05,
        'left_feature_clip':3.,'factor_normalization':'frozen selected-policy perdate center/scale',
        'column_order':'bin,process,gate; pairedEuropean anchor last',
        'selection_rule':'minimum mean heldout variance over both spent split directions; finalrefitall8192spentrows',
        'no_fresh_selection_or_fitting':True,'no_empirical_valuation_recentering':True,
        'no_regression_intercept_at_valuation':True,'both_endpoints_same_shared_shift':True,
        'gap_pointwise_unchanged':True,'gap_improvement_claimed':False,'shared_shift_expectation_zero':True,
        'upper_is_Rogers_max_after_shift':False,'continuous_European_price_used_to_center':False,
        'independent_reference_mean_used':False,'all_paths_retained':True,
        'no_new_paths_generated_before_design':True,'price_error_budget':'gap +1.96*(lower_priceSE+gapSE)',
        'price_allowance':'.001+1e-4*abs(V)','new_policy_fits':0,'new_dual_control_fits':0,
        'optimal_value_ready':False,'stock_delta_ready':False,'runtime_projection_only':True}
    if any(design.get(k)!=v for k,v in expected.items()):
        raise ValueError('Changed price-budget design, sample budget, frozen control or interpretation')
    if {s['seed'] for s in STREAMS}&set(design.get('forbidden_prior_seeds',[])):
        raise ValueError('Independent price streams overlap spent fitting or earlier valuation')


def collect_price_budget_campaign(folder):
    folder=Path(folder).resolve();output=folder.parents[1]
    manifest,entries=_manifest(folder,inventory=True)
    design_path=folder/'design_before_valuation.json';design=read(design_path);validate_design(design)
    if (any(manifest.get(k)!=v for k,v in design.items()) or manifest.get('complete') is not True
            or manifest.get('completed_paths')!=49152 or manifest.get('all_streams_retained') is not True
            or manifest.get('design_sha256')!=_sha(design_path)):
        raise ValueError('Incomplete independent price-budget campaign')
    prior=output/'rough_bellman_validation_v1/bellman_control_v1'
    prior_design=read(prior/'design_before_training.json')
    expected=set(prior_design['source_sha256'])|{'examples/rough_price_budget.py',
        'src/rvlab/experiments/_pricing_rough_price_budget.py',
        'src/rvlab/experiments/_pricing_rough_price_budget_controls.py'}
    if len(expected)!=52:raise ValueError('The original numerical dependency set changed')
    verify_sources(folder,design,expected);entries+=verify_inputs(design,output)
    required=set(prior_design['input_sha256'])|{
        *['rough_bellman_validation_v1/bellman_control_v1/'+n for n in
          ('manifest.json','design_before_training.json','selection_before_final.json')],
        *['rough_price_budget_v1/spent_shared_fit_v1/'+n for n in
          ('manifest.json','design_before_fitting.json','weights_before_final.json','all_fit_coefficients.npz',
           'spent_trials.csv','preservation_inventory.json')],
        *['rough_price_budget_v1/spent_terminal_anchor_v1/'+n for n in ('manifest.json','preservation_inventory.json')]}
    if set(design['input_sha256'])!=required:
        raise ValueError('A frozen spent fit, conditional replay or prior input was omitted')
    from ._pricing_rough_bellman_control_collection import (
        collect_control_campaign,validate_policy_diagnostics,_numerical_probes)
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    from ._pricing_rough_frozen_policy import _decode_models
    old=collect_control_campaign(prior);entries+=old['input_files']
    bundle=load_larger_policy_bundle(output/'rough_larger_policy_v1/32768_paths_two_starts')
    old_arrays=[_npz(prior/f'final_{i}_samples.npz') for i in range(2)]
    spent=validate_spent_fit(folder.parent/'spent_shared_fit_v1',output,bundle,old_arrays)
    entries+=spent['input_files'];seal=read(folder/'weights_before_final.json')
    if (seal!=spent['seal'] or _sha(folder/'weights_before_final.json')!=design.get('coefficient_sha256')
            or not _time(seal['sealed_at'])<_time(design['created_at'])):
        raise ValueError('Fresh valuation preceded or changed the training-only coefficient seal')
    forbidden=set(prior_design['forbidden_prior_seeds'])|{1092005042,1093005042,1094005042,1095005042}
    if set(design['forbidden_prior_seeds'])!=forbidden:
        raise ValueError('Price-budget seed firewall omitted spent or historical paths')
    # Keep the unsuccessful 32-index replay as provenance and verify it against
    # the complete later spent replay, never promote it to another final pool.
    probe=folder.parent/'spent_terminal_probe32_v1';_,probe_entries=_preserved(probe);entries+=probe_entries
    probe_design=read(probe/'design_before_replay.json')
    verify_sources(probe,probe_design,set(prior_design['source_sha256'])|{
        'examples/rough_price_budget_spent.py','src/rvlab/experiments/_pricing_rough_price_budget.py'})
    entries+=verify_inputs(probe_design,output);probe_manifest=read(probe/'manifest.json')
    if (probe_manifest.get('new_outer_paths')!=0 or probe_manifest.get('used_for_fresh_acceptance') is not False
            or probe_manifest.get('complete') is not True):raise ValueError('Spent runtime probe became acceptance evidence')
    small=_npz(probe/'stream_0_samples.npz')
    complete=_npz(folder.parent/'spent_terminal_anchor_v1/stream_1_samples.npz')
    index=small['absolute_path_indices']
    if not np.array_equal(index,np.arange(32)):raise ValueError('The spent probe index set changed')
    for name,value in small.items():
        if name=='absolute_path_indices':continue
        wanted=complete[name] if name=='physical_time_grid' else complete[name][index]
        if value.shape!=wanted.shape or not np.allclose(value,wanted,rtol=0.,atol=2e-12):
            raise ValueError('Selective replay differs from the complete original stream')
    selection=read(prior/'selection_before_final.json');controls=selection['controls']
    if selection['selected_control']!='target 4 bins penalty0' or list(controls).index(selection['selected_control'])!=3:
        raise ValueError('The selected dual control or its retained endpoint order changed')
    tail=read(output/'rough_tail_attribution_v1/joint_switch_v1/selection_before_final.json')
    proxy_path=output/'rough_bellman_validation_v1/spent_proxy_aligned_v2'
    proxy_seal=read(proxy_path/'proxy_before_fresh_paths.json')
    with np.load(proxy_path/proxy_seal['coefficient_archive'],allow_pickle=False) as z:
        models=(None,)+_decode_models(proxy_seal['models'],z)+(None,)
    streams=[];records=[];execution=[];primitive_probes=[];conditional_probes=[];checks=[]
    for i,spec in enumerate(STREAMS):
        record=read(folder/f'stream_{i}_manifest.json')
        expected_record={'index':i,**spec,'design_sha256':_sha(design_path),
            'coefficient_sha256':design['coefficient_sha256'],'source_sha256':design['source_sha256'],
            'sources_unchanged':True,'frozen_policy':'GH16 blend1','frozen_dual_control':'target 4 bins penalty0',
            'new_fitting_or_selection':False,'all_paths_retained':True,'gap_pointwise_unchanged':True,
            'empirical_valuation_recentering':False}
        if (any(record.get(k)!=v for k,v in expected_record.items())
                or set(record.get('sample_sha256',{}))!={'price','conditional','shift'}
                or not _time(design['created_at'])<_time(record['started_at'])<_time(record['completed_at'])
                or record.get('elapsed_seconds',0)<=0):
            raise ValueError('A final stream changed its frozen identity, role, budget or chronology')
        arrays={}
        for name in ('price','conditional','shift'):
            path=folder/f'stream_{i}_{name}_samples.npz'
            if _sha(path)!=record['sample_sha256'][name]:raise ValueError('A final full-path archive changed')
            arrays[name]=_npz(path)
            schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays[name].items()}
            if schema!=record.get('array_schema',{}).get(name):raise ValueError('Final array schema differs from its sealed record')
        a=arrays['price'];n=spec['paths']
        result=validate_control_samples(a,controls,bundle,prior_design['prior_control'],prior_design['grid_control'],tail,paths=n)
        validate_policy_diagnostics({'diagnostics':record['old_diagnostics']},result,a)
        primitive_probes.extend({'seed':spec['seed'],**p} for p in _numerical_probes(a,bundle,models))
        cv=validate_conditional_european({**a,**arrays['conditional']},bundle,**spec)
        diagnostic=record.get('conditional_diagnostics',{})
        expected_diagnostic={'seed':spec['seed'],'full_stream_paths':n,'evaluated_indices':list(range(n)),
            'all_random_draws_consumed':True,'same_batch_size':bundle['proxy'].config.batch_size,
            'orthogonal_stock_noise_integrated':True,'both_estimators_target_same_discrete_european_value':True,
            'continuous_price_used_to_center':False,'empirical_valuation_mean_subtracted':False,
            'physical_steps':128,'frozen_exercise_policy_changed':False}
        if (any(diagnostic.get(k)!=v for k,v in expected_diagnostic.items())
                or not 0<=diagnostic.get('max_full_stream_stock_error',np.inf)<=2e-12
                or not 0<=diagnostic.get('max_full_stream_factor_error',np.inf)<=2e-13):
            raise ValueError('A final conditional replay changed its stream, law or centering diagnostics')
        if not np.allclose(arrays['conditional']['shared_control'],cv['shared_european_control'],rtol=0.,atol=2e-12):
            raise ValueError('Final European anchor changed its centering identity')
        conditional_probes.extend({'role':'independent final','seed':spec['seed'],**p} for p in cv['probes'])
        basis,metadata=reconstruct_predictable_basis(a,bundle)
        basis=np.column_stack((basis,cv['shared_european_control']))
        saved=arrays['shift']
        required_shift={'basis','unchanged_lower','unchanged_upper','shared_lower','shared_upper','shift'}
        if (set(saved)!=required_shift or saved['basis'].shape!=(n,193)
                or not np.allclose(saved['basis'],basis,rtol=0.,atol=2e-12)
                or not np.array_equal(saved['unchanged_lower'],a['candidate3_lower'])
                or not np.array_equal(saved['unchanged_upper'],a['candidate3_upper'])):
            raise ValueError('A final predictable feature, stopping mask or original endpoint changed')
        translated=validate_common_shift(saved['unchanged_lower'],saved['unchanged_upper'],basis,spent['weights'],
            shifted_lower=saved['shared_lower'],shifted_upper=saved['shared_upper'],saved_shift=saved['shift'])
        streams.append({'seed':spec['seed'],'values':{
            'unchanged control':{'lower':saved['unchanged_lower'],'upper':saved['unchanged_upper']},
            'shared control':{'lower':translated['lower'],'upper':translated['upper']}}})
        records.append(record)
        execution.append({**spec,'started_at':record['started_at'],'completed_at':record['completed_at'],
            'elapsed_seconds':record['elapsed_seconds'],'used_for_fitting_or_selection':False,
            'all_paths_retained':True,'physical_steps':128,'exercise_intervals':64})
        checks.append({'seed':spec['seed'],'paths':n,'features':basis.shape[1],
            'max_basis_reconstruction_error':float(np.max(abs(saved['basis']-basis))),
            'max_gap_change':float(np.max(abs(translated['upper']-translated['lower']-
                (saved['unchanged_upper']-saved['unchanged_lower'])))),
            'max_stock_replay_error':cv['max_stock_error'],'max_factor_replay_error':cv['max_factor_error'],
            'left_state_gates_only':True,'intercept_or_recentering_applied':False})
    if manifest.get('records')!=records:raise ValueError('A predeclared final stream was omitted or replaced')
    tables=summarize_price_streams(streams)
    tables.update(spent['tables'])
    from ._pricing_rough_price_budget_gap_diagnostics import gap_diagnostics
    primary=tables['price_comparison'].loc[tables['price_comparison'].primary].iloc[0].to_dict()
    tables.update(gap_diagnostics(
        [a['candidate3_upper']-a['candidate3_lower'] for a in old_arrays],
        [s['values']['unchanged control']['upper']-s['values']['unchanged control']['lower'] for s in streams],
        [1094005042,1095005042],[s['seed'] for s in streams],primary))
    original=price_budget([a['candidate3_lower'] for a in old_arrays],[a['candidate3_upper'] for a in old_arrays],
        label='stage20 selected control',role='spent development evidence')
    tables['spent_budget_diagnosis']=pd.DataFrame([diagnose_spent_budget(original)])
    tables['coefficients']=pd.DataFrame({'column':seal['basis_columns'],'coefficient':spent['weights'],
        'frozen_before_valuation':True,'fitted_on_reported_final_paths':False})
    tables['execution']=pd.DataFrame(execution)
    tables['primitive_probes']=pd.DataFrame(primitive_probes)
    tables['conditional_probes']=pd.concat((tables['conditional_probes'],pd.DataFrame(conditional_probes)),ignore_index=True)
    tables['path_identity_checks']=pd.DataFrame(checks)
    return {'tables':tables,'manifest':manifest,'input_files':entries}


def price_budget_study(context,*,seed=42,compute_figures=True):
    if isinstance(seed,bool) or not isinstance(seed,Integral) or seed!=42:
        raise ValueError('The declared price-budget research seed is 42')
    if not isinstance(compute_figures,bool):raise TypeError('compute_figures must be boolean')
    context.progress('Separating the stopping bracket from price uncertainty on three sealed independent streams')
    output=Path(context.output_dir).resolve()
    collected=collect_price_budget_campaign(output/'rough_price_budget_v1/shared_control_v1')
    tables=collected['tables'];unique={}
    for row in collected['input_files']:
        path=str(Path(row['path']).resolve());normalized={**row,'path':path}
        if path in unique and unique[path]!=normalized:raise ValueError('Conflicting price-budget evidence fingerprints')
        unique[path]=normalized
    entries=[unique[k] for k in sorted(unique)]
    names=('_pricing_rough_price_budget_collection.py','_pricing_rough_price_budget_samples.py',
        '_pricing_rough_price_budget_basis.py','_pricing_rough_price_budget_tables.py',
        '_pricing_rough_bellman_control_collection.py','_pricing_rough_bellman_control_samples.py',
        '_pricing_rough_bellman_control_tables.py','_pricing_rough_stopping_delta.py',
        '_pricing_rough_gap_time_stopping.py','_pricing_rough_policy_validation_stopping.py',
        '_pricing_rough_tail_samples.py','_pricing_rough_price_budget_trials.py',
        '_pricing_rough_price_budget_gap_diagnostics.py')
    sources={name:_sha(Path(__file__).with_name(name)) for name in names}
    fingerprint=hashlib.sha256(json.dumps({'version':'rough_price_budget_v1','files':entries,
        'analysis_sources':sources},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    tables['provenance']=pd.DataFrame(entries)
    tables['configuration']=pd.DataFrame([
        {'item':'Synthetic contract','value':'Spot100, strike100 put;60/365years; four-factor lift; explicit empty cash-dividend schedule'},
        {'item':'Fixed rough-Heston parameters','value':'v0=.04, theta=.045, kappa=1.5, vol_of_vol=.3, rho=-.6, hurst=.15'},
        {'item':'Fixed pricing problem','value':'GH16 blend1 policy; selected target4bin dual; same128 physical steps/64exercise dates'},
        {'item':'Common control','value':'192 bounded predictable stopped increments plus an exactly zero-mean paired European anchor;193 coefficients fixed on spent data'},
        {'item':'Independent valuation','value':'49152 paths: three seeds of16384; no fitting, selection or adaptive path count'},
        {'item':'Price allowance','value':'.001+1e-4|V|; budget gap+1.96*(lowerSE+gapSE); unchanged'},
        {'item':'Interpretation','value':'The control shifts both endpoints, preserves their pathwise gap and their expectation bounds; shifted upper is not itself a Rogers maximum'},
        {'item':'Scope','value':'One finite-lift discrete stopping problem; no optimal value, Greek, kernel or real-SPY readiness certificate'}])
    manifest={'backend':'rough_heston_price_budget','study_version':'rough_price_budget_v1',
        'profile_fingerprint':fingerprint,'analysis_sources':sources,'input_files':entries,
        'primary_complete':True,'historical_replay':False,'current_engine_execution':False,
        'execution_mode':'read-only independent reconstruction of separately sealed precision experiments',
        'campaign':collected['manifest'],'primary_paths':49152,'seed_count':3,
        'primary_policy_fits':0,'primary_dual_control_fits':0,'spent_control_regressions':51,
        'new_control_stock_delta_validated':False,'optimal_delta_ready':False,
        'american_pricing_ready':False,'stock_delta_ready':False,
        'all_paths_retained':True,'gap_pointwise_unchanged':True,
        'policy_and_dual_slack_separately_identified':False,
        'standard_error':'path-count weighted independent seed means; within-seed variances; no recentering',
        'price_atol':.001,'price_rtol':1e-4,'delta_atol':.002,'delta_rtol':.005,
        'confidence_multiplier':1.96,'spread_fraction':.1,'tolerances_changed':False,
        'limitations':['The unchanged gap contains policy error and dual slack; these are not separately identified.',
            'A common zero-mean estimator shift reduces noise without improving the stopping rule.',
            'Sampling intervals describe this finite-law experiment, not kernel or continuous-exercise bias.',
            'The new common control is not a validated stock-delta estimator.',
            'Missing independent SPY spot and historically available dividend expectations remain blockers.']}
    result={'tables':tables,'figures':{},'manifest':manifest}
    if compute_figures:
        from ._pricing_rough_price_budget_reporting import add_price_budget_reporting
        result=add_price_budget_reporting(result)
    return result

