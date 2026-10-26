"""Read-only source, chronology, primitive and selection audit of Bellman controls."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ._pricing_rough_stopping_delta import _manifest,_sha
from ._pricing_rough_policy_validation import _entry
from ._pricing_rough_gap_time_stopping import _preserved
from ._pricing_rough_bellman_control_samples import BASELINE,validate_specs,validate_control_samples
from ._pricing_rough_bellman_control_tables import row,summarize_control_streams

ROOT=Path(__file__).resolve().parents[3]
TRAIN=1092005042; SELECT=1093005042; FINALS=(1094005042,1095005042)
TARGETS=['naive old-q grid8 constant upper','old-q grid8 q-endpoint-zero',
    'Bellman-q grid4 q-endpoint-zero','Bellman-q grid8 q-endpoint-zero']


def read(path):return json.loads(Path(path).read_text())
def time(value):return datetime.fromisoformat(value)


def validate_design(design):
    expected={'version':'rough_bellman_control_v1',
        'training':{'seed':TRAIN,'paths':2048,'role':'bounded control coefficient fitting'},
        'selection':{'seed':SELECT,'paths':2048,'role':'control selection only'},
        'final_streams':[{'seed':seed,'paths':4096} for seed in FINALS],
        'targets':TARGETS,'bins':[1,4],'penalties':[0.,1e-4],'coefficient_sum_per_bin_max':1.,
        'extra_shrink_candidates':False,'controls_if_all_fits_complete':5,'frozen_policy':'GH16 blend1',
        'operator_steps':128,'physical_steps':128,'exercise_intervals':64,
        'proxy_fit_seed':1077005042,'proxy_diagnostic_seed':1078005042,'proxy_correction_bound':.05,
        'proxy_ridge':.001,'proxy_degree':1,'proxy_factor_clip':3.,
        'pure_targets_diagnostic_only':True,'no_policy_refit_or_reselection':True,
        'all_paths_retained':True,'all_exercise_dates_retained':True,'nodal_european_grid':'original129 physical grid',
        'all_target_primitives_retained':True,'price_allowance':'.001+1e-4*abs(V)',
        'delta_allowance':'.002+.005*abs(delta)','confidence_multiplier':1.96,'spread_fraction':.1,
        'tolerances_changed':False,'optimal_value_ready':False,'new_control_stock_delta_validated':False,
        'no_new_paths_before_design':True,
        'selection_rule':'max positive paired gap-reduction95percentLCB; unchanged fallback score0',
        'designated_challenger_rule':'highest nonbaseline selectionLCB regardless of fallback acceptance'}
    if any(design.get(k)!=v for k,v in expected.items()):
        raise ValueError('Changed Bellman control numerical design, budgets, rule or acceptance tolerances')
    if {TRAIN,SELECT,*FINALS}&set(design['forbidden_prior_seeds']):
        raise ValueError('Fresh control paths overlap a declared prior seed')


def validate_selection(stream,controls,seal,frame):
    expected=[]; selected=BASELINE; score=0.; challenger=None; challenger_score=-np.inf
    base=stream['values'][BASELINE]; basegap=base['upper']-base['lower']
    for i,name in enumerate(controls):
        v=stream['values'][name]; stats=row([basegap-v['upper']+v['lower']]); low=stats['interval_low']
        expected.append({'control':name,'paired_gap_gain':stats['estimate'],'gap_gain_se':stats['standard_error'],
            'gain_low':low,'gain_high':stats['interval_high'],'selection_score':low,
            'selection_seed':SELECT,'final_paths_used_for_selection':False})
        if low>score:selected,score=name,low
        if i and low>challenger_score:challenger,challenger_score=name,low
    for value in expected:value.update(selected_control=value['control']==selected,designated_challenger=value['control']==challenger)
    if (seal.get('selected_control')!=selected or seal.get('designated_challenger')!=challenger
            or list(frame.columns)!=list(expected[0]) or len(frame)!=len(expected)):
        raise ValueError('Control selection differs from the pre-final selection-only paired LCB')
    for actual,wanted in zip(frame.to_dict('records'),expected):
        for key,v in wanted.items():
            a=actual[key]
            if isinstance(v,float):
                if not np.isclose(a,v,atol=2e-13,rtol=1e-11):raise ValueError('Selection statistic changed')
            elif a!=v:raise ValueError('Selection role, candidate order or seed changed')
    return pd.DataFrame(expected)


def _numerical_probes(arrays,bundle,models):
    """All-path primitive identities above plus bounded fresh conditional re-evaluation.

    These explicit probes verify saved conditional means against the frozen
    transition implementation; they are not claimed as an exhaustive second
    numerical integration of every conditional state.
    """
    from dataclasses import replace
    from ._pricing_rough_bellman_control_aligned import aligned_node_proxy,target_nodal_premiums
    from ._pricing_rough_exercise_block import exercise_block_law
    from ._pricing_rough_node_refinement_fast import refined_stock_knots,signed_shared_expectations
    from ._pricing_rough_local_control import _linear_coefficients
    from rvlab.pricing.rough_estimators import _AffineSurrogate
    from rvlab.pricing.rough_stopping_fourier import _affine_values
    proxy=bundle['proxy']; selected=bundle['selected_policy']; grid=arrays['actual_time_grid']; times=arrays['times']
    affine=_AffineSurrogate(proxy.params,proxy.market,proxy.kernel,np.array([proxy.strike]),np.array([proxy.maturity]),np.array([proxy.is_call]),grid,proxy.control_config)
    node_proxy=aligned_node_proxy(proxy,affine); nodes=np.r_[0.,refined_stock_knots(proxy.config.stock_knots,8)]
    from ._pricing_rough_bellman_policy import BellmanEvaluator
    evaluator=BellmanEvaluator(proxy,selected,orders=(16,))
    rows=[]
    for j in (16,32,48,63):
        indices=np.array([0,1]); a,b=np.searchsorted(grid,times[j:j+2]); factors=arrays['factors'][indices,j]
        law=exercise_block_law(proxy,factors,times=grid[a:b+1]); new=law['new_factors']; p=law['probabilities']
        values=target_nodal_premiums(proxy,selected,models,j+1,new,fine_nodes=nodes,node_proxy=node_proxy)
        coefficients=_linear_coefficients(nodes,values)
        mean=np.log(arrays['stocks'][indices,j,0])[:,None,None]+law['log_stock_shifts']; clock=law['orthogonal']
        ex=signed_shared_expectations(mean,clock,coefficients,strike=proxy.strike,knots=nodes[1:])
        ep=np.sum(ex.mean(axis=-1)*p[...,None],axis=1)
        ee=np.sum(_affine_values(affine,float(times[j+1]),new,mean[...,None],clock[...,None])[...,0].mean(axis=2)*p,axis=1)
        actual_values=target_nodal_premiums(proxy,selected,models,j+1,arrays['factors'][indices,j+1],fine_nodes=nodes,node_proxy=node_proxy)
        actual=np.stack([np.interp(arrays['stocks'][idx,j+1,0]/proxy.strike,nodes,actual_values[k,target])
            for k,idx in enumerate(indices) for target in range(4)]).reshape(2,4)
        errors={'expected_european_error':np.max(abs(ee-arrays['target_expected_european'][indices,j+1])),
            'expected_premium_error':np.max(abs(ep-arrays['target_expected_premiums'][indices,j+1])),
            'actual_premium_error':np.max(abs(actual-arrays['target_actual_premiums'][indices,j+1]))}
        active_indices=np.flatnonzero(arrays['gh16_computed_mask'][:,j])[:2]
        errors['frozen_policy_continuation_error']=0.
        if len(active_indices):
            c=evaluator.conditional_expectations(j,arrays['stocks'][active_indices,j,0],arrays['factors'][active_indices,j])['continuation'][:,0]
            errors['frozen_policy_continuation_error']=float(np.max(abs(c-arrays['gh16_continuation'][active_indices,j])))
        if any(v>2e-11 for v in errors.values()):raise ValueError('A saved aligned conditional primitive or frozen policy continuation failed independent probe reconstruction')
        rows.append({'left_exercise_index':j,'path_indices':'0,1','policy_probe_states':len(active_indices),**errors,'all_path_accumulation_validated_separately':True,
            'exhaustive_conditional_mean_recomputation':False})
    return rows




def validate_proxy_lineage(proxy_path,output,main_created_at):
    proxy_path=Path(proxy_path); output=Path(output).resolve()
    manifest=read(proxy_path/'manifest.json'); seal=read(proxy_path/'proxy_before_fresh_paths.json')
    design=read(proxy_path/'design_before_fit.json')
    expected={*['rough_larger_policy_v1/32768_paths_two_starts/'+n for n in
        ('frozen_policy_before_final.json','frozen_policy_coefficients.npz')],
        *['rough_bellman_policy_v1/one_step_v1/'+n for n in
        ('selection_before_final.json','manifest.json','final_0_samples.npz','final_0_manifest.json',
            'final_1_samples.npz','final_1_manifest.json')]}
    if (set(manifest.get('input_sha256',{}))!=expected
            or seal.get('input_sha256')!=manifest['input_sha256']
            or design.get('input_sha256')!=manifest['input_sha256']
            or seal.get('source_sha256')!=manifest.get('source_sha256')
            or design.get('source_sha256')!=manifest.get('source_sha256')
            or seal.get('created_at')!=manifest.get('created_at') or design.get('created_at')!=manifest.get('created_at')
            or not time(seal['created_at'])<time(main_created_at)
            or manifest.get('fit_seed')!=1077005042 or manifest.get('diagnostic_seed')!=1078005042
            or manifest.get('new_outer_paths')!=0 or manifest.get('complete') is not True
            or manifest.get('regression_row_clip_or_deletion') is not False):
        raise ValueError('Frozen proxy training lineage, dates or spent-only role changed')
    entries=[]
    for name,digest in manifest['input_sha256'].items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or _sha(path)!=digest:
            raise ValueError('A nested spent proxy training/diagnostic input changed')
        entries.append(_entry(path))
    for name,digest in manifest['source_sha256'].items():
        if _sha(ROOT/name)!=digest or _sha(proxy_path/'source'/name)!=digest:
            raise ValueError('Nested frozen proxy numerical source changed')
    for i in range(2):
        record=read(output/f'rough_bellman_policy_v1/one_step_v1/final_{i}_manifest.json')
        if record.get('seed')!=(1077005042,1078005042)[i] or record.get('paths')!=4096 or record.get('sample_sha256')!=_sha(
                output/f'rough_bellman_policy_v1/one_step_v1/final_{i}_samples.npz'):
            raise ValueError('Nested spent label stream identity or complete path count changed')
    return entries


def validate_policy_diagnostics(record,result,arrays):
    diagnostic=record.get('diagnostics',{}).get('fixed_policy',{})
    mask=result['computed_mask']
    expected={'operator_steps':128,'outer_steps':128,'computed_states':int(mask.sum()),
        'only_alive_ITM_evaluated':True,'all_paths_retained':True,'all_bumps_recompute_continuation_and_stops':True}
    if any(diagnostic.get(k)!=v for k,v in expected.items()) or not np.array_equal(diagnostic.get('operator_grid'),arrays['actual_time_grid']):
        raise ValueError('Frozen policy diagnostics changed operator law, mask size or retained paths')
    rows=diagnostic.get('dates',[]); dates=np.flatnonzero(mask.sum(axis=0)).tolist()
    if [r.get('exercise_index') for r in rows]!=dates:
        raise ValueError('Frozen policy diagnostic date coverage changed')
    for r,j in zip(rows,dates):
        flags={'states':int(mask[:,j].sum()),'orders':[16],'physical_increments':2,
            'terminal_exact_payoff':j==63,'only_premium_capped':True,
            'actual_future_state_used':False,'all_factor_branches_and_stock_orderings':True}
        if (any(r.get(k)!=v for k,v in flags.items()) or not 0<=r.get('max_probability_error',-1)<=1e-12
                or np.asarray(r.get('max_cv_direct_difference')).shape!=(1,)
                or not np.isfinite(r['max_cv_direct_difference']).all()
                or np.any(np.asarray(r['max_cv_direct_difference'])<0)):
            raise ValueError('Frozen policy per-date diagnostics changed conditional law or nonanticipation')


def collect_control_campaign(folder):
    folder=Path(folder).resolve(); output=folder.parents[1]
    manifest,entries=_manifest(folder,inventory=True)
    design_path=folder/'design_before_training.json'; design=read(design_path); validate_design(design)
    if (any(manifest.get(k)!=v for k,v in design.items()) or manifest.get('complete') is not True
            or manifest.get('completed_final_paths')!=8192):raise ValueError('Incomplete or changed control campaign')
    tail_path=output/'rough_tail_attribution_v1/joint_switch_v1'; tail=read(tail_path/'selection_before_final.json')
    old_design=read(tail_path/'design_before_training.json')
    expected_sources=set(old_design['source_sha256'])|{'examples/rough_bellman_control.py',
        *['src/rvlab/experiments/'+n for n in ('_pricing_rough_bellman_control.py','_pricing_rough_bellman_control_aligned.py',
            '_pricing_rough_bellman_delta.py','_pricing_rough_bellman_policy.py','_pricing_rough_stock_delta.py')]}
    if len(expected_sources)!=49 or set(design.get('source_sha256',{}))!=expected_sources:raise ValueError('A numerical control dependency is omitted/replaced')
    for name,digest in design['source_sha256'].items():
        if _sha(ROOT/name)!=digest or _sha(folder/'source'/name)!=digest:raise ValueError('Frozen control numerical source changed')
    required=set(old_design['input_sha256'])|{
        'rough_tail_attribution_v1/joint_switch_v1/selection_before_final.json',
        *['rough_bellman_policy_v1/one_step_v1/'+n for n in ('manifest.json','selection_before_final.json')],
        *['rough_bellman_validation_v1/spent_proxy_aligned_v2/'+n for n in ('proxy_before_fresh_paths.json','proxy_coefficients.npz','preservation_inventory.json')]}
    if set(design.get('input_sha256',{}))!=required:raise ValueError('A frozen proxy or prior input was omitted/replaced')
    for name,digest in design['input_sha256'].items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or _sha(path)!=digest:raise ValueError('Frozen control input changed')
        entries.append(_entry(path))
    proxy_path=folder.parent/'spent_proxy_aligned_v2'; _,proxy_entries=_preserved(proxy_path); entries+=proxy_entries; proxy_manifest=read(proxy_path/'manifest.json')
    entries+=validate_proxy_lineage(proxy_path,output,design['created_at'])
    if (proxy_manifest.get('new_outer_paths')!=0 or proxy_manifest.get('used_for_fresh_acceptance') is not False
            or proxy_manifest['benchmark']['final_budget_recommendation']!=4096):raise ValueError('Spent proxy evidence promoted to final acceptance')
    seal_proxy=read(proxy_path/'proxy_before_fresh_paths.json')
    if seal_proxy['model_dates']!=list(range(1,64)) or _sha(proxy_path/seal_proxy['coefficient_archive'])!=seal_proxy['coefficient_archive_sha256']:
        raise ValueError('Frozen control proxy model coefficients changed')
    from ._pricing_rough_frozen_policy import _decode_models
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    with np.load(proxy_path/seal_proxy['coefficient_archive'],allow_pickle=False) as z:models=(None,)+_decode_models(seal_proxy['models'],z)+(None,)
    bundle=load_larger_policy_bundle(output/'rough_larger_policy_v1/32768_paths_two_starts')
    if (set(bundle['manifest']['all_selection_training_seeds'])|{1043005042,1054005042,1055005042,1066005042,1077005042,1078005042,1089005042,1090005042,1091005042})-set(design['forbidden_prior_seeds']):
        raise ValueError('Control seed firewall omits prior fitted or final streams')
    if (design['prior_control']!=old_design['prior_control'] or design['grid_control']!=old_design['grid_control']
            or design['common_control']!=tail['selected_control']
            or read(output/'rough_bellman_policy_v1/one_step_v1/selection_before_final.json')['selected_policy']!='GH16 blend1'):
        raise ValueError('Control experiment changed the fixed policy or common martingale')
    fitted=read(folder/'controls_before_selection.json'); selection=read(folder/'selection_before_final.json')
    controls=fitted['controls']; trials=pd.read_csv(folder/'control_trials.csv',float_precision='round_trip')
    expected_trials=[f'target {b} bins penalty{p:g}' for b in (1,4) for p in (0.,1e-4)]
    if (trials.control.tolist()!=expected_trials or not trials.status.isin(['completed','optimizer_failure']).all()):raise ValueError('A predeclared optimizer attempt/failure was omitted')
    validate_specs(controls,set(trials.loc[trials.status.eq('completed'),'control']))
    if (fitted.get('training_seed')!=TRAIN or fitted.get('selection_paths_generated_before_seal') is not False
            or fitted.get('design_sha256')!=_sha(design_path) or selection.get('controls')!=controls
            or selection.get('control_order')!=list(controls) or selection.get('selection_seed')!=SELECT
            or selection.get('design_sha256')!=_sha(design_path) or selection.get('control_sha256')!=_sha(folder/'controls_before_selection.json')
            or selection.get('final_paths_generated_before_seal') is not False or selection.get('frozen_policy')!='GH16 blend1'):
        raise ValueError('Fitted controls or pre-final selection seal changed')
    execution=[]; probe_rows=[]; streams=[]; final_records=[]; selection_table=None
    for stage,seed,paths in [('training',TRAIN,2048),('selection',SELECT,2048),*[(f'final_{i}',s,4096) for i,s in enumerate(FINALS)]]:
        record=read(folder/f'{stage}_manifest.json'); sample=folder/f'{stage}_samples.npz'
        wanted={'seed':seed,'paths':paths,'source_sha256':design['source_sha256'],'sources_unchanged':True,
            'design_sha256':_sha(design_path),'sample_sha256':_sha(sample)}
        if stage=='training':wanted['control_sha256']=_sha(folder/'controls_before_selection.json')
        else:wanted['selection_sha256']=_sha(folder/'selection_before_final.json')
        if any(record.get(k)!=v for k,v in wanted.items()):raise ValueError('A control stage changed its source/seed/path/seal/sample identity')
        if not time(design['created_at'])<time(record['started_at'])<time(record['completed_at']) or record['elapsed_seconds']<=0:
            raise ValueError('Invalid stage chronology or runtime')
        if stage=='training' and not time(record['started_at'])<time(fitted['sealed_at'])<=time(record['completed_at']):raise ValueError('Coefficient seal precedes training')
        if stage=='selection' and not time(fitted['sealed_at'])<time(record['started_at'])<time(selection['sealed_at'])<=time(record['completed_at']):raise ValueError('Selection did not follow the coefficient seal')
        if stage.startswith('final'):
            if (record.get('index')!=int(stage[-1]) or record.get('control_order')!=list(controls)
                    or not time(selection['sealed_at'])<time(record['started_at'])):raise ValueError('Final paths preceded or changed selection')
            final_records.append(record)
        flags=record.get('diagnostics',{}).get('target_control',{})
        required_flags={'targets':TARGETS,'physical_increments_per_interval':2,'only_q_capped':True,'snell_premium_capped':False,
            'right_factor_reliability_inside_expectation':True,'all_paths_retained':True,'conditional_mixture_expectation_exact':True,
            'new_exercise_policy':False,'nodal_european_grid_matches_physical_grid':True,'nodal_european_grid_points':129,
            'actual_expected_primitives_saved':True,'proxy_correction_bound':.05}
        if any(flags.get(k)!=v for k,v in required_flags.items()) or flags.get('max_probability_error',1.)>1e-12:
            raise ValueError('Conditional target law/centering/scope diagnostics changed')
        with np.load(sample,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        result=validate_control_samples(arrays,controls,bundle,design['prior_control'],design['grid_control'],tail,paths=paths)
        result['seed']=seed
        validate_policy_diagnostics(record,result,arrays)
        for probe in _numerical_probes(arrays,bundle,models):probe_rows.append({'stage':stage,'seed':seed,**probe})
        if stage=='training':
            for i,name in enumerate(controls):
                if not i:continue
                trial=trials.loc[trials.control.eq(name)].iloc[0]; v=result['values'][name]
                objective=float(np.mean(v['upper']-v['lower'])+controls[name]['penalty']*np.sum(controls[name]['weights']))
                if not np.isclose(objective,trial.objective,atol=2e-9,rtol=1e-9):raise ValueError('Coefficient objective does not use its declared training paths')
        elif stage=='selection':
            selection_table=validate_selection(result,controls,selection,pd.read_csv(folder/'control_selection.csv',float_precision='round_trip'))
        else:streams.append(result)
        execution.append({'stage':stage,'seed':seed,'paths':paths,'started_at':record['started_at'],
            'completed_at':record['completed_at'],'elapsed_seconds':record['elapsed_seconds'],
            'computed_alive_ITM_states':int(result['computed_mask'].sum()),'used_for_control_fitting':stage=='training',
            'used_for_control_selection':stage=='selection','used_for_final_acceptance':stage.startswith('final')})
    if (manifest.get('records')!=final_records or manifest.get('selected_control')!=selection['selected_control']
            or manifest.get('designated_challenger')!=selection['designated_challenger']
            or manifest.get('control_order')!=list(controls) or manifest.get('selection_sha256')!=_sha(folder/'selection_before_final.json')
            or manifest.get('training_manifest_sha256')!=_sha(folder/'training_manifest.json')
            or manifest.get('selection_manifest_sha256')!=_sha(folder/'selection_manifest.json')):
        raise ValueError('Control completion changed/omitted predeclared final evidence')
    tables=summarize_control_streams(streams,selection,TARGETS)
    tables.update(gap_selection=selection_table,gap_trials=trials,
        gap_proxy_fit=pd.read_csv(proxy_path/'proxy_fit_diagnostics.csv',float_precision='round_trip'),
        gap_execution=pd.DataFrame(execution),gap_primitive_probes=pd.DataFrame(probe_rows),
        gap_grid_alignment=pd.read_csv(proxy_path/'mixed_grid_alignment.csv',float_precision='round_trip'))
    return {'tables':tables,'manifest':manifest,'input_files':entries}
