"""Independent predictable-compensator precision study for a frozen control.

No policy/control fitting, seed selection, path trimming or price-tolerance
change occurs. Numerical conditional integration targets the same expected
stopped contrast, rather than changing the old realized sample mean.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter

import numpy as np
import pandas as pd


def precision_repeat_specs():
    return [{'seed':510005042+1000000*i,'paths':2048} for i in range(8)]


def _sources():
    from ._pricing_rough_control_centering import _centering_sources
    names=('_pricing_rough_centering_precision.py','_pricing_rough_centering_precision_paths.py',
        '_pricing_rough_centering_precision_quadrature.py','_pricing_rough_centering_precision_statistics.py')
    return {**_centering_sources(),**{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}}


def _frozen(output_dir):
    from ._pricing_rough_frozen_policy import load_frozen_policy_bundle
    p=Path(output_dir)/'rough_martingale_centering_v1/frozen_joint_policy'
    frozen=load_frozen_policy_bundle(p)
    reference=Path(output_dir)/'rough_control_centering_v1/campaign_v1/collect/manifest.json'
    old=json.loads(reference.read_text())
    if old['profile_fingerprint']!='499c48aa4d689aa26d4ffb8413ccf4c9537b10e1b16a7bc4dd8921484231b3b6' or not old['primary_complete']:
        raise ValueError('Expected complete previous independent-path centering campaign')
    if old['sources']!={k:v for k,v in _sources().items() if k in old['sources']}:
        raise ValueError('Frozen historical numerical sources changed')
    if not old['frozen_reconstruction']['all_bit_identical']:
        raise ValueError('Historical frozen coefficient reconstruction failed')
    return frozen,old,reference


def _validate_design(repeats,forbidden):
    from ._pricing_rough_control_centering import _validated_repeats
    return _validated_repeats(repeats,forbidden)


def _calculate_repeat(frozen, spec, folder, levels):
    from ._pricing_rough_centering_precision_paths import full_microstate_paths,evaluate_precision_paths
    from ._pricing_rough_stopping_attribution import _save_sample_archive
    data=full_microstate_paths(frozen,**spec)
    result=evaluate_precision_paths(frozen,data,quadrature_nodes=levels)
    states=result['states'];counts=result['absolute_gate_counts']
    sample=_save_sample_archive(folder/f"seed_{spec['seed']}_samples.npz",**result['samples'],
        projected_by_level=result['projected_by_level'],absolute_gate_counts=counts,
        raw_gain=result['raw_gain'],controlled_gain=result['controlled_gain'])
    sim=data['simulation']
    trace=_save_sample_archive(folder/f"seed_{spec['seed']}_microstates.npz",times=sim.times,stocks=sim.spots,
        factors=sim.factors,signed_left_gates=data['signed_left_gates'],
        original_stops=data['original_stops'],selected_stops=data['selected_stops'],
        original_stop_times=data['original_stop_times'],selected_stop_times=data['selected_stop_times'])
    destination=folder/f"seed_{spec['seed']}_conditional_states.csv"
    states.to_csv(destination,index=False)
    raw=destination.read_bytes()
    return {'status':'completed',**spec,'sample_archives':[sample,trace],
        'state_table':{'path':str(destination),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
        'gated_microstates':len(states),'disagreeing_paths':int(np.count_nonzero(counts)),
        'actual_integration_steps':len(sim.times)-1,'exercise_dates':len(frozen['proxy'].exercise_times),
        'quadrature_nodes':list(levels),'new_policy_or_control_fit':False,'all_paths_retained':True,
        'maximum_first_refinement':0. if states.empty else float(states.first_refinement.max()),
        'maximum_last_refinement':0. if states.empty else float(states.last_refinement.max()),
        'maximum_tail_envelope':0. if states.empty else float(states.tail_envelope.max())}


def centering_precision_study(context, *, seed=42, repeat_specs=None,repeat_indices=None,
        reuse_checkpoints=True,compute_figures=True):
    from numbers import Integral
    import platform,scipy
    from ._pricing_rough_foundation import _checkpoint,_fingerprint,_jsonable
    from ._pricing_rough_stopping_attribution import _freeze_json,_verify_sample_archives
    from ._pricing_rough_centering_precision_statistics import summarize_precision_runs
    from ._pricing_rough_centering_statistics import summarize_independent_centering_runs
    if isinstance(seed,bool) or not isinstance(seed,Integral) or seed!=42:
        raise ValueError('This frozen historical investigation declares seed42 provenance')
    if not isinstance(reuse_checkpoints,bool) or not isinstance(compute_figures,bool):
        raise TypeError('reuse and figure flags must be boolean')
    frozen,old,reference=_frozen(context.output_dir)
    forbidden=set(old['forbidden_seeds'])|{r['seed'] for r in old['repeats']}|{890005042,890105042,430005042}
    repeats=_validate_design(precision_repeat_specs() if repeat_specs is None else repeat_specs,forbidden)
    if repeat_indices is not None and (not isinstance(repeat_indices,(list,tuple)) or len(set(repeat_indices))!=len(repeat_indices)
        or any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<len(repeats) for i in repeat_indices)):
        raise ValueError('repeat indices must be distinct valid integers')
    indices=set(range(len(repeats)) if repeat_indices is None else repeat_indices)
    sources=_sources();levels=(32,64,128)
    common={'study_version':'rough_centering_precision_v1','seed':42,'repeats':repeats,
        'forbidden_seeds':sorted(forbidden),'sources':sources,'quadrature_nodes':levels,'normal_limit':12.,
        'frozen_identity':old['frozen_identity'],'previous_centering_profile':old['profile_fingerprint'],
        'previous_centering_manifest_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'price_allowance':old['frozen_reconstruction']['price_allowance'],'acceptance':old['acceptance'],
        'primary_quantity':'projected_fine','estimator':'predictable_compensator_independent_conditional_integration',
        'conditioning':'all nonzero signed LEFT gates at every actual integration step',
        'primary_paths_fixed_before_outcomes':True,'negative_control_center_error':'b_j = sign * .01 * signed_LEFT_gate_j',
        'negative_control_shift_per_gated_step':.01,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__}}
    fingerprint=_fingerprint(common);folder=Path(context.output_dir)/'rough_centering_precision_v1'/fingerprint[:16]
    if not reuse_checkpoints:folder/='attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=True)
    seal=_freeze_json(folder/'design_before_fresh_valuation.json',common)
    snapshots=[]
    for name,digest in sources.items():
        src=Path(__file__).with_name(name)
        if not src.exists():src=Path(__file__).parents[1]/'pricing'/name
        if not src.exists():continue
        raw=src.read_bytes();actual=hashlib.sha256(raw).hexdigest()
        dest=folder/'source'/actual/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Changed source snapshot')
        if not dest.exists():dest.write_bytes(raw)
        snapshots.append({'source':str(src),'snapshot':str(dest),'sha256':actual,'bytes':len(raw)})
    records=[];provenance=[];failures=[];runs=[];negative=[];gain_runs=[];state_summaries=[]
    for index,spec in enumerate(repeats):
        if index not in indices:continue
        context.progress(f"Independent conditional-centering precision stream {index+1}/{len(repeats)} seed={spec['seed']}")
        def calculate(spec=spec):
            if _sources()!=sources:raise RuntimeError('Numerical sources changed before valuation')
            from rvlab.pricing import PricingError
            try:
                value=_calculate_repeat(frozen,spec,folder,levels)
            except (FloatingPointError,OverflowError) as exc:
                raise PricingError('Conditional-integration numerical failure: '+str(exc)) from exc
            if _sources()!=sources:raise RuntimeError('Numerical sources changed during valuation')
            return value
        record,origin=_checkpoint(folder,f'repeat_{index:02d}',{**common,'index':index},calculate,reuse=reuse_checkpoints)
        _verify_sample_archives(record);provenance.append(origin)
        records.append({'index':index,**spec,'status':record['status'],'elapsed_seconds':origin['original_elapsed_seconds'],
            'reused':origin['reused'],'checkpoint':origin['checkpoint'],
            **{k:record.get(k) for k in ('gated_microstates','disagreeing_paths','actual_integration_steps','exercise_dates',
                'maximum_first_refinement','maximum_last_refinement','maximum_tail_envelope')}})
        if record['status']!='completed':
            failures.append({'index':index,'error':record.get('error')});continue
        state=record['state_table'];payload=Path(state['path']).read_bytes()
        if len(payload)!=state['bytes'] or hashlib.sha256(payload).hexdigest()!=state['sha256']:
            raise ValueError('Changed saved conditional-state table')
        with np.load(record['sample_archives'][0]['path'],allow_pickle=False) as z:
            samples={k:z[k].copy() for k in ('raw_contrast','projected_coarse','projected_fine','quadrature_error','tail_error','arithmetic_error')}
            counts=z['absolute_gate_counts'].copy()
            if any(a.shape!=(spec['paths'],) or not np.isfinite(a).all() for a in samples.values()):
                raise ValueError('Saved primary samples do not match the declared path count')
            if counts.shape!=(spec['paths'],) or not np.isfinite(counts).all() or np.any(counts<0):
                raise ValueError('Saved gate counts do not align with every declared path')
            if any(z[k].shape!=(spec['paths'],) or not np.isfinite(z[k]).all() for k in ('raw_gain','controlled_gain')):
                raise ValueError('Saved policy gains do not match the declared path count')
            gain_runs.append({'seed':spec['seed'],'samples':{'raw_gain':z['raw_gain'].copy(),'controlled_gain':z['controlled_gain'].copy()}})
        runs.append({'seed':spec['seed'],'samples':samples})
        for sign in (-1,1):
            changed={k:v.copy() for k,v in samples.items()}
            for k in ('raw_contrast','projected_coarse','projected_fine'):changed[k]-=sign*.01*counts
            negative.append((sign,{'seed':spec['seed'],'samples':changed}))
    statistics=summarize_precision_runs(runs,price_allowance=common['price_allowance'],forbidden_seeds=forbidden) if runs else None
    tables={name:statistics[name] if statistics is not None else pd.DataFrame() for name in ('per_seed','pooled','numerical_budget')}
    controls=[]
    if negative:
        for sign in (-1,1):
            checked=summarize_precision_runs([run for s,run in negative if s==sign],price_allowance=common['price_allowance'],forbidden_seeds=forbidden)
            row=checked['numerical_budget'].loc[lambda t:t.scope=='pooled'].iloc[0].to_dict()
            fine=checked['pooled'].loc[lambda t:t.quantity=='projected_fine'].iloc[0]
            controls.append({'injected_adaptive_center_coefficient':sign*.01,
                'mechanism':'center error b_j=coefficient*signed_LEFT_gate_j; contrast shifts -coefficient*sum_abs_gates',
                'same_path_artificial_transformation':True,
                'estimate':fine.estimate,'standard_error':fine.standard_error,**row})
    tables['negative_controls']=pd.DataFrame(controls)
    tables['gains']=summarize_independent_centering_runs(gain_runs,forbidden_seeds=forbidden)['pooled'] if gain_runs else pd.DataFrame()
    tables['repeats']=pd.DataFrame(records);tables['provenance']=pd.DataFrame(provenance);tables['failures']=pd.DataFrame(failures)
    previous=pd.read_csv(reference.parent/'pooled.csv',float_precision='round_trip')
    tables['previous_comparison']=previous.loc[previous.quantity=='contrast'].copy()
    v=previous.loc[previous.quantity=='contrast'].iloc[0]
    tables['planning']=pd.DataFrame([{'method':'plain iid replication','previous_paths':int(v.paths),
        'previous_standard_error':v.standard_error,'target_half_width':common['price_allowance'],
        'estimated_paths':int(np.ceil(v.paths*(1.96*v.standard_error/common['price_allowance'])**2)),
        'planning_only_not_executed':True}])
    tables['configuration']=pd.DataFrame([{'item':'fresh design','value':f'{len(repeats)} streams; {sum(x["paths"] for x in repeats)} paths; declared before outcomes'},
        {'item':'policies/control','value':'exact frozen case05 functions; no new fitting or selection'},
        {'item':'conditional integration','value':'all six branches; Gaussian Legendre 32/64/128 with explicit breakpoints and tails'},
        {'item':'scope','value':'same expected raw/control gain contrast; finite kernel/weak law/exercise calendar only'}])
    complete=len(runs)==len(repeats) and not failures
    manifest={**_jsonable(common),'backend':'rough_heston_centering_precision','profile_fingerprint':fingerprint,
        'checkpoint_dir':str(folder),'design_seal':seal,'source_snapshots':snapshots,'primary_complete':complete,
        'planned_streams':len(repeats),'completed_streams':len(runs),
        'source_elapsed_seconds':sum(x['original_elapsed_seconds'] for x in provenance),
        'checkpoint_replay':bool(provenance) and all(x['reused'] for x in provenance),
        'current_engine_execution':True,'historical_replay':False,
        'statistics':{} if statistics is None else statistics['metadata'],
        'policy_optimizer_calls':0,'dual_optimizer_calls':0,'calibration_optimizer_calls':0,
        'new_policy_or_control_selection':False,'projection_mean_assumed_zero':False,
        'raw_sample_interval_shrunk':False,'tolerances_changed':False,'american_pricing_ready':False,'stock_delta_ready':False,
        'limitations':['Conditional and raw estimators share an expectation, not individual sample values.',
            'Numerical endpoint intervals include empirical refinement/tail/arithmetic diagnostics, not a rigorous global integration bound.',
            'This addresses one frozen stopped-control expected contrast, not American pricing, delta, model discretization or historical state accuracy.']}
    result={'tables':tables,'figures':{},'manifest':manifest}
    if compute_figures:
        from ._pricing_rough_centering_precision_reporting import add_precision_reporting
        result=add_precision_reporting(result,output_dir=context.output_dir)
    return result
