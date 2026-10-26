"""Independent paired-bump and time-law statistics for the shared193 control.

No pricing, simulation, fitting or result selection is performed. Every
declared stream and representation is retained. The delta target is the same
frozen original128-GH16 stopping rule, not the optimal American value.
"""
from numbers import Integral

import numpy as np
import pandas as pd

RELATIVE=np.array([-.005,-.0025,0.,.0025,.005])
BUMPS=(.005,.0025)
ESTIMATORS=('raw','stable','shared')


def moments(chunks):
    n=sum(map(len,chunks))
    mean=float(sum(a.sum() for a in chunks)/n)
    se=float(np.sqrt(sum(len(a)*np.var(a,ddof=1) for a in chunks))/n)
    return {'estimate':mean,'standard_error':se,'half_width':1.96*se,
        'interval_low':mean-1.96*se,'interval_high':mean+1.96*se}


def _validate(streams,expected):
    if not expected or len(streams)!=len(expected):
        raise ValueError('Every predeclared delta stream is required')
    seeds=[s.get('seed') for s in expected]
    if (len(set(seeds))!=len(seeds) or any(isinstance(s,bool) or not isinstance(s,Integral) for s in seeds)):
        raise ValueError('Distinct predeclared independent stream seeds required')
    for stream,spec in zip(streams,expected):
        if (stream.get('seed')!=spec['seed'] or stream.get('steps')!=spec['steps']
                or spec['steps'] not in (128,256) or isinstance(spec['paths'],bool)
                or not isinstance(spec['paths'],Integral) or spec['paths']<2
                or list(stream.get('representations',{}))!=spec['representations']
                or not spec['representations'] or len(set(spec['representations']))!=len(spec['representations'])
                or any(r not in ('original','micro') for r in spec['representations'])
                or not np.array_equal(stream.get('relative_spots'),RELATIVE)
                or not np.array_equal(stream.get('initial_spots'),100.*(1.+RELATIVE))):
            raise ValueError('A delta stream changed seed, resolution, stock bumps or representation')
        reference=None
        for rep,values in stream['representations'].items():
            if set(values)!=set(ESTIMATORS)|{'stops'}:
                raise ValueError('All three fixed estimators and own stopping indices are required')
            stops=np.asarray(values['stops'])
            if (stops.shape!=(spec['paths'],5) or stops.dtype.kind not in 'iu'
                    or np.any(stops<1) or np.any(stops>64)):
                raise ValueError('Each stock bump must retain its own bounded stopping index')
            for name in ESTIMATORS:
                a=np.asarray(values[name])
                if a.shape!=(spec['paths'],5) or not np.isfinite(a).all():
                    raise ValueError('Every finite path and stock bump must remain aligned')
            if reference is None:reference=values
            elif (not np.array_equal(values['stops'],reference['stops'])
                    or not np.array_equal(values['raw'],reference['raw'])
                    or not np.array_equal(values['stable'],reference['stable'])):
                raise ValueError('Changing the control representation must not change policy, raw payoff or stable comparator')


def delta(stream,representation,estimator,bump):
    value=stream['representations'][representation][estimator]
    a,b=(0,4) if bump==BUMPS[0] else (1,3)
    return (value[:,b]-value[:,a])/(200.*bump)


def _contrast(chunks,reference):
    row=moments(chunks);a=.002+.005*abs(reference)
    error=abs(row['estimate'])+row['half_width']
    return {**row,'delta_allowance':a,'change_plus_uncertainty':error,
        'tolerance_bias_resolved':error<=a,'zero_in_interval':row['interval_low']<=0<=row['interval_high']}


def summarize_shared_delta(streams,*,expected_streams):
    """Apply unchanged sampling, paired-bump and independent-time checks.

    expected_streams is fixed before valuation. It contains seed, steps, paths
    and ordered representations for every stream. Raw/control and representation
    shifts remain visible even when their tolerance-scale uncertainty is large.
    """
    _validate(streams,expected_streams)
    groups=sorted({(s['steps'],r) for s in streams for r in s['representations']})
    sampling=[];bumps=[];controls=[];representations=[];per_seed=[];exercise=[]
    for steps,rep in groups:
        group=[s for s in streams if s['steps']==steps and rep in s['representations']]
        n=sum(len(s['representations'][rep]['raw']) for s in group)
        flags={'physical_steps':steps,'representation':rep,'paths':n,'streams':len(group),
            'exercise_intervals':64,'policy_operator_steps':128,'same_frozen_rule':True,
            'optimal_delta_ready':False,'used_for_selection':False}
        for estimator in ESTIMATORS:
            for bump in BUMPS:
                chunks=[delta(s,rep,estimator,bump) for s in group]
                row={**flags,'estimator':estimator,'bump_relative':bump,**moments(chunks)}
                a=.002+.005*abs(row['estimate'])
                row.update(delta_allowance=a,sampling_pass=row['half_width']<=a)
                sampling.append(row)
                for s in group:
                    per_seed.append({**flags,'seed':s['seed'],'paths':len(s['representations'][rep]['raw']),
                        'estimator':estimator,'bump_relative':bump,**moments([delta(s,rep,estimator,bump)]),
                        'role':'all prespecified seed diagnostics retained'})
            fine=moments([delta(s,rep,estimator,BUMPS[1]) for s in group])['estimate']
            contrast=_contrast([delta(s,rep,estimator,BUMPS[1])-delta(s,rep,estimator,BUMPS[0]) for s in group],fine)
            bumps.append({**flags,'estimator':estimator,'coarse_bump':BUMPS[0],'fine_bump':BUMPS[1],
                **contrast,'bump_check_pass':contrast['tolerance_bias_resolved']})
        for bump in BUMPS:
            ref=moments([delta(s,rep,'shared',bump) for s in group])['estimate']
            for left,right in (('raw','shared'),('stable','shared'),('raw','stable')):
                controls.append({**flags,'comparison':f'{left} minus {right}','bump_relative':bump,
                    **_contrast([delta(s,rep,left,bump)-delta(s,rep,right,bump) for s in group],ref),
                    'marginal_interval':True})
        for s in group:
            stops=s['representations'][rep]['stops']
            for bump in BUMPS:
                a,b=(0,4) if bump==BUMPS[0] else (1,3)
                exercise.append({**flags,'seed':s['seed'],'paths':len(stops),'bump_relative':bump,
                    'different_stopping_fraction':float(np.mean(stops[:,a]!=stops[:,b])),
                    'all_bumps_recomputed_own_stops':True})
    for steps in sorted({s['steps'] for s in streams}):
        group=[s for s in streams if s['steps']==steps and set(s['representations'])=={'original','micro'}]
        if not group:continue
        for estimator in ESTIMATORS:
            for bump in BUMPS:
                ref=moments([delta(s,'micro',estimator,bump) for s in group])['estimate']
                representations.append({'physical_steps':steps,'paths':sum(len(s['representations']['micro']['raw']) for s in group),
                    'comparison':'micro minus original','estimator':estimator,'bump_relative':bump,
                    **_contrast([delta(s,'micro',estimator,bump)-delta(s,'original',estimator,bump) for s in group],ref),
                    'same_paths':True,'same_frozen_rule':True,'control_processes_are_identical':False,
                    'used_for_selection':False})
    frame=pd.DataFrame(sampling);time=[]
    for rep in ('original','micro'):
        resolutions=sorted({steps for steps,r in groups if r==rep})
        for coarse,fine in zip(resolutions[:-1],resolutions[1:]):
            for estimator in ESTIMATORS:
                for bump in BUMPS:
                    match=frame.loc[frame.representation.eq(rep)&frame.estimator.eq(estimator)&frame.bump_relative.eq(bump)]
                    a=match.loc[match.physical_steps.eq(coarse)].iloc[0]
                    b=match.loc[match.physical_steps.eq(fine)].iloc[0]
                    change=float(b.estimate-a.estimate);se=float(np.hypot(a.standard_error,b.standard_error))
                    allowance=.002+.005*abs(float(b.estimate));error=abs(change)+1.96*se
                    time.append({'representation':rep,'estimator':estimator,'bump_relative':bump,
                        'coarse_steps':coarse,'fine_steps':fine,'coarse_paths':int(a.paths),'fine_paths':int(b.paths),
                        'delta_change':change,'standard_error':se,'interval_low':change-1.96*se,
                        'interval_high':change+1.96*se,'change_plus_uncertainty':error,
                        'delta_allowance':allowance,'time_check_pass':error<=allowance,
                        'independent_resolutions':True,'policy_operator_steps':128,'same_frozen_rule':True,
                        'same_exercise_calendar':True,'optimal_delta_ready':False,'used_for_selection':False})
    combined=[]
    if (128,'original') in groups and (256,'micro') in groups:
        for estimator in ESTIMATORS:
            for bump in BUMPS:
                match=frame.loc[frame.estimator.eq(estimator)&frame.bump_relative.eq(bump)]
                a=match.loc[match.physical_steps.eq(128)&match.representation.eq('original')].iloc[0]
                b=match.loc[match.physical_steps.eq(256)&match.representation.eq('micro')].iloc[0]
                change=float(b.estimate-a.estimate);se=float(np.hypot(a.standard_error,b.standard_error))
                allowance=.002+.005*abs(float(b.estimate));error=abs(change)+1.96*se
                combined.append({'coarse_representation':'original','fine_representation':'micro',
                    'coarse_steps':128,'fine_steps':256,'estimator':estimator,'bump_relative':bump,
                    'coarse_paths':int(a.paths),'fine_paths':int(b.paths),'delta_change':change,
                    'standard_error':se,'interval_low':change-1.96*se,'interval_high':change+1.96*se,
                    'change_plus_uncertainty':error,'delta_allowance':allowance,
                    'combined_check_pass':error<=allowance,'independent_resolutions':True,
                    'same_frozen_rule':True,'control_representation_changed':True,
                    'physical_law_changed':True,'optimal_delta_ready':False,'used_for_selection':False})
    return {'delta_sampling':frame,'bump_refinement':pd.DataFrame(bumps),'raw_control':pd.DataFrame(controls),
        'representation_change':pd.DataFrame(representations),'time_refinement':pd.DataFrame(time),
        'combined_refinement':pd.DataFrame(combined),'delta_per_seed':pd.DataFrame(per_seed),
        'exercise_changes':pd.DataFrame(exercise)}


__all__=['summarize_shared_delta','delta','RELATIVE','BUMPS','ESTIMATORS']

