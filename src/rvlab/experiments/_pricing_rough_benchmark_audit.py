"""Independent fixed-budget statistics for generic rough stopping benchmarks.

This analysis never fits a rule or selects a control. Bounds concern each
declared finite transition law and exercise calendar. Refinements compare
lower-policy expectations; small changes alone do not bound optimal prices.
"""
from numbers import Integral
import math

import numpy as np
import pandas as pd

CONTROLS=('raw','european','european_premium')
RELATIVE=np.array([-.005,-.0025,0.,.0025,.005])
BUMPS=(.005,.0025)


def _close(actual,expected,label,atol=2e-13):
    if np.shape(actual)!=np.shape(expected) or not np.allclose(actual,expected,atol=atol,rtol=0.):
        raise ValueError(label)


def moments(chunks):
    """Mean and within-stream variance, retaining all declared independent paths."""
    n=sum(len(x) for x in chunks)
    if n<2 or any(len(x)<2 or np.ndim(x)!=1 or not np.isfinite(x).all() for x in chunks):
        raise ValueError('Finite one-dimensional independent stream samples required')
    mean=float(sum(x.sum() for x in chunks)/n)
    se=float(np.sqrt(sum(len(x)*np.var(x,ddof=1) for x in chunks))/n)
    return dict(paths=n,estimate=mean,standard_error=se,half_width=1.96*se,
        interval_low=mean-1.96*se,interval_high=mean+1.96*se)


def _contrast(chunks,reference,quantity):
    row=moments(chunks)
    allowance=(.001+1e-4*abs(reference) if quantity=='price' else .002+.005*abs(reference))
    error=abs(row['estimate'])+row['half_width']
    return {**row,'allowance':allowance,'change_plus_uncertainty':error,
        'tolerance_resolved':error<=allowance,
        'zero_in_interval':row['interval_low']<=0<=row['interval_high']}


def delta(lower,bump):
    if bump not in BUMPS:raise ValueError('Only predeclared relative stock bumps are supported')
    a,b=(0,4) if bump==BUMPS[0] else (1,3)
    return (lower[:,b]-lower[:,a])/(200.*bump)


def _validate_streams(streams,specs):
    if not specs or len(streams)!=len(specs):raise ValueError('Every predeclared stream, including failures, is required')
    seeds=[x['seed'] for x in specs]
    if len(set(seeds))!=len(seeds) or any(isinstance(x,bool) or not isinstance(x,Integral) or x<0 for x in seeds):
        raise ValueError('Distinct nonnegative valuation seeds required')
    if set(seeds)&{x['training_seed'] for x in specs}:
        raise ValueError('All valuation streams must be independent of every training stream')
    by_trial={}
    keys=('trial_id','seed','training_seed','paths','steps','exercise_steps','factors','strike','maturity')
    for stream,spec in zip(streams,specs):
        if any(stream.get(k)!=spec.get(k) for k in keys):raise ValueError('A benchmark stream changed identity or numerical settings')
        if (isinstance(spec['paths'],bool) or not isinstance(spec['paths'],Integral) or spec['paths']<2
                or spec['seed']==spec['training_seed'] or not 0<spec['maturity']<=1.
                or spec['strike']<=0 or spec['steps']<spec['exercise_steps']):
            raise ValueError('Invalid fixed benchmark budget or independent seed declaration')
        if stream.get('failure'):
            if stream.get('controls'):raise ValueError('Failed streams cannot silently supply partial acceptance samples')
        else:
            if not isinstance(stream.get('policy_fingerprint'),str) or not stream['policy_fingerprint']:
                raise ValueError('Every complete stream must identify its sealed stopping policy')
            if not np.array_equal(stream.get('initial_spots'),100.*(1+RELATIVE)):
                raise ValueError('All five fixed CRN initial stocks are required')
            stops=np.asarray(stream.get('stops'))
            if (stops.shape!=(spec['paths'],5) or stops.dtype.kind not in 'iu'
                    or np.any(stops<0) or np.any(stops>spec['exercise_steps'])
                    or set(stream.get('controls',{}))!=set(CONTROLS)):
                raise ValueError('Own stopping dates and all three controls are required')
            for control,value in stream['controls'].items():
                if set(value)!= {'lower','upper'}:raise ValueError('Explicit lower and genuine dual upper samples required')
                lo,hi=np.asarray(value['lower']),np.asarray(value['upper'])
                if (lo.shape!=(spec['paths'],5) or hi.shape!=lo.shape
                        or not np.isfinite(lo).all() or not np.isfinite(hi).all()
                        or np.any(hi-lo< -2e-13)):
                    raise ValueError('Incomplete finite pathwise-ordered bound samples')
        by_trial.setdefault(spec['trial_id'],[]).append((stream,spec))
    for items in by_trial.values():
        complete=[x[0] for x in items if not x[0].get('failure')]
        if (len({x[1]['training_seed'] for x in items})!=1
                or len({x['policy_fingerprint'] for x in complete})>1):
            raise ValueError('A trial must value one sealed policy on every independent stream')
    return by_trial


def summarize_benchmark(streams,*,expected_streams,refinement_pairs=()):
    """Retain every trial; EP is the primary control, never a selected winner.

    A missing/failed stream makes its entire trial incomplete. Complete other
    trials remain reportable. Construction refinements explicitly include
    independently retrained rules and cannot certify optimal-American Greeks.
    """
    grouped=_validate_streams(streams,expected_streams)
    prices=[];sampling=[];bumps=[];changes=[];per_seed=[];status=[];failures=[];complete={}
    for trial,items in grouped.items():
        bad=[(s,p) for s,p in items if s.get('failure')]
        status.append(dict(trial_id=trial,declared_streams=len(items),complete_streams=len(items)-len(bad),
            declared_paths=sum(p['paths'] for _,p in items),complete=not bad,
            used_for_selection=False,primary_control='european_premium'))
        if bad:
            failures.extend(dict(trial_id=trial,seed=p['seed'],failure=str(s['failure']),
                primary_acceptance=False,remaining_streams_not_used_for_partial_acceptance=True) for s,p in bad)
            continue
        group=[s for s,_ in items];complete[trial]=group
        case={k:items[0][1][k] for k in ('trial_id','steps','exercise_steps','factors','strike','maturity')}
        flags={**case,'streams':len(group),'primary_control':'european_premium',
            'used_for_selection':False,'finite_transition_and_exercise_law_only':True,
            'optimal_american_delta_ready':False}
        for control in CONTROLS:
            lower=[s['controls'][control]['lower'] for s in group]
            upper=[s['controls'][control]['upper'] for s in group]
            lo=moments([x[:,2] for x in lower]);hi=moments([x[:,2] for x in upper])
            gap=moments([(y-x)[:,2] for x,y in zip(lower,upper)])
            budget=gap['estimate']+1.96*(lo['standard_error']+gap['standard_error'])
            allowance=.001+1e-4*abs(lo['estimate'])
            prices.append({**flags,'control':control,'primary':control=='european_premium',
                'paths':lo['paths'],'price':lo['estimate'],'price_se':lo['standard_error'],
                'upper':hi['estimate'],'upper_se':hi['standard_error'],
                'gap':gap['estimate'],'gap_se':gap['standard_error'],
                'price_error_budget':budget,'price_allowance':allowance,
                'price_budget_pass':budget<=allowance,'upper_is_dual_maximum':True,
                'continuous_american_price_certified':False})
            for bump in BUMPS:
                chunks=[delta(x,bump) for x in lower];summary=moments(chunks)
                a=.002+.005*abs(summary['estimate'])
                sampling.append({**flags,'control':control,'primary':control=='european_premium',
                    'bump_relative':bump,**summary,'delta_allowance':a,
                    'sampling_pass':summary['half_width']<=a})
                for s,x in zip(group,chunks):
                    per_seed.append({**flags,'seed':s['seed'],'control':control,'quantity':'delta',
                        'bump_relative':bump,**moments([x]),'diagnostic_only':True})
            ref=moments([delta(x,BUMPS[1]) for x in lower])['estimate']
            contrast=_contrast([delta(x,BUMPS[1])-delta(x,BUMPS[0]) for x in lower],ref,'delta')
            bumps.append({**flags,'control':control,'primary':control=='european_premium',
                'coarse_bump':BUMPS[0],'fine_bump':BUMPS[1],**contrast})
            for s,x in zip(group,lower):
                per_seed.append({**flags,'seed':s['seed'],'control':control,'quantity':'price',
                    'bump_relative':0.,**moments([x[:,2]]),'diagnostic_only':True})
        for a,b in (('raw','european'),('raw','european_premium'),('european','european_premium')):
            for quantity,bump in (('price',0.),('delta',BUMPS[0]),('delta',BUMPS[1])):
                def value(s,c):
                    x=s['controls'][c]['lower'];return x[:,2] if quantity=='price' else delta(x,bump)
                ref=moments([value(s,b) for s in group])['estimate']
                changes.append({**flags,'comparison':a+' minus '+b,'quantity':quantity,
                    'bump_relative':bump,**_contrast([value(s,a)-value(s,b) for s in group],ref,quantity),
                    'paired_same_paths':True,'marginal_normal_interval':True})
    refinement=[]
    seen=set()
    for pair in refinement_pairs:
        coarse,fine=pair['coarse_trial'],pair['fine_trial'];key=(coarse,fine)
        if key in seen or coarse not in grouped or fine not in grouped or coarse==fine:
            raise ValueError('Unique declared coarse/fine benchmark comparisons required')
        seen.add(key)
        if not isinstance(pair.get('same_full_policy'),bool):raise ValueError('Refinement must declare full-policy identity')
        if coarse not in complete or fine not in complete:
            refinement.append({**pair,'complete':False,'refinement_check_pass':False,
                'reason':'A complete independent coarse and fine trial is required','used_for_selection':False})
            continue
        cg,fg=complete[coarse],complete[fine]
        if {s['seed'] for s in cg}&{s['seed'] for s in fg}:raise ValueError('Resolution confidence intervals require independent streams')
        if pair['same_full_policy'] and any(s.get('policy_fingerprint')!=cg[0].get('policy_fingerprint')
                for s in cg+fg):raise ValueError('A fixed-policy refinement changed its stopping rule')
        for control in CONTROLS:
            for quantity,bump in (('price',0.),('delta',BUMPS[0]),('delta',BUMPS[1])):
                def value(s):
                    x=s['controls'][control]['lower'];return x[:,2] if quantity=='price' else delta(x,bump)
                a=moments([value(s) for s in cg]);b=moments([value(s) for s in fg])
                change=b['estimate']-a['estimate'];se=math.hypot(a['standard_error'],b['standard_error'])
                allowance=.001+1e-4*abs(b['estimate']) if quantity=='price' else .002+.005*abs(b['estimate'])
                error=abs(change)+1.96*se
                refinement.append({**pair,'complete':True,'control':control,'primary':control=='european_premium',
                    'quantity':quantity,'bump_relative':bump,'coarse_paths':a['paths'],'fine_paths':b['paths'],
                    'coarse_estimate':a['estimate'],'fine_estimate':b['estimate'],
                    'change':change,'standard_error':se,'interval_low':change-1.96*se,
                    'interval_high':change+1.96*se,'change_plus_uncertainty':error,'allowance':allowance,
                    'refinement_check_pass':error<=allowance,'independent_resolutions':True,
                    'optimal_american_accuracy_certified':False,'used_for_selection':False,
                    'scope':'fixed lower-policy law sensitivity' if pair['same_full_policy']
                        else 'joint model/calendar/policy construction sensitivity'})
    return {name:pd.DataFrame(value) for name,value in (
        ('price_bounds',prices),('delta_sampling',sampling),('bump_refinement',bumps),
        ('control_changes',changes),('refinement',refinement),('per_seed',per_seed),
        ('case_status',status),('failures',failures))}


def _reference_increment(policy,affine,law,left,right,new,time):
    """Preserved general premium integral, not the new benchmark adapter."""
    from rvlab.pricing.rough_stopping_fourier import (
        _affine_values,_interpolate,_premium_values,_expected_premium_values)
    m=np.log(left)[:,None,None]+law['correlated'][:,:,None,None]-.5*law['orthogonal'][...,None]
    v=law['orthogonal'][...,None];p=law['probabilities'];branches=law['new_factors']
    expected=np.sum(_affine_values(affine,time,branches,m,v).mean(axis=2)*p[...,None],axis=1)
    actual=_affine_values(affine,time,new[:,None],np.log(right)[:,None,None],
        np.zeros((len(right),1,1,5)))[:,0,0]
    ec=_interpolate(policy,time,branches)[:,:,None,None]
    ep=np.sum(_expected_premium_values(m,v,ec,strike=policy.strike,
        stock_knots=policy.config.stock_knots,cap=policy.premium_cap).mean(axis=2)*p[...,None],axis=1)
    ap=_premium_values(right,_interpolate(policy,time,new)[:,None],strike=policy.strike,
        stock_knots=policy.config.stock_knots,cap=policy.premium_cap)
    return np.stack((actual-expected,
        float(policy.market.discount_curve.discount(time))*(ap-ep)),axis=-1)


def _replay_and_probe(a,policy,spec):
    """Whole RNG/calendar conformance; bounded independent conditional means."""
    from scipy.special import ndtri
    from rvlab.pricing.rough_estimators import _AffineSurrogate
    from rvlab.pricing.rough_simulation import (
        _theta,_drift_flow_with_integral,_weak_transition_branches)
    from rvlab.pricing.rough_stopping_fourier import _affine_values
    params,market,kernel=policy.params,policy.market,policy.kernel
    grid=a['actual_time_grid'];times=a['times'];n=spec['paths'];nf=spec['factors']
    nodes=np.asarray(kernel.nodes);weights=np.asarray(kernel.weights)
    stocks=np.broadcast_to(a['initial_spots'],(n,5)).copy();factors=np.zeros((n,nf))
    matrix=-np.diag(nodes)-params.kappa*np.ones((nf,1))*weights
    rng=np.random.default_rng(spec['seed']);cache={};records=[];errors=np.zeros(2)
    args=(params,market,kernel,np.array([policy.strike]),np.array([policy.maturity]),np.array([policy.is_call]))
    actual_affine=_AffineSurrogate(*args,grid,policy.control_config)
    operator_affine=actual_affine if np.array_equal(grid,a['operator_time_grid']) else _AffineSurrogate(
        *args,a['operator_time_grid'],policy.control_config)
    ids=np.array(sorted({0,min(1,n-1),min(511,n-1),min(512,n-1)}));nt=len(times)
    js=sorted({0,(nt-1)//4,(nt-1)//2,3*(nt-1)//4,nt-2})
    calendar={float(t):j for j,t in enumerate(times)};panel={};total={j:np.zeros((len(ids),5,2)) for j in js}
    for j in js:
        start,end=np.searchsorted(grid,times[j:j+2])
        for k in range(start+1,end+1):panel[k]=j
    for k,(left,right) in enumerate(zip(grid[:-1],grid[1:]),1):
        dt=float(right-left);theta=_theta(params,.5*(left+right));key=(dt,theta)
        if key not in cache:cache[key]=_drift_flow_with_integral(matrix,
            np.full(nf,params.kappa*(theta-params.v0)),weights,params.v0,dt/2)
        flow=cache[key];growth=math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        for begin in range(0,n,512):
            end=min(begin+512,n);old=factors[begin:end];u=rng.random((end-begin,3))
            law=_weak_transition_branches(old,params,nodes=nodes,weights=weights,
                flow=flow[:nf,:nf],shift=flow[:nf,-1],integrated_flow=flow,
                dt=dt,theta=theta,growth=growth,stock_integral=policy.config.stock_integral)
            p=law['probabilities'];choice=(u[:,0]>p[:,0]).astype(int)+(u[:,0]>p[:,:2].sum(axis=1)).astype(int)
            order=(u[:,1]>=.5).astype(int);row=np.arange(end-begin)
            m=np.log(stocks[begin:end])[:,None,None]+law['correlated'][:,:,None,None]-.5*law['orthogonal'][...,None]
            q=law['orthogonal'][row,choice,order];z=ndtri(np.clip(u[:,2],np.finfo(float).tiny,1-np.finfo(float).eps))
            nextstock=np.exp(m[row,choice,order]+np.sqrt(q)[:,None]*z[:,None]);new=law['new_factors'][row,choice]
            chosen=ids[(ids>=begin)&(ids<end)]
            if k in panel and len(chosen):
                local=chosen-begin
                limited={name:value[local] for name,value in law.items() if isinstance(value,np.ndarray)}
                value=_reference_increment(policy,actual_affine,limited,stocks[chosen],nextstock[local],new[local],float(right))
                total[panel[k]][np.searchsorted(ids,chosen)]+=value
            stocks[begin:end]=nextstock;factors[begin:end]=new
        if float(right) in calendar:
            j=calendar[float(right)]
            errors[0]=max(errors[0],float(np.max(abs(stocks-a['stocks'][:,j]))))
            errors[1]=max(errors[1],float(np.max(abs(factors-a['factors'][:,j]))))
    if errors[0]>2e-12 or errors[1]>2e-13:raise ValueError('Saved paths changed whole-stream RNG or full factor memory')
    for j in js:
        actual=np.stack((a['european_martingale'][ids,j+1]-a['european_martingale'][ids,j],
            a['premium_martingale'][ids,j+1]-a['premium_martingale'][ids,j]),axis=-1)
        _close(actual,total[j],'Actual-law base martingale failed bounded conditional centering',2e-11)
        records.append(dict(kind='conditional centering',calendar_left=j,paths=ids.tolist(),
            all_bumps=True,actual_physical_steps=spec['steps'],exhaustive_conditional_repricing=False,
            max_error=float(np.max(abs(actual-total[j])))))
    for j in sorted(set(js+[nt-1])):
        value=_affine_values(operator_affine,float(times[j]),a['factors'][ids,j,None],
            np.log(a['stocks'][ids,j])[:,None,None],np.zeros((len(ids),1,1,5)))[:,0,0]
        _close(a['decision_european'][ids,j],value,'The full frozen European decision function changed grids',2e-11)
        records.append(dict(kind='frozen decision European',calendar_index=j,paths=ids.tolist(),
            operator_grid_points=len(a['operator_time_grid']),all_bumps=True,
            max_error=float(np.max(abs(a['decision_european'][ids,j]-value)))))
    return records,errors


def validate_benchmark_samples(a,policy,*,spec):
    """Independent complete-path algebra and bounded conditional/RNG audit.

    Numerical failures are retained by the campaign as failed streams, not
    repaired by this validator. The policy is loaded from its prevaluation
    coefficient seal; no training or empirical recentering occurs here.
    """
    from rvlab.pricing.rough_simulation import _time_grid
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    n=spec['paths'];nt=spec['exercise_steps']+1;nf=spec['factors']
    if (isinstance(n,bool) or not isinstance(n,Integral) or n<2 or spec['seed']==spec['training_seed']
            or policy.config.training_seed!=spec['training_seed'] or policy.strike!=spec['strike']
            or policy.maturity!=spec['maturity'] or len(policy.kernel.nodes)!=nf
            or len(policy.exercise_times)!=nt or policy.market.spot!=100. or policy.is_call
            or policy.exercise_style!='american' or policy.premium_cap!=.7 or policy.config.batch_size!=512
            or policy.market.dividends.events):
        raise ValueError('Benchmark policy, declared contract, independent seeds or dimensions changed')
    times=np.asarray(a['times']);discount=np.asarray(a['discounts'])
    operator=_time_grid(policy.params,policy.market,policy.exercise_times,policy.config.steps)
    physical=_time_grid(policy.params,policy.market,policy.exercise_times,spec['steps'])
    if (not np.array_equal(times,policy.exercise_times) or not np.array_equal(a['operator_time_grid'],operator)
            or not np.array_equal(a['actual_time_grid'],physical) or not np.array_equal(a['relative_spots'],RELATIVE)
            or not np.array_equal(a['initial_spots'],100.*(1+RELATIVE))
            or not np.array_equal(discount,policy.market.discount_curve.discount(times))
            or np.shape(a['eligible'])!=(nt,) or a['eligible'].dtype.kind!='b' or not a['eligible'].all()):
        raise ValueError('Original decision grid, actual law, discounting or CRN spots changed')
    shape=(n,nt,5)
    for name in ('stocks','discounted_payoffs','european_martingale','premium_martingale',
            'european_proxy','decision_european','decision_continuation'):
        if np.shape(a.get(name))!=shape or not np.isfinite(a[name]).all():
            raise ValueError('Finite complete path/date/stock primitive required: '+name)
    if (np.shape(a.get('factors'))!=(n,nt,nf) or not np.isfinite(a['factors']).all()
            or np.any(a['stocks']<=0) or not np.array_equal(a['stocks'][:,0],np.broadcast_to(a['initial_spots'],(n,5)))
            or np.any(a['factors'][:,0]!=0) or any(np.any(a[name][:,0]!=0)
                for name in ('european_martingale','premium_martingale'))):
        raise ValueError('Initial states, full factor histories or martingale origins changed')
    intrinsic=np.maximum(policy.strike-a['stocks'],0.)
    if not np.array_equal(a['discounted_payoffs'],intrinsic*discount[None,:,None]):
        raise ValueError('Undiscounted per-unit vanilla payoff construction changed')
    future=np.stack([a['decision_european'][:,j]/discount[j]+np.clip(
        model.value(a['stocks'][:,j],a['factors'][:,j],policy.strike),0.,policy.premium_cap)
        for j,model in enumerate(policy.continuations)],axis=1)
    _close(a['decision_continuation'],future,'Saved full continuation differs from sealed policy',2e-12)
    rule=(intrinsic>0)&(intrinsic>=future);rule[:,0]=policy.exercise_at_inception;rule[:,-1]=True
    if np.shape(a.get('decision_rule'))!=shape or a['decision_rule'].dtype.kind!='b' or not np.array_equal(a['decision_rule'],rule):
        raise ValueError('A stock bump changed its own nonanticipating stopping rule')
    stop=np.argmax(rule,axis=1)
    if a['stop_indices'].dtype.kind not in 'iu' or not np.array_equal(a['stop_indices'],stop):
        raise ValueError('Own bump stopping dates differ from first eligible exercise')
    if np.array_equal(physical,operator) and not np.array_equal(a['public_stop_indices'],stop):
        raise ValueError('Same-law replay differs from the direct public backend')
    row=np.arange(n)[:,None];bump=np.arange(5)[None,:];controls={}
    for name,m in (('raw',np.zeros(shape)),('european',a['european_martingale']),
            ('european_premium',a['european_martingale']+a['premium_martingale'])):
        adjusted=a['discounted_payoffs']-m;lower=adjusted[row,stop,bump];upper=adjusted.max(axis=1)
        _close(a[name+'_lower'],lower,'Lower estimator changed own stopping date/control')
        _close(a[name+'_upper'],upper,'Upper estimator is not the declared genuine dual maximum')
        if np.any(upper-lower< -2e-13):raise ValueError('A pathwise stopping gap became negative')
        controls[name]={'lower':a[name+'_lower'],'upper':a[name+'_upper']}
    probes,errors=_replay_and_probe(a,policy,spec)
    stream={k:spec[k] for k in ('trial_id','seed','training_seed','paths','steps','exercise_steps','factors','strike','maturity')}
    stream.update(policy_fingerprint=fourier_policy_fingerprint(policy),initial_spots=a['initial_spots'],
        relative_spots=a['relative_spots'],stops=stop,controls=controls)
    return {'stream':stream,'checks':dict(trial_id=spec['trial_id'],seed=spec['seed'],paths=n,
        all_own_bump_stops_checked=True,all_payoffs_and_dual_maxima_checked=True,
        all_calendar_factor_and_CRN_paths_checked=True,max_stock_error=float(errors[0]),
        max_factor_error=float(errors[1]),policy_operator_steps=policy.config.steps,
        outer_steps=spec['steps'],exercise_steps=spec['exercise_steps'],
        no_fitting_or_selection=True,all_paths_retained=True,
        conditional_repricing_exhaustive=False,optimal_american_accuracy_certified=False),
        'probes':probes}


__all__=['summarize_benchmark','validate_benchmark_samples','moments','delta','CONTROLS','RELATIVE','BUMPS']
