"""Stock bumps of the frozen GH16 Bellman rule, with a fixed inner operator.

Outer weak-time refinement changes the valuation law, not the exercise rule.
The latter always uses the original two-physical-step conditional operator.
All stock bumps recompute their own continuation and stopping decisions.
"""
from dataclasses import replace
from numbers import Integral
import numpy as np
import pandas as pd

from ._pricing_rough_bellman_policy import BellmanEvaluator
from ._pricing_rough_stock_delta import _summary
from rvlab.pricing.rough_policy_diagnostics import _martingale, evaluate_trace_policy
from rvlab.pricing.rough_simulation import _time_grid
from rvlab.pricing.rough_stopping_fourier import _affine_values, fourier_policy_fingerprint, value_fourier_stopping_policy

LEVELS=(.005,.0025)
POLICIES=('frozen selected policy','GH16 blend1')
TOLERANCES={'price_atol':.001,'price_rtol':.0001,'delta_atol':.002,
    'delta_rtol':.005,'confidence_multiplier':1.96,'spread_fraction':.1}

def _integer(value,name,minimum):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')

def bellman_policy_stops(trace,evaluator,*,outer_steps,forbidden_seeds=()):
    """Apply one frozen rule sequentially, computing only alive ITM states.

    Missing continuation entries are explicitly NaN and never used in a
    decision. Stopped paths remain present in every saved endpoint array.
    """
    _integer(outer_steps,'outer_steps',1)
    proxy=evaluator.proxy;selected=evaluator.selected;seed=trace.get('seed')
    if (evaluator.orders!=(16,) or isinstance(seed,bool) or not isinstance(seed,Integral)
            or seed<0 or seed in set(selected.training_seeds)|set(forbidden_seeds)):
        raise ValueError('Independent matching GH16 policy and recorded seed required')
    stocks=np.asarray(trace['stocks']);factors=np.asarray(trace['factors']);times=np.asarray(trace['times'])
    discounts=np.asarray(trace['discounts']);initial=np.asarray(trace['initial_spots'])
    expected_grid=_time_grid(proxy.params,proxy.market,proxy.exercise_times,outer_steps)
    if (trace.get('policy_fingerprint')!=fourier_policy_fingerprint(proxy)
            or trace.get('kernel_fingerprint')!=proxy.kernel.fingerprint
            or trace.get('strike')!=proxy.strike or not np.array_equal(times,evaluator.times)
            or not np.array_equal(trace['actual_time_grid'],expected_grid)
            or not np.array_equal(discounts,evaluator.discounts)
            or stocks.ndim!=3 or stocks.shape[1]!=len(times) or stocks.shape[2]!=len(initial)
            or factors.shape!=(len(stocks),len(times),len(proxy.kernel.nodes))
            or initial.ndim!=1 or not np.array_equal(stocks[:,0],np.broadcast_to(initial,stocks[:,0].shape))
            or not np.array_equal(trace['eligible'],np.ones(len(times),bool))
            or not all(np.isfinite(a).all() for a in (stocks,factors,initial)) or np.any(stocks<=0)):
        raise ValueError('Matching full-factor multistock trace and outer physical law required')
    n,nt,nb=stocks.shape;stops=np.full((n,nb),nt-1,dtype=int)
    mask=np.zeros(stocks.shape,bool);continuation=np.full(stocks.shape,np.nan);date_records=[]
    for j in range(1,nt-1):
        intrinsic=np.maximum((1 if proxy.is_call else -1)*(stocks[:,j]-proxy.strike),0.)
        active=(stops==nt-1)&(intrinsic>0)
        path,bump=np.nonzero(active);mask[:,j]=active
        if not len(path):continue
        values=evaluator.conditional_expectations(j,stocks[path,j,bump],factors[path,j])
        c=values['continuation'][:,0]
        if c.shape!=(len(path),) or not np.isfinite(c).all() or np.any(c<0):
            raise FloatingPointError('Nonfinite frozen Bellman continuation')
        continuation[path,j,bump]=c
        exercise=intrinsic[path,bump]>=c
        stops[path[exercise],bump[exercise]]=j
        date_records.append({'exercise_index':j,'states':len(path),**values['diagnostics']})
    return {'stops':stops,'computed_mask':mask,'continuation':continuation,
        'diagnostics':{'operator_steps':proxy.config.steps,'outer_steps':outer_steps,
            'operator_grid':evaluator.grid,'computed_states':int(mask.sum()),
            'only_alive_ITM_evaluated':True,'all_paths_retained':True,
            'all_bumps_recompute_continuation_and_stops':True,'dates':date_records}}

def bellman_delta_samples(bundle,*,seed,paths,steps,forbidden_seeds=()):
    """Generate one independent five-stock CRN trace and both frozen rules."""
    for value,name,minimum in ((seed,'seed',0),(paths,'paths',2),(steps,'steps',1)):
        _integer(value,name,minimum)
    proxy=bundle['proxy'];selected=bundle['selected_policy']
    forbidden=set(forbidden_seeds)|set(bundle['manifest']['all_selection_training_seeds'])
    forbidden|={proxy.config.training_seed,proxy.config.valuation_seed,bundle['weights'].training_seed}
    if seed in forbidden:raise ValueError('Bellman delta stream reuses fitting, selection or reserved paths')
    if proxy.config.steps!=128 or len(proxy.exercise_times)!=65:
        raise ValueError('This frozen Bellman policy requires its original128-law/64exercise calendar')
    relative=np.array([-.005,-.0025,0.,.0025,.005]);spots=proxy.market.spot*(1+relative)
    config=replace(proxy.config,valuation_paths=int(paths),valuation_seed=int(seed),steps=int(steps))
    valuation=value_fourier_stopping_policy(proxy,config=config,spots=spots,trace=True)
    trace=valuation['trace'];evaluator=BellmanEvaluator(proxy,selected,orders=(16,))
    looked=bellman_policy_stops(trace,evaluator,outer_steps=steps,forbidden_seeds=tuple(forbidden))
    baseline_euro=trace['european_proxy'].copy()
    if steps!=proxy.config.steps:
        for j,t in enumerate(trace['times']):
            baseline_euro[:,j]=_affine_values(evaluator.affine,float(t),trace['factors'][:,j,None,:],
                np.log(trace['stocks'][:,j])[:,None,None,:],np.zeros((paths,1,1,5)))[:,0,0]
    baseline_trace={**trace,'european_proxy':baseline_euro}
    baseline=evaluate_trace_policy(baseline_trace,policy=selected,martingale_weights=bundle['weights'])
    baseline_continuation=np.empty_like(baseline_euro)
    for j in range(len(trace['times'])):
        q=np.clip(selected.continuations[j].value(trace['stocks'][:,j],trace['factors'][:,j],proxy.strike),0.,selected.premium_cap)
        baseline_continuation[:,j]=baseline_euro[:,j]/trace['discounts'][j]+q
    martingale,_,_= _martingale(trace,bundle['weights'])
    path=np.arange(paths)[:,None];bump=np.arange(5)[None,:];stops=looked['stops']
    raw=trace['discounted_payoffs'][path,stops,bump];controlled=raw-martingale[path,stops,bump]
    arrays={'relative_spots':relative,'times':trace['times'],'discounts':trace['discounts'],'eligible':trace['eligible'],
        'actual_time_grid':trace['actual_time_grid'],'operator_time_grid':evaluator.grid,
        'strike':np.array(proxy.strike),'is_call':np.array(proxy.is_call),
        'stable_weights':bundle['weights'].weights,'stable_time_edges':bundle['weights'].time_edges,
        'stable_components':np.asarray(bundle['weights'].config.components,dtype='U32'),
        'initial_spots':trace['initial_spots'],'stocks':trace['stocks'],'factors':trace['factors'],
        'european_proxy':trace['european_proxy'],'payoff':trace['discounted_payoffs'],
        'baseline_european_proxy':baseline_euro,'baseline_continuation':baseline_continuation,
        'european_martingale':trace['european_martingale'],'premium_martingale':trace['premium_martingale'],
        'micro_boundary_martingale':trace['micro_boundary_martingale'],
        'micro_poststop_martingale':trace['micro_poststop_martingale'],'intrinsic_martingale':trace['intrinsic_martingale'],
        'stable_martingale':martingale,'bellman_computed_mask':looked['computed_mask'],
        'bellman_continuation':looked['continuation'],
        'baseline_stops':baseline.stop_indices,'baseline_raw':baseline.raw_samples,
        'baseline_controlled':baseline.controlled_samples,'bellman_stops':stops,
        'bellman_raw':raw,'bellman_controlled':controlled}
    for prefix in ('baseline','bellman'):
        for estimator in ('raw','controlled'):
            prices=arrays[prefix+'_'+estimator]
            arrays[prefix+'_'+estimator+'_deltas']=np.column_stack([
                (prices[:,4]-prices[:,0])/(2*proxy.market.spot*LEVELS[0]),
                (prices[:,3]-prices[:,1])/(2*proxy.market.spot*LEVELS[1])])
    return arrays,{'seed':seed,'paths':paths,'steps':steps,'physical_operator_steps':128,
        'actual_time_steps':len(trace['actual_time_grid'])-1,'exercise_steps':64,
        'operator_identity_unchanged':True,'original_two_physical_step_lookahead':True,
        'stable_martingale_weights':bundle['weights'].weights.tolist(),
        'stable_martingale_time_edges':bundle['weights'].time_edges.tolist(),
        'all_paths_retained':True,'no_fitting_or_selection':True,
        'target':'finite stock differences of frozen lower policies, not optimal American value',
        'optimal_delta_ready':False,'tolerances_changed':False,**looked['diagnostics']}

def summarize_bellman_delta(arrays):
    rows=[];comparisons=[]
    for prefix,name in zip(('baseline','bellman'),POLICIES):
        for estimator in ('raw','controlled'):
            values=arrays[prefix+'_'+estimator+'_deltas']
            for k,level in enumerate(LEVELS):
                stat=_summary(values[:,k]);allowance=.002+.005*abs(stat['mean'])
                rows.append({'policy':name,'estimator':estimator,'bump_relative':level,
                    'paths':len(values),**stat,'delta_allowance':allowance,
                    'sampling_passed':stat['ci_half_width']<=allowance,'optimal_delta_ready':False})
            change=values[:,1]-values[:,0];stat=_summary(change)
            allowance=.002+.005*abs(values[:,1].mean())
            comparisons.append({'comparison':'bump refinement','policy':name,'estimator':estimator,
                'coarse_bump':LEVELS[0],'fine_bump':LEVELS[1],**stat,'delta_allowance':allowance,
                'change_plus_uncertainty':abs(stat['mean'])+stat['ci_half_width'],
                'comparison_passed':abs(stat['mean'])+stat['ci_half_width']<=allowance})
    for estimator in ('raw','controlled'):
        for k,level in enumerate(LEVELS):
            change=arrays['bellman_'+estimator+'_deltas'][:,k]-arrays['baseline_'+estimator+'_deltas'][:,k]
            comparisons.append({'comparison':'Bellman minus baseline','policy':'paired policies','estimator':estimator,
                'coarse_bump':np.nan,'fine_bump':level,**_summary(change)})
    for prefix,name in zip(('baseline','bellman'),POLICIES):
        for k,level in enumerate(LEVELS):
            change=arrays[prefix+'_raw_deltas'][:,k]-arrays[prefix+'_controlled_deltas'][:,k]
            comparisons.append({'comparison':'raw minus controlled','policy':name,'estimator':'paired estimators',
                'coarse_bump':np.nan,'fine_bump':level,**_summary(change)})
    return pd.DataFrame(rows),pd.DataFrame(comparisons)

__all__=['LEVELS','POLICIES','TOLERANCES','bellman_policy_stops','bellman_delta_samples','summarize_bellman_delta']
