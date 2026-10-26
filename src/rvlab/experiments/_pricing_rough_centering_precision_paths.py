"""Full microstate replay for a frozen stopping-control precision experiment.

The cheap simulator shares the actual weak transition law. All original
exercise dates are retained; saving extra states does not create exercise.
"""
from __future__ import annotations

import numpy as np
from rvlab.pricing.rough_simulation import (
    RoughSimulationConfig, _time_grid, simulate_rough_heston)
from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_stopping_fourier import _affine_values
from rvlab.pricing.rough_policy_diagnostics import _policy_stops


def full_microstate_paths(frozen, *, seed, paths):
    policy=frozen['proxy']
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    if isinstance(paths,bool) or not isinstance(paths,(int,np.integer)) or paths<2:
        raise ValueError('paths must be an integer >=2')
    if policy.market.dividends.events:
        raise ValueError('this frozen precision experiment supports no cash events')
    forbidden=set(frozen['original_policy'].training_seeds+frozen['selected_policy'].training_seeds)
    forbidden.update((policy.config.training_seed,frozen['weights'].training_seed))
    if seed in forbidden:
        raise ValueError('microstate paths must be independent of every fitting stream')
    grid=_time_grid(policy.params,policy.market,policy.continuation_times,policy.config.steps)
    simulation=simulate_rough_heston(policy.params,policy.market,policy.kernel,
        observation_times=grid,config=RoughSimulationConfig(paths=paths,steps=policy.config.steps,
        seed=seed,batch_size=policy.config.batch_size,memory_budget_mb=policy.config.memory_budget_mb,
        stock_integral=policy.config.stock_integral))
    if not np.array_equal(simulation.times,grid):
        raise ValueError('saving microstates changed the integration grid')
    exercise=policy.exercise_times
    indices=np.searchsorted(grid,exercise)
    if not np.array_equal(grid[indices],exercise):
        raise ValueError('original exercise dates are missing from the microgrid')
    affine=_AffineSurrogate(policy.params,policy.market,policy.kernel,np.array([policy.strike]),
        np.array([policy.maturity]),np.array([policy.is_call]),grid,policy.control_config)
    euro=np.empty((paths,len(exercise),1))
    for j,index in enumerate(indices):
        euro[:,j,0]=european_values(affine,float(exercise[j]),simulation.spots[:,index],simulation.factors[:,index])
    stocks=simulation.spots[:,indices,None];factors=simulation.factors[:,indices]
    eligible=np.ones(len(exercise),bool)
    original=_policy_stops(frozen['original_policy'],exercise,eligible,stocks,factors,euro)[:,0]
    selected=_policy_stops(frozen['selected_policy'],exercise,eligible,stocks,factors,euro)[:,0]
    original_time,selected_time=exercise[original],exercise[selected]
    gates=((grid[None,:-1]<selected_time[:,None]).astype(np.int8)
        -(grid[None,:-1]<original_time[:,None]).astype(np.int8))
    discount=policy.market.discount_curve.discount(exercise)
    intrinsic=np.maximum((1 if policy.is_call else -1)*(stocks[:,:,0]-policy.strike),0.)
    p=np.arange(paths)
    raw_gain=discount[selected]*intrinsic[p,selected]-discount[original]*intrinsic[p,original]
    return {'simulation':simulation,'affine':affine,'exercise_indices':indices,
        'european_exercise_values':euro,'original_stops':original,'selected_stops':selected,
        'original_stop_times':original_time,'selected_stop_times':selected_time,
        'signed_left_gates':gates,'raw_gain':raw_gain,'seed':int(seed),'paths':int(paths)}


def european_values(affine,time,stock,factors):
    stock=np.asarray(stock,float);factors=np.asarray(factors,float)
    return _affine_values(affine,time,factors[:,None,:],
        np.log(stock)[:,None,None,None],np.zeros((len(stock),1,1,1)))[:,0,0,0]


def verify_microstate_replay(frozen, *, seed=890105042, paths=32):
    """Excluded independent smoke comparison with the complete control stream."""
    from dataclasses import replace
    from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy
    from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
    cheap=full_microstate_paths(frozen,seed=seed,paths=paths)
    policy=frozen['proxy']
    trace=value_fourier_stopping_policy(policy,config=replace(policy.config,
        valuation_seed=seed,valuation_paths=paths),trace=True)['trace']
    index=cheap['exercise_indices'];simulation=cheap['simulation']
    original=evaluate_trace_policy(trace,policy=frozen['original_policy'],martingale_weights=frozen['weights'])
    selected=evaluate_trace_policy(trace,policy=frozen['selected_policy'],martingale_weights=frozen['weights'])
    stock_error=float(np.max(np.abs(simulation.spots[:,index]-trace['stocks'][:,:,0])))
    factor_error=float(np.max(np.abs(simulation.factors[:,index]-trace['factors'])))
    euro_error=float(np.max(np.abs(cheap['european_exercise_values']-trace['european_proxy'])))
    assert stock_error<1e-10 and factor_error<1e-12 and euro_error<1e-10
    assert np.array_equal(cheap['original_stops'],original.stop_indices[:,0])
    assert np.array_equal(cheap['selected_stops'],selected.stop_indices[:,0])
    raw_error=float(np.max(np.abs(cheap['raw_gain']-(selected.raw_samples[:,0]-original.raw_samples[:,0]))))
    assert raw_error<1e-10
    return {'seed':seed,'paths':paths,'stock_max_difference':stock_error,'factor_max_difference':factor_error,
        'european_proxy_max_difference':euro_error,'raw_gain_max_difference':raw_error,
        'original_stops_identical':True,'selected_stops_identical':True,'no_new_exercise_dates':True,
        'excluded_from_primary_inference':True},cheap,trace


def evaluate_precision_paths(frozen, data, *, quadrature_nodes=(32,64,128)):
    """Evaluate all nonzero predictable gates, without selecting by residuals."""
    from rvlab.experiments._pricing_rough_centering_precision_quadrature import ConditionalCenteringQuadrature
    import pandas as pd
    sim=data['simulation'];policy=frozen['proxy'];fit=frozen['weights']
    engine=ConditionalCenteringQuadrature(policy,martingale_weights=fit,time_grid=sim.times)
    if (not isinstance(quadrature_nodes,(tuple,list)) or len(quadrature_nodes)<2
            or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x<4 for x in quadrature_nodes)
            or any(a>=b for a,b in zip(quadrature_nodes[:-1],quadrature_nodes[1:]))):
        raise ValueError('quadrature nodes must be strictly increasing integers >=4, with at least two levels')
    n=data['paths'];levels=len(quadrature_nodes)
    raw=np.zeros(n);projected=np.zeros((levels,n));error=np.zeros(n);tail=np.zeros(n);arithmetic=np.zeros(n)
    state_rows=[]
    for j,left in enumerate(sim.times[:-1]):
        gates=data['signed_left_gates'][:,j];index=np.flatnonzero(gates)
        if not len(index):continue
        checked=engine.evaluate(left_time=float(left),left_stocks=sim.spots[index,j],
            left_factors=sim.factors[index,j],
            original_policy_stopped=data['original_stop_times'][index]<=left,
            quadrature_nodes=quadrature_nodes)
        right=checked['right_time']
        actual=engine.actual_primitive_values(time=right,stocks=sim.spots[index,j+1],factors=sim.factors[index,j+1])
        primitive=actual-checked['analytic_primitive_expectation']
        discount=float(policy.market.discount_curve.discount(right))
        primitive*=np.array([1.,discount,discount])
        e,p,i=primitive.T;boundary=checked['boundary_multiplier']
        parts={'european':e,'premium':p,'micro_boundary':boundary*(i-e)}
        bin_index=int(np.searchsorted(fit.time_edges[1:],right,side='left'))
        weights=fit.weights[bin_index]
        realized=sum(w*parts[name] for w,name in zip(weights,fit.config.components))
        gate=gates[index]
        raw[index]+=gate*realized;projected[:,index]+=gate[None,:]*checked['drift_by_level']
        qerror=np.maximum(checked['refinement_and_tail_estimate']-checked['absolute_tail_envelope'],0.)
        error[index]+=qerror;tail[index]+=checked['absolute_tail_envelope']
        scale=1.+np.abs(actual).sum(axis=1)+np.abs(checked['analytic_primitive_expectation']).sum(axis=1)
        arith=1024*np.finfo(float).eps*(1.+np.abs(weights).sum())*scale
        arithmetic[index]+=arith
        for k,path in enumerate(index):
            state_rows.append({'path':int(path),'microstep':j,'left_time':float(left),'right_time':right,
                'signed_left_gate':int(gate[k]),'left_stock':float(sim.spots[path,j]),
                'original_stopped_at_left':bool(data['original_stop_times'][path]<=left),
                'raw_increment':float(realized[k]),'projected_coarse':float(checked['drift_by_level'][-2,k]),
                'projected_fine':float(checked['drift_by_level'][-1,k]),
                'first_refinement':float(abs(checked['drift_by_level'][1,k]-checked['drift_by_level'][0,k])),
                'last_refinement':float(abs(checked['drift_by_level'][-1,k]-checked['drift_by_level'][-2,k])),
                'quadrature_error_estimate':float(qerror[k]),'tail_envelope':float(checked['absolute_tail_envelope'][k]),
                'arithmetic_allowance':float(arith[k]),'boundary_multiplier':float(boundary[k])})
    samples={'raw_contrast':raw,'projected_coarse':projected[-2], 'projected_fine':projected[-1],
        'quadrature_error':error,'tail_error':tail,'arithmetic_error':arithmetic}
    return {'samples':samples,'projected_by_level':projected,'states':pd.DataFrame(state_rows),
        'absolute_gate_counts':np.abs(data['signed_left_gates']).sum(axis=1),
        'raw_gain':data['raw_gain'],'controlled_gain':data['raw_gain']-raw}
