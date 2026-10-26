"""Shared exactly centered European controls for frozen American price bounds.

The control is a difference of two unbiased estimators of the same finite-law
European price. No Fourier price or empirical valuation mean centers it. A
common endpoint translation preserves each stopping gap pathwise; the shifted
upper estimate retains its expectation bound, not a Rogers-max representation.
"""
from numbers import Integral
import math
import numpy as np
from scipy.special import ndtri
from rvlab.pricing.rough_estimators import _AffineSurrogate,_conditional_payoffs
from rvlab.pricing.rough_simulation import _time_grid,_theta,_drift_flow_with_integral,_weak_transition_branches


def conditional_european_replay(proxy,*,seed,paths,indices=None,reference=None,retain_primitives=True):
    """Replay the exact full random stream; integrate the orthogonal stock noise.

    ``indices`` limits expensive conditional-Fourier evaluations, never random
    draws or physical factor/stock propagation. Every seed/path pair therefore
    preserves its original absolute index at all exercise dates.
    """
    for value,name,minimum in ((seed,'seed',0),(paths,'paths',2)):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:raise ValueError(f'{name} must be integer >= {minimum}')
    if proxy.market.dividends.events or proxy.params.vol_of_vol<=0 or proxy.config.steps!=128:
        raise ValueError('This shared-control replay requires the frozen stochastic no-cash128-law')
    index=np.arange(paths) if indices is None else np.asarray(indices)
    if (index.ndim!=1 or index.dtype.kind not in 'iu' or not len(index) or np.any(index<0) or np.any(index>=paths)
            or np.any(np.diff(index)<=0)):
        raise ValueError('Declare distinct increasing absolute full-stream path indices')
    params,market,kernel=proxy.params,proxy.market,proxy.kernel
    count=len(kernel.nodes);nodes=np.asarray(kernel.nodes);weights=np.asarray(kernel.weights)
    grid=_time_grid(params,market,proxy.exercise_times,proxy.config.steps)
    affine=_AffineSurrogate(params,market,kernel,np.array([proxy.strike]),np.array([proxy.maturity]),np.array([proxy.is_call]),grid,proxy.control_config)
    matrix=-np.diag(nodes)-params.kappa*np.ones((count,1))*weights
    factors=np.zeros((paths,count));stock=np.full(paths,market.spot);rng=np.random.default_rng(seed);cache={}
    n=len(index);mu=np.full(n,math.log(market.spot));variance=np.zeros(n);martingale=np.zeros(n)
    shape=(n,len(grid));expected=np.zeros(shape);actual=np.zeros(shape);means=np.zeros(shape);clocks=np.zeros(shape)
    means[:,0]=mu;physical_factors=np.zeros((n,len(grid),count))
    physical_stocks=np.zeros(shape);physical_stocks[:,0]=stock[index]
    exercise_index={float(t):j for j,t in enumerate(proxy.exercise_times)}
    max_stock_error=0.;max_factor_error=0.
    if reference is not None:
        if (np.shape(reference['stocks'])!=(paths,len(proxy.exercise_times),1)
                or np.shape(reference['factors'])!=(paths,len(proxy.exercise_times),count)
                or not np.array_equal(reference['actual_time_grid'],grid)):
            raise ValueError('Full-stream saved states and exact physical grid required for replay audit')
    for k,(left,right) in enumerate(zip(grid[:-1],grid[1:]),1):
        dt=float(right-left);theta=_theta(params,.5*(left+right));key=(dt,theta)
        if key not in cache:
            cache[key]=_drift_flow_with_integral(matrix,np.full(count,params.kappa*(theta-params.v0)),weights,params.v0,dt/2)
        flow=cache[key];growth=math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        for begin in range(0,paths,proxy.config.batch_size):
            end=min(begin+proxy.config.batch_size,paths);old=factors[begin:end]
            random=rng.random((end-begin,3))
            law=_weak_transition_branches(old,params,nodes=nodes,weights=weights,flow=flow[:count,:count],
                shift=flow[:count,-1],integrated_flow=flow,dt=dt,theta=theta,growth=growth,stock_integral=proxy.config.stock_integral)
            p=law['probabilities'];choice=(random[:,0]>p[:,0]).astype(int)+(random[:,0]>p[:,:2].sum(axis=1)).astype(int)
            endpoint=(random[:,1]>=.5).astype(int);row=np.arange(end-begin)
            q=law['orthogonal'][row,choice,endpoint];correlated=law['correlated'][row,choice]
            # Same operation order as the frozen full-state Fourier simulator.
            fullmeans=np.log(stock[begin:end])[:,None,None]+law['correlated'][:,:,None]-.5*law['orthogonal']
            nextmean=fullmeans[row,choice,endpoint]
            normal=ndtri(np.clip(random[:,2],np.finfo(float).tiny,1-np.finfo(float).eps))
            stock[begin:end]=np.exp(nextmean+np.sqrt(q)*normal)
            factors[begin:end]=law['new_factors'][row,choice]
            chosen=np.flatnonzero((index>=begin)&(index<end))
            if len(chosen):
                local=index[chosen]-begin
                conditional_mean=mu[chosen,None,None]+law['correlated'][local,:,None]-.5*law['orthogonal'][local]
                conditional_clock=variance[chosen,None,None]+law['orthogonal'][local]
                values=affine.values(float(right),law['new_factors'][local],conditional_mean,conditional_clock,np.array([0]))[...,0]
                ex=np.sum(values.mean(axis=2)*p[local],axis=1)
                mu[chosen]=conditional_mean[np.arange(len(chosen)),choice[local],endpoint[local]]
                variance[chosen]=conditional_clock[np.arange(len(chosen)),choice[local],endpoint[local]]
                av=affine.values(float(right),factors[index[chosen],None],mu[chosen,None,None],variance[chosen,None,None],np.array([0]))[:,0,0,0]
                martingale[chosen]+=av-ex;expected[chosen,k]=ex;actual[chosen,k]=av
        means[:,k]=mu;clocks[:,k]=variance;physical_stocks[:,k]=stock[index];physical_factors[:,k]=factors[index]
        if reference is not None and float(right) in exercise_index:
            j=exercise_index[float(right)]
            max_stock_error=max(max_stock_error,float(np.max(abs(stock-reference['stocks'][:,j,0]))))
            max_factor_error=max(max_factor_error,float(np.max(abs(factors-reference['factors'][:,j]))))
    if not all(np.isfinite(v).all() for v in (mu,variance,martingale,stock,factors,expected,actual)) or np.any(variance<0):
        raise FloatingPointError('Conditional-European replay produced nonfinite or negative-clock primitives')
    # The physical replay itself must preserve all full-stream states, not just
    # the subset on which expensive conditional values were evaluated.
    if reference is not None and (max_stock_error>2e-12 or max_factor_error>2e-13):
        raise ValueError('Conditional replay changed saved physical states or random-stream order')
    conditional_payoff=float(market.discount_curve.discount(proxy.maturity))*_conditional_payoffs(mu,variance,proxy.strike,proxy.is_call)
    result={'absolute_path_indices':index,'conditional_european_payoff':conditional_payoff,
        'conditional_european_martingale':martingale,'conditional_european_estimate':conditional_payoff-martingale,
        'physical_time_grid':grid}
    if retain_primitives:result.update(conditional_log_means=means,conditional_variances=clocks,
        conditional_expected_next=expected,conditional_actual_next=actual,
        replay_physical_stocks=physical_stocks,replay_physical_factors=physical_factors)
    diagnostic={'seed':int(seed),'full_stream_paths':int(paths),'evaluated_indices':index.tolist(),
        'all_random_draws_consumed':True,'same_batch_size':proxy.config.batch_size,
        'orthogonal_stock_noise_integrated':True,'both_estimators_target_same_discrete_european_value':True,
        'continuous_price_used_to_center':False,'empirical_valuation_mean_subtracted':False,
        'max_full_stream_stock_error':max_stock_error,'max_full_stream_factor_error':max_factor_error,
        'physical_steps':len(grid)-1,'frozen_exercise_policy_changed':False}
    return result,diagnostic


def shared_european_control(arrays,conditional):
    index=conditional['absolute_path_indices']
    realized=arrays['payoff'][index,-1]-arrays['european_martingale'][index,-1]
    return realized-conditional['conditional_european_estimate']


__all__=['conditional_european_replay','shared_european_control']
