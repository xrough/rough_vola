"""Independent identities for shared zero-mean price-control experiments.

All physical random draws and conditional clocks are reconstructed. Affine
conditional means receive bounded, fixed-state probes rather than an expensive
second all-path Fourier valuation. Sample means never establish centering.
"""
from numbers import Integral
import math
import numpy as np
from scipy.special import ndtr,ndtri

from rvlab.pricing.rough_estimators import _AffineSurrogate
from rvlab.pricing.rough_simulation import (
    _time_grid,_theta,_drift_flow_with_integral,_weak_transition_branches)


def _close(actual,expected,message,*,atol=2e-12):
    if np.shape(actual)!=np.shape(expected) or not np.allclose(actual,expected,rtol=0.,atol=atol):
        raise ValueError(message)


def validate_conditional_european(arrays,bundle,*,seed,paths):
    """Audit an entire new stream and return its shared European control."""
    for value,name,minimum in ((seed,'seed',0),(paths,'paths',2)):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    proxy=bundle['proxy'];params=proxy.params;market=proxy.market;kernel=proxy.kernel
    if market.dividends.events or params.vol_of_vol<=0 or proxy.config.steps!=128:
        raise ValueError('The common price control requires the frozen stochastic no-cash128 law')
    times=proxy.exercise_times;grid=_time_grid(params,market,times,proxy.config.steps)
    nt=len(grid);count=len(kernel.nodes);nodes=np.asarray(kernel.nodes);weights=np.asarray(kernel.weights)
    required={'absolute_path_indices','conditional_european_payoff','conditional_european_martingale',
        'conditional_european_estimate','physical_time_grid','conditional_log_means','conditional_variances',
        'conditional_expected_next','conditional_actual_next','replay_physical_stocks','replay_physical_factors',
        'stocks','factors','actual_time_grid','payoff','european_martingale'}
    if not required.issubset(arrays):raise ValueError('Missing common-control conditional or physical primitives')
    if (not np.array_equal(arrays['absolute_path_indices'],np.arange(paths))
            or not np.array_equal(arrays['physical_time_grid'],grid)
            or not np.array_equal(arrays['actual_time_grid'],grid)):
        raise ValueError('The complete original random stream and physical grid must be retained')
    fields=('conditional_log_means','conditional_variances','conditional_expected_next',
        'conditional_actual_next','replay_physical_stocks')
    if (any(np.shape(arrays[k])!=(paths,nt) or not np.isfinite(arrays[k]).all() for k in fields)
            or np.shape(arrays['replay_physical_factors'])!=(paths,nt,count)
            or not np.isfinite(arrays['replay_physical_factors']).all()
            or np.any(arrays['conditional_variances']<0)
            or any(np.shape(arrays[k])!=(paths,) or not np.isfinite(arrays[k]).all() for k in
                ('conditional_european_payoff','conditional_european_martingale','conditional_european_estimate'))):
        raise ValueError('Malformed or nonfinite full-path common-control primitives')
    if (np.shape(arrays['stocks'])!=(paths,len(times),1)
            or np.shape(arrays['factors'])!=(paths,len(times),count)
            or np.shape(arrays['payoff'])!=(paths,len(times))
            or np.shape(arrays['european_martingale'])!=(paths,len(times))
            or not all(np.isfinite(arrays[k]).all() for k in ('stocks','factors','payoff','european_martingale'))):
        raise ValueError('Full original inception states, cashflows and European martingale required')
    mu=np.full(paths,math.log(market.spot));variance=np.zeros(paths)
    factor=np.zeros((paths,count));stock=np.full(paths,market.spot)
    _close(arrays['conditional_log_means'][:,0],mu,'Conditional log means must start at the actual spot')
    for name in ('conditional_variances','conditional_expected_next','conditional_actual_next','replay_physical_factors'):
        if np.any(arrays[name][:,0]!=0):raise ValueError('Common-control primitives must start at zero')
    _close(arrays['replay_physical_stocks'][:,0],stock,'Physical replay has another inception stock')
    matrix=-np.diag(nodes)-params.kappa*np.ones((count,1))*weights
    rng=np.random.default_rng(seed);cache={};probes=[];mass=0.
    affine=_AffineSurrogate(params,market,kernel,np.array([proxy.strike]),np.array([proxy.maturity]),
        np.array([proxy.is_call]),grid,proxy.control_config)
    probe_dates={1,32,64,128}&set(range(1,nt));probe_paths=np.arange(min(2,paths))
    max_stock_error=0.;max_factor_error=0.;max_mu_error=0.;max_clock_error=0.
    for k,(left,right) in enumerate(zip(grid[:-1],grid[1:]),1):
        dt=float(right-left);theta=_theta(params,.5*(left+right));key=(dt,theta)
        if key not in cache:
            cache[key]=_drift_flow_with_integral(matrix,np.full(count,params.kappa*(theta-params.v0)),weights,params.v0,dt/2)
        flow=cache[key];growth=math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        for begin in range(0,paths,proxy.config.batch_size):
            end=min(begin+proxy.config.batch_size,paths);old=factor[begin:end]
            random=rng.random((end-begin,3))
            law=_weak_transition_branches(old,params,nodes=nodes,weights=weights,flow=flow[:count,:count],
                shift=flow[:count,-1],integrated_flow=flow,dt=dt,theta=theta,growth=growth,stock_integral=proxy.config.stock_integral)
            probability=law['probabilities'];mass=max(mass,float(abs(probability.sum(axis=1)-1).max()))
            choice=(random[:,0]>probability[:,0]).astype(int)+(random[:,0]>probability[:,:2].sum(axis=1)).astype(int)
            order=(random[:,1]>=.5).astype(int);index=np.arange(end-begin)
            means=mu[begin:end,None,None]+law['correlated'][:,:,None]-.5*law['orthogonal']
            clocks=variance[begin:end,None,None]+law['orthogonal']
            normal=ndtri(np.clip(random[:,2],np.finfo(float).tiny,1-np.finfo(float).eps))
            stock_means=np.log(stock[begin:end])[:,None,None]+law['correlated'][:,:,None]-.5*law['orthogonal']
            stock[begin:end]=np.exp(stock_means[index,choice,order]+np.sqrt(law['orthogonal'][index,choice,order])*normal)
            factor[begin:end]=law['new_factors'][index,choice]
            mu[begin:end]=means[index,choice,order];variance[begin:end]=clocks[index,choice,order]
            wanted=probe_paths[(probe_paths>=begin)&(probe_paths<end)]
            if k in probe_dates and len(wanted):
                local=wanted-begin
                value=affine.values(float(right),law['new_factors'][local],means[local],clocks[local],np.array([0]))[...,0]
                expected=np.sum(value.mean(axis=2)*probability[local],axis=1)
                actual=affine.values(float(right),factor[wanted,None],mu[wanted,None,None],variance[wanted,None,None],np.array([0]))[:,0,0,0]
                e=float(abs(expected-arrays['conditional_expected_next'][wanted,k]).max())
                a=float(abs(actual-arrays['conditional_actual_next'][wanted,k]).max())
                if max(e,a)>2e-11:raise ValueError('Saved conditional affine mean failed the fixed-state probe')
                probes.append({'physical_index':k,'path_indices':wanted.tolist(),'expected_error':e,'actual_error':a,
                    'all_path_clocks_and_accumulation_validated':True,'exhaustive_affine_recomputation':False})
        max_stock_error=max(max_stock_error,float(abs(stock-arrays['replay_physical_stocks'][:,k]).max()))
        max_factor_error=max(max_factor_error,float(abs(factor-arrays['replay_physical_factors'][:,k]).max()))
        max_mu_error=max(max_mu_error,float(abs(mu-arrays['conditional_log_means'][:,k]).max()))
        max_clock_error=max(max_clock_error,float(abs(variance-arrays['conditional_variances'][:,k]).max()))
    if max(max_stock_error,max_mu_error)>2e-12 or max(max_factor_error,max_clock_error)>2e-13 or mass>1e-12:
        raise ValueError('Common-control random order, factor memory, log mean or variance clock changed')
    exercise=np.searchsorted(grid,times)
    _close(arrays['replay_physical_stocks'][:,exercise,None],arrays['stocks'],'Physical replay differs from saved exercise stock states')
    _close(arrays['replay_physical_factors'][:,exercise],arrays['factors'],'Physical replay differs from saved exercise factor states',atol=2e-13)
    std=np.sqrt(variance);forward=np.exp(mu+.5*variance)
    d2=np.divide(mu-np.log(proxy.strike),std,out=np.zeros(paths),where=std>0);d1=d2+std
    conditional=np.where(proxy.is_call,forward*ndtr(d1)-proxy.strike*ndtr(d2),
        proxy.strike*ndtr(-d2)-forward*ndtr(-d1))
    terminal=np.maximum((1 if proxy.is_call else -1)*(forward-proxy.strike),0.)
    conditional=float(market.discount_curve.discount(proxy.maturity))*np.where(variance==0,terminal,conditional)
    _close(arrays['conditional_european_payoff'],conditional,'Conditional European payoff is not the exact Gaussian terminal expectation')
    martingale=np.sum(arrays['conditional_actual_next'][:,1:]-arrays['conditional_expected_next'][:,1:],axis=1)
    _close(arrays['conditional_european_martingale'],martingale,'Conditional martingale does not accumulate its actual-minus-expected increments')
    estimate=conditional-martingale
    _close(arrays['conditional_european_estimate'],estimate,'Conditional controlled-European estimate changed its units or centering')
    control=arrays['payoff'][:,-1]-arrays['european_martingale'][:,-1]-estimate
    return {'shared_european_control':control,'conditional_payoff':conditional,'conditional_martingale':martingale,
        'conditional_estimate':estimate,'probes':probes,'max_stock_error':max_stock_error,
        'max_factor_error':max_factor_error,'max_log_mean_error':max_mu_error,'max_clock_error':max_clock_error,
        'max_probability_error':mass,'empirical_mean_used_to_center':False}


def validate_common_shift(lower,upper,features,coefficients,*,shifted_lower=None,shifted_upper=None,saved_shift=None):
    """Reconstruct an unrecentered, common translation and invariant gaps."""
    lower=np.asarray(lower,float);upper=np.asarray(upper,float);features=np.asarray(features,float)
    coefficient=np.asarray(coefficients,float)
    if (lower.ndim!=1 or upper.shape!=lower.shape or len(lower)<2 or features.ndim!=2
            or features.shape[0]!=len(lower) or features.shape[1]<1 or coefficient.shape!=(features.shape[1],)
            or not all(np.isfinite(v).all() for v in (lower,upper,features,coefficient)) or np.any(upper<lower-2e-11)):
        raise ValueError('Finite full-path endpoints and frozen common coefficients required')
    shift=features@coefficient;l=lower-shift;u=upper-shift
    if saved_shift is not None:_close(saved_shift,shift,'Common control was recentered or changed its frozen coefficients')
    if shifted_lower is not None:_close(shifted_lower,l,'Lower endpoint did not use the same common control')
    if shifted_upper is not None:_close(shifted_upper,u,'Upper endpoint did not use the same common control')
    _close(u-l,upper-lower,'A common price control changed a stopping gap')
    return {'shift':shift,'lower':l,'upper':u,'gap':upper-lower,'all_paths_retained':True,
        'gap_unchanged':True,'empirical_valuation_mean_subtracted':False}


__all__=['validate_conditional_european','validate_common_shift']
