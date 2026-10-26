"""Exact finite-mixture controls between the original exercise dates.

Every physical weak step remains unchanged. Consecutive one-step mixture laws
are composed, integrating independent Gaussian stock shocks analytically.
Only the martingale observation calendar changes, never exercise eligibility.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import expit
from rvlab.pricing.rough_simulation import _theta,_drift_flow_with_integral,_weak_transition_branches,_nonnegative_clock
from rvlab.pricing.rough_estimators import _AffineSurrogate,_conditional_payoffs
from rvlab.pricing.rough_stopping_fourier import _affine_values,_interpolate,_premium_values,_expected_premium_values,_continuation_coefficients,fourier_policy_fingerprint


def exercise_block_law(policy,factors,*,times):
    """Compose one or two actual weak transitions as a Gaussian mixture.

    The factor branching tree is retained exactly. The Gaussian means are
    log-stock *shifts*, so initial stock enters separately. Stock orderings
    remain equally weighted endpoints at every physical step.
    """
    factors=np.asarray(factors,float);times=np.asarray(times,float)
    if (factors.ndim!=2 or len(factors)<1 or factors.shape[1]!=len(policy.kernel.nodes)
        or not np.isfinite(factors).all() or times.ndim!=1 or not 2<=len(times)<=3
        or not np.isfinite(times).all() or not np.all(np.diff(times)>0)):
        raise ValueError('Declare finite aligned factor states and one or two ordered physical steps')
    if policy.market.dividends.events:
        raise ValueError('Exercise-block controls require explicitly no cash events')
    params=policy.params;nodes=np.asarray(policy.kernel.nodes);w=np.asarray(policy.kernel.weights)
    count=len(nodes);matrix=-np.diag(nodes)-params.kappa*np.ones((count,1))*w
    n=len(factors);branches=factors[:,None,:];probability=np.ones((n,1));mean=np.zeros((n,1,1));variance=np.zeros_like(mean)
    for left,right in zip(times[:-1],times[1:]):
        b=branches.shape[1];orders=mean.shape[2];old=branches.reshape(n*b,count)
        dt=float(right-left);theta=_theta(params,float((left+right)/2))
        flow=_drift_flow_with_integral(matrix,np.full(count,params.kappa*(theta-params.v0)),w,params.v0,dt if params.vol_of_vol==0 else dt/2)
        growth=math.log(policy.market.discount_curve.discount(left)/policy.market.discount_curve.discount(right))
        if params.vol_of_vol==0:
            new=(old@flow[:count,:count].T+flow[:count,-1]).reshape(n,b,1,count)
            clock=_nonnegative_clock(old@flow[count,:count]+flow[count,-1]).reshape(n,b,1,1)
            corr=np.full((n,b,1),growth);p=np.ones((n,b,1))
        else:
            law=_weak_transition_branches(old,params,nodes=nodes,weights=w,flow=flow[:count,:count],
                shift=flow[:count,-1],integrated_flow=flow,dt=dt,theta=theta,growth=growth,stock_integral=policy.config.stock_integral)
            new=law['new_factors'].reshape(n,b,3,count);clock=law['orthogonal'].reshape(n,b,3,2)
            corr=law['correlated'].reshape(n,b,3);p=law['probabilities'].reshape(n,b,3)
        nb=new.shape[2];no=clock.shape[-1]
        mean=(mean[:,:,None,:,None]+corr[:,:,:,None,None]-.5*clock[:,:,:,None,:]).reshape(n,b*nb,orders*no)
        variance=(variance[:,:,None,:,None]+clock[:,:,:,None,:]).reshape(n,b*nb,orders*no)
        probability=(probability[:,:,None]*p).reshape(n,b*nb)
        branches=new.reshape(n,b*nb,count)
    return {'new_factors':branches,'probabilities':probability,'log_stock_shifts':mean,'orthogonal':variance,
        'endpoint_probability':1./mean.shape[2],'physical_steps':len(times)-1}


def block_control_trace(policy,source_trace,*,decision_policy=None,batch_size=128):
    """Build a new exactly centered martingale on a shared physical path trace."""
    from rvlab.pricing.rough_policy_diagnostics import _policy_stops
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('batch_size must be positive')
    times=np.asarray(source_trace['times']);grid=np.asarray(source_trace['actual_time_grid'])
    from rvlab.pricing.rough_simulation import _time_grid
    expected_grid=_time_grid(policy.params,policy.market,policy.continuation_times,policy.config.steps)
    if (not np.array_equal(grid,expected_grid)
        or source_trace.get('kernel_fingerprint')!=policy.kernel.fingerprint
        or source_trace.get('strike')!=policy.strike
        or not np.array_equal(source_trace.get('initial_spots'),[policy.market.spot])
        or not np.array_equal(source_trace.get('discounts'),policy.market.discount_curve.discount(times))):
        raise ValueError('Block controls require matching physical law, market inputs, kernel and time grid')
    reference=decision_policy.base_policy if hasattr(decision_policy,'base_policy') else (decision_policy or policy)
    if (source_trace.get('policy_fingerprint')!=fourier_policy_fingerprint(reference)
        or reference.params!=policy.params or reference.control_config!=policy.control_config):
        raise ValueError('Source trace must identify the frozen policy and unchanged European function')
    if not np.array_equal(times,policy.exercise_times):raise ValueError('Original exercise calendar must be retained')
    stock=np.asarray(source_trace['stocks']);factors=np.asarray(source_trace['factors']);euro=np.asarray(source_trace['european_proxy'])
    if stock.ndim!=3 or stock.shape[-1]!=1 or factors.shape[:2]!=stock.shape[:2]:
        raise ValueError('Block research trace requires one inception stock and full factor states')
    affine=_AffineSurrogate(policy.params,policy.market,policy.kernel,np.array([policy.strike]),
        np.array([policy.maturity]),np.array([policy.is_call]),grid,policy.control_config)
    em=np.zeros_like(stock);pm=np.zeros_like(stock);im=np.zeros_like(stock);bm=np.zeros_like(stock)
    maxmass=0.;maxblock=0
    for j,(left,right) in enumerate(zip(times[:-1],times[1:])):
        a=int(np.searchsorted(grid,left));b=int(np.searchsorted(grid,right))
        if grid[a]!=left or grid[b]!=right or not 1<=b-a<=2:
            raise ValueError('Current block engine supports one or two physical steps per exercise interval')
        maxblock=max(maxblock,b-a)
        for start in range(0,len(stock),batch_size):
            end=min(start+batch_size,len(stock));u=factors[start:end,j];s=stock[start:end,j,0]
            law=exercise_block_law(policy,u,times=grid[a:b+1])
            maxmass=max(maxmass,float(np.max(abs(law['probabilities'].sum(axis=1)-1.))))
            mean=np.log(s)[:,None,None]+law['log_stock_shifts'];clock=law['orthogonal'];new=law['new_factors']
            ev=_affine_values(affine,float(right),new,mean[...,None],clock[...,None])[...,0]
            coeff=_interpolate(policy,float(right),new)[:,:,None,None]
            pv=_expected_premium_values(mean[...,None],clock[...,None],coeff,strike=policy.strike,
                stock_knots=policy.config.stock_knots,cap=policy.premium_cap)[...,0]
            iv=_conditional_payoffs(mean,clock,policy.strike,policy.is_call)
            p=law['probabilities']
            expected=lambda v:np.sum(v.mean(axis=2)*p,axis=1)
            nex=factors[start:end,j+1];ns=stock[start:end,j+1,0]
            actualp=_premium_values(ns,_interpolate(policy,float(right),nex),strike=policy.strike,
                stock_knots=policy.config.stock_knots,cap=policy.premium_cap)
            d=float(policy.market.discount_curve.discount(right))
            ei=euro[start:end,j+1,0]-expected(ev)
            pi=d*(actualp-expected(pv));ii=d*(np.maximum((1 if policy.is_call else -1)*(ns-policy.strike),0.)-expected(iv))
            ld=float(policy.market.discount_curve.discount(left))
            q=_premium_values(s,_continuation_coefficients(policy,float(left),u),strike=policy.strike,
                stock_knots=policy.continuations[0].knots,cap=policy.premium_cap)
            indicator=expit((ld*(np.maximum((1 if policy.is_call else -1)*(s-policy.strike),0.)-q)-euro[start:end,j,0])/(ld*.05))
            em[start:end,j+1,0]=em[start:end,j,0]+ei
            pm[start:end,j+1,0]=pm[start:end,j,0]+pi
            im[start:end,j+1,0]=im[start:end,j,0]+ii
            bm[start:end,j+1,0]=bm[start:end,j,0]+indicator*(ii-ei)
    result=dict(source_trace)
    result.update(european_martingale=em,premium_martingale=pm,intrinsic_martingale=im,
        micro_boundary_martingale=bm,micro_poststop_martingale=np.zeros_like(em),
        policy_fingerprint=fourier_policy_fingerprint(policy),policy_training_seed=policy.config.training_seed,
        additional_control_training_seeds=tuple(policy.diagnostics.get('additional_control_training_seeds',())),
        block_control_scope='exact composed physical weak steps; original exercise dates only',
        block_max_probability_error=maxmass,block_max_physical_steps=maxblock,
        micro_boundary_feature_source='LEFT exercise-interval boundary state',micro_boundary_width=.05)
    rule=policy if decision_policy is None else decision_policy
    result['stop_indices']=_policy_stops(rule,times,result['eligible'],stock,factors,euro)
    result['continuation_proxy']=euro.copy()
    discounts=policy.market.discount_curve.discount(times)
    for j,time in enumerate(times):
        q=_premium_values(stock[:,j,0],_continuation_coefficients(policy,float(time),factors[:,j]),
            strike=policy.strike,stock_knots=policy.continuations[0].knots,cap=policy.premium_cap)
        result['continuation_proxy'][:,j,0]+=discounts[j]*q
    if not all(np.isfinite(result[k]).all() for k in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale')):
        raise FloatingPointError('Composed exercise-block control produced nonfinite values')
    return result
