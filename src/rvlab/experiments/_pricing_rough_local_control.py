"""Bounded local Snell corrections added to a frozen stable martingale.

Only synthetic inception, empty cash schedules and one/two physical steps per
original exercise interval are supported. All approximation functions are
frozen before independent control training; no simulated path is clipped.
"""
from __future__ import annotations

from numbers import Integral,Real
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtr


def _normal_interval(left,right):
    return np.where(left>0,ndtr(-left)-ndtr(-right),ndtr(right)-ndtr(left))


def signed_linear_expectation(mean,variance,coefficients,*,strike,knots):
    """Exact Gaussian integral of a signed continuous stock-linear spline."""
    mean,variance=np.broadcast_arrays(np.asarray(mean,float),np.asarray(variance,float))
    coeff=np.asarray(coefficients,float)
    shape=np.broadcast_shapes(mean.shape,coeff.shape[:-2])
    mean=np.broadcast_to(mean,shape);variance=np.broadcast_to(variance,shape)
    coeff=np.broadcast_to(coeff,shape+(len(knots)+1,3))
    if (not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<0)
            or not np.isfinite(coeff).all() or np.any(coeff[...,2]!=0)):
        raise ValueError("Finite Gaussian states and linear spline coefficients required")
    if not np.isfinite(strike) or strike<=0 or np.any(np.diff(knots)<=0) or np.any(np.asarray(knots)<=0):
        raise ValueError("Positive strike and strictly increasing positive knots required")
    logbound=np.r_[-np.inf,np.log(np.asarray(knots,float)),np.inf]
    logmean=mean-np.log(strike);root=np.sqrt(np.maximum(variance,np.finfo(float).tiny))
    left=(logbound[:-1]-logmean[...,None])/root[...,None]
    right=(logbound[1:]-logmean[...,None])/root[...,None]
    p=_normal_interval(left,right)
    first=np.exp(logmean+.5*variance)[...,None]*_normal_interval(left-root[...,None],right-root[...,None])
    value=np.sum(coeff[...,0]*p+coeff[...,1]*first,axis=-1)
    deterministic=variance==0
    if np.any(deterministic):
        z=np.exp(logmean);index=np.searchsorted(knots,z,side="right")
        c=np.take_along_axis(coeff,index[...,None,None],axis=-2)[...,0,:]
        value=np.where(deterministic,c[...,0]+c[...,1]*z,value)
    if not np.isfinite(value).all():raise FloatingPointError("Signed Gaussian spline integral is nonfinite")
    return value


def _linear_coefficients(nodes,values):
    slope=np.diff(values,axis=-1)/np.diff(nodes)
    intercept=values[...,:-1]-slope*nodes[:-1]
    slope=np.concatenate((slope,np.zeros(slope.shape[:-1]+(1,))),axis=-1)
    intercept=np.concatenate((intercept,values[...,-1:]),axis=-1)
    return np.stack((intercept,slope,np.zeros_like(slope)),axis=-1)


def _state_reliability(model,factors,scale):
    standardized=np.max(abs((factors-model.center)/model.scale),axis=-1)
    # At ten scale units exp(-x**8) is zero to machine precision. This limits
    # only the declared bounded approximation feature, never a simulated state.
    return np.exp(-np.minimum(standardized/scale,10.)**8)


def _correction_coefficients(proxy,selected,j,factors,reliability_scale):
    model=proxy.value_premiums[j]
    if float(proxy.continuation_times[j])==proxy.maturity:
        return np.zeros(factors.shape[:-1]+(len(proxy.config.stock_knots)+1,3))
    nodes=model.stock_nodes
    euro=model.european_values(factors)
    old=model.continuation_values(factors)
    stocks=proxy.strike*nodes
    expanded=np.broadcast_to(stocks,factors.shape[:-1]+(len(stocks),))
    new=selected.continuations[j].value(expanded,factors,proxy.strike)
    new=np.maximum(new,0.) if selected.premium_cap is None else np.clip(new,0.,selected.premium_cap)
    payoff=np.maximum((1 if proxy.is_call else -1)*(stocks-proxy.strike),0.)
    old=np.maximum(old,payoff-euro);new=np.maximum(new,payoff-euro)
    if proxy.premium_cap is not None:
        old=np.clip(old,0.,proxy.premium_cap);new=np.clip(new,0.,proxy.premium_cap)
    reliability=_state_reliability(selected.continuations[j],factors,reliability_scale)
    return _linear_coefficients(nodes,(new-old)*reliability[...,None])


def local_correction_martingales(proxy,selected,trace,*,widths=(None,.1,.3),reliability_scale=3.,batch_size=128):
    """Return exact-centered corrections with RIGHT reliability and LEFT gates.

    Reliability is part of the value function at the right state and is
    integrated over every conditional branch. The exercise-margin multiplier
    is measured before the transition, hence preserves conditional centering.
    """
    from ._pricing_rough_exercise_block import exercise_block_law
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    from rvlab.pricing.rough_simulation import _time_grid
    if isinstance(batch_size,bool) or not isinstance(batch_size,Integral) or batch_size<1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(widths,(tuple,list)) or not widths or len(set(widths))!=len(widths):
        raise ValueError("Declare distinct nonempty gate widths")
    times=np.asarray(trace["times"]);grid=np.asarray(trace["actual_time_grid"])
    if (not np.array_equal(times,proxy.continuation_times) or not np.array_equal(times,selected.times)
            or trace.get("policy_fingerprint")!=fourier_policy_fingerprint(proxy)):
        raise ValueError("Local correction requires the declared original proxy and exercise calendar")
    if (isinstance(reliability_scale,bool) or not isinstance(reliability_scale,Real)
            or not np.isfinite(reliability_scale) or reliability_scale<=0):raise ValueError("reliability_scale must be positive")
    if any(w is not None and (isinstance(w,bool) or not isinstance(w,Real) or not np.isfinite(w) or w<=0) for w in widths):raise ValueError("Gate widths must be positive or None")
    stock=np.asarray(trace["stocks"]);factors=np.asarray(trace["factors"]);euro=np.asarray(trace["european_proxy"])
    expected_grid=_time_grid(proxy.params,proxy.market,proxy.continuation_times,proxy.config.steps)
    if (trace.get("kernel_fingerprint")!=proxy.kernel.fingerprint or trace.get("strike")!=proxy.strike
            or not np.array_equal(trace.get("initial_spots"),np.array([proxy.market.spot]))
            or not np.array_equal(trace.get("discounts"),proxy.market.discount_curve.discount(times))
            or not np.array_equal(grid,expected_grid)
            or selected.base_policy.exercise_style!="american"
            or fourier_policy_fingerprint(selected.base_policy)!=fourier_policy_fingerprint(proxy)):
        raise ValueError("Trace model, market, kernel or physical grid differs from the frozen proxy")
    if (stock.ndim!=3 or stock.shape[-1]!=1 or stock.shape[1]!=len(times) or len(stock)<1
            or euro.shape!=stock.shape or factors.shape!=(len(stock),len(times),len(proxy.kernel.nodes))
            or not np.isfinite(stock).all() or np.any(stock<=0) or not np.isfinite(factors).all() or not np.isfinite(euro).all()):
        raise ValueError("Finite aligned full-factor states and one positive inception stock required")
    result=np.zeros(stock.shape[:2]+(len(widths),));maxmass=0.
    knots=proxy.config.stock_knots
    for j,(left,right) in enumerate(zip(times[:-1],times[1:])):
        a=int(np.searchsorted(grid,left));b=int(np.searchsorted(grid,right))
        if grid[a]!=left or grid[b]!=right or not 1<=b-a<=2:raise ValueError("One/two existing physical steps required")
        for start in range(0,len(stock),batch_size):
            end=min(start+batch_size,len(stock));old=factors[start:end,j];s=stock[start:end,j,0]
            law=exercise_block_law(proxy,old,times=grid[a:b+1]);new=law["new_factors"]
            maxmass=max(maxmass,float(np.max(abs(law["probabilities"].sum(axis=1)-1))))
            coefficient=_correction_coefficients(proxy,selected,j+1,new,reliability_scale)
            mean=np.log(s)[:,None,None]+law["log_stock_shifts"];clock=law["orthogonal"]
            expected=signed_linear_expectation(mean,clock,coefficient[:,:,None],strike=proxy.strike,knots=knots)
            expected=np.sum(expected.mean(axis=2)*law["probabilities"],axis=1)
            actual_coeff=_correction_coefficients(proxy,selected,j+1,factors[start:end,j+1],reliability_scale)
            z=stock[start:end,j+1,0]/proxy.strike;index=np.searchsorted(knots,z,side="right")
            actual=np.take_along_axis(actual_coeff,index[:,None,None],axis=-2)[:,0]
            actual=actual[:,0]+actual[:,1]*z
            d=float(proxy.market.discount_curve.discount(right));increment=d*(actual-expected)
            leftd=float(proxy.market.discount_curve.discount(left))
            q=selected.continuations[j].value(s,old,proxy.strike)
            q=np.maximum(q,0.) if selected.premium_cap is None else np.clip(q,0.,selected.premium_cap)
            margin=leftd*(np.maximum((1 if proxy.is_call else -1)*(s-proxy.strike),0.)-q)-euro[start:end,j,0]
            gate=np.column_stack([np.ones_like(margin) if width is None else np.exp(-.5*(margin/width)**2) for width in widths])
            result[start:end,j+1]=result[start:end,j]+increment[:,None]*gate
    if not np.isfinite(result).all():raise FloatingPointError("Local correction martingales nonfinite")
    return result,{"max_probability_error":maxmass,"widths":widths,"reliability_scale":reliability_scale,
        "right_reliability_integrated_inside_branch_expectation":True,"left_gates_predictable":True,
        "all_paths_retained":True,"no_simulation_law_or_exercise_calendar_changes":True}


def correction_basis(corrections,times,bins):
    corrections=np.asarray(corrections,float);times=np.asarray(times,float)
    if (corrections.ndim!=3 or corrections.shape[1]!=len(times) or isinstance(bins,bool)
            or not isinstance(bins,Integral) or bins not in (1,4)
            or not np.isfinite(corrections).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        raise ValueError("Declare finite aligned corrections and one/four bins")
    indices=np.linspace(0,len(times)-1,bins+1,dtype=int)
    increments=np.diff(corrections,axis=1);pieces=[]
    for left,right in zip(indices[:-1],indices[1:]):
        masked=np.zeros_like(increments);masked[:,left:right]=increments[:,left:right]
        pieces.append(np.concatenate((np.zeros((len(corrections),1,corrections.shape[2])),np.cumsum(masked,axis=1)),axis=1))
    return np.concatenate(pieces,axis=2)


def fit_correction_weights(payoff,baseline_martingale,stops,basis,*,bins,penalty=0.,coefficient_budget=1.):
    """Convex sampled gap minimization; zero weights remain a fixed fallback."""
    payoff,baseline_martingale,stops,basis=map(np.asarray,(payoff,baseline_martingale,stops,basis))
    if (payoff.ndim!=2 or basis.ndim!=3 or baseline_martingale.shape!=payoff.shape
            or isinstance(bins,bool) or not isinstance(bins,Integral) or bins not in (1,4)):
        raise ValueError("Aligned correction fit arrays and one/four bins required")
    n,t=payoff.shape;d=basis.shape[2]
    if (basis.shape[:2]!=(n,t) or d%bins or stops.shape!=(n,) or n<2 or stops.dtype.kind not in "iu"
            or np.any(stops<0) or np.any(stops>=t) or not all(np.isfinite(x).all() for x in (payoff,baseline_martingale,basis))
            or isinstance(penalty,bool) or not isinstance(penalty,Real) or not np.isfinite(penalty) or penalty<0
            or isinstance(coefficient_budget,bool) or not isinstance(coefficient_budget,Real)
            or not np.isfinite(coefficient_budget) or not 0<coefficient_budget<=1):raise ValueError("Finite aligned correction fit arrays and bounded coefficients required")
    path=np.arange(n);lower=payoff[path,stops]-baseline_martingale[path,stops]
    basegap=payoff-baseline_martingale-lower[:,None]
    changes=basis[path,stops][:,None,:]-basis
    rows=n*t
    a=sparse.hstack((sparse.csr_matrix(changes.reshape(rows,d)),sparse.csr_matrix((-np.ones(rows),(np.arange(rows),np.repeat(path,t))),shape=(rows,n))),format="csr")
    group=np.zeros((bins,d+n))
    for b in range(bins):group[b,b*(d//bins):(b+1)*(d//bins)]=1.
    a=sparse.vstack((a,sparse.csr_matrix(group)),format="csr");rhs=np.r_[-basegap.ravel(),np.full(bins,coefficient_budget)]
    solved=linprog(np.r_[np.full(d,penalty),np.full(n,1/n)],A_ub=a,b_ub=rhs,
        bounds=[(0.,coefficient_budget)]*d+[(0.,None)]*n,method="highs",
        options={"dual_feasibility_tolerance":1e-9,"primal_feasibility_tolerance":1e-9})
    if not solved.success:return None,{"status":"optimizer_failure","message":solved.message,"iterations":solved.nit}
    return solved.x[:d],{"status":"completed","message":solved.message,"iterations":solved.nit,
        "objective":float(solved.fun),"penalty":penalty,"bins":bins,"coefficient_budget":coefficient_budget}


def correction_endpoints(payoff,baseline_martingale,stops,basis,weights):
    martingale=baseline_martingale+np.einsum("ntd,d->nt",basis,weights)
    raw=payoff[np.arange(len(payoff)),stops]
    lower=raw-martingale[np.arange(len(payoff)),stops];upper=np.max(payoff-martingale,axis=1)
    if np.any(upper<lower-2e-11):raise FloatingPointError("Corrected martingale violated endpoint ordering")
    return lower,upper


__all__=["signed_linear_expectation","local_correction_martingales","correction_basis",
    "fit_correction_weights","correction_endpoints"]
