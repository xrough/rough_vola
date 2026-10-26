"""Coefficient-gather implementation of the unchanged stock-node control.

This separate implementation preserves the initial unoptimized source. It
evaluates each factor-dependent stock polynomial once, gathering only the
interval needed at each node instead of expanding all intervals per node.
"""
from dataclasses import replace
from numbers import Integral,Real
import numpy as np

from ._pricing_rough_local_control import _linear_coefficients,_normal_interval,_state_reliability


def polynomial_node_values(model,nodes,factors):
    """Evaluate exactly the frozen piecewise polynomial with shared factors."""
    coefficient=model.coefficients(factors)
    if model.knots:
        index=np.searchsorted(model.knots,nodes,side='right')
        coefficient=coefficient[...,index,:]
    else:
        coefficient=coefficient[...,None,:]
    return coefficient[...,0]+nodes*(coefficient[...,1]+nodes*coefficient[...,2])


def refined_stock_knots(coarse,multiple):
    """Subdivide existing positive-stock intervals; retain the zero-stock tail."""
    coarse=np.asarray(coarse,float)
    if (coarse.ndim!=1 or len(coarse)<2 or not np.isfinite(coarse).all() or np.any(coarse<=0)
            or np.any(np.diff(coarse)<=0) or isinstance(multiple,bool)
            or not isinstance(multiple,Integral) or multiple not in (4,8)):
        raise ValueError("Positive ordered stock knots and four/eight subdivisions required")
    return np.r_[coarse[0],np.concatenate([np.linspace(a,b,multiple+1)[1:] for a,b in zip(coarse[:-1],coarse[1:])])]


def interpolate_nodes(nodes,values,targets):
    """Deterministic linear interpolation with the original constant upper tail."""
    indices=np.clip(np.searchsorted(nodes,targets,side="right")-1,0,len(nodes)-2)
    weight=(targets-nodes[indices])/(nodes[indices+1]-nodes[indices])
    result=values[...,indices]+weight*(values[...,indices+1]-values[...,indices])
    return np.where(targets>=nodes[-1],values[...,-1,None],result)


def snell_refinement_coefficients(proxy,selected,j,factors,*,fine_nodes,reliability_scale=3.):
    """Fine-minus-coarse selected Snell premiums on one shared nested grid."""
    coarse_nodes=np.r_[0.,proxy.config.stock_knots]
    coarse_positions=np.searchsorted(fine_nodes,coarse_nodes)
    if not np.array_equal(fine_nodes[coarse_positions],coarse_nodes):raise ValueError("Coarse nodes must be exact fine-grid members")
    if j==len(proxy.continuation_times)-1:
        return np.zeros(factors.shape[:-1]+(2,len(fine_nodes),3)),np.zeros(factors.shape[:-1]+(len(fine_nodes),))
    # European factor/Fourier arrays are shared across all node resolutions.
    evaluator=replace(proxy.value_premiums[j],stock_nodes=fine_nodes)
    euro=evaluator.european_values(factors)
    stocks=proxy.strike*fine_nodes
    q=polynomial_node_values(selected.continuations[j],fine_nodes,factors)
    q=np.maximum(q,0.) if selected.premium_cap is None else np.clip(q,0.,selected.premium_cap)
    payoff=np.maximum((1 if proxy.is_call else -1)*(stocks-proxy.strike),0.)
    premium=np.maximum(q,payoff-euro)
    premium=np.maximum(premium,0.) if selected.premium_cap is None else np.clip(premium,0.,selected.premium_cap)
    coarse=interpolate_nodes(coarse_nodes,premium[...,coarse_positions],fine_nodes)
    reliability=_state_reliability(selected.continuations[j],factors,reliability_scale)
    corrections=[]
    for multiple in (4,8):
        nodes=np.r_[0.,refined_stock_knots(proxy.config.stock_knots,multiple)]
        positions=np.searchsorted(fine_nodes,nodes)
        if not np.array_equal(fine_nodes[positions],nodes):raise ValueError("Refinement levels must nest exactly")
        refined=interpolate_nodes(nodes,premium[...,positions],fine_nodes)
        corrections.append(_linear_coefficients(fine_nodes,(refined-coarse)*reliability[...,None]))
    return np.stack(corrections,axis=-3),premium


def signed_shared_expectations(mean,variance,coefficients,*,strike,knots):
    """Integrate multiple signed splines with one shared Gaussian moment table."""
    mean,variance=np.broadcast_arrays(np.asarray(mean,float),np.asarray(variance,float))
    coeff=np.asarray(coefficients,float)
    if (coeff.shape[-1]!=3 or np.any(coeff[...,2]!=0) or np.any(variance<0)
            or not np.isfinite(strike) or strike<=0 or np.any(np.asarray(knots)<=0) or np.any(np.diff(knots)<=0)
            or not all(np.isfinite(x).all() for x in (mean,variance,coeff))):
        raise ValueError("Finite linear splines and nonnegative Gaussian clocks required")
    # mean=(path,branch,ordering); coeff=(path,branch,level,interval,3).
    logmean=mean-np.log(strike);root=np.sqrt(np.maximum(variance,np.finfo(float).tiny))
    bounds=np.r_[-np.inf,np.log(knots),np.inf]
    left=(bounds[:-1]-logmean[...,None])/root[...,None]
    right=(bounds[1:]-logmean[...,None])/root[...,None]
    p=_normal_interval(left,right)
    first=np.exp(logmean+.5*variance)[...,None]*_normal_interval(left-root[...,None],right-root[...,None])
    value=np.sum(coeff[...,None,:,0]*p[...,None,:,:]+coeff[...,None,:,1]*first[...,None,:,:],axis=-1)
    zero=variance==0
    if np.any(zero):
        z=np.exp(logmean);index=np.searchsorted(knots,z,side="right")
        c=np.broadcast_to(coeff[...,None,:,:],coeff.shape[:-2]+(mean.shape[-1],)+coeff.shape[-2:])
        indices=np.broadcast_to(index[...,None,:,None,None],c.shape[:-2]+(1,1))
        actual=np.take_along_axis(c,indices,axis=-2)[...,0,:]
        deterministic=actual[...,0]+actual[...,1]*z[...,None,:]
        value=np.where(zero[...,None,:],deterministic,value)
    if not np.isfinite(value).all():raise FloatingPointError("Shared signed Gaussian expectation nonfinite")
    return value


def node_refinement_martingales(proxy,selected,trace,*,widths=(None,.1,.3),reliability_scale=3.,batch_size=64):
    """Two nested stock-grid residuals with predictable exercise-margin gates."""
    from ._pricing_rough_exercise_block import exercise_block_law
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    from rvlab.pricing.rough_simulation import _time_grid
    if (isinstance(batch_size,bool) or not isinstance(batch_size,Integral) or batch_size<1
            or not widths or len(set(widths))!=len(widths)
            or isinstance(reliability_scale,bool) or not isinstance(reliability_scale,Real)
            or not np.isfinite(reliability_scale) or reliability_scale<=0
            or any(w is not None and (isinstance(w,bool) or not isinstance(w,Real) or not np.isfinite(w) or w<=0) for w in widths)):
        raise ValueError("Declare a positive batch and distinct positive/None gate widths")
    times=np.asarray(trace['times']);grid=np.asarray(trace['actual_time_grid'])
    stock=np.asarray(trace['stocks']);factors=np.asarray(trace['factors']);euro=np.asarray(trace['european_proxy'])
    if (not np.array_equal(times,proxy.continuation_times) or not np.array_equal(times,selected.times)
            or trace.get('policy_fingerprint')!=fourier_policy_fingerprint(proxy)
            or fourier_policy_fingerprint(selected.base_policy)!=fourier_policy_fingerprint(proxy)
            or selected.base_policy.exercise_style!='american'
            or trace.get('kernel_fingerprint')!=proxy.kernel.fingerprint or trace.get('strike')!=proxy.strike
            or not np.array_equal(grid,_time_grid(proxy.params,proxy.market,proxy.continuation_times,proxy.config.steps))
            or not np.array_equal(trace.get('initial_spots'),np.array([proxy.market.spot]))
            or not np.array_equal(trace.get('discounts'),proxy.market.discount_curve.discount(times))
            or stock.ndim!=3 or stock.shape[-1]!=1 or euro.shape!=stock.shape
            or factors.shape!=(len(stock),len(times),len(proxy.kernel.nodes))
            or not all(np.isfinite(x).all() for x in (stock,factors,euro)) or np.any(stock<=0)):
        raise ValueError('Trace changes frozen model, market, calendar or full-factor states')
    fine_nodes=np.r_[0.,refined_stock_knots(proxy.config.stock_knots,8)]
    result=np.zeros(stock.shape[:2]+(2*len(widths),));maxmass=0.
    for j,(left,right) in enumerate(zip(times[:-1],times[1:])):
        a,b=np.searchsorted(grid,[left,right])
        if grid[a]!=left or grid[b]!=right or not 1<=b-a<=2:raise ValueError('One/two existing physical steps required')
        for start in range(0,len(stock),batch_size):
            end=min(start+batch_size,len(stock));old=factors[start:end,j];s=stock[start:end,j,0]
            law=exercise_block_law(proxy,old,times=grid[a:b+1]);new=law['new_factors']
            maxmass=max(maxmass,float(np.max(abs(law['probabilities'].sum(axis=1)-1))))
            coeff,_=snell_refinement_coefficients(proxy,selected,j+1,new,fine_nodes=fine_nodes,reliability_scale=reliability_scale)
            mean=np.log(s)[:,None,None]+law['log_stock_shifts'];clock=law['orthogonal']
            values=signed_shared_expectations(mean,clock,coeff,strike=proxy.strike,knots=fine_nodes[1:])
            expected=np.sum(values.mean(axis=-1)*law['probabilities'][...,None],axis=1)
            actual_coeff,_=snell_refinement_coefficients(proxy,selected,j+1,factors[start:end,j+1],fine_nodes=fine_nodes,reliability_scale=reliability_scale)
            z=stock[start:end,j+1,0]/proxy.strike;index=np.searchsorted(fine_nodes[1:],z,side='right')
            indices=np.broadcast_to(index[:,None,None,None],actual_coeff.shape[:2]+(1,1))
            c=np.take_along_axis(actual_coeff,indices,axis=-2)[...,0,:]
            actual=c[...,0]+c[...,1]*z[:,None]
            d=float(proxy.market.discount_curve.discount(right));increment=d*(actual-expected)
            ld=float(proxy.market.discount_curve.discount(left));q=selected.continuations[j].value(s,old,proxy.strike)
            q=np.maximum(q,0.) if selected.premium_cap is None else np.clip(q,0.,selected.premium_cap)
            margin=ld*(np.maximum((1 if proxy.is_call else -1)*(s-proxy.strike),0.)-q)-euro[start:end,j,0]
            gate=np.column_stack([np.ones_like(margin) if width is None else np.exp(-.5*(margin/width)**2) for width in widths])
            result[start:end,j+1]=result[start:end,j]+(increment[...,None]*gate[:,None,:]).reshape(end-start,-1)
    if not np.isfinite(result).all():raise FloatingPointError('Stock-grid refinement martingales nonfinite')
    return result,{'max_probability_error':maxmass,'widths':widths,'right_reliability_scale':reliability_scale,
        'subdivisions':(4,8),'positive_nodes':tuple(len(refined_stock_knots(proxy.config.stock_knots,m)) for m in (4,8)),
        'coarse_nodes_exact_subsets':True,'shared_european_values':True,'all_paths_retained':True,
        'right_reliability_inside_conditional_expectation':True,'left_gates_predictable':True,
        'policy_dynamics_exercise_and_physical_steps_unchanged':True}


__all__=['refined_stock_knots','interpolate_nodes','snell_refinement_coefficients',
    'signed_shared_expectations','node_refinement_martingales','polynomial_node_values']
