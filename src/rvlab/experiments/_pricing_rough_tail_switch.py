"""Predictable intrinsic/European replacement controls for retained tail gaps.

The first three corrections act strictly after a frozen exercise rule stops,
so they leave that rule's controlled lower sample unchanged path by path.
The broader corrections remain exactly centered but do not have that stronger
identity. These are experiment-local controls, not a new pricing backend.
"""
from numbers import Integral
import numpy as np
from scipy.special import expit

FEATURES=("selected poststop intrinsic", "selected poststop intrinsic soft .1",
    "selected poststop intrinsic soft .3", "boundary intrinsic soft .03",
    "boundary intrinsic soft .1", "boundary intrinsic soft .3", "OTM European")
FAMILIES={"poststop":(0,1,2),"extended":tuple(range(7))}


def tail_switch_corrections(proxy,selected,trace,current_martingale,*,forbidden_seeds=()):
    """Return seven exact-centered corrections using only LEFT information."""
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    from rvlab.pricing.rough_policy_diagnostics import _policy_stops
    from rvlab.pricing.rough_simulation import _time_grid
    times=np.asarray(trace['times']);stock=np.asarray(trace['stocks']);factors=np.asarray(trace['factors'])
    euro=np.asarray(trace['european_proxy']);payoff=np.asarray(trace['discounted_payoffs'])
    em=np.asarray(trace['european_martingale']);im=np.asarray(trace['intrinsic_martingale'])
    m=np.asarray(current_martingale,float);discount=np.asarray(trace['discounts'])
    if (any(isinstance(s,bool) or not isinstance(s,Integral) or s<0 for s in forbidden_seeds)
            or trace.get('seed') in forbidden_seeds
            or trace.get('policy_fingerprint')!=fourier_policy_fingerprint(proxy)
            or fourier_policy_fingerprint(selected.base_policy)!=fourier_policy_fingerprint(proxy)
            or not np.array_equal(times,proxy.continuation_times) or not np.array_equal(times,selected.times)
            or not np.array_equal(trace.get('actual_time_grid'),_time_grid(proxy.params,proxy.market,times,proxy.config.steps))
            or trace.get('kernel_fingerprint')!=proxy.kernel.fingerprint or trace.get('strike')!=proxy.strike
            or not np.array_equal(trace.get('initial_spots'),[proxy.market.spot])
            or not np.array_equal(discount,proxy.market.discount_curve.discount(times))
            or stock.ndim!=3 or stock.shape[-1]!=1 or euro.shape!=stock.shape or payoff.shape!=stock.shape
            or factors.shape!=(len(stock),len(times),len(proxy.kernel.nodes))
            or em.shape!=stock.shape or im.shape!=stock.shape or m.shape!=stock.shape[:2]
            or np.any(stock<=0) or np.any(payoff<0) or np.any(m[:,0]!=0)
            or np.any(em[:,0]!=0) or np.any(im[:,0]!=0)
            or not all(np.isfinite(a).all() for a in (times,stock,factors,euro,payoff,discount,em,im,m))):
        raise ValueError('Tail switches require matching frozen model, full states and independent finite martingales')
    stops=_policy_stops(selected,times,trace['eligible'],stock,factors,euro)[:,0]
    margin=np.empty(m.shape)
    for j in range(len(times)):
        q=selected.continuations[j].value(stock[:,j,0],factors[:,j],proxy.strike)
        q=np.maximum(q,0.) if selected.premium_cap is None else np.clip(q,0.,selected.premium_cap)
        margin[:,j]=(payoff[:,j,0]-euro[:,j,0])/discount[j]-q
    left=margin[:,:-1];post=(np.arange(len(times)-1)[None,:]>=stops[:,None]).astype(float)
    gates=np.stack((post,post*expit(left/.1),post*expit(left/.3),
        expit(left/.03),expit(left/.1),expit(left/.3),(payoff[:,:-1,0]==0).astype(float)),axis=-1)
    intrinsic=np.diff(im[:,:,0]-m,axis=1);european=np.diff(em[:,:,0]-m,axis=1)
    increments=gates*np.concatenate((np.repeat(intrinsic[...,None],6,axis=-1),european[...,None]),axis=-1)
    corrections=np.concatenate((np.zeros((len(m),1,len(FEATURES))),np.cumsum(increments,axis=1)),axis=1)
    if (not np.isfinite(corrections).all() or np.any(gates<0) or np.any(gates>1)
            or np.any(corrections[np.arange(len(m)),stops,:3]!=0)):
        raise FloatingPointError('Tail-switch centering or poststop lower-sample identity failed')
    diagnostics={'features':FEATURES,'families':FAMILIES,'left_margin_widths':[.03,.1,.3],
        'poststop_uses_selected_rule':True,'first_three_corrections_zero_at_selected_stop':True,
        'left_gates_predictable':True,'source_martingales_exact_centered':True,'all_paths_retained':True,
        'no_new_conditional_pricing_approximation':True,'policy_dynamics_and_calendars_unchanged':True}
    return {'corrections':corrections,'stops':stops,'margin':margin,'gates':gates,'diagnostics':diagnostics}


def current_control_workspace(proxy,selected,old_weights,prior_control,grid_control,trace):
    """Reconstruct the exact preceding selected control without refitting it."""
    from ._pricing_rough_local_control import local_correction_martingales,correction_basis
    from ._pricing_rough_node_refinement_fast import node_refinement_martingales
    from rvlab.pricing.rough_dual import apply_rough_dual_weights
    stable=apply_rough_dual_weights(trace,old_weights)['martingale'][:,:,0]
    prior,prior_diagnostics=local_correction_martingales(proxy,selected,trace)
    prior_basis=correction_basis(prior,trace['times'],prior_control['bins'])
    prior_m=stable+np.einsum('ntd,d->nt',prior_basis,np.asarray(prior_control['weights']))
    grid,grid_diagnostics=node_refinement_martingales(proxy,selected,trace)
    grid_basis=correction_basis(grid,trace['times'],grid_control['bins'])
    current=prior_m+np.einsum('ntd,d->nt',grid_basis,np.asarray(grid_control['weights']))
    switches=tail_switch_corrections(proxy,selected,trace,current)
    return {'payoff':trace['discounted_payoffs'][:,:,0],
        'raw':trace['discounted_payoffs'][np.arange(len(current)),switches['stops'],0],
        'stops':switches['stops'],'stable_old_martingale':stable,'prior_local_corrections':prior,
        'prior_martingale':prior_m,'grid_corrections':grid,'current_martingale':current,
        'tail_corrections':switches['corrections'],'times':trace['times'],
        'diagnostics':{'prior_local':prior_diagnostics,'prior_grid':grid_diagnostics,'tail_switch':switches['diagnostics']}}


__all__=['FEATURES','FAMILIES','tail_switch_corrections','current_control_workspace']
