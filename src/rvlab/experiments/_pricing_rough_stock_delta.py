"""Independent frozen-policy stock-bump diagnostics for the no-cash lift.

This research estimator re-applies the decision rule at every stock bump; it
does not differentiate a recorded stopping time. A terminal European payoff
minus its conditional Gaussian expectation is a zero-mean control under the
same discrete law, even when the American policy stopped earlier. It is not
an optional-stopping martingale and does not preserve pathwise price bounds.
Its coefficient is declared before valuation (one by default), never fitted
on the reported paths. These are finite-bump sensitivities of a frozen lower
policy, not certified derivatives of the optimal American value.
"""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral, Real

import numpy as np
import pandas as pd

from rvlab.pricing.rough_estimators import _AffineSurrogate, _conditional_payoffs
from rvlab.pricing.rough_policy_diagnostics import _policy_stops
from rvlab.pricing.rough_simulation import RoughSimulationConfig, simulate_rough_heston
from rvlab.pricing.rough_stopping_fourier import _affine_values


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _bump_levels(levels):
    if not isinstance(levels, (list, tuple)) or len(levels) < 2:
        raise ValueError("bump_relatives must contain at least two descending positive levels")
    result=[]
    for value in levels:
        if (isinstance(value, (bool,np.bool_)) or not isinstance(value,Real)
                or not np.isfinite(value) or value <= 0 or value >= .25):
            raise ValueError("bump_relatives must contain finite positive levels below .25")
        result.append(float(value))
    if any(a <= b for a,b in zip(result,result[1:])):
        raise ValueError("bump_relatives must be strictly descending")
    return tuple(result)


def _summary(samples):
    values=np.asarray(samples,float)
    if values.ndim != 1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError("samples must be a finite path vector with at least two paths")
    mean=float(values.mean()); se=float(values.std(ddof=1)/np.sqrt(len(values)))
    return {"mean":mean,"standard_error":se,"ci_half_width":1.96*se,
        "interval_low":mean-1.96*se,"interval_high":mean+1.96*se}


def terminal_conditional_control(simulation, *, initial_spot, spots, strike, is_call):
    """A separately valid zero-mean control at every bumped inception stock.

    Conditional stock moments retain the *full* realized variance-driver and
    stock-splitting-order paths. Integrating them does not replace rough
    variance by its average, alter exercise, or assume an analytic Heston
    mean. Positive cash dividends invalidate the multiplicative construction.
    """
    if simulation.conditional_log_spots is None or simulation.conditional_variances is None:
        raise ValueError("terminal conditional control requires explicitly no cash dividends")
    spots=np.asarray(spots,float)
    if (spots.ndim != 1 or not len(spots) or not np.isfinite(spots).all()
            or np.any(spots<=0) or not np.isfinite(initial_spot) or initial_spot<=0):
        raise ValueError("initial_spot and spots must be finite positive values")
    growth=spots/initial_spot
    terminal=simulation.spots[:,-1,None]*growth[None,:]
    raw=np.maximum((1 if is_call else -1)*(terminal-strike),0.)
    conditional=_conditional_payoffs(simulation.conditional_log_spots[:,-1,None]
        +np.log(growth)[None,:],simulation.conditional_variances[:,-1,None],strike,is_call)
    discount=float(simulation.discounts[-1])
    return discount*(raw-conditional)


def frozen_policy_delta_samples(frozen, *, seed, paths, bump_relatives=(.01,.005,.0025),
        control_coefficient=1., steps=None):
    """Revalue all frozen decisions on fresh paths and return paired bump data.

    ``steps`` can add time refinement while retaining the original eligible
    exercise dates. Changing steps changes the random-stream interpretation;
    separate step studies therefore require separate independent seeds.
    """
    _integer(seed,"seed",0); _integer(paths,"paths",2)
    levels=_bump_levels(bump_relatives)
    if (isinstance(control_coefficient,(bool,np.bool_))
            or not isinstance(control_coefficient,Real) or not np.isfinite(control_coefficient)):
        raise ValueError("control_coefficient must be a finite predeclared real")
    proxy=frozen["proxy"]
    if proxy.market.dividends.events:
        raise ValueError("frozen stock-delta study supports an explicitly empty cash schedule")
    forbidden={proxy.config.training_seed,proxy.config.valuation_seed,frozen["weights"].training_seed}
    for name in ("original_policy","selected_policy"):
        forbidden.update(frozen[name].training_seeds)
    if seed in forbidden:
        raise ValueError("stock-delta valuation must be independent of all fitting and reserved streams")
    if steps is not None: _integer(steps,"steps",1)
    actual_steps=proxy.config.steps if steps is None else int(steps)
    simulation=simulate_rough_heston(proxy.params,proxy.market,proxy.kernel,
        observation_times=proxy.exercise_times,
        config=RoughSimulationConfig(paths=paths,steps=actual_steps,seed=seed,
            batch_size=proxy.config.batch_size,memory_budget_mb=proxy.config.memory_budget_mb,
            stock_integral=proxy.config.stock_integral))
    s0=proxy.market.spot
    relative=np.asarray(sorted(set([0.]+[-b for b in levels]+list(levels))))
    spots=s0*(1.+relative)
    stock=simulation.spots[:,:,None]*(spots/s0)[None,None,:]
    eligible=np.ones(len(proxy.exercise_times),bool)
    affine=_AffineSurrogate(proxy.params,proxy.market,proxy.kernel,
        np.array([proxy.strike]),np.array([proxy.maturity]),np.array([proxy.is_call]),
        proxy.exercise_times,proxy.control_config)
    euro=np.empty_like(stock)
    for j,time in enumerate(proxy.exercise_times):
        euro[:,j]=_affine_values(affine,float(time),simulation.factors[:,j,None,:],
            np.log(stock[:,j])[:,None,None,:],np.zeros((paths,1,1,len(spots))))[:,0,0]
    payoff=simulation.discounts[None,:,None]*np.maximum(
        (1 if proxy.is_call else -1)*(stock-proxy.strike),0.)
    terminal_control=terminal_conditional_control(simulation,initial_spot=s0,
        spots=spots,strike=proxy.strike,is_call=proxy.is_call)
    samples={}; stops={}; rows=[]
    path=np.arange(paths)[:,None]; bump=np.arange(len(spots))[None,:]
    for name in ("original_policy","selected_policy"):
        stop=_policy_stops(frozen[name],proxy.exercise_times,eligible,stock,simulation.factors,euro)
        raw=payoff[path,stop,bump]
        controlled=raw-float(control_coefficient)*terminal_control
        stops[name]=stop
        samples[name]={"raw_prices":raw,"controlled_prices":controlled}
        for level in levels:
            low=int(np.flatnonzero(relative==-level)[0]); high=int(np.flatnonzero(relative==level)[0])
            center=int(np.flatnonzero(relative==0)[0]); width=s0*level
            for estimator,values in (("raw",raw),("terminal conditional control",controlled)):
                key=f"{estimator}:{level:g}"
                delta=(values[:,high]-values[:,low])/(2*width)
                samples[name][key]=delta
                summary=_summary(delta); allowance=.002+.005*abs(summary["mean"])
                rows.append({"policy":name,"estimator":estimator,"bump_relative":level,
                    "seed":int(seed),"paths":int(paths),"steps":actual_steps,**summary,
                    "delta_allowance":allowance,"sampling_passed":summary["ci_half_width"]<=allowance,
                    "exercise_change_fraction":float(np.mean(stop[:,low]!=stop[:,high])),
                    "one_sided_left":float(np.mean((values[:,center]-values[:,low])/width)),
                    "one_sided_right":float(np.mean((values[:,high]-values[:,center])/width)),
                    "optimal_delta_ready":False})
    comparisons=[]
    for name in samples:
        for estimator in ("raw","terminal conditional control"):
            for coarse,fine in zip(levels[:-1],levels[1:]):
                values=samples[name][f"{estimator}:{fine:g}"]-samples[name][f"{estimator}:{coarse:g}"]
                comparisons.append({"comparison":"bump refinement","policy":name,"estimator":estimator,
                    "coarse_bump":coarse,"fine_bump":fine,**_summary(values)})
    for estimator in ("raw","terminal conditional control"):
        for level in levels:
            key=f"{estimator}:{level:g}"
            comparisons.append({"comparison":"selected minus original","policy":"paired policies",
                "estimator":estimator,"coarse_bump":np.nan,"fine_bump":level,
                **_summary(samples["selected_policy"][key]-samples["original_policy"][key])})
    return {"estimates":pd.DataFrame(rows),"comparisons":pd.DataFrame(comparisons),
        "samples":samples,"stops":stops,"relative_spots":relative,
        "terminal_control":terminal_control,"simulation":simulation,
        "diagnostics":{"seed":int(seed),"paths":int(paths),"steps":actual_steps,
            "independent_training_valuation":True,"valuation_used_for_fitting":False,
            "fixed_parameters_factors_discount_dividends":True,
            "decisions_reapplied_at_each_bump":True,"stopping_times_held_fixed":False,
            "control_coefficient":float(control_coefficient),
            "control_center":"conditional Gaussian expectation under actual discrete law",
            "target":"finite central differences of each frozen exercise-policy lower value",
            "pathwise_price_bounds_preserved":False,"optimal_delta_ready":False,
            "tolerances_changed":False}}


def frozen_policy_martingale_delta_samples(frozen, *, seed, paths,
        bump_relatives=(.01,.005,.0025), steps=None):
    """Use the already-audited common stopped martingale at every stock bump.

    Every coefficient is frozen before the valuation seed. Re-applying both
    policies to one seven-stock trace gives valid paired policy differences,
    bump differences, and the raw/control comparison. No price bound is
    converted into a derivative bound. Valuation grids retain all original
    continuation/exercise dates when an independent time refinement is added.
    """
    from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
    from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy
    _integer(seed,"seed",0); _integer(paths,"paths",2)
    levels=_bump_levels(bump_relatives)
    proxy=frozen["proxy"]
    if proxy.market.dividends.events:
        raise ValueError("frozen stock-delta study supports an explicitly empty cash schedule")
    forbidden={proxy.config.training_seed,proxy.config.valuation_seed,frozen["weights"].training_seed}
    for name in ("original_policy","selected_policy"):
        forbidden.update(frozen[name].training_seeds)
    if seed in forbidden:
        raise ValueError("stock-delta valuation must be independent of all fitting and reserved streams")
    if steps is not None: _integer(steps,"steps",1)
    actual_steps=proxy.config.steps if steps is None else int(steps)
    config=replace(proxy.config,valuation_seed=int(seed),valuation_paths=int(paths),steps=actual_steps)
    relative=np.asarray(sorted(set([0.]+[-b for b in levels]+list(levels))))
    s0=proxy.market.spot; spots=s0*(1.+relative)
    valuation=value_fourier_stopping_policy(proxy,config=config,spots=spots,trace=True)
    trace=valuation["trace"]; samples={}; stops={};rows=[]
    for name in ("original_policy","selected_policy"):
        evaluated=evaluate_trace_policy(trace,policy=frozen[name],martingale_weights=frozen["weights"])
        raw,controlled=evaluated.raw_samples,evaluated.controlled_samples
        stops[name]=evaluated.stop_indices
        samples[name]={"raw_prices":raw,"controlled_prices":controlled}
        for level in levels:
            low=int(np.flatnonzero(relative==-level)[0]); high=int(np.flatnonzero(relative==level)[0])
            center=int(np.flatnonzero(relative==0)[0]); width=s0*level
            for estimator,values in (("raw",raw),("stopped martingale",controlled)):
                key=f"{estimator}:{level:g}"
                delta=(values[:,high]-values[:,low])/(2*width)
                samples[name][key]=delta
                summary=_summary(delta); allowance=.002+.005*abs(summary["mean"])
                rows.append({"policy":name,"estimator":estimator,"bump_relative":level,
                    "seed":int(seed),"paths":int(paths),"steps":actual_steps,**summary,
                    "delta_allowance":allowance,"sampling_passed":summary["ci_half_width"]<=allowance,
                    "exercise_change_fraction":float(np.mean(evaluated.stop_indices[:,low]!=evaluated.stop_indices[:,high])),
                    "one_sided_left":float(np.mean((values[:,center]-values[:,low])/width)),
                    "one_sided_right":float(np.mean((values[:,high]-values[:,center])/width)),
                    "optimal_delta_ready":False})
    comparisons=[]
    for name in samples:
        for estimator in ("raw","stopped martingale"):
            for coarse,fine in zip(levels[:-1],levels[1:]):
                values=samples[name][f"{estimator}:{fine:g}"]-samples[name][f"{estimator}:{coarse:g}"]
                comparisons.append({"comparison":"bump refinement","policy":name,"estimator":estimator,
                    "coarse_bump":coarse,"fine_bump":fine,**_summary(values)})
    for estimator in ("raw","stopped martingale"):
        for level in levels:
            key=f"{estimator}:{level:g}"
            comparisons.append({"comparison":"selected minus original","policy":"paired policies",
                "estimator":estimator,"coarse_bump":np.nan,"fine_bump":level,
                **_summary(samples["selected_policy"][key]-samples["original_policy"][key])})
    return {"estimates":pd.DataFrame(rows),"comparisons":pd.DataFrame(comparisons),
        "samples":samples,"stops":stops,"relative_spots":relative,
        "diagnostics":{"seed":int(seed),"paths":int(paths),"steps":actual_steps,
            "actual_time_steps":int(valuation["diagnostics"]["actual_time_steps"]),
            "independent_training_valuation":True,"valuation_used_for_fitting":False,
            "fixed_parameters_factors_discount_dividends":True,
            "decisions_reapplied_at_each_bump":True,"stopping_times_held_fixed":False,
            "same_common_martingale_for_policies":True,
            "martingale_weights":frozen["weights"].weights.tolist(),
            "target":"finite central differences of each frozen exercise-policy lower value",
            "optimal_delta_ready":False,"tolerances_changed":False}}
