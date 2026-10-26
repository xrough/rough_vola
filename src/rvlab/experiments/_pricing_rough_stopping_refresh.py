"""Research controls refreshed around an independently fitted exercise rule.

This opt-in helper changes approximation functions, never the transition law,
exercise dates, tolerance or public pricing defaults.  New dual coefficients
must be trained on a separate stream before any final comparison is made.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from rvlab.pricing.rough_obstacle_premium import RoughObstaclePremium


@dataclass(frozen=True)
class TensorObstaclePremium(RoughObstaclePremium):
    """The exact-centered nodal envelope of a frozen tensor continuation."""

    continuation_degree: int | None = None
    continuation_factor_clip: float = 4.
    exercise_obstacle: bool = True

    def continuation_values(self, factors):
        if self.continuation_degree is None:
            return super().continuation_values(factors)
        from rvlab.pricing.rough_tensor_premium import _factor_features
        factors = np.asarray(factors, float)
        feature = _factor_features(factors, self.continuation_center,
            self.continuation_scale, self.continuation_degree,
            self.continuation_factor_clip)
        beta = np.einsum("...f,sf->...s", feature, self.continuation_beta)
        z = self.stock_nodes
        stock = np.stack((np.ones_like(z), z-1., (z-1.)**2), axis=-1)
        if len(self.continuation_knots):
            stock = np.column_stack((stock,
                np.maximum(z[:, None]-self.continuation_knots, 0.)**2))
        value = np.einsum("...s,zs->...z", beta, stock)
        return (np.maximum(value, 0.) if self.premium_cap is None
            else np.clip(value, 0., self.premium_cap))

    def nodal_premiums(self, factors):
        if self.exercise_obstacle:
            return super().nodal_premiums(factors)
        return self.continuation_values(factors)


@dataclass(frozen=True)
class CoefficientMixture:
    """Flattened immutable interpolation of two frozen tensor decision curves.

    Every array is a direct dataclass field, so the existing policy fingerprint
    hashes all coefficient bytes.  These interpolated rules are used only as
    predictable microstep boundary features; they add no exercise dates.
    """
    left_center: np.ndarray
    left_scale: np.ndarray
    left_beta: np.ndarray
    right_center: np.ndarray
    right_scale: np.ndarray
    right_beta: np.ndarray
    knots: tuple
    left_degree: int | None
    right_degree: int | None
    left_clip: float
    right_clip: float
    weight: float

    def __post_init__(self):
        for name in ("left_center","left_scale","left_beta","right_center","right_scale","right_beta"):
            value=np.asarray(getattr(self,name),float).copy()
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    def _model(self, side):
        from rvlab.pricing.rough_stopping import _Continuation
        from rvlab.pricing.rough_tensor_premium import TensorPremium
        values=[getattr(self,side+"_"+name) for name in ("center","scale","beta")]
        degree=getattr(self,side+"_degree")
        if degree is None:
            return _Continuation(*values,self.knots,0,0.)
        return TensorPremium(*values,self.knots,degree,getattr(self,side+"_clip"),0,0.)

    def coefficients(self, factors):
        return ((1-self.weight)*self._model("left").coefficients(factors)
            +self.weight*self._model("right").coefficients(factors))

    def value(self, spots, factors, strike):
        return ((1-self.weight)*self._model("left").value(spots,factors,strike)
            +self.weight*self._model("right").value(spots,factors,strike))


def _microdate_decisions(selected_policy, times):
    original=np.asarray(selected_policy.times)
    models=[]
    for time in times:
        j=int(np.searchsorted(original,time))
        if original[j]==time:
            models.append(selected_policy.continuations[j])
            continue
        left,right=selected_policy.continuations[j-1:j+1]
        if left.knots != right.knots:
            raise ValueError("Decision interpolation requires matching stock knots")
        values={side+"_"+key:getattr(model,key)
            for side,model in (("left",left),("right",right))
            for key in ("center","scale","beta")}
        values.update(knots=left.knots,left_degree=getattr(left,"degree",None),
            right_degree=getattr(right,"degree",None),left_clip=getattr(left,"factor_clip",4.),
            right_clip=getattr(right,"factor_clip",4.),weight=float((time-original[j-1])/(original[j]-original[j-1])))
        models.append(CoefficientMixture(**values))
    return tuple(models)


def refresh_tensor_obstacle_control(proxy, selected_policy, *, stock_knots=None):
    """Materialize the frozen selected rule and rebuild only its value control.

    The selected policy must already be fitted and selected independently.
    Its historical training/selection seeds propagate to the path firewall.
    This helper performs no fit, simulation or adaptive model selection.
    """
    from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy
    from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    from rvlab.pricing.rough_tensor_premium import TensorPremium
    if not isinstance(proxy, FourierStoppingPolicy):
        raise TypeError("A Fourier stopping proxy is required")
    if not isinstance(selected_policy, ControlledExercisePolicy):
        raise TypeError("A frozen controlled exercise policy is required")
    if selected_policy.base_policy is not proxy:
        raise ValueError("The exercise rule must belong to the supplied proxy")
    if not np.array_equal(selected_policy.times, proxy.continuation_times):
        raise ValueError("Exercise-rule and control calendar must match exactly")
    if selected_policy.premium_floor != 0.:
        raise ValueError("The current refreshed control requires zero premium floor")
    knots = proxy.config.stock_knots if stock_knots is None else stock_knots
    # Reuse the established flattened affine arrays.  The original additive
    # continuation stored here is replaced below; no control is relabelled.
    original = build_rough_obstacle_proxy(proxy, stock_knots=knots)
    models = []
    for old, decision in zip(original.value_premiums, selected_policy.continuations):
        values = {field.name: getattr(old, field.name) for field in fields(old)}
        values.update(continuation_center=decision.center,
            continuation_scale=decision.scale, continuation_beta=decision.beta,
            continuation_knots=np.asarray(decision.knots))
        if isinstance(decision, TensorPremium):
            values.update(continuation_degree=decision.degree,
                continuation_factor_clip=decision.factor_clip)
        models.append(TensorObstaclePremium(**values))
    diagnostics = dict(proxy.diagnostics)
    diagnostics.update(control_proxy_method="obstacle envelope of frozen selected tensor rule",
        selected_exercise_policy_preserved=True, original_control_replaced=True,
        additional_control_training_seeds=tuple(selected_policy.training_seeds),
        valuation_used_in_control_fitting=False, american_pricing_ready=False,
        optimal_stock_delta_ready=False)
    return replace(original, continuations=selected_policy.continuations,
        value_premiums=tuple(models), exercise_at_inception=selected_policy.exercise_at_inception,
        diagnostics=diagnostics)


def fit_microdate_value_control(proxy, selected_policy, *, training_paths,
        training_seed, value_stock_knots=None, control_stock_knots=None,
        degree=2, ridge=1e-4):
    """Refit the frozen policy's value at every existing integration date.

    Future cashflows are controlled by the exactly centered European proxy.
    Regression uses a new independent training stream.  Decisions remain the
    old selected rule, with its original exercise calendar.  The fitted control
    is a conditional-value approximation; a new dual fit and independent final
    paths are required before reporting an improvement.
    """
    from numbers import Integral
    from time import perf_counter
    from rvlab.pricing.rough_simulation import _time_grid
    from rvlab.pricing.rough_stopping_fourier import _stream
    from rvlab.pricing.rough_stopping import _payoff, _zero_continuation
    from rvlab.pricing.rough_tensor_premium import tensor_premium_fitter
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    started=perf_counter()
    if (isinstance(training_paths,bool) or not isinstance(training_paths,Integral) or training_paths<2
            or isinstance(training_seed,bool) or not isinstance(training_seed,Integral) or training_seed<0):
        raise ValueError("Independent value training needs at least two paths and a nonnegative integer seed")
    forbidden=tuple(selected_policy.training_seeds)+(proxy.config.valuation_seed,)
    if training_seed in forbidden:
        raise ValueError("Value training seed overlaps an earlier policy or reserved valuation stream")
    # Reuse input validation and the exact selected decision materialization.
    decision=refresh_tensor_obstacle_control(proxy,selected_policy)
    times=_time_grid(proxy.params,proxy.market,proxy.continuation_times,proxy.config.steps)
    fitconfig=replace(proxy.config,training_paths=int(training_paths),training_seed=int(training_seed),
        stock_knots=tuple(value_stock_knots or selected_policy.continuations[0].knots))
    snapshots=_stream(proxy.params,proxy.market,proxy.kernel,proxy.strike,proxy.maturity,
        proxy.is_call,times,fitconfig,proxy.control_config,training=True)
    decisions=_microdate_decisions(selected_policy,times)
    fit=tensor_premium_fitter(degree=degree,ridge=ridge)
    discounts=proxy.market.discount_curve.discount(times)
    exercise=set(map(float,proxy.exercise_times))
    target=discounts[-1]*_payoff(snapshots[float(times[-1])][0],proxy.strike,proxy.is_call)-snapshots[float(times[-1])][2]
    value_models=[None]*len(times)
    value_models[-1]=_zero_continuation(snapshots[float(times[-1])][1],fitconfig)
    for j in range(len(times)-2,0,-1):
        stock,factors,martingale,euro=snapshots[float(times[j])]
        if float(times[j]) in exercise:
            continuation=np.clip(decisions[j].value(stock,factors,proxy.strike),0.,proxy.premium_cap)
            intrinsic=_payoff(stock,proxy.strike,proxy.is_call)
            stop=(intrinsic>0)&(intrinsic>=euro/discounts[j]+continuation)
            target=np.where(stop,discounts[j]*intrinsic-martingale,target)
        labels=(target+martingale-euro)/discounts[j]
        value_models[j]=fit(stock,factors,labels,proxy.strike,fitconfig)
    value_models[0]=_zero_continuation(snapshots[0.][1],fitconfig)
    dummy=tuple(_zero_continuation(snapshots[float(time)][1],fitconfig) for time in times)
    shell=replace(decision,continuation_times=times,continuations=dummy,value_premiums=dummy,
        config=fitconfig)
    envelope=build_rough_obstacle_proxy(shell,stock_knots=control_stock_knots or proxy.config.stock_knots)
    models=[]
    for time,old,value in zip(times,envelope.value_premiums,value_models):
        settings={field.name:getattr(old,field.name) for field in fields(old)}
        settings.update(continuation_center=value.center,continuation_scale=value.scale,
            continuation_beta=value.beta,continuation_knots=np.asarray(value.knots),
            continuation_degree=getattr(value,"degree",None),
            continuation_factor_clip=getattr(value,"factor_clip",4.),exercise_obstacle=float(time) in exercise)
        models.append(TensorObstaclePremium(**settings))
    diagnostics=dict(decision.diagnostics)
    diagnostics.update(control_proxy_method="independent frozen-policy value fit at every integration date",
        value_training_seed=int(training_seed),value_training_paths=int(training_paths),
        value_training_elapsed_seconds=perf_counter()-started,value_regression_degree=degree,
        value_regression_ridge=ridge,value_stock_knots=tuple(fitconfig.stock_knots),
        value_control_dates=len(times),exercise_dates=len(proxy.exercise_times),
        original_exercise_calendar_preserved=True,
        additional_control_training_seeds=tuple(dict.fromkeys(selected_policy.training_seeds+(int(training_seed),))),
        value_training_control="actual-law exactly centered European Fourier increments",
        value_training_final_valuation_used=False,american_pricing_ready=False,optimal_stock_delta_ready=False)
    # Control knots and exercise knots have separate roles.  The config follows
    # the former; each frozen continuation object carries its own latter grid.
    return replace(envelope,continuations=decisions,value_premiums=tuple(models),diagnostics=diagnostics)


__all__ = ["TensorObstaclePremium", "CoefficientMixture", "refresh_tensor_obstacle_control",
    "fit_microdate_value_control"]
