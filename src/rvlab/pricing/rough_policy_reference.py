"""Independent constant-variance reference decisions on a fixed exercise calendar.

This is a diagnostic policy, not a rough-Heston approximation or a replacement
for the conditional expectations used by a martingale control.  A deterministic
Gaussian dynamic program gives continuation margins on a fine stock mesh.
Its actions can then be evaluated on the *original* simulation trace with the
original, frozen martingale, including its original post-stop history gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .rough_policy_diagnostics import ControlledExercisePolicy
from .rough_stopping_reference import (
    _gaussian_stock_linear_step, black_scholes_stopping_surface,
)


@dataclass(frozen=True)
class _ReferenceDecision:
    """Encode a reference action for the existing continuation-policy protocol.

    These signed sentinels are decision comparators, not option-value premiums.
    They prevent the base policy's approximate European proxy from changing the
    independent dynamic program's decisions. Never use them as dual controls.
    """
    stocks: np.ndarray
    continuation_margin: np.ndarray

    def __post_init__(self):
        for name in ("stocks", "continuation_margin"):
            array = np.asarray(getattr(self, name), float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def value(self, stocks, factors, strike):
        margin = np.interp(np.asarray(stocks, float), self.stocks, self.continuation_margin)
        return np.where(margin > 0., 1e100, -1e100)


def build_black_scholes_reference_policy(base_policy, *, volatility=None,
        exercise_times=None, spot_nodes=32769, log_half_width=2.):
    """Build a deterministic DP-informed policy for a constant-variance case.

    The policy must be valued on a trace with exactly ``exercise_times``.  It
    does not train on, inspect, or select against valuation paths.  Spatial and
    domain sensitivity remain the responsibility of the reference study; this
    approximate policy does not make a rigorous claim about numerical bias.
    """
    from .rough_stopping_fourier import FourierStoppingPolicy
    if not isinstance(base_policy, FourierStoppingPolicy):
        raise TypeError("reference policy requires a FourierStoppingPolicy")
    if base_policy.exercise_style != "american":
        raise ValueError("reference policy requires American exercise")
    params = base_policy.params
    if params.vol_of_vol != 0 or params.v0 != getattr(params, "theta", None):
        raise ValueError("reference policy requires constant variance, not a rough/Heston substitution")
    expected = math.sqrt(params.v0)
    if volatility is None:
        volatility = expected
    if (isinstance(volatility, (bool, np.bool_)) or not isinstance(volatility, Real)
            or not np.isfinite(volatility) or volatility != expected):
        raise ValueError("reference volatility must equal the declared constant variance")
    times = base_policy.exercise_times if exercise_times is None else np.asarray(exercise_times)
    if not np.array_equal(times, base_policy.exercise_times):
        raise ValueError("reference policy must share the base policy exercise calendar")
    surface = black_scholes_stopping_surface(base_policy.market,
        volatility=volatility, strike=base_policy.strike, exercise_times=times,
        is_call=base_policy.is_call, spot_nodes=spot_nodes,
        log_half_width=log_half_width, exercise_style="american")
    stocks, values = surface["stocks"], surface["values"]
    discounts = np.asarray(base_policy.market.discount_curve.discount(times), float)
    payoff = np.maximum(stocks-base_policy.strike if base_policy.is_call
        else base_policy.strike-stocks, 0.)
    decisions, boundaries = [], []
    initial_margin = None
    for j in range(len(times)-1):
        duration = times[j+1]-times[j]
        growth = math.log(discounts[j]/discounts[j+1])
        if base_policy.is_call:
            tail = {"left_intercept": 0., "left_slope": 0.,
                "right_intercept": -base_policy.strike*float(np.min(discounts[j+1:]))/discounts[j+1],
                "right_slope": 1.}
        else:
            tail = {"left_intercept": base_policy.strike*float(np.max(discounts[j+1:]))/discounts[j+1],
                "left_slope": -1., "right_intercept": 0., "right_slope": 0.}
        continuation = discounts[j+1]/discounts[j]*_gaussian_stock_linear_step(
            values[j+1], stocks, growth-.5*volatility**2*duration,
            volatility**2*duration, **tail)
        margin = continuation-payoff
        decisions.append(_ReferenceDecision(stocks, margin))
        exercise = (payoff > 0.) & (margin <= 0.)
        boundary = None if not exercise.any() else float(
            stocks[np.flatnonzero(exercise)[0 if base_policy.is_call else -1]])
        boundaries.append({"time": float(times[j]), "exercise_stock_boundary": boundary})
        if j == 0:
            initial_margin = float(np.interp(base_policy.market.spot, stocks, margin))
    decisions.append(_ReferenceDecision(stocks, np.zeros_like(stocks)))
    intrinsic = max(base_policy.market.spot-base_policy.strike if base_policy.is_call
        else base_policy.strike-base_policy.market.spot, 0.)
    initial_stop = intrinsic > 0 and (initial_margin is None or initial_margin <= 0.)
    # Retaining the base seed is conservative: the deterministic reference does
    # not use that training sample, while shared-trace comparisons still do.
    seeds = (int(base_policy.config.training_seed),)
    return ControlledExercisePolicy(base_policy, times, tuple(decisions), initial_stop,
        seeds, None, None,
        {"method": "independent Gaussian-DP reference decisions",
         "training_paths": 0, "valuation_used_in_training": False,
         "reference_price": float(np.interp(base_policy.market.spot, stocks, values[0])),
         "reference_settings": dict(surface["settings"]),
         "reference_boundaries": boundaries,
         "actions_override_approximate_european_proxy": True,
         "decision_comparator_is_not_a_value_proxy": True,
         "rough_model_substitution": False,
         "scope": "constant variance on identical finite exercise dates; approximate stock mesh",
         "american_pricing_ready": False, "optimal_stock_delta_ready": False})


__all__ = ["build_black_scholes_reference_policy"]
