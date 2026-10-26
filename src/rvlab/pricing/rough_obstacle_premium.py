"""Pointwise obstacle envelopes as exact-centered rough stopping controls.

The European affine function and frozen continuation premium are evaluated
at stock nodes using the complete factor state. Taking their obstacle maximum
before stock interpolation avoids a second regression of the nonsmooth Snell
premium. The resulting piecewise-linear function is the actual control; its
conditional Gaussian expectation is evaluated exactly by the existing engine.
It changes neither the rough transition law nor the frozen stopping rule.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .rough_estimators import _AffineSurrogate, _conditional_payoffs
from .rough_simulation import _nonnegative_clock, RoughSimulationError
from .rough_stopping_fourier import FourierStoppingPolicy


DEFAULT_OBSTACLE_STOCK_KNOTS = (.7, .85, .9, .925, .95, .975, 1., 1.025, 1.05, 1.15, 1.3)


def _readonly(value):
    array = np.asarray(value).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RoughObstaclePremium:
    """Flattened immutable Fourier and regression data for one calendar date."""
    stock_nodes: np.ndarray
    frequencies: np.ndarray
    quadrature: np.ndarray
    affine_exponent: np.ndarray
    affine_factor: np.ndarray
    remaining_integral_coefficient: np.ndarray
    remaining_integral_constant: float
    growth: float
    relative_discount: float
    strike: float
    is_call: bool
    remaining_time: float
    premium_cap: float | None
    continuation_center: np.ndarray
    continuation_scale: np.ndarray
    continuation_beta: np.ndarray
    continuation_knots: np.ndarray

    def __post_init__(self):
        for name in ("stock_nodes", "frequencies", "quadrature", "affine_exponent",
                "affine_factor", "remaining_integral_coefficient", "continuation_center",
                "continuation_scale", "continuation_beta", "continuation_knots"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def european_values(self, factors):
        """Date-value affine European proxy at every normalized stock node."""
        factors = np.asarray(factors, float)
        shape = factors.shape[:-1]
        if (factors.ndim < 1 or factors.shape[-1] != len(self.continuation_center)
                or not np.isfinite(factors).all()):
            raise ValueError("full factor coordinates must align with obstacle proxy")
        factors = factors.reshape(-1, factors.shape[-1])
        stocks = self.strike*self.stock_nodes
        if self.remaining_time == 0:
            value = np.maximum((1 if self.is_call else -1)*(stocks-self.strike), 0.)
            return np.broadcast_to(value, shape+(len(stocks),)).copy()
        clock = _nonnegative_clock(factors@self.remaining_integral_coefficient+
            self.remaining_integral_constant)
        square = self.frequencies**2+.25
        with np.errstate(over="ignore", invalid="ignore"):
            cf = np.exp(self.affine_exponent+factors@self.affine_factor.T)
            reference = np.exp(-.5*clock[:, None]*square)
            integrand = (cf-reference)*(self.quadrature/square)
            forward = stocks[1:]*np.exp(self.growth)
            phase = np.exp(1j*np.log(forward/self.strike)[:, None]*self.frequencies)
            correction = np.real(integrand@phase.T)*np.sqrt(forward*self.strike)/np.pi
            mean = np.log(stocks[1:])[None, :]+self.growth-.5*clock[:, None]
            baseline = _conditional_payoffs(mean, clock[:, None], self.strike, self.is_call)
            positive = self.relative_discount*(baseline-correction)
        zero = np.full((len(factors), 1), 0. if self.is_call else self.strike*self.relative_discount)
        value = np.concatenate((zero, positive), axis=-1)
        if not np.isfinite(value).all():
            raise RoughSimulationError("obstacle European proxy produced nonfinite values")
        return value.reshape(shape+(len(stocks),))

    def continuation_values(self, factors):
        factors = np.asarray(factors, float)
        r = (factors-self.continuation_center)/self.continuation_scale
        n, h = r.shape[-1], len(self.continuation_knots)
        beta = self.continuation_beta
        a = beta[0]+r@beta[3+h:3+h+n]
        b = beta[1]+r@beta[3+h+n:3+h+2*n]
        i, j = np.triu_indices(n)
        a = a+(r[..., i]*r[..., j])@beta[3+h+2*n:]
        z = self.stock_nodes
        value = a[..., None]+z*(b[..., None]+z*beta[2])
        value += np.maximum(z[:, None]-self.continuation_knots, 0.)**2@beta[3:3+h]
        return np.maximum(value, 0.) if self.premium_cap is None else np.clip(value, 0., self.premium_cap)

    def nodal_premiums(self, factors):
        european = self.european_values(factors)
        if self.remaining_time == 0:
            return np.zeros_like(european)
        payoff = np.maximum((1 if self.is_call else -1)*
            (self.strike*self.stock_nodes-self.strike), 0.)
        premium = np.maximum(self.continuation_values(factors), payoff-european)
        premium = np.maximum(premium, 0.)
        return premium if self.premium_cap is None else np.minimum(premium, self.premium_cap)

    def coefficients(self, factors):
        premium = self.nodal_premiums(factors)
        slope = np.diff(premium, axis=-1)/np.diff(self.stock_nodes)
        intercept = premium[..., :-1]-slope*self.stock_nodes[:-1]
        slope = np.concatenate((slope, np.zeros(premium.shape[:-1]+(1,))), axis=-1)
        intercept = np.concatenate((intercept, premium[..., -1:]), axis=-1)
        return np.stack((intercept, slope, np.zeros_like(slope)), axis=-1)


def build_rough_obstacle_proxy(policy, *, stock_knots=DEFAULT_OBSTACLE_STOCK_KNOTS):
    """Replace value-control proxies while preserving all exercise decisions."""
    if not isinstance(policy, FourierStoppingPolicy):
        raise TypeError("rough obstacle proxy requires FourierStoppingPolicy")
    if any(e.amount > 0 and e.time <= policy.maturity for e in policy.market.dividends.events):
        raise ValueError("rough obstacle proxy requires an explicitly empty cash schedule")
    if not all(hasattr(model, "beta") and hasattr(model, "center") for model in policy.continuations):
        raise ValueError("obstacle proxy currently requires the original polynomial continuation basis")
    knots = np.asarray(stock_knots)
    if (knots.dtype.kind not in "iuf" or knots.ndim != 1 or not len(knots)
            or not np.isfinite(knots).all() or np.any(knots <= 0) or np.any(np.diff(knots) <= 0)):
        raise ValueError("obstacle stock knots must be finite positive and increasing")
    times = policy.continuation_times
    affine = _AffineSurrogate(policy.params, policy.market, policy.kernel, np.array([policy.strike]),
        np.array([policy.maturity]), np.array([policy.is_call]), times, policy.control_config)
    models = []
    maturity_discount = float(policy.market.discount_curve.discount(policy.maturity))
    for time, continuation in zip(times, policy.continuations):
        a, b = affine.coefficients[(policy.maturity, float(time))]
        coefficient, constant = affine._remaining_integral(float(time), policy.maturity)
        discount = maturity_discount/float(policy.market.discount_curve.discount(time))
        models.append(RoughObstaclePremium(np.r_[0., knots.astype(float)], affine.frequencies,
            affine.quadrature, a, b, coefficient, float(constant), float(-np.log(discount)),
            discount, policy.strike, policy.is_call, policy.maturity-float(time), policy.premium_cap,
            continuation.center, continuation.scale, continuation.beta, np.asarray(continuation.knots)))
    diagnostics = dict(policy.diagnostics)
    diagnostics.update(control_proxy_method="full-factor nodal obstacle envelope before stock interpolation",
        exercise_policy_unchanged=True, proxy_stock_knots=knots.tolist(),
        proxy_stock_interpolation="piecewise linear in S/K; constant upper-stock tail",
        proxy_obstacle_enforced="at calendar-date stock nodes only; interpolation is the actual exact-centered proxy",
        valuation_used_in_control_fitting=False, american_pricing_ready=False, optimal_stock_delta_ready=False)
    config = replace(policy.config, stock_knots=tuple(float(k) for k in knots))
    return replace(policy, config=config, value_premiums=tuple(models), diagnostics=diagnostics)


__all__ = ["RoughObstaclePremium", "build_rough_obstacle_proxy", "DEFAULT_OBSTACLE_STOCK_KNOTS"]
