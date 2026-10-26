"""Heston models with a deterministic, nonnegative mean-reversion schedule.

``curve`` specifies theta(t), rather than independently prescribing expected
variance at each expiry.  The resulting forward-variance curve is
``xi(t) = E[V_t]``.  For rough Heston the model is

    V_t = v0 + integral K(t-s) kappa (theta(s)-V_s) ds
             + integral K(t-s) vol_of_vol sqrt(V_s) dB_s,

with K(t)=t**(H-.5)/Gamma(H+.5). Nonnegative v0 and theta provide a
nonnegative immigration input. This restricted parameterisation avoids
silently treating an arbitrary positive curve as an admissible Volterra
initial condition. Times and curve knots are years from valuation. The last
theta level is extrapolated, and no historical conditional state is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Real
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad_vec
from scipy.special import gamma

from ._validation import broadcast_inputs, scalar_or_array
from .black_scholes import black_scholes_price
from .fourier import price_from_cf
from .heston import _heston_riccati_basis, heston_cf, heston_price
from .rough_heston import (
    _terminal_integral, rough_heston_cf, rough_heston_price,
    solve_fractional_riccati,
)
from .types import (
    BlackScholesParams, HestonParams, PricingConfig, PricingError,
    RoughHestonParams, _nonnegative,
)


@dataclass(frozen=True)
class MeanReversionCurve:
    """Piecewise-constant nonnegative theta, with right-continuous knots.

    ``levels[j]`` applies between knot j-1 and knot j, and the final level
    continues indefinitely. This is a drift schedule; use ``forward_variance``
    to obtain the corresponding expected variance curve.
    """

    knot_times: tuple[float, ...]
    levels: tuple[float, ...]

    def __post_init__(self):
        try:
            knots, levels = tuple(self.knot_times), tuple(self.levels)
        except TypeError as exc:
            raise ValueError("curve knot_times and levels must be sequences") from exc
        if len(levels) != len(knots) + 1:
            raise ValueError("curve requires exactly one more level than knot_times")
        previous = 0.0
        for index, knot in enumerate(knots):
            if (isinstance(knot, bool) or not isinstance(knot, Real)
                    or not math.isfinite(knot) or knot <= previous):
                raise ValueError(f"curve knot_times[{index}] must be finite, positive, and increasing")
            previous = knot
        for index, level in enumerate(levels):
            _nonnegative(f"curve levels[{index}]", level)
        object.__setattr__(self, "knot_times", tuple(float(x) for x in knots))
        object.__setattr__(self, "levels", tuple(float(x) for x in levels))

    @property
    def is_constant(self):
        return all(level == self.levels[0] for level in self.levels)

    def __call__(self, maturity):
        times = _times(maturity)
        result = np.asarray(self.levels)[np.searchsorted(self.knot_times, times, side="right")]
        return scalar_or_array(result)


@dataclass(frozen=True)
class HestonCurveParams:
    """Heston with theta(t) provided by a MeanReversionCurve."""

    v0: float
    kappa: float
    vol_of_vol: float
    rho: float
    curve: MeanReversionCurve

    def __post_init__(self):
        for name in ("v0", "kappa", "vol_of_vol"):
            _nonnegative(name, getattr(self, name))
        if (isinstance(self.rho, bool) or not isinstance(self.rho, Real)
                or not math.isfinite(self.rho) or abs(self.rho) > 1):
            raise ValueError("rho must be finite and in [-1, 1]")
        if not isinstance(self.curve, MeanReversionCurve):
            raise TypeError("curve must be MeanReversionCurve")


@dataclass(frozen=True)
class RoughHestonCurveParams(HestonCurveParams):
    """Volterra Heston with nonnegative time-dependent theta(t)."""

    hurst: float

    def __post_init__(self):
        super().__post_init__()
        if (isinstance(self.hurst, bool) or not isinstance(self.hurst, Real)
                or not math.isfinite(self.hurst) or not 0 < self.hurst <= 0.5):
            raise ValueError("hurst must be finite and in (0, 0.5]")


def _times(maturity):
    result = np.asarray(maturity, dtype=float)
    if not np.all(np.isfinite(result) & (result >= 0)):
        raise ValueError("maturities must be finite and nonnegative")
    return result


def _check_params(params):
    if not isinstance(params, HestonCurveParams):
        raise TypeError("params must be HestonCurveParams or RoughHestonCurveParams")


def _legacy_params(params):
    kwargs = dict(v0=params.v0, theta=params.curve.levels[0], kappa=params.kappa,
                  vol_of_vol=params.vol_of_vol, rho=params.rho)
    if isinstance(params, RoughHestonCurveParams):
        return RoughHestonParams(**kwargs, hurst=params.hurst)
    return HestonParams(**kwargs)


@lru_cache(maxsize=8192)
def _relaxation_moments(alpha, x):
    """E_alpha(-x), E_alpha,2(-x), and 1-E_alpha,2(-x).

    Small x uses convergent series to retain small drift contributions. Other
    values use the positive spectral mixture of the completely monotone
    Mittag-Leffler relaxation. Its inverse-CDF substitution gives a bounded
    integral on [0,1], including the near-alpha=1 concentration, without an
    oscillating or exponentially cancelling large-x series.
    """
    if x == 0.0:
        return 1.0, 1.0, 0.0
    if alpha == 1.0:
        relaxation = math.exp(-x)
        integral = -math.expm1(-x) / x
        if x < 1e-4:
            complement = x * (0.5 + x * (-1/6 + x * (1/24 - x/120)))
        else:
            complement = 1.0 - integral
        return relaxation, integral, complement
    if x <= 0.75:
        r_terms, j_terms = [1.0], [1.0]
        power = 1.0
        for index in range(1, 250):
            power *= -x
            r_term = power / gamma(alpha * index + 1.0)
            j_term = power / gamma(alpha * index + 2.0)
            r_terms.append(r_term)
            j_terms.append(j_term)
            if max(abs(r_term), abs(j_term)) < 2e-17:
                break
        else:
            raise PricingError("Mittag-Leffler relaxation series did not converge")
        return math.fsum(r_terms), math.fsum(j_terms), -math.fsum(j_terms[1:])
    angle = math.pi * alpha
    log_x = math.log(x)

    def integrand(position):
        if position <= 0.0:
            return np.array([1.0, 1.0, 0.0])
        if position >= 1.0:
            return np.array([0.0, 0.0, 1.0])
        log_z = (math.log(math.sin(angle * position))
                 - math.log(math.sin(angle * (1.0 - position))))
        log_a = (log_x + log_z) / alpha
        if log_a > 7.0:
            j_value = math.exp(-log_a)
            return np.array([0.0, j_value, 1.0 - j_value])
        a_value = math.exp(log_a)
        if a_value < 1e-4:
            complement = a_value * (0.5 + a_value * (-1/6 + a_value * (1/24 - a_value/120)))
            j_value = 1.0 - complement
        else:
            j_value = -math.expm1(-a_value) / a_value
            complement = 1.0 - j_value
        return np.array([math.exp(-a_value), j_value, complement])

    # Locate where the exponential argument crosses one. Explicit split
    # points retain a narrow small-u tail at large x and near-alpha=1 limits.
    transition = math.atan2(math.sin(angle), x + math.cos(angle)) / angle
    points = sorted({p for p in (transition / 4, transition, 4 * transition,
                                1 - (1-transition)/4)
                     if 0 < p < 1})
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            values, error = quad_vec(integrand, 0.0, 1.0, points=points,
                                     epsabs=2e-13, epsrel=2e-11, limit=300)
        except (IntegrationWarning, OverflowError, ValueError) as exc:
            raise PricingError("Mittag-Leffler relaxation quadrature failed") from exc
    if (not np.all(np.isfinite(values)) or not np.isfinite(error)
            or np.any(values < -2e-12) or np.any(values > 1 + 2e-12)
            or error > 1e-9):
        raise PricingError("Mittag-Leffler relaxation quadrature did not meet tolerance")
    return tuple(float(value) for value in np.clip(values, 0, 1))


def _mean_components(params, maturity):
    """Nonnegative relaxation and immigration weights, avoiding jump cancellation."""
    alpha = params.hurst + 0.5 if isinstance(params, RoughHestonCurveParams) else 1.0
    if maturity == 0:
        return params.v0, 0.0
    if params.kappa == 0:
        return params.v0, params.v0 * maturity

    def moments(time):
        try:
            argument = params.kappa * time**alpha
        except OverflowError as exc:
            raise PricingError("Forward variance relaxation argument exceeds floating-point range") from exc
        if not math.isfinite(argument):
            raise PricingError("Forward variance relaxation argument is nonfinite")
        return _relaxation_moments(alpha, argument)

    r, j, _ = moments(maturity)
    mean, integral = params.v0 * r, params.v0 * maturity * j
    starts = (0.0,) + params.curve.knot_times
    ends = params.curve.knot_times + (maturity,)
    for start, end, level in zip(starts, ends, params.curve.levels):
        if start >= maturity:
            break
        end = min(end, maturity)
        left, right = maturity - start, maturity - end
        rl, _, ql = moments(left)
        rr, _, qr = moments(right)
        mean_weight, integral_weight = rr-rl, left*ql-right*qr
        if (mean_weight < -2e-11
                or integral_weight < -2e-11 * max(maturity, 1.0)):
            raise PricingError("Forward variance relaxation produced negative immigration weights")
        mean += level * max(mean_weight, 0.0)
        integral += level * max(integral_weight, 0.0)
    if not np.all(np.isfinite([mean, integral])):
        raise PricingError("Forward variance curve produced nonfinite moments")
    return mean, integral


def forward_variance(params, maturity):
    """Expected instantaneous variance E[V_t], including its value v0 at zero."""
    _check_params(params)
    times = _times(maturity)
    values = np.array([_mean_components(params, float(t))[0] for t in times.flat])
    return scalar_or_array(values.reshape(times.shape))


def integrated_forward_variance(params, maturity):
    """Expected integrated variance integral_0^T E[V_t] dt."""
    _check_params(params)
    times = _times(maturity)
    values = np.array([_mean_components(params, float(t))[1] for t in times.flat])
    return scalar_or_array(values.reshape(times.shape))


def _cf_inputs(u, maturity):
    frequencies, times = np.broadcast_arrays(np.asarray(u, complex), _times(maturity))
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("CF frequencies must be finite")
    return frequencies, times


def heston_curve_cf(params, u, maturity):
    """Normalized European log-forward-return CF for the scheduled Heston model."""
    if not isinstance(params, HestonCurveParams) or isinstance(params, RoughHestonCurveParams):
        raise TypeError("params must be HestonCurveParams")
    if params.curve.is_constant and params.vol_of_vol != 0:
        return heston_cf(_legacy_params(params), u, maturity)
    u, t = _cf_inputs(u, maturity)
    if params.vol_of_vol == 0:
        output = np.exp(-0.5 * (u*u + 1j*u) * integrated_forward_variance(params, t))
    else:
        initial, _ = _heston_riccati_basis(params, u, t)
        exponent = params.v0 * initial
        if params.kappa:
            for index, level in enumerate(params.curve.levels):
                start = 0.0 if index == 0 else params.curve.knot_times[index-1]
                end = params.curve.knot_times[index] if index < len(params.curve.knot_times) else np.inf
                _, left = _heston_riccati_basis(params, u, np.maximum(t-start, 0))
                _, right = _heston_riccati_basis(params, u, np.maximum(t-end, 0))
                exponent += params.kappa * level * (left-right)
        with np.errstate(over="ignore", invalid="ignore"):
            output = np.exp(exponent)
    if not np.all(np.isfinite(output)):
        raise PricingError("Heston curve characteristic function produced nonfinite values")
    return complex(output) if np.ndim(output) == 0 else output


def _linear_primitive(values, maturity, points):
    """Integral of linear interpolants, with exact partial cells at curve knots."""
    n = len(values)-1
    dt = maturity/n
    cumulative = np.vstack([np.zeros(values.shape[1], complex),
                            np.cumsum(0.5*dt*(values[:-1]+values[1:]), axis=0)])
    output = []
    for point in points:
        location = float(np.clip(point/dt, 0, n))
        index = min(int(location), n-1)
        fraction = location-index
        output.append(cumulative[index] + dt * (
            fraction*values[index] + 0.5*fraction*fraction*(values[index+1]-values[index])))
    return np.asarray(output)


@lru_cache(maxsize=32)
def _rough_basis(kappa, vol_of_vol, rho, hurst, maturity, steps, knots, frequencies):
    """Cache only affine terminal bases, never a full O(steps*nodes) path."""
    parameters = RoughHestonParams(0.0, 0.0, kappa, vol_of_vol, rho, hurst)
    _, values = solve_fractional_riccati(parameters, np.asarray(frequencies), maturity,
                                        n_steps=steps)
    initial = _terminal_integral(values, maturity, 0.5-hurst)
    boundaries = np.maximum(maturity-np.asarray((0.0,)+knots), 0)
    points = np.r_[boundaries, 0.0]
    primitive = _linear_primitive(values, maturity, points)
    levels = kappa * (primitive[:-1]-primitive[1:])
    initial.setflags(write=False)
    levels.setflags(write=False)
    return initial, levels


def rough_heston_curve_cf(params, u, maturity, *, config=None):
    """Normalized CF, with the original fractional Riccati equation and kernel."""
    if not isinstance(params, RoughHestonCurveParams):
        raise TypeError("params must be RoughHestonCurveParams")
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    if params.hurst == 0.5:
        ordinary = HestonCurveParams(params.v0, params.kappa, params.vol_of_vol,
                                     params.rho, params.curve)
        return heston_curve_cf(ordinary, u, maturity)
    if params.curve.is_constant and params.vol_of_vol != 0:
        return rough_heston_cf(_legacy_params(params), u, maturity, config=config)
    config = PricingConfig() if config is None else config
    u, t = _cf_inputs(u, maturity)
    if params.vol_of_vol == 0:
        with np.errstate(over="ignore", invalid="ignore"):
            output = np.exp(-0.5 * (u*u + 1j*u) * integrated_forward_variance(params, t))
        if not np.all(np.isfinite(output)):
            raise PricingError("Deterministic rough Heston curve characteristic function is nonfinite")
        return complex(output) if np.ndim(output) == 0 else output
    shape = u.shape
    frequencies, times = u.reshape(-1), t.reshape(-1)
    output = np.ones(frequencies.shape, complex)
    active = (times > 0) & (frequencies != 0) & (frequencies != -1j)
    for tenor in np.unique(times[active]):
        selected = active & (times == tenor)
        initial, levels = _rough_basis(
            params.kappa, params.vol_of_vol, params.rho, params.hurst,
            float(tenor), config.riccati_steps, params.curve.knot_times,
            tuple(frequencies[selected]),
        )
        exponent = params.v0*initial + np.asarray(params.curve.levels) @ levels
        with np.errstate(over="ignore", invalid="ignore"):
            output[selected] = np.exp(exponent)
    if not np.all(np.isfinite(output)):
        raise PricingError("Rough Heston curve characteristic function produced nonfinite values")
    output = output.reshape(shape)
    return complex(output) if output.ndim == 0 else output


def _deterministic_price(params, *, forward, discount, strike, maturity, is_call):
    f, d, k, t, calls = broadcast_inputs(forward=forward, discount=discount,
                                       strike=strike, maturity=maturity, is_call=is_call)
    integrated = np.asarray(integrated_forward_variance(params, t))
    variance = np.divide(integrated, t, out=np.full_like(t, params.v0), where=t > 0)
    output = np.empty(t.shape)
    for level in np.unique(variance):
        selected = variance == level
        output[selected] = black_scholes_price(
            BlackScholesParams(float(np.sqrt(level))), forward=f[selected],
            discount=d[selected], strike=k[selected], maturity=t[selected],
            is_call=calls[selected])
    return scalar_or_array(output)


def heston_curve_price(params, *, forward, discount, strike, maturity,
                       is_call=True, config=None):
    """European Heston prices per underlying unit, with explicit forward/discount."""
    if not isinstance(params, HestonCurveParams) or isinstance(params, RoughHestonCurveParams):
        raise TypeError("params must be HestonCurveParams")
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    kwargs = dict(forward=forward, discount=discount, strike=strike,
                  maturity=maturity, is_call=is_call)
    if params.curve.is_constant and params.vol_of_vol != 0:
        return heston_price(_legacy_params(params), **kwargs, config=config)
    if params.vol_of_vol == 0:
        return _deterministic_price(params, **kwargs)
    return price_from_cf(lambda u, t: heston_curve_cf(params, u, t),
                         reference_variance=params.v0, **kwargs, config=config)


def rough_heston_curve_price(params, *, forward, discount, strike, maturity,
                             is_call=True, config=None):
    """European Volterra Heston prices with fractional and Fourier controls."""
    if not isinstance(params, RoughHestonCurveParams):
        raise TypeError("params must be RoughHestonCurveParams")
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    kwargs = dict(forward=forward, discount=discount, strike=strike,
                  maturity=maturity, is_call=is_call)
    if params.hurst == 0.5:
        ordinary = HestonCurveParams(params.v0, params.kappa, params.vol_of_vol,
                                     params.rho, params.curve)
        return heston_curve_price(ordinary, **kwargs, config=config)
    if params.curve.is_constant and params.vol_of_vol != 0:
        return rough_heston_price(_legacy_params(params), **kwargs, config=config)
    if params.vol_of_vol == 0:
        return _deterministic_price(params, **kwargs)
    config = PricingConfig() if config is None else config
    return price_from_cf(lambda u, t: rough_heston_curve_cf(params, u, t, config=config),
                         reference_variance=params.v0, **kwargs, config=config)


__all__ = ["MeanReversionCurve", "HestonCurveParams", "RoughHestonCurveParams",
           "forward_variance", "integrated_forward_variance", "heston_curve_cf",
           "rough_heston_curve_cf", "heston_curve_price", "rough_heston_curve_price"]
