"""Positive exponential lifts of the rough-Heston kernel.

This module implements the *dynamics* foundation for a future exercise engine.
It does not implement American exercise or infer a conditional memory state.
The lift starts at model inception with U_i(0)=0 and

    dU_i = -x_i U_i dt + kappa (theta(t)-V) dt + eta sqrt(V) dW,
    V = v0 + sum_i w_i U_i.

The finite-lift Fourier transform is independent of the simulator. Its
exponential product-integration Riccati solver has an explicit time budget;
factor count and time resolution must both be refined. The positive kernel
fit is a local, deterministic numerical construction, not a reproduction of
the optimal quadratures in Bayer--Breneis (arXiv:2309.07023).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gamma

from ._validation import broadcast_inputs, scalar_or_array
from .black_scholes import black_scholes_price
from .fourier import price_from_cf
from .heston import heston_cf
from .rough_heston import _implicit_riccati_step
from .types import BlackScholesParams, HestonParams, PricingConfig, PricingError
from .variance_curve import (
    HestonCurveParams, MeanReversionCurve, heston_curve_cf,
)


def _real(name, value, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _times(value, name="times"):
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.all(np.isfinite(result) & (result >= 0)):
        raise ValueError(f"{name} must be finite nonnegative real numbers")
    return result.astype(float)


@dataclass(frozen=True)
class ExponentialKernel:
    """Immutable positive mixture K_N(t)=sum weights[i]*exp(-nodes[i]*t).

    Nodes are strictly increasing and nonnegative. A zero node is permitted
    for the exact Heston kernel. ``hurst`` and ``horizon`` are optional audit
    metadata; a declared horizon is enforced by pricing and moment routines.
    K_N(0) is finite, unlike a fractional kernel for H<.5. Use integrated cell
    masses to assess its error in the singular first time cell.
    """

    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    hurst: float | None = None
    horizon: float | None = None

    def __post_init__(self):
        try:
            nodes, weights = tuple(self.nodes), tuple(self.weights)
        except TypeError as exc:
            raise ValueError("kernel nodes and weights must be nonempty sequences") from exc
        if not nodes or len(nodes) != len(weights):
            raise ValueError("kernel requires equally sized nonempty nodes and weights")
        nodes = tuple(_real(f"nodes[{i}]", x) for i, x in enumerate(nodes))
        weights = tuple(_real(f"weights[{i}]", w, positive=True) for i, w in enumerate(weights))
        if any(b <= a for a, b in zip(nodes, nodes[1:])):
            raise ValueError("kernel nodes must be strictly increasing")
        if self.hurst is not None:
            hurst = _real("hurst", self.hurst, positive=True)
            if hurst > .5:
                raise ValueError("hurst must be in (0, .5]")
            object.__setattr__(self, "hurst", hurst)
        if self.horizon is not None:
            object.__setattr__(self, "horizon", _real("horizon", self.horizon, positive=True))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def fingerprint(self):
        """Stable SHA-256 of coefficients and declared approximation domain."""
        payload = dict(nodes=self.nodes, weights=self.weights,
                       hurst=self.hurst, horizon=self.horizon)
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def __call__(self, times):
        t = _times(times)
        return scalar_or_array(np.exp(-t[..., None] * self.nodes) @ self.weights)

    def integrated(self, times):
        """Exact integral from zero, including constant-forcing first cells."""
        t = _times(times)
        nodes = np.asarray(self.nodes)
        terms = np.empty(t.shape + nodes.shape)
        positive = nodes > 0
        terms[..., positive] = -np.expm1(-t[..., None]*nodes[positive])/nodes[positive]
        terms[..., ~positive] = t[..., None]
        return scalar_or_array(terms @ self.weights)


def approximate_fractional_kernel(
    hurst, horizon, factors, *, min_time_ratio=1e-7, fit_points=256, max_nfev=400,
):
    """Fit a positive exponential kernel on [0,horizon].

    Exact fractional *cell masses* on a logarithmic grid, including [0,t_min],
    are fitted with residuals (mass_N-mass)/sqrt(mass). This uses no evaluation
    of singular K(0). The objective approximates an integrated weighted kernel
    error; neither optimizer convergence nor factor count certifies prices.
    The dimensionless fit is deterministic and cached. All returned nodes and
    weights are positive for H<.5. H=.5 returns the exact single constant
    factor, irrespective of the requested approximation budget.
    """
    h = _real("hurst", hurst, positive=True)
    if h > .5:
        raise ValueError("hurst must be in (0, .5]")
    horizon = _real("horizon", horizon, positive=True)
    for name, value, lower in (("factors", factors, 1), ("fit_points", fit_points, 32),
                               ("max_nfev", max_nfev, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    ratio = _real("min_time_ratio", min_time_ratio, positive=True)
    if ratio >= 1:
        raise ValueError("min_time_ratio must be less than one")
    if fit_points < 4*factors:
        raise ValueError("fit_points must be at least four times factors")
    return _approximate(h, horizon, int(factors), ratio, int(fit_points), int(max_nfev))


@lru_cache(maxsize=64)
def _approximate(h, horizon, factors, ratio, points, max_nfev):
    if h == .5:
        return ExponentialKernel((0.,), (1.,), h, horizon)
    alpha = h+.5
    times = np.r_[0., np.geomspace(ratio, 1., points)]
    target = np.diff(times**alpha / gamma(alpha+1))
    measure_constant = 1/(gamma(alpha)*gamma(1-alpha))
    nodes = np.geomspace(.1, 1e4, factors)
    if factors == 1:
        nodes = np.array([1.])
    weights = measure_constant*nodes**(1-alpha)*(math.log(1e5)/max(factors-1, 1))

    def residuals(coordinates):
        x, w = np.exp(coordinates[:factors]), np.exp(coordinates[factors:])
        primitive = -np.expm1(-times[:, None]*x)/x
        masses = np.diff(primitive, axis=0) @ w
        return (masses-target)/np.sqrt(target)

    fit = least_squares(residuals, np.log(np.r_[nodes, weights]),
                        bounds=(np.full(2*factors, -18.),
                                np.r_[np.full(factors, 25.), np.full(factors, 20.)]),
                        max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise PricingError(f"Positive kernel fit failed: {fit.message}; nfev={fit.nfev}")
    nodes = np.exp(fit.x[:factors])/horizon
    weights = np.exp(fit.x[factors:])*horizon**(alpha-1)
    ordering = np.argsort(nodes)
    return ExponentialKernel(tuple(nodes[ordering]), tuple(weights[ordering]), h, horizon)


def _check(params, kernel, maturity):
    if not isinstance(params, (HestonParams, HestonCurveParams)):
        raise TypeError("params must be Heston/RoughHeston parameters or curve parameters")
    if not isinstance(kernel, ExponentialKernel):
        raise TypeError("kernel must be ExponentialKernel")
    if (kernel.hurst is not None and hasattr(params, "hurst")
            and kernel.hurst != params.hurst):
        raise ValueError("kernel hurst metadata differs from model hurst")
    times = _times(maturity, "maturities")
    if kernel.horizon is not None and np.any(times > kernel.horizon*(1+1e-13)):
        raise ValueError("maturity exceeds the declared kernel approximation horizon")
    return times


def _curve(params):
    return params.curve if isinstance(params, HestonCurveParams) else MeanReversionCurve((), (params.theta,))


def _mean_moments(params, maturity, kernel):
    """Exact affine ODE propagation, with an integrated-variance accumulator."""
    n = len(kernel.nodes)
    weights = np.asarray(kernel.weights)
    matrix = np.zeros((n+2, n+2))
    matrix[:n, :n] = -np.diag(kernel.nodes)-params.kappa*np.ones((n, 1))*weights
    matrix[n+1, :n] = weights
    matrix[n+1, n] = params.v0
    state = np.r_[np.zeros(n), 1., 0.]
    curve = _curve(params)
    edges = (0.,)+tuple(k for k in curve.knot_times if k < maturity)+(float(maturity),)
    for start, end in zip(edges, edges[1:]):
        matrix[:n, n] = params.kappa*(curve(.5*(start+end))-params.v0)
        state = expm((end-start)*matrix) @ state
    variance = params.v0+weights @ state[:n]
    integral = state[n+1]
    if (not np.all(np.isfinite([variance, integral]))
            or variance < -1e-10 or integral < -1e-10):
        raise PricingError("Lifted deterministic moments are nonfinite or materially negative")
    return max(float(variance), 0.), max(float(integral), 0.)


def lifted_forward_variance(params, maturity, kernel):
    """Exact finite-lift E[V_t], using calendar-time mean-reversion knots."""
    times = _check(params, kernel, maturity)
    values = np.array([_mean_moments(params, float(t), kernel)[0] for t in times.flat])
    return scalar_or_array(values.reshape(times.shape))


def lifted_integrated_variance(params, maturity, kernel):
    """Exact finite-lift E[integral_0^T V_t dt], not average-vol substitution."""
    times = _check(params, kernel, maturity)
    values = np.array([_mean_moments(params, float(t), kernel)[1] for t in times.flat])
    return scalar_or_array(values.reshape(times.shape))


def _one_factor_params(params, kernel):
    x, w = kernel.nodes[0], kernel.weights[0]
    kappa = x+w*params.kappa
    curve = _curve(params)
    levels = tuple((x*params.v0+w*params.kappa*level)/kappa if kappa else level
                   for level in curve.levels)
    return HestonCurveParams(params.v0, kappa, w*params.vol_of_vol, params.rho,
                             MeanReversionCurve(curve.knot_times, levels))


def _solve_lifted_riccati(params, frequencies, maturity, kernel, steps):
    """Second-order exponential product integration, all frequencies batched.

    Exponential decay is integrated exactly and F(psi) interpolated linearly.
    The nonlinear endpoint equation is solved on its continuous quadratic
    branch. Curve discontinuities are aligned exactly in reverse Riccati time.
    The returned exponent includes v0*integral F and calendar theta forcing.
    """
    z = np.asarray(frequencies, complex).reshape(-1)
    if maturity == 0:
        return np.zeros_like(z)
    base = np.linspace(0., maturity, steps+1)
    curve = _curve(params)
    reverse_knots = [maturity-k for k in curve.knot_times if 0 < k < maturity]
    times = np.unique(np.r_[base, reverse_knots])
    x, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    values = np.zeros((len(x), len(z)), complex)
    psi = np.zeros(len(z), complex)
    forcing = -.5*(z*z+1j*z)
    linear = 1j*z*params.rho*params.vol_of_vol-params.kappa
    quadratic = .5*params.vol_of_vol**2
    previous_f = forcing.copy()
    exponent = np.zeros(len(z), complex)
    for start, end in zip(times, times[1:]):
        dt = end-start
        decay = np.exp(-x*dt)
        phi = np.ones_like(x)
        nonzero = x > 0
        phi[nonzero] = -np.expm1(-x[nonzero]*dt)/(x[nonzero]*dt)
        endpoint = np.full_like(x, .5*dt)
        small = (x*dt) < 1e-3
        q = x[small]*dt
        endpoint[small] = dt*(.5+q*(-1/6+q*(1/24-q/120)))
        endpoint[~small] = (1-phi[~small])/x[~small]
        initial = dt*phi-endpoint
        history = (weights*decay) @ values + (weights @ initial)*previous_f
        scale = float(weights @ endpoint)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            next_psi = _implicit_riccati_step(history, forcing, linear, quadratic, scale)
            next_f = forcing+linear*next_psi+quadratic*next_psi*next_psi
            values = (decay[:, None]*values+initial[:, None]*previous_f
                      +endpoint[:, None]*next_f)
            theta = curve(maturity-.5*(start+end))
            exponent += .5*dt*(params.v0*(previous_f+next_f)
                                      +params.kappa*theta*(psi+next_psi))
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(exponent)):
            raise PricingError("Lifted Riccati solver diverged; refine steps or revise parameters")
        psi, previous_f = next_psi, next_f
    return exponent


def lifted_heston_cf(params, u, maturity, kernel, *, config=None):
    """Finite-lift normalized European log-forward CF at model inception.

    Single-exponential lifts are exactly Heston (with mapped coefficients),
    and use that analytic control. Multiple factors use an independent
    exponential-Riccati discretization; riccati_steps remains an accuracy
    budget, rather than a guarantee. No cash-dividend jump is represented.
    """
    times = _check(params, kernel, maturity)
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    config = PricingConfig() if config is None else config
    frequencies, times = np.broadcast_arrays(np.asarray(u, complex), times)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("CF frequencies must be finite")
    if params.vol_of_vol == 0:
        output = np.exp(-.5*(frequencies**2+1j*frequencies)
                        *lifted_integrated_variance(params, times, kernel))
    elif len(kernel.nodes) == 1:
        output = heston_curve_cf(_one_factor_params(params, kernel), frequencies, times)
    else:
        shape = frequencies.shape
        z, t = frequencies.reshape(-1), times.reshape(-1)
        output = np.ones(z.shape, complex)
        active = (t > 0) & (z != 0) & (z != -1j)
        for tenor in np.unique(t[active]):
            selected = active & (t == tenor)
            exponent = _solve_lifted_riccati(params, z[selected], float(tenor), kernel,
                                            config.riccati_steps)
            with np.errstate(over="ignore", invalid="ignore"):
                output[selected] = np.exp(exponent)
        output = output.reshape(shape)
    if not np.all(np.isfinite(output)):
        raise PricingError("Lifted Heston CF produced nonfinite values")
    return complex(output) if np.ndim(output) == 0 else output


def lifted_heston_price(params, *, forward, discount, strike, maturity,
                        kernel, is_call=True, config=None):
    """European prices for this finite lift, with explicit forwards/discounts."""
    _check(params, kernel, maturity)
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    if params.vol_of_vol == 0:
        f, d, k, t, calls = broadcast_inputs(forward=forward, discount=discount,
                                            strike=strike, maturity=maturity, is_call=is_call)
        integrated = np.asarray(lifted_integrated_variance(params, t, kernel))
        variance = np.divide(integrated, t, out=np.full_like(t, params.v0), where=t > 0)
        output = np.empty(t.shape)
        for value in np.unique(variance):
            selected = variance == value
            output[selected] = black_scholes_price(
                BlackScholesParams(math.sqrt(value)), forward=f[selected], discount=d[selected],
                strike=k[selected], maturity=t[selected], is_call=calls[selected])
        return scalar_or_array(output)
    return price_from_cf(lambda u, t: lifted_heston_cf(params, u, t, kernel, config=config),
                         reference_variance=params.v0, forward=forward, discount=discount,
                         strike=strike, maturity=maturity, is_call=is_call, config=config)


__all__ = ["ExponentialKernel", "approximate_fractional_kernel", "lifted_forward_variance",
           "lifted_integrated_variance", "lifted_heston_cf", "lifted_heston_price"]
