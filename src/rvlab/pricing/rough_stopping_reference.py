"""Independent Gaussian dynamic programming on a declared exercise calendar.

This is a numerical reference for the constant-variance stopping experiments,
not a new public contract style. Values are interpolated linearly in STOCK;
each Gaussian transition integrates that interpolant analytically. Uniform
log-stock spacing makes the two moment sums convolutions. No Monte Carlo,
regression policy, Heston Fourier proxy or stopping-control helper is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
import math

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ndtr

from .vanilla_types import SpotMarket, UnsupportedModelError


@dataclass(frozen=True)
class StoppingCalendarReference:
    price: float
    delta: float
    settings: object
    diagnostics: object

    def __post_init__(self):
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _cdf_band(a, b):
    """Stable Gaussian interval probability, including far-right tails."""
    return np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))


def _gaussian_stock_linear_step(values, stocks, mean, variance, *,
        left_intercept, left_slope, right_intercept, right_slope):
    """Exact expectation of a stock-linear interpolant with linear tails."""
    if variance == 0:
        future = stocks*math.exp(mean)
        result = np.interp(future, stocks, values)
        return np.where(future < stocks[0], left_intercept+left_slope*future,
            np.where(future > stocks[-1], right_intercept+right_slope*future, result))
    n = len(stocks)
    spacing = math.log(stocks[1]/stocks[0])
    std = math.sqrt(variance)
    offsets = np.arange(1-n, n)*spacing
    lower, upper = (offsets-mean)/std, (offsets+spacing-mean)/std
    probability = _cdf_band(lower, upper)
    first_moment = np.exp(mean+.5*variance-offsets)*_cdf_band(lower-std, upper-std)
    ratio_minus_one = math.expm1(spacing)
    scaled_slope = np.diff(values)/ratio_minus_one
    intercept = values[:-1]-scaled_slope
    result = (fftconvolve(intercept, probability[::-1])[n-1:2*n-1]
        +fftconvolve(scaled_slope, first_moment[::-1])[n-1:2*n-1])
    # The integral beyond the two finite grid endpoints uses known deep-ITM
    # asymptotes, independently checked by expanding the log-stock domain.
    left = (np.log(stocks[0]/stocks)-mean)/std
    right = (np.log(stocks[-1]/stocks)-mean)/std
    forward = stocks*math.exp(mean+.5*variance)
    result += left_intercept*ndtr(left)+left_slope*forward*ndtr(left-std)
    result += right_intercept*ndtr(-right)+right_slope*forward*ndtr(std-right)
    if not np.isfinite(result).all():
        raise FloatingPointError("Gaussian stopping-reference convolution is nonfinite")
    return result


def _calendar_grid(market, volatility, strike, times, is_call, nodes, width, style, return_surface=False):
    if len(times) == 1:
        if return_surface:
            stocks = market.spot*np.exp(np.linspace(-width, width, nodes))
            return stocks, np.maximum(stocks-strike if is_call else strike-stocks, 0.)[None, :]
        price = max(market.spot-strike if is_call else strike-market.spot, 0.)
        delta = (.5 if is_call else -.5) if market.spot == strike else (
            float(market.spot > strike) if is_call else -float(market.spot < strike))
        return price, delta
    stocks = market.spot*np.exp(np.linspace(-width, width, nodes))
    if not stocks[1] < strike < stocks[-2]:
        raise ValueError("reference stock domain does not resolve the strike; increase log_half_width")
    discounts = np.asarray(market.discount_curve.discount(times))
    payoff = np.maximum(stocks-strike if is_call else strike-stocks, 0.)
    values = payoff.copy()
    surface = np.empty((len(times), nodes)) if return_surface else None
    if surface is not None:
        surface[-1] = values
    for j in range(len(times)-2, -1, -1):
        duration = times[j+1]-times[j]
        growth = math.log(discounts[j]/discounts[j+1])
        future_discounts = discounts[j+1:] if style == "american" else discounts[-1:]
        if is_call:
            tail = dict(left_intercept=0., left_slope=0.,
                right_intercept=-strike*float(np.min(future_discounts))/discounts[j+1], right_slope=1.)
        else:
            tail = dict(left_intercept=strike*float(np.max(future_discounts))/discounts[j+1],
                left_slope=-1., right_intercept=0., right_slope=0.)
        continuation = discounts[j+1]/discounts[j]*_gaussian_stock_linear_step(
            values, stocks, growth-.5*volatility**2*duration, volatility**2*duration, **tail)
        values = np.maximum(payoff, continuation) if style == "american" else continuation
        if surface is not None:
            surface[j] = values
    if surface is not None:
        return stocks, surface
    center = nodes//2
    # Three-point differentiation in stock, with the actual unequal spacings.
    delta = float(np.gradient(values, stocks, edge_order=2)[center])
    return float(values[center]), delta


def black_scholes_stopping_reference(market, *, volatility, strike, exercise_times,
        is_call=False, spot_nodes=16385, log_half_width=2.,
        exercise_style="american", check_refinement=True):
    """No-cash BS reference with the SAME finite exercise dates as a simulation.

    ``check_refinement`` doubles spatial intervals and separately doubles the
    domain at fixed fine spacing. The returned price uses the fine spatial
    grid; all changes and unchanged price/delta tolerances are reported.
    The finite calendar is not a continuous-American reference.
    """
    if not isinstance(market, SpotMarket):
        raise TypeError("market must be SpotMarket")
    for name, value in (("volatility", volatility), ("strike", strike), ("log_half_width", log_half_width)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or (value < 0 if name == "volatility" else value <= 0)):
            raise ValueError(f"invalid reference {name}")
    if (isinstance(spot_nodes, (bool, np.bool_)) or not isinstance(spot_nodes, Integral)
            or spot_nodes < 33 or spot_nodes % 2 == 0):
        raise ValueError("spot_nodes must be an odd integer >= 33")
    if not isinstance(is_call, (bool, np.bool_)) or not isinstance(check_refinement, (bool, np.bool_)):
        raise ValueError("is_call and check_refinement must be boolean")
    if exercise_style not in ("american", "european"):
        raise ValueError("reference exercise_style must be american or european")
    times = np.asarray(exercise_times)
    if (times.dtype.kind not in "iuf" or times.ndim != 1 or len(times) == 0
            or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("exercise_times must be increasing finite dates starting at zero")
    times = times.astype(float)
    maturity = float(times[-1])
    if maturity > market.dividends.coverage:
        raise ValueError("reference dividend information does not cover maturity")
    if any(event.amount > 0 and event.time <= maturity for event in market.dividends.events):
        raise UnsupportedModelError("Gaussian stopping-calendar reference does not support cash dividends")
    market.discount_curve.discount(maturity)
    options = (market, float(volatility), float(strike), times, bool(is_call))
    base = _calendar_grid(*options, int(spot_nodes), float(log_half_width), exercise_style)
    fine, domain = base, None
    if check_refinement:
        fine = _calendar_grid(*options, 2*int(spot_nodes)-1, float(log_half_width), exercise_style)
        domain = _calendar_grid(*options, 4*int(spot_nodes)-3, 2*float(log_half_width), exercise_style)
    price_allowance, delta_allowance = .001+1e-4*abs(fine[0]), .002+.005*abs(fine[1])
    changes = {"spatial_price_change": abs(fine[0]-base[0]), "spatial_delta_change": abs(fine[1]-base[1]),
        "domain_price_change": None if domain is None else abs(domain[0]-fine[0]),
        "domain_delta_change": None if domain is None else abs(domain[1]-fine[1])}
    return StoppingCalendarReference(fine[0], fine[1],
        {"backend": "gaussian_stock_linear_calendar_reference", "spot_nodes": int(spot_nodes),
            "fine_spot_nodes": 2*int(spot_nodes)-1 if check_refinement else int(spot_nodes),
            "log_half_width": float(log_half_width), "exercise_times": times.tolist(),
            "exercise_style": exercise_style, "volatility": float(volatility), "strike": float(strike)},
        {**changes, "price_allowance": price_allowance, "delta_allowance": delta_allowance,
            "price_stable": bool(check_refinement and changes["spatial_price_change"] <= price_allowance
                and changes["domain_price_change"] <= price_allowance),
            "delta_stable": bool(check_refinement and changes["spatial_delta_change"] <= delta_allowance
                and changes["domain_delta_change"] <= delta_allowance),
            "reference_change_diagnostic": max(changes["spatial_price_change"],
                changes["domain_price_change"] or 0.),
            "scope": "No-cash Black-Scholes on declared finite exercise calendar; not continuous exercise"})


def black_scholes_stopping_surface(market, *, volatility, strike, exercise_times,
        is_call=False, spot_nodes=4097, log_half_width=2., exercise_style="american"):
    """Return nondiscounted date values on a stock grid for surrogate construction.

    Arrays are read-only. A surrogate built from this approximate table must
    still center its *actual interpolant* under the simulation law; this table
    is never a substitute for that conditional expectation or the rough price.
    """
    checked = black_scholes_stopping_reference(market, volatility=volatility,
        strike=strike, exercise_times=exercise_times, is_call=is_call,
        spot_nodes=spot_nodes, log_half_width=log_half_width,
        exercise_style=exercise_style, check_refinement=False)
    times = np.asarray(exercise_times, float).copy()
    stocks, values = _calendar_grid(market, float(volatility), float(strike), times,
        bool(is_call), int(spot_nodes), float(log_half_width), exercise_style, True)
    for array in (times, stocks, values):
        array.setflags(write=False)
    return {"times": times, "stocks": stocks, "values": values,
        "settings": dict(checked.settings), "diagnostics": dict(checked.diagnostics)}
