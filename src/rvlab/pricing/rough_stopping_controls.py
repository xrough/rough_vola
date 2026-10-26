"""Exact local conditional expectations for finite-lift stopping controls.

The weak rough-Heston transition has six enumerable factor/stock-order
branches and one independent Gaussian stock shock.  A stopping surrogate
which is quadratic in normalized stock can be integrated over that Gaussian
*exactly*, after taking its maximum with the vanilla exercise payoff.  This
module supplies that local integral; it neither trains a policy nor replaces
the continuation value by its Gaussian expectation in a stopping decision.

Coefficients use ``q(z) = a + b*z + c*z**2``, with post-dividend
``z=max(S-D,0)/K``.  The cash convention is the simulator's limited-liability
jump.  ``payoff_side='both'`` permits exercise on either side of the event.
All expectations remain under the actual discrete transition law, so these
functions can center a genuine martingale without assuming that a simulated
stock or an approximate option price is itself an exact martingale.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from functools import lru_cache
import math

import numpy as np
from scipy.special import ndtr

from .rough_simulation import RoughSimulationError


def _arguments(state, coefficients, strike, is_call, cash_dividend, payoff_side,
               variance=None, stock_knots=()):
    if payoff_side not in ("pre", "post", "both"):
        raise ValueError("payoff_side must be 'pre', 'post', or 'both'")
    knots = np.asarray(stock_knots)
    if (knots.ndim != 1 or knots.dtype.kind not in "iuf" or not np.isfinite(knots).all()
            or np.any(knots <= 0) or np.any(np.diff(knots) <= 0)):
        raise ValueError("stock_knots must be a strictly increasing finite positive real sequence")
    arrays = [np.asarray(v) for v in (state, coefficients, strike, is_call, cash_dividend)]
    x, coefficients, strike, calls, cash = arrays
    if (coefficients.ndim == 0 or coefficients.shape[-1] != 3
            or any(a.dtype.kind not in "iuf" for a in (x, coefficients, strike, cash))
            or calls.dtype.kind != "b"):
        raise ValueError("state, strike, cash and three polynomial coefficients must be real; is_call must be boolean")
    if len(knots):
        if coefficients.ndim < 2 or coefficients.shape[-2] != len(knots)+1:
            raise ValueError("piecewise coefficients must have shape (..., len(stock_knots)+1, 3)")
        leading_shape = coefficients.shape[:-2]
    else:
        leading_shape = coefficients.shape[:-1]
        coefficients = coefficients[..., None, :]
    shape_inputs = [x.shape, leading_shape, strike.shape, calls.shape, cash.shape]
    if variance is not None:
        variance = np.asarray(variance)
        if variance.dtype.kind not in "iuf":
            raise ValueError("variance must be a finite nonnegative real array")
        shape_inputs.append(variance.shape)
    try:
        shape = np.broadcast_shapes(*shape_inputs)
    except ValueError as exc:
        raise ValueError("stopping states, coefficients and contracts must broadcast") from exc
    x, strike, calls, cash = [np.broadcast_to(a, shape) for a in (x, strike, calls, cash)]
    coefficients = np.broadcast_to(coefficients, shape+(len(knots)+1, 3)).astype(float)
    valid_state = np.isfinite(x) | ((variance is not None) & np.isneginf(x))
    if (not valid_state.all() or any(not np.isfinite(a).all() for a in (coefficients, strike, cash))
            or np.any(strike <= 0) or np.any(cash < 0)):
        raise ValueError("states and coefficients must be finite, strikes positive, and cash dividends nonnegative")
    if variance is not None:
        variance = np.broadcast_to(variance, shape).astype(float)
        if not np.isfinite(variance).all() or np.any(variance < 0):
            raise ValueError("variance must be a finite nonnegative real array")
    return (x.astype(float), coefficients, strike.astype(float), calls,
            cash.astype(float), variance, knots.astype(float))


def _pre_payoff(calls, payoff_side):
    if payoff_side == "pre":
        return np.ones(calls.shape, bool)
    if payoff_side == "post":
        return np.zeros(calls.shape, bool)
    # Calls decrease and puts increase at a cash jump.  Their dominating
    # intrinsic payoffs therefore come from these respective event sides.
    return calls


def _cap_argument(value_cap, calls, strike, *, exercise):
    """Validate payoff-dominating caps or a nonexercise proxy-only cap.

    A continuation or residual-premium proxy has no intrinsic floor when
    ``exercise=False``. Any nonnegative numeric cap is then permitted for
    either payoff direction; the same clipped function is integrated and
    evaluated, so its use does not alter exact martingale centering.
    """
    if value_cap is None:
        return None
    if isinstance(value_cap, str):
        if value_cap != "stock" or not np.all(calls):
            raise ValueError("value_cap='stock' is supported for calls only")
        return "stock"
    cap = np.asarray(value_cap)
    if cap.dtype.kind not in "iuf":
        raise ValueError("numeric value_cap must be a finite real cap")
    try:
        cap = np.broadcast_to(cap, strike.shape).astype(float)
    except ValueError as exc:
        raise ValueError("value_cap must broadcast to the stopping states") from exc
    if not exercise:
        if not np.isfinite(cap).all() or np.any(cap < 0):
            raise ValueError("nonexercise numeric value_cap must be finite and nonnegative")
        return cap
    if np.any(calls) or not np.isfinite(cap).all() or np.any(cap < strike):
        raise ValueError("numeric value_cap supports puts only and must be finite and at least strike")
    return cap


def stopping_value(spots, coefficients, *, strike, is_call, cash_dividend=0.,
                   payoff_side="pre", stock_knots=(), exercise=True,
                   value_cap=None):
    """Evaluate the nonnegative exercise/continuation surrogate pointwise.

    Leading coefficient axes broadcast with the other arguments.  Cash is
    subtracted only inside the continuation polynomial and (when requested)
    the post-event payoff; ``spots`` always denotes the pre-event stock.
    Without knots, coefficients have shape ``(..., 3)``. With strictly
    positive normalized post-stock ``stock_knots``, they have shape
    ``(..., len(stock_knots)+1, 3)``. At a knot the right segment is used;
    continuity is the fitting procedure's responsibility.
    ``exercise=False`` evaluates only ``max(q, 0)`` for integration dates at
    which the trained contract cannot exercise.
    Optional ``value_cap='stock'`` caps calls by pre-event stock; a numeric
    put cap must be at least strike (include any negative-rate allowance).
    With ``exercise=False`` any finite nonnegative numeric proxy cap is
    allowed for calls or puts, including residual premiums below strike.
    Caps bound the surrogate only, not the reported Monte Carlo estimator.
    """
    if not isinstance(exercise, (bool, np.bool_)):
        raise ValueError("exercise must be boolean")
    spots, coefficients, strike, calls, cash, _, knots = _arguments(
        spots, coefficients, strike, is_call, cash_dividend, payoff_side,
        stock_knots=stock_knots)
    cap = _cap_argument(value_cap, calls, strike, exercise=exercise)
    if np.any(spots < 0):
        raise ValueError("spots must be nonnegative")
    normalized = np.maximum(spots-cash, 0.)/strike
    segments = np.searchsorted(knots, normalized, side="right")
    coefficients = np.take_along_axis(coefficients, segments[..., None, None], axis=-2)[..., 0, :]
    continuation = coefficients[..., 0]+normalized*(coefficients[..., 1]
        +normalized*coefficients[..., 2])
    exercised = np.where(_pre_payoff(calls, payoff_side), spots, np.maximum(spots-cash, 0.))
    payoff = np.maximum(np.where(calls, exercised-strike, strike-exercised), 0.)
    result = np.maximum(payoff if exercise else 0., continuation)
    if cap is not None:
        result = np.minimum(result, spots if isinstance(cap, str) else cap)
    if not np.isfinite(result).all():
        raise RoughSimulationError("stopping surrogate evaluation overflowed")
    return result


def _interval_moments(mean, variance, lower, upper):
    """First three unnormalised lognormal moments on an open interval."""
    # Endpoints carry no mass here: this helper is used only at positive
    # variance.  Deterministic atoms are evaluated directly by the caller.
    lower = np.maximum(lower, 0.)
    upper = np.maximum(upper, 0.)
    active = upper > lower
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lo, hi = np.log(lower), np.log(upper)
        std = np.sqrt(variance)
        finite_band = (lower > 0) & np.isfinite(upper)
        # All moment powers must use the same band width. Subtracting two
        # separately standardized endpoints loses different low bits after
        # each power-dependent shift, which can corrupt the integral of a
        # large polynomial over a very thin cap-crossing interval.
        log_width = np.log1p(np.divide(upper-lower, lower,
            out=np.zeros_like(lower), where=finite_band))
        log_midpoint = lo+.5*log_width
        output = []
        for power in range(3):
            a = (lo-mean-power*variance)/std
            b = (hi-mean-power*variance)/std
            # Subtract survival probabilities in the upper tail, avoiding
            # the catastrophic 1-1 cancellation of a direct CDF difference.
            probability = np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
            middle = np.where(finite_band,
                (log_midpoint-mean-power*variance)/std, .5*a+.5*b)
            width = np.where(finite_band, log_width/std, b-a)
            close = (np.isfinite(middle) & (width > 0)
                & (width <= .01/np.maximum(1., np.abs(middle))))
            if np.any(close):
                # The ordinary CDF difference loses absolute ~eps near a
                # narrow interval around zero. Integrating the analytic
                # Gaussian-density Taylor series gives its small mass
                # directly. Here |middle*halfwidth| <= .005, so terms past
                # degree 8 are far below double roundoff, not a quadrature
                # or a changed pricing tolerance.
                m, half = middle[close], .5*width[close]
                m2, h2 = m*m, half*half
                total = 1+h2*((m2-1)/6+h2*((m2*m2-6*m2+3)/120
                    +h2*((m2**3-15*m2*m2+45*m2-15)/5040
                    +h2*(m2**4-28*m2**3+210*m2*m2-420*m2+105)/362880)))
                probability[close] = (np.exp(-.5*m*m)/np.sqrt(2*np.pi)
                    *width[close]*total)
            output.append(np.where(active, np.exp(power*mean+.5*power**2*variance)
                *np.maximum(probability, 0.), 0.))
    result = np.stack(output, axis=-1)
    if not np.isfinite(result).all():
        raise RoughSimulationError("conditional lognormal stopping moments overflowed")
    return result


@lru_cache(maxsize=16)
def _decimal_pi(precision):
    """Gauss--Legendre AGM pi with guard digits at the requested precision."""
    with localcontext() as context:
        context.prec = precision+12
        one, two = Decimal(1), Decimal(2)
        a, b, t, multiplier = one, one/two.sqrt(), one/4, one
        for _ in range(math.ceil(math.log2(precision))+2):
            next_a = (a+b)/2
            b = (a*b).sqrt()
            t -= multiplier*(a-next_a)**2
            a, multiplier = next_a, 2*multiplier
        return +(a+b)**2/(4*t)


def _decimal_normal_cdf(x, target, sqrt_two_pi):
    """Analytic positive erf series with a geometric remainder certificate."""
    if not x:
        return Decimal('.5'), Decimal(0)
    positive, x = x > 0, abs(x)
    exponential = (-x*x/2).exp()
    # Q(x) <= exp(-x²/2)/2 for x>=0. The caller scales this absolute
    # remainder by the complete moment coefficient before accepting it.
    tail_bound = exponential/2
    if tail_bound <= target:
        return (Decimal(1) if positive else Decimal(0)), tail_bound
    density = exponential/sqrt_two_pi
    total, term = x, x
    for index in range(10000):
        ratio = x*x/(2*index+3)
        following = term*ratio
        if ratio < 1:
            remainder = density*following/(1-ratio)
            if remainder <= target:
                shifted = density*total
                return (Decimal('.5')+shifted if positive else Decimal('.5')-shifted), remainder
        term, total = following, total+following
    raise RoughSimulationError("Decimal stopping Gaussian series exceeded its 10000-term budget")


def _decimal_polynomial_integral(coefficients, mean, variance, lower, upper,
                                 *, minimum_precision=60):
    """Recompute the same three analytic moments when float sums cancel.

    This fallback is not Gaussian quadrature: all inputs are reconstructed
    with Decimal.from_float, CDFs use a convergent analytic erf series, and
    the original polynomial is integrated by its three lognormal moments.
    A weighted series/tail, arithmetic and float-conversion error bound is
    returned. Budgets are 60--100 decimal digits and 10000 series terms.
    """
    try:
        decimal_coefficients = [Decimal.from_float(float(c)) for c in coefficients]
        mu, v = Decimal.from_float(float(mean)), Decimal.from_float(float(variance))
        log_weight = max([math.log(abs(float(c)))+p*float(mean)+.5*p*p*float(variance)
            for p, c in enumerate(coefficients) if c]+[0.])
        precision = max(minimum_precision, 45+math.ceil(max(log_weight, 0.)/math.log(10.)))
        if precision > 100:
            raise RoughSimulationError("stopping moment cancellation requires more than 100 decimal digits")
        with localcontext() as context:
            context.prec = precision
            sigma = v.sqrt()
            lo = None if lower == 0 else Decimal.from_float(float(lower)).ln()
            hi = None if np.isinf(upper) else Decimal.from_float(float(upper)).ln()
            weights = [c*(p*mu+Decimal(p*p)*v/2).exp()
                for p, c in enumerate(decimal_coefficients)]
            total_weight = sum(abs(weight) for weight in weights)
            target = Decimal('1e-28')/(1+total_weight)
            sqrt_two_pi = (2*_decimal_pi(precision)).sqrt()
            value, remainder = Decimal(0), Decimal(0)
            for power, weight in enumerate(weights):
                if not weight:
                    continue
                a, ae = ((Decimal(0), Decimal(0)) if lo is None else
                    _decimal_normal_cdf((lo-mu-power*v)/sigma, target, sqrt_two_pi))
                b, be = ((Decimal(1), Decimal(0)) if hi is None else
                    _decimal_normal_cdf((hi-mu-power*v)/sigma, target, sqrt_two_pi))
                value += weight*(b-a)
                remainder += abs(weight)*(ae+be)
            # Fifteen guard digits cover the bounded series/AGM arithmetic;
            # retain float conversion in the final certificate as well.
            remainder += (1+total_weight)*Decimal(10)**(-precision+15)
            converted = float(value)
            if not math.isfinite(converted):
                raise RoughSimulationError("Decimal stopping moment integral overflowed on float conversion")
            remainder += abs(Decimal.from_float(converted)-value)
            error = float(remainder)
            if not math.isfinite(error) or error > 1e-9*(1+abs(converted)):
                raise RoughSimulationError("Decimal stopping integral failed its unchanged local arithmetic precision check")
            return converted, error, precision
    except (ArithmeticError, ValueError) as exc:
        raise RoughSimulationError(f"Decimal stopping moment calculation failed: {exc}") from exc


def _polynomial_integral(coefficients, mean, variance, lower, upper):
    shape = np.broadcast_shapes(coefficients.shape[:-1], mean.shape, variance.shape,
        lower.shape, upper.shape)
    coefficients = np.broadcast_to(coefficients, shape+(3,))
    mean, variance, lower, upper = [np.broadcast_to(a, shape) for a in (mean, variance, lower, upper)]
    active = (upper > np.maximum(lower, 0.)) & np.any(coefficients != 0, axis=-1)
    result = np.zeros(shape)
    if not np.any(active):
        return result
    moments = _interval_moments(mean[active], variance[active], lower[active], upper[active])
    terms = coefficients[active].astype(np.longdouble)*moments.astype(np.longdouble)
    value = np.asarray(np.sum(terms, axis=-1), float)
    magnitude = np.asarray(np.sum(np.abs(terms), axis=-1), float)
    tolerance = 256*np.finfo(float).eps*np.maximum(1., magnitude)
    precision_bound = 8*np.finfo(float).eps*magnitude
    uncertain = precision_bound > 1e-9*(1+np.abs(value))
    if np.any(uncertain):
        selected_coefficients = coefficients[active]
        selected_mean, selected_variance, selected_lower, selected_upper = (
            a[active] for a in (mean, variance, lower, upper))
        for index in np.flatnonzero(uncertain):
            accurate, error, _ = _decimal_polynomial_integral(selected_coefficients[index],
                selected_mean[index], selected_variance[index], selected_lower[index], selected_upper[index])
            value[index] = accurate
            tolerance[index] = error+256*np.finfo(float).eps*max(1., abs(accurate))
    if np.any(value < -tolerance):
        raise RoughSimulationError("positive stopping-polynomial integral became materially negative")
    result[active] = np.maximum(value, 0.)
    return result


def _positive_quadratic(coefficients, mean, variance, lower, upper):
    """Integral of a quadratic's positive part on a positive-stock interval."""
    c, b, a = (coefficients[..., j] for j in range(3))
    quadratic = a != 0
    discriminant = b*b-4*a*c
    roots = quadratic & (discriminant > 0)
    root = np.sqrt(np.maximum(discriminant, 0.))
    q = -.5*(b+np.where(b >= 0, root, -root))
    r1 = np.divide(q, a, out=np.zeros_like(q), where=roots)
    r2 = np.divide(c, q, out=np.zeros_like(q), where=roots & (q != 0))
    lowroot, highroot = np.minimum(r1, r2), np.maximum(r1, r2)
    # A convex quadratic is positive outside its two real roots; a concave
    # one inside.  Repeated roots have no mass and need no special interval.
    left_end = np.where(roots, np.minimum(upper, lowroot), lower)
    right_start = np.where(roots, np.maximum(lower, highroot), upper)
    middle_lo = np.where(roots, np.maximum(lower, lowroot), lower)
    middle_hi = np.where(roots, np.minimum(upper, highroot), lower)
    full = (quadratic & ~roots & (a > 0)) | (~quadratic & (b == 0) & (c > 0))
    linear = ~quadratic & (b != 0)
    linear_root = np.divide(-c, b, out=np.zeros_like(b), where=linear)
    first_lo = np.where(roots & (a < 0), middle_lo, lower)
    first_hi = np.where(roots & (a < 0), middle_hi, lower)
    first_hi = np.where(roots & (a > 0), left_end, first_hi)
    first_hi = np.where(full, upper, first_hi)
    first_lo = np.where(linear & (b > 0), np.maximum(lower, linear_root), first_lo)
    first_hi = np.where(linear & (b > 0), upper, first_hi)
    first_hi = np.where(linear & (b < 0), np.minimum(upper, linear_root), first_hi)
    second_lo = np.where(roots & (a > 0), right_start, upper)
    lo, hi = np.stack((first_lo, second_lo), axis=-1), np.stack((first_hi, upper), axis=-1)
    return _polynomial_integral(coefficients[..., None, :], mean[..., None],
        variance[..., None], lo, hi).sum(axis=-1)


def _quadratic_roots(coefficients):
    """Real roots with scale-safe discriminants; invalid roots are infinity."""
    scale = np.max(np.abs(coefficients), axis=-1)
    normalized = np.divide(coefficients, scale[..., None], out=np.zeros_like(coefficients),
        where=scale[..., None] > 0)
    c, b, a = (normalized[..., j] for j in range(3))
    discriminant = b*b-4*a*c
    # Repeated roots must partition the interval too: otherwise a midpoint
    # can land exactly at the touching point and classify an entire interval
    # by that isolated equality rather than its polynomial on either side.
    quadratic = (a != 0) & (discriminant >= 0)
    q = -.5*(b+np.where(b >= 0, 1., -1.)*np.sqrt(np.maximum(discriminant, 0.)))
    first = np.divide(q, a, out=np.full_like(a, np.inf), where=quadratic)
    second = np.divide(c, q, out=np.full_like(a, np.inf), where=quadratic & (q != 0))
    linear = (a == 0) & (b != 0)
    first = np.where(linear, np.divide(-c, b, out=np.zeros_like(b), where=linear), first)
    return np.stack((first, second), axis=-1)


def _bounded_polynomial_integral(q, base, cap, mean, variance, lower, upper):
    """Integrate min(max(base,q),cap) without subtracting huge tail values.

    The algebraic positive-part difference can lose the complete bounded
    answer when an extrapolated q is enormous. Splitting at both crossing
    pairs integrates the actually selected bounded polynomial instead.
    """
    difference, cap_difference = q-base, q-cap
    base_roots = _quadratic_roots(difference)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_roots = np.log(base_roots)
        relevant = np.zeros(base_roots.shape, bool)
        for power in range(3):
            distance = np.abs((log_roots-mean[:, None]-power*variance[:, None])
                /np.sqrt(variance[:, None]))
            relevant |= ndtr(-distance) > 0
    lost_band = (np.all(difference == cap_difference, axis=-1)
        & np.any(base != cap, axis=-1)
        & np.any((base_roots > lower[:, None]) & (base_roots < upper[:, None])
            & relevant, axis=-1))
    if np.any(lost_band):
        raise RoughSimulationError("bounded stopping cap crossings are unresolved at floating-point precision")
    roots = np.concatenate((base_roots, _quadratic_roots(cap_difference)), axis=-1)
    roots = np.clip(roots, lower[:, None], upper[:, None])
    boundaries = np.sort(np.column_stack((lower, roots, upper)), axis=-1)
    lo, hi = boundaries[:, :-1], boundaries[:, 1:]
    with np.errstate(over="ignore", invalid="ignore"):
        mid = np.where(np.isfinite(hi), .5*lo+.5*hi, lo+np.maximum(1., np.abs(lo)))
    mid = np.minimum(mid, np.finfo(float).max)
    common = np.maximum.reduce((np.max(np.abs(q), axis=-1),
        np.max(np.abs(base), axis=-1), np.max(np.abs(cap), axis=-1), np.ones(len(q))))

    def evaluate(coefficients):
        a = coefficients/common[:, None]
        inverse = np.divide(1., mid, out=np.zeros_like(mid), where=mid > 1)
        # For large z compare q/z² to avoid overflow in q(z). All compared
        # polynomials share the same positive normalization.
        small_z = np.minimum(mid, 1.)
        ordinary = a[:, 0, None]+small_z*(a[:, 1, None]+small_z*a[:, 2, None])
        scaled = a[:, 2, None]+inverse*(a[:, 1, None]+inverse*a[:, 0, None])
        return np.where(mid > 1, scaled, ordinary)

    q_value, base_value, cap_value = evaluate(q), evaluate(base), evaluate(cap)
    selected = np.where((q_value <= base_value)[..., None], base[:, None, :], q[:, None, :])
    selected = np.where((q_value >= cap_value)[..., None], cap[:, None, :], selected)
    return _polynomial_integral(selected, mean[:, None], variance[:, None], lo, hi).sum(axis=-1)


def expected_stopping_value(log_mean, variance, coefficients, *, strike, is_call,
                            cash_dividend=0., payoff_side="pre", stock_knots=(),
                            exercise=True, value_cap=None):
    """Integrate ``stopping_value`` exactly under a lognormal stock law.

    ``log_mean`` and ``variance`` are the mean and variance of log pre-event
    stock.  A deterministic variance-zero law is evaluated without smoothing;
    ``log_mean=-inf`` denotes an absorbing zero-stock atom after a cash cap.
    No numerical Gaussian quadrature or nested sample mean is used.  This is
    a local conditional expectation, not an American price or a statistical
    upper bound by itself. ``exercise=False`` integrates ``max(q, 0)`` and
    never introduces an intrinsic payoff at a nonexercise integration date.
    """
    if not isinstance(exercise, (bool, np.bool_)):
        raise ValueError("exercise must be boolean")
    mean, coefficients, strike, calls, cash, variance, knots = _arguments(
        log_mean, coefficients, strike, is_call, cash_dividend, payoff_side, variance,
        stock_knots=stock_knots)
    cap = _cap_argument(value_cap, calls, strike, exercise=exercise)
    shape = mean.shape
    with np.errstate(over="ignore", invalid="ignore"):
        deterministic_spot = np.exp(mean)
    deterministic = (variance == 0) | np.isneginf(mean)
    if np.any(deterministic & ~np.isfinite(deterministic_spot)):
        raise RoughSimulationError("deterministic stopping stock overflowed")
    # Work only on stochastic entries to keep their moment denominators
    # strictly positive.  Flattening also gives identical scalar/array paths.
    mask = ~deterministic.ravel()
    result = np.empty(mean.size)
    if np.any(deterministic):
        idx = deterministic.ravel()
        selected = coefficients.reshape(-1, len(knots)+1, 3)[idx]
        if not len(knots):
            selected = selected[:, 0, :]
        result[idx] = stopping_value(deterministic_spot.ravel()[idx],
            selected, strike=strike.ravel()[idx],
            is_call=calls.ravel()[idx], cash_dividend=cash.ravel()[idx],
            payoff_side=payoff_side, stock_knots=knots, exercise=exercise,
            value_cap=(cap if cap is None or isinstance(cap, str) else cap.ravel()[idx]))
    if not np.any(mask):
        return result.reshape(shape)
    mean = mean.ravel()[mask]-np.log(strike.ravel()[mask])
    variance, strike, calls, cash = (a.ravel()[mask] for a in (variance, strike, calls, cash))
    coefficients = coefficients.reshape(-1, len(knots)+1, 3)[mask]
    if cap is not None and not isinstance(cap, str):
        cap = cap.ravel()[mask]
    pre = _pre_payoff(calls, payoff_side)
    d = cash/strike
    payoff_knot = np.where(pre, 1., 1.+d)
    internal = np.sort(np.column_stack((d, payoff_knot, d[:, None]+knots)
        if exercise else (d, d[:, None]+knots)), axis=1)
    boundaries = [np.zeros_like(d)]+list(internal.T)+[np.full_like(d, np.inf)]
    accumulated = np.zeros_like(d)
    cap_polynomial = None
    if cap is not None:
        cap_polynomial = np.zeros((len(d), 3))
        if isinstance(cap, str):
            cap_polynomial[:, 1] = strike
        else:
            cap_polynomial[:, 0] = cap
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mid = np.where(np.isfinite(upper), .5*(lower+upper), lower+1.)
        above_cash = mid > d
        segment = np.searchsorted(knots, np.maximum(mid-d, 0.), side="right")
        q = coefficients[np.arange(len(d)), segment].copy()
        q[:, 0] = np.where(above_cash, q[:, 0]-d*q[:, 1]+d*d*q[:, 2], q[:, 0])
        q[:, 1] = np.where(above_cash, q[:, 1]-2*d*q[:, 2], 0.)
        q[:, 2] = np.where(above_cash, q[:, 2], 0.)
        payoff_stock = np.where(pre, mid, np.maximum(mid-d, 0.))
        in_money = np.where(calls, payoff_stock > 1., payoff_stock < 1.)
        linear_stock = pre | above_cash
        sign = np.where(calls, 1., -1.)
        intercept = np.where(pre, -sign*strike, -sign*strike*(1.+d))
        intercept = np.where(~pre & ~above_cash, np.where(calls, 0., strike), intercept)
        base = np.stack((np.where(in_money, intercept, 0.),
            np.where(in_money & linear_stock, sign*strike, 0.), np.zeros_like(d)), axis=-1)
        if not exercise:
            base[:] = 0.
        if cap_polynomial is not None:
            accumulated += _bounded_polynomial_integral(q, base, cap_polynomial,
                mean, variance, lower, upper)
        elif exercise:
            accumulated += _polynomial_integral(base, mean, variance, lower, upper)
            accumulated += _positive_quadratic(q-base, mean, variance, lower, upper)
        else:
            accumulated += _positive_quadratic(q, mean, variance, lower, upper)
    if not np.isfinite(accumulated).all():
        raise RoughSimulationError("conditional stopping-surrogate expectation overflowed")
    if cap is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            expected_cap = (np.exp(mean+np.log(strike)+.5*variance)
                if isinstance(cap, str) else cap)
        tolerance = 256*np.finfo(float).eps*np.maximum(1., expected_cap)
        if (not np.isfinite(expected_cap).all() or np.any(accumulated < -tolerance)
                or np.any(accumulated > expected_cap+tolerance)):
            raise RoughSimulationError("bounded stopping expectation violated its nonnegative surrogate cap")
    result[mask] = accumulated
    return result.reshape(shape)


__all__ = ["stopping_value", "expected_stopping_value"]
