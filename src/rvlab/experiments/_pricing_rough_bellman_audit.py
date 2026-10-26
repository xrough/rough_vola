"""Independent scalar integration for approximate Bellman-policy audits.

This helper integrates a supplied endpoint function over a composed discrete
transition law. It is not a policy fitter or a martingale-centering routine.
Adaptive-quadrature errors and refinement changes are numerical diagnostics,
not rigorous integration bounds or guarantees of policy improvement.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr


def clipped_premium_breakpoints(model, factor, strike, cap):
    """Locate stock-spline knots and raw quadratic premium roots at0/cap.

    Coefficients are read from the frozen model, while its direct ``value``
    method remains the integrand. Roots are used only to split independent
    quadrature; no fitted coefficient or endpoint value is changed.
    """
    factor = np.asarray(factor, float)
    knots = np.asarray(model.knots, float)
    if (factor.ndim != 1 or not np.isfinite(factor).all() or knots.ndim != 1
            or not np.isfinite(knots).all() or np.any(knots <= 0) or np.any(np.diff(knots) <= 0)
            or not np.isfinite(strike) or strike <= 0 or not np.isfinite(cap) or cap <= 0):
        raise ValueError("Frozen premium breakpoints require finite states, ordered knots and positive strike/cap")
    coefficients = np.asarray(model.coefficients(factor[None]), float)[0]
    if not len(knots):
        coefficients = np.atleast_2d(coefficients)
    if coefficients.shape != (len(knots)+1, 3) or not np.isfinite(coefficients).all():
        raise ValueError("Frozen premium must expose finite quadratic coefficients by stock interval")
    bounds = np.r_[0., knots, np.inf]
    points = [1., *knots]
    for (a, b, c), left, right in zip(coefficients, bounds[:-1], bounds[1:]):
        for target in (0., cap):
            constant = a-target
            if c == 0:
                roots = [-constant/b] if b != 0 else []
            else:
                scale = max(abs(c), abs(b), abs(constant))
                aa, bb, cc = constant/scale, b/scale, c/scale
                discriminant = bb*bb-4*aa*cc
                if discriminant < 0:
                    roots = []
                else:
                    stable = -.5*(bb+np.copysign(np.sqrt(discriminant), bb))
                    roots = [0.] if stable == 0 else [stable/cc, aa/stable]
            for value in roots:
                if np.isfinite(value) and left < value < right and value > 0:
                    points.append(float(value))
    return strike*np.unique(points)


def _function_values(function, stocks, factor):
    stocks = np.asarray(stocks, float)
    values = np.asarray(function(stocks, np.broadcast_to(factor, stocks.shape+(len(factor),))), float)
    if values.shape != stocks.shape or not np.isfinite(values).all():
        raise ValueError("Endpoint audit functions must return finite values aligned with supplied stocks")
    return values


def _split_points(mean, sigma, factor, stock_breakpoints, kink_function, normal_limit):
    knots = stock_breakpoints(factor) if callable(stock_breakpoints) else stock_breakpoints
    knots = np.asarray(knots, float)
    if knots.ndim != 1 or not np.isfinite(knots).all() or np.any(knots <= 0):
        raise ValueError("Endpoint kink breakpoints must be finite positive stocks")
    mapped = (np.log(knots)-mean)/sigma
    points = np.unique(np.r_[np.linspace(-normal_limit, normal_limit, 25),
        mapped[(mapped > -normal_limit) & (mapped < normal_limit)]])
    roots = []
    if kink_function is not None:
        values = _function_values(kink_function, np.exp(mean+sigma*points), factor)
        for left, right, a, b in zip(points[:-1], points[1:], values[:-1], values[1:]):
            if a*b < 0:
                root = brentq(lambda z: float(_function_values(kink_function,
                    np.array([np.exp(mean+sigma*z)]), factor)[0]), left, right,
                    xtol=2e-13, rtol=8*np.finfo(float).eps)
                roots.append(root)
        # A whole zero interval contributes no extra crossing. Its existing
        # endpoints are already retained in the numerical partition.
    return np.unique(np.r_[points, roots]), len(roots)


def adaptive_block_expectation(law, left_stocks, value_function, *, stock_breakpoints=(),
        kink_function=None, normal_limit=12., absolute_tolerance=1e-9, relative_tolerance=1e-10,
        envelope_constant=None, envelope_stock=None):
    """Integrate each branch/order using an independent scalar normal integral.

    ``value_function(stocks,factors)`` returns right-date, undiscounted values.
    The caller supplies calendar discounting separately. A ``kink_function``
    such as intrinsic-minus-continuation locates sign-changing Snell roots;
    spline knots and polynomial clipping roots may be supplied as positive
    ``stock_breakpoints`` (or a callable of the right factor state).

    Optional nonnegative ``A,B`` declare the caller's global envelope
    ``abs(value(s,u)) <= A+B*s``. Only then is an omitted-tail value bound
    returned. Without that declaration Gaussian tail probability alone does
    not control value error. No omitted-tail probability is renormalized.
    """
    stock = np.asarray(left_stocks, float)
    try:
        factors = np.asarray(law["new_factors"], float)
        probability = np.asarray(law["probabilities"], float)
        shifts = np.asarray(law["log_stock_shifts"], float)
        variance = np.asarray(law["orthogonal"], float)
        endpoint_probability = float(law["endpoint_probability"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("An explicit composed branch/order transition law is required") from error
    if (stock.ndim != 1 or not len(stock) or not np.isfinite(stock).all() or np.any(stock <= 0)
            or factors.ndim != 3 or factors.shape[0] != len(stock) or factors.shape[-1] < 1
            or probability.shape != factors.shape[:2] or shifts.ndim != 3
            or shifts.shape[:2] != probability.shape or shifts.shape[-1] < 1
            or variance.shape != shifts.shape or np.any(variance < 0)
            or not all(np.isfinite(v).all() for v in (factors, probability, shifts, variance))
            or np.any(probability < 0) or not np.allclose(probability.sum(axis=1), 1., rtol=0., atol=2e-12)
            or endpoint_probability != 1./shifts.shape[-1]):
        raise ValueError("Finite aligned factor branches, Gaussian clocks and normalized unmodified probabilities required")
    for name, value in (("normal_limit", normal_limit), ("absolute_tolerance", absolute_tolerance),
            ("relative_tolerance", relative_tolerance)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 8 <= normal_limit <= 14:
        raise ValueError("normal_limit must lie in[8,14]")
    if (envelope_constant is None) != (envelope_stock is None):
        raise ValueError("Declare both terms of the global value envelope, or neither")
    if envelope_constant is not None and any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            or not np.isfinite(v) or v < 0 for v in (envelope_constant, envelope_stock)):
        raise ValueError("The declared global value envelope must have finite nonnegative coefficients")
    n = len(stock)
    expectation, errors, tails = np.zeros(n), np.zeros(n), np.zeros(n)
    roots, evaluations = np.zeros(n, int), np.zeros(n, int)
    for path in range(n):
        for branch in range(probability.shape[1]):
            factor = factors[path, branch]
            for order in range(shifts.shape[-1]):
                mass = probability[path, branch]*endpoint_probability
                if mass == 0:
                    continue
                mean = float(np.log(stock[path])+shifts[path, branch, order])
                clock = float(variance[path, branch, order])
                if clock == 0:
                    expectation[path] += mass*float(_function_values(value_function, np.array([np.exp(mean)]), factor)[0])
                    evaluations[path] += 1
                    continue
                sigma = np.sqrt(clock)
                points, count = _split_points(mean, sigma, factor, stock_breakpoints, kink_function, normal_limit)
                roots[path] += count
                def integrand(z):
                    evaluations[path] += 1
                    value = float(_function_values(value_function, np.array([np.exp(mean+sigma*z)]), factor)[0])
                    return value*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
                integral, error = quad(integrand, -normal_limit, normal_limit, points=points[1:-1],
                    epsabs=absolute_tolerance, epsrel=relative_tolerance, limit=max(200, 2*len(points)))
                expectation[path] += mass*integral
                errors[path] += mass*error
                if envelope_constant is not None:
                    tails[path] += mass*(envelope_constant*2*ndtr(-normal_limit)
                        +envelope_stock*np.exp(mean+.5*clock)
                        *(ndtr(-normal_limit-sigma)+ndtr(sigma-normal_limit)))
    if not all(np.isfinite(v).all() for v in (expectation, errors, tails)):
        raise FloatingPointError("Independent Bellman endpoint integration returned nonfinite diagnostics")
    return {"expectation": expectation, "quadrature_error_estimate": errors,
        "normal_tail_probability": 2*ndtr(-normal_limit),
        "tail_value_bound": tails if envelope_constant is not None else None,
        "tail_envelope_declared_by_caller": envelope_constant is not None,
        "normal_limit": normal_limit, "detected_snell_crossings": roots,
        "function_evaluations": evaluations, "quadrature_error_is_rigorous_bound": False,
        "discounting_applied": False, "gaussian_tail_renormalized": False,
        "policy_improvement_guaranteed": False, "changes_martingale_centering": False}


__all__ = ["adaptive_block_expectation", "clipped_premium_breakpoints"]
