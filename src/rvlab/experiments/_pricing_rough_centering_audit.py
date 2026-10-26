"""Independent scalar Gaussian checks of rough stopping-control centering.

This is a numerical audit at fixed, independently generated LEFT states. It
does not refit controls, select paths by outcomes, or infer Monte Carlo bias
from a finite collection of conditional tests.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit

from rvlab.pricing.rough_estimators import _AffineSurrogate, _conditional_payoffs
from rvlab.pricing.rough_simulation import (
    _drift_flow_with_integral, _theta, _weak_transition_branches)
from rvlab.pricing.rough_stopping_controls import expected_stopping_value
from rvlab.pricing.rough_stopping_fourier import (
    FourierStoppingPolicy, _affine_values, _continuation_coefficients,
    _interpolate, fourier_policy_fingerprint)


def _scalar_premium(stock, coefficient, strike, knots, cap):
    """Direct scalar polynomial, independent of vectorized evaluation helpers."""
    z = stock/strike
    coefficient = np.asarray(coefficient)
    c = (coefficient if coefficient.ndim == 1 else
        coefficient[int(np.searchsorted(knots, z, side="right"))])
    value = max(0., float(c[0]+z*(c[1]+z*c[2])))
    return value if cap is None else min(value, cap)


def _integration_points(mean, variance, strike, coefficient, knots, cap, limit):
    points = [-limit, limit]
    stocks = [strike*float(k) for k in knots]
    edges = [0., *knots, np.inf]
    for j, c in enumerate(np.atleast_2d(coefficient)):
        for level in ((0.,) if cap is None else (0., cap)):
            roots = np.roots([c[2], c[1], c[0]-level])
            stocks.extend(strike*float(root.real) for root in roots
                if abs(root.imag) < 1e-12 and edges[j] < root.real < edges[j+1])
    for stock in stocks:
        if stock > 0:
            value = (math.log(stock)-mean)/math.sqrt(variance)
            if -limit < value < limit:
                points.append(value)
    return np.unique(points)


def _source_identity():
    pricing = Path(__file__).resolve().parents[1]/"pricing"
    paths = [*pricing.glob("*.py"), Path(__file__)]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def audit_conditional_centers(policy, trace, *, state_indices,
                              martingale_weights=None,
                              absolute_quadrature_tolerance=2e-11,
                              normal_limit=12.):
    """Enumerate six branches and integrate actual functions at fixed states.

    ``state_indices`` contains predeclared ``(path, exercise_date_index)``
    pairs on the independently simulated trace. Gaussian integration uses
    scalar actual European values and a separately evaluated premium
    polynomial. Its cap crossings and stock knots are explicit breakpoints.
    Conditional expectations are compared before any learned outer weights.
    The trace and all reported stock/factor states remain unchanged.
    """
    if not isinstance(policy, FourierStoppingPolicy):
        raise TypeError("conditional audit requires FourierStoppingPolicy")
    if (trace.get("policy_fingerprint") != fourier_policy_fingerprint(policy)
            or trace.get("seed") == policy.config.training_seed):
        raise ValueError("conditional audit requires a matching independent state trace")
    if (isinstance(absolute_quadrature_tolerance, bool)
            or not np.isfinite(absolute_quadrature_tolerance)
            or absolute_quadrature_tolerance <= 0 or not np.isfinite(normal_limit)
            or not 8 <= normal_limit <= 14):
        raise ValueError("positive finite quadrature tolerance and normal limit in [8,14] required")
    indices = list(state_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("declare nonempty distinct (path, exercise index) states")
    times, grid = np.asarray(trace["times"]), np.asarray(trace["actual_time_grid"])
    stock, factors = np.asarray(trace["stocks"]), np.asarray(trace["factors"])
    if stock.shape[-1] != 1:
        raise ValueError("conditional audit accepts a single inception stock")
    selected_features = None
    if martingale_weights is not None:
        from rvlab.pricing.rough_dual import RoughDualWeights, _predictable_features
        if (not isinstance(martingale_weights, RoughDualWeights)
                or martingale_weights.policy_fingerprint != fourier_policy_fingerprint(policy)
                or trace["seed"] in (martingale_weights.training_seed, martingale_weights.policy_training_seed)):
            raise ValueError("selected martingale must be frozen for this policy on independent streams")
        selected_features = _predictable_features(trace, martingale_weights.config.predictable_features)
    for pair in indices:
        if (len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in pair)
                or not 0 <= pair[0] < len(stock) or not 0 <= pair[1] < len(times)-1):
            raise ValueError("state indices must identify nonterminal left exercise states")
    before = _source_identity()
    params, market, kernel = policy.params, policy.market, policy.kernel
    nodes, weights = np.asarray(kernel.nodes), np.asarray(kernel.weights)
    count = len(nodes)
    if params.vol_of_vol <= 0:
        raise ValueError("six-branch audit requires positive vol_of_vol")
    matrix = -np.diag(nodes)-params.kappa*np.ones((count, 1))*weights
    affine = _AffineSurrogate(params, market, kernel, np.array([policy.strike]),
        np.array([policy.maturity]), np.array([policy.is_call]), grid, policy.control_config)
    branch_rows, component_rows, state_rows = [], [], []
    for state_id, (path, date) in enumerate(indices):
        left = float(times[date])
        position = int(np.searchsorted(grid, left))
        if position >= len(grid)-1 or grid[position] != left:
            raise ValueError("trace exercise date is not an exact integration-grid date")
        right = float(grid[position+1]); dt = right-left
        old = factors[path, date][None, :]
        left_stock = float(stock[path, date, 0])
        theta = _theta(params, .5*(left+right))
        flow = _drift_flow_with_integral(matrix,
            np.full(count, params.kappa*(theta-params.v0)), weights, params.v0, dt/2)
        growth = math.log(market.discount_curve.discount(left)/market.discount_curve.discount(right))
        law = _weak_transition_branches(old, params, nodes=nodes, weights=weights,
            flow=flow[:count, :count], shift=flow[:count, -1], integrated_flow=flow,
            dt=dt, theta=theta, growth=growth, stock_integral=policy.config.stock_integral)
        branches = law["new_factors"]
        clocks = law["orthogonal"][..., None]
        means = math.log(left_stock)+law["correlated"][:, :, None, None]-.5*clocks
        analytic_euro = _affine_values(affine, right, branches, means, clocks)[0, ..., 0]
        coefficient = _interpolate(policy, right, branches)[0]
        analytic_premium = expected_stopping_value(means[0, ..., 0], clocks[0, ..., 0],
            coefficient[:, None], strike=policy.strike, is_call=False, exercise=False,
            stock_knots=policy.config.stock_knots, value_cap=policy.premium_cap)
        analytic_intrinsic = _conditional_payoffs(means, clocks, policy.strike, policy.is_call)[0, ..., 0]
        signed, errors = np.zeros(3), np.zeros(3)
        discount = float(market.discount_curve.discount(right))
        for branch in range(3):
            factor = branches[0, branch]
            scalar_coeff = coefficient[branch]
            for order in range(2):
                mean, variance = float(means[0, branch, order, 0]), float(clocks[0, branch, order, 0])
                sigma = math.sqrt(variance)
                def actual_values(z):
                    logstock = mean+sigma*z
                    next_stock = math.exp(logstock)
                    euro = affine.values(right, factor[None, None, :],
                        np.full((1, 1, 1), logstock), np.zeros((1, 1, 1)), np.array([0])).item()
                    premium = _scalar_premium(next_stock, scalar_coeff, policy.strike,
                        policy.config.stock_knots, policy.premium_cap)
                    intrinsic = max((1 if policy.is_call else -1)*(next_stock-policy.strike), 0.)
                    return euro, premium, intrinsic
                if variance == 0:
                    references, quad_errors = np.asarray(actual_values(0.)), np.zeros(3)
                else:
                    points = _integration_points(mean, variance, policy.strike, scalar_coeff,
                        policy.config.stock_knots, policy.premium_cap, normal_limit)
                    threshold = (math.log(policy.strike)-mean)/sigma
                    if -normal_limit < threshold < normal_limit:
                        points = np.unique(np.r_[points, threshold])
                    references, quad_errors = [], []
                    for component in range(3):
                        values = [quad(lambda z: actual_values(z)[component]*
                            math.exp(-.5*z*z)/math.sqrt(2*math.pi), a, b,
                            epsabs=absolute_quadrature_tolerance, epsrel=absolute_quadrature_tolerance,
                            limit=100) for a, b in zip(points[:-1], points[1:])]
                        references.append(sum(value for value, error in values))
                        quad_errors.append(sum(error for value, error in values))
                    references, quad_errors = np.asarray(references), np.asarray(quad_errors)
                analytic = np.array([analytic_euro[branch, order], analytic_premium[branch, order], analytic_intrinsic[branch, order]])
                probability = float(law["probabilities"][0, branch])*.5
                signed += probability*(references-analytic)
                errors += probability*quad_errors
                for j, name in enumerate(("european", "premium", "intrinsic")):
                    branch_rows.append(dict(state_id=state_id, branch=branch, order=order,
                        component=name, probability=probability, analytic_center=float(analytic[j]),
                        scalar_integral=float(references[j]), difference=float(references[j]-analytic[j]),
                        quadrature_error_estimate=float(quad_errors[j]), variance=variance))
        left_discount = float(market.discount_curve.discount(left))
        left_q = _scalar_premium(left_stock,
            _continuation_coefficients(policy, left, old)[0], policy.strike,
            policy.continuations[0].knots, policy.premium_cap)
        margin = left_discount*(max((1 if policy.is_call else -1)*(left_stock-policy.strike), 0.)-left_q)
        margin -= float(trace["european_proxy"][path, date, 0])
        indicator = float(expit(margin/(left_discount*.05)))
        stopped = bool(date >= trace["stop_indices"][path, 0])
        e, p, i = signed*np.array([1., discount, discount])
        ee, ep, ei = errors*np.array([1., discount, discount])
        increments = {"european": (e, ee), "premium": (p, ep), "intrinsic": (i, ei),
            "micro_boundary": (indicator*(i-e), indicator*(ei+ee)),
            "micro_poststop": (indicator*stopped*(i-e-p), indicator*stopped*(ei+ee+ep))}
        if martingale_weights is not None:
            config = martingale_weights.config
            bin_index = int(np.searchsorted(martingale_weights.time_edges[1:], right, side="left"))
            coefficients = martingale_weights.weights.reshape(len(martingale_weights.time_edges)-1,
                1+len(config.predictable_features), len(config.components))[bin_index]
            multipliers = coefficients[0].copy()
            if selected_features is not None:
                multipliers += selected_features[path, date, 0]@coefficients[1:]
            means_and_errors = {**increments, "intrinsic_minus_european":(i-e, ei+ee)}
            weighted_mean = sum(w*means_and_errors[name][0] for w, name in zip(multipliers, config.components))
            weighted_error = sum(abs(w)*means_and_errors[name][1] for w, name in zip(multipliers, config.components))
            increments["selected_martingale"] = (weighted_mean, weighted_error)
        for name, (value, error) in increments.items():
            component_rows.append(dict(state_id=state_id, component=name,
                conditional_increment_mean=float(value), quadrature_error_estimate=float(error),
                left_boundary_multiplier=indicator, original_policy_stopped=stopped))
        state_rows.append(dict(state_id=state_id, path=int(path), exercise_date_index=int(date),
            left_time=left, right_time=right, left_stock=left_stock,
            original_policy_stopped=stopped, left_boundary_multiplier=indicator,
            **{f"factor_{j}":float(u) for j, u in enumerate(old[0])}))
    after = _source_identity()
    if before != after:
        raise RuntimeError("pricing/audit source changed during conditional centering check")
    return {"tables": {"branches":pd.DataFrame(branch_rows), "components":pd.DataFrame(component_rows),
            "states":pd.DataFrame(state_rows)},
        "manifest": {"policy_fingerprint":fourier_policy_fingerprint(policy),
            "state_trace_seed":int(trace["seed"]), "state_indices":[list(pair) for pair in indices],
            "absolute_quadrature_tolerance":absolute_quadrature_tolerance, "normal_limit":normal_limit,
            "selected_martingale_checked":martingale_weights is not None,
            "selected_martingale_components":None if martingale_weights is None else list(martingale_weights.config.components),
            "selected_martingale_weights":None if martingale_weights is None else martingale_weights.weights.tolist(),
            "selected_martingale_time_edges":None if martingale_weights is None else martingale_weights.time_edges.tolist(),
            "source_hashes":before, "source_unchanged":True, "valuation_used_for_selection":False,
            "scope":"Local six-branch/Gaussian conditional mean tests at predeclared independent LEFT states",
            "continuous_rough_heston_certified":False, "american_pricing_ready":False,
            "tolerances_changed":False}}


__all__ = ["audit_conditional_centers"]
