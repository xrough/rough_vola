"""Spot-based vanilla prices and stock deltas with explicit exercise style."""
from __future__ import annotations

from types import MappingProxyType
from dataclasses import replace
import hashlib
import time

import numpy as np
import pandas as pd

from .types import BlackScholesParams, HestonParams, RoughHestonParams, PricingError
from .variance_curve import HestonCurveParams, RoughHestonCurveParams
from .vanilla_types import (SpotMarket, FDConfig, FrozenFDGrid, VanillaResult, UnsupportedModelError, _real)
from ._vanilla_common import build_grid
from ._vanilla_bs import solve_bs


def _contracts(strike, maturity, is_call):
    raw = np.asarray(is_call, dtype=object)
    if any(not isinstance(v, (bool, np.bool_)) for v in raw.flat):
        raise ValueError("is_call must contain booleans")
    for name, arg in (("strike", strike), ("maturity", maturity)):
        if any(isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in np.asarray(arg, object).flat):
            raise ValueError(f"{name} must be real numeric values")
    try:
        k, t, calls = np.broadcast_arrays(np.asarray(strike, float), np.asarray(maturity, float), np.asarray(is_call, bool))
    except (TypeError, ValueError) as exc:
        raise ValueError("strike, maturity and is_call must broadcast") from exc
    if not k.size or not np.isfinite(k).all() or (k <= 0).any():
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("maturity must be finite and nonnegative")
    return k, t, calls


def _deterministic(params, market, strikes, maturity, calls, style):
    events = [e for e in market.dividends.events if e.time <= maturity]
    times = sorted(set([0., maturity, *[v for v in market.discount_curve.times if v < maturity], *[e.time for e in events]]))
    cash = {e.time: e.amount for e in events}
    def prices(initial_spot, start=0., *, before_event=True):
        spot, previous = float(initial_spot), start
        candidates = []
        for t in times:
            if t < start:
                continue
            spot *= market.discount_curve.discount(previous)/market.discount_curve.discount(t)
            discounted = market.discount_curve.discount(t)/market.discount_curve.discount(start)
            if style == "american":
                candidates.append(discounted*np.maximum(np.where(calls, 1., -1.)*(spot-strikes), 0.))
            if t in cash and (t > start or before_event):
                spot = max(spot-cash[t], 0.)
            if style == "american" or t == maturity:
                candidates.append(discounted*np.maximum(np.where(calls, 1., -1.)*(spot-strikes), 0.))
            previous = t
        return np.max(candidates, axis=0)
    def sensitivities(spot, start=0., *, before_event=True):
        value = prices(spot, start, before_event=before_event)
        h = min(.25*spot, max(1e-7, 1e-6*spot)) if spot > 0 else 1e-7
        right = (prices(spot+h, start, before_event=before_event)-value)/h
        left = ((value-prices(spot-h, start, before_event=before_event))/h
                if spot > 0 else right)
        return value, .5*(left+right), left, right
    values = sensitivities(market.spot)
    event_sensitivities = []
    for event in events:
        pre = sensitivities(market.spot, event.time)
        post_spot = max(market.spot-event.amount, 0.)
        post = sensitivities(post_spot, event.time, before_event=False)
        event_sensitivities.append({"time": event.time, "cash_dividend": event.amount,
            "reference_spot": market.spot, "post_spot": post_spot,
            "pre_delta": pre[1].tolist(), "post_delta": post[1].tolist(),
            "pre_delta_left": pre[2].tolist(), "pre_delta_right": pre[3].tolist(),
            "post_delta_left": post[2].tolist(), "post_delta_right": post[3].tolist()})
    return values, event_sensitivities


def _run(params, market, k, t, calls, config, grid, style):
    fields = {key: np.empty(len(k)) for key in ("prices", "deltas", "delta_left", "delta_right")}
    boundaries, solvers, events = [], [], []
    for maturity in np.unique(t):
        mask = t == maturity
        if isinstance(params, BlackScholesParams):
            deterministic = params.volatility == 0
        else:
            # Future immigration after this contract expires cannot activate
            # its otherwise absorbing variance state. A knot at expiry has
            # zero duration and is likewise irrelevant to the diffusion.
            levels = (params.curve.levels[:1+int(np.searchsorted(params.curve.knot_times, maturity, side="left"))]
                      if hasattr(params, "curve") else (params.theta,))
            deterministic = params.v0 == 0 and params.kappa*max(levels) == 0
        if maturity == 0 or deterministic:
            values, event_sensitivities = _deterministic(params, market, k[mask], float(maturity), calls[mask], style)
            result = dict(zip(fields, values))
            result.update(exercise_boundaries=[], diagnostics={"backend": "deterministic_optimal_stopping",
                "linear_solves": 0, "dividend_event_sensitivities": event_sensitivities})
        elif isinstance(params, BlackScholesParams) or params.vol_of_vol == 0:
            result = solve_bs(params, market, k[mask], float(maturity), calls[mask], config, grid, exercise_style=style)
        else:
            from ._vanilla_heston import solve_heston
            result = solve_heston(params, market, k[mask], float(maturity), calls[mask], config, grid, exercise_style=style)
        for name in fields:
            array = np.asarray(result[name], float)
            if array.shape != k[mask].shape or not np.isfinite(array).all():
                raise PricingError(f"{style} vanilla backend returned invalid {name}")
            fields[name][mask] = array
        indices = np.flatnonzero(mask)
        for record in result["exercise_boundaries"]:
            record = {"maturity": float(maturity), **record}
            if "contract_index" in record:
                record["contract_index"] = int(indices[record["contract_index"]])
            boundaries.append(record)
        solvers.append({"maturity": float(maturity), **result["diagnostics"]})
        events.extend({"maturity": float(maturity), "contract_indices": indices.tolist(), **event}
                      for event in result["diagnostics"].get("dividend_event_sensitivities", []))
    return fields, {"solvers": solvers, "linear_solves": sum(row.get("linear_solves", 0) for row in solvers),
                    "dividend_event_sensitivities": events}, boundaries


def _expanded_grid(grid, config):
    def extend(a, maximum):
        if maximum <= a[-1]:
            return a
        n = max(1, int(np.ceil((maximum-a[-1])/(a[-1]-a[-2]))))
        return np.r_[a, np.linspace(a[-1], maximum, n+1)[1:]]
    return FrozenFDGrid(extend(grid.spot, 2*grid.spot[-1]),
                        None if grid.variance is None else extend(grid.variance, 2*grid.variance[-1]))


def evaluate_vanilla(params, *, market, strike, maturity, is_call=True,
                     exercise_style, config=None, check_refinement=False, grid=None,
                     price_atol=.001, price_rtol=1e-4,
                     delta_atol=.002, delta_rtol=.005,
                     kernel=None, refinement_kernel=None, conditional_state=None):
    """Price European/American vanilla contracts and compute *stock* deltas.

    Parameters, discount factors and expected cash events are held fixed in
    the stock sensitivity. ``delta_left/right`` are local spatial one-sided
    derivatives; dividend-event diagnostics separately report pre/post-event
    sensitivities at a declared reference spot (not a forecast stock path).
    Rough Heston requires an explicit finite ``kernel`` and returns an
    independent-training primal/dual simulation estimate. Its optional checks
    additionally require a caller-declared ``refinement_kernel``; omission
    blocks readiness. Conditional historical rough states remain unsupported.
    Finite-difference refinements test space, time, variance, domain and a joint grid
    independently. Missing/failed probes block stability without erasing prices.
    """
    started = time.perf_counter()
    if exercise_style not in {"european", "american"}:
        raise ValueError("exercise_style must be explicitly european or american")
    if not isinstance(check_refinement, (bool, np.bool_)):
        raise ValueError("check_refinement must be a boolean")
    for name, value in (("price_atol", price_atol), ("price_rtol", price_rtol), ("delta_atol", delta_atol), ("delta_rtol", delta_rtol)):
        _real(value, name, nonnegative=True)
    if isinstance(params, (RoughHestonParams, RoughHestonCurveParams)):
        if grid is not None:
            raise ValueError("rough Heston uses an explicit kernel, not a finite-difference grid")
        return _evaluate_rough(params, market, strike, maturity, is_call,
            exercise_style, config, check_refinement, kernel, refinement_kernel,
            conditional_state, price_atol, price_rtol, delta_atol, delta_rtol)
    if kernel is not None or refinement_kernel is not None or conditional_state is not None:
        raise ValueError("kernel, refinement_kernel and conditional_state apply only to rough Heston")
    if not isinstance(params, (BlackScholesParams, HestonParams, HestonCurveParams)):
        raise TypeError("unsupported vanilla parameter object")
    if not isinstance(market, SpotMarket):
        raise TypeError("market must be SpotMarket with an explicit dividend schedule")
    if config is None:
        config = FDConfig(spot_steps=400, time_steps=400) if isinstance(params, BlackScholesParams) else FDConfig()
    if not isinstance(config, FDConfig):
        raise TypeError("config must be FDConfig")
    k, t, calls = _contracts(strike, maturity, is_call)
    shape = k.shape
    k, t, calls = k.ravel(), t.ravel(), calls.ravel()
    horizon = float(np.max(t))
    market.discount_curve.discount(horizon)
    if market.dividends.coverage + 1e-12 < horizon:
        raise ValueError("dividend expectation coverage ends before option maturity")
    stochastic = not isinstance(params, BlackScholesParams) and params.vol_of_vol > 0
    if grid is None:
        grid = build_grid(market, k, config, variance=stochastic)
    if not isinstance(grid, FrozenFDGrid) or grid.spot[-1] <= max(market.spot, float(k.max())):
        raise ValueError("grid must cover spot and strikes with an upper boundary beyond them")
    if stochastic and (grid.variance is None or params.v0 >= grid.variance[-1]):
        raise ValueError("variance grid must extend above v0")
    fields, diagnostics, boundaries = _run(params, market, k, t, calls, config, grid, exercise_style)
    base = fields
    checks, price_stable, delta_stable = [], None, None
    price_changes = {}
    references = {key: np.full_like(k, np.nan) for key in fields}
    pricing_evaluations = 1
    if check_refinement:
        axes = ["spot", "time", *(["variance"] if stochastic else []), "domain", "joint"]
        for axis in axes:
            settings = config.refined(axis)
            if axis == "time":
                refined_grid = grid
            elif axis == "domain":
                refined_grid = _expanded_grid(grid, config)
            else:
                fresh = build_grid(market, k, settings, variance=stochastic)
                refined_grid = FrozenFDGrid(grid.spot if axis == "variance" else fresh.spot,
                    grid.variance if axis == "spot" else fresh.variance)
            pricing_evaluations += 1
            try:
                probe, detail, _ = _run(params, market, k, t, calls, settings, refined_grid, exercise_style)
                dp, dd = np.abs(probe["prices"]-base["prices"]), np.abs(probe["deltas"]-base["deltas"])
                price_changes[axis] = dp.copy()
                ps = bool(np.all(dp <= price_atol+price_rtol*np.abs(probe["prices"])))
                ds = bool(np.all(dd <= delta_atol+delta_rtol*np.abs(probe["deltas"])))
                checks.append({"axis": axis, "status": "ok", "price_stable": ps, "delta_stable": ds,
                               "max_price_change": float(dp.max()), "max_delta_change": float(dd.max()), "error": None})
                diagnostics["linear_solves"] += detail["linear_solves"]
                if axis == "joint":
                    references = probe
            except (PricingError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                checks.append({"axis": axis, "status": "failed", "price_stable": False, "delta_stable": False,
                               "max_price_change": float("inf"), "max_delta_change": float("inf"), "error": str(exc)})
        price_stable = all(check["price_stable"] for check in checks)
        delta_stable = all(check["delta_stable"] for check in checks)
    def reshape(a):
        a = np.asarray(a).reshape(shape)
        return float(a) if shape == () else a
    diagnostics.update(checks=pd.DataFrame(checks), pricing_evaluations=pricing_evaluations,
        price_changes=price_changes,
        reference_prices=reshape(references["prices"]), reference_deltas=reshape(references["deltas"]),
        elapsed_seconds=time.perf_counter()-started,
        grid_sha256=hashlib.sha256(grid.spot.tobytes()+(grid.variance.tobytes() if grid.variance is not None else b"")).hexdigest(),
        dividend_convention="liquidator: S_after=max(S_before-cash,0); payout=min(cash,S_before)")
    return VanillaResult(**{key: reshape(value) for key, value in fields.items()},
        exercise_style=exercise_style, config=config, diagnostics=MappingProxyType(diagnostics),
        exercise_boundaries=boundaries, price_stable=price_stable, delta_stable=delta_stable, grid=grid)


def _evaluate_rough(params, market, strike, maturity, is_call, style, config,
                    check_refinement, kernel, refinement_kernel, conditional_state,
                    price_atol, price_rtol, delta_atol, delta_rtol):
    """Keep stopping, sampling and numerical approximation evidence distinct."""
    from .rough_lift import ExponentialKernel
    from .rough_stopping import RoughAmericanConfig, estimate_rough_american
    if not isinstance(kernel, ExponentialKernel):
        raise ValueError("rough Heston requires an explicit ExponentialKernel; validate its approximation separately")
    if refinement_kernel is not None and not isinstance(refinement_kernel, ExponentialKernel):
        raise TypeError("refinement_kernel must be ExponentialKernel")
    if (refinement_kernel is not None and refinement_kernel.nodes == kernel.nodes
            and refinement_kernel.weights == kernel.weights):
        raise ValueError("refinement_kernel coefficients must differ from the pricing kernel")
    exact_kernel = params.hurst == .5 and kernel.nodes == (0.,) and kernel.weights == (1.,)
    config = RoughAmericanConfig() if config is None else config
    started = time.perf_counter()
    options = dict(strike=strike, maturity=maturity, is_call=is_call,
                   exercise_style=style, conditional_state=conditional_state)
    base = estimate_rough_american(params, market, kernel, config=config, **options)
    _validate_rough_estimate(base)
    diagnostics = dict(base.diagnostics)
    checks, changes = [], {}
    evaluations = 1
    pbudget = price_atol+price_rtol*np.abs(base.prices)
    dbudget = delta_atol+delta_rtol*np.abs(base.deltas)
    # The paired dual gap controls exercise-policy error for the discrete
    # model. Its CI and the primal sampling CI share the original price budget.
    price_uncertainty = (np.maximum(base.dual_gaps, 0.)
        +1.96*(base.dual_gap_standard_errors+base.standard_errors))
    delta_uncertainty = 1.96*base.delta_standard_errors
    nonsmooth = (np.asarray(maturity) == 0) & (np.asarray(strike) == market.spot)
    nonsmooth = np.broadcast_to(nonsmooth, np.asarray(base.deltas).shape).copy()
    if check_refinement:
        axes = ["time", "exercise", "sampling", "training", "bump", "kernel", "joint"]
        for axis in axes:
            try:
                if axis in ("kernel", "joint") and refinement_kernel is None and not exact_kernel:
                    raise ValueError("independent refinement_kernel was not supplied")
                if axis == "kernel" and exact_kernel and refinement_kernel is None:
                    checks.append(dict(axis=axis, status="exact_limit", price_stable=True,
                        delta_stable=True, max_price_change=0., max_delta_change=0., error=None))
                    continue
                settings = config.refined(axis) if axis != "kernel" else config
                if axis in ("training", "joint"):
                    # A new training stream probes policy construction, not
                    # merely another prefix of the original regression sample.
                    seed = settings.training_seed+104729
                    if seed == settings.valuation_seed:
                        seed += 1
                    settings = replace(settings, training_seed=seed, retrain_bumps=True)
                probe_kernel = (refinement_kernel if axis in ("kernel", "joint")
                    and refinement_kernel is not None else kernel)
                evaluations += 1
                probe = estimate_rough_american(params, market, probe_kernel,
                    config=settings, **options)
                _validate_rough_estimate(probe)
                dp = np.abs(probe.prices-base.prices)
                dd = np.abs(probe.deltas-base.deltas)
                # A sum of endpoint SEs is conservative without assuming
                # independence between common-random-number refinements.
                pc = dp+1.96*(probe.standard_errors+base.standard_errors)
                dc = dd+1.96*(probe.delta_standard_errors+base.delta_standard_errors)
                policy_delta = np.abs(probe.retrained_deltas-probe.deltas)
                if settings.retrain_bumps:
                    dc = np.maximum(dc, policy_delta+1.96*(
                        probe.retrained_delta_standard_errors+probe.delta_standard_errors))
                gap = np.maximum(probe.dual_gaps, 0.)+1.96*(
                    probe.dual_gap_standard_errors+probe.standard_errors)
                ps = bool(np.all(pc <= pbudget) and np.all(gap <= pbudget))
                if axis == "bump":
                    old_gap = np.abs(base.delta_right-base.delta_left)
                    new_gap = np.abs(probe.delta_right-probe.delta_left)
                    # A persistent one-sided slope discontinuity is not a
                    # smooth delta, even if symmetric bumps agree perfectly.
                    nonsmooth |= (new_gap > 2*dbudget) & (new_gap >= .9*old_gap)
                ds = bool(np.all(dc <= dbudget))
                changes[axis] = dp.copy()
                checks.append(dict(axis=axis, status="ok", price_stable=ps,
                    delta_stable=ds, max_price_change=float(dp.max()),
                    max_delta_change=float(dd.max()), error=None))
            except (PricingError, ValueError, FloatingPointError, MemoryError,
                    np.linalg.LinAlgError) as exc:
                checks.append(dict(axis=axis, status="failed", price_stable=False,
                    delta_stable=False, max_price_change=float("inf"),
                    max_delta_change=float("inf"), error=str(exc)))
    price_stable = (bool(np.all(price_uncertainty <= pbudget))
        and all(row["price_stable"] for row in checks)) if check_refinement else None
    delta_stable = (bool(np.all(delta_uncertainty <= dbudget)) and not np.any(nonsmooth)
        and all(row["delta_stable"] for row in checks)) if check_refinement else None
    diagnostics.update(checks=pd.DataFrame(checks), price_changes=changes,
        pricing_evaluations=evaluations, price_stable=price_stable,
        delta_stable=delta_stable, american_pricing_ready=style == "american" and bool(price_stable),
        stock_delta_ready=bool(price_stable and delta_stable),
        nonsmooth_stock_sensitivity=nonsmooth,
        price_absolute_tolerance=price_atol, price_relative_tolerance=price_rtol,
        delta_absolute_tolerance=delta_atol, delta_relative_tolerance=delta_rtol,
        refinement_kernel_fingerprint=(None if refinement_kernel is None else refinement_kernel.fingerprint),
        price_uncertainty_allowance=price_uncertainty,
        delta_sampling_allowance=delta_uncertainty,
        readiness_reason="explicit numerical checks passed" if price_stable and delta_stable
            else "sampling, stopping, time, exercise, kernel and policy/delta checks are incomplete or exceed tolerance",
        elapsed_seconds=time.perf_counter()-started)
    return replace(base, diagnostics=diagnostics)


def _validate_rough_estimate(estimate):
    """Invalid bounds or sampling errors must never disappear in diagnostics."""
    fields = ("prices", "standard_errors", "dual_prices", "dual_standard_errors",
        "dual_gaps", "dual_gap_standard_errors", "deltas", "delta_standard_errors",
        "one_sided_deltas", "one_sided_delta_standard_errors", "bump_deltas",
        "bump_delta_standard_errors")
    if any(not np.isfinite(getattr(estimate, name)).all() for name in fields):
        raise PricingError("rough stopping returned nonfinite price/delta diagnostics")
    for name in fields:
        if "standard_errors" in name and np.any(getattr(estimate, name) < 0):
            raise PricingError("rough stopping returned negative standard errors")
    tolerance = 1e-10*(1+np.abs(estimate.prices))
    if np.any(estimate.dual_gaps < -tolerance) or not np.allclose(
            estimate.dual_prices-estimate.prices, estimate.dual_gaps, rtol=1e-8, atol=1e-10):
        raise PricingError("rough stopping returned inconsistent primal/dual gaps")
    if estimate.config.retrain_bumps and (not np.isfinite(estimate.retrained_deltas).all()
            or not np.isfinite(estimate.retrained_delta_standard_errors).all()
            or np.any(estimate.retrained_delta_standard_errors < 0)):
        raise PricingError("rough stopping returned invalid retrained stock-delta diagnostics")


__all__ = ["evaluate_vanilla"]
