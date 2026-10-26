"""Training-only input stresses and a deliberately flexible smile benchmark.

These diagnostics do not construct independent stock observations, alter the
strict market-data pipeline, or provide a dynamic model for hedging. Forward
experiments keep model parameters and contract membership fixed. The quadratic
benchmark fits each maturity separately and is only an interpolation comparator.
"""

from __future__ import annotations

from dataclasses import asdict
from numbers import Integral
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr

from rvlab.pricing import (
    BlackScholesParams, HestonParams, PricingConfig, PricingError,
    RoughHestonParams, price,
)
from rvlab.pricing._validation import broadcast_inputs
from rvlab.pricing.fourier import price_from_cf
from rvlab.pricing.heston import heston_cf
from rvlab.pricing.rough_heston import rough_heston_cf
from ._pricing_data import implied_volatility
from ._pricing_fits import _observation_arrays


def _inputs(quotes):
    if "exercise_style" in quotes:
        bad = quotes.exercise_style.astype(str).str.lower().ne("european").to_numpy()
        if bad.any():
            raise ValueError("European synthetic benchmarks reject American observations at row positions "
                             f"{np.flatnonzero(bad).tolist()}; use American calibration")
    target, scale, training = _observation_arrays(quotes)
    f, d, k, t, calls = broadcast_inputs(**{
        name: quotes[name].to_numpy() for name in
        ("forward", "discount", "strike", "maturity", "is_call")})
    if np.any(t <= 0):
        raise ValueError("benchmark maturities must be strictly positive")
    if "contract_id" in quotes and quotes.contract_id.duplicated().any():
        raise ValueError("benchmark contract_id values must be unique")
    keys = (quotes.expiry_date.astype(str).to_numpy() if "expiry_date" in quotes
            else np.array([format(x, ".17g") for x in t]))
    for key in np.unique(keys):
        rows = keys == key
        for name, values in (("forward", f), ("discount", d), ("maturity", t)):
            if not np.all(values[rows] == values[rows][0]):
                raise ValueError(f"{key}: {name} must be constant within an expiry")
    if "snapshot_date" in quotes and quotes.snapshot_date.nunique() != 1:
        raise ValueError("benchmark quotes must contain one snapshot date")
    return f, d, k, t, calls, target, scale, training, keys


def _budget(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _errors(values, target, scale, training):
    error = values - target
    out = {}
    for name, mask in (("train", training), ("test", ~training)):
        valid = np.any(mask) and np.all(np.isfinite(error[mask]))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask]**2))) if valid else None
        out[f"{name}_scaled_rmse"] = (
            float(np.sqrt(np.mean((error[mask] / scale[mask])**2))) if valid else None)
        out[f"{name}_inside_scale"] = int(np.sum(np.abs(error[mask]) <= scale[mask])) if valid else None
    out["training_objective"] = (
        float(np.mean((error[training] / scale[training])**2))
        if np.all(np.isfinite(error[training])) else None)
    return out


def _prediction(quotes, values, target, scale, *, scenario, forward=None, discount=None):
    out = quotes.copy(deep=True)
    if forward is not None:
        out["forward"] = forward
    if discount is not None:
        out["discount"] = discount
    out["scenario"] = scenario
    out["target_price"] = target
    out["price_scale"] = scale
    out["model_price"] = values
    out["price_residual"] = values - target
    out["residual"] = (values - target) / scale
    return out


def forward_sensitivity(params, quotes, anchors, config, *, max_nfev=100):
    """Hold model parameters fixed while stressing explicit forward inputs.

    Fit one additive forward shift per expiry on training prices only, first
    within anchor minima/maxima and then within a declared +/- $5 stress box.
    Anchor extrema are sensitivity ranges, never confidence intervals. Rate
    stresses shift continuously compounded yields by +/- 100 bp and recompute
    each forward from the *same* anchor pairs. No contracts are reselected.

    Return ``summary``, ``predictions`` and ``shifts`` DataFrames plus JSON-safe
    ``metadata``. Fourier CF values are cached within this fixed-parameter call,
    so optimizer trials over forwards do not repeatedly solve the Riccati ODE.
    These diagnostics do not independently establish numerical readiness.
    """
    started = time.perf_counter()
    _budget(max_nfev, "max_nfev")
    if not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig")
    if not isinstance(params, (BlackScholesParams, HestonParams, RoughHestonParams)):
        raise TypeError("params must be a supported pricing parameter object")
    f, d, k, t, calls, target, scale, train, keys = _inputs(quotes)
    if "expiry_date" not in quotes:
        raise ValueError("forward stresses require expiry_date")
    required = {"expiry_date", "strike", "forward", "market_price_call", "market_price_put"}
    if not isinstance(anchors, pd.DataFrame) or required - set(anchors):
        raise ValueError(f"anchors require {sorted(required)}")
    anchor_rows = anchors.copy(deep=True)
    if "snapshot_date" in quotes and "snapshot_date" in anchor_rows:
        anchor_rows = anchor_rows.loc[
            anchor_rows.snapshot_date == quotes.snapshot_date.iloc[0]].copy()
    groups = sorted(np.unique(keys))
    group_index = np.array([groups.index(key) for key in keys])
    lower, upper, paired = [], [], {}
    for group in groups:
        a = anchor_rows.loc[anchor_rows.expiry_date.astype(str) == group].copy()
        if a.empty:
            raise ValueError(f"{group}: no forward anchors")
        numeric = a[["strike", "forward", "market_price_call", "market_price_put"]].to_numpy(float)
        if not np.all(np.isfinite(numeric)) or np.any(numeric[:, :2] <= 0) or np.any(numeric[:, 2:] < 0):
            raise ValueError(f"{group}: anchor values must be finite with positive strikes/forwards and nonnegative prices")
        if a.strike.duplicated().any():
            raise ValueError(f"{group}: duplicate anchor strikes")
        if not np.any(train & (keys == group)):
            raise ValueError(f"{group}: a training contract is required to fit a forward shift")
        baseline_forward = f[keys == group][0]
        lower.append(float(a.forward.min() - baseline_forward))
        upper.append(float(a.forward.max() - baseline_forward))
        paired[group] = a
    lower, upper = np.asarray(lower), np.asarray(upper)
    # A zero-width anchor box is a fixed shift rather than an invalid SciPy box.
    cache = {}
    price_calls = 0
    cf_calls = 0

    def cached_cf(u, tenor):
        nonlocal cf_calls
        key = (float(tenor), u.shape, u.tobytes())
        if key not in cache:
            cf_calls += 1
            cache[key] = (rough_heston_cf(params, u, tenor, config=config)
                          if isinstance(params, RoughHestonParams)
                          else heston_cf(params, u, tenor))
        return cache[key]

    def evaluate(forward, discount):
        nonlocal price_calls
        price_calls += 1
        kwargs = dict(forward=forward, discount=discount, strike=k, maturity=t, is_call=calls)
        values = (price(params, **kwargs, config=config) if isinstance(params, BlackScholesParams)
                  else price_from_cf(cached_cf, reference_variance=params.v0,
                                     **kwargs, config=config))
        return np.asarray(values, dtype=float)

    summaries, predictions, shifts = [], [], []
    scenarios = ("baseline", "anchor_min", "anchor_max", "forward_fit_anchor_box",
                 "forward_fit_5dollar_box", "rate_minus_100bp", "rate_plus_100bp")
    for scenario in scenarios:
        stamp = time.perf_counter()
        initial_calls = price_calls
        forward, discount = f.copy(), d.copy()
        optimizer_success, status_code, message, nfev = None, None, None, None
        shift = np.zeros(len(groups))
        error = None
        status = "priced"
        lo = hi = None
        try:
            if scenario in ("anchor_min", "anchor_max"):
                shift = lower.copy() if scenario == "anchor_min" else upper.copy()
                forward += shift[group_index]
            elif scenario.startswith("forward_fit"):
                lo, hi = ((lower, upper) if scenario == "forward_fit_anchor_box"
                          else (np.maximum(-5., -np.array([f[keys == x][0] for x in groups]) + 1e-8),
                                np.full(len(groups), 5.)))
                free = lo < hi
                shift = .5 * (lo + hi)
                shift[free] = np.clip(0., lo[free], hi[free])
                if np.any(free):
                    def residual(z):
                        delta = shift.copy()
                        delta[free] = z
                        return (evaluate(f + delta[group_index], d) - target)[train] / scale[train]
                    opt = least_squares(residual, shift[free], bounds=(lo[free], hi[free]),
                        max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-8)
                    shift[free] = opt.x
                    optimizer_success, status_code = bool(opt.success), int(opt.status)
                    message, nfev = str(opt.message), int(opt.nfev)
                else:
                    optimizer_success, status_code, message, nfev = True, 1, "All forward shifts fixed by anchor ranges", 0
                status = "converged" if optimizer_success else "exhausted"
                forward += shift[group_index]
            elif scenario.startswith("rate_"):
                rate_shift = -.01 if scenario == "rate_minus_100bp" else .01
                discount *= np.exp(-rate_shift * t)
                for i, group in enumerate(groups):
                    mask = keys == group
                    a = paired[group]
                    new_forward = np.median(a.strike +
                        (a.market_price_call - a.market_price_put) / discount[mask][0])
                    forward[mask] = new_forward
                    shift[i] = new_forward - f[mask][0]
            values = evaluate(forward, discount)
        except PricingError as exc:
            status, error = "numerical_failure", str(exc)
            values = np.full(len(quotes), np.nan)
        summaries.append({"scenario": scenario, "status": status,
            "optimizer_success": optimizer_success, "optimizer_status": status_code,
            "optimizer_message": message, "nfev": nfev, "error": error,
            "elapsed_seconds": time.perf_counter() - stamp,
            "pricing_evaluations": price_calls - initial_calls,
            "config": asdict(config), "diagnostic_only": True,
            **_errors(values, target, scale, train)})
        predictions.append(_prediction(quotes, values, target, scale,
                                       scenario=scenario, forward=forward, discount=discount))
        for i, group in enumerate(groups):
            on_bound = None
            if lo is not None:
                on_bound = ("fixed" if lo[i] == hi[i] else
                    "lower" if np.isclose(shift[i], lo[i], atol=1e-7, rtol=0.) else
                    "upper" if np.isclose(shift[i], hi[i], atol=1e-7, rtol=0.) else None)
            shifts.append({"scenario": scenario, "expiry_date": group,
                "baseline_forward": float(f[keys == group][0]),
                "forward_shift": float(shift[i]), "forward": float(forward[keys == group][0]),
                "discount": float(discount[keys == group][0]),
                "anchor_shift_min": float(lower[i]), "anchor_shift_max": float(upper[i]),
                "active_bound": on_bound, "status": status})
    return {"summary": pd.DataFrame(summaries), "summary_records": summaries,
        "predictions": pd.concat(predictions),
        "shifts": pd.DataFrame(shifts), "metadata": {
            "parameters": asdict(params), "config": asdict(config), "max_nfev": int(max_nfev),
            "elapsed_seconds": time.perf_counter() - started,
            "pricing_evaluations": price_calls, "characteristic_function_evaluations": cf_calls,
            "selection": "training residuals only; fixed contracts and parameters",
            "anchor_ranges_are_confidence_intervals": False,
            "independent_stock_observation": False, "eligible_for_hedging": False,
            "numerical_readiness_independently_checked": False,
            "rate_shift_convention": "continuous yield +/- .01; same anchor IDs and median parity construction"}}


def _black_total_variance(forward, discount, strike, total_variance, is_call):
    stddev = np.sqrt(total_variance)
    z = np.log(forward / strike) / stddev
    d1, d2 = z + .5 * stddev, z - .5 * stddev
    otm_call = discount * (forward * ndtr(d1) - strike * ndtr(d2))
    otm_put = discount * (strike * ndtr(-d2) - forward * ndtr(-d1))
    parity = discount * (forward - strike)
    call = np.where(forward <= strike, otm_call, otm_put + parity)
    put = np.where(forward >= strike, otm_put, otm_call - parity)
    return np.where(is_call, call, put)


def _bernstein(x):
    return np.column_stack(((1. - x)**2, 2. * x * (1. - x), x**2))


def _shape_checks(coefficients, log_min, log_max, forward, discount, observed_strikes, expiry, grid_size):
    rows = []
    for label, strikes in (("observed_strikes", np.unique(observed_strikes)),
                           ("dense_grid", np.linspace(min(observed_strikes), max(observed_strikes), grid_size))):
        strikes = np.sort(strikes)
        x = strikes / forward
        u = (np.log(x) - log_min) / (log_max - log_min)
        w = _bernstein(u) @ coefficients
        call = _black_total_variance(forward, discount, strikes, w, True) / (discount * forward)
        slopes = np.diff(call) / np.diff(x)
        convexity = np.diff(slopes) / (.5 * (np.diff(x)[:-1] + np.diff(x)[1:]))
        minimum_convexity = float(convexity.min()) if len(convexity) else None
        min_slope = float(slopes.min()) if len(slopes) else None
        max_slope = float(slopes.max()) if len(slopes) else None
        lower_margin = float(np.min(call - np.maximum(1. - x, 0.)))
        upper_excess = float(np.max(call - 1.))
        passed = (minimum_convexity is not None and minimum_convexity >= -1e-8
                  and min_slope >= -1. - 1e-8 and max_slope <= 1e-8
                  and lower_margin >= -1e-8 and upper_excess <= 1e-8)
        rows.append({"expiry_date": expiry, "grid": label, "points": len(strikes),
            "min_total_variance": float(w.min()), "min_normalized_call_convexity": minimum_convexity,
            "min_normalized_call_slope": min_slope, "max_normalized_call_slope": max_slope,
            "min_normalized_call_lower_margin": lower_margin,
            "max_normalized_call_upper_excess": upper_excess, "passes_local_checks": bool(passed)})
    return rows


def quadratic_smile_benchmark(quotes, *, max_nfev=200, grid_size=201):
    """Fit positive quadratic total variance separately for each expiry.

    ``w(log(K/F)) = a + b*log(K/F) + c*log(K/F)**2`` uses three positive
    Bernstein coefficients on the range of known contract strikes. This
    ensures positive variance over that entire interval, including a dense
    check grid. Two deterministic starts use training IVs only; their training
    price objectives select the illustrative fit. The objective remains the
    original dollar residual divided by the unmodified half-spread/floor.

    This is a flexible static interpolation benchmark, not a stochastic model,
    a conditional-state estimate, or a global arbitrage-free surface. Local
    call monotonicity/convexity checks cover observed strikes and a dense grid;
    extrapolation, calendar arbitrage and dynamic consistency are not certified.
    """
    started = time.perf_counter()
    _budget(max_nfev, "max_nfev")
    _budget(grid_size, "grid_size")
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    f, d, k, t, calls, target, scale, train, keys = _inputs(quotes)
    values = np.full(len(quotes), np.nan)
    parameters, checks, attempts = [], [], []
    for group in sorted(np.unique(keys)):
        stamp = time.perf_counter()
        mask, fitting = keys == group, train & (keys == group)
        log_k = np.log(k[mask] / f[mask])
        low, high = float(log_k.min()), float(log_k.max())
        if np.unique(k[fitting]).size < 3 or high <= low:
            parameters.append({"expiry_date": group, "status": "insufficient_training",
                "error": "three distinct training strikes are required", "elapsed_seconds": time.perf_counter() - stamp})
            continue
        u = (log_k - low) / (high - low)
        design = _bernstein(u)
        local_train = train[mask]
        iv = np.asarray(implied_volatility(target[fitting], forward=f[fitting], discount=d[fitting],
            strike=k[fitting], maturity=t[fitting], is_call=calls[fitting]))
        train_w = iv**2 * t[fitting]
        train_u = (np.log(k[fitting] / f[fitting]) - low) / (high - low)
        fit_coefficients = np.linalg.lstsq(_bernstein(train_u), train_w, rcond=None)[0]
        starts = (np.clip(fit_coefficients, 1e-10, 4.),
                  np.full(3, np.clip(np.median(train_w), 1e-10, 4.)))
        fitted = []
        for i, initial in enumerate(starts):
            trial_started = time.perf_counter()
            def residual(z):
                prices = _black_total_variance(f[mask], d[mask], k[mask], design @ z, calls[mask])
                return (prices[local_train] - target[mask][local_train]) / scale[mask][local_train]
            try:
                opt = least_squares(residual, initial, bounds=(np.full(3, 1e-10), np.full(3, 4.)),
                    max_nfev=max_nfev, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-8)
                objective = float(np.mean(residual(opt.x)**2))
                fitted.append((objective, i, opt))
                attempts.append({"expiry_date": group, "start": i, "training_objective": objective,
                    "optimizer_success": bool(opt.success), "optimizer_status": int(opt.status),
                    "optimizer_message": str(opt.message), "nfev": int(opt.nfev),
                    "status": "converged" if opt.success else "exhausted", "error": None,
                    "elapsed_seconds": time.perf_counter() - trial_started})
            except PricingError as exc:
                attempts.append({"expiry_date": group, "start": i, "training_objective": None,
                    "optimizer_success": False, "optimizer_status": None,
                    "optimizer_message": None, "nfev": None, "status": "numerical_failure",
                    "error": str(exc), "elapsed_seconds": time.perf_counter() - trial_started})
        if not fitted:
            parameters.append({"expiry_date": group, "status": "numerical_failure",
                "error": "all training starts failed", "elapsed_seconds": time.perf_counter() - stamp})
            continue
        objective, selected, opt = min(fitted, key=lambda row: (row[0], row[1]))
        coefficients = opt.x
        values[mask] = _black_total_variance(f[mask], d[mask], k[mask], design @ coefficients, calls[mask])
        # Convert Bernstein coefficients on u=(log(K/F)-low)/(high-low)
        # to the explicitly documented a+b*k+c*k^2 parameterization.
        slope_u = 2. * (coefficients[1] - coefficients[0])
        curvature_u = coefficients[0] - 2. * coefficients[1] + coefficients[2]
        width = high - low
        a = coefficients[0] - slope_u * low / width + curvature_u * low**2 / width**2
        b = slope_u / width - 2. * curvature_u * low / width**2
        c = curvature_u / width**2
        parameters.append({"expiry_date": group, "status": "converged" if opt.success else "exhausted",
            "optimizer_success": bool(opt.success), "selected_start": selected,
            "training_objective": objective, "a": float(a), "b": float(b), "c": float(c),
            **{f"bernstein_{i}": float(x) for i, x in enumerate(coefficients)},
            "active_bounds": {f"bernstein_{i}": "lower" if bound < 0 else "upper"
                              for i, bound in enumerate(opt.active_mask) if bound},
            "log_moneyness_min": low, "log_moneyness_max": high,
            "nfev": int(opt.nfev), "train_count": int(fitting.sum()),
            "test_count": int((mask & ~train).sum()),
            "elapsed_seconds": time.perf_counter() - stamp, "error": None})
        checks.extend(_shape_checks(coefficients, low, high, f[mask][0], d[mask][0], k[mask], group, grid_size))
    parameter_table = pd.DataFrame(parameters)
    all_valid = bool(np.all(np.isfinite(values)))
    all_converged = all_valid and bool((parameter_table.status == "converged").all())
    status = "converged" if all_converged else "exhausted" if all_valid else "incomplete"
    summary = {"scenario": "independent_expiry_quadratic_total_variance", "status": status,
        "optimizer_success": all_converged, "diagnostic_only": True,
        "free_parameters": 3 * len(parameter_table),
        "total_nfev": sum(row["nfev"] or 0 for row in attempts),
        "elapsed_seconds": time.perf_counter() - started,
        **_errors(values, target, scale, train)}
    return {"summary": pd.DataFrame([summary]), "summary_records": [summary],
        "predictions": _prediction(quotes, values, target, scale,
            scenario="independent_expiry_quadratic_total_variance"),
        "parameters": parameter_table, "shape_checks": pd.DataFrame(checks),
        "attempts": pd.DataFrame(attempts), "metadata": {
            "selection": "minimum training price objective across two deterministic starts per expiry",
            "objective": "unchanged spread-normalized dollar prices",
            "total_variance_parameterization": "positive Bernstein quadratic on observed log-moneyness range",
            "coefficient_bounds": [1e-10, 4.], "max_nfev_per_start": int(max_nfev),
            "price_scale_floor": .005,
            "dense_grid_size": int(grid_size), "dynamic_model": False,
            "global_arbitrage_free_guarantee": False,
            "independent_stock_observation": False, "eligible_for_hedging": False,
            "limitations": ["Independent maturity smiles; no dynamic or calendar consistency",
                "Shape checks cover observed range only; no extrapolation certificate",
                "More flexibility per expiry is not evidence of superior hedging performance"]}}


__all__ = ["forward_sensitivity", "quadratic_smile_benchmark"]
