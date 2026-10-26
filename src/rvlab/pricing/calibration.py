"""Bounded deterministic calibration of one European option surface.

Each row is one contract. Inputs use per-share prices, explicit forwards and
discount factors, and maturities in years. No data loading, quote filtering,
exercise-style adjustment, or market-input estimation happens here.

When both ``bid`` and ``ask`` are supplied, the target is their midpoint and
the residual is divided by ``max(half_spread, price_scale_floor)``. Otherwise
``market_price`` is required and residuals are in per-share price units. A
supplied ``market_price`` column is retained but ignored when bid/ask exist.
All rows must be valid; invalid data and numerical failures are never dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
import time
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .api import price
from .diagnostics import PricingDiagnostics, pricing_diagnostics
from .types import (
    PricingConfig, PricingError,
)
from .parameters import ModelParams, parameter_values, replace_parameters
from .optimization import OptimizerConfig, _Coordinates, _finite_difference_jacobian


@dataclass(frozen=True)
class CalibrationResult:
    """A fitted surface, optimizer outcome, and independent refinement check.

    ``objective`` is mean squared scaled residual, not SciPy's half sum of
    squares. ``fitted_quotes`` preserves input row order and index, including
    duplicate index labels. It is a detached DataFrame and remains editable.
    ``ready`` requires optimizer convergence and numerical stability; it is
    not a test of fit quality, parameter identification, or trading usefulness.
    """

    params: ModelParams
    fitted_quotes: pd.DataFrame
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    nfev: int
    njev: int | None
    objective: float
    refined_objective: float
    price_rmse: float
    refined_price_rmse: float
    max_price_change: float
    numerical_stable: bool
    config: PricingConfig
    refined_config: PricingConfig
    active_bounds: Mapping[str, str]
    fixed_parameters: Mapping[str, float]
    elapsed_seconds: float
    diagnostics: PricingDiagnostics | None = None
    pricing_evaluations: int = 0
    optimality: float = float("nan")
    physical_optimality: float = float("nan")
    optimizer_settings: Mapping | None = None
    metadata: Mapping | None = None
    final_jacobian: np.ndarray | None = None
    jacobian_singular_values: tuple[float, ...] = ()
    free_parameters: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.optimizer_success and self.numerical_stable

    @property
    def success(self) -> bool:
        """Optimizer convergence only; inspect ``ready`` before accepting a fit."""
        return self.optimizer_success

    @property
    def status(self) -> int:
        return self.optimizer_status

    @property
    def message(self) -> str:
        return self.optimizer_message


def _locations(index: pd.Index, mask: np.ndarray) -> str:
    positions = np.flatnonzero(mask)
    shown = positions[:8]
    labels = [index[int(i)] for i in shown]
    tail = f" ({len(positions)} rows total)" if len(positions) > len(shown) else ""
    return f"row positions {shown.tolist()} (index labels {labels!r}){tail}"


def _numeric_column(quotes: pd.DataFrame, name: str) -> np.ndarray:
    bad_type = np.array([isinstance(v, (bool, np.bool_, complex, np.complexfloating))
                         for v in quotes[name].to_numpy()])
    if np.any(bad_type):
        raise ValueError(f"{name} must contain real numeric values at {_locations(quotes.index, bad_type)}")
    numeric = pd.to_numeric(quotes[name], errors="coerce")
    # Nullable pandas dtypes used pd.NA in older supported pandas versions.
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    bad = ~np.isfinite(values)
    if np.any(bad):
        raise ValueError(f"{name} must be finite and numeric at {_locations(quotes.index, bad)}")
    return values


def _market_array(value, name: str, quotes: pd.DataFrame) -> np.ndarray:
    # np.asarray deliberately ignores Series labels: alignment is positional.
    try:
        raw = np.broadcast_to(np.asarray(value, dtype=object), (len(quotes),))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be scalar or broadcast to {len(quotes)} quote rows") from exc
    bad_type = np.array([isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in raw])
    if np.any(bad_type):
        raise ValueError(f"{name} must contain real numeric values at {_locations(quotes.index, bad_type)}")
    values = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    bad = ~np.isfinite(values) | (values <= 0)
    if np.any(bad):
        raise ValueError(f"{name} must be finite and positive at {_locations(quotes.index, bad)}")
    return values


def _prepare_quotes(quotes, forward, discount, price_scale_floor, *,
                    exercise_style="european", bounds_label="European"):
    """Validate aligned prices without relabelling the original exercise style.

    The private exercise override is used only after the American-call
    reduction's market assumptions have been verified. It is never exposed
    by the European public calibrator.
    """
    if not isinstance(quotes, pd.DataFrame):
        raise TypeError("quotes must be a pandas DataFrame")
    if quotes.empty:
        raise ValueError("quotes must contain at least one contract")
    if not quotes.columns.is_unique:
        raise ValueError("quotes must have unique column names")
    if exercise_style == "american" and "exercise_style" not in quotes:
        raise ValueError("American calibration requires explicit exercise_style='american' on every row")
    if "exercise_style" in quotes:
        wrong_style = quotes["exercise_style"].astype(str).str.lower().ne(exercise_style).to_numpy()
        if wrong_style.any():
            prefix = ("European calibrate_surface rejects American or other non-European exercise styles at "
                      if exercise_style == "european" else
                      "American call calibration requires American exercise styles at ")
            raise ValueError(prefix + _locations(quotes.index, wrong_style)
                             + "; use an exercise-aware calibration interface")
    missing = {"strike", "maturity", "is_call"} - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes are missing required columns: {sorted(missing)}")
    has_bid, has_ask = "bid" in quotes, "ask" in quotes
    if has_bid != has_ask:
        raise ValueError("quotes must provide both bid and ask, or neither")
    if not has_bid and "market_price" not in quotes:
        raise ValueError("quotes require market_price or both bid and ask")

    strike = _numeric_column(quotes, "strike")
    maturity = _numeric_column(quotes, "maturity")
    for name, values in (("strike", strike), ("maturity", maturity)):
        bad = values <= 0
        if np.any(bad):
            raise ValueError(f"{name} must be positive at {_locations(quotes.index, bad)}")
    raw_flags = quotes["is_call"].to_numpy()
    bad_flags = np.array([not isinstance(v, (bool, np.bool_)) for v in raw_flags])
    if np.any(bad_flags):
        raise ValueError(f"is_call must contain booleans at {_locations(quotes.index, bad_flags)}")
    is_call = np.asarray(raw_flags, dtype=bool)
    f = _market_array(forward, "forward", quotes)
    d = _market_array(discount, "discount", quotes)

    if has_bid:
        bid, ask = _numeric_column(quotes, "bid"), _numeric_column(quotes, "ask")
        bad = (bid < 0) | (ask < bid)
        if np.any(bad):
            raise ValueError(f"require 0 <= bid <= ask at {_locations(quotes.index, bad)}")
        target = bid + 0.5 * (ask - bid)
        scales = np.maximum(0.5 * (ask - bid), price_scale_floor)
    else:
        target = _numeric_column(quotes, "market_price")
        scales = np.ones(len(quotes))

    lower = d * np.where(is_call, np.maximum(f - strike, 0), np.maximum(strike - f, 0))
    upper = d * np.where(is_call, f, strike)
    bad = ~np.isfinite(lower) | ~np.isfinite(upper)
    if np.any(bad):
        raise ValueError(f"market inputs produce nonfinite price bounds at {_locations(quotes.index, bad)}")
    tolerance = 1e-10 * np.maximum(1.0, upper)
    bad = (target < lower - tolerance) | (target > upper + tolerance)
    if np.any(bad):
        raise ValueError(f"target price violates {bounds_label} no-arbitrage bounds at {_locations(quotes.index, bad)}")
    return strike, maturity, is_call, f, d, target, scales


def _parameter_bounds(initial_params, bounds):
    values = parameter_values(initial_params)
    names = tuple(values)
    defaults = {
        "volatility": (0.0, 5.0), "v0": (0.0, 2.0), "theta": (0.0, 2.0),
        "kappa": (0.0, 20.0), "vol_of_vol": (0.0, 5.0),
        "rho": (-0.999, 0.999), "hurst": (0.01, 0.5),
    }
    if bounds is not None and not isinstance(bounds, Mapping):
        raise TypeError("bounds must map parameter names to (lower, upper) pairs")
    supplied = {} if bounds is None else dict(bounds)
    unknown = set(supplied) - set(names)
    if unknown:
        raise ValueError(f"unknown parameter bounds: {sorted(unknown)}")
    lower, upper = {}, {}
    for name in names:
        pair = supplied.get(name, defaults["theta"] if name.startswith("theta_") else defaults[name])
        try:
            if len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, Real) for v in pair):
                raise ValueError
            lo, hi = (float(value) for value in pair)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bounds for {name} must be a (lower, upper) pair") from exc
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"bounds for {name} must be finite with lower <= upper")
        value = values[name]
        if not lo <= value <= hi:
            raise ValueError(f"initial {name}={value} is outside bounds [{lo}, {hi}]")
        lower[name], upper[name] = lo, hi
    # Check physical parameter domains as well as the optimizer's box shape.
    try:
        replace_parameters(initial_params, **lower)
        replace_parameters(initial_params, **upper)
    except ValueError as exc:
        raise ValueError(f"bounds must remain inside model parameter domains: {exc}") from exc
    free = tuple(name for name in names if lower[name] < upper[name])
    fixed = {name: lower[name] for name in names if lower[name] == upper[name]}
    return free, fixed, lower, upper


def calibrate_surface(
    quotes: pd.DataFrame,
    initial_params: ModelParams,
    *,
    forward,
    discount,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    config: PricingConfig | None = None,
    max_nfev: int = 200,
    price_scale_floor: float = 0.005,
    stability_atol: float = 0.005,
    stability_rtol: float = 1e-4,
    optimizer: OptimizerConfig | None = None,
) -> CalibrationResult:
    """Fit one parameter set to all supplied contracts with bounded least squares.

    ``quotes`` requires ``strike``, ``maturity`` and boolean ``is_call`` plus
    ``market_price`` or both ``bid``/``ask``. Maturities must be strictly
    positive. Forward and discount are scalars or arrays broadcastable to the
    row count, aligned by position even when a Series is supplied.

    Bounds are partial overrides of the numerical search box: volatility
    [0,5], v0/theta [0,2], kappa [0,20], vol_of_vol [0,5], rho [-.999,.999],
    and hurst [.01,.5]. These are configurable search limits, not economic
    restrictions. Equal endpoints fix a parameter at its initial value.
    Curve models expose ``theta_0``, ``theta_1``, ... in place of ``theta``;
    these nonnegative mean-reversion levels default to [0,2]. Their supplied
    knot times stay fixed. Expected forward variance is derived from them.

    Optional ``optimizer`` controls Jacobian differencing, termination, and
    monotone physical/asinh/log1p coordinates. The default preserves physical
    coordinates. Transformations keep the same physical box and objective,
    including attainable zero levels. SciPy optimality uses its coordinates;
    ``physical_optimality`` reports a common physical-coordinate KKT measure.

    The final surface is checked with separate quadrature, Fourier-cutoff and
    Riccati refinements, plus a joint comparison. The cutoff is tested out to
    twice the fitted range. Every check must satisfy ``stability_atol +
    stability_rtol * abs(comparison_price)``. These are empirical convergence
    checks, not certified error bounds. Both fitted and reference objectives
    are retained. Optimizer exhaustion returns an unsuccessful result.
    Pricing failures during optimization raise ``PricingError`` with context.
    Failed post-fit probes remain in ``diagnostics`` and block readiness while
    preserving the fitted parameters and optimizer outcome.
    """
    return _calibrate_prepared_surface(
        quotes, initial_params, forward=forward, discount=discount, bounds=bounds,
        config=config, max_nfev=max_nfev, price_scale_floor=price_scale_floor,
        stability_atol=stability_atol, stability_rtol=stability_rtol, optimizer=optimizer,
    )


def _calibrate_prepared_surface(
    quotes, initial_params, *, forward, discount, bounds=None, config=None,
    max_nfev=200, price_scale_floor=.005, stability_atol=.005,
    stability_rtol=1e-4, optimizer=None, prepared=None, price_function=None,
    diagnostics_function=None, metadata=None, evaluation_callback=None,
    evaluation_observer=None,
):
    """Shared bounded optimizer, weighting and parameter-coordinate machinery.

    Exercise-aware callers supply guarded prepared quotes and pricing /
    diagnostic callbacks. Public European validation is never bypassed by
    changing or dropping an input label.
    """
    started = time.perf_counter()
    if optimizer is None:
        optimizer = OptimizerConfig()
    elif not isinstance(optimizer, OptimizerConfig):
        raise TypeError("optimizer must be OptimizerConfig or None")
    if config is None:
        config = PricingConfig()
    elif not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    for name, value, strictly_positive in (
        ("price_scale_floor", price_scale_floor, True),
        ("stability_atol", stability_atol, False),
        ("stability_rtol", stability_rtol, False),
    ):
        if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                or value < 0 or (strictly_positive and value == 0)):
            domain = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"{name} must be finite and {domain}")

    if prepared is None:
        prepared = _prepare_quotes(quotes, forward, discount, price_scale_floor)
    strike, maturity, is_call, f, d, target, scales = prepared
    free, fixed, lower, upper = _parameter_bounds(initial_params, bounds)
    coordinates = _Coordinates(free, lower, upper, optimizer)
    pricing_evaluations = 0

    def evaluate(params, settings, stage):
        nonlocal pricing_evaluations
        if evaluation_callback is not None:
            evaluation_callback(params, settings, stage)
        pricing_evaluations += 1
        try:
            if price_function is None:
                values = np.asarray(price(params, forward=f, discount=d, strike=strike,
                                          maturity=maturity, is_call=is_call,
                                          config=settings), dtype=float)
            else:
                values = np.asarray(price_function(params, settings), dtype=float)
        except PricingError as exc:
            raise PricingError(f"{stage} failed for {params!r}: {exc}") from exc
        if values.shape != target.shape or not np.all(np.isfinite(values)):
            raise PricingError(f"{stage} produced invalid prices for {params!r}")
        if evaluation_observer is not None:
            evaluation_observer(params, settings, stage, values.copy())
        return values

    def physical_residuals(values):
        params = replace_parameters(initial_params, **fixed, **dict(zip(free, values)))
        return (evaluate(params, config, "calibration pricing") - target) / scales

    cached_coordinates, cached_residuals = None, None

    def residuals(values):
        nonlocal cached_coordinates, cached_residuals
        result = physical_residuals(coordinates.decode(values))
        cached_coordinates, cached_residuals = np.asarray(values).copy(), result.copy()
        return result

    def transformed_jacobian(values):
        # Differencing in transformed coordinates can shrink a physical step
        # below price roundoff near zero levels. Differentiate physically and
        # apply the exact chain rule, preserving both the floor and the box.
        base = cached_residuals if np.array_equal(values, cached_coordinates) else None
        physical_jacobian, _ = _finite_difference_jacobian(
            physical_residuals, coordinates.decode(values), lower=coordinates.lower,
            upper=coordinates.upper, method=optimizer.jacobian, base=base,
        )
        return physical_jacobian*coordinates.derivative(values)

    if free:
        fitted = least_squares(
            residuals, coordinates.encode([parameter_values(initial_params)[name] for name in free]),
            bounds=coordinates.bounds,
            jac=optimizer.jacobian if optimizer.parameterization == "physical" else transformed_jacobian,
            method="trf", x_scale="jac", max_nfev=int(max_nfev),
            # TRF nudges a boundary start only slightly into its box. Its first
            # trust-region steps can then be tiny despite a large gradient;
            # objective-change termination would incorrectly accept that start.
            ftol=optimizer.ftol, xtol=optimizer.xtol, gtol=optimizer.gtol,
        )
        params = replace_parameters(initial_params, **fixed, **dict(zip(free, coordinates.decode(fitted.x))))
        optimizer_success, status, message = bool(fitted.success), int(fitted.status), str(fitted.message)
        nfev, njev = int(fitted.nfev), int(fitted.njev) if fitted.njev is not None else None
        optimality = float(fitted.optimality)
        physical_optimality = coordinates.physical_optimality(fitted.x, fitted.jac, fitted.fun)
        final_jacobian = np.asarray(fitted.jac, dtype=float)/coordinates.derivative(fitted.x)
        active = {name: "lower" if side < 0 else "upper"
                  for name, side in zip(free, fitted.active_mask) if side != 0}
    else:
        params = replace_parameters(initial_params, **fixed)
        optimizer_success, status, message = True, 1, "All parameters are fixed; no optimization was required."
        nfev, njev, active = 0, 0, {}
        optimality = 0.0
        physical_optimality = 0.0
        final_jacobian = np.empty((len(target), 0))

    final_jacobian = final_jacobian.copy()
    final_jacobian.setflags(write=False)
    singular_values = tuple(float(v) for v in np.linalg.svd(final_jacobian, compute_uv=False))
    model_prices = evaluate(params, config, "final calibration pricing")
    if diagnostics_function is None:
        diagnostics = pricing_diagnostics(
            params, forward=f, discount=d, strike=strike, maturity=maturity,
            is_call=is_call, config=config, atol=stability_atol, rtol=stability_rtol,
        )
        pricing_evaluations += diagnostics.pricing_evaluations
    else:
        # The injected diagnostic calls evaluate(), so every actual call is
        # already counted, observed and subject to its caller's time budget.
        diagnostics = diagnostics_function(params, evaluate, config,
                                           stability_atol, stability_rtol)
    refined_config = diagnostics.reference_config
    refined_prices = (diagnostics.reference_prices if diagnostics.reference_prices is not None
                      else np.full_like(model_prices, np.nan))
    price_residuals, refined_price_residuals = model_prices - target, refined_prices - target
    residuals_scaled, refined_scaled = price_residuals / scales, refined_price_residuals / scales
    changes = np.abs(refined_prices - model_prices)
    numerical_stable = diagnostics.stable

    output = quotes.copy(deep=True)
    for name, values in (
        ("forward", f), ("discount", d), ("target_price", target),
        ("model_price", model_prices), ("refined_model_price", refined_prices),
        ("price_scale", scales), ("price_residual", price_residuals),
        ("residual", residuals_scaled), ("refined_price_residual", refined_price_residuals),
        ("refined_residual", refined_scaled), ("refinement_difference", changes),
    ):
        output[name] = values
    return CalibrationResult(
        params=params, fitted_quotes=output,
        optimizer_success=optimizer_success, optimizer_status=status,
        optimizer_message=message, nfev=nfev, njev=njev,
        objective=float(np.mean(residuals_scaled ** 2)),
        refined_objective=float(np.mean(refined_scaled ** 2)),
        price_rmse=float(np.sqrt(np.mean(price_residuals ** 2))),
        refined_price_rmse=float(np.sqrt(np.mean(refined_price_residuals ** 2))),
        max_price_change=diagnostics.max_price_change, numerical_stable=numerical_stable,
        config=config, refined_config=refined_config,
        active_bounds=MappingProxyType(active), fixed_parameters=MappingProxyType(fixed),
        elapsed_seconds=time.perf_counter() - started,
        diagnostics=diagnostics, pricing_evaluations=pricing_evaluations,
        optimality=optimality,
        physical_optimality=physical_optimality,
        optimizer_settings=MappingProxyType(coordinates.settings()),
        metadata=MappingProxyType(dict(metadata)) if metadata is not None else None,
        final_jacobian=final_jacobian, jacobian_singular_values=singular_values,
        free_parameters=free,
    )


__all__ = ["CalibrationResult", "calibrate_surface"]
