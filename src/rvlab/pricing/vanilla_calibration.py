"""Exercise-aware calibration sharing the existing bounded optimizer machinery."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from numbers import Integral, Real
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .calibration import _numeric_column, _locations, _parameter_bounds
from .parameters import parameter_values, replace_parameters
from .optimization import OptimizerConfig, _Coordinates, _finite_difference_jacobian
from .types import BlackScholesParams, HestonParams, RoughHestonParams, PricingError
from .variance_curve import HestonCurveParams, RoughHestonCurveParams
from .vanilla import evaluate_vanilla
from .vanilla_types import FDConfig, SpotMarket, UnsupportedModelError
from ._vanilla_common import build_grid, zero_spot_put_value


@dataclass(frozen=True)
class VanillaCalibrationResult:
    params: object
    fitted_quotes: pd.DataFrame
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    nfev: int
    njev: int | None
    price_stable: bool
    delta_stable: bool
    price_rmse: float
    refined_price_rmse: float
    objective: float
    refined_objective: float
    max_price_change: float
    max_delta_change: float
    active_bounds: Mapping
    fixed_parameters: Mapping
    pricing_evaluations: int
    elapsed_seconds: float
    config: FDConfig
    diagnostics: Mapping
    exercise_style: str
    optimality: float
    physical_optimality: float
    optimizer_settings: Mapping

    @property
    def ready(self):
        """Accepted numerical price fit; never certifies historical hedge readiness."""
        return self.optimizer_success and self.price_stable

    @property
    def hedge_delta_ready(self):
        return self.ready and self.delta_stable

    @property
    def numerical_stable(self):
        return self.price_stable

    @property
    def success(self):
        return self.optimizer_success


def _prepare(quotes, market, style, floor):
    if not isinstance(quotes, pd.DataFrame) or quotes.empty or not quotes.columns.is_unique:
        raise ValueError("quotes must be a nonempty DataFrame with unique columns")
    required = {"strike", "maturity", "is_call"}
    if required-set(quotes):
        raise ValueError(f"quotes missing columns: {sorted(required-set(quotes))}")
    if "exercise_style" in quotes:
        mismatch = quotes["exercise_style"].astype(str).str.lower().ne(style).to_numpy()
        if mismatch.any():
            raise ValueError("quote exercise style disagrees with calibration at "+_locations(quotes.index, mismatch))
    strike, maturity = _numeric_column(quotes, "strike"), _numeric_column(quotes, "maturity")
    for name, values in (("strike", strike), ("maturity", maturity)):
        if (values <= 0).any():
            raise ValueError(f"{name} must be positive at "+_locations(quotes.index, values <= 0))
    raw = quotes["is_call"].to_numpy()
    bad = np.array([not isinstance(v, (bool, np.bool_)) for v in raw])
    if bad.any():
        raise ValueError("is_call must contain booleans at "+_locations(quotes.index, bad))
    calls = raw.astype(bool)
    if ("bid" in quotes) != ("ask" in quotes):
        raise ValueError("quotes require both bid and ask")
    if "bid" in quotes:
        bid, ask = _numeric_column(quotes, "bid"), _numeric_column(quotes, "ask")
        bad = (bid < 0) | (ask < bid)
        if bad.any():
            raise ValueError("require 0 <= bid <= ask at "+_locations(quotes.index, bad))
        targets, scales = bid+.5*(ask-bid), np.maximum(.5*(ask-bid), floor)
    elif "market_price" in quotes:
        targets = _numeric_column(quotes, "market_price")
        scales = np.ones(len(quotes))
    else:
        raise ValueError("quotes require market_price or bid/ask")
    lower = np.maximum(np.where(calls, market.spot-strike, strike-market.spot), 0.) if style == "american" else np.zeros(len(quotes))
    upper = np.array([market.spot if call else k*zero_spot_put_value(market, 0., t, style)
                      for k, t, call in zip(strike, maturity, calls)])
    bad = (targets < lower-1e-9*np.maximum(1., upper)) | (targets > upper+1e-9*np.maximum(1., upper))
    if bad.any():
        raise ValueError(f"target violates {style} price bounds at "+_locations(quotes.index, bad))
    return strike, maturity, calls, targets, scales


def calibrate_vanilla_surface(quotes, initial_params, *, market, exercise_style="american",
        bounds=None, config=None, max_nfev=200, price_scale_floor=.005,
        optimizer=None, stability_atol=.001, stability_rtol=1e-4,
        delta_atol=.002, delta_rtol=.005, spread_refinement_fraction=.1):
    """Calibrate raw exercise-matched prices on one frozen finite-difference grid.

    Price-only observations are unweighted. Bid/ask fits use midpoint prices
    and max(half spread, floor). Every independent price probe must additionally
    change prices by no more than the stated fraction of each weighting scale.
    Repricing evaluations include final Greek/grid diagnostics, independently
    of SciPy's optimizer counter. Fixed parameters and unsuccessful fits persist.
    """
    started = time.perf_counter()
    if isinstance(initial_params, (RoughHestonParams, RoughHestonCurveParams)) or not isinstance(initial_params, (BlackScholesParams, HestonParams, HestonCurveParams)):
        raise UnsupportedModelError("finite-difference calibration supports BS and Heston (including Heston curves), not American rough Heston")
    if not isinstance(market, SpotMarket):
        raise TypeError("market must be SpotMarket")
    if exercise_style not in {"american", "european"}:
        raise ValueError("exercise_style must be american or european")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be positive integer")
    for name, value, positive in (("price_scale_floor", price_scale_floor, True),
            ("stability_atol", stability_atol, False), ("stability_rtol", stability_rtol, False),
            ("delta_atol", delta_atol, False), ("delta_rtol", delta_rtol, False),
            ("spread_refinement_fraction", spread_refinement_fraction, True)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 or positive and value == 0:
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if config is None:
        config = FDConfig(spot_steps=400, time_steps=400) if isinstance(initial_params, BlackScholesParams) else FDConfig()
    if not isinstance(config, FDConfig):
        raise TypeError("config must be FDConfig")
    if optimizer is None:
        optimizer = OptimizerConfig()
    if not isinstance(optimizer, OptimizerConfig):
        raise TypeError("optimizer must be OptimizerConfig")
    k, t, calls, target, scales = _prepare(quotes, market, exercise_style, price_scale_floor)
    # Validate complete market coverage before any optimizer work.
    market.discount_curve.discount(float(t.max()))
    if market.dividends.coverage < float(t.max())-1e-12:
        raise ValueError("dividend expectation coverage ends before calibration maturity")
    free, fixed, lower, upper = _parameter_bounds(initial_params, bounds)
    stochastic = not isinstance(initial_params, BlackScholesParams) and upper["vol_of_vol"] > 0
    grid = build_grid(market, k, config, variance=stochastic)
    if stochastic and upper["v0"] >= grid.variance[-1]:
        raise ValueError("variance grid must extend beyond the entire v0 calibration bound")
    coords = _Coordinates(free, lower, upper, optimizer)
    pricing_evaluations = 0
    grid_fingerprints = set()
    def evaluate(params, *, refine=False):
        nonlocal pricing_evaluations
        try:
            result = evaluate_vanilla(params, market=market, strike=k, maturity=t,
                is_call=calls, exercise_style=exercise_style, config=config, grid=grid,
                check_refinement=refine, price_atol=stability_atol, price_rtol=stability_rtol,
                delta_atol=delta_atol, delta_rtol=delta_rtol)
        except PricingError as exc:
            raise PricingError(f"vanilla calibration failed for {params!r}: {exc}") from exc
        pricing_evaluations += result.diagnostics["pricing_evaluations"]
        grid_fingerprints.add(result.diagnostics["grid_sha256"])
        return result
    def physical_residuals(values):
        params = replace_parameters(initial_params, **fixed, **dict(zip(free, values)))
        return (np.asarray(evaluate(params).prices)-target)/scales
    cached_x, cached_y = None, None
    def residuals(x):
        nonlocal cached_x, cached_y
        y = physical_residuals(coords.decode(x))
        cached_x, cached_y = x.copy(), y.copy()
        return y
    def jacobian(x):
        base = cached_y if np.array_equal(x, cached_x) else None
        jac, _ = _finite_difference_jacobian(physical_residuals, coords.decode(x),
            lower=coords.lower, upper=coords.upper, method=optimizer.jacobian, base=base)
        return jac*coords.derivative(x)
    if free:
        fit = least_squares(residuals, coords.encode([parameter_values(initial_params)[n] for n in free]),
            bounds=coords.bounds, jac=jacobian, method="trf", x_scale="jac", max_nfev=int(max_nfev),
            ftol=optimizer.ftol, xtol=optimizer.xtol, gtol=optimizer.gtol)
        params = replace_parameters(initial_params, **fixed, **dict(zip(free, coords.decode(fit.x))))
        success, status, message = bool(fit.success), int(fit.status), str(fit.message)
        nfev, njev = int(fit.nfev), int(fit.njev) if fit.njev is not None else None
        active = {name: "lower" if side < 0 else "upper" for name, side in zip(free, fit.active_mask) if side}
        optimality = float(fit.optimality)
        physical_optimality = coords.physical_optimality(fit.x, fit.jac, fit.fun)
    else:
        params = replace_parameters(initial_params, **fixed)
        success, status, message, nfev, njev, active = True, 1, "All parameters fixed; no optimization required.", 0, 0, {}
        optimality = physical_optimality = 0.
    result = evaluate(params, refine=True)
    model = np.asarray(result.prices)
    refined = np.asarray(result.diagnostics["reference_prices"])
    checks = result.diagnostics["checks"].copy()
    spread_stable = True
    if "bid" in quotes:
        for axis, changes in result.diagnostics["price_changes"].items():
            passed = bool(np.all(changes <= spread_refinement_fraction*scales))
            checks.loc[checks.axis == axis, "spread_stable"] = passed
            spread_stable &= passed
    price_stable = bool(result.price_stable and spread_stable)
    output = quotes.copy(deep=True)
    for name, values in (("target_price", target), ("price_scale", scales), ("model_price", model),
            ("refined_model_price", refined), ("price_residual", model-target),
            ("refined_price_residual", refined-target), ("residual", (model-target)/scales),
            ("refined_residual", (refined-target)/scales), ("delta", result.deltas),
            ("refined_delta", result.diagnostics["reference_deltas"])):
        output[name] = values
    output["exercise_style"] = exercise_style
    diagnostics = {**result.diagnostics, "checks": checks, "spread_stable": spread_stable,
                   "grid_fingerprints": sorted(grid_fingerprints), "fixed_grid": len(grid_fingerprints) == 1,
                   "backend": "american_vanilla_fd" if exercise_style == "american" else "european_vanilla_fd"}
    return VanillaCalibrationResult(params, output, success, status, message, nfev, njev,
        price_stable, bool(result.delta_stable), float(np.sqrt(np.mean((model-target)**2))),
        float(np.sqrt(np.mean((refined-target)**2))), float(np.mean(((model-target)/scales)**2)),
        float(np.mean(((refined-target)/scales)**2)), float(checks.max_price_change.max()),
        float(checks.max_delta_change.max()), MappingProxyType(active), MappingProxyType(fixed),
        pricing_evaluations, time.perf_counter()-started, config, MappingProxyType(diagnostics),
        exercise_style, optimality, physical_optimality,
        MappingProxyType({**coords.settings(), "physical_bounds": {
            name: (lower[name], upper[name]) for name in lower}}))


__all__ = ["VanillaCalibrationResult", "calibrate_vanilla_surface"]
