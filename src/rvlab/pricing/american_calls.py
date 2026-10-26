"""American calls reduced exactly under explicit no-early-exercise assumptions.

Under zero distributions, nonnegative deterministic forward rates and a true
martingale discounted stock in a frictionless market, an American vanilla
call equals its European value. This module checks the representable market
conditions and records the economic martingale assumption. It does not infer
that assumption from a characteristic function returning one at ``-i``.
The affine Volterra-Heston martingale result is established by Abi Jaber,
Larsson and Pulido, Theorem 7.1(iii) / Lemma 7.3 (arXiv:1708.08796v3),
https://arxiv.org/abs/1708.08796. It imposes no negative-correlation-only rule.

The rough model here starts a new implied surface at valuation. No historical
conditional state, stock Greeks, stopping approximation or put calibration is
provided by this reduction.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import math
from numbers import Real
from types import MappingProxyType
import time

import numpy as np
import pandas as pd

from ._validation import broadcast_inputs
from .api import price
from .calibration import (
    CalibrationResult, _calibrate_prepared_surface, _locations, _numeric_column,
    _prepare_quotes,
)
from .diagnostics import PricingDiagnostics
from .optimization import OptimizerConfig
from .types import (
    BlackScholesParams, HestonParams, RoughHestonParams, PricingConfig, PricingError,
)
from .vanilla_types import SpotMarket, UnsupportedModelError


_SUPPORTED = (BlackScholesParams, HestonParams, RoughHestonParams)


def _market_inputs(params, market, strike, maturity, is_call, conditional_state):
    if type(params) not in _SUPPORTED:
        raise UnsupportedModelError(
            "American-call equivalence supports constant BlackScholesParams, "
            "HestonParams and RoughHestonParams only")
    if conditional_state is not None:
        raise UnsupportedModelError(
            "American-call equivalence does not support historical conditional states")
    if not isinstance(market, SpotMarket):
        raise TypeError("market must be SpotMarket with independent observed spot and explicit dividends")
    _, _, k, t, flags = broadcast_inputs(
        forward=market.spot, discount=1., strike=strike, maturity=maturity, is_call=is_call)
    if not np.all(flags):
        raise ValueError("American-call equivalence accepts calls only; puts require another engine")
    horizon = float(np.max(t)) if t.size else 0.
    if horizon > market.dividends.coverage + 1e-12:
        raise ValueError("American-call equivalence requires complete dividend coverage through every maturity")
    if market.dividends.events:
        raise ValueError("American-call equivalence requires an explicitly empty dividend schedule")
    # Coverage check uses the curve's ordinary no-extrapolation policy.
    d = np.asarray(market.discount_curve.discount(t), dtype=float)
    starts = np.asarray(market.discount_curve.times[:-1])
    log_discounts = np.log(market.discount_curve.discounts)
    if np.any(np.diff(log_discounts)[starts < horizon] > 0):
        raise ValueError(
            "American-call equivalence requires nonnegative forward rates on every interval through maturity")
    return market.spot/d, d, k, t, flags


def price_american_calls(
    params: BlackScholesParams | HestonParams | RoughHestonParams,
    *, market: SpotMarket, strike, maturity, is_call=True,
    config: PricingConfig | None = None, conditional_state=None,
):
    """Price American calls per underlying unit using the exact reduction.

    ``market`` must declare complete empty expected-dividend coverage. Every
    relevant log-discount interval must have nonnegative forward rate. The
    caller additionally assumes no other carry benefit, frictionless trading,
    and a true-martingale discounted stock under the specified dynamics.
    Forwards are derived from independent spot as ``S / D(T)``; they are never
    inferred from option parity. Scalar and array broadcasting match ``price``.
    """
    f, d, k, t, flags = _market_inputs(
        params, market, strike, maturity, is_call, conditional_state)
    return price(params, forward=f, discount=d, strike=k, maturity=t,
                 is_call=flags, config=config)


def _reduction_metadata(params):
    return {
        "exercise_style": "american", "payoff": "vanilla_call",
        "backend": "black_scholes_analytic" if type(params) is BlackScholesParams else
                   "rough_heston_fractional_riccati_fourier" if type(params) is RoughHestonParams and params.hurst < .5 else
                   "heston_characteristic_function_fourier",
        "reduction": "american_call_equals_european_no_early_exercise",
        "forward_source": "independent_observed_spot_divided_by_discount",
        "assumptions": (
            "complete_explicitly_empty_expected_dividend_schedule",
            "nonnegative_deterministic_forward_rates_on_every_relevant_interval",
            "frictionless_market_without_additional_carry_benefit",
            "true_martingale_discounted_stock_assumed_not_numerically_proved",
        ),
        "conditioning": "new_implied_surface_at_valuation_without_historical_state",
        "greeks_validated": False, "hedging_ready": False,
    }


def _call_diagnostics(params, evaluate, config, atol, rtol, *, scales, spread_fraction):
    """Apply price AND spread budgets to each separate numerical comparison."""
    started = time.perf_counter()
    cache = {}

    def probe(settings):
        if settings not in cache:
            try:
                cache[settings] = (evaluate(params, settings, "American-call refinement pricing"), None)
            except PricingError as exc:
                cache[settings] = (None, str(exc))
        return cache[settings]

    base, error = probe(config)
    if base is None:
        raise PricingError(f"base American-call diagnostic failed: {error}")
    quadrature = replace(config, quadrature_nodes=2*config.quadrature_nodes)
    mid = replace(quadrature, fourier_cutoff=1.5*config.fourier_cutoff)
    far = replace(quadrature, fourier_cutoff=2*config.fourier_cutoff)
    active_rough = type(params) is RoughHestonParams and params.hurst < .5
    reference = replace(far, riccati_steps=2*config.riccati_steps) if active_rough else far
    comparisons = [("quadrature", config, quadrature), ("cutoff_mid", quadrature, mid),
                   ("cutoff_far", mid, far), ("cutoff_cumulative", quadrature, far)]
    if active_rough:
        comparisons.append(("riccati", far, reference))
    comparisons.append(("joint", config, reference))
    rows = []
    for name, old_config, new_config in comparisons:
        check_started = time.perf_counter()
        old, old_error = probe(old_config)
        new, new_error = probe(new_config)
        if old is None or new is None:
            price_stable = spread_stable = False
            maximum = fraction = None
            error = "; ".join(str(e) for e in (old_error, new_error) if e is not None)
            status = "numerical_failure"
        else:
            changes = np.abs(new-old)
            maximum = float(np.max(changes)) if changes.size else 0.
            price_stable = bool(np.all(changes <= atol+rtol*np.abs(new)))
            fraction = float(np.max(changes/scales)) if scales is not None else None
            spread_stable = bool(np.all(changes <= spread_fraction*scales)) if scales is not None else True
            status = "stable" if price_stable and spread_stable else "unstable"
            error = None
        rows.append({
            "check": name, "baseline_config": asdict(old_config), "config": asdict(new_config),
            **asdict(new_config), "status": status, "error": error,
            "max_price_change": maximum, "price_stable": price_stable,
            "spread_stable": spread_stable, "max_spread_fraction": fraction,
            "spread_refinement_fraction": spread_fraction if scales is not None else None,
            "numerical_stable": price_stable and spread_stable,
            "elapsed_seconds": time.perf_counter()-check_started,
        })
    checks = pd.DataFrame(rows)
    reference_prices, _ = probe(reference)
    return PricingDiagnostics(
        config=config, reference_config=reference, base_prices=base.copy(),
        reference_prices=reference_prices.copy() if reference_prices is not None else None,
        checks=checks, stable=bool(checks.numerical_stable.all()),
        pricing_evaluations=len(cache), elapsed_seconds=time.perf_counter()-started,
    )


def calibrate_american_call_surface(
    quotes: pd.DataFrame,
    initial_params: BlackScholesParams | HestonParams | RoughHestonParams,
    *, market: SpotMarket,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    config: PricingConfig | None = None, max_nfev: int = 120,
    price_scale_floor: float = .005, stability_atol: float = .001,
    stability_rtol: float = 1e-4, spread_refinement_fraction: float = .1,
    optimizer: OptimizerConfig | None = None, conditional_state=None,
    evaluation_callback=None, evaluation_observer=None,
) -> CalibrationResult:
    """Fit explicitly American-labelled calls under the guarded reduction.

    Midpoints, half-spread weights and parameter coordinates are identical to
    ``calibrate_surface``. Equal bounds fix parameters. Original exercise
    labels, ordering and duplicate index labels are preserved. ``ready``
    requires optimizer convergence and EVERY numerical probe to satisfy the
    price tolerance and, for bid/ask inputs, the effective half-spread budget.

    ``evaluation_callback(params, settings, stage)`` runs immediately before
    each actual price evaluation, including refinements. Raise ``TimeoutError``
    (or another non-pricing exception) to interrupt a campaign deadline; pricing
    exceptions on diagnostic grids remain recorded numerical failures. ``evaluation_observer(params,
    settings, stage, prices)`` receives a detached array after successful calls
    and can checkpoint valid candidates on interruption. Optimizer nfev is
    reported separately from actual pricing evaluations. Final Jacobian and
    singular values concern scaled training residuals in physical coordinates;
    they do not establish unique identification.
    """
    if (isinstance(price_scale_floor, bool) or not isinstance(price_scale_floor, Real)
            or not math.isfinite(price_scale_floor) or price_scale_floor <= 0):
        raise ValueError("price_scale_floor must be finite and positive")
    if (isinstance(spread_refinement_fraction, bool) or not isinstance(spread_refinement_fraction, Real)
            or not math.isfinite(spread_refinement_fraction) or spread_refinement_fraction <= 0):
        raise ValueError("spread_refinement_fraction must be finite and positive")
    for name, callback in (("evaluation_callback", evaluation_callback), ("evaluation_observer", evaluation_observer)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")
    if not isinstance(quotes, pd.DataFrame):
        raise TypeError("quotes must be a pandas DataFrame")
    if not quotes.columns.is_unique:
        raise ValueError("quotes must have unique column names")
    missing = {"strike", "maturity", "is_call", "exercise_style"}-set(quotes.columns)
    if missing:
        raise ValueError(f"American call quotes are missing required columns: {sorted(missing)}")
    if quotes.empty:
        raise ValueError("quotes must contain at least one contract")
    k, t = _numeric_column(quotes, "strike"), _numeric_column(quotes, "maturity")
    for name, values in (("strike", k), ("maturity", t)):
        bad = values <= 0
        if bad.any():
            raise ValueError(f"{name} must be positive at " + _locations(quotes.index, bad))
    raw_flags = quotes.is_call.to_numpy()
    bad = np.array([not isinstance(flag, (bool, np.bool_)) for flag in raw_flags])
    if bad.any():
        raise ValueError("is_call must contain booleans at " + _locations(quotes.index, bad))
    bad = ~raw_flags.astype(bool)
    if bad.any():
        raise ValueError("American-call equivalence accepts calls only at " + _locations(quotes.index, bad))
    f, d, _, _, _ = _market_inputs(initial_params, market, k, t, raw_flags, conditional_state)
    prepared = _prepare_quotes(quotes, f, d, price_scale_floor,
                               exercise_style="american", bounds_label="American-call equivalence")
    # Save arrays from validated preparation; optimizer never changes contracts.
    k, t, flags, f, d, _, scales = prepared

    def prices(params, settings):
        return price(params, forward=f, discount=d, strike=k, maturity=t,
                     is_call=flags, config=settings)

    def diagnostics(params, evaluate, settings, atol, rtol):
        return _call_diagnostics(params, evaluate, settings, atol, rtol,
                                 scales=scales if "bid" in quotes else None,
                                 spread_fraction=spread_refinement_fraction)

    metadata = _reduction_metadata(initial_params)
    metadata["dividend_coverage"] = market.dividends.coverage
    metadata["spread_refinement_fraction"] = spread_refinement_fraction
    result = _calibrate_prepared_surface(
        quotes, initial_params, forward=f, discount=d, bounds=bounds, config=config,
        max_nfev=max_nfev, price_scale_floor=price_scale_floor,
        stability_atol=stability_atol, stability_rtol=stability_rtol,
        optimizer=optimizer, prepared=prepared, price_function=prices,
        diagnostics_function=diagnostics, metadata=metadata,
        evaluation_callback=evaluation_callback, evaluation_observer=evaluation_observer,
    )
    # H=.5 can be attained at an upper bound, so backend provenance describes
    # the fitted dynamics rather than assuming the initial H remains fixed.
    if result.metadata["backend"] != _reduction_metadata(result.params)["backend"]:
        result = replace(result, metadata=MappingProxyType({**result.metadata, **_reduction_metadata(result.params)}))
    return result


__all__ = ["price_american_calls", "calibrate_american_call_surface"]
