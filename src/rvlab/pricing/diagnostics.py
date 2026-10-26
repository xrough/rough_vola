"""Separate Fourier-tail, quadrature and fractional-time convergence checks.

All comparisons hold the fitted parameters and contract inputs fixed.  These
are empirical convergence checks, not a certified error bound.  In particular,
an optimizer can converge on an inaccurate integration grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Real
import time

import numpy as np
import pandas as pd

from ._validation import broadcast_inputs
from .api import price
from .types import PricingConfig, PricingError, RoughHestonParams
from .parameters import ModelParams
from .variance_curve import RoughHestonCurveParams


@dataclass(frozen=True)
class PricingDiagnostics:
    """Fixed-parameter prices and independent numerical comparisons.

    Arrays retain the broadcast input shape, including a zero-dimensional
    array for scalar inputs.  ``reference_prices`` is absent when that grid
    fails.  The detached ``checks`` table records unsuccessful probes instead
    of replacing their prices with zeros.  ``stable`` requires every check to
    succeed and satisfy the supplied price tolerance.
    """

    config: PricingConfig
    reference_config: PricingConfig
    base_prices: np.ndarray
    reference_prices: np.ndarray | None
    checks: pd.DataFrame
    stable: bool
    pricing_evaluations: int
    elapsed_seconds: float

    @property
    def numerical_stable(self) -> bool:
        return self.stable

    @property
    def max_price_change(self) -> float:
        """Largest available comparison; failed checks still block stability."""
        changes = self.checks["max_price_change"].dropna()
        return float(changes.max()) if len(changes) else float("nan")


def pricing_diagnostics(
    params: ModelParams,
    *,
    forward,
    discount,
    strike,
    maturity,
    is_call=True,
    config: PricingConfig | None = None,
    atol: float = .005,
    rtol: float = 1e-4,
) -> PricingDiagnostics:
    """Check pricing accuracy by varying numerical controls separately.

    For a fitted grid ``(N, L, M)``, compare quadrature at ``2N``; extend the
    Fourier range to ``1.5L`` and then ``2L`` at that same quadrature/time
    resolution; and test both the successive and cumulative tail changes.
    Active rough Heston additionally doubles the Riccati resolution at
    ``(2N, 2L)``.  A joint comparison against the original grid guards against
    error accumulation.  Heston and the explicitly delegated ``hurst=.5``
    limit do not run a redundant time-resolution probe.

    The base grid must produce valid finite prices or ``PricingError`` is
    raised.  Failures on diagnostic grids are retained in ``checks`` and
    make the outcome unstable.  Malformed caller inputs raise normally.
    Contract order and broadcasting are unchanged.
    """
    started = time.perf_counter()
    if config is None:
        config = PricingConfig()
    elif not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")

    f, d, k, t, flags = broadcast_inputs(
        forward=forward, discount=discount, strike=strike,
        maturity=maturity, is_call=is_call,
    )
    kwargs = dict(forward=f, discount=d, strike=k, maturity=t, is_call=flags)
    cache: dict[PricingConfig, tuple[np.ndarray | None, str | None]] = {}

    def evaluate(settings):
        if settings not in cache:
            try:
                values = np.asarray(price(params, **kwargs, config=settings), dtype=float)
                if values.shape != f.shape or not np.all(np.isfinite(values)):
                    raise PricingError("pricing diagnostic produced invalid or nonfinite prices")
                cache[settings] = (values.copy(), None)
            except PricingError as exc:
                cache[settings] = (None, str(exc))
        return cache[settings]

    base, base_error = evaluate(config)
    if base is None:
        raise PricingError(f"base pricing diagnostic failed: {base_error}")
    quadrature = replace(config, quadrature_nodes=2 * config.quadrature_nodes)
    cutoff_mid = replace(quadrature, fourier_cutoff=1.5 * config.fourier_cutoff)
    cutoff_far = replace(quadrature, fourier_cutoff=2 * config.fourier_cutoff)
    active_rough = isinstance(params, (RoughHestonParams, RoughHestonCurveParams)) and params.hurst < .5
    reference = (replace(cutoff_far, riccati_steps=2 * config.riccati_steps)
                 if active_rough else cutoff_far)
    comparisons = [
        ("quadrature", config, quadrature),
        ("cutoff_mid", quadrature, cutoff_mid),
        ("cutoff_far", cutoff_mid, cutoff_far),
        ("cutoff_cumulative", quadrature, cutoff_far),
    ]
    if active_rough:
        comparisons.append(("riccati", cutoff_far, reference))
    comparisons.append(("joint", config, reference))

    rows = []
    for name, baseline_config, settings in comparisons:
        check_started = time.perf_counter()
        baseline, baseline_error = evaluate(baseline_config)
        current, current_error = evaluate(settings)
        error = None
        if baseline is None or current is None:
            unavailable = []
            if baseline_error is not None:
                unavailable.append(f"baseline grid: {baseline_error}")
            if current_error is not None:
                unavailable.append(f"comparison grid: {current_error}")
            error = "; ".join(unavailable)
            maximum, stable = None, False
            status = "numerical_failure"
        else:
            changes = np.abs(current - baseline)
            maximum = float(np.max(changes)) if changes.size else 0.
            stable = bool(np.all(changes <= atol + rtol * np.abs(current)))
            status = "stable" if stable else "unstable"
        rows.append({
            "check": name, "baseline_config": asdict(baseline_config),
            "config": asdict(settings), **asdict(settings),
            "status": status, "error": error,
            "max_price_change": maximum, "numerical_stable": stable,
            "elapsed_seconds": time.perf_counter() - check_started,
        })
    checks = pd.DataFrame(rows)
    reference_prices, _ = evaluate(reference)
    return PricingDiagnostics(
        config=config, reference_config=reference,
        base_prices=base.copy(),
        reference_prices=reference_prices.copy() if reference_prices is not None else None,
        checks=checks, stable=bool(checks["numerical_stable"].all()),
        pricing_evaluations=len(cache),
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = ["PricingDiagnostics", "pricing_diagnostics"]
