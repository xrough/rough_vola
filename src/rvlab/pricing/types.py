"""Parameters and numerical controls for European vanilla pricing.

Variances are annualised variances, times are years, and model prices are per
underlying unit. Contract multipliers, calendars, curves and spot-to-forward
conversion belong to the caller. No Feller condition is imposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral, Real


class PricingError(RuntimeError):
    """A numerical price could not be computed reliably with these settings."""


def _nonnegative(name: str, value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class BlackScholesParams:
    """Constant annualised volatility, expressed as a decimal."""

    volatility: float

    def __post_init__(self) -> None:
        _nonnegative("volatility", self.volatility)


@dataclass(frozen=True)
class HestonParams:
    """Risk-neutral Heston parameters; v0 and theta are variances.

    dV = kappa * (theta - V) dt + vol_of_vol * sqrt(V) dB.
    rho is the correlation of stock and variance Brownian innovations.
    """

    v0: float
    theta: float
    kappa: float
    vol_of_vol: float
    rho: float

    def __post_init__(self) -> None:
        for name in ("v0", "theta", "kappa", "vol_of_vol"):
            _nonnegative(name, getattr(self, name))
        if (isinstance(self.rho, bool) or not isinstance(self.rho, Real)
                or not math.isfinite(self.rho) or abs(self.rho) > 1):
            raise ValueError("rho must be finite and in [-1, 1]")


@dataclass(frozen=True)
class RoughHestonParams(HestonParams):
    """Volterra Heston parameters with K(t)=t**(H-.5)/Gamma(H+.5).

    The drift and diffusion coefficients live inside the kernel convolution;
    their time scaling therefore depends on H. H=.5 is exactly Heston.
    """

    hurst: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if (isinstance(self.hurst, bool) or not isinstance(self.hurst, Real)
                or not math.isfinite(self.hurst)
                or not 0 < self.hurst <= 0.5):
            raise ValueError("hurst must be finite and in (0, 0.5]")


@dataclass(frozen=True)
class PricingConfig:
    """Fixed Fourier and fractional-Riccati resolutions.

    These settings are an explicit numerical budget, not an accuracy promise.
    Compare refined prices for the parameter range and quote set being used.
    """

    quadrature_nodes: int = 128
    fourier_cutoff: float = 80.0
    riccati_steps: int = 256

    def __post_init__(self) -> None:
        for name in ("quadrature_nodes", "riccati_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.fourier_cutoff, bool) or not isinstance(self.fourier_cutoff, Real)
                or not math.isfinite(self.fourier_cutoff) or self.fourier_cutoff <= 0):
            raise ValueError("fourier_cutoff must be finite and positive")

    def refined(self) -> PricingConfig:
        """Refine time resolution, quadrature resolution and integration range."""
        return replace(self, quadrature_nodes=2 * self.quadrature_nodes,
                       fourier_cutoff=1.25 * self.fourier_cutoff,
                       riccati_steps=2 * self.riccati_steps)


ModelParams = BlackScholesParams | HestonParams | RoughHestonParams

__all__ = ["BlackScholesParams", "HestonParams", "RoughHestonParams",
           "PricingConfig", "PricingError", "ModelParams"]
