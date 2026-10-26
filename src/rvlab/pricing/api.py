"""One functional entry point for European vanilla model prices."""

from __future__ import annotations

from .black_scholes import black_scholes_price
from .types import BlackScholesParams, HestonParams, RoughHestonParams, PricingConfig
from .variance_curve import (
    HestonCurveParams, RoughHestonCurveParams, heston_curve_price, rough_heston_curve_price,
)


def price(params, *, forward, discount, strike, maturity, is_call=True,
          config: PricingConfig | None = None):
    """Price a scalar or broadcast quote grid using an explicit parameter type."""
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    kwargs = dict(forward=forward, discount=discount, strike=strike,
                  maturity=maturity, is_call=is_call)
    if isinstance(params, RoughHestonCurveParams):
        return rough_heston_curve_price(params, **kwargs, config=config)
    if isinstance(params, HestonCurveParams):
        return heston_curve_price(params, **kwargs, config=config)
    if isinstance(params, BlackScholesParams):
        return black_scholes_price(params, **kwargs)
    if isinstance(params, RoughHestonParams):
        from .rough_heston import rough_heston_price
        return rough_heston_price(params, **kwargs, config=config)
    if isinstance(params, HestonParams):
        from .heston import heston_price
        return heston_price(params, **kwargs, config=config)
    raise TypeError("params must be a supported Black–Scholes, Heston or rough-Heston parameter object")


__all__ = ["price"]
