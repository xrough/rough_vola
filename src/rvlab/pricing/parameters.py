"""Scalar calibration coordinates, keeping curve knot times fixed.

The curve levels are mean-reversion levels ``theta_0``, ``theta_1``, ...;
they are not samples of the expected forward-variance curve. Knot locations
are structural inputs and are never optimized by ``calibrate_surface``.
"""

from dataclasses import fields, replace

from .types import BlackScholesParams, HestonParams, RoughHestonParams
from .variance_curve import HestonCurveParams, RoughHestonCurveParams, MeanReversionCurve

ModelParams = (BlackScholesParams | HestonParams | RoughHestonParams
               | HestonCurveParams | RoughHestonCurveParams)


def parameter_values(params: ModelParams) -> dict[str, float]:
    """Return ordered scalar coordinates, with fixed curve knots omitted."""
    if isinstance(params, (HestonCurveParams, RoughHestonCurveParams)):
        result = {field.name: getattr(params, field.name)
                  for field in fields(params) if field.name != "curve"}
        result.update({f"theta_{i}": level for i, level in enumerate(params.curve.levels)})
        return result
    if isinstance(params, (BlackScholesParams, HestonParams, RoughHestonParams)):
        return {field.name: getattr(params, field.name) for field in fields(params)}
    raise TypeError("params must be a supported Black–Scholes, Heston or rough-Heston parameter object")


def replace_parameters(params: ModelParams, **values) -> ModelParams:
    """Replace scalar coordinates without altering a curve's knot times."""
    unknown = set(values) - set(parameter_values(params))
    if unknown:
        raise ValueError(f"unknown model parameters: {sorted(unknown)}")
    if isinstance(params, (HestonCurveParams, RoughHestonCurveParams)):
        scalars = {name: value for name, value in values.items() if not name.startswith("theta_")}
        levels = tuple(values.get(f"theta_{i}", level)
                       for i, level in enumerate(params.curve.levels))
        return replace(params, **scalars, curve=MeanReversionCurve(params.curve.knot_times, levels))
    return replace(params, **values)


__all__ = ["ModelParams", "parameter_values", "replace_parameters"]
