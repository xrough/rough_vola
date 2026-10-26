"""Full-factor tensor stock splines for experimental premium regressions.

The fitted function is an approximation, never a conditional model price.
It can supply a frozen value control or a controlled exercise regression;
independent policy selection and valuation are required for the latter.
Multiplying stock spline features by factor features permits the stock
curvature and boundary shape to vary with the complete rough factor state.
It retains a piecewise quadratic stock representation, so the existing weak
branch / Gaussian expectation remains exact for the fitted control.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from numbers import Integral, Real

import numpy as np


def _factor_features(factors, center, scale, degree, clip):
    r = np.clip((factors-center)/scale, -clip, clip)
    columns = [np.ones(r.shape[:-1]+(1,)), r]
    if degree == 2:
        i, j = np.triu_indices(r.shape[-1])
        columns.append(r[..., i]*r[..., j])
    return np.concatenate(columns, axis=-1)


@dataclass(frozen=True)
class TensorPremium:
    """Immutable factor-dependent coefficients on each stock spline interval."""
    center: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    knots: tuple[float, ...]
    degree: int
    factor_clip: float
    rank: int
    condition: float

    def __post_init__(self):
        for name in ("center", "scale", "beta"):
            value = np.asarray(getattr(self, name), float).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def coefficients(self, factors):
        features = _factor_features(np.asarray(factors, float), self.center,
            self.scale, self.degree, self.factor_clip)
        stock_beta = np.einsum("...f,sf->...s", features, self.beta)
        a = stock_beta[..., 0]-stock_beta[..., 1]+stock_beta[..., 2]
        b = stock_beta[..., 1]-2*stock_beta[..., 2]
        c = stock_beta[..., 2]
        result = np.stack((a, b, c), axis=-1)[..., None, :]
        result = np.broadcast_to(result, features.shape[:-1]+(len(self.knots)+1, 3)).copy()
        for index, knot in enumerate(self.knots):
            gamma = stock_beta[..., 3+index]
            result[..., index+1:, 0] += gamma[..., None]*knot*knot
            result[..., index+1:, 1] -= 2*gamma[..., None]*knot
            result[..., index+1:, 2] += gamma[..., None]
        return result if self.knots else result[..., 0, :]

    def value(self, spots, factors, strike):
        spots, factors = np.asarray(spots, float), np.asarray(factors, float)
        if spots.ndim > factors.ndim-1:
            extra = spots.ndim-(factors.ndim-1)
            factors = factors.reshape(factors.shape[:-1]+(1,)*extra+(factors.shape[-1],))
            factors = np.broadcast_to(factors, spots.shape+(factors.shape[-1],))
        z = spots/strike
        coefficient = self.coefficients(factors)
        if self.knots:
            index = np.searchsorted(self.knots, z, side="right")
            coefficient = np.take_along_axis(coefficient, index[..., None, None], axis=-2)[..., 0, :]
        return coefficient[..., 0]+z*(coefficient[..., 1]+z*coefficient[..., 2])


def fit_tensor_premium(spots, factors, targets, strike, config, *,
                       degree=1, ridge=1e-4, factor_clip=4.):
    """Training-only normalized ridge fit with a bounded full-factor basis.

    The factor clip limits extrapolation in the approximation; it does not
    clip simulated states or change the transition law. Every factor remains
    a separate coordinate. No validation outcome or acceptance tolerance is
    used by this fit.
    """
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, Integral) or degree not in (1, 2):
        raise ValueError("degree must be one or two")
    for name, value, positive in (("ridge", ridge, False), ("factor_clip", factor_clip, True)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
            raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} real")
    spots, factors, targets = np.asarray(spots, float), np.asarray(factors, float), np.asarray(targets, float)
    if (spots.ndim != 1 or factors.ndim != 2 or targets.shape != spots.shape
            or factors.shape[0] != len(spots) or factors.shape[1] == 0 or len(spots) < 2
            or not all(np.isfinite(value).all() for value in (spots, factors, targets))
            or np.any(spots <= 0) or not np.isfinite(strike) or strike <= 0):
        raise ValueError("finite positive stocks/strike and aligned training factors/targets required")
    center, scale = factors.mean(axis=0), factors.std(axis=0)
    scale[scale < 128*np.finfo(float).eps*np.maximum(1., np.abs(center))] = 1.
    feature = _factor_features(factors, center, scale, degree, factor_clip)
    z = spots/strike
    stock_feature = np.column_stack((np.ones(len(z)), z-1., (z-1.)**2,
        np.maximum(z[:, None]-np.asarray(config.stock_knots), 0.)**2))
    design = (stock_feature[:, :, None]*feature[:, None, :]).reshape(len(z), -1)
    column_scale = np.sqrt(np.mean(design*design, axis=0))
    column_scale[column_scale < np.finfo(float).eps] = 1.
    x = design/column_scale
    gram, rhs = x.T@x/len(z), x.T@targets/len(z)
    eigenvalues, vectors = np.linalg.eigh(gram)
    threshold = max(float(eigenvalues[-1]), 1.)*len(eigenvalues)*np.finfo(float).eps
    active = eigenvalues > threshold
    denominator = np.maximum(eigenvalues, 0.)+ridge
    inverse = np.divide(1., denominator, out=np.zeros_like(denominator),
        where=(denominator > 0) & (active | (ridge > 0)))
    beta = ((vectors@(inverse*(vectors.T@rhs)))/column_scale).reshape(stock_feature.shape[1], feature.shape[1])
    if not np.isfinite(beta).all():
        raise ValueError("tensor premium regression produced nonfinite coefficients")
    condition = float(eigenvalues[-1]/eigenvalues[active][0]) if np.any(active) else float("inf")
    return TensorPremium(center, scale, beta, tuple(config.stock_knots), int(degree),
        float(factor_clip), int(np.sum(active)), condition)


def tensor_premium_fitter(*, degree=1, ridge=1e-4, factor_clip=4.):
    """Fit callback for value-proxy or independently tested policy fitting."""
    return partial(fit_tensor_premium, degree=degree, ridge=ridge, factor_clip=factor_clip)


__all__ = ["TensorPremium", "fit_tensor_premium", "tensor_premium_fitter"]
