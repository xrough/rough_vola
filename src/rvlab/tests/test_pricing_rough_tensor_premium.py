"""The richer proxy retains the exact piecewise-quadratic stock interface."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_tensor_premium import fit_tensor_premium, tensor_premium_fitter


@pytest.mark.parametrize("degree", [1, 2])
def test_tensor_coefficients_reconstruct_values_and_factor_curvature(degree):
    rng = np.random.default_rng(42)
    u = rng.normal(size=(800, 2))
    s = rng.uniform(60., 140., 800)
    z = s/100
    y = (1+u[:, 0])*(z-1)**2 + .3*u[:, 1]*np.maximum(z-.95, 0)**2
    config = replace(RoughAmericanConfig(), stock_knots=(.85, .95, 1.05))
    model = fit_tensor_premium(s, u, y, 100., config, degree=degree, ridge=0, factor_clip=20.)
    np.testing.assert_allclose(model.value(s, u, 100.), y, atol=2e-8)
    coeff = model.coefficients(u)
    index = np.searchsorted(config.stock_knots, z, side="right")
    chosen = coeff[np.arange(len(z)), index]
    np.testing.assert_allclose(chosen[:, 0]+z*(chosen[:, 1]+z*chosen[:, 2]), y, atol=2e-8)
    for array in (model.beta, model.center, model.scale):
        assert not array.flags.writeable


def test_tensor_broadcasting_and_deterministic_factor():
    rng = np.random.default_rng(42)
    u = np.column_stack((rng.normal(size=200), np.zeros(200)))
    s = rng.uniform(70., 130., 200)
    model = tensor_premium_fitter()(s, u, np.sin(s/100), 100., RoughAmericanConfig())
    points = np.column_stack((s, s+1))
    assert model.value(points, u, 100).shape == points.shape
    assert np.isfinite(model.coefficients(np.zeros((3, 2, 2)))).all()


@pytest.mark.parametrize("kwargs", [{"degree": 3}, {"degree": True}, {"ridge": -1.},
    {"factor_clip": 0.}, {"factor_clip": float("inf")}])
def test_tensor_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        fit_tensor_premium(np.ones(4), np.zeros((4, 2)), np.ones(4), 1., RoughAmericanConfig(), **kwargs)
