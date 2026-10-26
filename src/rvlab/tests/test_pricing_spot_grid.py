"""Spot-mesh regressions for near-coincident payoff and observation nodes."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.pricing import (
    DiscountCurve, DividendSchedule, FDConfig, HestonParams, PricingConfig,
    SpotMarket, evaluate_vanilla, heston_price,
)
from rvlab.pricing._vanilla_common import _insert_spot_nodes, build_grid


def test_required_spot_nodes_replace_nearby_base_nodes_without_losing_strikes():
    base = np.arange(5., dtype=float)
    required = np.array([1.+1e-9, 1.+2e-9, 2.3])
    actual = _insert_spot_nodes(base, required)
    assert set(required).issubset(actual)
    assert actual[0] == base[0] and actual[-1] == base[-1]
    assert 1. not in actual
    assert 2. in actual  # Outside the quarter-cell exclusion.
    assert np.all(np.diff(actual) > 0)
    # Distinct, very close quoted strikes remain distinct structural inputs.
    assert len(actual[np.abs(actual-1.) < 1e-8]) == 2


def test_domain_expansion_preserves_inserted_inner_spot_mesh_exactly():
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    strikes = np.array([94., 99., 101., 106.])
    config = FDConfig(80, 24, 60)
    base = build_grid(market, strikes, config)
    larger = build_grid(market, strikes, config.refined("domain"))
    np.testing.assert_array_equal(larger.spot[:len(base.spot)], base.spot)
    assert set(np.r_[market.spot, strikes]).issubset(base.spot)
    # Neither model coefficients nor variance resolution remesh stock nodes.
    changed_variance = build_grid(market, strikes, replace(config, variance_steps=48))
    np.testing.assert_array_equal(base.spot, changed_variance.spot)


@pytest.mark.parametrize("steps,concentration", [(160, .1), (240, .05)])
def test_short_tenor_heston_spot_mesh_agrees_with_independent_fourier(steps, concentration):
    # These six exact strikes used to fall near old sinh-grid nodes. Inserting
    # both made tiny terminal cells and left >$.003 of spot bias at 160 nodes.
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    params = HestonParams(.04, .045, 1.5, .3, -.6)
    strikes = np.array([94., 97., 99., 101., 103., 106.])
    calls = strikes >= market.spot
    maturity = 14/365
    numerical = evaluate_vanilla(params, market=market, strike=strikes,
        maturity=maturity, is_call=calls, exercise_style="european",
        config=FDConfig(steps, 64, 200, spot_concentration=concentration))
    reference = heston_price(params, forward=market.spot*np.exp(.04*maturity),
        discount=np.exp(-.04*maturity), strike=strikes, maturity=maturity,
        is_call=calls, config=PricingConfig(512, 500., 1024))
    np.testing.assert_allclose(numerical.prices, reference, atol=.0008, rtol=0)


@pytest.mark.parametrize("width", [0., -1., np.nan, True])
def test_spot_concentration_requires_positive_finite_real(width):
    with pytest.raises(ValueError, match="spot_concentration"):
        FDConfig(spot_concentration=width)


def test_spot_concentration_is_explicit_and_preserved_by_refinement():
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    config = FDConfig(160, 48, 180, spot_concentration=.05)
    assert config.refined("spot").spot_concentration == .05
    assert config.refined("domain").spot_concentration == .05
    narrow = build_grid(market, [94., 99., 101., 106.], config)
    wide = build_grid(market, [94., 99., 101., 106.], replace(config, spot_concentration=.1))
    assert not np.array_equal(narrow.spot, wide.spot)
    larger = build_grid(market, [94., 99., 101., 106.], config.refined("domain"))
    np.testing.assert_array_equal(larger.spot[:len(narrow.spot)], narrow.spot)
