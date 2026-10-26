"""Numerical derivative audits preserve bounds and isolate the drift-product ridge."""

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_curve_geometry as geometry
from rvlab.pricing import PricingConfig, PricingError
from rvlab.pricing.variance_curve import HestonCurveParams, MeanReversionCurve


CONFIG = PricingConfig(48, 100., 64)
PARAMS = HestonCurveParams(.04, 1.2, .3, -.6,
                          MeanReversionCurve((.05,), (.05, 1e-35)))
BOUNDS = {"v0": (.0025, .25), "kappa": (.01, 20.), "vol_of_vol": (.01, 2.),
          "rho": (-.95, 0.), "theta_0": (0., 2.), "theta_1": (0., 2.)}


@pytest.mark.parametrize("method,tolerance", [("2-point", 2e-7), ("3-point", 2e-10)])
def test_floored_steps_recover_quadratic_jacobian_at_both_bounds(method, tolerance):
    point = np.array([1e-35, .2, .9])

    def residual(x):
        assert np.all(x >= 0) and np.all(x <= [1, 1, .9])
        return [x[0] + 2*x[1]**2 + 3*x[2], x[0]**2 + x[1] + 4*x[2]**2]

    jacobian, details = geometry._finite_difference_jacobian(
        residual, point, lower=[0, 0, 0], upper=[1, 1, .9], method=method)
    np.testing.assert_allclose(jacobian, [[1, .8, 3], [0, 1, 7.2]], atol=tolerance)
    assert details[0]["scheme"] == "forward"
    assert details[2]["scheme"] == "backward"
    assert abs(details[0]["actual_steps"][0]) > 1e-8
    np.testing.assert_array_equal(point, [1e-35, .2, .9])


def test_three_point_stencil_adapts_to_a_box_narrower_than_default_step():
    recorded = []

    def residual(x):
        recorded.append(x.copy())
        assert .4999999 <= x[0] <= .5000001
        return [2*x[0]**2]

    jacobian, details = geometry._finite_difference_jacobian(
        residual, [.5], lower=[.4999999], upper=[.5000001], method="3-point")
    assert jacobian[0, 0] == pytest.approx(2., abs=1e-8)
    assert len(recorded) == 3 and details[0]["scheme"] in {"forward", "backward"}


@pytest.mark.parametrize("kwargs", [
    {"values": [0], "lower": [0], "upper": [0], "method": "2-point"},
    {"values": [2], "lower": [0], "upper": [1], "method": "3-point"},
    {"values": [0], "lower": [0], "upper": [1], "method": "relative"},
    {"values": [np.nan], "lower": [0], "upper": [1], "method": "2-point"},
])
def test_bad_finite_difference_domains_fail_before_function_evaluation(kwargs):
    def residual(x):
        pytest.fail("invalid differentiation domain should not be evaluated")

    with pytest.raises(ValueError):
        geometry._finite_difference_jacobian(residual, **kwargs)


def test_nonfinite_and_shape_changing_residuals_are_identifiable_failures():
    with pytest.raises(PricingError, match="finite"):
        geometry._finite_difference_jacobian(lambda x: [np.nan], [.1],
            lower=[0], upper=[1], method="3-point")
    with pytest.raises(PricingError, match="shape"):
        geometry._finite_difference_jacobian(lambda x: [1] if x[0] == .1 else [1, 2],
            [.1], lower=[0], upper=[1], method="2-point")


def _quotes():
    # The held-out row deliberately has unusable inputs. It must never reach
    # the derivative audit, which diagnoses a training-only objective.
    quotes = pd.DataFrame({"strike": 100 + np.arange(7)*.0001, "maturity": .08,
        "is_call": True, "forward": 100., "discount": 1.,
        "bid": 2.98, "ask": 3., "split": ["train"]*6 + ["heldout"]})
    quotes.loc[6, ["strike", "bid", "ask", "forward"]] = np.nan
    quotes.index = [3, 3, 1, 7, 9, 1, 1]
    return quotes


def _analytic_prices(params, *, strike, **kwargs):
    row = np.rint((np.asarray(strike)-100)*1e4).astype(int)
    assert np.all((row >= 0) & (row < 6))
    # Every free coordinate is identifiable in this toy map. Kappa and theta
    # enter through immigration products plus an independent kappa effect,
    # so the transformed kappa derivative has a known algebraic answer.
    values = np.array([params.v0, params.kappa*params.curve.levels[0],
        params.kappa*params.curve.levels[1], .01*params.kappa**2,
        params.vol_of_vol, .1*params.rho])
    return 3 + values[row]


def test_geometry_detects_raw_relative_step_failure_and_keeps_kkt_sign(monkeypatch):
    monkeypatch.setattr(geometry, "price", _analytic_prices)
    quotes = _quotes()
    original = quotes.copy(deep=True)
    result = geometry.curve_geometry_diagnostics(PARAMS, quotes, bounds=BOUNDS, config=CONFIG)
    columns = result["tables"]["jacobian_columns"].set_index("parameter")
    row = columns.loc["theta_1"]
    assert row.three_point_scheme == "forward" and row.lower_active
    assert row.three_point_norm == pytest.approx(120., rel=1e-9)
    assert row.gradient == pytest.approx(120., rel=1e-9)
    assert row.projected_gradient == 0.
    assert columns.relative_derivative_disagreement.max() < 2e-6
    naive = result["tables"]["naive_relative_steps"].iloc[0]
    assert naive.parameter == "theta_1" and naive.status == "lost_sensitivity"
    assert naive.naive_derivative_norm == 0. and naive.floored_derivative_norm > 0
    assert not naive.used_for_optimization
    ridge = result["tables"]["ridge_summary"].iloc[0]
    assert ridge.kappa_column_norm_fixed_theta == pytest.approx(np.hypot(5., 2.4), rel=1e-9)
    assert ridge.kappa_column_norm_fixed_immigration == pytest.approx(2.4, rel=1e-9)
    summary = result["tables"]["geometry_summary"]
    assert set(summary.coordinates) == {"physical", "relative", "immigration", "relative_immigration"}
    assert summary.full_column_rank.all() and (summary["rank"] == 6).all()
    assert set(result["tables"]["jacobian_values"].row_position) == set(range(6))
    assert result["metadata"]["n_training"] == 6 and not result["metadata"]["heldout_used"]
    assert result["metadata"]["config"]["riccati_steps"] == CONFIG.riccati_steps
    pd.testing.assert_frame_equal(quotes, original)


def test_fixed_coordinate_is_excluded_and_zero_kappa_has_no_immigration_map(monkeypatch):
    monkeypatch.setattr(geometry, "price", _analytic_prices)
    bounds = {**BOUNDS, "kappa": (0., 0.), "theta_0": (.05, .05)}
    params = HestonCurveParams(.04, 0., .3, -.6, PARAMS.curve)
    result = geometry.curve_geometry_diagnostics(params, _quotes(), bounds=bounds, config=CONFIG)
    assert "kappa" not in result["metadata"]["parameters"]
    assert "theta_0" not in result["metadata"]["parameters"]
    summary = result["tables"]["geometry_summary"]
    assert set(summary.coordinates) == {"physical", "relative"}
    assert summary.condition_number.isna().all()  # zero theta derivatives imply rank deficiency
    assert not result["tables"]["ridge_summary"].iloc[0].immigration_coordinates_available


def test_numerical_pricing_failure_is_not_replaced_by_zero(monkeypatch):
    def failed(*args, **kwargs):
        raise PricingError("declared fractional solver failure")

    monkeypatch.setattr(geometry, "price", failed)
    with pytest.raises(PricingError, match="declared fractional"):
        geometry.curve_geometry_diagnostics(PARAMS, _quotes(), bounds=BOUNDS, config=CONFIG)


@pytest.mark.parametrize("kwargs,match", [
    ({"coordinate_floor": 0.}, "coordinate_floor"),
    ({"bounds": None}, "bounds"),
    ({"config": (48, 100, 64)}, "config"),
])
def test_invalid_geometry_controls(kwargs, match):
    arguments = {"bounds": BOUNDS, "config": CONFIG, **kwargs}
    with pytest.raises((ValueError, TypeError), match=match):
        geometry.curve_geometry_diagnostics(PARAMS, _quotes(), **arguments)
