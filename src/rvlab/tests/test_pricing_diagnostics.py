"""Independent accuracy checks must not hide cancellation or failed probes."""

import numpy as np
import pytest

from rvlab.pricing import BlackScholesParams, HestonParams, PricingConfig, PricingError, RoughHestonParams
from rvlab.pricing import diagnostics


CONFIG = PricingConfig(16, 40., 32)
INPUTS = dict(forward=100., discount=.99, strike=[97., 100., 103.],
              maturity=.1, is_call=[False, True, True])


def check(params=None, **kwargs):
    return diagnostics.pricing_diagnostics(
        params or HestonParams(.04, .04, 1., .2, -.5),
        **INPUTS, config=CONFIG, **kwargs,
    )


def test_mixed_refinement_cancellation_does_not_establish_stability(monkeypatch):
    def cancellation(params, **kwargs):
        config = kwargs["config"]
        error = (.1 * (config.quadrature_nodes / CONFIG.quadrature_nodes - 1)
                 - .1 * (config.fourier_cutoff / CONFIG.fourier_cutoff - 1) / .25)
        return np.full(3, 2. + error)

    monkeypatch.setattr(diagnostics, "price", cancellation)
    # The former combined check would see exact cancellation.
    np.testing.assert_allclose(cancellation(None, config=CONFIG),
                               cancellation(None, config=CONFIG.refined()))
    result = check(atol=.005, rtol=0.)
    assert not result.stable
    rows = result.checks.set_index("check")
    assert rows.loc["quadrature", "max_price_change"] == pytest.approx(.1)
    assert not rows.loc["quadrature", "numerical_stable"]
    assert not rows.loc["cutoff_mid", "numerical_stable"]


def test_cumulative_tail_change_must_pass_even_when_each_increment_passes(monkeypatch):
    def slowly_decaying_tail(params, **kwargs):
        error = .008 * (kwargs["config"].fourier_cutoff / CONFIG.fourier_cutoff - 1)
        return np.full(3, 2. + error)

    monkeypatch.setattr(diagnostics, "price", slowly_decaying_tail)
    result = check(atol=.005, rtol=0.)
    rows = result.checks.set_index("check")
    assert rows.loc["cutoff_mid", "numerical_stable"]
    assert rows.loc["cutoff_far", "numerical_stable"]
    assert rows.loc["cutoff_cumulative", "max_price_change"] == pytest.approx(.008)
    assert not rows.loc["cutoff_cumulative", "numerical_stable"]
    assert not result.stable


def test_probe_failure_and_dependent_comparison_are_retained(monkeypatch):
    calls = []

    def fail_middle(params, **kwargs):
        config = kwargs["config"]
        calls.append(config)
        if config.fourier_cutoff == 60.:
            raise PricingError("fractional solver failed at a diagnostic grid")
        return np.full(3, 2.)

    monkeypatch.setattr(diagnostics, "price", fail_middle)
    result = check()
    rows = result.checks.set_index("check")
    assert not result.stable
    assert rows.loc["cutoff_mid", "status"] == "numerical_failure"
    assert rows.loc["cutoff_far", "status"] == "numerical_failure"
    assert "fractional solver failed" in rows.loc["cutoff_far", "error"]
    assert np.isnan(rows.loc["cutoff_mid", "max_price_change"])
    assert rows.loc["cutoff_cumulative", "numerical_stable"]
    np.testing.assert_array_equal(result.base_prices, 2.)
    np.testing.assert_array_equal(result.reference_prices, 2.)
    assert len(calls) == len(set(calls)) == 4
    assert result.pricing_evaluations == 4


def test_failed_reference_is_absent_and_blocks_stability(monkeypatch):
    def fail_reference(params, **kwargs):
        if kwargs["config"].riccati_steps == 2 * CONFIG.riccati_steps:
            raise PricingError("refined Riccati failed")
        return np.full(3, 2.)

    monkeypatch.setattr(diagnostics, "price", fail_reference)
    result = check(RoughHestonParams(.04, .04, 1., .2, -.5, .15))
    assert result.reference_prices is None
    assert not result.numerical_stable
    failed = result.checks.loc[result.checks.status == "numerical_failure"]
    assert failed.check.tolist() == ["riccati", "joint"]
    assert failed.max_price_change.isna().all()


@pytest.mark.parametrize("failure", ["error", "nonfinite", "shape"])
def test_invalid_base_price_raises_identifiable_failure(monkeypatch, failure):
    def fail(params, **kwargs):
        if failure == "error":
            raise PricingError("base calculation diverged")
        return np.full(3, np.nan) if failure == "nonfinite" else np.zeros((3, 1))

    monkeypatch.setattr(diagnostics, "price", fail)
    with pytest.raises(PricingError, match="base pricing diagnostic failed"):
        check()


def test_nonfinite_probe_never_becomes_a_zero_price(monkeypatch):
    def fail_refined(params, **kwargs):
        return np.full(3, np.nan if kwargs["config"].quadrature_nodes > 16 else 2.)

    monkeypatch.setattr(diagnostics, "price", fail_refined)
    result = check()
    assert not result.stable
    assert result.reference_prices is None
    assert result.checks.max_price_change.isna().all()
    assert result.checks.error.str.contains("nonfinite").all()


def test_actual_bs_broadcast_shape_order_and_scalar_prices():
    params = BlackScholesParams(.2)
    inputs = dict(forward=np.array([[100.], [103.]]), discount=.99,
                  strike=np.array([105., 95., 100.]), maturity=.1,
                  is_call=np.array([True, False, True]))
    result = diagnostics.pricing_diagnostics(params, **inputs, config=CONFIG)
    expected = diagnostics.price(params, **inputs)
    assert result.stable and result.max_price_change == 0.
    assert result.base_prices.shape == result.reference_prices.shape == (2, 3)
    np.testing.assert_array_equal(result.base_prices, expected)
    np.testing.assert_array_equal(result.reference_prices, expected)
    scalar = diagnostics.pricing_diagnostics(
        params, forward=100., discount=1., strike=100., maturity=.1,
    )
    assert scalar.base_prices.shape == scalar.reference_prices.shape == ()


@pytest.mark.parametrize("params", [BlackScholesParams(.2), HestonParams(.04, .04, 1., .2, -.5),
                                   RoughHestonParams(.04, .04, 1., .2, -.5, .5)])
def test_time_resolution_skipped_for_models_using_no_fractional_solver(monkeypatch, params):
    seen = []

    def constant(params, **kwargs):
        seen.append(kwargs["config"])
        return np.full(3, 2.)

    monkeypatch.setattr(diagnostics, "price", constant)
    result = check(params)
    assert result.stable
    assert "riccati" not in result.checks.check.tolist()
    assert {config.riccati_steps for config in seen} == {CONFIG.riccati_steps}
    assert len(seen) == len(set(seen)) == 4


def test_fractional_time_comparison_holds_fourier_controls_fixed(monkeypatch):
    monkeypatch.setattr(diagnostics, "price", lambda *args, **kwargs: np.full(3, 2.))
    result = check(RoughHestonParams(.04, .04, 1., .2, -.5, .15))
    row = result.checks.set_index("check").loc["riccati"]
    assert row.baseline_config["quadrature_nodes"] == row.config["quadrature_nodes"] == 32
    assert row.baseline_config["fourier_cutoff"] == row.config["fourier_cutoff"] == 80.
    assert row.baseline_config["riccati_steps"] == 32
    assert row.config["riccati_steps"] == 64
    assert result.reference_config == PricingConfig(32, 80., 64)
    assert result.pricing_evaluations == 5


@pytest.mark.parametrize("kwargs", [{"atol": -.1}, {"rtol": np.nan}, {"atol": True},
                                    {"rtol": 1j}, {"atol": "0.1"}])
def test_invalid_accuracy_tolerances(kwargs):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        check(**kwargs)


def test_invalid_contract_inputs_and_config_remain_caller_errors():
    with pytest.raises(TypeError, match="PricingConfig"):
        diagnostics.pricing_diagnostics(BlackScholesParams(.2), **INPUTS, config={})
    with pytest.raises(ValueError, match="strike must be finite and positive"):
        diagnostics.pricing_diagnostics(BlackScholesParams(.2), **(INPUTS | {"strike": -1.}))
