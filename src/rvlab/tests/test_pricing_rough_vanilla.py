"""Exercise-aware public routing and conservative rough stopping diagnostics."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.pricing import (
    BlackScholesParams, DiscountCurve, DividendSchedule, ExponentialKernel,
    RoughAmericanConfig, RoughAmericanEstimate, RoughHestonParams, SpotMarket,
    evaluate_vanilla,
)
from rvlab.pricing import rough_stopping


PARAMS = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
MARKET = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
KERNEL = ExponentialKernel((1.,), (1.,), hurst=.15, horizon=1.)
REFINED = ExponentialKernel((.5, 2.), (.5, .5), hurst=.15, horizon=1.)


def _estimate(config, *, gap=0., se=0., delta=.4):
    scalar = dict(prices=2., standard_errors=se, dual_prices=2.+gap,
        dual_standard_errors=se, dual_gaps=gap, dual_gap_standard_errors=se,
        deltas=delta, delta_standard_errors=se, raw_prices=2., raw_standard_errors=.1,
        exercise_probabilities=.1, retrained_deltas=delta,
        retrained_delta_standard_errors=se)
    paired = dict(one_sided_deltas=[delta, delta], one_sided_delta_standard_errors=[se, se],
        bump_deltas=[delta, delta], bump_delta_standard_errors=[se, se])
    return RoughAmericanEstimate(**scalar, **paired, config=config, policies=(), diagnostics={})


def _run(**kwargs):
    return evaluate_vanilla(PARAMS, market=MARKET, strike=100., maturity=.1,
        exercise_style="american", **kwargs)


def test_explicit_kernel_dispatch_keeps_exercise_and_contract_order(monkeypatch):
    calls = []
    def estimate(params, market, kernel, **kwargs):
        calls.append((params, market, kernel, kwargs))
        result = _estimate(kwargs["config"])
        names = ("prices", "standard_errors", "dual_prices", "dual_standard_errors",
            "dual_gaps", "dual_gap_standard_errors", "deltas", "delta_standard_errors",
            "one_sided_deltas", "one_sided_delta_standard_errors", "bump_deltas",
            "bump_delta_standard_errors", "raw_prices", "raw_standard_errors",
            "exercise_probabilities", "retrained_deltas", "retrained_delta_standard_errors")
        return replace(result, **{name: np.broadcast_to(getattr(result, name),
            (2,)+np.asarray(getattr(result, name)).shape) for name in names})
    monkeypatch.setattr(rough_stopping, "estimate_rough_american", estimate)
    config = RoughAmericanConfig(training_paths=16, valuation_paths=8, steps=4)
    result = evaluate_vanilla(PARAMS, market=MARKET, strike=[105., 95.],
        maturity=[.2, .1], is_call=[False, True], exercise_style="european",
        kernel=KERNEL, config=config)
    assert calls[0][:3] == (PARAMS, MARKET, KERNEL)
    assert calls[0][3]["strike"] == [105., 95.]
    assert calls[0][3]["exercise_style"] == "european"
    assert result.config is config
    assert result.price_stable is None and result.delta_stable is None
    assert not result.diagnostics["american_pricing_ready"]


def test_missing_kernel_refinement_blocks_readiness(monkeypatch):
    monkeypatch.setattr(rough_stopping, "estimate_rough_american",
        lambda *a, **kw: _estimate(kw["config"]))
    result = _run(kernel=KERNEL, check_refinement=True)
    rows = result.diagnostics["checks"].set_index("axis")
    assert rows.loc["kernel", "status"] == "failed"
    assert "not supplied" in rows.loc["kernel", "error"]
    assert result.price_stable is False and result.delta_stable is False


@pytest.mark.parametrize("gap,se,accepted", [(0., 0., True), (.01, 0., False), (0., .001, False)])
def test_unchanged_stopping_and_sampling_allowances(monkeypatch, gap, se, accepted):
    monkeypatch.setattr(rough_stopping, "estimate_rough_american",
        lambda *a, **kw: _estimate(kw["config"], gap=gap, se=se))
    result = _run(kernel=KERNEL, refinement_kernel=REFINED, check_refinement=True)
    assert result.price_stable is accepted
    assert result.diagnostics["price_absolute_tolerance"] == .001
    assert result.diagnostics["price_relative_tolerance"] == 1e-4


def test_policy_retraining_delta_can_block_greek_readiness(monkeypatch):
    seeds = []
    def estimate(*args, **kwargs):
        config = kwargs["config"]
        seeds.append(config.training_seed)
        result = _estimate(config)
        return replace(result, retrained_deltas=.5) if config.retrain_bumps else result
    monkeypatch.setattr(rough_stopping, "estimate_rough_american", estimate)
    result = _run(kernel=KERNEL, refinement_kernel=REFINED, check_refinement=True)
    assert result.price_stable is True
    assert result.delta_stable is False
    assert len(set(seeds)) == 2
    assert not result.diagnostics["stock_delta_ready"]


@pytest.mark.parametrize("kwargs,error", [
    ({}, "explicit ExponentialKernel"),
    ({"kernel": KERNEL, "grid": object()}, "finite-difference grid"),
    ({"kernel": KERNEL, "refinement_kernel": KERNEL}, "must differ"),
    ({"kernel": KERNEL, "check_refinement": 1}, "boolean"),
    ({"kernel": KERNEL, "price_atol": -1}, "nonnegative"),
])
def test_invalid_rough_routing(kwargs, error):
    with pytest.raises(ValueError, match=error):
        _run(**kwargs)


def test_fd_does_not_silently_ignore_rough_arguments():
    with pytest.raises(ValueError, match="only to rough Heston"):
        evaluate_vanilla(BlackScholesParams(.2), market=MARKET, strike=100.,
            maturity=.1, exercise_style="american", kernel=KERNEL)


def test_expiration_public_backend_and_one_sided_delta():
    result = evaluate_vanilla(PARAMS, market=MARKET, strike=[95., 100., 105.],
        maturity=0., is_call=True, exercise_style="american", kernel=KERNEL,
        config=RoughAmericanConfig(training_paths=4, valuation_paths=4, steps=2))
    np.testing.assert_array_equal(result.prices, [5., 0., 0.])
    np.testing.assert_array_equal(result.standard_errors, 0.)
    np.testing.assert_array_equal(result.dual_gaps, 0.)
    np.testing.assert_allclose(result.delta_left, [1., 0., 0.])
    np.testing.assert_allclose(result.delta_right, [1., 1., 0.])


@pytest.mark.parametrize("gap,se", [(-.01, 0.), (0., -.01), (0., np.nan)])
def test_invalid_backend_diagnostics_raise(monkeypatch, gap, se):
    from rvlab.pricing import PricingError
    monkeypatch.setattr(rough_stopping, "estimate_rough_american",
        lambda *a, **kw: _estimate(kw["config"], gap=gap, se=se))
    with pytest.raises(PricingError):
        _run(kernel=KERNEL)


def test_kernel_metadata_change_is_not_a_refinement():
    with pytest.raises(ValueError, match="coefficients must differ"):
        _run(kernel=KERNEL, refinement_kernel=replace(KERNEL, horizon=2.))


def test_european_checks_do_not_claim_american_readiness(monkeypatch):
    monkeypatch.setattr(rough_stopping, "estimate_rough_american",
        lambda *a, **kw: _estimate(kw["config"]))
    result = evaluate_vanilla(PARAMS, market=MARKET, strike=100., maturity=.1,
        exercise_style="european", kernel=KERNEL, refinement_kernel=REFINED,
        check_refinement=True)
    assert result.price_stable
    assert not result.diagnostics["american_pricing_ready"]


def test_persistent_one_sided_kink_blocks_smooth_delta_readiness(monkeypatch):
    def estimate(*a, **kw):
        return replace(_estimate(kw["config"]), one_sided_deltas=[0., 1.])
    monkeypatch.setattr(rough_stopping, "estimate_rough_american", estimate)
    result = _run(kernel=KERNEL, refinement_kernel=REFINED, check_refinement=True)
    assert result.price_stable and not result.delta_stable
    assert result.diagnostics["nonsmooth_stock_sensitivity"]


def test_exact_heston_kernel_needs_no_approximation_probe(monkeypatch):
    monkeypatch.setattr(rough_stopping, "estimate_rough_american",
        lambda *a, **kw: _estimate(kw["config"]))
    result = evaluate_vanilla(replace(PARAMS, hurst=.5), market=MARKET,
        strike=100., maturity=.1, exercise_style="american",
        kernel=ExponentialKernel((0.,), (1.,), hurst=.5, horizon=1.),
        check_refinement=True)
    assert result.price_stable and result.delta_stable
    assert result.diagnostics["checks"].set_index("axis").loc["kernel", "status"] == "exact_limit"
