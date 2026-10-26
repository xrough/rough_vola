"""Bounded warm-start continuation and honest retention of failed stages."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.pricing import (
    BlackScholesParams, HestonParams, PricingConfig, PricingError, heston_price,
)
from rvlab.pricing import refinement


CONFIGS = (PricingConfig(64, 60., 128), PricingConfig(128, 300., 256),
           PricingConfig(192, 600., 512))
INITIAL = BlackScholesParams(.3)
QUOTES = pd.DataFrame({"strike": [100.], "maturity": [.1],
                       "is_call": [True], "market_price": [2.]})


def _returned(volatility, *, ready, nfev):
    return SimpleNamespace(params=BlackScholesParams(volatility), ready=ready, nfev=nfev)


def test_continuation_warm_starts_and_stops_at_first_ready_stage(monkeypatch):
    first = _returned(.24, ready=False, nfev=7)
    second = _returned(.22, ready=True, nfev=4)
    calls = []

    def calibrate(quotes, params, **kwargs):
        calls.append((quotes, params, kwargs))
        return (first, second)[len(calls) - 1]

    monkeypatch.setattr(refinement, "calibrate_surface", calibrate)
    bounds = {"volatility": (.01, .5)}
    result = refinement.calibrate_surface_refined(
        QUOTES, INITIAL, configs=CONFIGS, forward=100., discount=.99,
        bounds=bounds, max_nfev=11, price_scale_floor=.02,
        stability_atol=.001, stability_rtol=1e-5,
    )
    assert result.ready and result.result is second
    assert len(result.stages) == len(calls) == 2
    assert calls[0][1] is INITIAL and calls[1][1] is first.params
    assert [call[2]["config"] for call in calls] == list(CONFIGS[:2])
    for quotes, _params, arguments in calls:
        assert quotes is QUOTES
        assert arguments["bounds"] is bounds
        assert arguments["max_nfev"] == 11
        assert arguments["price_scale_floor"] == .02
        assert arguments["stability_atol"] == .001
        assert arguments["stability_rtol"] == 1e-5
    assert result.total_nfev == 11
    assert all(stage.error is None for stage in result.stages)
    assert result.elapsed_seconds >= 0
    with pytest.raises(FrozenInstanceError):
        result.stages = ()
    with pytest.raises(FrozenInstanceError):
        result.stages[0].config = CONFIGS[2]


def test_failed_initial_stage_retries_original_start_and_preserves_exception(monkeypatch):
    failure = PricingError("fractional grid diverged")
    final = _returned(.22, ready=True, nfev=3)
    starts = []

    def calibrate(_quotes, params, **_kwargs):
        starts.append(params)
        if len(starts) == 1:
            raise failure
        return final

    monkeypatch.setattr(refinement, "calibrate_surface", calibrate)
    result = refinement.calibrate_surface_refined(
        QUOTES, INITIAL, configs=CONFIGS, forward=100., discount=.99,
    )
    assert starts == [INITIAL, INITIAL]
    assert result.stages[0].error is failure
    assert result.stages[0].result is None
    assert result.ready and result.result is final
    assert result.total_nfev == 3


def test_failed_later_stage_keeps_last_returned_parameters_and_diagnostics(monkeypatch):
    first = _returned(.24, ready=False, nfev=5)
    final = _returned(.22, ready=False, nfev=2)
    failure = PricingError("tail refinement failed")
    starts = []

    def calibrate(_quotes, params, **_kwargs):
        starts.append(params)
        if len(starts) == 1:
            return first
        if len(starts) == 2:
            raise failure
        return final

    monkeypatch.setattr(refinement, "calibrate_surface", calibrate)
    result = refinement.calibrate_surface_refined(
        QUOTES, INITIAL, configs=CONFIGS, forward=100., discount=.99,
    )
    assert starts == [INITIAL, first.params, first.params]
    assert result.stages[1].error is failure
    assert result.stages[1].initial_params is first.params
    assert result.result is final and not result.ready
    assert result.total_nfev == 7


def test_final_failure_cannot_accept_an_earlier_returned_result(monkeypatch):
    previous = _returned(.24, ready=False, nfev=6)
    calls = []

    def calibrate(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            return previous
        raise PricingError("finer grid failed")

    monkeypatch.setattr(refinement, "calibrate_surface", calibrate)
    result = refinement.calibrate_surface_refined(
        QUOTES, INITIAL, configs=CONFIGS[:2], forward=100., discount=.99,
    )
    assert result.result is previous
    assert not result.ready
    assert result.stages[-1].result is None


def test_all_failed_stages_have_no_result_or_invented_evaluation_counts(monkeypatch):
    def calibrate(*_args, **_kwargs):
        raise PricingError("no valid prices")

    monkeypatch.setattr(refinement, "calibrate_surface", calibrate)
    result = refinement.calibrate_surface_refined(
        QUOTES, INITIAL, configs=CONFIGS, forward=100., discount=.99,
    )
    assert len(result.stages) == 3
    assert result.result is None and not result.ready
    assert result.total_nfev == 0
    assert all(stage.error is not None and stage.result is None for stage in result.stages)


@pytest.mark.parametrize("configs", [[], (), None, "settings", {}, [CONFIGS[0], None]])
def test_invalid_grid_sequences_are_rejected(configs):
    with pytest.raises(ValueError, match="configs"):
        refinement.calibrate_surface_refined(
            QUOTES, INITIAL, configs=configs, forward=100., discount=.99,
        )


@pytest.mark.parametrize("failure", [ValueError("malformed rows"), TypeError("wrong type")])
def test_input_errors_propagate_instead_of_becoming_numerical_failures(monkeypatch, failure):
    def calibrate(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(refinement, "calibrate_surface", calibrate)
    with pytest.raises(type(failure), match=str(failure)):
        refinement.calibrate_surface_refined(
            QUOTES, INITIAL, configs=CONFIGS, forward=100., discount=.99,
        )


def test_actual_optimizer_exhaustion_retains_all_declared_stages():
    result = refinement.calibrate_surface_refined(
        QUOTES, INITIAL, configs=CONFIGS[:2], forward=100., discount=.99,
        max_nfev=1,
    )
    assert len(result.stages) == 2
    assert not result.ready
    assert result.total_nfev == 2
    assert all(not stage.result.optimizer_success for stage in result.stages)


def test_short_dated_heston_is_recalibrated_after_tail_refinement():
    truth = HestonParams(.019978211577066824, .25, .16949918021056282,
                         .6, -.8331920907620385)
    strike = np.tile([614., 620., 626., 639., 645., 651.], 3)
    maturity = np.repeat([15. / 365., 22. / 365., 29. / 365.], 6)
    quotes = pd.DataFrame({"strike": strike, "maturity": maturity,
                           "is_call": strike > 633.})
    quotes["market_price"] = heston_price(
        truth, forward=633., discount=.998, strike=strike, maturity=maturity,
        is_call=quotes.is_call.to_numpy(), config=PricingConfig(192, 1200., 256),
    )
    fixed = {name: (value, value) for name, value in vars(truth).items() if name != "v0"}
    initial = HestonParams(.025, truth.theta, truth.kappa, truth.vol_of_vol, truth.rho)
    result = refinement.calibrate_surface_refined(
        quotes, initial, configs=(CONFIGS[0], CONFIGS[-1]),
        forward=633., discount=.998, bounds=fixed, max_nfev=30,
        stability_atol=1e-5, stability_rtol=0.,
    )
    assert len(result.stages) == 2
    assert result.stages[0].result.optimizer_success
    assert not result.stages[0].result.numerical_stable
    assert result.ready
    assert result.result.price_rmse < 1e-6
    assert result.result.params.v0 == pytest.approx(truth.v0, abs=1e-8)
    assert result.stages[1].initial_params is result.stages[0].result.params
