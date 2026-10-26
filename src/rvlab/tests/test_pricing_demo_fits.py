"""Focused tests for reproducible demonstration outcomes, not optimizer speed."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_fits as demos
from rvlab.pricing import BlackScholesParams, PricingConfig, PricingError
from rvlab.pricing import diagnostics


@pytest.fixture
def quotes():
    return demos.synthetic_surface("bs")


def fake_result(*, objective=.01, success=True, stable=True, config=demos.COARSE_CONFIG):
    numerical_checks = pd.DataFrame([{
        "check": "joint", "baseline_config": asdict(config), "config": asdict(config),
        "status": "stable" if stable else "unstable", "error": None,
        "max_price_change": 0. if stable else .1,
        "numerical_stable": stable, "elapsed_seconds": .001,
    }])
    return SimpleNamespace(
        params=BlackScholesParams(.2), optimizer_success=success,
        optimizer_status=1 if success else 0, optimizer_message="done" if success else "budget",
        nfev=3, njev=2, numerical_stable=stable, ready=success and stable,
        objective=objective, refined_objective=objective, active_bounds={},
        config=config, refined_config=replace(config,
            quadrature_nodes=2 * config.quadrature_nodes,
            fourier_cutoff=2 * config.fourier_cutoff),
        optimality=1e-8, physical_optimality=1e-8,
        optimizer_settings={"parameterization": "physical", "jacobian": "2-point"},
        pricing_evaluations=12,
        max_price_change=0. if stable else .1,
        diagnostics=SimpleNamespace(checks=numerical_checks),
    )


def patch_fit(monkeypatch, result):
    monkeypatch.setattr(demos, "calibrate_surface", lambda *args, **kwargs: result)


def test_synthetic_contracts_and_truth(quotes):
    assert len(quotes) == 18
    assert quotes.contract_id.is_unique
    assert quotes.split.value_counts().to_dict() == {"train": 12, "heldout": 6}
    np.testing.assert_array_equal(quotes.is_call, quotes.strike >= quotes.forward)
    np.testing.assert_allclose(quotes.market_price, quotes.reference_price)
    assert "bid" not in quotes and "ask" not in quotes
    assert quotes.attrs["truth"] == {"volatility": .2}
    assert quotes.attrs["reference_config"] == asdict(demos.REFERENCE_CONFIG)
    assert set(quotes.loc[quotes.split == "heldout", "strike"]) == {97., 103.}


def test_synthetic_noise_is_seeded_and_not_spread_weighted(quotes):
    noisy = demos.synthetic_surface("bs", noise_std=.005, seed=42)
    again = demos.synthetic_surface("bs", noise_std=.005, seed=42)
    pd.testing.assert_frame_equal(noisy, again)
    expected = np.random.default_rng(42).normal(0, .005, len(quotes))
    np.testing.assert_allclose(noisy.market_price - quotes.market_price, expected, atol=1e-15)
    assert "bid" not in noisy


def test_synthetic_observation_resolution_is_explicit(monkeypatch):
    seen = []
    original = demos.price

    def observe(params, **kwargs):
        seen.append(kwargs["config"])
        return original(params, **kwargs)

    monkeypatch.setattr(demos, "price", observe)
    config = PricingConfig(48, 40., 64)
    out = demos.synthetic_surface("bs", config=config)
    assert seen == [config]
    assert out.attrs["reference_config"] == asdict(config)


@pytest.mark.parametrize("kwargs", [
    {"model": "bad"}, {"noise_std": -1.}, {"noise_std": np.nan},
    {"noise_std": True}, {"seed": True}, {"seed": 2.5},
])
def test_bad_synthetic_arguments(kwargs):
    with pytest.raises(ValueError):
        demos.synthetic_surface(**kwargs)


def test_bs_recovered_prices_not_uniqueness(quotes):
    attempt = demos.fit_attempt("bs", quotes, BlackScholesParams(.3), max_nfev=30)
    assert attempt.result is not None
    assert attempt.summary["ready"]
    assert attempt.summary["test_rmse"] < 1e-7
    assert attempt.summary["train_rmse"] < 1e-7
    np.testing.assert_allclose(attempt.predictions.model_price, quotes.reference_price, atol=1e-7)
    assert attempt.summary["free_parameter_count"] == 1
    assert attempt.summary["fixed_parameter_count"] == 0
    json.dumps(attempt.summary, allow_nan=False)


def test_training_only_and_original_order(monkeypatch, quotes):
    quotes = quotes.iloc[[6, 5, 1, 12, 10, 0]].copy()
    quotes.index = [3, 3, 9, 2, 4, 8]
    seen = []

    def calibrate(training, params, **kwargs):
        seen.append(training.copy())
        np.testing.assert_array_equal(kwargs["forward"], training.forward)
        return fake_result()

    monkeypatch.setattr(demos, "calibrate_surface", calibrate)
    attempt = demos.fit_attempt("train-only", quotes, BlackScholesParams(.3))
    assert len(seen) == 1
    assert (seen[0].split == "train").all()
    assert set(seen[0].contract_id).isdisjoint(quotes.loc[quotes.split != "train", "contract_id"])
    np.testing.assert_array_equal(attempt.predictions.index, quotes.index)
    np.testing.assert_array_equal(attempt.predictions.contract_id, quotes.contract_id)


def test_spread_scaling_and_floor(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result())
    quotes["bid"] = quotes.market_price - .002
    quotes["ask"] = quotes.market_price + .002
    quotes.loc[0, "ask"] = quotes.loc[0, "market_price"] + .018
    attempt = demos.fit_attempt("weighted", quotes, BlackScholesParams(.3))
    expected = np.maximum((quotes.ask - quotes.bid) / 2, .005)
    np.testing.assert_allclose(attempt.predictions.price_scale, expected)
    np.testing.assert_allclose(attempt.predictions.residual,
                               attempt.predictions.price_residual / expected)


def test_optimizer_exhaustion_is_retained(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result(success=False))
    attempt = demos.fit_attempt("budget", quotes, BlackScholesParams(.3), max_nfev=1)
    assert attempt.result is not None
    assert attempt.summary["status"] == "exhausted"
    assert not attempt.summary["optimizer_success"]
    assert attempt.summary["numerical_stable"]
    assert not attempt.summary["ready"]
    assert attempt.summary["test_rmse"] is not None


def test_numerical_failure_is_not_zero_prices(monkeypatch, quotes):
    def fail(*args, **kwargs):
        raise PricingError("fractional Riccati diverged")

    monkeypatch.setattr(demos, "calibrate_surface", fail)
    attempt = demos.fit_attempt("bad", quotes, BlackScholesParams(.3))
    assert attempt.result is None
    assert attempt.summary["status"] == "numerical_failure"
    assert "diverged" in attempt.summary["error"]
    assert attempt.predictions.model_price.isna().all()
    assert attempt.summary["test_rmse"] is None
    assert not attempt.summary["ready"]
    json.dumps(attempt.summary, allow_nan=False)


def test_input_errors_are_not_misrepresented_as_numerics(monkeypatch, quotes):
    def bad(*args, **kwargs):
        raise ValueError("malformed observation")

    monkeypatch.setattr(demos, "calibrate_surface", bad)
    with pytest.raises(ValueError, match="malformed"):
        demos.fit_attempt("bad-input", quotes, BlackScholesParams(.3))


def test_heldout_refinement_can_block_ready(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result())
    original = diagnostics.price

    def shifted(params, **kwargs):
        values = np.asarray(original(params, **kwargs))
        if kwargs["config"].quadrature_nodes > demos.COARSE_CONFIG.quadrature_nodes:
            values = values.copy()
            values[quotes.split != "train"] += .02
        return values

    monkeypatch.setattr(diagnostics, "price", shifted)
    attempt = demos.fit_attempt("heldout-unstable", quotes, BlackScholesParams(.3))
    assert attempt.summary["calibration_ready"]
    assert not attempt.summary["numerical_stable"]
    assert not attempt.summary["ready"]
    assert attempt.summary["status"] == "unstable"
    assert attempt.summary["max_price_change"] == pytest.approx(.02)


def test_heldout_refinement_failure_keeps_coarse_and_core_result(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result())
    original = diagnostics.price

    def fail_fine(params, **kwargs):
        if kwargs["config"].quadrature_nodes > demos.COARSE_CONFIG.quadrature_nodes:
            raise PricingError("held-out fine pricing failed")
        return original(params, **kwargs)

    monkeypatch.setattr(diagnostics, "price", fail_fine)
    attempt = demos.fit_attempt("fine-failure", quotes, BlackScholesParams(.3))
    assert attempt.result is not None
    assert attempt.summary["status"] == "refinement_failure"
    assert attempt.predictions.model_price.notna().all()
    assert attempt.predictions.refined_model_price.isna().all()
    assert attempt.summary["train_rmse"] is not None
    assert attempt.summary["refined_test_rmse"] is None
    assert not attempt.summary["ready"]


def test_core_instability_cannot_be_overridden_by_stable_heldout_checks(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result(stable=False))
    attempt = demos.fit_attempt("core-unstable", quotes, BlackScholesParams(.3))
    assert not attempt.summary["calibration_numerical_stable"]
    assert not attempt.summary["numerical_stable"]
    assert not attempt.summary["ready"]
    assert attempt.summary["status"] == "unstable"
    assert attempt.summary["max_price_change"] == pytest.approx(.1)
    assert {row["scope"] for row in attempt.summary["numerical_checks"]} == {"training", "all_contracts"}


def test_core_probe_failure_keeps_params_and_error_when_all_contract_checks_pass(monkeypatch, quotes):
    result = fake_result(stable=False)
    result.diagnostics.checks.loc[0, ["error", "status"]] = ["training cutoff failed", "numerical_failure"]
    patch_fit(monkeypatch, result)
    attempt = demos.fit_attempt("core-probe-failed", quotes, BlackScholesParams(.3))
    assert attempt.result is result
    assert attempt.predictions.model_price.notna().all()
    assert attempt.summary["status"] == "refinement_failure"
    assert "training cutoff failed" in attempt.summary["error"]
    assert not attempt.summary["ready"]
    json.dumps(attempt.summary, allow_nan=False)


def test_partial_probe_failure_summary_uses_json_null_for_unavailable_changes(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result())
    original = diagnostics.price

    def fail_middle(params, **kwargs):
        if kwargs["config"].fourier_cutoff == 1.5 * demos.COARSE_CONFIG.fourier_cutoff:
            raise PricingError("middle cutoff failed")
        return original(params, **kwargs)

    monkeypatch.setattr(diagnostics, "price", fail_middle)
    attempt = demos.fit_attempt("partial-probe-failure", quotes, BlackScholesParams(.3))
    assert attempt.summary["status"] == "refinement_failure"
    failed_rows = [row for row in attempt.summary["numerical_checks"] if row["error"]]
    assert failed_rows and all(row["max_price_change"] is None for row in failed_rows)
    json.dumps(attempt.summary, allow_nan=False)


def test_final_result_config_controls_diagnostics_and_prediction_settings(monkeypatch, quotes):
    actual_config = PricingConfig(128, 200., 512)
    result = fake_result(config=actual_config)
    patch_fit(monkeypatch, result)
    seen = []
    original = demos.pricing_diagnostics

    def observe(*args, **kwargs):
        seen.append(kwargs["config"])
        return original(*args, **kwargs)

    monkeypatch.setattr(demos, "pricing_diagnostics", observe)
    attempt = demos.fit_attempt("actual-grid", quotes, BlackScholesParams(.3))
    assert seen == [actual_config]
    assert attempt.summary["initial_config"] == asdict(demos.COARSE_CONFIG)
    assert attempt.summary["final_config"] == asdict(actual_config)
    assert attempt.summary["refined_config"] == asdict(result.refined_config)
    assert attempt.summary["pricing_evaluations"] == 16
    assert attempt.summary["optimality"] == 1e-8


def test_continuation_preserves_stages_and_uses_training_contracts_only(monkeypatch, quotes):
    from rvlab.pricing import refinement

    final_config = PricingConfig(128, 200., 512)
    seen = []

    def stage(training, initial, **kwargs):
        seen.append((training.copy(), initial, kwargs["config"]))
        return fake_result(stable=kwargs["config"] == final_config, config=kwargs["config"])

    monkeypatch.setattr(refinement, "calibrate_surface", stage)
    attempt = demos.fit_attempt("continuation", quotes, BlackScholesParams(.3),
                                refine_configs=[final_config])
    assert len(seen) == 2
    assert all((rows.split == "train").all() for rows, _, _ in seen)
    assert seen[0][1] == BlackScholesParams(.3)
    assert seen[1][1] == BlackScholesParams(.2)
    assert attempt.summary["ready"]
    assert attempt.summary["total_nfev"] == 6
    assert attempt.summary["pricing_evaluations"] == 28
    assert [row["ready"] for row in attempt.summary["stages"]] == [False, True]
    assert attempt.summary["final_config"] == asdict(final_config)
    json.dumps(attempt.summary, allow_nan=False)


def test_failed_last_continuation_stage_cannot_accept_previous_params(monkeypatch, quotes):
    from rvlab.pricing import refinement

    last_config = PricingConfig(128, 200., 512)
    previous = fake_result(stable=False)

    def stage(training, initial, **kwargs):
        if kwargs["config"] == last_config:
            raise PricingError("final grid diverged")
        return previous

    monkeypatch.setattr(refinement, "calibrate_surface", stage)
    attempt = demos.fit_attempt("continuation-failed", quotes, BlackScholesParams(.3),
                                refine_configs=[last_config])
    assert attempt.result is previous
    assert attempt.summary["status"] == "refinement_failure"
    assert not attempt.summary["ready"]
    assert attempt.summary["stages"][-1]["params"] is None
    assert attempt.summary["stages"][-1]["error"] == "final grid diverged"
    json.dumps(attempt.summary, allow_nan=False)


def test_fixed_parameters_are_visible(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result())
    attempt = demos.fit_attempt("fixed", quotes, BlackScholesParams(.2),
                                bounds={"volatility": (.2, .2)})
    assert attempt.summary["free_parameter_count"] == 0
    assert attempt.summary["fixed_parameters"] == {"volatility": .2}


def test_deterministic_starts_interior_and_repeated():
    starts = demos.deterministic_starts()
    assert starts == demos.deterministic_starts()
    assert len(starts) == 3
    assert starts[0] == demos.SYNTHETIC_INITIALS["rough_heston"]
    for start in starts:
        for name, value in asdict(start).items():
            lo, hi = demos.DEMO_BOUNDS[name]
            assert lo < value < hi


def test_selection_uses_training_only_and_keeps_unstable(monkeypatch, quotes):
    patch_fit(monkeypatch, fake_result(objective=.01, success=False))
    low_train = demos.fit_attempt("best-train", quotes, BlackScholesParams(.3))
    low_train.summary.update(test_rmse=100., ready=False)
    patch_fit(monkeypatch, fake_result(objective=.02))
    low_test = demos.fit_attempt("best-test", quotes, BlackScholesParams(.3))
    low_test.summary.update(test_rmse=0., ready=True)
    assert demos.select_attempt([low_test, low_train]) is low_train
    assert demos.select_attempt([]) is None


def test_refinement_table_retains_failures(monkeypatch, quotes):
    original = demos.price
    bad = PricingConfig(24, 10, 32)

    def fail_one(params, **kwargs):
        if kwargs["config"] == bad:
            raise PricingError("insufficient settings")
        return original(params, **kwargs)

    monkeypatch.setattr(demos, "price", fail_one)
    out = demos.refinement_table(BlackScholesParams(.2), quotes,
                                [bad, demos.COARSE_CONFIG, demos.REFERENCE_CONFIG])
    assert out.status.tolist() == ["numerical_failure", "ok", "ok"]
    assert pd.isna(out.loc[0, "max_change_vs_finest"])
    assert not out.loc[0, "stable_vs_finest"]
    assert out.loc[2, "max_change_vs_finest"] == 0
    assert out.loc[2, "price_rmse"] < 1e-12


def test_failed_finest_is_not_a_successful_reference(monkeypatch, quotes):
    original = demos.price

    def fail_fine(params, **kwargs):
        if kwargs["config"] == demos.REFERENCE_CONFIG:
            raise PricingError("fine settings diverged")
        return original(params, **kwargs)

    monkeypatch.setattr(demos, "price", fail_fine)
    out = demos.refinement_table(BlackScholesParams(.2), quotes,
                                [demos.COARSE_CONFIG, demos.REFERENCE_CONFIG])
    assert out.max_change_vs_finest.isna().all()
    assert not out.stable_vs_finest.any()


@pytest.mark.parametrize("change", [
    lambda q: q.assign(split="validation"),
    lambda q: q.assign(split="secret"),
    lambda q: q.drop(columns="forward"),
    lambda q: q.assign(bid=q.market_price),
])
def test_invalid_demonstration_inputs(quotes, change):
    with pytest.raises(ValueError):
        demos.fit_attempt("invalid", change(quotes), BlackScholesParams(.3))
