"""Fresh resolutions retain independent uncertainty and the historical pilot."""
import numpy as np
import pytest

from rvlab.experiments import _pricing_rough_time_precision as study


def samples(paths=16):
    rng = np.random.default_rng(809)
    relative = study.RELATIVE_SPOTS.copy()
    result = {"relative_spots": relative}
    for index, policy in enumerate(study.POLICIES):
        result[policy + "__stops"] = np.full((paths, 5), 64, dtype=int)
        for estimator, label in (("raw", "raw_prices"), ("stopped martingale", "controlled_prices")):
            slope = -.4 + rng.normal(0., .01, (paths, 1)) - .001 * index
            prices = 10. + .1 * rng.normal(size=(paths, 1)) + 100 * relative * slope
            result[policy + "__" + label] = prices
            for level in study.LEVELS:
                lo, hi = np.flatnonzero(relative == -level)[0], np.flatnonzero(relative == level)[0]
                result[policy + "__" + f"{estimator}:{level:g}"] = (prices[:, hi] - prices[:, lo]) / (200 * level)
    return result


def test_five_stock_adapter_reapplies_sealed_policies_without_retraining(monkeypatch):
    bundle = {name: object() for name in ("proxy", "weights", "previous_selected_policy", "selected_policy")}
    bundle.update(selected_name="new", manifest={"selected_name": "new", "all_selection_training_seeds": [9]})
    captured = {}
    def replay(frozen, **kwargs):
        captured.update(frozen=frozen, kwargs=kwargs)
        return {"diagnostics": {"stopping_times_held_fixed": False}}
    monkeypatch.setattr(study, "frozen_policy_martingale_delta_samples", replay)
    result = study.new_policy_time_samples(bundle, seed=10, paths=16, steps=512)
    assert captured["kwargs"] == dict(seed=10, paths=16, steps=512, bump_relatives=(.005, .0025))
    assert captured["frozen"]["original_policy"] is bundle["previous_selected_policy"]
    assert captured["frozen"]["selected_policy"] is bundle["selected_policy"]
    assert not result["diagnostics"]["old_256_pilot_pooled"]
    assert not result["diagnostics"]["optimal_value_delta_certified"]
    with pytest.raises(ValueError, match="independent"):
        study.new_policy_time_samples(bundle, seed=9, paths=16, steps=256)
    with pytest.raises(ValueError, match="256 or 512"):
        study.new_policy_time_samples(bundle, seed=10, paths=16, steps=128)


def test_saved_payoffs_prove_all_finite_differences_and_failures_are_retained():
    arrays = samples()
    study.validate_time_arrays(arrays, paths=16)
    estimates, changes = study.summarize_time_arrays(arrays)
    assert len(arrays) == 15 and len(estimates) == 8 and len(changes) == 8
    assert not estimates.optimal_delta_ready.any()
    np.testing.assert_allclose(estimates.delta_allowance, .002 + .005 * estimates["mean"].abs())
    row = estimates.iloc[0]
    values = arrays[row.policy + "__" + f"{row.estimator}:{row.bump_relative:g}"]
    assert row.standard_error == pytest.approx(values.std(ddof=1) / 4)


@pytest.mark.parametrize("change", ["omit", "extra", "price", "raw_negative", "stops", "relative"])
def test_tampered_five_stock_arrays_are_rejected(change):
    arrays = samples()
    if change == "omit": arrays.pop("selected_policy__raw:0.005")
    if change == "extra": arrays["extra"] = np.zeros(16)
    if change == "price": arrays["selected_policy__controlled_prices"][0, 0] += .01
    if change == "raw_negative": arrays["selected_policy__raw_prices"][0, 2] = -.01
    if change == "stops": arrays["selected_policy__stops"][0, 0] = 65
    if change == "relative": arrays["relative_spots"][0] = -.01
    with pytest.raises(ValueError): study.validate_time_arrays(arrays, paths=16)


def test_distinct_laws_are_compared_with_sum_of_independent_variances():
    baseline, _ = study.summarize_time_arrays(samples())
    fine = baseline.copy()
    fine["mean"] += .01
    fine["standard_error"] *= 2
    changes = study.independent_resolution_comparison(baseline, fine, coarse_steps=256, fine_steps=512)
    np.testing.assert_allclose(changes.standard_error, np.sqrt(5) * baseline.standard_error)
    np.testing.assert_allclose(changes["mean"], .01)
    assert (changes.coarse_steps == 256).all() and (changes.fine_steps == 512).all()
    assert not changes.comparison_passed.any()
    assert changes.independent_streams.all() and not changes.optimal_delta_ready.any()
    with pytest.raises(ValueError, match="increase"):
        study.independent_resolution_comparison(baseline, fine, coarse_steps=512, fine_steps=256)
    with pytest.raises(ValueError, match="eight unique"):
        study.independent_resolution_comparison(baseline.iloc[1:], fine, coarse_steps=256, fine_steps=512)


def test_acceptance_tolerances_are_unchanged():
    assert study.TOLERANCES == dict(price_atol=.001, price_rtol=.0001,
        delta_atol=.002, delta_rtol=.005, confidence_multiplier=1.96, spread_fraction=.1)
