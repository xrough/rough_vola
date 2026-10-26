"""The larger-fit policy is measured explicitly, with paired sample identities."""
from copy import deepcopy

import numpy as np
import pytest

from rvlab.experiments import _pricing_rough_new_policy_delta as module


def arrays(paths=16):
    rng = np.random.default_rng(77)
    relative = np.array([-.01, -.005, -.0025, 0., .0025, .005, .01])
    result = {"relative_spots": relative}
    for index, name in enumerate(module.POLICIES):
        result[name + "__stops"] = np.zeros((paths, 7), dtype=int)
        for estimator, label in (("raw", "raw_prices"), ("stopped martingale", "controlled_prices")):
            common = rng.normal(size=(paths, 1))
            prices = 3. + common + 100 * relative * (-.4 + .002 * rng.normal(size=(paths, 1)))
            prices += index * (.001 + relative * .02)
            result[name + "__" + label] = prices
            for level in module.LEVELS:
                lo, hi = np.flatnonzero(relative == -level)[0], np.flatnonzero(relative == level)[0]
                result[name + "__" + f"{estimator}:{level:g}"] = (prices[:, hi] - prices[:, lo]) / (200 * level)
    return result


def test_adapter_measures_new_and_previous_rules_without_retraining(monkeypatch):
    sentinel = {name: object() for name in ("proxy", "weights", "previous_selected_policy", "selected_policy")}
    bundle = {**sentinel, "selected_name": "new",
        "manifest": {"selected_name": "new", "all_selection_training_seeds": [11, 12]}}
    captured = {}
    def replay(policies, **kwargs):
        captured.update(policies=policies, kwargs=kwargs)
        return {"diagnostics": {"stopping_times_held_fixed": False}}
    monkeypatch.setattr(module, "frozen_policy_martingale_delta_samples", replay)
    result = module.new_policy_delta_samples(bundle, seed=13, paths=16, steps=128)
    assert captured["policies"]["original_policy"] is sentinel["previous_selected_policy"]
    assert captured["policies"]["selected_policy"] is sentinel["selected_policy"]
    assert captured["policies"]["weights"] is sentinel["weights"]
    assert captured["kwargs"] == dict(seed=13, paths=16, steps=128, bump_relatives=module.LEVELS)
    assert not result["diagnostics"]["optimal_value_delta_certified"]
    with pytest.raises(ValueError, match="independent"):
        module.new_policy_delta_samples(bundle, seed=12, paths=16, steps=128)
    bundle["manifest"]["selected_name"] = "other"
    with pytest.raises(ValueError, match="sealed identity"):
        module.new_policy_delta_samples(bundle, seed=13, paths=16, steps=128)


def test_paired_payoff_identity_and_full_keyset():
    sample = arrays()
    module.validate_arrays(sample, paths=16)
    estimates, comparisons = module.summarize_arrays(sample)
    assert len(estimates) == 12 and len(comparisons) == 14
    assert not estimates.optimal_delta_ready.any()
    np.testing.assert_allclose(estimates.delta_allowance, .002 + .005 * estimates["mean"].abs())
    pair = comparisons[comparisons.comparison == "selected minus original"]
    assert len(pair) == 6
    for _, row in pair.iterrows():
        key = f"{row.estimator}:{row.fine_bump:g}"
        values = sample["selected_policy__" + key] - sample["original_policy__" + key]
        assert row["mean"] == pytest.approx(values.mean())
        assert row.standard_error == pytest.approx(values.std(ddof=1) / 4)


@pytest.mark.parametrize("change", ["missing", "forged_delta", "stops", "nan", "relative"])
def test_saved_sample_validation_rejects_changed_identity(change):
    sample = arrays()
    if change == "missing": del sample["original_policy__raw:0.01"]
    if change == "forged_delta": sample["selected_policy__stopped martingale:0.0025"][0] += .01
    if change == "stops": sample["selected_policy__stops"][0, 0] = 65
    if change == "nan": sample["selected_policy__raw_prices"][0, 0] = np.nan
    if change == "relative": sample["relative_spots"][0] += .001
    with pytest.raises((ValueError, AssertionError)):
        module.validate_arrays(sample, paths=16)


def test_time_changes_use_sum_of_independent_variances_and_keep_failed_gate():
    base, _ = module.summarize_arrays(arrays())
    refined = deepcopy(base)
    refined["mean"] += .02
    compared = module.independent_time_comparison(base, refined)
    np.testing.assert_allclose(compared.standard_error, np.sqrt(2) * base.standard_error)
    np.testing.assert_allclose(compared["mean"], .02)
    assert not compared.comparison_passed.any()
    assert compared.independent_streams.all()
    assert compared.same_exercise_calendar.all()
    assert not compared.optimal_delta_ready.any()
    with pytest.raises(ValueError, match="unique"):
        module.independent_time_comparison(base.iloc[1:], refined)


def test_unchanged_tolerances_are_exact():
    assert module.TOLERANCES == dict(price_atol=.001, price_rtol=.0001,
        delta_atol=.002, delta_rtol=.005, confidence_multiplier=1.96, spread_fraction=.1)
