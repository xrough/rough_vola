"""Majority attribution uses paired improvement-minus-random-gap contrasts."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_tail_attribution import _dominance_statistics


def stream(gap, policy_gain, dual_decrease):
    gap, policy_gain, dual_decrease = np.broadcast_arrays(
        np.asarray(gap, float), np.asarray(policy_gain, float), np.asarray(dual_decrease, float))
    assert gap.ndim == 1
    lower = np.column_stack([np.full(len(gap), 2.), 2. + policy_gain])
    chosen_lower = np.minimum(lower, (2. + gap - dual_decrease)[:, None])
    return {"values": {"baseline": {"lower": lower, "upper": 2. + gap},
        "chosen": {"lower": chosen_lower, "upper": 2. + gap - dual_decrease}}}


def collect(streams, *, policy=1, control="chosen"):
    return _dominance_statistics(streams, policy, control,
        baseline_policy=0, baseline_control="baseline").set_index("contrast")


@pytest.mark.parametrize("dominant", ["policy", "dual"])
def test_known_majority_identifies_only_the_demonstrated_source(dominant):
    p, d = (.8, .1) if dominant == "policy" else (.1, .8)
    data = collect([stream(np.ones(4), p, d), stream(np.ones(6), p, d)])
    policy = data.loc["policy gain minus half baseline gap"]
    dual = data.loc["dual decrease minus half baseline gap"]
    assert policy.resolved_positive == (dominant == "policy")
    assert dual.resolved_positive == (dominant == "dual")
    assert policy["mean"] == pytest.approx(p - .5)
    assert dual["mean"] == pytest.approx(d - .5)
    assert policy.paths == dual.paths == 10
    assert data.paired_whole_paths.all()
    assert not data.optimal_policy_error_identified.any()


def test_unresolved_majority_is_retained_even_when_improvement_is_positive():
    gains = np.array([.2, .4, .6, .8])
    data = collect([stream(np.ones(4), gains, gains)])
    for name in ("policy gain minus half baseline gap", "dual decrease minus half baseline gap"):
        row = data.loc[name]
        assert row["mean"] == pytest.approx(0.)
        assert row.interval_low < 0 < row.interval_high
        assert not row.resolved_positive
    assert data.loc["policy lower gain", "resolved_positive"]
    assert data.loc["dual upper decrease", "resolved_positive"]


def test_positive_covariance_can_resolve_majority_when_fixed_threshold_test_cannot():
    gap = np.array([.2, 1.8, .4, 1.6])
    improvement = .5 * gap + .05
    data = collect([stream(gap, np.zeros(4), improvement)])
    direct = data.loc["dual upper decrease"]
    majority = data.loc["dual decrease minus half baseline gap"]
    assert majority["mean"] == pytest.approx(.05)
    assert majority.standard_error < 1e-15
    assert majority.resolved_positive
    # Treating half the observed gap as fixed loses the covariance information.
    assert direct.interval_low < .5 * gap.mean()


def test_random_gap_uncertainty_prevents_false_majority_from_fixed_threshold():
    gap = np.array([.8, 1.2, .9, 1.1])
    improvement = np.full(4, .55)
    data = collect([stream(gap, np.zeros(4), improvement)])
    majority = data.loc["dual decrease minus half baseline gap"]
    paired = improvement - .5 * gap
    expected_se = paired.std(ddof=1) / np.sqrt(len(paired))
    assert majority.standard_error == pytest.approx(expected_se)
    assert majority.interval_low < 0 < majority.interval_high
    assert not majority.resolved_positive
    # The invalid fixed-threshold approach would declare a majority with SE0.
    assert improvement.mean() > .5 * gap.mean()
    assert improvement.std(ddof=1) == pytest.approx(0.)


def test_stream_uncertainty_uses_whole_path_paired_contrasts_and_fixed_counts():
    gaps = [np.array([.8, 1., 1.2]), np.array([.7, .9, 1.1, 1.3, 1.5])]
    gains = [np.array([.1, .2, .3]), np.array([.15, .18, .2, .21, .23])]
    decreases = [np.array([.4, .6, .7]), np.array([.3, .5, .7, .8, .9])]
    data = collect([stream(g, p, d) for g, p, d in zip(gaps, gains, decreases)])
    chunks = [d - .5 * g for d, g in zip(decreases, gaps)]
    expected_mean = np.concatenate(chunks).mean()
    expected_se = np.sqrt(sum(len(v) * np.var(v, ddof=1) for v in chunks)) / 8
    row = data.loc["dual decrease minus half baseline gap"]
    assert row["mean"] == pytest.approx(expected_mean)
    assert row.standard_error == pytest.approx(expected_se)
    assert row.interval_low == pytest.approx(expected_mean - 1.96 * expected_se)
    assert row.interval_high == pytest.approx(expected_mean + 1.96 * expected_se)
    assert row.confidence_multiplier == 1.96 and row.paths == 8


def test_baseline_fallback_produces_exact_zero_gains_and_no_majority_claim():
    data = collect([stream(np.array([.2, .4, .6, .8]), .1, .1)], policy=0, control="baseline")
    for name in ("policy lower gain", "dual upper decrease"):
        row = data.loc[name]
        assert row["mean"] == row.standard_error == row.interval_low == row.interval_high == 0.
        assert not row.resolved_positive
    for name in ("policy gain minus half baseline gap", "dual decrease minus half baseline gap"):
        assert not data.loc[name, "resolved_positive"]


def test_perfect_baseline_gap_does_not_create_an_undefined_or_spurious_majority():
    data = collect([stream(np.zeros(4), 0., 0.)], policy=0, control="baseline")
    np.testing.assert_array_equal(data["mean"], np.zeros(4))
    np.testing.assert_array_equal(data.standard_error, np.zeros(4))
    assert not data.resolved_positive.any()
