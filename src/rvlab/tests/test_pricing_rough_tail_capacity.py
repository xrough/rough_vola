"""Prefix maxima bound an entire poststop family without a causal error split."""
from copy import deepcopy
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_tail_capacity import (
    poststop_capacity, check_poststop_upper, summarize_poststop_capacity)


def chunk(capacity, gap=1.):
    capacity, gap = np.broadcast_arrays(np.asarray(capacity, float), np.asarray(gap, float))
    lower = np.full(len(capacity), 2.)
    prefix = lower + gap - capacity
    payoff = np.column_stack([np.zeros(len(lower)), prefix, lower, lower + gap])
    return poststop_capacity(payoff, np.zeros_like(payoff), np.full(len(lower), 2, dtype=int))


def test_prefix_bound_is_tighter_than_after_stop_max_and_preserves_exact_accounting():
    values = chunk(np.array([.4, .1]), np.array([1., .5]))
    np.testing.assert_allclose(values["capacity"], [.4, .1])
    np.testing.assert_allclose(values["remaining_prefix_gap"], [.6, .4])
    np.testing.assert_allclose(values["capacity"] + values["remaining_prefix_gap"], values["baseline_gap"])
    candidate = values["prefix_upper"] + np.array([.2, .0])
    decrease = check_poststop_upper(values, candidate)
    assert np.all(decrease <= values["capacity"] + 2e-11)
    with pytest.raises(ValueError, match="prefix"):
        check_poststop_upper(values, values["prefix_upper"] - .01)


@pytest.mark.parametrize("fraction,claim", [(.4, True), (.5, False), (.6, False)])
def test_family_cannot_remove_half_only_when_upper_confidence_bound_is_negative(fraction, claim):
    result = summarize_poststop_capacity([chunk(np.full(4, fraction))]).iloc[0]
    assert result.mean_ceiling == pytest.approx(fraction)
    assert result.baseline_gap == pytest.approx(1.)
    assert result.half_gap_margin == pytest.approx(fraction - .5)
    assert result.poststop_family_cannot_remove_half == claim
    assert not result.acceptance_input and not result.used_for_selection
    assert not result.causal_policy_or_dual_error_identified


def test_correlated_random_half_gap_is_propagated_in_capacity_confidence_interval():
    gap = np.array([.8, 1.2, .9, 1.1])
    cap = np.full(4, .45)
    row = summarize_poststop_capacity([chunk(cap, gap)]).iloc[0]
    paired = cap - .5 * gap
    expected = paired.std(ddof=1) / 2
    assert row.half_gap_margin_standard_error == pytest.approx(expected)
    assert row.half_gap_margin_interval_high == pytest.approx(paired.mean() + 1.96 * expected)
    assert row.half_gap_margin_interval_high > 0 and not row.poststop_family_cannot_remove_half
    # A fixed-threshold shortcut would incorrectly claim .45 < half the gap.
    assert cap.std(ddof=1) == pytest.approx(0.) and cap.mean() < .5 * gap.mean()


def test_zero_capacity_and_maturity_stop_are_handled_without_dividing_by_zero():
    payoff = np.array([[0., 2., 1.], [0., 1., 3.]])
    values = poststop_capacity(payoff, np.zeros_like(payoff), np.array([2, 2]))
    np.testing.assert_array_equal(values["capacity"], [0., 0.])
    assert summarize_poststop_capacity([values]).iloc[0].mean_ceiling == 0.
    perfect = poststop_capacity(payoff, np.zeros_like(payoff), np.array([1, 2]))
    row = summarize_poststop_capacity([perfect]).iloc[0]
    assert row.baseline_gap == row.mean_ceiling == row.half_gap_margin == 0.
    assert np.isnan(row.ceiling_gap_fraction) and not row.poststop_family_cannot_remove_half


def test_ineligible_pre_stop_payoff_cannot_raise_the_prefix_bound():
    payoff = np.array([[0., 100., 2., 3.], [0., 100., 2., 3.]])
    values = poststop_capacity(payoff, np.zeros_like(payoff), np.array([2, 2]),
        eligible=np.array([True, False, True, True]))
    np.testing.assert_array_equal(values["prefix_upper"], [2., 2.])
    np.testing.assert_array_equal(values["capacity"], [1., 1.])


def test_stratified_uncertainty_uses_all_path_vectors():
    chunks = [chunk(np.array([.2, .4, .6])), chunk(np.array([.1, .3, .5, .7, .8]))]
    row = summarize_poststop_capacity(chunks).iloc[0]
    vectors = [v["capacity"] - .5 * v["baseline_gap"] for v in chunks]
    expected_se = np.sqrt(sum(len(v) * np.var(v, ddof=1) for v in vectors)) / 8
    assert row.paths == 8 and row.streams == 2
    assert row.half_gap_margin_standard_error == pytest.approx(expected_se)
    assert row.half_gap_margin == pytest.approx(np.concatenate(vectors).mean())


@pytest.mark.parametrize("kind", ["stop", "nonfinite", "control_origin", "eligibility", "tampered_identity"])
def test_invalid_or_incomplete_capacity_evidence_is_rejected(kind):
    payoff = np.array([[0., 2., 3.], [0., 2., 3.]])
    control = np.zeros_like(payoff); stops = np.array([1, 1])
    if kind == "stop": stops[0] = 3
    if kind == "nonfinite": payoff[0, 1] = np.nan
    if kind == "control_origin": control[0, 0] = .01
    if kind == "eligibility":
        with pytest.raises(ValueError):
            poststop_capacity(payoff, control, stops, eligible=np.array([True, False, True]))
        return
    if kind == "tampered_identity":
        values = deepcopy(poststop_capacity(payoff, control, stops)); values["capacity"][0] += .01
        with pytest.raises(ValueError): summarize_poststop_capacity([values])
        return
    with pytest.raises(ValueError): poststop_capacity(payoff, control, stops)
