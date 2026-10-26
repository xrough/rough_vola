"""Independent paired-inference checks, without pricing or fitting."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_bellman_tables import (
    continuation_diagnostics, final_statistics, selection_statistics,
)


SPECS = {"baseline": {"order": 0, "blend": 0.},
    "half": {"order": 16, "blend": .5}, "full": {"order": 16, "blend": 1.}}


def stream(seed, gain=.8):
    base = np.array([2., 2.1, 1.9, 2.05])
    change = np.array([gain, gain+.01, gain-.01, gain])
    lower = np.column_stack((base, base+.5*change, base+change))
    raw = lower+np.array([.1, -.2, .05, .1])[:, None]
    return {"seed": seed, "lower": lower, "raw": raw, "upper": base+1.,
        "stops": np.tile([64, 40, 32], (4, 1))}


def test_selection_uses_paired_lcb_and_keeps_best_challenger():
    table, selected, challenger = selection_statistics(stream(1), SPECS, seed=1)
    assert selected == challenger == "full"
    assert table.selected_policy.sum() == table.selected_challenger.sum() == 1
    assert not table.final_paths_used_for_selection.any()
    negative, selected, challenger = selection_statistics(stream(1, gain=-.2), SPECS, seed=1)
    assert selected == "baseline" and challenger == "half"
    assert negative.loc[negative.policy.eq("half"), "selection_score"].item() < 0


def test_selection_ties_follow_declared_order_without_final_reselection():
    s = stream(1)
    s["lower"][:, 2] = s["lower"][:, 1]
    _, selected, challenger = selection_statistics(s, SPECS, seed=1)
    assert selected == challenger == "half"
    final = final_statistics([stream(2), stream(3)], SPECS,
        selected_policy=selected, selected_challenger=challenger)
    assert final["comparison"].loc[final["comparison"].selected_policy, "policy"].item() == "half"


def test_common_upper_gap_change_is_exact_negative_policy_gain():
    tables = final_statistics([stream(2), stream(3)], SPECS,
        selected_policy="full", selected_challenger="full")
    paired = tables["paired_changes"]
    np.testing.assert_allclose(paired.gap_change, -paired.lower_gain, atol=1e-15)
    np.testing.assert_allclose(paired.gap_change_low, -paired.lower_gain_high, atol=1e-15)
    np.testing.assert_allclose(paired.gap_change_high, -paired.lower_gain_low, atol=1e-15)
    comparison = tables["comparison"]
    assert comparison.upper.nunique() == comparison.upper_se.nunique() == 1
    assert comparison.common_upper_identical.all()
    assert not comparison.american_pricing_ready.any()


def test_half_gap_contrast_keeps_covariance_instead_of_fixed_gap_subtraction():
    a, b = stream(2), stream(3)
    a["upper"] += np.array([0., .1, -.05, .05])
    b["upper"] += np.array([.02, -.02, .03, -.03])
    tables = final_statistics([a, b], SPECS, selected_policy="full", selected_challenger="full")
    actual = tables["policy_error"].set_index("policy").loc["full"]
    chunks = [(s["lower"][:, 2]-s["lower"][:, 0])-.5*(s["upper"]-s["lower"][:, 0]) for s in (a,b)]
    expected_se = np.sqrt(sum(len(x)*np.var(x, ddof=1) for x in chunks))/8
    assert actual.majority_margin == pytest.approx(np.concatenate(chunks).mean())
    assert actual.majority_margin_se == pytest.approx(expected_se)
    assert actual.majority_margin_low == pytest.approx(actual.majority_margin-1.96*expected_se)
    assert bool(actual.policy_majority_proved)
    assert not bool(actual.full_policy_error_identified)


def test_price_acceptance_uses_same_budget_and_no_greek_promotion():
    tables = final_statistics([stream(2), stream(3)], SPECS,
        selected_policy="full", selected_challenger="full")
    for row in tables["comparison"].itertuples():
        assert row.price_allowance == pytest.approx(.001+1e-4*abs(row.price))
        assert row.price_error_budget == pytest.approx(row.gap+1.96*(row.price_se+row.gap_se))
        assert not row.price_budget_pass and not row.new_rule_stock_delta_validated


def test_all_policy_paths_and_seedwise_results_remain_visible():
    tables = final_statistics([stream(2), stream(3)], SPECS,
        selected_policy="baseline", selected_challenger="half")
    assert len(tables["per_seed"]) == 6
    assert set(tables["per_seed"].seed) == {2, 3}
    assert (tables["per_seed"].paths == 4).all()
    stops = tables["exercise_times"].groupby("policy").agg({"paths": "sum", "fraction": "sum"})
    assert (stops.paths == 8).all()
    np.testing.assert_allclose(stops.fraction, 1.)
    tails = tables["tail_distribution"]
    assert len(tails) == 3 and not tails.used_for_selection.any() and not tails.acceptance_input.any()


def test_continuation_comparison_uses_only_explicit_computed_states():
    mask = np.zeros((4, 65), bool); mask[:, 1:3] = True
    c = np.full((4, 65, 3), np.nan)
    c[mask] = [1., 1.0005, 1.0025]
    s = {"computed_mask": mask, "continuation": c, "intrinsic": np.full((4, 65), 1.001)}
    table = continuation_diagnostics([s, s], [8,16,32])
    assert table.computed_states.tolist() == [16,16]
    np.testing.assert_allclose(table.mean_absolute_change, [.0005,.002])
    assert table.one_step_decision_change_fraction.tolist() == [0.,1.]
    assert table.within_original_price_allowance_fraction.tolist() == [1.,0.]
    assert not table.certified_operator_error_bound.any()
