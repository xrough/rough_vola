"""Paired inference must retain path identity and condition on frozen policies."""
from copy import deepcopy

import numpy as np
import pytest

from rvlab.experiments._pricing_policy_comparison_stats import (
    CONTROLS, POLICIES, summarize_case,
)


def sample(n, offset=0.):
    path = np.arange(n, dtype=float)+offset
    lower = (path[:, None, None, None]+np.array([-.3, .3])[None, :, None, None]
        +np.array([0., .2, .5])[None, None, :, None])
    out = {"times": np.array([0., .04, .08]),
        "stop_indices": np.ones((n, 2, 3, 1), np.int64),
        "policy_names": np.array(POLICIES), "replica_ids": np.array([0, 1])}
    for j, control in enumerate(CONTROLS):
        out[control+"_lower"] = lower+(j*.01)*path[:, None, None, None]
        out[control+"_upper"] = np.full((n, 1), 30.)
    return out


def report(samples=None, **kwargs):
    return summarize_case([sample(4), sample(6, 2.)] if samples is None else samples,
        case_id="put_90d_k100", arm="path_matched", **kwargs)


def pick(frame, **conditions):
    selected = frame
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    assert len(selected) == 1
    return selected.iloc[0]


def moments(chunks):
    n = sum(len(x) for x in chunks)
    mean = sum(x.sum() for x in chunks)/n
    se = np.sqrt(sum(len(x)*x.var(ddof=1) for x in chunks))/n
    return mean, se


def test_weighted_stream_mean_and_se_use_all_paths_once():
    chunks = [sample(4), sample(6, 2.)]
    row = pick(report(chunks)["values"], policy="cashflow_lsm", control="raw")
    expected = [x["raw_lower"].mean(axis=1)[:, 0, 0] for x in chunks]
    mean, se = moments(expected)
    assert row["paths"] == 10
    assert row["price"] == pytest.approx(mean)
    assert row["price_se"] == pytest.approx(se)
    assert row["price_interval_low"] == pytest.approx(mean-1.96*se)
    assert row["price_interval_high"] == pytest.approx(mean+1.96*se)


def test_training_replicas_are_averaged_within_path_not_counted_as_new_paths():
    chunks = [sample(4), sample(6)]
    for chunk in chunks:
        n = len(chunk["stop_indices"])
        alternating = 3*np.where(np.arange(n) % 2, 1., -1.)
        for control in CONTROLS:
            chunk[control+"_lower"][:] = 5.
            chunk[control+"_lower"][:, 0, :, 0] += alternating[:, None]
            chunk[control+"_lower"][:, 1, :, 0] -= alternating[:, None]
    result = report(chunks)
    assert (result["values"]["paths"] == 10).all()
    assert (result["values"]["price"] == 5.).all()
    assert (result["values"]["price_se"] == 0.).all()
    assert (result["training_variation"].query("record_type == 'individual_replica'")
        ["standard_error"] > 0).all()
    assert not result["values"]["training_uncertainty_in_interval"].any()


def test_price_gate_keeps_gap_se_and_original_tolerances():
    chunks = [sample(4), sample(6, 2.)]
    for chunk in chunks:
        for control in CONTROLS:
            chunk[control+"_lower"][:] = chunk["raw_lower"]
            # Every policy has a constant pathwise gap (different by policy).
            chunk[control+"_upper"] = chunk[control+"_lower"].mean(axis=1)[:, 0]+2.
    row = pick(report(chunks)["values"], policy="cashflow_lsm", control="raw")
    assert row["gap"] == pytest.approx(2.)
    assert row["gap_se"] == pytest.approx(0., abs=1e-15)
    assert row["upper_se"] > 0
    assert row["price_error_budget"] == pytest.approx(2.+1.96*row["price_se"])
    assert row["price_allowance"] == pytest.approx(.001+1e-4*abs(row["price"]))
    assert not row["price_budget_pass"]


def test_zero_gap_constant_payoff_passes_without_special_case_or_tolerance_relaxation():
    chunks = [sample(4), sample(6)]
    for chunk in chunks:
        chunk["stop_indices"][:] = 0
        for control in CONTROLS:
            chunk[control+"_lower"][:] = 10.
            chunk[control+"_upper"][:] = 10.
    result = report(chunks)
    assert result["values"]["price_budget_pass"].all()
    assert (result["values"]["price_error_budget"] == 0.).all()
    assert (result["values"]["price_allowance"] == .002).all()
    assert result["controls"]["variance_ratio_to_raw"].isna().all()
    assert result["controls"]["variance_reduction_fraction"].isna().all()


def test_paired_policy_gain_uses_pair_covariance_and_common_upper_cancels():
    chunks = [sample(4), sample(6, 2.)]
    result = report(chunks)
    for control in CONTROLS:
        row = pick(result["paired_policy_gains"], control=control,
            candidate_policy="fourier_premium_lsm")
        assert row["estimate"] == pytest.approx(.2)
        assert row["standard_error"] == pytest.approx(0., abs=2e-16)
        assert row["same_paths_same_upper"]
        assert row["gap_reduction_equals_controlled_gain"]
        assert not row["independent_dual_improvement"]
        a = pick(result["values"], policy="cashflow_lsm", control=control)
        b = pick(result["values"], policy="fourier_premium_lsm", control=control)
        assert a["gap"]-b["gap"] == pytest.approx(row["estimate"])


def test_control_variance_and_paired_shift_preserve_same_frozen_policy():
    chunks = [sample(4), sample(6, 2.)]
    for chunk in chunks:
        chunk["european_lower"][:] = 3.
    result = report(chunks)
    row = pick(result["controls"], policy="cashflow_lsm", control="european")
    assert row["variance_ratio_to_raw"] == 0.
    assert row["variance_reduction_fraction"] == 1.
    assert row["price_se"] == 0.
    assert row["raw_price_se"] > 0
    expected = [x["european_lower"].mean(axis=1)[:, 0, 0]
        -x["raw_lower"].mean(axis=1)[:, 0, 0] for x in chunks]
    mean, se = moments(expected)
    assert row["paired_shift_from_raw"] == pytest.approx(mean)
    assert row["paired_shift_se"] == pytest.approx(se)
    assert row["same_frozen_policy"]
    assert not row["control_unbiasedness_established_by_this_test"]


def test_training_disagreement_is_conditional_paired_difference_not_population_ci():
    result = report()
    row = pick(result["training_variation"], policy="cashflow_lsm", control="raw",
        record_type="paired_replica_difference")
    assert row["estimate"] == pytest.approx(.6)
    assert row["standard_error"] == pytest.approx(0., abs=3e-16)
    assert row["reference_replica_id"] == 0
    assert row["comparison_replica_id"] == 1
    assert not row["between_training_population_uncertainty_estimated"]
    assert len(result["training_variation"]) == 27
    assert len(result["paired_policy_gains"]) == 6
    assert result["paired_policy_gains"]["pointwise_interval"].all()
    assert not result["paired_policy_gains"]["used_for_policy_selection"].any()


def test_exercise_distribution_preserves_both_replica_rules():
    chunks = [sample(4), sample(6)]
    for chunk in chunks:
        chunk["stop_indices"][:, 0] = 0
        chunk["stop_indices"][:, 1] = 2
    frame = report(chunks)["exercise"]
    avg = pick(frame, policy="cashflow_lsm", record_type="replica_average")
    assert avg["paths"] == 10
    assert avg["inception_probability"] == .5
    assert avg["early_exercise_probability"] == .5
    assert avg["expiry_probability"] == .5
    assert avg["mean_stop_years"] == pytest.approx(.04)
    dist = frame[(frame["policy"] == "cashflow_lsm")
        & (frame["record_type"] == "exercise_date_distribution")]
    np.testing.assert_array_equal(dist["probability"], [.5, 0., .5])
    assert dist["probability"].sum() == 1.


@pytest.mark.parametrize("defect", ["one_stream", "replica_ids", "calendar", "calendar_zero",
    "stops_float", "stops_range", "stops_shape", "policy_order", "replica_order",
    "missing_control", "nan_lower", "nan_upper", "upper_policy_axis", "dominance"])
def test_malformed_or_confounded_samples_raise_identifiable_errors(defect):
    chunks = [sample(4), sample(6)]
    kwargs = {}
    if defect == "one_stream": chunks = chunks[:1]
    elif defect == "replica_ids": kwargs["replica_ids"] = (0, 0)
    elif defect == "calendar": chunks[1]["times"][1] += .001
    elif defect == "calendar_zero": chunks[0]["times"][0] = .001
    elif defect == "stops_float": chunks[0]["stop_indices"] = chunks[0]["stop_indices"].astype(float)
    elif defect == "stops_range": chunks[0]["stop_indices"][0, 0, 0] = 3
    elif defect == "stops_shape": chunks[0]["stop_indices"] = chunks[0]["stop_indices"][:, :, :, 0]
    elif defect == "policy_order": chunks[0]["policy_names"] = np.array(POLICIES[::-1])
    elif defect == "replica_order": chunks[0]["replica_ids"] = np.array([1, 0])
    elif defect == "missing_control": del chunks[0]["european_lower"]
    elif defect == "nan_lower": chunks[0]["raw_lower"][0, 0, 0] = np.nan
    elif defect == "nan_upper": chunks[0]["raw_upper"][0] = np.nan
    elif defect == "upper_policy_axis": chunks[0]["raw_upper"] = np.ones((4, 3, 1))
    else: chunks[0]["raw_lower"][0, 0, 0] = 31.
    with pytest.raises(ValueError):
        report(chunks, **kwargs)


def test_summary_never_mutates_samples_or_uses_training_metrics_for_selection():
    chunks = [sample(4), sample(6)]
    before = deepcopy(chunks)
    first = report(chunks)
    second = report(chunks, training_metrics={"best_policy": "bellman_lookahead",
        "walltime": {"cashflow_lsm": 1., "bellman_lookahead": 1e6}})
    for a, b in zip(chunks, before):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    for key in first:
        assert first[key].equals(second[key])


@pytest.mark.parametrize("kwargs", [{"case_id": ""}, {"arm": ""}])
def test_explicit_study_identity_is_required(kwargs):
    options = {"case_id": "case", "arm": "arm", **kwargs}
    with pytest.raises(ValueError):
        summarize_case([sample(4), sample(6)], **options)
