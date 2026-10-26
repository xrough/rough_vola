"""Independent centering summaries retain pairing and heterogeneous streams."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_centering_statistics import (
    stopped_martingale_contrast, summarize_independent_centering_runs)


def test_optional_stopping_contrast_reconstructs_signed_predictable_increments():
    values = np.array([[0., 1., 3., 2.], [0., 2., -1., 4.], [0., -3., 2., 1.]])
    output = stopped_martingale_contrast(values, [1, 3, 0], [3, 1, 0])
    np.testing.assert_array_equal(output["signed_left_gate"], [[0, 1, 1], [0, -1, -1], [0, 0, 0]])
    np.testing.assert_array_equal(output["endpoint_contrast"], [1., -2., 0.])
    np.testing.assert_array_equal(output["gated_increment_sums"], output["endpoint_contrast"])
    assert output["max_reconstruction_error"] == 0
    assert output["path_is_sampling_unit"]


@pytest.mark.parametrize("mode", ["nonzero_start", "nonfinite", "wrong_shape", "float_stops", "past_calendar"])
def test_optional_stopping_contrast_rejects_invalid_or_unaligned_inputs(mode):
    values, original, selected = np.array([[0., 1., 2.], [0., -1., 3.]]), [1, 1], [2, 2]
    if mode == "nonzero_start": values[0, 0] = .1
    elif mode == "nonfinite": values[1, 2] = np.inf
    elif mode == "wrong_shape": values = values.ravel()
    elif mode == "float_stops": original = [1., 1.]
    elif mode == "past_calendar": selected = [3, 2]
    with pytest.raises(ValueError):
        stopped_martingale_contrast(values, original, selected)


def test_path_weighted_pool_uses_independent_heteroskedastic_strata():
    # A high-variance, long stream must not acquire an inverse-variance target.
    a, b = np.array([-1., 0., 2.]), np.array([-6., -2., 1., 3., 7.])
    result = summarize_independent_centering_runs([
        {"seed": 101, "samples": {"contrast": a}},
        {"seed": 102, "samples": {"contrast": b}}])
    row = result["pooled"].iloc[0]
    expected_se = np.sqrt(len(a)*a.var(ddof=1)+len(b)*b.var(ddof=1))/(len(a)+len(b))
    assert row.estimate == pytest.approx(np.r_[a, b].mean())
    assert row.standard_error == pytest.approx(expected_se)
    assert row.iid_pooled_standard_error == pytest.approx(np.r_[a, b].std(ddof=1)/8**.5)
    assert row.confidence_low == pytest.approx(row.estimate-1.96*expected_se)
    inverse_variance_mean = (a.mean()/a.var(ddof=1)+b.mean()/b.var(ddof=1))/(1/a.var(ddof=1)+1/b.var(ddof=1))
    assert abs(row.estimate-inverse_variance_mean) > .1
    assert not row.acceptance_input and not row.sampling_noise_confirmed
    assert result["metadata"]["path_is_sampling_unit"]
    assert not result["metadata"]["zero_mean_nonrejection_proves_centering"]


def test_paired_contrast_uncertainty_is_not_sum_of_unpaired_errors():
    common = np.arange(12, dtype=float)-5.5
    raw, controlled = 10*common+.002*common**2, 10*common+.001*common**2
    result = summarize_independent_centering_runs([
        {"seed": 110, "samples": {"raw": raw, "controlled": controlled,
            "contrast": raw-controlled}}])
    rows = result["pooled"].set_index("quantity")
    assert rows.loc["contrast", "standard_error"] == pytest.approx((raw-controlled).std(ddof=1)/12**.5)
    assert rows.loc["contrast", "standard_error"] < rows.loc["raw", "standard_error"]/1000


def test_zero_and_negative_degenerate_contrasts_are_not_clipped():
    result = summarize_independent_centering_runs([
        {"seed": 120, "samples": {"zero": [0., 0.], "negative": [-.1, -.1]}}])
    rows = result["pooled"].set_index("quantity")
    assert rows.loc["zero", "z_score"] == 0 and rows.loc["zero", "zero_consistent"]
    assert rows.loc["negative", "estimate"] == -.1
    assert rows.loc["negative", "confidence_high"] == -.1
    assert not rows.loc["negative", "zero_consistent"]
    assert np.isnan(rows.loc["negative", "z_score"])


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "overlap"), ("forbidden", "overlap"), ("nonfinite", "finite aligned"),
    ("unaligned", "finite aligned"), ("matrix", "finite aligned"),
    ("missing_quantity", "same quantities"), ("one_path", "at least two"),
    ("boolean_seed", "nonnegative integers"), ("extra_field", "exactly seed")])
def test_invalid_or_overlapping_streams_are_rejected(mutation, match):
    runs = [{"seed": 130, "samples": {"a": [1., 2.], "b": [2., 3.]}},
        {"seed": 131, "samples": {"a": [1., 2.], "b": [2., 3.]}}]
    forbidden = ()
    if mutation == "duplicate": runs[1]["seed"] = 130
    elif mutation == "forbidden": forbidden = (130,)
    elif mutation == "nonfinite": runs[1]["samples"]["a"][0] = np.inf
    elif mutation == "unaligned": runs[1]["samples"]["a"].append(3.)
    elif mutation == "matrix": runs[1]["samples"]["a"] = [[1., 2.]]
    elif mutation == "missing_quantity": del runs[1]["samples"]["a"]
    elif mutation == "one_path": runs[1]["samples"] = {"a": [1.], "b": [2.]}
    elif mutation == "boolean_seed": runs[1]["seed"] = True
    elif mutation == "extra_field": runs[1]["paths"] = 2
    with pytest.raises(ValueError, match=match):
        summarize_independent_centering_runs(runs, forbidden_seeds=forbidden)


@pytest.mark.parametrize("multiplier", [0, -1, np.inf, np.nan, True, "1.96"])
def test_confidence_multiplier_must_remain_explicit_finite_positive(multiplier):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_independent_centering_runs([
            {"seed": 140, "samples": {"contrast": [1., 2.]}}], confidence_multiplier=multiplier)
