"""Paired predictable-compensator inference preserves bias and error budgets."""
from itertools import product

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_centering_precision_statistics import summarize_precision_runs


def _run(seed=510005042, raw=None, fine=None, coarse=None, **errors):
    raw = np.asarray([-2., -1., 1., 2.] if raw is None else raw, float)
    fine = np.asarray([-.0001, .0001, -.0001, .0001] if fine is None else fine, float)
    coarse = fine.copy() if coarse is None else np.asarray(coarse, float)
    return {"seed": seed, "samples": {"raw_contrast": raw,
        "projected_coarse": coarse, "projected_fine": fine,
        **{name: np.full(len(raw), errors.get(name, 0.))
            for name in ("quadrature_error", "tail_error", "arithmetic_error")}}}


def test_predictable_compensator_keeps_expectation_without_assuming_zero():
    shocks = np.array(list(product((-1., 1.), repeat=3)))
    original = np.where(shocks[:, 0] > 0, 1, 3)
    selected = np.where(shocks[:, 0] < 0, 2, 3)
    endpoints = np.arange(1, 4)
    gates = (selected[:, None] >= endpoints).astype(int)-(original[:, None] >= endpoints).astype(int)
    center_error = .03
    raw = np.sum(gates*(shocks+center_error), axis=1)
    projected = np.sum(gates*center_error, axis=1)
    assert raw.mean() == pytest.approx(projected.mean())
    assert projected.mean() == pytest.approx(.015)
    result = summarize_precision_runs([_run(raw=raw, fine=projected)], price_allowance=.0012860844852291117)
    rows = result["pooled"].set_index("quantity")
    assert rows.loc["projected_fine", "estimate"] == pytest.approx(.015)
    assert rows.loc["raw_minus_projected", "estimate"] == pytest.approx(0.)
    assert not result["metadata"]["projection_mean_assumed_zero"]


def test_injected_nonzero_center_error_is_preserved_and_fails_allowance():
    result = summarize_precision_runs([_run(fine=np.full(4, .01))], price_allowance=.0012860844852291117)
    budget = result["numerical_budget"].iloc[-1]
    assert budget.diagnostic_low == .01 and budget.diagnostic_high == .01
    assert not budget.diagnostic_interval_within_price_allowance
    assert not budget.american_pricing_ready and not budget.stock_delta_ready


def test_whole_path_sampling_and_numerical_budget_are_separate():
    fine = np.array([-.0001, .0001, -.0001, .0001])
    coarse = fine-np.array([1., -2., 3., -4.])*1e-6
    result = summarize_precision_runs([_run(fine=fine, coarse=coarse,
        quadrature_error=2e-6, tail_error=3e-6, arithmetic_error=4e-6)], price_allowance=.0012860844852291117)
    budget = result["numerical_budget"].iloc[-1]
    assert budget.absolute_refinement_change == pytest.approx(2.5e-6)
    assert budget.numerical_envelope == pytest.approx(11.5e-6)
    assert budget.sampling_half_width == pytest.approx(1.96*fine.std(ddof=1)/2)
    assert budget.diagnostic_half_width == pytest.approx(budget.sampling_half_width+11.5e-6)
    assert budget.diagnostic_interval_within_price_allowance
    assert not budget.numerical_budget_is_certified_bound
    assert not result["metadata"]["variance_reduction_guaranteed"]
    assert not result["metadata"]["sampling_interval_certifies_numerical_accuracy"]


def test_pool_weights_paths_and_budget_means_without_inverse_variance_selection():
    a = _run(seed=1, raw=[-2., 2.], fine=[.001, -.001], quadrature_error=1e-6)
    b = _run(seed=2, raw=[-3., 0., 3., 9.], fine=[-.003, -.001, .001, .003], quadrature_error=4e-6)
    result = summarize_precision_runs([a, b], price_allowance=.01)
    rows = result["pooled"].set_index("quantity")
    af, bf = a["samples"]["projected_fine"], b["samples"]["projected_fine"]
    expected = np.sqrt(len(af)*af.var(ddof=1)+len(bf)*bf.var(ddof=1))/6
    assert rows.loc["projected_fine", "standard_error"] == pytest.approx(expected)
    assert result["numerical_budget"].iloc[-1].quadrature_error == pytest.approx(3e-6)
    assert result["numerical_budget"].iloc[-1].paths == 6


def test_paired_raw_minus_projected_uses_same_path_covariance():
    fine = np.array([-2., -1., 1., 2.])
    raw = fine+np.array([-.01, .01, -.01, .01])
    result = summarize_precision_runs([_run(raw=raw, fine=fine)], price_allowance=.001)
    rows = result["pooled"].set_index("quantity")
    assert rows.loc["raw_minus_projected", "standard_error"] == pytest.approx((raw-fine).std(ddof=1)/2)
    assert rows.loc["raw_minus_projected", "standard_error"] < rows.loc["raw_contrast", "standard_error"]/100


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "overlap"), ("forbidden", "overlap"),
    ("nonfinite", "finite aligned"), ("unaligned", "finite aligned"),
    ("one_path", "at least two"), ("matrix", "finite aligned"),
    ("negative_allowance", "nonnegative"), ("missing", "six required"),
    ("extra_sample", "six required"), ("extra_field", "exactly seed"),
    ("boolean_seed", "nonnegative integers")])
def test_malformed_and_spent_streams_are_rejected(mutation, match):
    runs = [_run(seed=1), _run(seed=2)]
    forbidden = ()
    if mutation == "duplicate": runs[1]["seed"] = 1
    elif mutation == "forbidden": forbidden = (2,)
    elif mutation == "nonfinite": runs[1]["samples"]["projected_fine"][0] = np.inf
    elif mutation == "unaligned": runs[1]["samples"]["projected_coarse"] = [1., 2.]
    elif mutation == "one_path": runs[1]["samples"] = {k:[0.] for k in runs[1]["samples"]}
    elif mutation == "matrix": runs[1]["samples"]["projected_coarse"] = [[1., 2., 3., 4.]]
    elif mutation == "negative_allowance": runs[1]["samples"]["tail_error"][0] = -1.
    elif mutation == "missing": del runs[1]["samples"]["arithmetic_error"]
    elif mutation == "extra_sample": runs[1]["samples"]["extra"] = np.zeros(4)
    elif mutation == "extra_field": runs[1]["paths"] = 4
    elif mutation == "boolean_seed": runs[1]["seed"] = True
    with pytest.raises(ValueError, match=match):
        summarize_precision_runs(runs, price_allowance=.001, forbidden_seeds=forbidden)


@pytest.mark.parametrize("allowance", [0., -.001, np.nan, np.inf, True, ".001"])
def test_allowance_must_be_explicit_finite_positive(allowance):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_precision_runs([_run()], price_allowance=allowance)


def test_numerical_envelope_can_fail_even_with_exactly_zero_projected_mean():
    result = summarize_precision_runs([_run(fine=np.zeros(4), tail_error=.002)], price_allowance=.0012860844852291117)
    budget = result["numerical_budget"].iloc[-1]
    assert budget.sampling_half_width == 0
    assert not budget.diagnostic_interval_within_price_allowance
    assert budget.diagnostic_low == -.002 and budget.diagnostic_high == .002


def test_endpoint_interval_retains_sampling_variation_in_numerical_allowances():
    run = _run(fine=np.zeros(4))
    run["samples"]["quadrature_error"] = np.array([0., 0., 0., .0004])
    result = summarize_precision_runs([run], price_allowance=.0012860844852291117)
    budget = result["numerical_budget"].iloc[-1]
    expected_se = run["samples"]["quadrature_error"].std(ddof=1)/2
    assert budget.sampling_half_width == 0
    assert budget.diagnostic_high == pytest.approx(.0001)
    assert budget.upper_endpoint_standard_error == pytest.approx(expected_se)
    assert budget.uncertainty_endpoint_high == pytest.approx(.0001+1.96*expected_se)
    assert budget.uncertainty_endpoint_low == pytest.approx(-.0001-1.96*expected_se)
    assert budget.uncertainty_endpoint_high > budget.diagnostic_high
