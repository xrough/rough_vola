"""Price-budget diagnosis retains additive uncertainty and whole-path strata."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_price_budget_tables import (
    statistics,price_budget,summarize_price_streams,diagnose_spent_budget)


def test_budget_attributes_failure_to_sampling_without_identifying_policy_error():
    low=np.array([2.8,2.9,2.8,2.9]); high=low+.001
    row=price_budget([low],[high],label="baseline",role="spent")
    assert row["gap"]<row["price_allowance"]<row["price_error_budget"]
    assert row["price_error_budget"]==pytest.approx(row["gap"]+1.96*(row["price_se"]+row["gap_se"]))
    d=diagnose_spent_budget(row)
    assert d["sampling_causes_recorded_budget_failure"]
    assert d["projected_paths_with_unchanged_variance"]>len(low)
    assert not d["optimal_policy_error_identified"]
    assert "no new acceptance" in d["role"]


def test_common_shift_can_reduce_mc_error_while_leaving_gap_identical():
    streams=[]
    for seed in (31,42):
        shock=np.array([-.01,.01,-.01,.01])
        lower=2.86+shock; upper=lower+np.array([.0002,.0004,.0006,.0008])
        streams.append({"seed":seed,"values":{
            "unchanged control":{"lower":lower,"upper":upper},
            "shared control":{"lower":lower-shock,"upper":upper-shock}}})
    tables=summarize_price_streams(streams)
    base,new=tables["price_comparison"].iloc[0],tables["price_comparison"].iloc[1]
    assert new.gap==pytest.approx(base.gap)
    assert new.price_se<base.price_se and new.price_budget_pass and not base.price_budget_pass
    assert new.price_variance_ratio_vs_baseline<1
    assert len(tables["budget_decomposition"])==6 and len(tables["per_seed"])==4
    assert not new.american_pricing_ready and not new.stock_delta_ready
    change=tables["paired_changes"].query('primary and metric == "gap change"').iloc[0]
    assert abs(change.estimate)<1e-13 and change.half_width<1e-13


def test_shared_control_gap_drift_rejected():
    with pytest.raises(ValueError,match="pathwise stopping gap"):
        summarize_price_streams([{"seed":1,"values":{
            "unchanged control":{"lower":np.ones(3),"upper":np.ones(3)+.01},
            "shared control":{"lower":np.ones(3),"upper":np.ones(3)+.02}}}])


def test_stratified_uncertainty_does_not_treat_distinct_seed_means_as_paired_paths():
    r=statistics([np.array([0.,2.]),np.array([10.,12.])])
    assert r["estimate"]==6 and r["standard_error"]==pytest.approx(np.sqrt(8)/4)


@pytest.mark.parametrize("chunks",[[],[np.array([1.])],[np.array([1.,np.nan])],
    [np.zeros((2,2))]])
def test_malformed_or_deleted_path_vectors_rejected(chunks):
    with pytest.raises(ValueError):statistics(chunks)


def test_duplicate_seeds_and_endpoint_misalignment_rejected():
    with pytest.raises(ValueError,match="Unique"):
        summarize_price_streams([{"seed":1},{"seed":1}])
    with pytest.raises(ValueError,match="aligned"):
        price_budget([np.ones(3)],[np.ones(2)],label="x",role="x")
    with pytest.raises(ValueError,match="ordering"):
        price_budget([np.ones(3)],[np.zeros(3)],label="x",role="x")
