"""Majority attribution uses paired whole paths, not an ex-post tail label."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_tail_attribution import _dominance_statistics


def _streams(extra=.001):
    streams=[]
    for shift in (0.,.01):
        gap=np.array([.01,.02,.03,.04])+shift
        lower=np.arange(1.,5.)
        gain=.5*gap+extra
        streams.append({"values":{
            "baseline":{"lower":np.column_stack((lower,lower+gain)),"upper":lower+gap},
            "challenger":{"lower":np.column_stack((lower,lower+gain)),"upper":lower+.8*gap}}})
    return streams


def test_majority_contrast_preserves_common_gap_covariance():
    result=_dominance_statistics(_streams(),1,"challenger",baseline_policy=0,
        baseline_control="baseline").set_index("contrast")
    policy=result.loc["policy lower gain"]
    majority=result.loc["policy gain minus half baseline gap"]
    assert majority["mean"]==pytest.approx(.001)
    assert majority.standard_error==pytest.approx(0.,abs=2e-16)
    assert majority.standard_error<policy.standard_error/1000
    assert majority.resolved_positive and majority.majority_contrast
    assert result.loc["dual upper decrease","resolved_positive"]
    assert not result.loc["dual decrease minus half baseline gap","resolved_positive"]
    assert (result.paths==8).all() and result.paired_whole_paths.all()
    assert not result.optimal_policy_error_identified.any()


def test_positive_component_gain_does_not_imply_a_resolved_majority():
    perturb=np.array([-.005,.005,-.005,.005])+.0001
    result=_dominance_statistics(_streams(perturb),1,"challenger",baseline_policy=0,
        baseline_control="baseline").set_index("contrast")
    majority=result.loc["policy gain minus half baseline gap"]
    assert majority["mean"]>0. and majority.interval_low<0.<majority.interval_high
    assert not majority.resolved_positive
    assert result.loc["policy lower gain","resolved_positive"]


def test_unchanged_policy_or_control_has_no_fabricated_component_gain():
    result=_dominance_statistics(_streams(),0,"baseline",baseline_policy=0,
        baseline_control="baseline").set_index("contrast")
    for key in ("policy lower gain","dual upper decrease"):
        row=result.loc[key]
        assert row["mean"]==0. and row.standard_error==0.
        assert not row.resolved_positive and not row.majority_contrast
    assert not result.resolved_positive.any()
