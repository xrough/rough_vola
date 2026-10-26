"""Read-only graphics retain failed checks, raw rows and all frozen policies."""
from copy import deepcopy

import pandas as pd

from rvlab.experiments._pricing_rough_stopping_delta_reporting import add_stopping_delta_reporting


def test_missing_failed_campaign_does_not_fabricate_figures():
    result = {"tables": {"failures": pd.DataFrame([{"error": "independent failure"}])},
        "figures": {}, "manifest": {"primary_complete": False}}
    output = add_stopping_delta_reporting(result)
    assert not output["figures"]
    assert output["tables"] is result["tables"]
    assert output["manifest"] is result["manifest"]


def _result(with_time=True):
    estimates = pd.DataFrame([{"policy": p, "estimator": e, "bump_relative": b,
        "mean": -.40+.001*j, "ci_half_width": .08 if e == "raw" else .001,
        "delta_allowance": .004, "sampling_passed": e != "raw"}
        for p in ("original_policy", "selected_policy")
        for e in ("raw", "stopped martingale") for j, b in enumerate((.01, .005, .0025))])
    comparisons = pd.DataFrame([{"comparison": "bump refinement", "policy": p,
        "estimator": "stopped martingale", "coarse_bump": .005, "fine_bump": .0025,
        "mean": .001, "ci_half_width": .0005}
        for p in ("original_policy", "selected_policy")])
    tables = {"delta_estimates": estimates, "delta_comparisons": comparisons,
        "stopping_comparison": pd.DataFrame([{"control": "historical", "policy": "selected",
            "gap": .003, "gap_se": .0002, "price_error_budget": .005,
            "price_allowance": .001286, "price_budget_pass": False}])}
    if with_time:
        tables["delta_time_estimates"] = estimates.copy()
    return {"tables": tables, "figures": {}, "manifest": {"stock_delta_ready": False}}


def test_every_estimator_and_policy_remains_visible_without_altering_data():
    import matplotlib.pyplot as plt
    result = _result()
    old_tables = {name: frame.copy(deep=True) for name, frame in result["tables"].items()}
    old_manifest = deepcopy(result["manifest"])
    output = add_stopping_delta_reporting(result)
    assert len(output["figures"]) == 3
    for name, frame in old_tables.items():
        pd.testing.assert_frame_equal(frame, result["tables"][name])
    assert result["manifest"] == old_manifest
    assert not result["figures"]
    fig = output["figures"]["policy_delta_uncertainty"]
    assert len(fig.axes) == 4
    for axis in fig.axes[:2]:
        legend = [t.get_text() for t in axis.get_legend().get_texts()]
        assert legend == ["Raw payoff", "Stopped martingale"]
    # The failed budget is drawn above one, not clipped to the acceptance line.
    budget_axis = output["figures"]["stopping_gap_and_budget"].axes[1]
    assert budget_axis.patches[0].get_width() > 1.
    for figure in output["figures"].values():
        plt.close(figure)


def test_missing_time_comparison_is_explicit_not_replaced_by_coarse_results():
    import matplotlib.pyplot as plt
    output = add_stopping_delta_reporting(_result(with_time=False))
    axis = output["figures"]["delta_bump_and_time"].axes[1]
    assert "No completed independent time-refinement result" in axis.texts[0].get_text()
    for figure in output["figures"].values():
        plt.close(figure)
