"""Read-only graphics retain failed checks, raw rows and all frozen policies."""
from copy import deepcopy

import pandas as pd

from rvlab.experiments._pricing_rough_policy_validation_reporting import add_policy_validation_reporting


def test_missing_failed_campaign_does_not_fabricate_figures():
    result = {"tables": {"failures": pd.DataFrame([{"error": "independent failure"}])},
        "figures": {}, "manifest": {"primary_complete": False}}
    output = add_policy_validation_reporting(result)
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
            "price_allowance": .001286, "price_budget_pass": False,
            "selected_before_final": True}])}
    if with_time:
        tables["delta_time_estimates"] = estimates.copy()
    return {"tables": tables, "figures": {}, "manifest": {"stock_delta_ready": False}}


def test_every_estimator_and_policy_remains_visible_without_altering_data():
    import matplotlib.pyplot as plt
    result = _result()
    old_tables = {name: frame.copy(deep=True) for name, frame in result["tables"].items()}
    old_manifest = deepcopy(result["manifest"])
    output = add_policy_validation_reporting(result)
    assert len(output["figures"]) == 3
    for name, frame in old_tables.items():
        pd.testing.assert_frame_equal(frame, result["tables"][name])
    assert result["manifest"] == old_manifest
    assert not result["figures"]
    fig = output["figures"]["policy_delta_uncertainty"]
    assert len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "Previous dense policy", "New policy (ridge 1e−5)"]
    assert [t.get_text() for t in fig.axes[1].get_legend().get_texts()] == [
        "Previous dense · raw", "Previous dense · stopped control",
        "New ridge1e-5 · raw", "New ridge1e-5 · stopped control"]
    # The failed budget is drawn above one, not clipped to the acceptance line.
    budget_axis = output["figures"]["stopping_gap_and_budget"].axes[1]
    assert budget_axis.patches[0].get_width() > 1.
    assert budget_axis.get_yticklabels()[0].get_text() == "historical · selected"
    for figure in output["figures"].values():
        plt.close(figure)


def test_missing_time_comparison_is_explicit_not_replaced_by_coarse_results():
    import matplotlib.pyplot as plt
    output = add_policy_validation_reporting(_result(with_time=False))
    axis = output["figures"]["delta_bump_and_time"].axes[1]
    assert "No completed independent time-refinement result" in axis.texts[0].get_text()
    for figure in output["figures"].values():
        plt.close(figure)


def test_latest_state_panel_does_not_promote_readiness_or_choose_a_better_final_row():
    from rvlab.experiments._pricing_rough_policy_validation_reporting import latest_policy_validation_state
    result = {"manifest": {"rough_policy_validation": {
        "new_policy_name": "new frozen ridge1e-5", "individual_delta_check_count": 13,
        "new_policy_individual_delta_checks_passed": True,
        "american_pricing_ready": False, "stock_delta_ready": False}},
        "tables": {"rough_policy_validation_stopping_comparison": pd.DataFrame([
            {"selected_before_final": False, "price_error_budget": .0005, "price_allowance": .0013,
                "price_budget_pass": True},
            {"selected_before_final": True, "price_error_budget": .003, "price_allowance": .0013,
                "price_budget_pass": False}])}}
    old = deepcopy(result)
    table = latest_policy_validation_state(result).set_index("Question")
    assert table.loc["Finite-policy stock sensitivity", "Current evidence"] == "13 correlated checks · passed"
    assert table.loc["Selected stopping estimator", "Current evidence"] == "$0.003000 vs $0.001300 · failed"
    assert table.loc["Readiness flags", "Current evidence"] == "American price: False; stock delta: False"
    assert result["manifest"] == old["manifest"]
    pd.testing.assert_frame_equal(result["tables"]["rough_policy_validation_stopping_comparison"],
        old["tables"]["rough_policy_validation_stopping_comparison"])


def test_latest_state_panel_keeps_missing_or_ambiguous_results_explicit():
    from rvlab.experiments._pricing_rough_policy_validation_reporting import latest_policy_validation_state
    assert latest_policy_validation_state({"manifest": {}, "tables": {}}).empty
    result = {"manifest": {"rough_policy_validation": {"new_policy_name": "frozen"}}, "tables": {}}
    text = latest_policy_validation_state(result).to_string(index=False)
    assert "No completed individual-check summary" in text
    assert "No unique completed selected-control price check" in text
    result["tables"]["rough_policy_validation_stopping_comparison"] = pd.DataFrame([
        {"selected_before_final": True, "price_error_budget": .0005, "price_allowance": .0013,
            "price_budget_pass": True},
        {"selected_before_final": True, "price_error_budget": .0004, "price_allowance": .0013,
            "price_budget_pass": True}])
    assert "No unique completed selected-control price check" in latest_policy_validation_state(result).to_string(index=False)


def test_latest_state_panel_requires_an_explicit_prevaluation_selection_flag():
    from rvlab.experiments._pricing_rough_policy_validation_reporting import latest_policy_validation_state
    result = {"manifest": {"rough_policy_validation": {"new_policy_name": "frozen"}},
        "tables": {"rough_policy_validation_stopping_comparison": pd.DataFrame([
            {"price_error_budget": .0005, "price_allowance": .0013, "price_budget_pass": True}])}}
    assert "No unique completed selected-control price check" in latest_policy_validation_state(result).to_string(index=False)
