"""Reporting preserves accepted/challenger identities and every saved path."""
from copy import deepcopy

import pandas as pd

from rvlab.experiments._pricing_rough_bellman_reporting import (
    BASELINE, add_bellman_reporting, curated_bellman_comparisons, latest_bellman_state)


def _result():
    policies=[BASELINE,"Bellman GH16 half","unselected final winner"]
    manifest={"baseline_policy":BASELINE,"selected_policy":BASELINE,
        "selected_challenger":policies[1],"american_pricing_ready":False,
        "stock_delta_ready":False,"new_rule_stock_delta_validated":False}
    comparison=pd.DataFrame([{"policy":p,"selected_policy":p==BASELINE,
        "selected_challenger":p==policies[1],"gap":g,"gap_se":.0001,
        "price_error_budget":g+.001,"price_allowance":.0013,"price_budget_pass":False}
        for p,g in zip(policies,(.002,.0019,.001))])
    error=pd.DataFrame([{"policy":p,"selected_policy":p==BASELINE,
        "selected_challenger":p==policies[1],"identified_policy_gain":g,
        "gain_interval_low":g-se,"gain_interval_high":g+se,
        "majority_margin":g-.001,"majority_margin_low":g-.001-se,
        "majority_margin_high":g-.001+se,"policy_majority_proved":False}
        for p,g,se in zip(policies,(0.,.0001,.001),(0.,.00015,.00015))])
    tails=pd.DataFrame([{"policy":p,"gap_p90":.003,"gap_p99":.03,"gap_maximum":.2,
        "used_for_selection":False,"acceptance_input":False} for p in policies])
    times=pd.DataFrame([{"policy":p,"stop_index":i,"fraction":.8 if i==64 else .2/64}
        for p in policies for i in range(65)])
    return {"manifest":manifest,"tables":{"comparison":comparison,"policy_error":error,
        "tail_distribution":tails,"exercise_times":times},"figures":{}}


def test_all_three_graphics_are_readonly_and_keep_training_choices():
    import matplotlib.pyplot as plt
    result=_result();before=deepcopy(result)
    rendered=add_bellman_reporting(result)
    assert len(rendered["figures"])==3 and not result["figures"]
    for name,table in result["tables"].items():pd.testing.assert_frame_equal(table,before["tables"][name])
    assert result["manifest"]==before["manifest"]
    curated=curated_bellman_comparisons(result["tables"],result["manifest"])
    assert len(curated)==3 and list(curated.gap)==[.002,.002,.0019]
    assert "unselected final winner" not in curated.policy.values
    assert "Accepted rule" in curated.iloc[1].display_role
    gains=rendered["figures"]["policy_gain_and_majority"]
    assert "paired 95%" in gains.axes[1].get_xlabel()
    labels=[label.get_text() for label in gains.axes[0].get_yticklabels()]
    assert "accepted" in labels[0] and "challenger" in labels[1]
    assert len(rendered["figures"]["gap_and_price_budget"].axes[1].patches)==3
    for figure in rendered["figures"].values():plt.close(figure)


def test_failed_or_missing_study_does_not_create_policy_results():
    failures=pd.DataFrame([{"error":"bounded operator attempt failed"}])
    result={"manifest":{"primary_complete":False},"figures":{},"tables":{"failures":failures}}
    output=add_bellman_reporting(result)
    assert not output["figures"] and output["tables"] is result["tables"]
    assert curated_bellman_comparisons(result["tables"],result["manifest"]).empty
    assert latest_bellman_state({"tables":{},"manifest":{}}).empty


def test_coincident_roles_keep_rows_but_plot_each_policy_once_with_all_exercise_mass():
    import matplotlib.pyplot as plt
    result=_result();rendered=add_bellman_reporting(result)
    exercise=rendered["figures"]["exercise_and_tails"].axes[0]
    assert len(exercise.get_yticklabels())==2
    assert abs(sum(bar.get_width() for bar in exercise.patches)-2.)<1e-14
    assert len(rendered["figures"]["exercise_and_tails"].axes[1].lines)==2
    for figure in rendered["figures"].values():plt.close(figure)


def test_zero_tail_quantile_remains_zero_in_saved_data():
    import matplotlib.pyplot as plt
    result=_result();result["tables"]["tail_distribution"].loc[0,"gap_p90"]=0.
    before=result["tables"]["tail_distribution"].copy(deep=True)
    rendered=add_bellman_reporting(result)
    pd.testing.assert_frame_equal(result["tables"]["tail_distribution"],before)
    assert len(rendered["figures"]["exercise_and_tails"].axes[1].lines[0].get_ydata())==2
    for figure in rendered["figures"].values():plt.close(figure)


def test_missing_or_ambiguous_policy_has_no_fabricated_curated_replacement():
    result=_result();data=result["tables"]["comparison"]
    result["tables"]["comparison"]=pd.concat((data,data.iloc[[1]]),ignore_index=True)
    curated=curated_bellman_comparisons(result["tables"],result["manifest"])
    assert len(curated)==2 and set(curated.policy)=={BASELINE}
    assert curated_bellman_comparisons(result["tables"],{}).empty


def test_opening_status_reads_recorded_challenger_without_final_reselection():
    result=_result();wrapped={"manifest":{"rough_bellman_policy":result["manifest"]},
        "tables":{"rough_bellman_policy_"+k:v for k,v in result["tables"].items()}}
    before=deepcopy(wrapped)
    status=latest_bellman_state(wrapped).set_index("Question")
    assert status.loc["Accepted rule · training gate","Current evidence"]==BASELINE
    assert status.loc["Designated nonbaseline challenger","Current evidence"]=="Bellman GH16 half"
    assert status.loc["Challenger paired gain · marginal 95%","Current evidence"]=="$+0.000100 [$-0.000050, $+0.000250]"
    assert status.loc["Accepted rule price gate","Current evidence"]=="$0.003000 vs $0.001300 · failed"
    assert status.loc["New-rule stock delta validated","Current evidence"]=="False"
    assert wrapped["manifest"]==before["manifest"]
    for name,table in wrapped["tables"].items():pd.testing.assert_frame_equal(table,before["tables"][name])
