"""Tail graphics preserve paired bounds, frozen choices and every saved row."""
from copy import deepcopy

import pandas as pd

from rvlab.experiments._pricing_rough_tail_reporting import (
    add_tail_reporting, curated_tail_comparisons, latest_tail_state)


def _result():
    manifest={"selected_policy":"challenged rule","selected_control":"selected control",
        "dominant_source":"inconclusive","american_pricing_ready":False,
        "stock_delta_ready":False,"new_rule_stock_delta_validated":False}
    comparison=pd.DataFrame([{"policy":p,"control":c,"selected_policy":p==manifest["selected_policy"],
        "selected_control":c==manifest["selected_control"],"paths":8192,"gap":gap,
        "gap_se":.0001,"price_error_budget":gap+.0009,"price_allowance":.0013,
        "price_budget_pass":False} for p,c,gap in (
            ("frozen selected policy","current baseline",.002),
            ("challenged rule","current baseline",.0019),
            ("frozen selected policy","selected control",.0018),
            ("challenged rule","selected control",.0017),
            ("unselected final winner","other control",.0001))])
    dominance=pd.DataFrame([{"challenge":k,"bound_improvement":.0002,"improvement_low":.0001,
        "improvement_high":.0003,"majority_margin":-.0008,"majority_margin_low":-.001,
        "majority_margin_high":-.0006,"majority_dominance_proved":False} for k in ("dual","policy")])
    tails=comparison.iloc[[0,2,3]].loc[:,["policy","control"]].copy()
    tails=tails.assign(gap_p90=.003,gap_p99=.03,gap_maximum=.25,
        largest_decile_gap_share=.8,largest_percentile_gap_share=.3,
        used_for_selection=False,acceptance_input=False)
    spent=pd.DataFrame([{"group":label,"path_fraction":p,"gap_share":g,"acceptance_input":False}
        for label,p,g in (("winner before stop",.33,.56),("winner after stop",.07,.44),
            ("winning factors beyond3sd",.05,.21))])
    return {"tables":{"comparison":comparison,"dominance":dominance,
        "gap_distribution":tails,"spent_state_summary":spent},"figures":{},"manifest":manifest}


def test_all_four_readonly_figures_preserve_data_and_preselected_contrasts():
    import matplotlib.pyplot as plt
    result=_result();old=deepcopy(result)
    output=add_tail_reporting(result)
    assert len(output["figures"])==4 and not result["figures"]
    for name,table in result["tables"].items():pd.testing.assert_frame_equal(table,old["tables"][name])
    assert result["manifest"]==old["manifest"]
    curated=curated_tail_comparisons(result["tables"],result["manifest"])
    assert len(curated)==4 and "unselected final winner" not in curated.policy.values
    assert list(curated.gap)==[.002,.0019,.0018,.0017]
    axes=output["figures"]["component_bounds"].axes
    assert "paired 95%" in axes[1].get_xlabel()
    budget=output["figures"]["gap_and_price_budget"].axes[1]
    assert len(budget.patches)==4 and all(bar.get_width()>1 for bar in budget.patches)
    for figure in output["figures"].values():plt.close(figure)


def test_failed_or_missing_campaign_does_not_fabricate_figures():
    result={"tables":{"failures":pd.DataFrame([{"error":"numerical failure"}])},
        "figures":{},"manifest":{"primary_complete":False}}
    output=add_tail_reporting(result)
    assert not output["figures"] and output["tables"] is result["tables"]
    assert not curated_tail_comparisons(_result()["tables"],{}).shape[0]


def test_fallback_coincidences_do_not_create_new_estimates_or_pick_final_best():
    result=_result();result["manifest"].update(selected_policy="frozen selected policy",selected_control="current baseline")
    table=curated_tail_comparisons(result["tables"],result["manifest"])
    assert len(table)==4 and table.gap.nunique()==1 and table.gap.iloc[0]==.002
    assert len(set(table.display_role))==4


def test_missing_or_ambiguous_pair_has_no_synthetic_replacement():
    result=_result();data=result["tables"]["comparison"]
    result["tables"]["comparison"]=pd.concat((data,data.iloc[[3]]),ignore_index=True)
    curated=curated_tail_comparisons(result["tables"],result["manifest"])
    assert "Both training-selected choices" not in curated.display_role.values


def test_zero_tail_quantile_is_never_replaced_by_positive_data():
    import matplotlib.pyplot as plt
    result=_result();result["tables"]["gap_distribution"].loc[0,"gap_p90"]=0.
    original=result["tables"]["gap_distribution"].copy(deep=True)
    output=add_tail_reporting(result)
    pd.testing.assert_frame_equal(result["tables"]["gap_distribution"],original)
    lines=output["figures"]["residual_tail_diagnostics"].axes[0].lines
    assert len(lines[0].get_ydata())==2
    for figure in output["figures"].values():plt.close(figure)


def test_latest_panel_uses_recorded_dominance_and_explicit_frozen_selection():
    result=_result();manifest=result["manifest"]
    wrapped={"manifest":{"rough_tail_attribution":manifest},
        "tables":{"rough_tail_attribution_comparison":result["tables"]["comparison"]}}
    before=deepcopy(wrapped)
    table=latest_tail_state(wrapped).set_index("Question")
    assert table.loc["Majority attribution","Current evidence"]=="inconclusive"
    assert table.loc["Selected-pair price gate","Current evidence"]=="$0.002600 vs $0.001300 · failed"
    assert table.loc["New-rule stock delta validated","Current evidence"]=="False"
    assert table.loc["Readiness flags","Current evidence"]=="American price: False; stock delta: False"
    assert wrapped["manifest"]==before["manifest"]
    pd.testing.assert_frame_equal(wrapped["tables"]["rough_tail_attribution_comparison"],
        before["tables"]["rough_tail_attribution_comparison"])
    assert latest_tail_state({"manifest":{},"tables":{}}).empty
    wrapped["tables"]["rough_tail_attribution_comparison"]=pd.concat((result["tables"]["comparison"],
        result["tables"]["comparison"].iloc[[3]]),ignore_index=True)
    assert "No unique completed training-selected pair" in latest_tail_state(wrapped).to_string(index=False)
