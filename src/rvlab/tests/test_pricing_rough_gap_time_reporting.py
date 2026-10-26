"""Refinement displays retain historical roles, failed gates and sealed selection."""
from copy import deepcopy

import pandas as pd

from rvlab.experiments._pricing_rough_gap_time_reporting import (
    add_gap_time_reporting, latest_gap_time_state)


def _result():
    roles = [("historical 128",128,False,8192), ("historical 256",256,False,2048),
        ("new 256",256,True,8192), ("new 512",512,True,2048)]
    grid = pd.DataFrame([{"grid_role":role,"steps":steps,"new_paths":new,"paths":paths,
        "policy":policy,"estimator":estimator,"bump_relative":bump,"mean":-.4,
        "ci_half_width":.03 if estimator=="raw" else .001,"delta_allowance":.004}
        for role,steps,new,paths in roles for policy in ("original_policy","selected_policy")
        for estimator in ("raw","stopped martingale") for bump in (.005,.0025)])
    time = pd.DataFrame([{"policy":policy,"estimator":estimator,"bump_relative":bump,
        "coarse_steps":128,"fine_steps":256,"mean":.001,"ci_half_width":.001,
        "change_plus_uncertainty":.002 if bump==.005 else .005,"delta_allowance":.004,
        "comparison_passed":bump==.005}
        for policy in ("original_policy","selected_policy") for estimator in ("raw","stopped martingale")
        for bump in (.005,.0025)])
    finer=time.copy();finer["coarse_steps"]=256;finer["fine_steps"]=512
    stopping = pd.DataFrame([{"control":control,"gap":gap,"gap_se":.0001,
        "price_error_budget":budget,"price_allowance":.0013,"price_budget_pass":budget<=.0013,
        "selected_before_final":selected} for control,gap,budget,selected in
        (("Baseline",.0024,.0033,False),("Training selected",.002,.003,True),
         ("Smaller final gap",.0019,.0031,False))])
    return {"tables":{"delta_grid_estimates":grid,"delta_time_comparisons":time,
        "delta_finer_time_comparisons":finer,"stopping_comparison":stopping},
        "figures":{},"manifest":{"stock_delta_ready":False}}


def test_refinement_figures_keep_historical_pilot_failed_checks_and_selection_without_mutation():
    import matplotlib.pyplot as plt
    result=_result();before=deepcopy(result)
    output=add_gap_time_reporting(result)
    assert len(output["figures"])==3
    assert not result["figures"]
    for name,table in result["tables"].items():
        pd.testing.assert_frame_equal(table,before["tables"][name])
    assert result["manifest"]==before["manifest"]
    grid=output["figures"]["delta_grid_precision"]
    labels=[label.get_text() for label in grid.axes[0].get_xticklabels()]
    assert len(labels)==4 and "precision only" in labels[1]
    assert "new · n=8,192" in labels[2]
    assert len(grid.axes[1].get_legend().get_texts())==4
    time=output["figures"]["delta_independent_time_checks"]
    assert len(time.axes[1].patches)==4
    assert max(bar.get_width() for bar in time.axes[1].patches)>1
    stopping=output["figures"]["stopping_gap_and_budget"]
    labels=[label.get_text() for label in stopping.axes[0].get_yticklabels()]
    assert labels[1].endswith("[selected before final paths]")
    assert not labels[2].endswith("[selected before final paths]")
    for figure in output["figures"].values():plt.close(figure)


def test_partial_or_failed_campaign_does_not_fabricate_refinements():
    result={"tables":{"failures":pd.DataFrame([{"error":"independent failure"}])},
        "figures":{},"manifest":{"primary_complete":False}}
    output=add_gap_time_reporting(result)
    assert output["figures"]=={} and output["tables"] is result["tables"]
    result["tables"]["delta_grid_estimates"]=pd.DataFrame([{"mean":-.4}])
    assert not add_gap_time_reporting(result)["figures"]


def test_no_time_table_reuses_a_different_resolution():
    import matplotlib.pyplot as plt
    result=_result()
    result["tables"].pop("delta_time_comparisons")
    result["tables"].pop("delta_finer_time_comparisons")
    output=add_gap_time_reporting(result)
    assert "delta_independent_time_checks" not in output["figures"]
    for figure in output["figures"].values():plt.close(figure)


def test_latest_panel_uses_recorded_gates_and_training_selection_not_final_best():
    source=_result()
    result={"manifest":{"rough_gap_time_refinement":{"individual_delta_check_count":10,
        "new_time_delta_individual_checks_passed":False,"american_pricing_ready":False,
        "stock_delta_ready":False,"new_stopping_control_delta_validated":False}},
        "tables":{"rough_gap_time_refinement_stopping_comparison":source["tables"]["stopping_comparison"]}}
    before=deepcopy(result)
    table=latest_gap_time_state(result).set_index("Question")
    assert table.loc["Physical-time stock sensitivity","Current evidence"]=="10 correlated checks · not all passed"
    assert table.loc["Selected stopping estimator","Current evidence"]=="$0.003000 vs $0.001300 · failed"
    assert table.loc["Readiness flags","Current evidence"]=="American price: False; stock delta: False"
    assert table.loc["New price control's delta sampling","Current evidence"]=="False"
    assert result["manifest"]==before["manifest"]
    pd.testing.assert_frame_equal(result["tables"]["rough_gap_time_refinement_stopping_comparison"],
        before["tables"]["rough_gap_time_refinement_stopping_comparison"])


def test_latest_panel_missing_or_ambiguous_selection_stays_explicit():
    assert latest_gap_time_state({"manifest":{},"tables":{}}).empty
    result={"manifest":{"rough_gap_time_refinement":{}},"tables":{}}
    # An empty stage remains absent; it cannot claim completed evidence.
    assert latest_gap_time_state(result).empty
    result["manifest"]["rough_gap_time_refinement"]={"primary_complete":False}
    assert "No completed individual-check summary" in latest_gap_time_state(result).to_string(index=False)
    for flags in (None,[True,True]):
        data=pd.DataFrame([{"price_error_budget":.001,"price_allowance":.002,"price_budget_pass":True}]*2)
        if flags is not None:data["selected_before_final"]=flags
        result["tables"]["rough_gap_time_refinement_stopping_comparison"]=data
        assert "No unique completed selected-control price check" in latest_gap_time_state(result).to_string(index=False)


def test_deterministic_interpolation_figure_does_not_relabel_proxy_errors_as_price_gates():
    import matplotlib.pyplot as plt
    data=pd.DataFrame([{"date_index":date,"positive_nodes":nodes,
        "rms_interpolation_error":.004/nodes,"max_absolute_interpolation_error":.04/nodes,
        "new_acceptance_evidence":False} for date in (16,32) for nodes in (20,77,153)])
    result={"tables":{"stopping_interpolation_audit":data},"figures":{},
        "manifest":{"american_pricing_ready":False}}
    before=deepcopy(result)
    output=add_gap_time_reporting(result)
    assert list(output["figures"])==["stopping_interpolation"]
    figure=output["figures"]["stopping_interpolation"]
    assert all(len(axis.lines)==2 and axis.get_yscale()=="log" for axis in figure.axes)
    assert "no new acceptance paths" in figure._suptitle.get_text()
    pd.testing.assert_frame_equal(data,before["tables"]["stopping_interpolation_audit"])
    assert output["manifest"] is result["manifest"] and not output["manifest"]["american_pricing_ready"]
    plt.close(figure)
