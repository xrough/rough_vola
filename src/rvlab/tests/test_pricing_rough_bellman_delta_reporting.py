"""Sensitivity figures preserve recorded failures and distinct time-law pools."""
from copy import deepcopy

import pandas as pd

from rvlab.experiments._pricing_rough_bellman_delta_reporting import (
    add_bellman_delta_reporting, bellman_delta_refinement_summary, bellman_delta_sampling_summary,
    bellman_gap_summary, bellman_gap_selected_changes)


def _result():
    sampling=[];bumps=[];times=[]
    for steps,paths in ((128,4096),(256,2048)):
        for policy in ("baseline","bellman"):
            for estimator in ("raw","controlled"):
                for bump in (.005,.0025):
                    value=-.4+(policy=="bellman")*.001+(steps==256)*.0001
                    width=.04 if estimator=="raw" else .001
                    sampling.append({"policy":policy,"estimator":estimator,"physical_steps":steps,
                        "paths":paths,"bump_relative":bump,"estimate":value,"half_width":width,
                        "interval_low":value-width,"interval_high":value+width,
                        "delta_allowance":.004,"sampling_pass":width<=.004,"policy_operator_steps":128})
                change=.0001 if steps==128 else .004
                width=.0001 if steps==128 else .001
                bumps.append({"policy":policy,"estimator":estimator,"physical_steps":steps,
                    "estimate":change,"interval_low":change-width,"interval_high":change+width,
                    "change_plus_uncertainty":change+width,"delta_allowance":.004,
                    "bump_check_pass":change+width<=.004})
            for bump in (.005,.0025):
                if steps != 128:
                    continue
                mean=.001 if bump==.005 else .003;width=.001 if bump==.005 else .002
                times.append({"policy":policy,"estimator":"controlled","bump_relative":bump,
                    "coarse_steps":128,"fine_steps":256,"delta_change":mean,
                    "interval_low":mean-width,"interval_high":mean+width,
                    "change_plus_uncertainty":mean+width,"delta_allowance":.004,
                    "time_check_pass":mean+width<=.004,"independent_resolutions":True})
    return {"tables":{"delta_sampling":pd.DataFrame(sampling),"bump_refinement":pd.DataFrame(bumps),
        "time_refinement":pd.DataFrame(times)},"manifest":{"stock_delta_ready":False},"figures":{}}


def test_two_readonly_figures_keep_both_policies_raw_failures_and_separate_pools():
    import matplotlib.pyplot as plt
    result=_result();before=deepcopy(result);rendered=add_bellman_delta_reporting(result)
    assert set(rendered["figures"])=={"delta_sampling_vs_allowance","delta_refinement"}
    assert not result["figures"] and result["manifest"]==before["manifest"]
    for name,table in result["tables"].items():pd.testing.assert_frame_equal(table,before["tables"][name])
    figure=rendered["figures"]["delta_sampling_vs_allowance"]
    assert len(figure.axes[0].get_yticklabels())==4
    labels=[x.get_text() for x in figure.axes[0].get_yticklabels()]
    assert any("4,096" in label for label in labels) and any("2,048" in label for label in labels)
    assert len(figure.axes[1].patches)==8
    assert any(bar.get_width()>1 for bar in figure.axes[1].patches)
    assert any(bar.get_width()<1 for bar in figure.axes[1].patches)
    assert figure.axes[0].xaxis.get_offset_text().get_text()==""
    for figure in rendered["figures"].values():plt.close(figure)


def test_sampling_summary_keeps_both_declared_bumps_and_fixed_128_operator():
    result=_result();summary=bellman_delta_sampling_summary(result["tables"])
    assert len(summary)==4 and set(summary.policy)=={"bellman"} and set(summary.estimator)=={"controlled"}
    assert list(summary.physical_steps)==[128,128,256,256]
    assert list(summary.bump_relative)==[.005,.0025,.005,.0025]
    assert set(summary.policy_operator_steps)=={128}


def test_refinement_display_reads_paired_and_independent_budgets_without_overriding_failures():
    result=_result();summary=bellman_delta_refinement_summary(result["tables"])
    assert list(summary.check_family)==["Paired bump","Paired bump","Independent time","Independent time"]
    assert list(summary.recorded_check_pass)==[True,False,True,False]
    assert list(summary.change_plus_uncertainty)==[.0002,.005,.002,.005]
    assert summary.iloc[-1].estimate==.003
    assert summary.iloc[-1].independent_resolutions is True or summary.iloc[-1].independent_resolutions==True


def test_missing_or_failed_campaign_does_not_fabricate_sensitivities():
    failures=pd.DataFrame([{"error":"fixed-budget sensitivity failure"}])
    result={"tables":{"failures":failures},"manifest":{},"figures":{}}
    assert not add_bellman_delta_reporting(result)["figures"]
    assert bellman_delta_sampling_summary(result["tables"]).empty
    assert bellman_delta_refinement_summary(result["tables"]).empty


def test_zero_sampling_uncertainty_remains_zero_and_is_labelled_on_log_display():
    import matplotlib.pyplot as plt
    result=_result();data=result["tables"]["delta_sampling"]
    target=data.policy.eq("bellman")&data.estimator.eq("controlled")&data.bump_relative.eq(.0025)&data.physical_steps.eq(128)
    data.loc[target,"half_width"]=0.
    data.loc[target,"interval_low"]=data.loc[target,"estimate"]
    data.loc[target,"interval_high"]=data.loc[target,"estimate"]
    original=data.copy(deep=True);rendered=add_bellman_delta_reporting(result)
    pd.testing.assert_frame_equal(data,original)
    axis=rendered["figures"]["delta_sampling_vs_allowance"].axes[1]
    assert any(text.get_text()=="0" for text in axis.texts)
    assert any(bar.get_width()==0. for bar in axis.patches)
    for figure in rendered["figures"].values():plt.close(figure)


def test_latest_panel_counts_recorded_individual_flags_without_readiness_promotion():
    from rvlab.experiments._pricing_rough_bellman_delta_reporting import latest_bellman_delta_state
    result=_result();wrapped={"manifest":{"rough_bellman_delta":{
        "frozen_operator_steps":128,"american_pricing_ready":False,"stock_delta_ready":False,
        "optimal_delta_ready":False}},"tables":{"rough_bellman_delta_"+k:v for k,v in result["tables"].items()}}
    before=deepcopy(wrapped);state=latest_bellman_delta_state(wrapped).set_index("Question")
    assert state.loc["GH16 individual delta checks","Current evidence"]=="6/8 recorded; expected 8, correlated"
    assert "N=4,096" in state.loc["GH16 128-step paths · 95% half-width","Current evidence"]
    assert "N=2,048" in state.loc["GH16 256-step paths · 95% half-width","Current evidence"]
    assert state.loc["Optimal delta ready","Current evidence"]=="False"
    assert state.loc["Readiness flags","Current evidence"]=="American price: False; stock delta: False"
    assert wrapped["manifest"]==before["manifest"]
    for name,table in wrapped["tables"].items():pd.testing.assert_frame_equal(table,before["tables"][name])
    assert latest_bellman_delta_state({"tables":{},"manifest":{}}).empty


def _gap_tables():
    names=["latest selected tail control","target 1 bins penalty0","target 1 bins penalty0.0001",
        "target 4 bins penalty0","target 4 bins penalty0.0001"]
    comparison=pd.DataFrame([{"control":name,"selected_control":i==1,"designated_challenger":i==1,
        "paths":8192,"gap":.0024-i*.0001,"gap_se":.0001,"price_error_budget":.0033-i*.0001,
        "price_allowance":.001286,"price_budget_pass":False,"new_control_delta_validated":False}
        for i,name in enumerate(names)])
    changes=pd.DataFrame([{"control":name,"selected_control":i==1,"metric":metric,"estimate":value,
        "interval_low":value-.0001,"interval_high":value+.0001}
        for i,name in enumerate(names) for metric,value in (
            ("gap_gain",.0001),("upper_decrease",.00012),("lower_change",-.00002),
            ("raw_minus_controlled",.001))])
    return {"gap_comparison":comparison,"gap_paired_changes":changes}


def test_gap_display_keeps_training_choice_even_when_an_unselected_final_gap_is_lower():
    import matplotlib.pyplot as plt
    result=_result();result["tables"].update(_gap_tables());before=deepcopy(result)
    rendered=add_bellman_delta_reporting(result)
    assert set(rendered["figures"])=={"delta_sampling_vs_allowance","delta_refinement","stopping_gap_and_budget"}
    summary=bellman_gap_summary(result["tables"])
    pd.testing.assert_frame_equal(summary,result["tables"]["gap_comparison"])
    selected=bellman_gap_selected_changes(result["tables"])
    assert len(selected)==3 and set(selected.control)=={"target 1 bins penalty0"}
    assert set(selected.metric)=={"gap_gain","upper_decrease","lower_change"}
    figure=rendered["figures"]["stopping_gap_and_budget"]
    labels=figure.axes[0].get_yticklabels()
    assert len(labels)==5 and sum("Selected before" in label.get_text() for label in labels)==1
    assert "1 bin · no penalty" in labels[1].get_text() and labels[1].get_fontweight()=="bold"
    assert all(bar.get_width()>1 for bar in figure.axes[1].patches)
    for name,table in result["tables"].items():pd.testing.assert_frame_equal(table,before["tables"][name])
    assert result["manifest"]==before["manifest"]
    for figure in rendered["figures"].values():plt.close(figure)


def test_latest_panel_uses_gap_own_8192_paths_and_unchanged_failed_budget():
    from rvlab.experiments._pricing_rough_bellman_delta_reporting import latest_bellman_delta_state
    result=_result();result["tables"].update(_gap_tables())
    wrapped={"manifest":{"rough_bellman_delta":{"frozen_operator_steps":128,
        "american_pricing_ready":False,"stock_delta_ready":False,"optimal_delta_ready":False}},
        "tables":{"rough_bellman_delta_"+name:table for name,table in result["tables"].items()}}
    state=latest_bellman_delta_state(wrapped).set_index("Question")
    assert state.loc["Selected stopping control","Current evidence"]=="target 1 bins penalty0; N=8,192"
    assert "0.003200 / 0.001286; recorded pass: False"==state.loc["Selected price budget / allowance","Current evidence"]
    assert state.loc["New price-control delta validated","Current evidence"]=="False"
    assert bellman_gap_summary({}).empty and bellman_gap_selected_changes({}).empty
