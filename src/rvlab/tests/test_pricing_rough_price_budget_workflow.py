"""The price-budget investigation replays all earlier campaigns unchanged."""
from types import SimpleNamespace
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize("options,error",[
    ({"run_rough_price_budget":1},ValueError),
    ({"rough_price_budget_kwargs":[]},TypeError),
    *[({"rough_price_budget_kwargs":{key:False}},ValueError)
      for key in ("seed","reuse","reuse_checkpoints")],
    *[({"run_rough_price_budget":True,key:{"compute_figures":False}},ValueError)
      for key in ("rough_bellman_delta_kwargs","rough_bellman_policy_kwargs",
        "rough_tail_attribution_kwargs","rough_gap_time_refinement_kwargs",
        "rough_policy_validation_kwargs","rough_stopping_delta_kwargs")]])
def test_invalid_options_fail_before_pricing(options,error):
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):pricing_research(SimpleNamespace(),**options)


def test_price_budget_has_one_current_campaign_and_preserves_all_earlier_results(monkeypatch,tmp_path):
    from rvlab.experiments import _pricing_research as research
    from rvlab.pricing import PricingConfig,FDConfig
    bindings=[
        ("rough_american","_pricing_rough_american","load_historical_american"),
        ("rough_american_accuracy","_pricing_rough_stopping_attribution","load_historical_accuracy"),
        ("rough_stopping_attribution","_pricing_rough_dual_tightening","load_historical_attribution"),
        ("rough_dual_tightening","_pricing_rough_policy_improvement","load_historical_tightening"),
        ("rough_policy_improvement","_pricing_rough_control_centering","load_historical_policy"),
        ("rough_control_centering","_pricing_rough_uncertainty_history","load_historical_centering"),
        ("rough_centering_precision","_pricing_rough_stopping_delta_history","load_historical_centering_precision"),
        ("rough_stopping_delta","_pricing_rough_policy_validation_history","load_historical_stopping_delta"),
        ("rough_policy_validation","_pricing_rough_gap_time_history","load_historical_policy_validation"),
        ("rough_gap_time_refinement","_pricing_rough_tail_history","load_historical_gap_time_refinement"),
        ("rough_tail_attribution","_pricing_rough_bellman_history","load_historical_tail_attribution"),
        ("rough_bellman_policy","_pricing_rough_bellman_delta_history","load_historical_bellman_policy"),
        ("rough_bellman_delta","_pricing_rough_price_budget_history","load_historical_bellman_delta")]
    loads=[]
    for stage,module,loader in bindings:
        def load(*args,_stage=stage,**kwargs):
            loads.append(_stage)
            return {"tables":{"preserved":pd.DataFrame([{"value":.12345}])},
                "figures":{},"manifest":{"backend":_stage,"historical_replay":True}}
        monkeypatch.setattr("rvlab.experiments."+module+"."+loader,load)
    calls=[]
    def collect(*args,**kwargs):
        calls.append(kwargs)
        return {"tables":{"price_comparison":pd.DataFrame([{"price_budget_pass":True}])},
            "figures":{},"manifest":{"backend":"rough_heston_price_budget",
                "american_pricing_ready":False,"stock_delta_ready":False}}
    monkeypatch.setitem(sys.modules,"rvlab.experiments._pricing_rough_price_budget_collection",
        SimpleNamespace(price_budget_study=collect))
    monkeypatch.setitem(sys.modules,"rvlab.experiments._pricing_rough_bellman_delta_collection",
        SimpleNamespace(bellman_delta_study=lambda *a,**k:pytest.fail("Spent campaign reran")))
    monkeypatch.setattr(research,"_pricing_figures",lambda *a:{})
    result=research.pricing_research(SimpleNamespace(progress=lambda _:None,output_dir=tmp_path),
        fourier_config=PricingConfig(32,40.,64),fd_config=FDConfig(24,8,20),
        run_rough_price_budget=True,rough_price_budget_kwargs={"compute_figures":False})
    assert loads==[s for s,_,_ in bindings]
    for stage in loads:
        assert not result["manifest"][stage+"_executed"]
        assert result["manifest"][stage+"_included"]
        assert result["manifest"][stage+"_execution_mode"]=="preserved_historical_records"
        assert result["tables"][stage+"_preserved"].value.iloc[0]==.12345
    assert calls==[{"seed":42,"compute_figures":False}]
    assert result["manifest"]["rough_price_budget_executed"]
    assert result["manifest"]["calibration_optimizer_calls"]==0
