"""Only the stopping/delta follow-up executes; nine old campaigns are replayed."""
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.mark.parametrize("options,error",[
    ({"run_rough_stopping_delta":1},ValueError),
    ({"run_rough_stopping_delta":"true"},ValueError),
    ({"rough_stopping_delta_kwargs":[]},TypeError),
    ({"rough_stopping_delta_kwargs":{"seed":7}},ValueError),
    ({"rough_stopping_delta_kwargs":{"reuse":False}},ValueError),
    ({"rough_stopping_delta_kwargs":{"reuse_checkpoints":False}},ValueError),
    *[({"run_rough_stopping_delta":True,name:{"trial_specs":[]}},ValueError)
      for name in ("rough_study_kwargs","rough_precision_kwargs","rough_american_kwargs",
          "rough_american_accuracy_kwargs","rough_stopping_attribution_kwargs",
          "rough_dual_tightening_kwargs","rough_policy_improvement_kwargs",
          "rough_control_centering_kwargs","rough_centering_precision_kwargs")],
])
def test_stopping_delta_workflow_rejects_ambiguous_or_malformed_options_before_pricing(options,error):
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):
        pricing_research(SimpleNamespace(),**options)


def test_stopping_delta_study_unknown_keywords_fail_before_work():
    from rvlab.experiments._pricing_rough_stopping_delta import stopping_delta_study
    with pytest.raises(TypeError,match="unexpected keyword"):
        stopping_delta_study(None,unknown_numerical_override=True)


@pytest.mark.parametrize("old_flags",[True,False])
def test_stopping_delta_workflow_only_executes_current_campaign_even_with_older_flags_disabled(monkeypatch,tmp_path,old_flags):
    from rvlab.experiments import _pricing_research as research
    from rvlab.experiments import _pricing_rough_foundation as foundation
    from rvlab.experiments import _pricing_rough_precision as precision
    from rvlab.experiments import _pricing_rough_american as american
    from rvlab.experiments import _pricing_rough_american_accuracy as accuracy
    from rvlab.experiments import _pricing_rough_stopping_attribution as attribution
    from rvlab.experiments import _pricing_rough_dual_tightening as tightening
    from rvlab.experiments import _pricing_rough_policy_improvement as policy
    from rvlab.experiments import _pricing_rough_control_centering as centering
    from rvlab.experiments import _pricing_rough_uncertainty_history as history
    from rvlab.experiments import _pricing_rough_centering_precision as prior_precision
    from rvlab.experiments import _pricing_rough_stopping_delta_history as precision_history
    from rvlab.experiments import _pricing_rough_stopping_delta as current
    from rvlab.pricing import PricingConfig,FDConfig
    previous,loads,executions={},[],[]
    bindings=(
        ("rough_foundation",american,"load_historical_foundation",foundation,"rough_foundation_study"),
        ("rough_precision",american,"load_historical_precision",precision,"rough_precision_study"),
        ("rough_american",american,"load_historical_american",american,"rough_american_study"),
        ("rough_american_accuracy",attribution,"load_historical_accuracy",accuracy,"rough_american_accuracy_study"),
        ("rough_stopping_attribution",tightening,"load_historical_attribution",attribution,"rough_stopping_attribution_study"),
        ("rough_dual_tightening",policy,"load_historical_tightening",tightening,"rough_dual_tightening_study"),
        ("rough_policy_improvement",centering,"load_historical_policy",policy,"rough_policy_improvement_study"),
        ("rough_control_centering",history,"load_historical_centering",centering,"control_centering_study"),
        ("rough_centering_precision",precision_history,"load_historical_centering_precision",prior_precision,"centering_precision_study"),
    )
    for stage,load_module,loader,execute_module,execute in bindings:
        previous[stage]={"tables":{"sentinel":pd.DataFrame([{"stage":stage,"old_result":.012345}])},
            "figures":{},"manifest":{"backend":stage,"historical_replay":True,
                "source_identity":"frozen "+stage,"source_elapsed_seconds":12.345}}
        def load(*args,_stage=stage,**kwargs):
            loads.append(_stage)
            return previous[_stage]
        monkeypatch.setattr(load_module,loader,load)
        monkeypatch.setattr(execute_module,execute,
            lambda *args,**kwargs:pytest.fail("An older campaign was repriced under current source"))
    result_current={"tables":{"contrast":pd.DataFrame([{"mean":.0000001,"acceptance_input":False}])},
        "figures":{},"manifest":{"backend":"independent stopping/delta follow-up"}}
    def execute_current(context,**options):
        executions.append(options)
        return result_current
    monkeypatch.setattr(current,"stopping_delta_study",execute_current)
    monkeypatch.setattr(research,"_pricing_figures",lambda *args:{})
    context=SimpleNamespace(progress=lambda message:None,output_dir=tmp_path)
    flags={"run_"+stage:old_flags for stage in previous}
    result=research.pricing_research(context,fourier_config=PricingConfig(32,40.,64),
        fd_config=FDConfig(24,8,20),rough_reuse_checkpoints=False,
        run_rough_stopping_delta=True,rough_stopping_delta_kwargs={"repeat_indices":[1],"compute_figures":False},**flags)
    expected=list(previous) if old_flags else list(previous)[2:]
    assert loads==expected
    for stage in expected:
        assert not result["manifest"][stage+"_executed"]
        assert result["manifest"][stage+"_included"]
        assert result["manifest"][stage+"_execution_mode"]=="preserved_historical_records"
        assert result["manifest"][stage] is previous[stage]["manifest"]
        assert result["tables"][stage+"_sentinel"] is previous[stage]["tables"]["sentinel"]
    if not old_flags:
        for stage in ("rough_foundation","rough_precision"):
            assert not result["manifest"][stage+"_executed"]
            assert not result["manifest"][stage+"_included"]
            assert stage not in result["manifest"]
    assert result["manifest"]["rough_stopping_delta_executed"]
    assert result["manifest"]["rough_stopping_delta"] is result_current["manifest"]
    assert result["tables"]["rough_stopping_delta_contrast"] is result_current["tables"]["contrast"]
    assert executions==[{"seed":42,"repeat_indices":[1],"compute_figures":False}]
    assert result["manifest"]["calibration_optimizer_calls"]==0
