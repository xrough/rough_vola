"""The centering investigation replays historical campaigns without repricing."""
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.mark.parametrize("options,error", [
    ({"run_rough_control_centering":1}, ValueError),
    ({"rough_control_centering_kwargs":[]}, TypeError),
    ({"rough_control_centering_kwargs":{"seed":7}}, ValueError),
    ({"rough_control_centering_kwargs":{"reuse":False}}, ValueError),
    ({"rough_control_centering_kwargs":{"reuse_checkpoints":False}}, ValueError),
    *[({"run_rough_control_centering":True, name:{"trial_specs":[]}}, ValueError)
      for name in ("rough_study_kwargs", "rough_precision_kwargs", "rough_american_kwargs",
          "rough_american_accuracy_kwargs", "rough_stopping_attribution_kwargs",
          "rough_dual_tightening_kwargs", "rough_policy_improvement_kwargs")],
])
def test_centering_workflow_rejects_ambiguous_or_historical_options_before_pricing(options, error):
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):
        pricing_research(SimpleNamespace(), **options)


def test_centering_study_rejects_unknown_keywords_before_work():
    from rvlab.experiments._pricing_rough_control_centering import control_centering_study
    with pytest.raises(TypeError, match="unexpected keyword"):
        control_centering_study(None, unknown_numerical_override=True)


def test_centering_workflow_only_replays_all_seven_older_campaigns(monkeypatch, tmp_path):
    from rvlab.experiments import _pricing_research as research
    from rvlab.experiments import _pricing_rough_foundation as foundation
    from rvlab.experiments import _pricing_rough_precision as precision
    from rvlab.experiments import _pricing_rough_american as american
    from rvlab.experiments import _pricing_rough_american_accuracy as accuracy
    from rvlab.experiments import _pricing_rough_stopping_attribution as attribution
    from rvlab.experiments import _pricing_rough_dual_tightening as tightening
    from rvlab.experiments import _pricing_rough_policy_improvement as policy
    from rvlab.experiments import _pricing_rough_control_centering as centering
    from rvlab.experiments import _pricing_rough_centering_reporting as reporting
    from rvlab.pricing import PricingConfig, FDConfig
    previous, loads, executions = {}, [], []
    bindings = (
        ("rough_foundation", american, "load_historical_foundation", foundation, "rough_foundation_study"),
        ("rough_precision", american, "load_historical_precision", precision, "rough_precision_study"),
        ("rough_american", american, "load_historical_american", american, "rough_american_study"),
        ("rough_american_accuracy", attribution, "load_historical_accuracy", accuracy, "rough_american_accuracy_study"),
        ("rough_stopping_attribution", tightening, "load_historical_attribution", attribution, "rough_stopping_attribution_study"),
        ("rough_dual_tightening", policy, "load_historical_tightening", tightening, "rough_dual_tightening_study"),
        ("rough_policy_improvement", centering, "load_historical_policy", policy, "rough_policy_improvement_study"),
    )
    for stage, load_module, loader, execute_module, execute in bindings:
        previous[stage] = {"tables":{"sentinel":pd.DataFrame([{"stage":stage, "old_result":.012345}])},
            "figures":{}, "manifest":{"backend":stage, "historical_replay":True, "source_identity":"frozen "+stage}}
        def load(*args, _stage=stage, **options):
            loads.append(_stage)
            return previous[_stage]
        monkeypatch.setattr(load_module, loader, load)
        monkeypatch.setattr(execute_module, execute,
            lambda *args, **options:pytest.fail("An older campaign was repriced under current source"))
    current = {"tables":{"contrast":pd.DataFrame([{"mean":0., "acceptance_input":False}])},
        "figures":{}, "manifest":{"backend":"frozen centering investigation"}}
    def execute_current(context, **options):
        executions.append(options)
        return current
    monkeypatch.setattr(centering, "control_centering_study", execute_current)
    # This test covers workflow routing. Artifact/hash validation is covered
    # by the reporting tests with complete sealed campaign fixtures.
    monkeypatch.setattr(reporting, "add_centering_execution", lambda value, **kwargs:value)
    monkeypatch.setattr(reporting, "add_centering_supplement", lambda value, **kwargs:value)
    monkeypatch.setattr(research, "_pricing_figures", lambda *args:{})
    context = SimpleNamespace(progress=lambda _:None, output_dir=tmp_path)
    flags = {"run_"+stage:True for stage in previous}
    result = research.pricing_research(context, fourier_config=PricingConfig(32,40.,64),
        fd_config=FDConfig(24,8,20), rough_reuse_checkpoints=False,
        run_rough_control_centering=True, rough_control_centering_kwargs={"repeat_specs":[], "run_audit":False}, **flags)
    assert loads == list(previous)
    for stage in previous:
        assert not result["manifest"][stage+"_executed"]
        assert result["manifest"][stage+"_included"]
        assert result["manifest"][stage+"_execution_mode"] == "preserved_historical_records"
        assert result["manifest"][stage] is previous[stage]["manifest"]
        assert result["tables"][stage+"_sentinel"] is previous[stage]["tables"]["sentinel"]
    assert result["manifest"]["rough_control_centering_executed"]
    assert result["manifest"]["rough_control_centering"] is current["manifest"]
    assert executions == [{"seed":42, "reuse_checkpoints":False, "repeat_specs":[], "run_audit":False}]
    assert result["manifest"]["calibration_optimizer_calls"] == 0
