"""Latest policy studies replay earlier evidence through guarded loaders only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _stub_gallery(monkeypatch, research):
    """Keep this routing test independent of numerical engine cost/accuracy."""
    surface = pd.DataFrame({"forward": [100.], "discount": [.99], "strike": [100.],
        "maturity": [.1], "is_call": [False], "market_price": [2.]})
    monkeypatch.setattr(research, "synthetic_surface", lambda *a, **k: surface.copy())
    monkeypatch.setattr(research, "_ivs", lambda table: table)
    monkeypatch.setattr(research, "price", lambda params, **kwargs:
        np.ones(np.shape(kwargs["strike"])))
    monkeypatch.setattr(research, "_european_limits", lambda config: [])
    monkeypatch.setattr(research, "refinement_table", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(research, "_pricing_figures", lambda *a, **k: {})
    def value(params, *, strike, **kwargs):
        shape = np.shape(strike)
        return SimpleNamespace(prices=np.ones(shape), deltas=np.zeros(shape),
            exercise_boundaries=(), diagnostics={"checks": pd.DataFrame()})
    monkeypatch.setattr(research, "evaluate_vanilla", value)


def test_policy_workflow_replays_every_earlier_stage_without_current_repricing(monkeypatch, tmp_path):
    from rvlab.experiments import _pricing_research as research
    from rvlab.experiments import _pricing_rough_foundation as foundation
    from rvlab.experiments import _pricing_rough_precision as precision
    from rvlab.experiments import _pricing_rough_american as american
    from rvlab.experiments import _pricing_rough_american_accuracy as accuracy
    from rvlab.experiments import _pricing_rough_stopping_attribution as attribution
    from rvlab.experiments import _pricing_rough_dual_tightening as tightening
    from rvlab.experiments import _pricing_rough_policy_improvement as policy
    from rvlab.experiments import _pricing_rough_policy_reporting as reporting
    _stub_gallery(monkeypatch, research)
    stages = ("rough_foundation", "rough_precision", "rough_american",
        "rough_american_accuracy", "rough_stopping_attribution", "rough_dual_tightening")
    history = {stage: {"tables": {"original_prices": pd.DataFrame({"price": [i+.123456789]})},
        "figures": {}, "manifest": {"backend": stage+"_original_backend",
            "historical_replay": True, "current_engine_execution": False,
            "profile_fingerprint": stage+"_original_profile",
            "historical_archive_manifest_sha256": stage+"_verified_inventory",
            "source_elapsed_seconds": 101.+i}} for i, stage in enumerate(stages)}
    calls = []
    context = SimpleNamespace(progress=lambda _: None, output_dir=tmp_path)
    def loader(stage):
        def load(actual_context):
            assert actual_context is context
            calls.append(stage)
            return history[stage]
        return load
    for module, name, stage in (
        (american, "load_historical_foundation", "rough_foundation"),
        (american, "load_historical_precision", "rough_precision"),
        (american, "load_historical_american", "rough_american"),
        (attribution, "load_historical_accuracy", "rough_american_accuracy"),
        (tightening, "load_historical_attribution", "rough_stopping_attribution"),
        (policy, "load_historical_tightening", "rough_dual_tightening"),
    ):
        monkeypatch.setattr(module, name, loader(stage))
    def forbidden_repricing(*args, **kwargs):
        pytest.fail("Historical stages must use byte-verified replay loaders")
    for module, name in (
        (foundation, "rough_foundation_study"), (precision, "rough_precision_study"),
        (american, "rough_american_study"), (accuracy, "rough_american_accuracy_study"),
        (attribution, "rough_stopping_attribution_study"), (tightening, "rough_dual_tightening_study"),
    ):
        monkeypatch.setattr(module, name, forbidden_repricing)
    monkeypatch.setattr(american, "arithmetic_repair_comparison", lambda *a: (pd.DataFrame(), pd.DataFrame()))
    current = {"tables": {"comparison": pd.DataFrame({"policy": ["current policy"]})},
        "figures": {}, "manifest": {"backend": "rough_heston_policy_improvement"}}
    current_calls = []
    def run_current(actual_context, **kwargs):
        assert actual_context is context
        current_calls.append(kwargs)
        return current
    monkeypatch.setattr(policy, "rough_policy_improvement_study", run_current)
    reporting_calls = []
    monkeypatch.setattr(reporting, "add_policy_execution", lambda result, **kwargs:
        reporting_calls.append(kwargs) or result)
    options = {"trial_specs": [{"predeclared": "test only"}], "probe_archives": ()}
    result = research.pricing_research(context, seed=17, rough_reuse_checkpoints=False,
        run_rough_foundation=True, rough_study_kwargs={}, run_rough_precision=True,
        rough_precision_kwargs={}, run_rough_american=True, rough_american_kwargs={},
        run_rough_american_accuracy=True, rough_american_accuracy_kwargs={},
        run_rough_stopping_attribution=True, rough_stopping_attribution_kwargs={},
        run_rough_dual_tightening=True, rough_dual_tightening_kwargs={},
        run_rough_policy_improvement=True, rough_policy_improvement_kwargs=options)
    assert calls == list(stages)
    assert current_calls == [{"seed": 17, "reuse_checkpoints": False, **options}]
    assert reporting_calls == [{"output_dir": tmp_path}]
    for stage in stages:
        assert result["manifest"][stage+"_included"] is True
        assert result["manifest"][stage+"_executed"] is False
        assert result["manifest"][stage+"_execution_mode"] == "preserved_historical_records"
        assert result["manifest"][stage] is history[stage]["manifest"]
        assert result["tables"][stage+"_original_prices"] is history[stage]["tables"]["original_prices"]
    assert result["manifest"]["rough_policy_improvement_executed"] is True
    assert result["manifest"]["rough_policy_improvement"] is current["manifest"]
    assert result["tables"]["rough_policy_improvement_comparison"] is current["tables"]["comparison"]
    assert result["manifest"]["calibration_optimizer_calls"] == 0


@pytest.mark.parametrize("historical_kwargs", ["rough_study_kwargs", "rough_precision_kwargs",
    "rough_american_kwargs", "rough_american_accuracy_kwargs",
    "rough_stopping_attribution_kwargs", "rough_dual_tightening_kwargs"])
def test_policy_workflow_rejects_nonempty_historical_options_before_pricing(monkeypatch, historical_kwargs):
    from rvlab.experiments import _pricing_research as research
    monkeypatch.setattr(research, "synthetic_surface", lambda *a, **k:
        pytest.fail("Historical option validation must precede any pricing"))
    with pytest.raises(ValueError, match="empty"):
        research.pricing_research(SimpleNamespace(), run_rough_policy_improvement=True,
            **{historical_kwargs: {"trial_specs": []}})


@pytest.mark.parametrize("options,error", [
    ({"run_rough_policy_improvement": 1}, ValueError),
    ({"rough_policy_improvement_kwargs": []}, TypeError),
    ({"rough_policy_improvement_kwargs": {"seed": 7}}, ValueError),
    ({"rough_policy_improvement_kwargs": {"reuse": True}}, ValueError),
    ({"rough_policy_improvement_kwargs": {"reuse_checkpoints": False}}, ValueError),
])
def test_policy_workflow_rejects_ambiguous_controls_before_pricing(monkeypatch, options, error):
    from rvlab.experiments import _pricing_research as research
    monkeypatch.setattr(research, "synthetic_surface", lambda *a, **k:
        pytest.fail("Policy workflow controls must be validated before pricing"))
    with pytest.raises(error):
        research.pricing_research(SimpleNamespace(), **options)
