"""Bellman displays keep failed challenges, all paths and accepted identities."""
from pathlib import Path

import pandas as pd
import pytest


def _cells():
    nbformat=pytest.importorskip("nbformat")
    path=Path(__file__).resolve().parents[3]/"notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb"
    n=nbformat.read(path,as_version=4)
    cells=[c.source for c in n.cells if c.cell_type=="code" and
        c.source.startswith('if results["manifest"].get("rough_bellman_policy_included"')]
    assert len(cells)==3
    return cells


def _env(manifest,tables,frames,text):
    return {"results":{"manifest":manifest,"figures":{}},"tables":tables,"pd":pd,
        "display":text.append,"Markdown":str,
        "present_table":lambda f,**k:frames.append((f.copy(),k)),
        "present_figure":lambda *a,**k:pytest.fail("Fixture has no figures")}


def test_disabled_bellman_study_needs_no_saved_results():
    frames,text=[],[];env=_env({}, {},frames,text)
    for cell in _cells():exec(cell,env)
    assert not frames
    assert text==["The state-dependent Bellman policy study is disabled in this execution."]


def test_failed_operator_attempt_is_visible_without_fabricated_policy_gain():
    failures=pd.DataFrame([{"error":"bounded Bellman operator attempt failed"}])
    frames,text=[],[]
    env=_env({"rough_bellman_policy_executed":True},
        {"rough_bellman_policy_failures":failures},frames,text)
    for cell in _cells():exec(cell,env)
    assert any("No completed independent Bellman-policy evidence" in item for item in text)
    assert len(frames)==1
    pd.testing.assert_frame_equal(frames[0][0],failures)
    assert frames[0][1]["collapsed"] is True


def test_all_rules_and_all_calendar_rows_survive_compact_presentation():
    policies=["frozen selected policy"]+[f"Bellman policy {i}" for i in range(6)]
    comparison=pd.DataFrame([{"policy":p,"price":2.86,"upper":2.862,
        "gap":.002,"price_error_budget":.003,"price_allowance":.0013,
        "price_budget_pass":False} for p in policies])
    evidence=pd.DataFrame([{"policy":p,"selected_policy":p==policies[0],
        "selected_challenger":p==policies[3],"identified_policy_gain":0.,
        "gain_interval_low":-.0001,"gain_interval_high":.0001,
        "policy_majority_proved":False} for p in policies])
    exercise=pd.DataFrame([{"policy":p,"stop_index":i,"fraction":1/65}
        for p in policies for i in range(65)])
    frames,text=[],[]
    manifest={"rough_bellman_policy_executed":True,"rough_bellman_policy":{
        "baseline_policy":policies[0],"selected_policy":policies[0],"selected_challenger":policies[3]}}
    env=_env(manifest,{"rough_bellman_policy_comparison":comparison,
        "rough_bellman_policy_policy_error":evidence,
        "rough_bellman_policy_exercise_times":exercise},frames,text)
    for cell in _cells():exec(cell,env)
    whole=next(f for f,k in frames if k["title"]=="Every policy challenge on independent final paths")
    pd.testing.assert_frame_equal(whole,evidence)
    curated=next(f for f,k in frames if k["title"]=="Accepted rule and separately designated challenger")
    assert list(curated.policy)==[policies[0],policies[0],policies[3]]
    calendar,options=next((f,k) for f,k in frames if k["title"].endswith("· exercise times"))
    pd.testing.assert_frame_equal(calendar,exercise)
    assert len(calendar)==455 and options["max_rows"]==30
    assert "30 of 455" in options["caption"] and "not an acceptance subset" in options["caption"]
    raw=next(f for f,k in frames if k["title"].endswith("· comparison"))
    pd.testing.assert_frame_equal(raw,comparison)
