"""New-rule delta displays keep full evidence and never promote missing results."""
from pathlib import Path

import pandas as pd
import pytest


def _cells():
    nbformat=pytest.importorskip("nbformat")
    path=Path(__file__).resolve().parents[3]/"notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb"
    n=nbformat.read(path,as_version=4)
    cells=[c.source for c in n.cells if c.cell_type=="code" and
        c.source.startswith('if results["manifest"].get("rough_bellman_delta_included"')]
    assert len(cells)==3
    return cells


def _env(manifest,tables,frames,text):
    return {"results":{"manifest":manifest,"figures":{}},"tables":tables,"pd":pd,
        "display":text.append,"Markdown":str,
        "present_table":lambda f,**k:frames.append((f.copy(),k)),
        "present_figure":lambda *a,**k:pytest.fail("Fixture has no figures")}


def test_disabled_delta_gap_study_needs_no_results():
    frames,text=[],[];env=_env({}, {},frames,text)
    for cell in _cells():exec(cell,env)
    assert not frames
    assert text==["The Bellman stock-delta and remaining-gap study is disabled in this execution."]


def test_failed_delta_attempt_is_visible_without_invented_sensitivity():
    failures=pd.DataFrame([{"error":"bounded independent-path attempt failed"}])
    frames,text=[],[];env=_env({"rough_bellman_delta_executed":True},
        {"rough_bellman_delta_failures":failures},frames,text)
    for cell in _cells():exec(cell,env)
    assert any("No completed independent Bellman-rule stock-delta evidence" in item for item in text)
    assert len(frames)==1
    pd.testing.assert_frame_equal(frames[0][0],failures)
    assert frames[0][1]["collapsed"] is True


def test_compact_new_rule_table_retains_all_raw_rules_and_large_record():
    sampling=pd.DataFrame([{"physical_steps":s,"paths":4096 if s==128 else 2048,
        "policy":p,"estimator":e,"bump_relative":b,"estimate":-.4,"half_width":.001,
        "delta_allowance":.004,"sampling_pass":True,"policy_operator_steps":128}
        for s in (128,256) for p in ("baseline","bellman")
        for e in ("raw","controlled") for b in (.005,.0025)])
    full=pd.DataFrame({"record_index":range(200),"all_paths_retained":True})
    frames,text=[],[];env=_env({"rough_bellman_delta_executed":True},
        {"rough_bellman_delta_delta_sampling":sampling,
        "rough_bellman_delta_large_record":full},frames,text)
    for cell in _cells():exec(cell,env)
    focused=next(f for f,k in frames if k["title"]=="The selected rule's own sensitivities")
    assert len(focused)==4 and set(focused.policy)=={"bellman"}
    assert list(focused.physical_steps)==[128,128,256,256]
    assert list(focused.paths)==[4096,4096,2048,2048]
    original=next(f for f,k in frames if k["title"].endswith("· delta sampling"))
    pd.testing.assert_frame_equal(original,sampling)
    record,options=next((f,k) for f,k in frames if k["title"].endswith("· large record"))
    pd.testing.assert_frame_equal(record,full)
    assert len(record)==200 and options["max_rows"]==30
    assert "30 of 200" in options["caption"] and "not an acceptance subset" in options["caption"]


def test_control_display_uses_own_final_paths_and_preserves_failed_alternatives():
    comparison=pd.DataFrame([{"control":name,"selected_control":i==1,"paths":8192,
        "gap":.0024-i*.0001,"gap_se":.0001,"price_error_budget":.003-i*.0001,
        "price_allowance":.001286,"price_budget_pass":False}
        for i,name in enumerate(("latest selected tail control","target 1 bins penalty0",
            "target 4 bins penalty0"))])
    paired=pd.DataFrame([{"control":"target 1 bins penalty0","selected_control":True,
        "metric":metric,"estimate":.0001,"interval_low":-.0001,"interval_high":.0003}
        for metric in ("gap_gain","upper_decrease","lower_change","raw_minus_controlled")])
    frames,text=[],[];env=_env({"rough_bellman_delta_executed":True},
        {"rough_bellman_delta_gap_comparison":comparison,
        "rough_bellman_delta_gap_paired_changes":paired},frames,text)
    for cell in _cells():exec(cell,env)
    controls=next(f for f,k in frames if k["title"]=="Remaining stopping gap under the same GH16 rule")
    pd.testing.assert_frame_equal(controls,comparison)
    assert set(controls.paths)=={8192} and not controls.price_budget_pass.any()
    changes=next(f for f,k in frames if k["title"]=="Where the selected control changes the bracket")
    assert len(changes)==3 and set(changes.metric)=={"gap_gain","upper_decrease","lower_change"}
    raw_record=next(f for f,k in frames if k["title"].endswith("· gap paired changes"))
    pd.testing.assert_frame_equal(raw_record,paired)
