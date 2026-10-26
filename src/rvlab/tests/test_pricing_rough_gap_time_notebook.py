"""Section17 uses saved results and never fabricates missing campaigns."""
from pathlib import Path

import pandas as pd
import pytest


def _cells():
    nbformat=pytest.importorskip("nbformat")
    path=Path(__file__).resolve().parents[3]/"notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb"
    notebook=nbformat.read(path,as_version=4)
    cells=[c.source for c in notebook.cells if c.cell_type=="code" and
        c.source.startswith('if results["manifest"].get("rough_gap_time_refinement_included"')]
    assert len(cells)==3
    return cells


def test_disabled_refinement_needs_no_campaign_results():
    seen=[]
    env={"results":{"manifest":{},"figures":{}},"tables":{},"pd":pd,
        "display":seen.append,"Markdown":str,
        "present_table":lambda *a,**k:pytest.fail("Disabled stage has no results"),
        "present_figure":lambda *a,**k:pytest.fail("Disabled stage has no figure")}
    for cell in _cells():exec(cell,env)
    assert seen==["The further stopping-gap and time-refinement study is disabled in this execution."]


def test_failed_refinement_reports_error_without_fabricated_sensitivities():
    seen=[]
    env={"results":{"manifest":{"rough_gap_time_refinement_executed":True},"figures":{}},
        "tables":{"rough_gap_time_refinement_failures":pd.DataFrame([{"error":"independent failure"}])},
        "pd":pd,"display":seen.append,"Markdown":str,
        "present_table":lambda f,**k:seen.append(f.to_dict("records")),
        "present_figure":lambda *a,**k:pytest.fail("Failed stage has no figure")}
    for cell in _cells():exec(cell,env)
    assert any("No completed independent time-refinement pool" in str(x) for x in seen)
    assert any("independent failure" in str(x) for x in seen)


def test_compact_new_rule_table_keeps_complete_raw_rows_and_grids_separate():
    rows=pd.DataFrame([{"policy":p,"estimator":e,"bump_relative":b,"paths":8192,
        "mean":-.4,"ci_half_width":.02 if e=="raw" else .001,
        "delta_allowance":.004,"sampling_passed":e!="raw"}
        for p in ("original_policy","selected_policy")
        for e in ("raw","stopped martingale") for b in (.005,.0025)])
    fine=rows.copy();fine["paths"]=2048
    frames=[]
    env={"results":{"manifest":{"rough_gap_time_refinement_executed":True},"figures":{}},
        "tables":{"rough_gap_time_refinement_delta_estimates":rows,
            "rough_gap_time_refinement_delta_512_estimates":fine},"pd":pd,
        "display":lambda x:None,"Markdown":str,
        "present_table":lambda f,**k:frames.append((f.copy(),k)),
        "present_figure":lambda *a,**k:pytest.fail("Fixture has no figures")}
    for cell in _cells():exec(cell,env)
    table=next(f for f,k in frames if k["title"]=="New physical-time pools, kept separate")
    assert len(table)==4 and list(table.physical_steps)==[256,256,512,512]
    assert list(table.paths)==[8192,8192,2048,2048]
    for name,expected in (("delta estimates",rows),("delta 512 estimates",fine)):
        table=next(f for f,k in frames if k["title"]=="Complete follow-up record · "+name)
        pd.testing.assert_frame_equal(table,expected)
