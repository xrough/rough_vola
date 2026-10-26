"""Actual display cells retain missing/failed precision evidence honestly."""
from pathlib import Path

import pandas as pd
import pytest


def _cells():
    nbformat = pytest.importorskip("nbformat")
    n = nbformat.read(Path(__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb", as_version=4)
    cells = [c.source for c in n.cells if c.cell_type == "code" and
        c.source.startswith('if results["manifest"].get("rough_centering_precision_included"')]
    assert len(cells) == 3
    return cells


def test_disabled_precision_notebook_needs_no_new_numerical_results():
    seen=[]
    env={"results":{"manifest":{},"figures":{}},"tables":{},"display":seen.append,"Markdown":str,
        "present_table":lambda *a,**k:pytest.fail("Disabled stage has no tables"),
        "present_figure":lambda *a,**k:pytest.fail("Disabled stage has no figures")}
    for source in _cells():exec(source,env)
    assert seen == ["The tolerance-scale conditional-centering study is disabled in this execution."]


def test_failed_precision_notebook_preserves_failure_without_fabricating_pool():
    seen=[]
    from rvlab.experiments.pricing_presentation import _table_html
    def present(frame, **kwargs):
        kwargs.pop("collapsed", None)
        seen.append(_table_html(frame, **kwargs))
    env={"pd":pd,"results":{"manifest":{"rough_centering_precision_executed":True},"figures":{}},
        "tables":{"rough_centering_precision_failures":pd.DataFrame([{"error":"independent integration failure retained"}])},
        "display":seen.append,"Markdown":str,"present_table":present,
        "present_figure":lambda *a,**k:pytest.fail("Failed stage has no figures")}
    for source in _cells():exec(source,env)
    text="\n".join(map(str,seen))
    assert "No completed independent conditional-precision pool" in text
    assert "independent integration failure retained" in text


def test_precision_notebook_keeps_all_fresh_seeds_and_realized_raw_values():
    frames=[]
    repeats=pd.DataFrame([{"quantity":q,"seed":s,"paths":2048,"estimate":v,"standard_error":.01,
        "confidence_low":v-.02,"confidence_high":v+.02}
        for s,v in ((510005042,-.04),(511005042,.06)) for q in ("raw_contrast","projected_fine")])
    pool=pd.DataFrame([{"quantity":q,"streams":2,"paths":4096,"estimate":v,"standard_error":.01,
        "confidence_low":v-.02,"confidence_high":v+.02}
        for q,v in (("raw_contrast",.01),("projected_coarse",1e-9),("projected_fine",1e-12))])
    budget=pd.DataFrame([{"scope":"pooled","paths":4096,"price_allowance":.001286,
        "sampling_half_width":1e-12,"numerical_envelope":1e-10,
        "uncertainty_endpoint_low":-1e-10,"uncertainty_endpoint_high":1e-10,
        "endpoint_interval_within_price_allowance":True}])
    env={"pd":pd,"results":{"manifest":{"rough_centering_precision_executed":True},"figures":{}},
        "tables":{"rough_centering_precision_per_seed":repeats,"rough_centering_precision_pooled":pool,
            "rough_centering_precision_numerical_budget":budget},"display":lambda x:None,"Markdown":str,
        "present_table":lambda f,**k:frames.append((f.copy(),k)),
        "present_figure":lambda *a,**k:pytest.fail("No figures recorded in fixture")}
    for source in _cells():exec(source,env)
    f=next(f for f,k in frames if k["title"] == "Every declared fresh stream, with paired realized and conditional quantities")
    assert f.seed.tolist()==[510005042,510005042,511005042,511005042]
    assert f.loc[f.quantity.eq("raw_contrast"),"estimate"].tolist()==[-.04,.06]
