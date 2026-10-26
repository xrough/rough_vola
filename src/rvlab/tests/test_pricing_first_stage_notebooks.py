"""The preserved v1 reports retain their original compact research structure."""
from pathlib import Path
import ast

import nbformat


ROOT=Path(__file__).resolve().parents[3]
NAMES=('01_pricing_and_exercise.ipynb','02_synthetic_calibration.ipynb','03_spy_calibration.ipynb')


def _read(name):return nbformat.read(ROOT/'notebooks/legacy/pricing/first_stage_policy_baseline_2026-10-21/notebooks/pricing'/name,4)


def test_three_active_reports_have_compact_question_based_structure():
    for name,sections,maximum in zip(NAMES,(9,7,5),(30,25,20)):
        n=_read(name)
        assert n.metadata.rvlab_artifact_family=='first_stage_v1'
        assert len(n.cells)<=maximum
        headings=[c.source for c in n.cells if c.cell_type=='markdown' and c.source.startswith('## ')]
        assert len(headings)==sections
        for cell in n.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
                if 'present_figure(' in cell.source or 'present_table(' in cell.source or 'present_archive(' in cell.source:
                    assert cell.metadata.rvlab_presentation is True
                    assert cell.metadata.jupyter.source_hidden is True
        assert sum('results = run_first_stage_' in c.source for c in n.cells)==1
        assert any('setup_first_stage(' in c.source for c in n.cells)


def test_pricing_report_retains_all_cases_and_separates_focused_versus_generic_claims():
    n=_read(NAMES[0]);text='\n'.join(c.source for c in n.cells if c.cell_type=='markdown')
    assert 'K=90/100/110' in text and '30/365' in text and '90/365' in text
    assert 'predeclared primary' in text and 'not a winner selected from final errors' in text
    assert 'not isolated estimates of one error component' in text
    assert 'specific' in ' '.join(text.lower().split()) or 'one contract' in text
    assert 'all 23 chronological' in text
    code='\n'.join(c.source for c in n.cells if c.cell_type=='code')
    assert 'RUN_ROUGH_' not in code and 'rough_reuse_checkpoints' not in code
    assert 'report_generic_cases' in code and 'rough_benchmark_refinement' in code


def test_calibration_report_retains_adverse_validation_and_current_source_scope():
    text='\n'.join(c.source for c in _read(NAMES[1]).cells)
    assert 'zero optimizer calls' in text or 'zero optimizer' in text.lower()
    assert 'All 72 prices and deltas reproduce exactly' in text
    assert 'neither a new calibration' in text
    assert 'reverses after refinement' in text
    assert '10,000 nonlinear calibrations' in text
    assert 'report_matched_maturity_recovery' in text


def test_blocked_spy_report_cannot_display_fabricated_fits_even_when_tables_exist():
    n=_read(NAMES[2]);source=next(c.source for c in n.cells if c.cell_type=='code' and 'calibration_ready' in c.source)
    displayed=[];messages=[]
    exec(source,{'tables':{'study_design':__import__('pandas').DataFrame(),
        'calibration_summary':__import__('pandas').DataFrame([{'invented':True}])},
        'results':{'manifest':{'calibration_ready':False}},'Markdown':str,'display':messages.append,
        'present_table':lambda table,**kw:displayed.append(table)})
    assert len(displayed)==1 and displayed[0].empty
    assert len(messages)==1 and 'No calibration residuals, fitted parameters or rankings' in messages[0]
    text='\n'.join(c.source for c in n.cells if c.cell_type=='markdown')
    assert 'independent observed spot' in text and 'historically available' in text
    assert 'American-rough calibration is a separate unsupported workflow' in text
