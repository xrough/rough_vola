"""Notebook stock-delta displays keep frozen-policy and evidence roles explicit."""
from pathlib import Path
import ast

import nbformat


ROOT = Path(__file__).resolve().parents[3]


def _notebook():
    return nbformat.read(ROOT/'notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb', as_version=4)


def test_shared_delta_displays_use_recorded_evidence_in_three_marked_cells():
    selected = [cell for cell in _notebook().cells if cell.cell_type == 'code'
        and cell.source.startswith('if results["manifest"].get("rough_shared_delta_included"')]
    assert len(selected) == 3
    for cell in selected:
        ast.parse(cell.source)
        assert cell.metadata.rvlab_presentation is True
        assert cell.metadata.jupyter.source_hidden is True
        assert {'presentation', 'hide-input'}.issubset(cell.metadata.tags)
        assert 'run_pricing_research(' not in cell.source
    text = '\n'.join(cell.source for cell in selected)
    assert all(key in text for key in ('rough_shared_delta_delta_sampling_precision',
        'rough_shared_delta_delta_refinement_checks', 'rough_shared_delta_shared_delta_trials'))


def test_missing_new_delta_evidence_does_not_display_a_historical_or_partial_result():
    source = next(cell.source for cell in _notebook().cells if cell.cell_type == 'code'
        and cell.source.startswith('if results["manifest"].get("rough_shared_delta_included"'))
    messages = []; tables_shown = []
    namespace = {'results': {'manifest': {'rough_shared_delta_included': True}, 'figures': {}},
        'tables': {}, 'pd': __import__('pandas'), 'Markdown': str, 'display': messages.append,
        'present_table': lambda *a, **k: tables_shown.append(a),
        'present_figure': lambda *a, **k: None}
    exec(compile(source, 'shared-delta-display', 'exec'), namespace)
    assert not tables_shown and len(messages) == 1
    assert 'No partial delta, accepted check or readiness claim' in messages[0]


def test_policy_target_refinement_semantics_and_historical_price_gate_stay_distinct():
    text = '\n'.join(cell.source for cell in _notebook().cells if cell.cell_type == 'markdown')
    assert '**Historical study.** The complete executed 196,608-path price evidence' in text
    assert 'All seventeen preceding rough-engine studies remain verified historical replays' in text
    assert '64-date lower-policy value' in text
    assert 'neither a derivative of the dual upper estimator nor an optimal-American delta' in text
    assert 'changes both the physical law and control representation' in text
    assert 'An interval containing zero is not evidence of tolerance-scale equivalence' in text
