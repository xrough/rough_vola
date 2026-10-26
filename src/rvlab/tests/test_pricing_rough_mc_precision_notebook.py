"""Actual notebook displays retain fixed-pool and historical evidence roles."""
from pathlib import Path
import ast

import nbformat


ROOT = Path(__file__).resolve().parents[3]


def _notebook():
    return nbformat.read(ROOT/'notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb', as_version=4)


def test_new_precision_displays_remain_separate_marked_presentation_cells():
    notebook = _notebook()
    selected = [cell for cell in notebook.cells if cell.cell_type == 'code'
        and cell.source.startswith('if results["manifest"].get("rough_mc_precision_included"')]
    assert len(selected) == 3
    for cell in selected:
        ast.parse(cell.source)
        assert cell.metadata.rvlab_presentation is True
        assert cell.metadata.jupyter.source_hidden is True
        assert {'presentation', 'hide-input'}.issubset(cell.metadata.tags)
        assert 'run_pricing_research(' not in cell.source
    assert all(key in '\n'.join(cell.source for cell in selected) for key in (
        'rough_mc_precision_complete_pool_budget', 'rough_mc_precision_precision_and_replicas',
        'rough_mc_precision_gap_tail_context'))


def test_missing_full_pool_cannot_display_a_partial_price_as_completed():
    notebook = _notebook()
    source = next(cell.source for cell in notebook.cells if cell.cell_type == 'code'
        and cell.source.startswith('if results["manifest"].get("rough_mc_precision_included"'))
    messages = []; tables_shown = []
    namespace = {'results': {'manifest': {'rough_mc_precision_included': True}, 'figures': {}},
        'tables': {}, 'pd': __import__('pandas'), 'Markdown': str, 'display': messages.append,
        'present_table': lambda *a, **k: tables_shown.append(a),
        'present_figure': lambda *a, **k: None}
    exec(compile(source, 'mc-precision-display', 'exec'), namespace)
    assert not tables_shown and len(messages) == 1
    assert 'No partial-pool prices, passes or readiness claims' in messages[0]


def test_old_failed_pool_and_new_diagnostic_replicas_have_distinct_roles():
    text = '\n'.join(cell.source for cell in _notebook().cells if cell.cell_type == 'markdown')
    assert '**Historical study.** The complete executed 49,152-path evidence' in text
    assert 'Replica `k` pools rows `[k×16,384:(k+1)×16,384]` from each' in text
    assert 'All four are retained, with no stopping, acceptance or selection role.' in text
    assert 'All sixteen preceding rough-engine studies remain verified historical replays' in text
