"""The public notebook edition is readable and independent of private archives."""
import ast
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[3]
NAMES = ("01_pricing_and_exercise.ipynb", "02_synthetic_calibration.ipynb", "03_spy_pilot.ipynb")


def read(name):
    return nbformat.read(ROOT/"notebooks"/name, 4)


def test_three_public_notebooks_are_valid_compact_python_documents():
    assert {p.name for p in (ROOT/"notebooks").glob("*.ipynb")} == set(NAMES)
    for name in NAMES:
        notebook = read(name)
        nbformat.validate(notebook)
        assert 3 <= len(notebook.cells) <= 35
        assert any(c.cell_type == "markdown" for c in notebook.cells)
        assert any(c.cell_type == "code" for c in notebook.cells)
        assert notebook.metadata.kernelspec.language == "python"
        for cell in notebook.cells:
            if cell.cell_type == "code":
                # Notebook magics are valid IPython; validate the same Python
                # translation used by its kernel, including every code cell.
                ast.parse(TransformerManager().transform_cell(cell.source))


def test_public_notebook_code_does_not_require_the_original_research_archive():
    forbidden = ("first_stage_pre_report_2026-10-20", "first_stage_policy_baseline_2026-10-21",
                 "setup_policy_comparison", "run_policy_comparison_", "Rough-Pricing",
                 "/Users/", "demo/python/research", "demo/data/")
    for name in NAMES:
        code = "\n".join(c.source for c in read(name).cells if c.cell_type == "code")
        for path in forbidden:
            assert path not in code, (name, path)


def test_public_reading_order_contains_exactly_the_public_notebooks():
    order = [line.strip() for line in (ROOT/"notebooks/.order").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert order == list(NAMES)
