"""Shared fixtures. Every test runs on synthetic data so none of them need the
gitignored market files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Tests render figures off screen and never need a desktop GUI backend.
os.environ.setdefault("MPLBACKEND", "Agg")

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



# These exact tests inspect previous report editions. Their scientific engines
# and numerical unit tests remain in the default suite. The public repository
# intentionally does not include the multi-gigabyte historical report archive.
_ARCHIVE_MODULES = {
    "test_pricing_first_stage_notebooks.py",
    "test_pricing_rough_bellman_delta_notebook.py",
    "test_pricing_rough_bellman_notebook.py",
    "test_pricing_rough_centering_notebook.py",
    "test_pricing_rough_centering_precision_notebook.py",
    "test_pricing_rough_gap_time_notebook.py",
    "test_pricing_rough_mc_precision_notebook.py",
    "test_pricing_rough_policy_notebook.py",
    "test_pricing_rough_policy_validation_notebook.py",
    "test_pricing_rough_price_budget_notebook.py",
    "test_pricing_rough_shared_delta_notebook.py",
    "test_pricing_rough_stopping_delta_notebook.py",
    "test_pricing_rough_tail_notebook.py",
}
_ARCHIVE_TESTS = {
    "test_config.py": {
        "test_notebook_runner_rejects_a_mismatched_selected_kernel"},
    "test_pricing_american_workflows.py": {
        "test_preserved_american_notebooks_do_not_contain_old_empirical_fits"},
    "test_pricing_rough_american_accuracy_study.py": {
        "test_all_failed_follow_up_displays_an_audit_without_fabricated_prices"},
    "test_pricing_rough_american_study.py": {
        "test_actual_notebook_all_failed_stopping_display_is_an_audit_not_an_exception"},
    "test_pricing_rough_dual_tightening_study.py": {
        "test_notebook_tightening_display_preserves_all_failed_cases_without_assuming_comparisons"},
    "test_pricing_rough_foundation.py": {
        "test_recorded_artifact_displays_need_no_parameter_cell_globals"},
    "test_pricing_rough_precision.py": {
        "test_precision_notebook_displays_refresh_from_artifacts_without_parameter_globals",
        "test_foundation_notebook_labels_preserved_outputs_without_claiming_current_execution",
        "test_actual_notebook_parameter_and_run_cells_obey_workflow_argument_contract"},
    "test_pricing_rough_stopping_attribution_study.py": {
        "test_current_notebook_attribution_display_handles_all_failed_cases"},
}


def pytest_addoption(parser):
    parser.addoption("--run-archive-tests", action="store_true", default=False,
                     help="Run original report-edition tests after separately restoring their historical archive")


def pytest_collection_modifyitems(config, items):
    for item in items:
        function = getattr(item, "originalname", item.name.split("[")[0])
        if item.path.name in _ARCHIVE_MODULES or function in _ARCHIVE_TESTS.get(item.path.name, set()):
            item.add_marker(pytest.mark.archive)
        if item.get_closest_marker("archive") and not config.getoption("--run-archive-tests"):
            item.add_marker(pytest.mark.skip(reason=(
                "Original research-report edition is an optional historical archive, omitted from the public source tree; "
                "restore it and pass --run-archive-tests. Numerical model tests remain enabled.")))


@pytest.fixture(scope="session", autouse=True)
def _synthetic_mode():
    """Force the synthetic path so the suite is reproducible anywhere."""
    os.environ["RVLAB_FORCE_SYNTHETIC"] = "1"
    yield
    os.environ.pop("RVLAB_FORCE_SYNTHETIC", None)


@pytest.fixture(scope="session")
def panel():
    from rvlab.data import load_smile_panel
    return load_smile_panel()


@pytest.fixture(scope="session")
def intraday():
    """A continuous 1-minute panel with 3 expiries per bar and no overnight gaps."""
    import numpy as np
    import pandas as pd

    ts = pd.date_range("2026-01-05 09:30", periods=600, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ts": np.repeat(ts, 3),
        "expiry": np.tile(["a", "b", "c"], len(ts)),
        "x": rng.standard_normal(3 * len(ts)),
    })
