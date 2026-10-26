"""Portable filesystem defaults used by notebooks and installed rvlab packages."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_output_dir_environment_override_is_resolved(tmp_path):
    output = tmp_path / "artifacts"
    src = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["RVLAB_OUTPUT_DIR"] = str(output)
    env["PYTHONPATH"] = str(src)

    result = subprocess.run(
        [sys.executable, "-c", "from rvlab.config import OUTPUT_DIR; print(OUTPUT_DIR)"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == output.resolve()


def test_notebook_runner_rejects_a_mismatched_selected_kernel(tmp_path):
    """The dependency preflight and Jupyter kernel must be one interpreter."""
    bash = shutil.which("bash")
    other_command = shutil.which("false")
    notebook_modules = (
        "ipykernel", "nbconvert", "jupyter_client", "numpy", "scipy",
        "pandas", "sklearn", "statsmodels", "matplotlib", "joblib",
        "seaborn", "pyarrow", "optuna", "tidyfinance", "xgboost",
        "lightgbm",
    )
    unusable = []
    for name in notebook_modules:
        try:
            importlib.import_module(name)
        except Exception:
            unusable.append(name)
    if not bash or not other_command or unusable:
        pytest.skip("requires bash and the complete notebook extra")

    data_dir = tmp_path / "jupyter"
    kernel_dir = data_dir / "kernels" / "rvlab-mismatched-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [other_command, "-f", "{connection_file}"],
        "display_name": "Deliberately mismatched test kernel",
        "language": "python",
    }))

    repo = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update({
        "JUPYTER_PATH": str(data_dir),
        "RVLAB_PYTHON": sys.executable,
        "RVLAB_KERNEL": "rvlab-mismatched-test",
    })
    result = subprocess.run(
        [bash, str(repo / "scripts" / "run_notebooks.sh"), "--legacy",
         str(tmp_path / "never-executed.ipynb")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not RVLAB_PYTHON" in result.stdout + result.stderr
