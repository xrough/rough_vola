"""Portable notebook execution and detached diagnostic artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import socket
import sys

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import pricing


@pytest.fixture(autouse=True)
def _noninteractive_backend():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)


def test_manifest_json_conversion_is_finite_recursive_and_detached():
    @dataclass(frozen=True)
    class Params:
        v0: float
    original = {"params": Params(.04), "values": np.array([1., np.nan, np.inf, -np.inf]),
        "scalar_array": np.array(2.5), "float": np.float64(np.nan),
        "missing": [pd.NA, pd.NaT], "path": Path("somewhere"),
        "timestamp": pd.Timestamp("2025-08-07 19:45Z"), "observation_date": date(2025, 8, 5)}
    clean = pricing._jsonable(original)
    encoded = json.dumps(clean, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["params"] == {"v0": .04}
    assert decoded["values"] == [1., None, None, None]
    assert decoded["scalar_array"] == 2.5
    assert decoded["missing"] == [None, None]
    assert decoded["float"] is None
    assert decoded["observation_date"] == "2025-08-05"
    original["values"][0] = 100.
    assert clean["values"][0] == 1.


@pytest.mark.parametrize("name", ["..", ".", "../escape", "nested/escape", "/escape", "", "demo.csv"])
def test_setup_rejects_names_that_are_not_notebook_stems(name, tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="simple filename|stem"):
        pricing.setup_demo(name, output_dir=tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_artifacts_detach_tables_manifest_and_figures(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    context = pricing.setup_demo("small_demo.ipynb", output_dir=tmp_path / "artifacts")
    table = pd.DataFrame({"price": [1.25, 2.5], "unstable": [False, True]})
    original = table.copy(deep=True)
    figure, axis = plt.subplots()
    axis.plot([0, 1], [1, 2])
    result = {"tables": {"fit_prices": table}, "figures": {"smile": figure},
        "manifest": {"diagnostic": np.nan, "bounds": {"v0": [.0025, .25]}}}
    try:
        saved = pricing._finish(context, result)
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(table, original)
    pd.testing.assert_frame_equal(pd.read_csv(context.output_dir / "fit_prices.csv"), original)
    assert (context.output_dir / "smile.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = json.loads((context.output_dir / "manifest.json").read_text())
    assert manifest["diagnostic"] is None
    assert manifest["interpreter"] == sys.executable
    assert manifest["tables"] == {"fit_prices": 2}
    assert manifest["figures"] == ["smile"]
    assert manifest["source_sha256"] and all(len(value) == 64 for value in manifest["source_sha256"].values())
    table.loc[0, "price"] = 999.
    saved["manifest"]["bounds"]["v0"][0] = 99.
    assert pd.read_csv(context.output_dir / "fit_prices.csv").price.iloc[0] == 1.25
    assert json.loads((context.output_dir / "manifest.json").read_text())["bounds"]["v0"][0] == .0025


@pytest.mark.parametrize("kind", ["tables", "figures"])
def test_artifact_names_cannot_escape_destination(tmp_path, kind):
    pytest.importorskip("matplotlib")
    context = pricing.setup_demo("small_demo", output_dir=tmp_path / "artifacts")
    class Figure:
        def savefig(self, path, **kwargs):
            Path(path).write_bytes(b"should not be written")
    result = {"tables": {}, "figures": {}, "manifest": {}}
    result[kind]["../escape"] = pd.DataFrame({"price": [1.]}) if kind == "tables" else Figure()
    with pytest.raises(ValueError, match="artifact|filename|stem"):
        pricing._finish(context, result)
    assert not (tmp_path / "escape.csv").exists()
    assert not (tmp_path / "escape.png").exists()


def _notebook_dependencies():
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    return nbformat


def test_runner_cleans_pinned_temporary_kernel_on_execution_failure(tmp_path, monkeypatch):
    nbformat = _notebook_dependencies()
    import nbclient
    import jupyter_client
    source = tmp_path / "source.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")]), source)
    original_bytes = source.read_bytes()
    captured = {}
    class Manager:
        has_kernel = True
        def __init__(self, *, kernel_name, kernel_spec_manager):
            captured["manager"] = self
            captured["kernel_name"] = kernel_name
            captured["directory"] = Path(kernel_spec_manager.kernel_dirs[0])
            captured["spec"] = json.loads((captured["directory"] / kernel_name / "kernel.json").read_text())
        def shutdown_kernel(self, *, now):
            captured["shutdown"] = now
            self.has_kernel = False
    class Client:
        def __init__(self, notebook, **kwargs):
            captured["options"] = kwargs
        def execute(self, **kwargs):
            captured["cleanup_kc"] = kwargs.get("cleanup_kc")
            raise RuntimeError("deliberate notebook failure")
    monkeypatch.setattr(jupyter_client, "AsyncKernelManager", Manager)
    monkeypatch.setattr(nbclient, "NotebookClient", Client)
    destination = tmp_path / "executed.ipynb"
    with pytest.raises(RuntimeError, match="deliberate notebook failure"):
        pricing.execute_notebook(source, destination, timeout=7)
    assert captured["spec"]["argv"][0] == sys.executable
    assert captured["spec"]["argv"][1:3] == ["-m", "ipykernel_launcher"]
    assert captured["options"]["allow_errors"] is False
    assert captured["options"]["timeout"] == 7
    assert captured["cleanup_kc"] is True
    assert captured["shutdown"] is True
    assert not captured["directory"].exists()
    assert source.read_bytes() == original_bytes
    assert not destination.exists()


def _require_kernel_ports():
    # Managed filesystem sandboxes sometimes also forbid the local sockets
    # Jupyter needs. Such environments can still run the mocked cleanup test.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("sandbox prohibits local kernel ports; run outside the sandbox")


def test_fresh_kernels_pin_interpreter_and_preserve_source(tmp_path):
    nbformat = _notebook_dependencies()
    _require_kernel_ports()
    source = tmp_path / "fresh.ipynb"
    cell = """import json, os, sys
assert 'sentinel_from_previous_run' not in globals()
sentinel_from_previous_run = 42
print(json.dumps({'interpreter': sys.executable, 'cwd': os.getcwd(),
                  'backend': os.environ['MPLBACKEND']}))
"""
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell)]), source)
    original = source.read_bytes()
    for i in range(2):
        destination = tmp_path / "executed" / f"run{i}.ipynb"
        assert pricing.execute_notebook(source, destination, timeout=20) == destination.resolve()
        notebook = nbformat.read(destination, as_version=4)
        assert notebook.cells[0].execution_count == 1
        output = json.loads("".join(item.get("text", "") for item in notebook.cells[0].outputs))
        assert Path(output["interpreter"]) == Path(sys.executable)
        assert Path(output["cwd"]) == pricing.ROOT
        assert output["backend"] == "Agg"
        assert notebook.metadata.rvlab_execution.interpreter == sys.executable
        assert notebook.metadata.kernelspec.name == "python3"
    assert source.read_bytes() == original


@pytest.mark.parametrize("code,exception", [
    ("raise ValueError('intentional failure')", "CellExecutionError"),
    ("import time; time.sleep(10)", "CellTimeoutError"),
])
def test_notebook_failure_does_not_overwrite_source_or_existing_output(tmp_path, code, exception):
    nbformat = _notebook_dependencies()
    _require_kernel_ports()
    from nbclient import exceptions
    source, destination = tmp_path / "failure.ipynb", tmp_path / "existing.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), source)
    source_bytes = source.read_bytes()
    destination.write_text("previous successful output")
    with pytest.raises(getattr(exceptions, exception)):
        pricing.execute_notebook(source, destination, timeout=1)
    assert source.read_bytes() == source_bytes
    assert destination.read_text() == "previous successful output"
