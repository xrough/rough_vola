"""Portable public runner contracts, with no real kernels or archived payloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import nbformat
import pytest

REPO = Path(__file__).resolve().parents[3]
NAMES = ("01_pricing_and_exercise.ipynb", "02_synthetic_calibration.ipynb", "03_spy_pilot.ipynb")


def runner():
    spec = importlib.util.spec_from_file_location("public_notebook_runner", REPO/"scripts/run_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_runner_lists_exactly_three_notebooks_without_execution(monkeypatch, capsys):
    module = runner()
    monkeypatch.setattr(module, "execute", lambda *a,**k: pytest.fail("listing launched a notebook"))
    assert module.main(["--list"]) == 0
    assert [Path(x).name for x in capsys.readouterr().out.splitlines()] == list(NAMES)


def test_public_runner_rejects_unknown_notebooks_and_nonpositive_timeout():
    module = runner()
    for argv in (["unknown.ipynb"], ["--timeout","0"], ["--timeout","-1"]):
        with pytest.raises(SystemExit) as exc:
            module.main(argv)
        assert exc.value.code == 2


def test_public_runner_executes_all_or_an_explicit_subset_to_detached_outputs(tmp_path, monkeypatch):
    module = runner()
    calls = []
    def execute(path, output, timeout):
        calls.append((path.name,output.name,timeout))
        return {"notebook":path.name,"error_outputs":0}
    monkeypatch.setattr(module,"execute",execute)
    assert module.main(["--all","--output-dir",str(tmp_path),"--timeout","37"]) == 0
    assert calls == [(name,name,37)for name in NAMES]
    receipt=json.loads((tmp_path/"notebook_execution.json").read_text())
    assert not receipt["private_data_included"] and not receipt["historical_campaigns_recomputed"]
    calls.clear()
    assert module.main([NAMES[1],"--output-dir",str(tmp_path)]) == 0
    assert [x[0]for x in calls] == [NAMES[1]]


def test_failed_notebook_is_not_recorded_as_a_successful_execution(tmp_path, monkeypatch):
    module = runner()
    def fail(*a,**k):raise RuntimeError("deliberate notebook failure")
    monkeypatch.setattr(module,"execute",fail)
    with pytest.raises(RuntimeError,match="deliberate"):
        module.main(["--all","--output-dir",str(tmp_path)])
    assert not (tmp_path/"notebook_execution.json").exists()


def test_executor_refuses_external_notebook_paths(tmp_path):
    module = runner()
    with pytest.raises(ValueError,match="Only the three public"):
        module.execute(tmp_path/NAMES[0])


def test_executor_pins_interpreter_and_preserves_source_when_writing_a_copy(tmp_path, monkeypatch):
    module=runner()
    monkeypatch.setattr(module,"ROOT",tmp_path)
    # The public runner's selected location remains its own responsibility.
    source=tmp_path/"notebooks"/NAMES[0]
    source.parent.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("answer = 42")]),source)
    before=source.read_bytes()
    monkeypatch.setattr(module,"source_hashes",lambda:{"unit_source":"unchanged"})
    observed=[]
    class Manager:
        def __init__(self,*,kernel_name,kernel_spec_manager):
            spec=Path(kernel_spec_manager.kernel_dirs[0])/kernel_name/"kernel.json"
            config=json.loads(spec.read_text());observed.append(config)
            self.has_kernel=False
    class Client:
        def __init__(self,notebook,**kwargs):self.notebook=notebook
        def execute(self,**kwargs):return self.notebook
    import jupyter_client,nbclient
    monkeypatch.setattr(jupyter_client,"AsyncKernelManager",Manager)
    monkeypatch.setattr(nbclient,"NotebookClient",Client)
    destination=tmp_path/"executed"/NAMES[0]
    result=module.execute(source,destination,timeout=12)
    assert observed[0]["argv"][:3]==[sys.executable,"-m","ipykernel_launcher"]
    assert observed[0]["env"]["PYTHONPATH"].split(__import__('os').pathsep)[0]==str(tmp_path/"src")
    assert observed[0]["env"]["OPENBLAS_NUM_THREADS"]=="1"
    assert source.read_bytes()==before and destination.is_file()
    assert result["error_outputs"]==0 and result["kernel_pinned_to_selected_interpreter"]
    saved=nbformat.read(destination,4)
    assert saved.metadata.public_execution.private_archive_required is False
    assert saved.metadata.public_execution.source_sha256=={"unit_source":"unchanged"}
