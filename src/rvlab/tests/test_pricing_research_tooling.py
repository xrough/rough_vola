"""Consolidated entry points cannot execute or mislabel superseded evidence."""

import json
import sys
from types import SimpleNamespace

import pytest

from rvlab.experiments import pricing
from rvlab.experiments.pricing_presentation import load_recorded_results, refresh_presentations


@pytest.mark.parametrize("selection,indexes", [
    ("pricing", [0]), ("calibration", [1]), ("spy", [2]),
    ("quick", [0, 1]), ("all", [0, 1, 2]),
])
def test_cli_selects_only_active_notebooks(selection, indexes, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pricing, "execute_notebook", lambda *args: calls.append(args))
    assert pricing.main(["--execute", selection, "--output-dir", str(tmp_path)]) == 0
    assert calls == [(pricing.ROOT / "notebooks/pricing" / pricing.NOTEBOOKS[i],
                      tmp_path / pricing.NOTEBOOKS[i]) for i in indexes]


def test_old_cli_selection_fails_instead_of_rerunning_the_preserved_series():
    with pytest.raises(SystemExit) as error:
        pricing.main(["--execute", "american"])
    assert error.value.code == 2


def test_calibration_cli_allows_an_explicit_cold_run_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(pricing, "execute_notebook", lambda *a, **k: calls.append((a, k)))
    assert pricing.main(["--execute", "calibration", "--timeout", "7200"]) == 0
    assert calls[0][1] == {"timeout": 7200}
    with pytest.raises(SystemExit):
        pricing.main(["--execute", "calibration", "--timeout", "0"])


def test_facades_delegate_to_one_research_module_without_reinterpreting_results(monkeypatch):
    calls = []
    module = SimpleNamespace(**{name: (lambda context, _name=name, **kwargs:
        calls.append((_name, context, kwargs)) or {"study": _name})
        for name in ("pricing_research", "calibration_research", "spy_research")})
    monkeypatch.setitem(sys.modules, "rvlab.experiments._pricing_research", module)
    monkeypatch.setattr(pricing, "_finish", lambda context, result: result)
    for name in ("pricing", "calibration", "spy"):
        assert getattr(pricing, f"run_{name}_research")("context", seed=42) == {"study": f"{name}_research"}
    assert calls == [(f"{name}_research", "context", {"seed": 42}) for name in ("pricing", "calibration", "spy")]


def test_new_outputs_have_an_independent_namespace(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(pricing, "ROOT", tmp_path)
    for name in pricing.NOTEBOOKS:
        assert pricing.setup_demo(name).output_dir == (
            tmp_path / "notebooks/_output/pricing/research_v1" / name.removesuffix(".ipynb"))


def _manifest(tmp_path, stem, **updates):
    folder = tmp_path / stem
    folder.mkdir(exist_ok=True)
    manifest = {"notebook": stem, "tables": {}, "figures": [],
        "evidence_version": pricing.RESEARCH_EVIDENCE_VERSION,
        "data_kind": "synthetic", "backend": "mixed_european_fourier_american_fd",
        "exercise_style": ["european", "american"]}
    manifest.update(updates)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("stem", sorted(pricing.RESEARCH_STEMS))
def test_new_ids_reject_previous_artifact_versions(tmp_path, stem):
    _manifest(tmp_path, stem, evidence_version="american_v1")
    with pytest.raises(ValueError, match="current pricing_research_v1"):
        load_recorded_results(stem, artifact_root=tmp_path)


@pytest.mark.parametrize("stem", ["01_pricing_and_exercise", "02_synthetic_calibration"])
def test_synthetic_evidence_permits_mixed_styles_but_rejects_spy_inputs(tmp_path, stem):
    current = _manifest(tmp_path, stem)
    assert load_recorded_results(stem, artifact_root=tmp_path)["manifest"] == current
    _manifest(tmp_path, stem, spy_inputs={"source": "market"})
    with pytest.raises(ValueError, match="synthetic observations only"):
        load_recorded_results(stem, artifact_root=tmp_path)


def test_new_spy_id_requires_american_backend_even_when_version_is_current(tmp_path):
    stem = "03_spy_calibration"
    _manifest(tmp_path, stem)
    with pytest.raises(ValueError, match="American backend"):
        load_recorded_results(stem, artifact_root=tmp_path)
    current = _manifest(tmp_path, stem, backend="american_vanilla_fd", exercise_style="american",
                        data_kind="market", status="blocked_inputs")
    assert load_recorded_results(stem, artifact_root=tmp_path)["manifest"] == current


def test_manifest_identity_is_checked_before_reading_artifact_tables(tmp_path):
    _manifest(tmp_path, "03_spy_calibration", notebook="02_spy_pricing")
    with pytest.raises(ValueError, match="different notebook"):
        load_recorded_results("03_spy_calibration", artifact_root=tmp_path)


def test_supported_executors_preserve_archive_sources_and_side_outputs():
    pytest.importorskip("nbformat")
    archived = pricing.ROOT / "notebooks/legacy/pricing/01_synthetic_pricing.ipynb"
    with pytest.raises(ValueError, match="preserved evidence"):
        pricing.execute_notebook(archived)
    with pytest.raises(ValueError, match="preserved unchanged"):
        refresh_presentations([archived])
