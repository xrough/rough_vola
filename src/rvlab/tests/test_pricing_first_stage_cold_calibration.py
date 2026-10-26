"""Cold reproduction is explicit, bounded and isolated from report/history paths."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path=Path(__file__).resolve().parents[3]/'examples/first_stage_cold_calibration.py'
    spec=importlib.util.spec_from_file_location('first_stage_cold_calibration',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_cold_recipe_retains_declared_budgets_and_disables_cache_substitution(tmp_path,monkeypatch):
    module=_module();monkeypatch.setattr(module,'ROOT',tmp_path)
    calls=[];context=SimpleNamespace()
    monkeypatch.setattr(module,'setup_first_stage',lambda *a,**k:context)
    monkeypatch.setattr(module,'run_calibration_research',lambda ctx,**kw:calls.append((ctx,kw)) or {'cold':True})
    destination=tmp_path/'notebooks/_output/pricing/first_stage_v1/cold_calibration_v1'
    assert module.run_cold(destination)=={'cold':True}
    ctx,kw=calls[0];assert ctx is context and kw['seed']==42
    assert kw['joint_max_nfev']==80 and kw['joint_continuation_max_nfev']==160
    assert kw['curve_diagnosis_max_nfev']==40 and kw['maturity_max_nfev']==80
    assert all(kw[name] is False for name in ('joint_reuse_checkpoints','curve_diagnosis_reuse','maturity_reuse','run_controlled_american','run_extended'))
    assert (kw['joint_fit_config'].spot_steps,kw['joint_fit_config'].variance_steps,kw['joint_fit_config'].time_steps)==(192,48,180)
    assert kw['maturity_fit_config'].variance_steps==64
    assert kw['joint_generation_config'].spot_steps==640 and kw['joint_generation_config'].time_steps==600


def test_cold_recipe_cannot_overwrite_old_evidence_or_an_active_report(tmp_path,monkeypatch):
    module=_module();monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'setup_first_stage',lambda *a,**k:pytest.fail('must reject before writing'))
    allowed=tmp_path/'notebooks/_output/pricing/first_stage_v1'
    for path in (tmp_path/'notebooks/_output/pricing/research_v1/cold',allowed,
            allowed/'02_synthetic_calibration'):
        with pytest.raises(ValueError):module.run_cold(path)
    existing=allowed/'existing';existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):module.run_cold(existing)
