"""Cold CLI provenance/phase tests; never execute the 40,960-path study."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from rvlab.experiments import _pricing_rough_benchmark as b


@pytest.fixture
def cold(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[3]/"examples/rough_benchmark_reproduce.py"
    spec = importlib.util.spec_from_file_location("cold_reproduction_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path/"repo"
    allowed = root/"notebooks/_output/pricing/first_stage_v1"
    allowed.mkdir(parents=True)
    pins = {}
    for i in range(61):
        name = f"src/historical_{i}.py"
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"historical_source = {i}\n")
        pins[name] = b.sha256(path)
    for name in module.NEW_SOURCES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test_source_snapshot = True\n")
    definitions = b.benchmark_trials()
    train = {row["policy_id"]: 1146005042+3000000*i for i, row in enumerate(definitions)
        if row["policy_id"] == row["trial_id"]}
    trials = [{**row, "index": i, "training_seed": train[row["policy_id"]],
        "training_paths": 2048, "valuation_paths": 2048,
        "valuation_seeds": [1147005042+3000000*i, 1148005042+3000000*i]}
        for i, row in enumerate(definitions)]
    original = {"trials": trials, "source_sha256": pins, "parameters": b.PARAMETERS,
        "trial_count": 10, "total_valuation_paths": 40960, "kernels": {"4": {}, "8": {}}}
    path = root/"approved_design.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "ALLOWED_ROOT", allowed)
    monkeypatch.setattr(module, "APPROVED_DESIGN", path)
    monkeypatch.setattr(module, "APPROVED_SHA", b.sha256(path))
    return module, allowed/"cold_test", path


def prepared(cold):
    module, folder, approved = cold
    module.prepare(folder, approved)
    return module, folder, approved


def test_destination_is_explicit_new_cold_child_and_rejects_escaped_symlink(cold, tmp_path):
    module, folder, _ = cold
    assert module.destination(folder) == folder
    for value in (tmp_path/"elsewhere", module.ALLOWED_ROOT/"independent_v1",
            module.ALLOWED_ROOT/"cold_", folder/"nested"):
        with pytest.raises(ValueError, match="cold_"):
            module.destination(value)
    outside = tmp_path/"outside"
    outside.mkdir()
    escaped = module.ALLOWED_ROOT/"cold_escape"
    escaped.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        module.destination(escaped)


def test_prepare_needs_only_design_and_sources_not_handmade_guard_folders(cold):
    module, folder, approved = prepared(cold)
    design = module.load(folder)
    assert design["maximum_workers"] == 1 and design["hand_generated_guard_folders_required"] is False
    assert design["same_seed_numerical_reproduction"] is True
    assert design["new_independent_validation"] is design["eligible_for_report_acceptance"] is False
    assert (folder/"approved_design.json").read_bytes() == approved.read_bytes()
    assert len(design["source_sha256"]) == 63
    assert not list(module.ALLOWED_ROOT.glob("**/spent_preflight_v1"))
    with pytest.raises(FileExistsError):
        module.prepare(folder, approved)


@pytest.mark.parametrize("kind", ["approved", "live_source", "snapshot", "role", "tolerance", "environment"])
def test_changed_sources_inputs_and_validation_role_fail_closed(cold, kind):
    module, folder, _ = prepared(cold)
    if kind == "approved":
        (folder/"approved_design.json").write_text("{}")
    elif kind == "live_source":
        (module.ROOT/"src/historical_0.py").write_text("changed\n")
    elif kind == "snapshot":
        (folder/"source/src/historical_0.py").write_text("changed\n")
    else:
        path = folder/"design_before_reproduction.json"
        value = json.loads(path.read_text())
        if kind == "role": value["new_independent_validation"] = 1
        elif kind == "tolerance": value["price_atol"] = .01
        else: value["software_environment"]["packages"]["numpy"] = "changed"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.load(folder)


def test_exclusive_phase_and_stale_lock_are_never_silently_replaced(cold):
    module, folder, _ = prepared(cold)
    with module.phase_lock(folder):
        original = (folder/"reproduction.lock").read_bytes()
        with pytest.raises(FileExistsError):
            with module.phase_lock(folder):
                pytest.fail("concurrent phase entered")
        assert (folder/"reproduction.lock").read_bytes() == original
    assert not (folder/"reproduction.lock").exists()
    (folder/"reproduction.lock").write_text("retained stale-lock evidence")
    with pytest.raises(FileExistsError):
        with module.phase_lock(folder):
            pass
    assert (folder/"reproduction.lock").read_text() == "retained stale-lock evidence"


def test_valuation_requires_all_training_and_physical_rule_is_not_retrained(cold):
    module, folder, _ = prepared(cold)
    with pytest.raises(ValueError, match="all nine"):
        module.main(["value", "--all", "--output-dir", str(folder)])
    assert not list(folder.glob("trial_*"))
    assert not (folder/"reproduction.lock").exists()
    with pytest.raises(ValueError, match="reuses"):
        module.main(["train", "--index", "6", "--output-dir", str(folder)])
    assert not (folder/"training_6").exists()


def test_failed_attempt_uses_supported_serializer_and_cannot_be_retried(cold, monkeypatch):
    module, folder, _ = prepared(cold)
    marker, calls = object(), []
    monkeypatch.setattr(module.b, "train_benchmark_policy", lambda *args: marker)
    monkeypatch.setattr(module.b, "ExponentialKernel", lambda **kwargs: object())
    def supported(policy, target):
        calls.append((policy, target))
        raise RuntimeError("explicit implementation fixture; no simulation")
    monkeypatch.setattr(module, "save_benchmark_policy", supported)
    module.main(["train", "--index", "2", "--output-dir", str(folder)])
    path = folder/"training_2/manifest.json"
    record = json.loads(path.read_text())
    assert calls[0][0] is marker and record["status"] == "failed"
    assert record["acceptance_evidence"] is False and "implementation fixture" in record["message"]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(["train", "--index", "2", "--output-dir", str(folder)])
    assert path.read_bytes() == original and len(calls) == 1


@pytest.mark.parametrize("empty_exception_message", [False, True])
def test_all_failed_fixture_closes_twenty_attempts_without_any_pricing(cold, monkeypatch,
        empty_exception_message):
    module, folder, _ = prepared(cold)
    design = module.load(folder)
    for index, trial in enumerate(design["trials"]):
        if index == 6: continue
        target = folder/f"training_{index}"
        target.mkdir()
        module.write(target/"manifest.json", {"index": index, "training_seed": trial["training_seed"],
            "training_paths": 2048, "source_sha256": design["source_sha256"],
            "design_sha256": b.sha256(folder/"design_before_reproduction.json"),
            "status": "failed", "message": "non-numerical test fixture",
            "completed_at": datetime.now(timezone.utc).isoformat()})
    def no_pricing(*args, **kwargs):
        pytest.fail("blocked reproduction attempted to generate paths")
    monkeypatch.setattr(module.b, "value_benchmark_policy", no_pricing)
    module.main(["value", "--all", "--output-dir", str(folder)])
    assert len(list(folder.glob("trial_*/manifest.json"))) == 20
    if empty_exception_message:
        # An exception such as RuntimeError() has no text. Its failed outcome
        # must remain truthy to the statistics layer, never become a fake
        # successful stream with missing numerical arrays.
        for path in folder.glob("trial_*/manifest.json"):
            record = json.loads(path.read_text())
            record.update(status="failed", error_type="RuntimeError", message="")
            path.write_text(json.dumps(record))
    module.main(["collect", "--output-dir", str(folder)])
    manifest = json.loads((folder/"manifest.json").read_text())
    assert manifest["complete"] and manifest["completed_paths"] == 0
    assert manifest["all_twenty_attempts_retained"] and len(manifest["records"]) == 20
    assert manifest["new_independent_validation"] is False
    if empty_exception_message:
        assert all(row["status"] == "failed" and row["message"] == ""
            for row in manifest["records"])
        assert manifest["tables"]["failures"] == 20
    assert (folder/"preservation_inventory.json").is_file()
    assert not (folder/"reproduction.lock").exists()
    with pytest.raises(FileExistsError, match="closed"):
        module.main(["value", "--index", "0", "--stream", "0", "--output-dir", str(folder)])


@pytest.mark.parametrize("args", [
    ["train", "--index", "0", "--stream", "1"],
    ["value", "--index", "0"], ["value", "--all", "--stream", "0"],
    ["prepare", "--index", "0"], ["collect", "--all"],
    ["train", "--all", "--index", "0"], ["value", "--index", "10", "--stream", "0"]])
def test_cli_rejects_ambiguous_and_out_of_scope_attempt_selection(cold, args):
    module, folder, _ = cold
    with pytest.raises(SystemExit) as error:
        module.main([*args, "--output-dir", str(folder)])
    assert error.value.code == 2
    assert not folder.exists()
