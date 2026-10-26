"""Protocol guards for read-only Bellman evidence, source and model pins."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rvlab.experiments import _pricing_rough_bellman_collection as collection


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_protocol(tmp_path, monkeypatch):
    root = tmp_path/"repo"; folder = tmp_path/"output/rough_bellman_policy_v1/one_step_v1"
    names = {"src/rvlab/pricing/one.py", "examples/rough_tail_attribution.py", "examples/rough_bellman_policy.py"}
    names.update("src/rvlab/experiments/"+name for name in (
        "_pricing_rough_tail_switch.py", "_pricing_rough_tail_switch_guarded.py", "_pricing_rough_tail_policy.py",
        "_pricing_rough_local_control.py", "_pricing_rough_node_refinement_fast.py", "_pricing_rough_exercise_block.py",
        "_pricing_rough_large_policy.py", "_pricing_rough_frozen_policy.py", "_pricing_rough_policy_improvement.py",
        "_pricing_rough_bellman_policy.py"))
    pins = {}
    for name in names:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# sealed source\n")
        snapshot = folder/"source"/name; snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes()); pins[name] = digest(path)
    monkeypatch.setattr(collection, "ROOT", root)
    return root, folder, {"source_sha256": pins}


def test_exact_current_and_snapshotted_numerical_source_set_passes(source_protocol):
    _, folder, design = source_protocol
    collection._sources(folder, design)


@pytest.mark.parametrize("tamper", ["omitted", "unexpected", "current", "snapshot", "digest", "new_pricing_file"])
def test_changed_omitted_or_replaced_sources_are_rejected(source_protocol, tamper):
    root, folder, design = source_protocol; name = "src/rvlab/experiments/_pricing_rough_bellman_policy.py"
    if tamper == "omitted": design["source_sha256"].pop(name)
    if tamper == "unexpected": design["source_sha256"]["../outside.py"] = "0"*64
    if tamper == "current": (root/name).write_text("# changed runtime\n")
    if tamper == "snapshot": (folder/"source"/name).write_text("# changed preserved source\n")
    if tamper == "digest": design["source_sha256"][name] = "0"*64
    if tamper == "new_pricing_file": (root/"src/rvlab/pricing/unsealed.py").write_text("# new\n")
    with pytest.raises(ValueError, match="source"):
        collection._sources(folder, design)


@pytest.fixture
def input_protocol(tmp_path, monkeypatch):
    from rvlab.experiments import _pricing_rough_large_policy as loader
    output = tmp_path/"output"; folder = output/"rough_bellman_policy_v1/one_step_v1"
    folder.mkdir(parents=True)
    model = output/"rough_larger_policy_v1/32768_paths_two_starts"
    prior_path = output/"rough_policy_validation_v1/local_control_v1/controls_before_final.json"
    grid_path = output/"rough_gap_time_refinement_v1/stock_grid_control_v2/controls_before_final.json"
    tail_path = output/"rough_tail_attribution_v1/joint_switch_v1/selection_before_final.json"
    prior = {"selected": "prior", "controls": {"prior": {"weights": [.1, .2, .3]}}}
    grid = {"selected": "grid", "controls": {"grid": {"weights": [.01]*24}}}
    control = {"family": "extended", "weights": [0., 0., 0., 0., .03, .02, 0.]}
    tail = {"selected_control": "extended 1 bins penalty0", "controls": {"extended 1 bins penalty0": control}}
    for path, value in ((model/"frozen_policy_before_final.json", {}), (prior_path, prior), (grid_path, grid), (tail_path, tail)):
        write(path, value)
    (model/"frozen_policy_coefficients.npz").write_bytes(b"opaque sealed coefficients; loader replaced only in protocol fixture")
    paths = [model/"frozen_policy_before_final.json", model/"frozen_policy_coefficients.npz", prior_path, grid_path, tail_path]
    bundle = {"selected_name": "dense tensor ridge1e-5", "manifest": {"all_selection_training_seeds": [10, 20]}}
    monkeypatch.setattr(loader, "load_larger_policy_bundle", lambda path: bundle)
    design = {"input_sha256": {str(path.relative_to(output)): digest(path) for path in paths},
        "forbidden_prior_seeds": [10, 20, 1043005042, 1054005042, 1055005042],
        "common_control_name": tail["selected_control"], "common_control": deepcopy(control),
        "frozen_regression_policy_name": bundle["selected_name"]}
    return output, folder, design, bundle, prior_path, grid_path, tail_path


def test_model_and_common_control_pins_preserve_old_rule_without_new_fitting(input_protocol):
    _, folder, design, bundle, *_ = input_protocol; entries = []
    actual, prior, grid, tail = collection._frozen_inputs(folder, design, entries)
    assert actual is bundle and len(entries) == 5
    assert prior["weights"] == [.1, .2, .3] and grid["weights"] == [.01]*24
    assert tail["selected_control"] == design["common_control_name"]


@pytest.mark.parametrize("tamper", ["omitted", "digest", "path_escape", "forbidden", "seed_overlap",
    "control_name", "control_weight", "policy_name", "previous_selection"])
def test_input_identity_and_independent_stream_guards(input_protocol, tamper):
    output, folder, design, bundle, _, _, tail_path = input_protocol
    if tamper == "omitted": design["input_sha256"].pop(next(iter(design["input_sha256"])))
    if tamper == "digest": design["input_sha256"][next(iter(design["input_sha256"]))] = "0"*64
    if tamper == "path_escape": design["input_sha256"]["../outside.json"] = "0"*64
    if tamper == "forbidden": design["forbidden_prior_seeds"].remove(1054005042)
    if tamper == "seed_overlap": bundle["manifest"]["all_selection_training_seeds"].append(collection.TRAIN_SEED)
    if tamper == "control_name": design["common_control_name"] = "poststop 1 bins penalty0"
    if tamper == "control_weight": design["common_control"]["weights"][4] += .01
    if tamper == "policy_name": design["frozen_regression_policy_name"] = "different regression"
    if tamper == "previous_selection":
        tail = json.loads(tail_path.read_text()); tail["selected_control"] = "different control"; write(tail_path, tail)
        design["input_sha256"][str(tail_path.relative_to(output))] = digest(tail_path)
    with pytest.raises(ValueError): collection._frozen_inputs(folder, design, [])


class ReachedNextAudit(RuntimeError): pass


@pytest.fixture
def campaign_protocol(tmp_path, monkeypatch):
    folder = tmp_path/"output/rough_bellman_policy_v1/one_step_v1"; folder.mkdir(parents=True)
    design = {"version": "rough_bellman_one_step_v1", "training": {"seed": collection.TRAIN_SEED, "paths": 2048, "role": "selection only"},
        "final_streams": [{"seed": seed, "paths": 4096} for seed in collection.FINAL_SEEDS],
        "physical_steps": 128, "exercise_intervals": 64, "all_paths_retained": True,
        "candidate_specs": collection._expected_specs(), "orders": [8, 16, 32], "batch_size": 32,
        "confidence_multiplier": 1.96, "tolerances_changed": False,
        "price_allowance": ".001+1e-4*abs(V)", "delta_allowance": ".002+.005*abs(delta)"}
    seal = {"candidate_specs": collection._expected_specs(), "policy_order": list(collection._expected_specs()),
        "selection_seed": collection.TRAIN_SEED, "final_paths_generated_before_seal": False}
    state = {"manifest": {**deepcopy(design), "complete": True, "completed_final_paths": 8192}}
    monkeypatch.setattr(collection, "_manifest", lambda path, inventory: (state["manifest"], []))
    monkeypatch.setattr(collection, "_sources", lambda *args: None)
    def next_audit(*args): raise ReachedNextAudit()
    monkeypatch.setattr(collection, "_frozen_inputs", next_audit)
    return folder, design, seal, state


def save_campaign(folder, design, seal, state):
    write(folder/"design_before_training.json", design)
    seal["design_sha256"] = digest(folder/"design_before_training.json")
    write(folder/"selection_before_final.json", seal)
    state["manifest"]["selection_sha256"] = digest(folder/"selection_before_final.json")


def test_declared_campaign_reaches_independent_model_input_audit(campaign_protocol):
    folder, design, seal, state = campaign_protocol; save_campaign(*campaign_protocol)
    with pytest.raises(ReachedNextAudit): collection.collect_bellman_campaign(folder, compute_figures=False)


@pytest.mark.parametrize("tamper", ["physical_steps", "exercise_steps", "training_seed", "training_role", "final_seed", "final_budget",
    "omit_candidate", "blend", "quadrature", "all_paths", "price_tolerance", "delta_tolerance", "confidence", "complete", "missing_paths", "manifest_drift"])
def test_changed_predeclared_law_selection_scope_or_tolerances_fail_before_loading_paths(campaign_protocol, tamper):
    folder, design, seal, state = campaign_protocol
    if tamper == "physical_steps": design["physical_steps"] = 256
    if tamper == "exercise_steps": design["exercise_intervals"] = 128
    if tamper == "training_seed": design["training"]["seed"] = collection.FINAL_SEEDS[0]
    if tamper == "training_role": design["training"]["role"] = "model fitting"
    if tamper == "final_seed": design["final_streams"][0]["seed"] = collection.TRAIN_SEED
    if tamper == "final_budget": design["final_streams"][0]["paths"] = 2048
    if tamper == "omit_candidate": design["candidate_specs"].pop("GH8 blend0.5")
    if tamper == "blend": design["candidate_specs"]["GH8 blend0.5"]["blend"] = .25
    if tamper == "quadrature": design["orders"] = [8, 16, 64]
    if tamper == "all_paths": design["all_paths_retained"] = False
    if tamper == "price_tolerance": design["price_allowance"] = ".01"
    if tamper == "delta_tolerance": design["delta_allowance"] = ".02"
    if tamper == "confidence": design["confidence_multiplier"] = 1.
    if tamper == "complete": state["manifest"]["complete"] = False
    if tamper == "missing_paths": state["manifest"]["completed_final_paths"] = 8191
    if tamper == "manifest_drift": state["manifest"]["physical_steps"] = 256
    if tamper not in ("complete", "missing_paths", "manifest_drift"):
        state["manifest"].update(deepcopy(design))
    save_campaign(folder, design, seal, state)
    with pytest.raises(ValueError, match="predeclared"):
        collection.collect_bellman_campaign(folder, compute_figures=False)


@pytest.mark.parametrize("tamper", ["design_pin", "selection_seed", "candidate", "ordering", "final_before_seal", "selection_pin", "manifest_ordering"])
def test_prefinal_seal_guards_reject_changed_order_or_final_selection_use(campaign_protocol, monkeypatch, tamper):
    folder, design, seal, state = campaign_protocol
    monkeypatch.setattr(collection, "_frozen_inputs", lambda *args: ({}, {}, {}, {}))
    state["manifest"]["policy_order"] = list(collection._expected_specs())
    if tamper == "selection_seed": seal["selection_seed"] = collection.FINAL_SEEDS[0]
    if tamper == "candidate": seal["candidate_specs"]["GH16 blend1"]["blend"] = .25
    if tamper == "ordering": seal["policy_order"] = seal["policy_order"][::-1]
    if tamper == "final_before_seal": seal["final_paths_generated_before_seal"] = True
    if tamper == "manifest_ordering": state["manifest"]["policy_order"] = state["manifest"]["policy_order"][::-1]
    save_campaign(folder, design, seal, state)
    if tamper == "design_pin": seal["design_sha256"] = "0"*64; write(folder/"selection_before_final.json", seal)
    if tamper == "selection_pin": state["manifest"]["selection_sha256"] = "0"*64
    with pytest.raises(ValueError, match="pre-final"):
        collection.collect_bellman_campaign(folder, compute_figures=False)
