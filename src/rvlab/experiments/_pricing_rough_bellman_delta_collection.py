"""Read-only verification of independent Bellman stock-delta and gap campaigns."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_stopping_delta import _manifest, _sha, _fixed_tolerances
from ._pricing_rough_policy_validation import _entry
from ._pricing_rough_gap_time_stopping import _preserved
from ._pricing_rough_bellman_delta_samples import validate_delta_samples
from ._pricing_rough_bellman_delta_tables import summarize_delta_streams

ROOT = Path(__file__).resolve().parents[3]
STREAMS = [{"seed": 1089005042, "paths": 2048, "steps": 128},
    {"seed": 1090005042, "paths": 2048, "steps": 128},
    {"seed": 1091005042, "paths": 2048, "steps": 256}]


def _read(path):
    return json.loads(Path(path).read_text())


def _verify_sources(folder, design, expected):
    if set(design.get("source_sha256", {})) != expected:
        raise ValueError("A Bellman Greek numerical dependency was omitted or replaced")
    for name, digest in design["source_sha256"].items():
        if _sha(ROOT/name) != digest or _sha(folder/"source"/name) != digest:
            raise ValueError("Frozen Bellman Greek source changed")


def collect_delta_campaign(folder):
    folder = Path(folder).resolve()
    output = folder.parents[1]
    manifest, entries = _manifest(folder, inventory=True)
    design_path = folder/"design_before_valuation.json"
    design = _read(design_path)
    _fixed_tolerances(design, required=True)
    expected = {"version": "rough_bellman_frozen_delta_v1", "streams": STREAMS,
        "policy_order": ["frozen selected policy", "GH16 blend1"],
        "frozen_policy": "GH16 blend1", "bump_relatives": [.005, .0025],
        "relative_spots": [-.005, -.0025, 0., .0025, .005], "operator_steps": 128,
        "exercise_steps": 64, "operator_fixed_across_outer_time_refinement": True,
        "all_bumps_recompute_continuations_and_stops": True, "all_paths_retained": True,
        "sampling_and_bump_and_time_gates_separate": True, "no_model_or_control_fitting": True,
        "no_new_paths_generated_before_design": True, "optimal_delta_ready": False}
    if (any(design.get(k) != v for k, v in expected.items())
            or any(manifest.get(k) != v for k, v in design.items())
            or manifest.get("complete") is not True or manifest.get("primary_paths") != 4096
            or manifest.get("time_paths") != 2048):
        raise ValueError("Changed Bellman Greek design, frozen rule, path budgets or completion")
    prior = output/"rough_bellman_policy_v1/one_step_v1"
    old = _read(prior/"design_before_training.json")
    sources = set(old["source_sha256"]) | {"examples/rough_bellman_delta.py",
        "src/rvlab/experiments/_pricing_rough_bellman_delta.py",
        "src/rvlab/experiments/_pricing_rough_stock_delta.py"}
    _verify_sources(folder, design, sources)
    required = {"rough_bellman_policy_v1/one_step_v1/"+n for n in
        ("manifest.json", "selection_before_final.json", "design_before_training.json")}
    required |= {"rough_bellman_validation_v1/delta_runtime_probe_v1/"+n for n in
        ("manifest.json", "preservation_inventory.json")}
    if not required.issubset(design.get("input_sha256", {})):
        raise ValueError("Delta campaign omits its prior rule or spent runtime probe")
    for name, digest in design["input_sha256"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or _sha(path) != digest:
            raise ValueError("Frozen Bellman Greek input changed")
        entries.append(_entry(path))
    probe = folder.parent/"delta_runtime_probe_v1"
    _, files = _preserved(probe)
    entries.extend(files)
    probe_manifest = _read(probe/"manifest.json")
    if (probe_manifest.get("no_new_paths_or_fitting") is not True
            or probe_manifest.get("frozen_center_stops_exact") is not True
            or probe_manifest.get("new_acceptance_evidence") is not False):
        raise ValueError("Spent runtime probe was promoted to fresh acceptance")
    if _read(prior/"selection_before_final.json").get("selected_policy") != "GH16 blend1":
        raise ValueError("The chosen Bellman exercise rule changed")
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    bundle = load_larger_policy_bundle(output/"rough_larger_policy_v1/32768_paths_two_starts")
    forbidden = set(design["forbidden_prior_seeds"])
    forbidden |= set(bundle["manifest"]["all_selection_training_seeds"]) | {
        1066005042, 1077005042, 1078005042, 1043005042, 1054005042, 1055005042}
    if forbidden & {s["seed"] for s in STREAMS}:
        raise ValueError("Fresh Greek seeds overlap spent training or valuation streams")
    streams, records, costs = [], [], []
    for i, spec in enumerate(STREAMS):
        record = _read(folder/f"stream_{i}_manifest.json")
        launch = _read(folder/f"stream_{i}_launch.json")
        sample = folder/f"stream_{i}_samples.npz"
        expected_record = {**spec, "index": i, "sample_sha256": _sha(sample),
            "design_sha256": _sha(design_path), "source_sha256": design["source_sha256"],
            "sources_unchanged": True}
        if any(record.get(k) != v for k, v in expected_record.items()):
            raise ValueError("Delta stream changed its path count, seed or frozen identities")
        # External launch records were written before dispatch, without editing frozen numerical source.
        launch_time = launch.get("started_at", launch.get("started_at_utc"))
        if (launch.get("design_sha256") != _sha(design_path) or not launch_time
                or not datetime.fromisoformat(design["created_at"]) <= datetime.fromisoformat(launch_time)
                < datetime.fromisoformat(record["completed_at"])):
            raise ValueError("Greek stream launch does not follow its design seal")
        diagnostics = record.get("diagnostics", {})
        flags = {"operator_identity_unchanged": True, "original_two_physical_step_lookahead": True,
            "all_paths_retained": True, "no_fitting_or_selection": True,
            "all_bumps_recompute_continuation_and_stops": True, "physical_operator_steps": 128,
            "outer_steps": spec["steps"], "tolerances_changed": False, "optimal_delta_ready": False}
        if any(diagnostics.get(k) != v for k, v in flags.items()):
            raise ValueError("Changed frozen-rule Greek interpretation or numerical law")
        with np.load(sample, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        schema = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
        if schema != record.get("array_schema"):
            raise ValueError("Greek sample schema differs from the full-path record")
        result = validate_delta_samples(arrays, bundle, **spec)
        streams.append(result)
        records.append(record)
        costs.append({**spec, "stage": "independent fixed-rule delta", "started_at": launch_time,
            "completed_at": record["completed_at"], "elapsed_seconds": record["elapsed_seconds"],
            "computed_alive_ITM_states": int(arrays["bellman_computed_mask"].sum()),
            "used_for_fitting_or_selection": False})
    if manifest.get("records") != records:
        raise ValueError("Delta completion omits a declared independent stream")
    tables = summarize_delta_streams(streams)
    tables["delta_execution"] = pd.DataFrame(costs)
    tables["spent_runtime_probe"] = pd.read_csv(probe/"timings.csv", float_precision="round_trip")
    return {"tables": tables, "input_files": entries, "manifest": manifest}


def bellman_delta_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The declared Bellman delta research seed is 42")
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    context.progress("Verifying frozen Bellman stock differences and independently selected stopping controls")
    output = Path(context.output_dir).resolve()
    delta = collect_delta_campaign(output/"rough_bellman_validation_v1/frozen_rule_delta_v1")
    from ._pricing_rough_bellman_control_collection import collect_control_campaign
    gap = collect_control_campaign(output/"rough_bellman_validation_v1/bellman_control_v1")
    tables = {**delta["tables"], **gap["tables"]}
    unique = {}
    for row in delta["input_files"]+gap["input_files"]:
        key = str(Path(row["path"]).resolve())
        normalized = {**row, "path": key}
        if key in unique and unique[key] != normalized:
            raise ValueError("Conflicting Bellman evidence fingerprints")
        unique[key] = normalized
    entries = [unique[k] for k in sorted(unique)]
    sources = {name: _sha(Path(__file__).with_name(name)) for name in (
        "_pricing_rough_bellman_delta_collection.py", "_pricing_rough_bellman_delta_samples.py",
        "_pricing_rough_bellman_delta_tables.py", "_pricing_rough_bellman_control_collection.py",
        "_pricing_rough_bellman_control_samples.py", "_pricing_rough_bellman_control_tables.py")}
    fingerprint = hashlib.sha256(json.dumps({"version": "rough_bellman_delta_v1", "files": entries,
        "analysis_sources": sources}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tables["provenance"] = pd.DataFrame(entries)
    tables["configuration"] = pd.DataFrame([
        {"item": "Stock deltas", "value": "Frozen GH16 rule; own continuation and stopping date at each ±0.5%/±0.25% stock bump; all other inputs fixed"},
        {"item": "Physical time check", "value": "128 versus 256 steps on independent paths; same 64 exercise dates and the original 128-step Bellman state function"},
        {"item": "Gap experiment", "value": "Frozen GH16 exercise policy; new martingale candidates trained and selected before independent final valuation"},
        {"item": "Accuracy gates", "value": "price .001+1e-4|V|; delta .002+.005|delta|; 95% multiplier 1.96; no tolerance relaxation"},
        {"item": "Scope", "value": "Finite-lift discrete-policy study; no optimal American delta or hedge-readiness certificate"}])
    manifest = {"backend": "rough_heston_bellman_delta", "study_version": "rough_bellman_delta_v1",
        "profile_fingerprint": fingerprint, "analysis_sources": sources, "input_files": entries,
        "primary_complete": True, "historical_replay": False, "current_engine_execution": False,
        "execution_mode": "read-only reconstruction of separately sealed delta and stopping-control campaigns",
        "delta_standard_error": "path-count-weighted independent stream means with within-stream variance; frozen runner CSV summaries remain preserved separately",
        "delta_campaign": delta["manifest"], "gap_campaign": gap["manifest"],
        "primary_policy_fits": 0, "primary_control_optimizer_calls": len(tables["gap_trials"]),
        "new_control_stock_delta_validated": False,
        "new_rule_delta_measured": True, "frozen_operator_steps": 128, "exercise_intervals": 64,
        "delta_primary_paths": 4096, "delta_time_paths": 2048,
        "price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002, "delta_rtol": .005,
        "confidence_multiplier": 1.96, "spread_fraction": .1, "tolerances_changed": False,
        "all_paths_retained": True, "american_pricing_ready": False, "stock_delta_ready": False,
        "optimal_delta_ready": False,
        "limitations": ["Stock differences target each finite frozen stopping rule, not the unknown optimal American value.",
            "Physical refinement holds the original Bellman operator fixed; refining that operator would define a different rule.",
            "Delta samples use the original audited stable martingale; the newly fitted gap-control weights are not validated Greek estimators.",
            "A small stopping price bracket does not establish a derivative error bound.",
            "Kernel, continuous exercise, broader contracts and real-SPY input readiness remain separate requirements."]}
    result = {"tables": tables, "figures": {}, "manifest": manifest}
    if compute_figures:
        from ._pricing_rough_bellman_delta_reporting import add_bellman_delta_reporting
        result = add_bellman_delta_reporting(result)
    return result
