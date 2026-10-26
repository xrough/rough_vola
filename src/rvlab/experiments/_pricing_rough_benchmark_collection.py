"""Fail-closed collection of the predefined generic rough-Heston benchmark.

No path generation, training, control selection or acceptance-driven retries
occur here. Read one saved stream at a time, independently reconstruct its
finite-law path/decision/control evidence, then retain small endpoint samples.
"""
from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_benchmark import (
    VERSION, ESTIMATORS, PARAMETERS, RELATIVE_SPOTS, TOLERANCES,
    benchmark_trials, benchmark_config, canonical, load_policy,
)
from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint


ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = "rough_generic_benchmark_v1/independent_v1"
DESIGN_SHA256 = "9903db3a1a93b46fb043fe174af270509ad128c575d88979a2ee9ca80e14f1c2"
OLD_DESIGN = "notebooks/_output/pricing/research_v1/01_pricing_and_exercise/rough_shared_delta_v1/independent_v1/design_before_valuation.json"
NEW_SOURCES = {"src/rvlab/experiments/_pricing_rough_benchmark.py",
               "src/rvlab/experiments/_pricing_rough_benchmark_audit.py",
               "examples/rough_benchmark.py"}
ARRAY_KEYS = {"times", "eligible", "discounts", "discounted_payoffs", "stocks", "factors",
    "european_martingale", "premium_martingale", "european_proxy", "decision_european",
    "decision_continuation", "decision_rule", "stop_indices", "initial_spots", "relative_spots",
    "actual_time_grid", "operator_time_grid", "public_stop_indices"} | {
        control + suffix for control in ESTIMATORS for suffix in ("_lower", "_upper")}
GUARD_RUNNER_SHA256 = "0a04814c40f02c43c0bed96575a8a9273728ce97ae94e7b487de366265cfdbbd"
FORMAL_RUNNER_SHA256 = "7b44f0cbf23fb7b42a8b8c96525d54fb23af70ff8c8793154af4b4d8478bb2a5"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Benchmark chronology requires explicit timezones")
    return result


def file_entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def closed_inventory(folder):
    folder = Path(folder).resolve()
    inventory_path = folder / "preservation_inventory.json"
    inventory = read(inventory_path)
    rows, entries, seen = inventory["files"], [], set()
    for row in rows:
        name = row["path"]
        path = (folder / name).resolve()
        if name in seen or not path.is_relative_to(folder):
            raise ValueError("Closed benchmark inventory contains duplicate or escaped paths")
        seen.add(name)
        entry = file_entry(path)
        if entry["bytes"] != row["bytes"] or entry["sha256"] != row["sha256"]:
            raise ValueError(f"A preserved benchmark artifact changed: {name}")
        entries.append(entry)
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*")
              if p.is_file() and p.name != "preservation_inventory.json"}
    if seen != actual:
        raise ValueError("A closed benchmark inventory omits or replaces files")
    if "entry_count" in inventory and inventory["entry_count"] != len(rows):
        raise ValueError("Closed inventory entry count changed")
    if "entries_sha256" in inventory and inventory["entries_sha256"] != hashlib.sha256(canonical(rows).encode()).hexdigest():
        raise ValueError("Closed inventory canonical fingerprint changed")
    entries.append(file_entry(inventory_path))
    return entries


def validate_design(design):
    expected = {"version": VERSION, "parameters": PARAMETERS, "trial_count": 10,
        "stream_count": 20, "policy_training_count": 9, "total_valuation_paths": 40960,
        "relative_spots": list(RELATIVE_SPOTS), "bump_relatives": [.005, .0025],
        "estimators": list(ESTIMATORS), "old_source_count": 58, "maximum_workers": 2,
        "regression_ridge": 1e-5, "premium_cap": .7, "batch_size": 512,
        "stock_integral": "exact_drift", "joint_refinement_included": True,
        "no_fresh_selection": True, "no_fitting_on_valuation_paths": True,
        "no_empirical_valuation_recentering": True, "all_paths_retained": True,
        "all_failures_retained": True, "acceptance_requires_all_streams_reported": True,
        "historical_or_pilot_paths_used_for_acceptance": False,
        "independent_training_valuation": True, "focused_GH16_shared193_reused": False,
        "all_policy_training_attempts_complete_before_any_valuation": True,
        "optimal_american_delta_ready": False, "new_hedging_readiness": False,
        **TOLERANCES}
    for key, value in expected.items():
        actual = design.get(key)
        if actual != value or isinstance(value, bool) and actual is not value:
            raise ValueError(f"Predeclared benchmark setting changed: {key}")
    definitions = benchmark_trials(include_joint=True)
    trials = design["trials"]
    if len(trials) != len(definitions):
        raise ValueError("Every base case and refinement must remain declared")
    train_map = {row["policy_id"]: 1146005042 + 3000000 * i for i, row in enumerate(definitions)
                 if row["policy_id"] == row["trial_id"]}
    for i, (trial, definition) in enumerate(zip(trials, definitions)):
        expected_trial = {**definition, "index": i, "training_seed": train_map[trial["policy_id"]],
            "training_paths": 2048, "valuation_paths": 2048,
            "valuation_seeds": [1147005042 + 3000000 * i, 1148005042 + 3000000 * i]}
        if trial != expected_trial:
            raise ValueError("Benchmark cases, ordering, policies, seeds or path counts changed")
    timestamp(design["created_at"])


def expected_streams(design):
    return [{"trial_id": row["trial_id"], "seed": seed, "training_seed": row["training_seed"],
             "paths": row["valuation_paths"], "steps": row["steps"],
             "exercise_steps": row["exercise_steps"], "factors": row["factors"],
             "strike": row["case"]["strike"], "maturity": row["case"]["maturity"]}
            for row in design["trials"] for seed in row["valuation_seeds"]]


def validate_record(record, spec, *, design, design_hash, training, training_hash):
    for key, value in spec.items():
        if record.get(key) != value:
            raise ValueError("A reported benchmark stream changed its declared identity")
    if (record.get("source_sha256") != design["source_sha256"] or
            record.get("design_sha256") != design_hash or
            record.get("training_manifest_sha256") != training_hash or
            record.get("new_fitting_or_selection") is not False or
            record.get("all_failures_retained") is not True):
        raise ValueError("Benchmark stream source, training seal or selection guard changed")
    start, end = timestamp(record["started_at"]), timestamp(record["completed_at"])
    if not timestamp(training["completed_at"]) < start <= end:
        raise ValueError("Independent valuation must follow completed training")
    if record["status"] == "completed":
        if (training["status"] != "completed" or record.get("all_paths_retained") is not True or
                record.get("policy_fingerprint") != training["policy_fingerprint"] or
                record.get("policy_sha256") != training["policy_sha256"]):
            raise ValueError("A valuation used a changed or incompletely trained rule")
    elif record["status"] == "blocked_training_failure":
        if training["status"] != "failed":
            raise ValueError("A stream falsely claims its policy training failed")
    elif record["status"] != "failed":
        raise ValueError("Unrecognized benchmark stream outcome")


def runner_guard_compatibility(previous, current):
    """The known pre-fresh change only strengthens prepare/input bookkeeping."""
    if sha(previous) != GUARD_RUNNER_SHA256 or sha(current) != FORMAL_RUNNER_SHA256:
        raise ValueError("Unknown runner version in the implementation guard")
    def numerical_body(path):
        tree = ast.parse(Path(path).read_text())
        result = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("prepare", "get_design"):
                continue
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id in ("IMPLEMENTATION_GUARD", "PILOT_AUDIT") for target in node.targets):
                continue
            result.append(ast.dump(node, include_attributes=False))
        return result
    if numerical_body(previous) != numerical_body(current):
        raise ValueError("The formal runner changed numerical generation or training code")
    return {"guard_runner_sha256": GUARD_RUNNER_SHA256, "formal_runner_sha256": FORMAL_RUNNER_SHA256,
        "whole_source_profiles_identical": False, "numerical_functions_and_constants_identical": True,
        "change_scope": "Added implementation-folder constants, prepare verification/input pins, and get_design input checks before formal training"}


def validate_policy(policy, trial, design, training):
    expected = benchmark_config(training_seed=trial["training_seed"],
        valuation_seed=trial["valuation_seeds"][0], training_paths=trial["training_paths"],
        valuation_paths=trial["valuation_paths"], steps=trial["steps"],
        exercise_steps=trial["exercise_steps"])
    if (fourier_policy_fingerprint(policy) != training["policy_fingerprint"] or
            asdict(policy.params) != design["parameters"] or
            canonical(asdict(policy.kernel)) != canonical(design["kernels"][str(trial["factors"])]) or
            policy.config != expected or policy.control_config != expected.premium_control or
            policy.strike != trial["case"]["strike"] or policy.maturity != trial["case"]["maturity"] or
            policy.is_call or policy.exercise_style != "american" or policy.premium_cap != .7 or
            policy.market.spot != 100. or policy.market.dividends.events or policy.market.dividends.coverage != 1.):
        raise ValueError("A trained policy changed the declared model, kernel, market, settings or identity")
    times = np.asarray(policy.market.discount_curve.times)
    discounts = np.asarray(policy.market.discount_curve.discounts)
    if (not np.array_equal(times, [0., 1.]) or
            not np.allclose(discounts, [1., np.exp(-.04)], rtol=0., atol=2e-15)):
        raise ValueError("A trained policy changed the declared discount curve")


def _source_and_input_checks(folder, design):
    old = read(ROOT / OLD_DESIGN)
    if (set(design["source_sha256"]) != set(old["source_sha256"]) | NEW_SOURCES or
            any(design["source_sha256"].get(k) != v for k, v in old["source_sha256"].items())):
        raise ValueError("Frozen old sources or new numerical source closure changed")
    entries = []
    for name, digest in design["source_sha256"].items():
        for path in (ROOT / name, folder / "source" / name):
            entry = file_entry(path)
            if entry["sha256"] != digest:
                raise ValueError(f"A frozen numerical source changed: {name}")
            entries.append(entry)
    base_relative = folder.parent.relative_to(ROOT)
    required = {OLD_DESIGN} | {str(base_relative / name / leaf)
        for name in ("spent_preflight_v1", "implementation_guard_v1", "pilot_independent_audit_v1")
        for leaf in ("manifest.json", "preservation_inventory.json")}
    if set(design["input_sha256"]) != required or design["input_sha256"].get(OLD_DESIGN) != sha(ROOT / OLD_DESIGN):
        raise ValueError("The seven parent/preflight input pins changed")
    for name, digest in design["input_sha256"].items():
        entry = file_entry(ROOT / name)
        if entry["sha256"] != digest:
            raise ValueError(f"A pre-fresh implementation input changed: {name}")
        entries.append(entry)
    base = folder.parent
    for name in ("spent_preflight_v1", "implementation_guard_v1", "pilot_independent_audit_v1"):
        entries.extend(closed_inventory(base / name))
        preflight = read(base / name / "manifest.json")
        if (preflight.get("complete") is not True or preflight.get("sources_unchanged") is not True or
                not timestamp(preflight["created_at"]) < timestamp(preflight["completed_at"]) < timestamp(design["created_at"])):
            raise ValueError("Implementation checks must complete before the formal design")
        if name != "spent_preflight_v1":
            old_sources = preflight.get("source_sha256", {})
            if set(old_sources) != set(design["source_sha256"]):
                raise ValueError("Independent implementation checks omitted numerical sources")
            changes = {key for key, value in design["source_sha256"].items() if old_sources.get(key) != value}
            if changes:
                if changes != {"examples/rough_benchmark.py"}:
                    raise ValueError("Independent implementation checks used a different numerical algorithm")
                runner_guard_compatibility(base / name / "source/examples/rough_benchmark.py",
                                           folder / "source/examples/rough_benchmark.py")
        if name == "pilot_independent_audit_v1" and preflight.get("all_checks_passed") is not True:
            raise ValueError("The independent cached-path implementation audit failed")
        if name == "implementation_guard_v1" and (
                len(preflight.get("records", [])) != 2 or
                any(row.get("status") != "completed" for row in preflight["records"])):
            raise ValueError("The fixed-decision stochastic implementation guard failed")
    forbidden = set(old["forbidden_prior_seeds"]) | {r["seed"] for r in old["streams"]} | {1094005042, 1095005042}
    training = {r["training_seed"] for r in design["trials"]}
    valuation = [s for r in design["trials"] for s in r["valuation_seeds"]]
    if (len(training) != 9 or len(valuation) != len(set(valuation)) or
            training.intersection(valuation) or (training | set(valuation)).intersection(forbidden)):
        raise ValueError("A benchmark path stream reused training or spent scientific observations")
    return entries


def collect_benchmark(folder):
    from ._pricing_rough_benchmark_audit import summarize_benchmark, validate_benchmark_samples
    folder = Path(folder).resolve()
    design_path = folder / "design_before_training.json"
    design_hash = sha(design_path)
    if design_hash != DESIGN_SHA256:
        raise ValueError("The approved final benchmark design changed")
    design = read(design_path)
    validate_design(design)
    entries = closed_inventory(folder)
    entries.extend(_source_and_input_checks(folder, design))
    manifest = read(folder / "manifest.json")
    if (any(manifest.get(k) != v for k, v in design.items()) or
            manifest.get("complete") is not True or manifest.get("all_streams_reported") is not True or
            manifest.get("all_declared_attempts_retained") is not True or
            manifest.get("sources_unchanged") is not True or
            manifest.get("design_sha256") != design_hash or len(manifest["records"]) != 20):
        raise ValueError("All ten benchmark trials and twenty stream outcomes must be closed before collection")
    policies, training_records, training_paths, training_execution = {}, [], {}, []
    for i, trial in enumerate(design["trials"]):
        if trial["policy_id"] != trial["trial_id"]:
            continue
        path = folder / f"training_{i}_manifest.json"
        record = read(path)
        if (record.get("index") != i or record.get("trial_id") != trial["trial_id"] or
                record.get("policy_id") != trial["policy_id"] or
                record.get("training_seed") != trial["training_seed"] or
                record.get("training_paths") != trial["training_paths"] or
                record.get("source_sha256") != design["source_sha256"] or
                record.get("design_sha256") != design_hash or
                not timestamp(design["created_at"]) < timestamp(record["started_at"]) <= timestamp(record["completed_at"])):
            raise ValueError("A training attempt differs from its frozen design or chronology")
        if record["status"] == "completed":
            policy_folder = folder / ("policy_" + trial["policy_id"])
            if (sha(policy_folder / "policy.json") != record["policy_sha256"] or
                    sha(policy_folder / "coefficients.npz") != record["coefficient_sha256"]):
                raise ValueError("A policy coefficient seal changed after training")
            policy = load_policy(policy_folder)
            validate_policy(policy, trial, design, record)
            policies[trial["policy_id"]] = policy
        elif record["status"] != "failed":
            raise ValueError("Unrecognized training outcome")
        training_records.append(record)
        training_paths[trial["policy_id"]] = path
        training_execution.append({"trial_id": trial["trial_id"], "training_seed": trial["training_seed"],
            "training_paths": trial["training_paths"], "status": record["status"],
            "elapsed_seconds": record["elapsed_seconds"], "started_at": record["started_at"],
            "completed_at": record["completed_at"], "exercise_at_inception":
                policies[trial["policy_id"]].exercise_at_inception if trial["policy_id"] in policies else None})
    if manifest["training_records"] != training_records:
        raise ValueError("A declared training failure or completed attempt was omitted")
    last_training = max(timestamp(r["completed_at"]) for r in training_records)
    end = timestamp(manifest["completed_at"])
    streams, records, checks, probes, execution = [], [], [], [], []
    specs = expected_streams(design)
    by_id = {r["policy_id"]: r for r in training_records}
    for i, (spec, trial) in enumerate(zip(specs, [r for r in design["trials"] for _ in range(2)])):
        stream_index, trial_index = i % 2, i // 2
        stream_folder = folder / f"trial_{trial_index}_stream_{stream_index}"
        record = read(stream_folder / "manifest.json")
        trained = by_id[trial["policy_id"]]
        validate_record(record, spec, design=design, design_hash=design_hash,
                        training=trained, training_hash=sha(training_paths[trial["policy_id"]]))
        if (record.get("index") != trial_index or record.get("stream_index") != stream_index or
                record.get("policy_id") != trial["policy_id"] or
                not last_training < timestamp(record["started_at"]) <= timestamp(record["completed_at"]) <= end):
            raise ValueError("A valuation started before all training ended or after campaign completion")
        if record["status"] == "completed":
            path = stream_folder / "samples.npz"
            if sha(path) != record["sample_sha256"]:
                raise ValueError("A saved benchmark path archive changed")
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != ARRAY_KEYS:
                    raise ValueError("A saved path archive omitted or replaced numerical fields")
                arrays = {key: saved[key] for key in saved.files}
            schema = {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()}
            if schema != record["array_schema"]:
                raise ValueError("A saved path archive changed shape or dtype")
            verified = validate_benchmark_samples(arrays, policies[trial["policy_id"]], spec=spec)
            streams.append(verified["stream"])
            checks.append(verified["checks"])
            probes.extend({"trial_id": trial["trial_id"], "seed": spec["seed"], **p}
                          for p in verified["probes"])
            del arrays, verified
        else:
            streams.append({**spec, "failure": {"status": record["status"],
                "message": record.get("message") or record["status"]}, "controls": {}})
        records.append(record)
        execution.append({**spec, "status": record["status"], "started_at": record["started_at"],
            "completed_at": record["completed_at"], "elapsed_seconds": record["elapsed_seconds"],
            "timing_scope": "Shared physical paths, full decision replay and all three controls; not isolated control runtimes"})
    if manifest["records"] != records:
        raise ValueError("A complete or failed valuation stream was omitted")
    count = sum(r["paths"] for r in records if r["status"] == "completed")
    if manifest["completed_valuation_paths"] != count:
        raise ValueError("Completed path counts disagree with retained stream outcomes")
    pairs = [{"coarse_trial": row["reference_trial"], "fine_trial": row["trial_id"],
              "same_full_policy": row["stage"] == "physical"}
             for row in design["trials"] if row["reference_trial"] is not None]
    tables = summarize_benchmark(streams, expected_streams=specs, refinement_pairs=pairs)
    tables.update(path_identity_checks=pd.DataFrame(checks), conditional_probes=pd.DataFrame(probes),
        execution=pd.DataFrame(execution), training_execution=pd.DataFrame(training_execution),
        trial_design=pd.DataFrame([{**{k: v for k, v in row.items() if k != "case"}, **row["case"]}
                                   for row in design["trials"]]))
    tables["implementation_source_compatibility"] = pd.DataFrame([
        {"fixture": name, **runner_guard_compatibility(
            folder.parent / name / "source/examples/rough_benchmark.py",
            folder / "source/examples/rough_benchmark.py")}
        for name in ("implementation_guard_v1", "pilot_independent_audit_v1")])
    return {"tables": tables, "campaign": manifest, "input_files": entries,
            "streams": streams, "expected_streams": specs, "refinement_pairs": pairs}


def price_bracket_separation(prices, pairs):
    """Independent-law primary contrasts; no attribution to one error source."""
    selected = (prices.loc[prices.control.eq("european_premium")].set_index("trial_id")
                if not prices.empty else pd.DataFrame())
    rows = []
    for pair in pairs:
        coarse, fine = pair["coarse_trial"], pair["fine_trial"]
        if coarse not in selected.index or fine not in selected.index:
            rows.append({**pair, "complete": False})
            continue
        for left, right, direction in ((coarse, fine, "coarse lower minus fine upper"),
                                        (fine, coarse, "fine lower minus coarse upper")):
            lower, upper = selected.loc[left], selected.loc[right]
            estimate = float(lower.price - upper.upper)
            se = float(np.hypot(lower.price_se, upper.upper_se))
            half_width = TOLERANCES["confidence_multiplier"] * se
            rows.append({**pair, "complete": True, "direction": direction,
                "control": "european_premium", "primary": True, "estimate": estimate,
                "standard_error": se, "half_width": half_width,
                "interval_low": estimate - half_width, "interval_high": estimate + half_width,
                "positive_separation": estimate - half_width > 0,
                "independent_valuation_streams": True, "conditional_on_training": True,
                "used_for_selection": False, "pure_kernel_bias_identified": False,
                "scope": "Finite-law/calendar price brackets; not isolated kernel, time or policy error"})
    return pd.DataFrame(rows)


def benchmark_study(context):
    from ._pricing_rough_benchmark_recovery_collection import collect_recovery
    context.progress("Auditing all ten predefined generic benchmark trials and independent path streams")
    collected = collect_benchmark(Path(context.output_dir) / CAMPAIGN)
    collected = collect_recovery(collected,
        Path(context.output_dir) / "rough_generic_benchmark_v1/serialization_repair_v1")
    collected["tables"]["bracket_separation"] = price_bracket_separation(
        collected["tables"]["price_bounds"], collected["refinement_pairs"])
    unique = {}
    for entry in collected["input_files"]:
        old = unique.setdefault(entry["path"], entry)
        if old != entry:
            raise ValueError("Conflicting benchmark input fingerprints")
    files = [unique[name] for name in sorted(unique)]
    analysis = [file_entry(Path(__file__)), file_entry(
        Path(__file__).with_name("_pricing_rough_benchmark_recovery_collection.py"))]
    fingerprint = hashlib.sha256(canonical({"files": files, "analysis_sources": analysis}).encode()).hexdigest()
    manifest = {"version": VERSION, "data_kind": "synthetic", "exercise_style": "american",
        "backend": "generic_fourier_premium_finite_lift_stopping",
        "complete": True, "all_declared_attempts_reported": True,
        "profile_fingerprint": fingerprint, "input_files": files, "analysis_sources": analysis,
        "campaign": collected["campaign"], "base_cases": 6, "representative_refinements": 4,
        "serialization_recovery": collected["recovery"],
        "original_valuation_paths": collected["campaign"]["completed_valuation_paths"],
        "recovered_original_never_drawn_paths": collected["recovery"]["completed_recovered_paths"],
        "trial_count": 10, "declared_valuation_paths": 40960,
        "completed_valuation_paths": collected["completed_paths"],
        "primary_control": "european_premium", "focused_GH16_shared193_reused": False,
        "no_paths_generated_in_notebook": True, "no_post_valuation_selection": True,
        "all_failures_retained": True, "readiness_requires_more_than_report_completion": True,
        "american_pricing_ready": False, "stock_delta_ready": False, "optimal_delta_ready": False,
        "conditional_on_training": True, "confidence_intervals_are_marginal": True,
        "control_runtime_isolation_measured": False, **TOLERANCES,
        "limitations": [
            "Six put contracts at one inception parameter set with explicit absence of cash dividends.",
            "This generic policy/control panel is distinct from the specialized GH16/shared193 case study.",
            "Bounds concern finite transition laws and exercise calendars; price intervals do not bound optimal stock deltas.",
            "Only the physical refinement freezes the full decision function; other refinements include independently retrained rules.",
            "Normal intervals concern valuation uncertainty conditional on each trained policy; they omit training and structural approximation errors.",
            "All control comparisons share physical-path generation and execution costs; isolated algorithm speedups are not measured.",
        ]}
    return {"tables": collected["tables"], "figures": {}, "manifest": manifest}
