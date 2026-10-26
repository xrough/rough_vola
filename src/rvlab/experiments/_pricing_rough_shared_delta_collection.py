"""Read-only collection of independently generated shared-control stock bumps.

The campaign fixes its source, rule, weights, seeds and budgets before fresh
paths. Historical price evidence supplies identities, never Greek samples.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PARENT = "rough_mc_precision_v1/independent_v1"
PARENT_PROFILE = "93d94ff9f30cfc6155d77797c05d3996dcc8dc1588b5e811b68f152c095dcac2"
CAMPAIGN = "rough_shared_delta_v1/independent_v1"
PARENT_MANIFEST_SHA = "bf07c44777e00457acf88b7421b85143b89be9323b73c939158b3a13e973bbc0"
PARENT_DESIGN_SHA = "81946dfa36c7c241bac6ef0c4d9f0c0762507142857b6e0977e8932bc42ebdd8"
COEFFICIENT_SHA = "dc847d22f3186d834b13a77fb82a55daf61bac614fc2b02753b2cee95c463fa2"
# Fixed from the255-path spent implementation/precision pilots, before fresh
# valuation. These streams cannot be enlarged or selected after seeing results.
STREAMS = (
    {"seed": 1132005042, "paths": 4096, "steps": 128, "representations": ["original", "micro"]},
    {"seed": 1133005042, "paths": 4096, "steps": 128, "representations": ["original", "micro"]},
    {"seed": 1134005042, "paths": 4096, "steps": 256, "representations": ["micro"]},
    {"seed": 1135005042, "paths": 4096, "steps": 256, "representations": ["micro"]},
)
COMMON_FIELDS = frozenset("""times discounts eligible actual_time_grid stocks factors
    relative_spots initial_spots operator_time_grid payoff calendar_european gh16_stops
    gh16_continuation gh16_computed_mask absolute_path_indices physical_time_grid
    physical_stocks physical_factors conditional_log_means conditional_variances
    conditional_expected_next conditional_actual_next conditional_european_payoff
    conditional_european_martingale conditional_european_estimate""".split())
REPRESENTATION_FIELDS = frozenset("""processes latest_martingale selected_martingale
    stage_corrections basis anchor shift raw stable unchanged_lower shared_lower
    unchanged_upper shared_upper""".split())


def sha(path):
    """Bounded-memory fingerprint, including multi-gigabyte historical arrays."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def entry(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def preserved(folder):
    """Verify all declared files and reject omitted or escaping artifacts."""
    folder = Path(folder).resolve()
    path = folder / "preservation_inventory.json"
    inventory = read(path)
    files = inventory["files"]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != inventory.get("inventory_sha256"):
        raise ValueError("Changed shared-delta preservation inventory")
    entries, names = [entry(path)], set()
    for row in files:
        name = row["path"]
        target = (folder / name).resolve()
        if (name in names or not target.is_relative_to(folder) or
                target.stat().st_size != row["bytes"] or sha(target) != row["sha256"]):
            raise ValueError("Changed, duplicated or escaping shared-delta artifact")
        names.add(name)
        entries.append({"path": str(target), "bytes": row["bytes"], "sha256": row["sha256"]})
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*")
        if p.is_file() and p != path}
    if names != actual:
        raise ValueError("Shared-delta inventory omits a campaign artifact")
    return inventory, entries


def verify_sources_and_inputs(folder, design, output):
    """The original54 numerical dependencies remain byte-for-byte frozen."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if (sha(output / PARENT / "manifest.json") != PARENT_MANIFEST_SHA or
            sha(output / PARENT / "design_before_valuation.json") != PARENT_DESIGN_SHA):
        raise ValueError("The sealed historical price campaign changed")
    parent = read(output / PARENT / "design_before_valuation.json")
    original = parent["source_sha256"]
    declared = design["source_sha256"]
    mandatory = set(parent["input_sha256"]) | {PARENT + "/" + name for name in (
        "manifest.json", "design_before_valuation.json", "weights_before_final.json",
        "preservation_inventory.json")} | {"rough_shared_delta_v1/spent_v1/" + name for name in (
        "design_before_evaluation.json", "manifest.json", "equivalence_checks.json",
        "planning_diagnostics.json", "preservation_inventory.json")}
    if (set(design["input_sha256"]) != mandatory or
            any(design["input_sha256"][name] != digest for name, digest in parent["input_sha256"].items())):
        raise ValueError("A mandatory frozen model, policy, coefficient or preflight input was omitted")
    required = {"examples/rough_shared_delta.py", *(
        "src/rvlab/experiments/" + name for name in (
            "_pricing_rough_shared_delta.py", "_pricing_rough_shared_delta_statistics.py",
            "_pricing_rough_shared_delta_samples.py"))}
    if len(original) != 54 or set(original) | required != set(declared):
        raise ValueError("A historical numerical dependency was omitted")
    entries = []
    for name, digest in declared.items():
        source, snapshot = (ROOT / name).resolve(), (folder / "source" / name).resolve()
        if (not source.is_relative_to(ROOT) or not snapshot.is_relative_to(folder / "source")
                or sha(source) != digest or sha(snapshot) != digest
                or name in original and original[name] != digest):
            raise ValueError("Frozen shared-delta numerical source changed")
        entries.extend((entry(source), entry(snapshot)))
    for name, digest in design["input_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha(path) != digest:
            raise ValueError("Frozen shared-delta policy, coefficient or input changed")
        entries.append(entry(path))
    return entries


def validate_design(design):
    expected = {"version": "rough_shared_delta_v1", "parent_campaign": PARENT,
        "streams": list(STREAMS), "total_paths": sum(s["paths"] for s in STREAMS),
        "frozen_policy": "GH16 blend1",
        "policy_operator_steps": 128, "exercise_intervals": 64, "kernel_factors": 4,
        "common_control_features": 193, "bump_relatives": [.005, .0025],
        "relative_spots": [-.005, -.0025, 0., .0025, .005],
        "physical_random_batch_size": 512, "work_batch_size": 32,
        "calendar_european_function_grid_points": 129,
        "physical_european_function_grid_points_by_steps": {"128": 129, "256": 257},
        "original_and_micro_are_distinct_centered_controls": True,
        "microstep_control_not_a_telescoping_block_bridge": True,
        "independent_primitive_panel": {"absolute_path_indices": [0, 1, 511, 512],
            "calendar_indices": [16, 32, 48, 63], "bump_indices": [0, 2, 4]},
        "no_fresh_selection_or_fitting": True,
        "no_empirical_valuation_recentering": True,
        "no_regression_intercept_at_valuation": True,
        "all_bumps_recompute_continuations_and_stops": True,
        "all_paths_retained": True, "prior_paths_used_for_acceptance": False,
        "price_atol": .001, "price_rtol": 1e-4,
        "delta_atol": .002, "delta_rtol": .005,
        "confidence_multiplier": 1.96, "spread_fraction": .1,
        "tolerances_changed": False, "optimal_american_delta_ready": False,
        "new_hedging_readiness": False}
    if not STREAMS or any((design.get(k) is not v) if isinstance(v, bool)
            else design.get(k) != v for k, v in expected.items()):
        raise ValueError("Changed shared-delta design, budget, rule or tolerances")
    seeds = [row["seed"] for row in STREAMS]
    forbidden = design.get("forbidden_prior_seeds")
    if (not isinstance(forbidden, list) or len(set(forbidden)) != len(forbidden)
            or any(isinstance(s, bool) or not isinstance(s, int) for s in forbidden)
            or len(set(seeds)) != len(seeds) or set(seeds) & set(forbidden)):
        raise ValueError("The independent stock-delta seed firewall failed")
    _timestamp(design["created_at"])


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Campaign timestamps require explicit timezones")
    return parsed


def _unique_entries(entries):
    unique = {}
    for row in entries:
        key = str(Path(row["path"]).resolve())
        value = {**row, "path": key}
        if key in unique and unique[key] != value:
            raise ValueError("Conflicting shared-delta input fingerprints")
        unique[key] = value
    return [unique[key] for key in sorted(unique)]


def load_controls(output):
    """Load already sealed objects; no model, policy or coefficient is fitted."""
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    from ._pricing_rough_frozen_policy import _decode_models
    output = Path(output)
    bundle = load_larger_policy_bundle(output / "rough_larger_policy_v1/32768_paths_two_starts")
    control = output / "rough_bellman_validation_v1/bellman_control_v1"
    design = read(control / "design_before_training.json")
    tail = read(output / "rough_tail_attribution_v1/joint_switch_v1/selection_before_final.json")
    target = read(control / "selection_before_final.json")
    proxy = output / "rough_bellman_validation_v1/spent_proxy_aligned_v2"
    seal = read(proxy / "proxy_before_fresh_paths.json")
    with np.load(proxy / seal["coefficient_archive"], allow_pickle=False) as saved:
        models = (None,) + _decode_models(seal["models"], saved) + (None,)
    weights = read(output / PARENT / "weights_before_final.json")
    return bundle, design["prior_control"], design["grid_control"], tail, target, models, weights


def _load_archive(path, schema):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    actual = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()}
    if actual != schema:
        raise ValueError("The complete stock-bump archive schema changed")
    return arrays


def validate_array_set(arrays, representations):
    expected = COMMON_FIELDS | {rep + "_" + name
        for rep in representations for name in REPRESENTATION_FIELDS}
    if set(arrays) != expected:
        raise ValueError("A complete bumped-path primitive was omitted or replaced")
    for name, value in arrays.items():
        if (value.dtype.kind not in "bifu" or np.any(np.isinf(value)) or
                name != "gh16_continuation" and not np.isfinite(value).all()):
            raise ValueError("A bumped-path primitive has invalid type or nonfinite values")


def validate_record(record, design, *, index, design_sha256):
    expected = {"index": index, **STREAMS[index], "design_sha256": design_sha256,
        "source_sha256": design["source_sha256"], "sources_unchanged": True,
        "all_paths_retained": True, "new_fitting_or_selection": False}
    if any((record.get(k) is not v) if isinstance(v, bool) else record.get(k) != v
            for k, v in expected.items()):
        raise ValueError("A stock-delta stream changed its identity, source or policy")
    if (not _timestamp(design["created_at"]) < _timestamp(record["started_at"]) <
            _timestamp(record["completed_at"]) or
            not np.isfinite(record.get("elapsed_seconds", np.nan)) or
            record["elapsed_seconds"] <= 0):
        raise ValueError("A stock-delta stream has invalid chronology or runtime")


def validate_completion_timing(design, manifest, records):
    completed = _timestamp(manifest["completed_at"])
    if (not _timestamp(design["created_at"]) < completed or
            any(_timestamp(row["completed_at"]) > completed for row in records)):
        raise ValueError("The aggregate completion seal predates a stock-delta stream")


def delta_gate_summary(tables):
    """All prescribed shared-estimator checks enter; no bump/start is selected."""
    declarations = (("delta_sampling", "sampling_pass", 6),
        ("bump_refinement", "bump_check_pass", 3),
        ("representation_change", "tolerance_bias_resolved", 2),
        ("time_refinement", "time_check_pass", 2),
        ("combined_refinement", "combined_check_pass", 2))
    rows = []
    for name, gate, expected in declarations:
        data = tables[name]
        shared = data.loc[data.estimator.eq("shared")]
        if len(shared) != expected or shared[gate].dtype.kind != "b":
            raise ValueError("A prescribed stock-delta check was omitted or changed")
        rows.append({"check": name, "checks": len(shared),
            "passed": int(shared[gate].sum()), "all_pass": bool(shared[gate].all()),
            "delta_atol": .002, "delta_rtol": .005, "confidence_multiplier": 1.96,
            "optimal_delta_ready": False})
    return pd.DataFrame(rows)


def validate_preflight(folder, design, output, weights):
    """Recompute the23 spent equivalence checks; do not trust a saved pass flag."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    _, entries = preserved(folder)
    before = read(folder / "design_before_evaluation.json")
    manifest = read(folder / "manifest.json")
    specs = [{"seed": 1094005042, "paths": 4096, "steps": 128, "reference": "old price control"},
        {"seed": 1089005042, "paths": 2048, "steps": 128, "reference": "old five-stock valuer"},
        {"seed": 1091005042, "paths": 2048, "steps": 256, "reference": "old five-stock valuer"}]
    analysis = {"src/rvlab/experiments/_pricing_rough_shared_delta_statistics.py",
        "src/rvlab/experiments/_pricing_rough_shared_delta_samples.py"}
    expected = {"version": "rough_shared_delta_spent_v1", "streams": specs,
        "evaluated_paths_each": 255, "fresh_paths": 0, "equivalence_atol": 2e-13,
        "physical_random_batch_size": 512, "work_batch_size": 32,
        "all_full_stream_random_draws_consumed": True,
        "original_and_micro_are_distinct_centered_controls": True,
        "no_fresh_selection_or_fitting": True}
    if (any((before.get(k) is not v) if isinstance(v, bool) else before.get(k) != v
                for k, v in expected.items()) or
            any(manifest.get(k) != v for k, v in before.items()) or
            manifest.get("complete") is not True or manifest.get("all_checks_passed") is not True or
            manifest.get("numerical_sources_unchanged") is not True or
            sha(folder / "manifest.json") != design.get("preflight_manifest_sha256") or
            set(before["source_sha256"]) != set(design["source_sha256"]) or
            any(h != design["source_sha256"][name] for name, h in before["source_sha256"].items()
                if name not in analysis) or
            not _timestamp(manifest["completed_at"]) < _timestamp(design["created_at"])):
        raise ValueError("A complete matching pre-fresh spent equivalence study is required")
    for name, digest in before["input_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha(path) != digest:
            raise ValueError("A spent equivalence reference changed")
        entries.append(entry(path))
    oldprice = output / "rough_bellman_validation_v1/bellman_control_v1"
    olddelta = output / "rough_bellman_validation_v1/frozen_rule_delta_v1"
    fit = output / "rough_price_budget_v1/spent_shared_fit_v1"
    rows = []
    records = manifest["records"]
    if len(records) != 3:
        raise ValueError("A spent equivalence stream was omitted")
    for i, spec in enumerate(specs):
        record = records[i]
        path = folder / f"stream_{i}_samples.npz"
        if (record.get("index") != i or any(record.get(k) != v for k, v in spec.items()) or
                record.get("evaluated_paths") != 255 or record.get("sample_sha256") != sha(path) or
                not _timestamp(before["created_at"]) <= _timestamp(record["started_at"]) <
                    _timestamp(record["completed_at"]) <= _timestamp(manifest["completed_at"])):
            raise ValueError("A spent full-stream identity or chronology changed")
        indices = np.r_[np.arange(251), 511, 512, 1023, spec["paths"] - 1]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        validate_array_set(arrays, ["original", "micro"] if spec["steps"] == 128 else ["micro"])
        if not np.array_equal(arrays["absolute_path_indices"], indices):
            raise ValueError("A spent comparison changed its absolute full-stream indices")
        pairs = []
        if i == 0:
            with np.load(oldprice / "final_0_samples.npz", allow_pickle=False) as old:
                for name, reference in (("stocks", "stocks"), ("factors", "factors"),
                        ("payoff", "payoff"), ("calendar_european", "european_proxy"),
                        ("gh16_stops", "gh16_stops"), ("original_latest_martingale", "latest_martingale"),
                        ("original_unchanged_lower", "candidate3_lower")):
                    value, previous = arrays[name], old[reference][indices]
                    if value.ndim == 3 and name != "factors": value = value[:, :, 2]
                    elif value.ndim == 2: value = value[:, 2]
                    if previous.ndim == 3 and reference != "factors": previous = previous[:, :, 0]
                    pairs.append((name, value, previous))
            with np.load(fit / "spent_0_basis.npz", allow_pickle=False) as old:
                basis = old["gated_anchor"][indices]
                pairs.extend((("original_basis", arrays["original_basis"][:, 2], basis),
                    ("original_shared_lower", arrays["original_shared_lower"][:, 2], old["lower"][indices] - basis @ weights)))
        else:
            with np.load(olddelta / f"stream_{0 if i == 1 else 2}_samples.npz", allow_pickle=False) as old:
                for name, reference in (("stocks", "stocks"), ("factors", "factors"), ("payoff", "payoff"),
                        ("calendar_european", "baseline_european_proxy"), ("gh16_stops", "bellman_stops"),
                        ("gh16_continuation", "bellman_continuation"), ("gh16_computed_mask", "bellman_computed_mask")):
                    pairs.append((name, arrays[name], old[reference][indices]))
        for name, actual, previous in pairs:
            if actual.shape != previous.shape or not np.array_equal(np.isnan(actual), np.isnan(previous)):
                raise ValueError("Spent equivalence dimensions or stopping masks changed")
            finite = np.isfinite(previous)
            error = (float(np.any(actual != previous)) if actual.dtype.kind == "b" else
                float(np.max(abs(actual[finite] - previous[finite]), initial=0.)))
            if error > 2e-13:
                raise ValueError(f"Spent implementation equivalence failed: {name}")
            rows.append({"stream": i, "field": name, "max_abs_error": error,
                "same_nan_mask": True, "allowance": 2e-13, "passed": True})
        del arrays, pairs
    checks = read(folder / "equivalence_checks.json")
    if (len(rows) != 23 or len(checks) != 23 or
            sha(folder / "equivalence_checks.json") != manifest["checks_sha256"] or
            sha(folder / "planning_diagnostics.json") != manifest["planning_sha256"]):
        raise ValueError("A spent implementation or budget-planning record changed")
    for actual, recorded in zip(rows, checks):
        if (any(actual[k] != recorded[k] for k in actual if k != "max_abs_error") or
                abs(actual["max_abs_error"] - recorded["max_abs_error"]) > 2e-14):
            raise ValueError("A spent equivalence result was relabelled")
    return manifest, entries, pd.DataFrame(rows), pd.DataFrame(read(folder / "planning_diagnostics.json"))


def development_trials(folder, manifest):
    rows = []
    benchmark = read(Path(folder) / "trials/rough_shared_delta_fourstep_benchmark.json")
    source = Path(folder) / "trials/fourstep_benchmark_helper.py"
    if (benchmark.get("fresh_paths") != 0 or
            benchmark.get("one_and_two_step_reference_bitexact") is not True or
            sha(source) != benchmark["new_helper_sha256"]):
        raise ValueError("The rejected exact four-step trial lost its source or scope")
    for item in benchmark["records"]:
        rows.append({"trial": f"Exact four-step block at exercise index {item['date']}",
            "physical_steps": 256, "evaluated_paths": item["states"], "spots": item["spots"],
            "elapsed_seconds": item["total_seconds"], "projected_paths": 4096,
            "projected_seconds": item["total_seconds"] * 4096 / item["states"] * 64,
            "outcome": "finite result; full study rejected for runtime",
            "used_for_acceptance": False,
            "note": "81 factor branches×16 Gaussian orders; linear projection over 64 intervals, not an observed full run"})
    for record in manifest["records"]:
        rows.append({"trial": f"Spent {record['steps']}-step pilot · seed {record['seed']}",
            "physical_steps": record["steps"], "evaluated_paths": record["evaluated_paths"],
            "spots": 5, "elapsed_seconds": record["elapsed_seconds"], "projected_paths": 4096,
            "projected_seconds": record["projected_full4096_seconds"],
            "outcome": "reference-equivalent; planning only", "used_for_acceptance": False,
            "note": "Full prior RNG stream consumed; selected 255 paths supply cost/precision planning only; projection excludes archive compression"})
    return pd.DataFrame(rows)


def validate_cached_roundtrips(folder, design, controls):
    """The final frozen guard must pass cached full small-stream implementations."""
    from ._pricing_rough_shared_delta_samples import validate_shared_delta_samples
    folder = Path(folder)
    path = folder / "implementation_guard_v3/manifest.json"
    record = read(path)
    specs = [{"seed": 1094005042, "paths": 4, "steps": 128, "representations": ["original", "micro"]},
        {"seed": 1091005042, "paths": 4, "steps": 256, "representations": ["micro"]}]
    if (record.get("version") != "rough_shared_delta_implementation_guard_v3" or
            record.get("complete") is not True or record.get("all_checks_passed") is not True or
            record.get("sources_unchanged") is not True or record.get("cached_inputs_unchanged") is not True or
            record.get("new_paths") != 0 or record.get("formal_scientific_valuation_paths") != 0 or
            record.get("source_sha256") != design["source_sha256"] or record.get("streams") != specs or
            not _timestamp(record["completed_at"]) < _timestamp(design["created_at"]) or
            len(record.get("records", [])) != 2):
        raise ValueError("The latest independent guard lacks complete pre-fresh implementation roundtrips")
    bundle, prior, grid, tail, target, models, seal = controls
    rows = []
    for i, spec in enumerate(specs):
        source = folder / f"implementation_roundtrip_v1/stream_{i}_samples.npz"
        if record["input_sha256"].get(str(source.resolve())) != sha(source):
            raise ValueError("A cached implementation fixture was replaced")
        with np.load(source, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        validate_array_set(arrays, spec["representations"])
        result = validate_shared_delta_samples(arrays, bundle, prior, grid, tail, target,
            **spec, coefficients=seal["weights"], models=models)
        row = record["records"][i]
        if (any(row.get(k) != v for k, v in spec.items()) or row.get("passed") is not True or
                row.get("bounded_physical_checks") != 4):
            raise ValueError("A cached implementation guard failure was hidden")
        rows.append({**spec, "passed": True, "used_for_acceptance": False,
            "purpose": "Four-path implementation fixture with a different RNG mapping from the scientific streams",
            "bounded_primitive_probes": len(result["probes"])})
    return pd.DataFrame(rows)


def collect_shared_delta_campaign(folder):
    from ._pricing_rough_shared_delta_samples import validate_shared_delta_samples
    from ._pricing_rough_shared_delta_statistics import summarize_shared_delta
    folder = Path(folder).resolve()
    output = folder.parents[1]
    _, entries = preserved(folder)
    design = read(folder / "design_before_valuation.json")
    validate_design(design)
    manifest = read(folder / "manifest.json")
    digest = sha(folder / "design_before_valuation.json")
    if (manifest.get("complete") is not True or manifest.get("all_streams_retained") is not True or
            manifest.get("completed_paths") != sum(s["paths"] for s in STREAMS) or
            manifest.get("design_sha256") != digest or
            any(manifest.get(k) != v for k, v in design.items())):
        raise ValueError("Every predeclared independent stream must complete before collection")
    entries += verify_sources_and_inputs(folder, design, output)
    parent = read(output / PARENT / "manifest.json")
    forbidden = set(parent["forbidden_prior_seeds"]) | {r["seed"] for r in parent["streams"]}
    if not forbidden.issubset(design["forbidden_prior_seeds"]):
        raise ValueError("A spent training/selection/valuation seed escaped the new firewall")
    weights_path = folder / "weights_before_final.json"
    if (sha(weights_path) != COEFFICIENT_SHA or design.get("coefficient_sha256") != COEFFICIENT_SHA or
            sha(output / PARENT / "weights_before_final.json") != COEFFICIENT_SHA):
        raise ValueError("The 193 common-control coefficients were changed or refitted")
    weights = np.asarray(read(weights_path)["weights"], float)
    preflight_path = (output / design["preflight_folder"]).resolve()
    if preflight_path != output / "rough_shared_delta_v1/spent_v1":
        raise ValueError("The declared spent preflight was replaced")
    before, files, equivalence, planning = validate_preflight(preflight_path, design, output, weights)
    entries += files
    controls = load_controls(output)
    bundle, prior, grid, tail, target, models, seal = controls
    if not np.array_equal(np.asarray(seal["weights"], float), weights):
        raise ValueError("The independently loaded original coefficients differ")
    roundtrips = validate_cached_roundtrips(preflight_path, design, controls)
    streams, records, checks, probes, execution = [], [], [], [], []
    for i, spec in enumerate(STREAMS):
        record = read(folder / f"stream_{i}_manifest.json")
        validate_record(record, design, index=i, design_sha256=digest)
        path = folder / f"stream_{i}_samples.npz"
        if record.get("sample_sha256") != sha(path) or record.get("coefficient_sha256") != COEFFICIENT_SHA:
            raise ValueError("An independent stock-bump archive or coefficient identity changed")
        arrays = _load_archive(path, record["array_schema"])
        validate_array_set(arrays, spec["representations"])
        verified = validate_shared_delta_samples(arrays, bundle, prior, grid, tail, target,
            **spec, coefficients=weights, models=models)
        streams.append(verified["stream"])
        checks.append(verified["checks"])
        probes.extend({"seed": spec["seed"], **p} for p in verified["probes"])
        execution.append({**spec, "started_at": record["started_at"],
            "completed_at": record["completed_at"], "elapsed_seconds": record["elapsed_seconds"],
            "cpu_seconds": record.get("cpu_seconds"),
            "utc_span_seconds": (_timestamp(record["completed_at"]) - _timestamp(record["started_at"])).total_seconds(),
            "all_paths_retained": True, "used_for_fitting_or_selection": False})
        records.append(record)
        # Retain only small endpoint arrays for statistics, then release all
        # date/physical/basis payloads before loading the next stream.
        del verified, arrays
    if manifest.get("records") != records:
        raise ValueError("An independent Greek stream was omitted or replaced")
    validate_completion_timing(design, manifest, records)
    tables = summarize_shared_delta(streams, expected_streams=list(STREAMS))
    tables.update(implementation_checks=equivalence, implementation_roundtrips=roundtrips,
        planning_diagnostics=planning,
        path_identity_checks=pd.DataFrame(checks), conditional_probes=pd.DataFrame(probes),
        execution=pd.DataFrame(execution), development_trials=development_trials(preflight_path, before),
        failures=pd.DataFrame(columns=["stage", "seed", "error_type", "message"]))
    return {"tables": tables, "manifest": manifest, "input_files": entries}


def shared_delta_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The fixed shared-control Greek research seed is 42")
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    context.progress("Verifying independent stock bumps and the frozen-policy time-refinement campaign")
    result = collect_shared_delta_campaign(Path(context.output_dir).resolve() / CAMPAIGN)
    tables = result["tables"]
    tables["acceptance"] = delta_gate_summary(tables)
    finite_pass = bool(tables["acceptance"].all_pass.all())
    controls = tables["raw_control"]
    raw_checks = controls.loc[controls.comparison.eq("raw minus shared")]
    stable_checks = controls.loc[controls.comparison.eq("stable minus shared")]
    if len(raw_checks) != 6 or len(stable_checks) != 6:
        raise ValueError("A required raw/control contrast was omitted")
    entries = _unique_entries(result["input_files"])
    names = ("_pricing_rough_shared_delta_collection.py", "_pricing_rough_shared_delta_statistics.py",
        "_pricing_rough_shared_delta_samples.py", "_pricing_rough_price_budget_basis.py")
    sources = {name: sha(Path(__file__).with_name(name)) for name in names}
    fingerprint = hashlib.sha256(json.dumps({"version": "rough_shared_delta_v1",
        "files": entries, "analysis_sources": sources}, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    tables["provenance"] = pd.DataFrame([
        {"item": "Target", "value": "Finite stock differences of the frozen 64-date GH16 lower-policy expectation"},
        {"item": "Controls", "value": "Frozen 193 coefficients; original128 control and distinct centered micro128/256 controls"},
        {"item": "Independence", "value": "Independent valuation seeds across streams and resolutions; common random numbers within each stock-bump stream"},
        {"item": "Selection", "value": "Every fixed stream, bump and representation retained; no valuation fitting, intercept or empirical recentering"},
        {"item": "Accuracy", "value": "Sampling, paired bumps, representation shifts and independent time changes assessed separately at unchanged tolerances"},
        {"item": "Scope", "value": "One four-factor synthetic ATM put; optimal exercise, kernel refinement, real-data calibration and hedge replay remain separate"},
    ])
    manifest = {"backend": "rough_heston_shared_stock_delta", "study_version": "rough_shared_delta_v1",
        "data_kind": "synthetic", "model": "rough_heston", "exercise_style": "american",
        "profile_fingerprint": fingerprint, "parent_profile_fingerprint": PARENT_PROFILE,
        "input_files": entries, "analysis_sources": sources, "campaign": result["manifest"],
        "primary_complete": True, "streams": list(STREAMS), "kernel_factors": 4,
        "primary_paths": 8192, "time_paths": 8192, "seed_count": 4,
        "exercise_intervals": 64, "policy_operator_steps": 128,
        "bump_relatives": [.005, .0025], "common_control_features": 193,
        "no_new_policy_or_control_fits": True, "all_paths_retained": True,
        "prior_paths_used_for_acceptance": False, "tolerances_changed": False,
        "price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002, "delta_rtol": .005,
        "confidence_multiplier": 1.96, "spread_fraction": .1,
        "historical_replay": False, "current_engine_execution": False,
        "execution_mode": "read-only reconstruction of a separately sealed independent campaign",
        "fixed_policy_delta_checks_pass": finite_pass,
        "new_control_stock_delta_validated": finite_pass,
        "raw_control_tolerance_bias_resolved": bool(raw_checks.tolerance_bias_resolved.all()),
        "stable_control_tolerance_bias_resolved": bool(stable_checks.tolerance_bias_resolved.all()),
        "all_raw_control_marginal_intervals_include_zero": bool(raw_checks.zero_in_interval.all()),
        "american_pricing_ready": False, "stock_delta_ready": False, "optimal_delta_ready": False,
        "limitations": [
            "These checks concern a finite-calendar frozen lower policy, not the optimal American delta.",
            "Fine-law microstep centering changes the variance-control representation; it is not a telescoping bridge or pathwise identity with the original control.",
            "The matched and combined refinement comparisons keep the original128-law GH16 decision function fixed.",
            "Normal 1.96 intervals are approximate; raw/control contrasts and all comparator failures remain visible.",
            "Four-factor kernel, exercise-calendar, broader contract coverage and dividend-event sensitivities are not certified by this study.",
            "Independent SPY spot and historically available cash-dividend expectations remain missing.",
        ]}
    collected = {"tables": tables, "figures": {}, "manifest": manifest}
    if compute_figures:
        from ._pricing_rough_shared_delta_reporting import add_shared_delta_reporting
        collected = add_shared_delta_reporting(collected)
    return collected
