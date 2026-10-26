"""Frozen-control replication and conditional-law centering audit.

The triggering valuation stream is retained separately. Fresh paths never
retrain or select a policy, martingale, seed, path exclusion or tolerance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parents[3]/"notebooks/legacy/pricing/rough_control_centering_2026-10-18"
POLICY_PROFILE = "16a0ade1e6eda5dc82ce6bd71be0f2d631e9fbbb876cfc3d414ec634c80cc023"
AUDIT_SEED = 430005042
AUDIT_STATES = tuple((p, j) for p in (0, 1, 2) for j in (0, 16, 32, 48))


def load_historical_policy(context=None, *, archive=None, compute_figures=True):
    from ._pricing_rough_precision import _read_verified_archive
    archive = ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive/"results/manifest.json").read_text())
    stage = recorded.get("rough_policy_improvement", {})
    if (recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get("evidence_version") != "pricing_research_v1" or
            stage.get("backend") != "rough_heston_policy_improvement" or
            stage.get("profile_fingerprint") != POLICY_PROFILE):
        raise ValueError("Preserved evidence is not the declared six-case policy campaign")
    prefix, tables = "rough_policy_improvement_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix): continue
        try: table = pd.read_csv(archive/"results"/(name+".csv"), float_precision="round_trip")
        except pd.errors.EmptyDataError: table = pd.DataFrame()
        if len(table) != count: raise ValueError(f"Changed historical policy table count: {name}")
        tables[name.removeprefix(prefix)] = table
    figures = {name.removeprefix(prefix): archive/"results"/(name+".png") for name in recorded["figures"] if name.startswith(prefix)} if compute_figures else {}
    manifest = deepcopy(stage)
    manifest.update(historical_replay=True, current_engine_execution=False,
        historical_archive=str(archive.resolve()), historical_archive_manifest_sha256=digest,
        historical_archive_files=len(inventory["files"]), historical_execution_created_at=recorded["created_at"],
        american_pricing_ready=False, stock_delta_ready=False)
    if context is not None: context.progress("Loading verified policy-improvement campaign; the triggering valuation stays historical")
    return {"tables": tables, "figures": figures, "manifest": manifest}


def control_centering_repeat_specs():
    """Eight fixed fresh streams, sealed before observing their outcomes."""
    return [{"seed": 410005042+1000000*i, "paths": 8192} for i in range(8)]


def _validated_repeats(specs, forbidden):
    from numbers import Integral
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError("Declare a nonempty repeat list")
    seen, output = set(), []
    for row in specs:
        if not isinstance(row, dict) or set(row) != {"seed", "paths"}:
            raise ValueError("Each repeat declares exactly seed and paths")
        if any(isinstance(row[k], (bool, np.bool_)) or not isinstance(row[k], Integral) or row[k] < (0 if k == "seed" else 2) for k in row):
            raise ValueError("Seeds must be nonnegative integers and path counts >=2")
        seed = int(row["seed"])
        if seed in seen or seed in forbidden: raise ValueError("Repeat seeds overlap a spent, fitting, audit or repeated stream")
        seen.add(seed); output.append({"seed": seed, "paths": int(row["paths"])})
    return output


def _tails(samples):
    x = np.asarray(samples, float); centered = x-x.mean(); variance = np.mean(centered**2)
    absolute = np.abs(x); count = max(1, int(np.ceil(.01*len(x))))
    return {"nonzero_fraction": float(np.mean(x != 0.)), "minimum": float(x.min()), "maximum": float(x.max()),
        "skewness": float(np.mean(centered**3)/variance**1.5) if variance else 0.,
        "excess_kurtosis": float(np.mean(centered**4)/variance**2-3.) if variance else 0.,
        "largest_one_percent_absolute_mass_fraction": float(np.sort(absolute)[-count:].sum()/absolute.sum()) if absolute.sum() else 0.}


def _repeat_samples(trace, frozen):
    from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
    from rvlab.pricing.rough_dual import apply_rough_dual_weights
    from ._pricing_rough_centering_statistics import stopped_martingale_contrast
    original = evaluate_trace_policy(trace, policy=frozen["original_policy"], martingale_weights=frozen["weights"])
    selected = evaluate_trace_policy(trace, policy=frozen["selected_policy"], martingale_weights=frozen["weights"])
    if not np.array_equal(original.dual_samples, selected.dual_samples):
        raise ValueError("Control audit changed the shared upper sample")
    m = apply_rough_dual_weights(trace, frozen["weights"])["martingale"][:, :, 0]
    o, s = original.stop_indices[:, 0], selected.stop_indices[:, 0]
    identity = stopped_martingale_contrast(m, o, s)
    raw = selected.raw_samples[:, 0]-original.raw_samples[:, 0]
    controlled = selected.controlled_samples[:, 0]-original.controlled_samples[:, 0]
    contrast = raw-controlled
    allowance = 128*np.finfo(float).eps*(1+np.max(np.abs(m)))
    if not np.allclose(contrast, identity["endpoint_contrast"], rtol=0., atol=allowance):
        raise ValueError("Raw/control contrast differs from the stopped martingale contrast")
    if identity["max_reconstruction_error"] > allowance:
        raise ValueError("Predictable stopping-window identity failed beyond arithmetic roundoff")
    p = np.arange(len(m))
    samples = {"contrast": contrast, "raw_gain": raw, "controlled_gain": controlled,
        "original_stopped_m": m[p, o], "selected_stopped_m": m[p, s], "terminal_m": m[:, -1]}
    fit = frozen["weights"]
    if fit.config.predictable_features: raise ValueError("This frozen component audit requires its declared feature-free outer weights")
    components = {"european": "european_martingale", "premium": "premium_martingale", "micro_boundary": "micro_boundary_martingale"}
    bins = np.searchsorted(fit.time_edges[1:], trace["times"][1:], side="left")
    total = np.zeros(len(m))
    for j, name in enumerate(fit.config.components):
        increments = np.diff(trace[components[name]][:, :, 0], axis=1)*fit.weights[bins, j]
        values = (identity["signed_left_gate"]*increments).sum(axis=1)
        samples[name+"_contrast"] = values; total += values
    if not np.allclose(total, contrast, rtol=0., atol=allowance): raise ValueError("Component contrasts failed to sum to the selected stopped control")
    return samples, {"max_endpoint_identity_error": float(np.max(np.abs(contrast-identity["endpoint_contrast"]))),
        "max_predictable_gate_identity_error": identity["max_reconstruction_error"],
        "arithmetic_allowance": float(allowance), "same_upper_samples": True,
        "stopping_decision_change_fraction": float(np.mean(o != s)), **_tails(contrast)}


def _bundle_summary(frozen):
    """Seal exact historical identities without copying every old coefficient."""
    from ._pricing_rough_frozen_policy import _load_trace
    metadata = frozen["manifest"]
    declaration = metadata["declaration"]
    record = json.loads(Path(metadata["source_checkpoint"]["path"]).read_text())["payload"]
    samples, diagnostics = _repeat_samples(_load_trace(record, "final_valuation"), frozen)
    contrast = samples["contrast"]; estimate = float(contrast.mean())
    se = float(contrast.std(ddof=1)/np.sqrt(len(contrast)))
    comparison = next(r for r in record["tables"]["comparison"] if r["selected_before_valuation"])
    manifest_file = Path(frozen["folder"])/"manifest.json"
    return {"proxy_fingerprint": declaration["proxy_fingerprint"],
        "original_policy_identity":declaration["policy_model_identities"]["original"]["sha256"],
        "selected_policy_identity":declaration["policy_model_identities"][declaration["selected_policy"]]["sha256"],
        "bundle_identity":hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "bundle_manifest":str(manifest_file), "coefficient_archive_sha256":metadata["coefficient_archive_sha256"],
        "source_checkpoint":metadata["source_checkpoint"], "sealed_decisions":metadata["sealed_decisions"],
        "replay_checks":metadata["replay"]["comparisons"], "all_bit_identical":metadata["replay"]["all_bit_identical"],
        "new_dual_fit":False, "new_selection":False, "price_allowance":comparison["price_allowance"],
        "historical_trigger":{"seed":350005042,"paths":len(contrast),"raw_minus_controlled_gain":estimate,
            "raw_minus_controlled_standard_error":se,"confidence_low":estimate-1.96*se,"confidence_high":estimate+1.96*se,
            "z_score":estimate/se,"raw_gain":float(samples["raw_gain"].mean()),"controlled_gain":float(samples["controlled_gain"].mean()),
            **diagnostics}}


def _centering_sources():
    from ._pricing_rough_american import _core_source_hashes
    files = ("_pricing_rough_frozen_policy.py", "_pricing_rough_centering_statistics.py", "_pricing_rough_centering_audit.py",
        "_pricing_rough_american.py", "_pricing_rough_foundation.py", "_pricing_rough_stopping_attribution.py", "_pricing_rough_policy_improvement.py")
    functions = (control_centering_repeat_specs, _validated_repeats, _tails, _repeat_samples, _bundle_summary, _centering_sources, control_centering_study)
    return {**_core_source_hashes(), **{name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
        "study_numerics": hashlib.sha256("\n".join(inspect.getsource(f) for f in functions).encode()).hexdigest()}


def control_centering_study(context, *, seed=42, repeat_specs=None, repeat_indices=None,
        reuse_checkpoints=True, compute_figures=True, run_audit=True):
    from datetime import datetime, timezone
    from numbers import Integral
    import platform
    import scipy
    from rvlab.pricing import PricingError
    from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy
    from ._pricing_rough_foundation import _checkpoint, _fingerprint, _jsonable
    from ._pricing_rough_american import _preserve_sources
    from ._pricing_rough_stopping_attribution import _save_sample_archive, _verify_sample_archives, _freeze_json
    from ._pricing_rough_frozen_policy import reconstruct_joint_frozen_policy
    from ._pricing_rough_centering_statistics import summarize_independent_centering_runs
    from ._pricing_rough_centering_audit import audit_conditional_centers
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("This fixed historical investigation uses the declared seed42 provenance")
    if any(not isinstance(v, bool) for v in (reuse_checkpoints, compute_figures, run_audit)):
        raise TypeError("Reuse, figure and audit flags must be boolean")
    frozen = reconstruct_joint_frozen_policy(context.output_dir)
    frozen = {**frozen, "manifest":_bundle_summary(frozen)}
    proxy = frozen["proxy"]
    forbidden = tuple(set(frozen["original_policy"].training_seeds+frozen["selected_policy"].training_seeds+
        (350005042, AUDIT_SEED)))
    repeats = _validated_repeats(control_centering_repeat_specs() if repeat_specs is None else repeat_specs, forbidden)
    if repeat_indices is not None and (not isinstance(repeat_indices, (tuple, list)) or
            any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(repeats) for i in repeat_indices) or len(set(repeat_indices)) != len(repeat_indices)):
        raise ValueError("repeat_indices must contain distinct valid indices")
    indices = set(range(len(repeats)) if repeat_indices is None else repeat_indices)
    sources = _centering_sources()
    # Only stable bundle identities enter the numerical profile, never replay
    # flags or elapsed wall times from reconstructing the historical functions.
    identity = {"proxy": frozen["manifest"]["proxy_fingerprint"],
        "original": frozen["manifest"]["original_policy_identity"], "selected": frozen["manifest"]["selected_policy_identity"],
        "bundle": frozen["manifest"]["bundle_identity"]}
    common = {"study_version": "rough_control_centering_v1", "seed": 42,
        "historical_policy_profile": POLICY_PROFILE, "fixed_case_index": 5,
        "frozen_identity": identity, "repeats": repeats, "forbidden_seeds": sorted(forbidden),
        "conditional_audit": {"seed": AUDIT_SEED, "paths": 64, "states": AUDIT_STATES,
            "absolute_quadrature_tolerance": 2e-11, "normal_limit": 12.},
        "sources": sources, "primary_quantity": "contrast", "trigger_excluded_from_primary_pool": True,
        "acceptance": {"price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002, "delta_rtol": .005,
            "ci_multiplier": 1.96, "effective_half_spread_fraction": .1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = _fingerprint(common)
    folder = Path(context.output_dir)/"rough_control_centering_v1"/fingerprint[:16]
    if not reuse_checkpoints: folder /= "attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots = _preserve_sources(folder, {k:v for k,v in sources.items() if k.endswith('.py') and Path(__file__).parents[1].joinpath('pricing', k).exists()})
    for name in (Path(__file__).name, *[key for key in sources if key.startswith("_pricing_")]):
        path = Path(__file__).with_name(name); payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
        dest = folder/"source"/digest/name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != payload: raise ValueError("Changed preserved centering source")
        if not dest.exists(): dest.write_bytes(payload)
        snapshots.append({"source": str(path), "snapshot": str(dest), "sha256": digest, "bytes": len(payload)})
    design_seal = _freeze_json(folder/"design_before_fresh_valuation.json", common)
    records, provenance, failures, runs = [], [], [], []
    for index, spec in enumerate(repeats):
        if index not in indices: continue
        context.progress(f"Frozen-control independent stream {index+1}/{len(repeats)}: seed {spec['seed']}")
        def calculate(spec=spec, index=index):
            if _centering_sources() != sources: raise PricingError("Numerical sources changed before independent repeat")
            trace = value_fourier_stopping_policy(proxy, config=replace(proxy.config, valuation_seed=spec["seed"], valuation_paths=spec["paths"]), trace=True)["trace"]
            samples, diagnostics = _repeat_samples(trace, frozen)
            archive = _save_sample_archive(folder/f"repeat_{index:02d}_endpoints.npz", **samples)
            if _centering_sources() != sources: raise PricingError("Numerical sources changed during independent repeat")
            return {"status": "completed", "seed": spec["seed"], "paths": spec["paths"], "sample_archives": [archive],
                "diagnostics": diagnostics, "policy_and_weights_retrained": False, "source_changed_during_trial": False}
        record, origin = _checkpoint(folder, f"repeat_{index:02d}", {**common, "repeat_index": index}, calculate, reuse=reuse_checkpoints)
        _verify_sample_archives(record); provenance.append(origin)
        records.append({"index": index, "seed": spec["seed"], "paths": spec["paths"], "status": record["status"],
            "checkpoint": origin["checkpoint"], "elapsed_seconds": origin["original_elapsed_seconds"], "reused": origin["reused"], **record.get("diagnostics", {})})
        if record["status"] != "completed":
            failures.append({"stage": "independent_repeat", "index": index, "error": record.get("error")}); continue
        with np.load(record["sample_archives"][0]["path"], allow_pickle=False) as arrays:
            runs.append({"seed": spec["seed"], "samples": {name: arrays[name].copy() for name in arrays.files}})
    statistics = summarize_independent_centering_runs(runs, forbidden_seeds=forbidden) if runs else {"per_seed": pd.DataFrame(), "pooled": pd.DataFrame(), "metadata": {}}
    tables = {"per_seed": statistics["per_seed"], "pooled": statistics["pooled"], "repeats": pd.DataFrame(records), "provenance": pd.DataFrame(provenance)}
    audit_manifest = None
    if run_audit:
        context.progress("Checking six-branch/Gaussian conditional centers at twelve independent left states")
        def audit():
            if _centering_sources() != sources: raise PricingError("Numerical sources changed before conditional audit")
            trace = value_fourier_stopping_policy(proxy, config=replace(proxy.config, valuation_seed=AUDIT_SEED, valuation_paths=64), trace=True)["trace"]
            checked = audit_conditional_centers(proxy, trace, state_indices=AUDIT_STATES, martingale_weights=frozen["weights"])
            if _centering_sources() != sources: raise PricingError("Numerical sources changed during conditional audit")
            archive = _save_sample_archive(folder/"conditional_state_trace.npz", **{k:v for k,v in trace.items() if isinstance(v,np.ndarray)})
            return {"status": "completed", "tables": {k:t.to_dict('records') for k,t in checked['tables'].items()}, "audit_manifest": checked['manifest'], "sample_archives": [archive]}
        checked, origin = _checkpoint(folder, "conditional_audit", common, audit, reuse=reuse_checkpoints)
        _verify_sample_archives(checked)
        if checked["status"] == "completed":
            tables.update({"conditional_"+name:pd.DataFrame(rows) for name, rows in checked['tables'].items()}); audit_manifest=checked['audit_manifest']
        else: failures.append({"stage":"conditional_audit", "error":checked.get('error')})
        tables["audit_provenance"] = pd.DataFrame([origin])
    trigger = frozen["manifest"]["historical_trigger"]
    tables["exploratory_trigger"] = pd.DataFrame([{**trigger, "quantity":"contrast", "estimate":trigger["raw_minus_controlled_gain"], "standard_error":trigger["raw_minus_controlled_standard_error"], "included_in_primary_pool":False}])
    tables["reconstruction"] = pd.DataFrame(frozen["manifest"]["replay_checks"])
    tables["failures"] = pd.DataFrame(failures)
    tables["configuration"] = pd.DataFrame([{"item":"fixed policy/control", "value":"original case05; exact coefficient identity verified"},
        {"item":"fresh design", "value":f"{len(repeats)} streams; {sum(x['paths'] for x in repeats)} paths; all declared before outcomes"},
        {"item":"original stream", "value":"350005042 excluded from primary pooled inference"},
        {"item":"conditional audit", "value":"12 predeclared fresh left states, 6 branches/Gaussian scalar integration"}])
    for key in ("conditional_components", "conditional_states", "conditional_branches"): tables.setdefault(key,pd.DataFrame())
    complete = len(runs)==len(repeats) and not failures
    if not tables["pooled"].empty:
        tables["pooled"]["primary_complete"] = complete
        tables["pooled"]["original_price_allowance"] = frozen["manifest"]["price_allowance"]
        tables["pooled"]["interval_within_price_allowance"] = np.maximum(abs(tables["pooled"].confidence_low),abs(tables["pooled"].confidence_high)) <= frozen["manifest"]["price_allowance"]
    manifest = {**_jsonable(common), "backend":"rough_heston_control_centering", "profile_fingerprint":fingerprint,
        "checkpoint_dir":str(folder), "source_snapshots":snapshots, "design_seal":design_seal,
        "primary_complete":complete, "planned_streams":len(repeats), "completed_streams":len(runs),
        "source_elapsed_seconds":sum(x['original_elapsed_seconds'] for x in provenance),
        "checkpoint_replay":bool(provenance) and all(x['reused'] for x in provenance), "current_engine_execution":True,
        "frozen_reconstruction":frozen["manifest"], "conditional_audit":audit_manifest, "statistics":statistics["metadata"],
        "policy_optimizer_calls":0, "dual_optimizer_calls":0, "calibration_optimizer_calls":0,
        "new_policy_or_control_selection":False, "trigger_in_primary_pool":False,
        "centering_error_identified":False, "sampling_noise_proven":False,
        "american_pricing_ready":False, "stock_delta_ready":False,
        "limitations":["Independent mean intervals cannot prove zero bias, especially at the unchanged price allowance.",
            "Conditional integration checks are finite local numerical tests, not a global error bound.",
            "Bounds concern the finite kernel, transition law and exercise dates; price/delta/readiness gates remain unchanged."]}
    result = {"tables":tables, "figures":{}, "manifest":manifest}
    if compute_figures:
        from ._pricing_rough_centering_reporting import add_centering_figures
        result = add_centering_figures(result)
    return result
