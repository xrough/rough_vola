"""Independent exercise-policy comparisons under one frozen dual martingale.

All adaptive choices precede final valuation.  The martingale, including the
original policy's post-stop gate, is applied from the original proxy trace.
Alternative stopping rules never reconstruct that gate or the upper sample.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_ARCHIVE = Path(__file__).resolve().parents[3]/"notebooks/legacy/pricing/rough_policy_improvement_2026-10-18"
HISTORICAL_TIGHTENING_PROFILE = "5cc13128feff1218bf39c210fc3edabba1149ab8f60b206490fbb119492d0531"


def load_historical_tightening(context=None, *, archive=None, compute_figures=True):
    from ._pricing_rough_precision import _read_verified_archive
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = POLICY_ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive/"results/manifest.json").read_text())
    stage = recorded.get("rough_dual_tightening", {})
    if (recorded.get("evidence_version") != "pricing_research_v1" or
            recorded.get("notebook") != "01_pricing_and_exercise" or
            stage.get("backend") != "rough_heston_dual_tightening" or
            stage.get("profile_fingerprint") != HISTORICAL_TIGHTENING_PROFILE):
        raise ValueError("Preserved evidence is not the declared six-case dual-tightening campaign")
    prefix, tables = "rough_dual_tightening_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        try:
            table = pd.read_csv(archive/"results"/(name+".csv"), float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical tightening table count: {name}")
        tables[name.removeprefix(prefix)] = table
    figures = {name.removeprefix(prefix): archive/"results"/(name+".png")
        for name in recorded["figures"] if name.startswith(prefix)} if compute_figures else {}
    manifest = deepcopy(stage)
    manifest.update(historical_replay=True, current_engine_execution=False,
        historical_archive=str(archive.resolve()), historical_archive_manifest_sha256=digest,
        historical_archive_files=len(inventory["files"]),
        historical_execution_created_at=recorded["created_at"],
        american_pricing_ready=False, stock_delta_ready=False)
    if context is not None:
        context.progress("Loading verified dual-tightening evidence; the original six cases are not repriced")
    return {"tables": tables, "figures": figures, "manifest": manifest}


def policy_improvement_trial_specs():
    """Predeclare policies, controls, budgets and six untouched final streams."""
    from ._pricing_rough_dual_tightening import dual_tightening_trial_specs
    cases = dual_tightening_trial_specs()
    candidates = {
        "additive refit": {"basis": "additive", "premium_floor": 0.},
        "signed additive": {"basis": "additive", "premium_floor": -.7},
        "tensor linear": {"basis": "tensor", "degree": 1, "ridge": 1e-4, "premium_floor": 0.},
        "tensor quadratic": {"basis": "tensor", "degree": 2, "ridge": 1e-4, "premium_floor": 0.},
        "tensor quadratic regularized": {"basis": "tensor", "degree": 2, "ridge": 1e-3, "premium_floor": 0.},
    }
    for case in cases:
        case["config"]["valuation_paths"] *= 2
        for options in case["dual_configs"].values():
            options["target_valuation_paths"] = case["config"]["valuation_paths"]
        # Same strongest families as the previous campaign, selected using
        # the base policy's separate dual-training holdout only.
        case["dual_configs"] = {name: case["dual_configs"][name]
            for name in ("soft boundary intrinsic", "micro boundary", "joint micro")}
        case["improvement_paths"] = 2*case["config"]["training_paths"]
        case["selection_paths"] = case["config"]["training_paths"]
        case["policy_candidates"] = deepcopy(candidates)
        case["policy_candidates"]["dense tensor quadratic"] = {"basis": "tensor", "degree": 2, "ridge": 1e-4, "premium_floor": 0., "stock_knots": case["obstacle_proxy"]["stock_knots"]}
    return cases


def _validated_policy_design(trial_specs, seed):
    from numbers import Integral
    from ._pricing_rough_dual_tightening import _validated_tightening_design
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    stripped, extras = [], []
    if not isinstance(trial_specs, (list, tuple)) or not trial_specs:
        raise ValueError("trial_specs must be a nonempty declared case list")
    for spec in trial_specs:
        if not isinstance(spec, dict):
            raise TypeError("Every policy case must be a dictionary")
        item = dict(spec)
        candidates = item.pop("policy_candidates", {})
        count = item.pop("selection_paths", None)
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError("Declare nonempty policy_candidates")
        for name, options in candidates.items():
            if (not isinstance(name, str) or not name or name == "original" or
                    not isinstance(options, dict) or set(options)-{"basis", "degree", "ridge", "premium_floor", "stock_knots"}
                    or options.get("basis") not in ("additive", "tensor")):
                raise ValueError("Policy candidates require distinct names and explicit basis settings")
            if options.get("degree", 1) not in (1, 2) or isinstance(options.get("degree", 1), bool):
                raise ValueError("Policy tensor degree must be one or two")
            ridge = options.get("ridge", 1e-4)
            floor = options.get("premium_floor", 0.)
            if (isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not np.isfinite(ridge) or ridge < 0 or
                    (floor is not None and (isinstance(floor, bool) or not isinstance(floor, (int, float)) or not np.isfinite(floor)))):
                raise ValueError("Policy ridge and floor must be finite valid levels")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
            raise ValueError("selection_paths must be an integer >=2")
        stripped.append(item); extras.append((deepcopy(candidates), int(count)))
    rows = _validated_tightening_design(stripped, seed+140_000_000)
    for index, (row, (candidates, count)) in enumerate(zip(rows, extras)):
        row.update(policy_candidates=candidates, selection_paths=count,
            selection_seed=int(seed)+340_000_000+1000*index)
        row["config"] = replace(row["config"], valuation_seed=int(seed)+350_000_000+1000*index)
        streams = (row["config"].training_seed, row["dual_seed"], row["improvement_seed"],
            row["selection_seed"], row["config"].valuation_seed)
        if len(set(streams)) != 5:
            raise ValueError("Policy investigation requires five distinct streams")
    return rows


def policy_comparison_tables(raw, lower, upper, stops, *, policy_names, selected_name):
    """Paired gains and a common-upper gap, with the unchanged error budget."""
    from ._pricing_rough_stopping_attribution import paired_attribution_tables
    raw, lower, upper, stops = map(np.asarray, (raw, lower, upper, stops))
    if (raw.ndim != 2 or lower.shape != raw.shape or stops.shape != raw.shape or
            upper.shape != (len(raw),) or selected_name not in policy_names):
        raise ValueError("Policy comparison requires aligned samples and a declared selected policy")
    result = paired_attribution_tables(raw, lower[..., None], upper[:, None],
        policy_names=policy_names, control_names=["fixed training-selected martingale"])
    matrix, changes = result["matrix"], result["policy_changes"]
    matrix["selected_before_valuation"] = matrix.policy.eq(selected_name)
    matrix["exercise_change_fraction"] = [float(np.mean(stops[:, i] != stops[:, 0])) for i in range(len(policy_names))]
    baseline_gap = matrix.iloc[0].paired_gap
    if not changes.empty:
        changes["gain_interval_low"] = changes.controlled_lower_change-1.96*changes.controlled_change_standard_error
        changes["gain_interval_high"] = changes.controlled_lower_change+1.96*changes.controlled_change_standard_error
        changes["resolved_lower_gain"] = changes.gain_interval_low.gt(0.)
        changes["selected_before_valuation"] = changes.policy.eq(selected_name)
        changes["exercise_change_fraction"] = matrix.iloc[1:].exercise_change_fraction.to_numpy()
        changes["removed_baseline_gap_fraction"] = changes.controlled_lower_change/baseline_gap if baseline_gap else np.nan
        # Ratio intervals use paired samples with covariance of gain and gap.
        gap = upper-lower[:, 0]
        for index, row in changes.iterrows():
            j = policy_names.index(row.policy)
            ratio = row.removed_baseline_gap_fraction
            se = float(np.std((lower[:, j]-lower[:, 0])-ratio*gap, ddof=1)/np.sqrt(len(raw))/baseline_gap) if baseline_gap else np.nan
            changes.loc[index, "gap_fraction_standard_error"] = se
            changes.loc[index, "gap_fraction_interval_low"] = ratio-1.96*se
            changes.loc[index, "gap_fraction_interval_high"] = ratio+1.96*se
        changes["tested_policy_removes_majority_resolved"] = changes.gap_fraction_interval_low.gt(.5)
        changes["same_upper_samples"] = True
        changes["absolute_rough_policy_error_identified"] = False
        changes["ratio_interval_method"] = "marginal paired delta method; baseline gap denominator is estimated"
    return {"comparison": matrix, "paired_policy_changes": changes}


def _policy_model_identity(policy):
    """Hash immutable decision coefficients before any final path is generated."""
    from dataclasses import fields
    models = []
    for model in policy.continuations:
        content = {}
        for field in fields(model):
            value = getattr(model, field.name)
            if isinstance(value, np.ndarray):
                content[field.name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            else:
                content[field.name] = repr(value)
        models.append({"class": type(model).__name__, "content": content})
    declaration = {"models": models, "exercise_at_inception": bool(policy.exercise_at_inception),
        "floor": getattr(policy, "premium_floor", 0.), "cap": policy.premium_cap,
        "training_seeds": getattr(policy, "training_seeds", (policy.config.training_seed,) if hasattr(policy, "config") else ())}
    return {"sha256": hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest(), **declaration}


def _policy_trial(case, params, market, kernel, folder):
    from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy, value_fourier_stopping_policy, fourier_policy_fingerprint
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    from rvlab.pricing.rough_dual import train_rough_dual_weights
    from rvlab.pricing.rough_policy_diagnostics import fit_controlled_policy_iteration, evaluate_trace_policy, select_trace_policy
    from rvlab.pricing.rough_tensor_premium import tensor_premium_fitter
    from ._pricing_rough_dual_tightening import _select_proxy_controls, _assert_same_policy_traces
    from ._pricing_rough_stopping_attribution import _freeze_json, _save_sample_archive, finite_reference_partition
    config, contract = case["config"], case["contract"]
    base = train_fourier_stopping_policy(params, market, kernel, strike=contract["strike"],
        maturity=contract["maturity"], is_call=contract["is_call"], config=config,
        control_config=config.premium_control, premium_cap=config.premium_cap)
    proxies = {"regression": base, "obstacle": build_rough_obstacle_proxy(base,
        stock_knots=case["obstacle_proxy"]["stock_knots"])}
    traces, fits, controls = {}, {}, {}
    units = {"regression": "unit baseline", "obstacle": "obstacle / unit"}
    for family, proxy in proxies.items():
        count = next(iter(case["dual_configs"].values())).training_paths
        trace = value_fourier_stopping_policy(proxy, config=replace(proxy.config,
            valuation_seed=case["dual_seed"], valuation_paths=count), trace=True)["trace"]
        traces[family+"_dual_training"] = trace
        if family != "regression": _assert_same_policy_traces(traces["regression_dual_training"], trace)
        controls[units[family]] = (family, (1., 1.))
        names = ("joint micro",) if family == "regression" else case["obstacle_proxy"]["controls"]
        for local in names:
            fit = train_rough_dual_weights(proxy, config=case["dual_configs"][local], training_trace=trace)
            name = local if family == "regression" else "obstacle / "+local
            fits[name] = (family, fit); controls[name] = (family, fit)
    selected_control, control_rows = _select_proxy_controls(fits, units)
    family, weights = controls[selected_control]
    proxy = proxies[family]
    trace = value_fourier_stopping_policy(proxy, config=replace(proxy.config,
        valuation_seed=case["improvement_seed"], valuation_paths=case["improvement_paths"]), trace=True)["trace"]
    traces["policy_improvement_training"] = trace
    candidates, failures = {"original": proxy}, []
    for name, options in case["policy_candidates"].items():
        try:
            fitter = tensor_premium_fitter(degree=options.get("degree", 1), ridge=options.get("ridge", 1e-4)) if options["basis"] == "tensor" else None
            candidates[name] = fit_controlled_policy_iteration(proxy, trace, martingale_weights=weights,
                premium_floor=options.get("premium_floor", 0.), premium_cap=config.premium_cap,
                fit_continuation=fitter, stock_knots=options.get("stock_knots"), ridge=options.get("ridge"))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            failures.append({"policy": name, "status": "numerical_failure", "error": str(error)})
    selection_trace = value_fourier_stopping_policy(proxy, config=replace(proxy.config,
        valuation_seed=case["selection_seed"], valuation_paths=case["selection_paths"]), trace=True)["trace"]
    traces["policy_selection"] = selection_trace
    selection = select_trace_policy(selection_trace, candidates, baseline_name="original", martingale_weights=weights)
    candidates = dict(selection["candidates"])
    selected_policy = selection["selected_name"]
    reference = None
    if params.vol_of_vol == 0 and params.v0 == params.theta:
        from rvlab.pricing.rough_policy_reference import build_black_scholes_reference_policy
        from rvlab.pricing.rough_stopping_reference import black_scholes_stopping_reference
        candidates["independent BS reference policy"] = build_black_scholes_reference_policy(proxy, spot_nodes=case["reference_nodes"])
        result = black_scholes_stopping_reference(market, volatility=np.sqrt(params.v0),
            strike=contract["strike"], exercise_times=base.exercise_times, is_call=contract["is_call"],
            spot_nodes=case["reference_nodes"], check_refinement=True)
        reference = {"price": result.price, "delta": result.delta, "settings": dict(result.settings), "diagnostics": dict(result.diagnostics)}
    identities = {name: _policy_model_identity(p) for name, p in candidates.items()}
    frozen = _freeze_json(Path(folder)/"selection_before_valuation.json", {
        "selected_control": selected_control, "control_selection": control_rows,
        "proxy_fingerprint": fourier_policy_fingerprint(proxy), "selected_policy": selected_policy,
        "policy_selection": selection["rows"], "policy_candidates": case["policy_candidates"], "policy_model_identities": identities,
        "policy_diagnostics": {name: dict(p.diagnostics) for name, p in candidates.items()},
        "streams": {"base": config.training_seed, "dual": case["dual_seed"], "improvement": case["improvement_seed"], "selection": case["selection_seed"], "valuation": config.valuation_seed},
        "dual_weights": None if isinstance(weights, tuple) else weights.weights.tolist(),
        "valuation_used_for_selection": False, "reference": reference,
        "same_original_stopping_history_gate_for_all_policies": True})
    # The final trace is generated only after the policy/control decisions seal.
    final_trace = value_fourier_stopping_policy(proxy, trace=True)["trace"]
    traces["final_valuation"] = final_trace
    raw, lower, stops, upper = [], [], [], None
    for policy in candidates.values():
        evaluated = evaluate_trace_policy(final_trace, policy=policy, martingale_weights=weights)
        if upper is None: upper = evaluated.dual_samples[:, 0]
        elif not np.array_equal(upper, evaluated.dual_samples[:, 0]):
            raise ValueError("Policy attribution changed the frozen upper sample")
        raw.append(evaluated.raw_samples[:, 0]); lower.append(evaluated.controlled_samples[:, 0]); stops.append(evaluated.stop_indices[:, 0])
    if identities != {name: _policy_model_identity(p) for name, p in candidates.items()}:
        raise ValueError("Policy coefficients changed after final valuation began")
    raw, lower, stops = np.column_stack(raw), np.column_stack(lower), np.column_stack(stops)
    names = list(candidates)
    tables = policy_comparison_tables(raw, lower, upper, stops, policy_names=names, selected_name=selected_policy)
    tables["training_selection"] = pd.DataFrame(selection["rows"])
    tables["control_selection"] = pd.DataFrame(control_rows)
    tables["failures"] = pd.DataFrame(failures)
    if reference:
        tables["reference_partition"] = finite_reference_partition(tables["comparison"], reference_price=reference["price"],
            reference_change=reference["diagnostics"]["reference_change_diagnostic"], reference_stable=reference["diagnostics"]["price_stable"],
            reference_scope="Independent GBM Gaussian dynamic program; identical finite exercise calendar; no cash dividends")
    archives = [_save_sample_archive(Path(folder)/"paired_policy_endpoints.npz", raw_payoffs=raw,
        controlled_primals=lower, dual_samples=upper, stop_indices=stops)]
    trace_metadata = {}
    for name, trace in traces.items():
        archives.append(_save_sample_archive(Path(folder)/(name+"_trace.npz"), **{key: value for key, value in trace.items() if isinstance(value, np.ndarray)}))
        trace_metadata[name] = {key: value for key, value in trace.items() if not isinstance(value, np.ndarray)}
    return {"status": "completed", "tables": {name: frame.to_dict("records") for name, frame in tables.items()},
        "policy_names": names, "selected_policy": selected_policy, "selected_control": selected_control,
        "frozen_decisions": frozen, "sample_archives": archives, "trace_metadata": trace_metadata,
        "reference": reference, "config": asdict(config), "same_upper_samples": True,
        "same_original_stopping_history_gate_for_all_policies": True,
        "policy_diagnostics": {name: dict(p.diagnostics) for name, p in candidates.items()},
        "dual_fit_diagnostics": {name: dict(fit.diagnostics) for name, (_, fit) in fits.items()}}


def _policy_sources():
    from . import _pricing_rough_stopping_attribution as attribution
    from . import _pricing_rough_dual_tightening as tightening
    from ._pricing_rough_american import _core_source_hashes, stopping_accuracy
    return {**_core_source_hashes(),
        "attribution_plumbing": hashlib.sha256(Path(attribution.__file__).read_bytes()).hexdigest(),
        "tightening_plumbing": hashlib.sha256(Path(tightening.__file__).read_bytes()).hexdigest(),
        "study_numerics": hashlib.sha256("\n".join(inspect.getsource(f) for f in
            (policy_improvement_trial_specs, _validated_policy_design, policy_comparison_tables, _policy_model_identity, _policy_trial, stopping_accuracy)).encode()).hexdigest()}


def rough_policy_improvement_study(context, *, seed=42, trial_specs=None, trial_indices=None,
        reuse_checkpoints=True, compute_figures=True, probe_archives=()):
    from datetime import datetime, timezone
    from numbers import Integral
    import platform
    import scipy
    from rvlab.pricing import approximate_fractional_kernel, PricingError
    from ._pricing_rough_foundation import _checkpoint, _fingerprint, _jsonable, _market
    from ._pricing_rough_american import _american_models, _preserve_sources
    from ._pricing_rough_dual_tightening import _load_tightening_probes
    from ._pricing_rough_stopping_attribution import _verify_sample_archives
    if not isinstance(reuse_checkpoints, bool) or not isinstance(compute_figures, bool):
        raise TypeError("Reuse and figure flags must be boolean")
    design = _validated_policy_design(trial_specs, seed)
    if trial_indices is not None and (not isinstance(trial_indices, (list, tuple)) or
            any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= len(design) for i in trial_indices) or len(set(trial_indices)) != len(trial_indices)):
        raise ValueError("trial_indices must contain distinct valid indices")
    selected = set(range(len(design)) if trial_indices is None else trial_indices)
    probes, probe_records = _load_tightening_probes(probe_archives)
    models, market, sources = _american_models(), _market(), _policy_sources()
    kernels = {n: approximate_fractional_kernel(.5 if n == 1 else .15, 1., n,
        **({} if n == 1 else {"min_time_ratio": .003 if n == 4 else .001, "max_nfev": 4000})) for n in sorted({c["factors"] for c in design})}
    declared = [{**row, "config": asdict(row["config"]), "dual_configs": {name: asdict(v) for name, v in row["dual_configs"].items()}} for row in design]
    common = {"study_version": "rough_policy_improvement_v1", "seed": int(seed), "design": declared,
        "models": {name: asdict(value) for name, value in models.items()}, "market": asdict(market),
        "kernels": {str(n): asdict(v) for n, v in kernels.items()}, "sources": sources, "preserved_probes": probe_records,
        "historical_tightening_profile": HISTORICAL_TIGHTENING_PROFILE,
        "excluded_api_smoke_seed": 900000042, "excluded_api_smoke_scope": "32 paths, 4 time steps, 2 exercise intervals; no acceptance evidence",
        "acceptance": {"price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002, "delta_rtol": .005, "ci_multiplier": 1.96, "effective_half_spread_fraction": .1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = _fingerprint(common)
    folder = Path(context.output_dir)/"rough_policy_improvement_v1"/fingerprint[:16]
    if not reuse_checkpoints: folder /= "attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots = _preserve_sources(folder, {k: v for k, v in sources.items() if k not in ("study_numerics", "attribution_plumbing", "tightening_plumbing")})
    for path in (Path(__file__), Path(__file__).with_name("_pricing_rough_stopping_attribution.py"), Path(__file__).with_name("_pricing_rough_dual_tightening.py")):
        payload = path.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
        destination = folder/"source"/digest/path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != payload: raise ValueError("Changed preserved policy study source")
        if not destination.exists(): destination.write_bytes(payload)
        snapshots.append({"source": str(path), "snapshot": str(destination), "sha256": digest, "bytes": len(payload)})
    collected, summaries, trials, provenance, failures = {}, [], [], [], []
    actual_dual_calls = 0
    for i, case in enumerate(design):
        if i not in selected: continue
        context.progress(f"Independent policy test {i+1}/{len(design)}: {case['case']}")
        def calculate(case=case, i=i):
            if _policy_sources() != sources: raise PricingError("Numerical sources changed before policy trial")
            record = _policy_trial(case, models[case["contract"]["model"]], market, kernels[case["factors"]], folder/f"case_{i:02d}")
            if _policy_sources() != sources: raise PricingError("Numerical sources changed during policy trial")
            record.update(source_changed_during_trial=False, source_after=sources)
            return record
        record, origin = _checkpoint(folder, f"case_{i:02d}", {**common, "case_index": i}, calculate, reuse=reuse_checkpoints)
        _verify_sample_archives(record); provenance.append(origin)
        trials.append({"case": case["case"], "case_index": i, "status": record["status"], "policy_training_seed": case["config"].training_seed,
            "dual_training_seed": case["dual_seed"], "improvement_seed": case["improvement_seed"], "selection_seed": case["selection_seed"], "valuation_seed": case["config"].valuation_seed,
            "training_paths": case["config"].training_paths, "improvement_paths": case["improvement_paths"], "selection_paths": case["selection_paths"], "valuation_paths": case["config"].valuation_paths,
            "steps": case["config"].steps, "exercise_steps": case["config"].exercise_steps, "elapsed_seconds": origin["original_elapsed_seconds"], "reused": origin["reused"], "checkpoint": origin["checkpoint"], "source_changed_during_trial": record.get("source_changed_during_trial")})
        if record["status"] != "completed":
            failures.append({"case": case["case"], "status": record["status"], "error": record.get("error")})
            summaries.append({"case": case["case"], "status": record["status"], "american_pricing_ready": False, "stock_delta_ready": False}); continue
        actual_dual_calls += sum(row.get("candidate") != "unit baseline" for fit in record["dual_fit_diagnostics"].values() for row in fit["candidates"])
        for name, rows in record["tables"].items():
            table = pd.DataFrame(rows); table.insert(0, "case", case["case"]); collected.setdefault(name, []).append(table)
        matrix = pd.DataFrame(record["tables"]["comparison"])
        original = matrix.iloc[0]; chosen = matrix.loc[matrix.selected_before_valuation].iloc[0]
        change = pd.DataFrame(record["tables"]["paired_policy_changes"])
        chosen_change = change.loc[change.selected_before_valuation]
        gain = chosen_change.iloc[0].to_dict() if len(chosen_change) else {}
        summaries.append({"case": case["case"], "status": "completed", "selected_policy": record["selected_policy"], "selected_control": record["selected_control"],
            "original_gap": original.paired_gap, "selected_gap": chosen.paired_gap, "paired_lower_gain": gain.get("controlled_lower_change", 0.),
            "gain_interval_low": gain.get("gain_interval_low", 0.), "gain_interval_high": gain.get("gain_interval_high", 0.),
            "resolved_lower_gain": gain.get("resolved_lower_gain", False), "removed_baseline_gap_fraction": gain.get("removed_baseline_gap_fraction", 0.),
            "gap_fraction_interval_low": gain.get("gap_fraction_interval_low", 0.), "gap_fraction_interval_high": gain.get("gap_fraction_interval_high", 0.),
            "tested_policy_removes_majority_resolved": gain.get("tested_policy_removes_majority_resolved", False),
            "exercise_change_fraction": chosen.exercise_change_fraction, "selected_price_gate_passed": bool(chosen.discrete_price_gate_passed),
            "selected_price_error_budget": chosen.discrete_price_error_budget, "selected_price_allowance": chosen.price_allowance,
            "completed_comparisons": len(matrix), "same_upper_samples": True, "american_pricing_ready": False, "stock_delta_ready": False})
    tables = {name: pd.concat(values, ignore_index=True, sort=False) for name, values in collected.items()}
    for name in ("comparison", "paired_policy_changes", "training_selection", "control_selection", "reference_partition"): tables.setdefault(name, pd.DataFrame())
    candidate_failures = tables.pop("failures", pd.DataFrame())
    tables.update(summary=pd.DataFrame(summaries), trials=pd.DataFrame(trials), provenance=pd.DataFrame(provenance),
        failures=pd.concat([candidate_failures, pd.DataFrame(failures)], ignore_index=True), probes=probes,
        probe_inventory=pd.DataFrame([{"archive": Path(r["folder"]).name, "files": r["files"], "inventory_sha256": r["inventory_sha256"], "independent_acceptance_evidence": False} for r in probe_records]))
    manifest = {**_jsonable(common), "backend": "rough_heston_policy_improvement", "profile_fingerprint": fingerprint,
        "checkpoint_dir": str(folder), "source_snapshots": snapshots, "source_elapsed_seconds": sum(p["original_elapsed_seconds"] for p in provenance),
        "checkpoint_replay": bool(provenance) and all(p["reused"] for p in provenance), "planned_case_count": len(design), "partial_campaign": len(selected) != len(design),
        "current_engine_execution": True, "five_independent_seed_namespaces": True, "valuation_used_for_selection": False, "same_upper_samples": True,
        "original_poststop_gate_frozen": True, "calibration_optimizer_calls": 0,
        "planned_dual_optimizer_calls": sum(sum(len(f["candidate_bins"]) for f in c["dual_configs"].values()) for i, c in enumerate(declared) if i in selected),
        "recorded_dual_optimizer_calls": actual_dual_calls,
        "optimizer_counts_incomplete_on_failed_cases": any(r["status"] != "completed" for r in trials),
        "american_pricing_ready": False, "stock_delta_ready": False,
        "limitations": ["A paired lower gain identifies removable policy error only among the tested rules.", "The remaining rough gap has no exact policy/upper decomposition.", "Bounds concern the declared finite kernel, weak transition law and exercise dates.", "No stock-delta validation or historical conditional-state readiness is inferred.", "Price, delta and spread tolerances are unchanged; independent SPY spot/dividend blockers remain."]}
    return {"tables": tables, "figures": _policy_figures(tables) if compute_figures else {}, "manifest": manifest}


def _policy_figures(tables):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    summary = tables.get("summary", pd.DataFrame())
    if summary.empty or "selected_gap" not in summary: return {}
    summary = summary.loc[summary.status.eq("completed")].reset_index(drop=True)
    labels = [s.replace("independent rough repeat", "Rough repeat").replace("central rough", "Rough").replace("curve specification", "Curve").replace("constant variance", "BS limit").replace("larger path budgets", "More paths").replace("joint time and exercise", "Finer dates") for s in summary.case]
    y = np.arange(len(summary)); figures = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].barh(y+.15, summary.original_gap, height=.28, label="Original policy")
    axes[0].barh(y-.15, summary.selected_gap, height=.28, label="Training-selected policy")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Common upper minus lower, dollars", title="Exercise-policy improvement")
    axes[0].invert_yaxis()
    ratio = summary.selected_price_error_budget/summary.selected_price_allowance
    axes[1].barh(y, ratio, color=np.where(ratio <= 1., "#31856b", "#bf6b4e"))
    axes[1].axvline(1., color="black", ls="--", lw=1)
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Price error budget / unchanged allowance", title="Residual acceptance test")
    axes[1].invert_yaxis()
    for ax in axes: ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0,.08,1,1)); figures["policy_gap_and_acceptance"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    error = np.stack((summary.paired_lower_gain-summary.gain_interval_low, summary.gain_interval_high-summary.paired_lower_gain))
    axes[0].errorbar(summary.paired_lower_gain, y, xerr=error, fmt="o", capsize=3)
    axes[0].axvline(0., color="black", lw=1)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Paired lower gain, dollars (95% marginal CI)", title="Identical dual path maxima")
    axes[0].invert_yaxis()
    ratioerr = np.stack((summary.removed_baseline_gap_fraction-summary.gap_fraction_interval_low, summary.gap_fraction_interval_high-summary.removed_baseline_gap_fraction))
    axes[1].errorbar(summary.removed_baseline_gap_fraction, y, xerr=ratioerr, fmt="o", capsize=3)
    axes[1].axvline(.5, color="black", ls="--", lw=1)
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Fraction of original common-M gap removed", title="Majority threshold is a diagnostic")
    axes[1].invert_yaxis()
    for ax in axes: ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.tight_layout(); figures["paired_policy_gains"] = fig
    return figures


__all__ = ["load_historical_tightening", "policy_improvement_trial_specs", "policy_comparison_tables", "rough_policy_improvement_study", "policy_improvement_probe_archives"]


def policy_improvement_probe_archives():
    """Sealed training-only pilots and reference diagnostics; never acceptance."""
    root = Path(__file__).resolve().parents[3]/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    names = ("rough_policy_basis_probe_2026-10-18", "rough_policy_grid_probe_2026-10-18", "rough_policy_reference_probe_2026-10-18", "rough_policy_reference_probe_2026-10-18_initial_config_failure", "rough_policy_reference_probe_2026-10-18_proxy_config_failure")
    paths = tuple(root/name for name in names)
    for path in paths:
        if not any((path/name).exists() for name in ("preservation_manifest.json", "preservation_inventory.json")):
            raise FileNotFoundError(f"Missing policy probe inventory: {path}")
    return paths
