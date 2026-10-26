"""Independent-path comparisons of stronger rough-Heston dual controls.

The preceding stopping-attribution campaign is immutable historical evidence.
This follow-up compares upper estimators under one fixed exercise policy, kernel,
transition law and exercise calendar. Selection belongs to training paths only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DUAL_TIGHTENING_ARCHIVE = (Path(__file__).resolve().parents[3]/
    "notebooks/legacy/pricing/rough_dual_tightening_2026-10-18")
HISTORICAL_ATTRIBUTION_PROFILE = "dd48a2ec788ba4bf072cead585345b11f4a79f3880dd0ed0f26196c4853fe678"


def load_historical_attribution(context=None, *, archive=None, compute_figures=True):
    """Replay all six original cases without repricing them under new controls."""
    from ._pricing_rough_precision import _read_verified_archive
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = DUAL_TIGHTENING_ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive/"results/manifest.json").read_text())
    stage = recorded.get("rough_stopping_attribution", {})
    if (recorded.get("evidence_version") != "pricing_research_v1" or
            recorded.get("notebook") != "01_pricing_and_exercise" or
            stage.get("backend") != "rough_heston_stopping_attribution" or
            stage.get("profile_fingerprint") != HISTORICAL_ATTRIBUTION_PROFILE):
        raise ValueError("Preserved evidence is not the declared six-case stopping-attribution campaign")
    prefix, tables = "rough_stopping_attribution_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        path = archive/"results"/(name+".csv")
        try:
            table = pd.read_csv(path, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical attribution table count: {path}")
        tables[name.removeprefix(prefix)] = table
    figures = {name.removeprefix(prefix): archive/"results"/(name+".png")
        for name in recorded["figures"] if name.startswith(prefix)} if compute_figures else {}
    manifest = deepcopy(stage)
    manifest.update({"historical_replay": True, "current_engine_execution": False,
        "historical_archive": str(archive.resolve()),
        "historical_archive_manifest_sha256": digest,
        "historical_archive_files": len(inventory["files"]),
        "historical_execution_created_at": recorded["created_at"],
        "american_pricing_ready": False, "stock_delta_ready": False})
    if context is not None:
        context.progress("Loading byte-verified six-case stopping attribution; previous controls are not recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}


def paired_dual_comparison(raw_payoffs, controlled_primals, dual_samples, *,
                           control_names, reference_control=0):
    """Compare shared-path controls, retaining paired covariance and failures.

    Inputs are (paths,), (paths, controls), (paths, controls). Raw cashflows are
    structurally shared: this helper cannot turn a policy change into a control
    improvement. Price gates retain the existing sampling-plus-gap allowance.
    """
    from ._pricing_rough_stopping_attribution import paired_attribution_tables
    raw = np.asarray(raw_payoffs, float)
    lower, upper = np.asarray(controlled_primals, float), np.asarray(dual_samples, float)
    if raw.ndim != 1 or lower.ndim != 2 or upper.shape != lower.shape or len(raw) != len(lower):
        raise ValueError("Dual comparison requires aligned (path,), (path, control), (path, control) arrays")
    if (isinstance(reference_control, (bool, np.bool_)) or
            not isinstance(reference_control, (int, np.integer)) or
            not 0 <= reference_control < lower.shape[1]):
        raise ValueError("reference_control must identify a declared control")
    order = [int(reference_control)]+[i for i in range(lower.shape[1]) if i != reference_control]
    if not isinstance(control_names, (tuple, list)) or len(control_names) != lower.shape[1]:
        raise ValueError("control_names must align with the control axis")
    names = [control_names[i] for i in order]
    result = paired_attribution_tables(raw[:, None], lower[:, None, order], upper[:, order],
        policy_names=["frozen exercise policy"], control_names=names)
    matrix = result["matrix"]
    changes = result["control_changes"]
    if not changes.empty:
        for name, error in (("dual_change", "dual_change_standard_error"),
                ("controlled_lower_change", "lower_change_standard_error"),
                ("gap_change", "gap_change_standard_error")):
            changes[name+"_interval_low"] = changes[name]-1.96*changes[error]
            changes[name+"_interval_high"] = changes[name]+1.96*changes[error]
        changes["resolved_upper_reduction"] = changes.dual_change_interval_high.lt(0.)
        changes["resolved_gap_reduction"] = changes.gap_change_interval_high.lt(0.)
        changes["lower_change_consistent_with_zero"] = (
            changes.controlled_lower_change_interval_low.le(0.) &
            changes.controlled_lower_change_interval_high.ge(0.))
        original = matrix.iloc[0].paired_gap
        changes["gap_reduction_fraction"] = -changes.gap_change/original if original else np.nan
        identical = {control_names[i]: np.array_equal(lower[:, i], lower[:, reference_control])
            for i in range(lower.shape[1])}
        changes["controlled_lower_samples_identical"] = changes.control.map(identical)
        matrix["controlled_lower_samples_identical_to_reference"] = matrix.control.map(identical)
    return {"comparison": matrix, "paired_changes": changes}


def _tightening_figures(tables):
    """Compact figures retain unchanged thresholds and paired uncertainty."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    comparison = tables.get("comparison", pd.DataFrame())
    if comparison.empty:
        return {}
    figures = {}
    cases = list(comparison.case.drop_duplicates()) if "case" in comparison else ["fixed-policy trial"]
    aliases = {"soft boundary intrinsic": "Prior soft boundary", "unit baseline": "Unit",
        "micro boundary": "Micro boundary", "micro poststop only": "Post-stop only",
        "joint micro": "Joint micro", "obstacle / unit": "Obstacle: unit",
        "obstacle / soft boundary intrinsic": "Obstacle: soft",
        "obstacle / micro boundary": "Obstacle: micro"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=.075, right=.98, bottom=.38, top=.90, wspace=.30)
    controls = list(comparison.control.drop_duplicates())
    offsets = np.linspace(-.28, .28, len(controls))
    colors = plt.get_cmap("tab10").colors
    handles, labels = [], []
    for i, control in enumerate(controls):
        subset = comparison.loc[comparison.control.eq(control)]
        x = np.array([cases.index(c) for c in subset.case]) if "case" in subset else np.arange(len(subset))
        handle = axes[0].errorbar(x+offsets[i], subset.paired_gap.to_numpy()*1000.,
            yerr=1.96*subset.paired_gap_standard_error.to_numpy()*1000.,
            fmt="o", capsize=3, color=colors[i%len(colors)], markersize=4.)
        handles.append(handle); labels.append(aliases.get(control, control))
        allowance = .001+1e-4*np.abs(subset.price.to_numpy())
        required = subset.paired_gap.to_numpy()+1.96*(subset.paired_gap_standard_error.to_numpy()+subset.price_standard_error.to_numpy())
        axes[1].plot(x+offsets[i], required/allowance, "o", color=colors[i%len(colors)], markersize=4.)
    limit = axes[1].axhline(1., color="black", linestyle="--", linewidth=1.)
    handles.append(limit); labels.append("Unchanged price gate")
    axes[0].set(ylabel="paired stopping gap ($ × 1,000)", title="Does the stronger control tighten the gap?")
    axes[1].set(ylabel="required error budget / price allowance", title="A tighter gap can still fail acceptance")
    for ax in axes:
        ax.set_xticks(range(len(cases)), cases, rotation=24, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", alpha=.2)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.01), ncol=3,
        fontsize=9, frameon=False, columnspacing=2.5)
    figures["gap_and_acceptance"] = fig
    changes = tables.get("paired_changes", pd.DataFrame())
    if not changes.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), constrained_layout=True)
        shown = changes.loc[changes.get("selected_before_valuation", pd.Series(True, index=changes.index))]
        if shown.empty: shown = changes
        labels = [str(row.get("case", "trial")) for row in shown.to_dict("records")]
        y = np.arange(len(shown))
        for ax, column, error, title in zip(axes,
                ("gap_change", "dual_change", "controlled_lower_change"),
                ("gap_change_standard_error", "dual_change_standard_error", "lower_change_standard_error"),
                ("Paired gap reduction", "Direct upper reduction", "Lower-control change")):
            sign = 1. if column == "controlled_lower_change" else -1.
            ax.errorbar(sign*shown[column].to_numpy()*1000., y,
                xerr=1.96*shown[error].to_numpy()*1000., fmt="o", capsize=3)
            ax.axvline(0., color="black", linewidth=1.)
            ax.set(title=title, xlabel="$ × 1,000; paired 95% interval")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="x", alpha=.2)
            ax.set_yticks(y, labels if ax is axes[0] else [""]*len(y))
            ax.invert_yaxis()
        figures["paired_upper_changes"] = fig
    poststop = tables.get("poststop_vs_unit", pd.DataFrame())
    if not poststop.empty:
        fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
        y = np.arange(len(poststop))
        ax.errorbar(-poststop.dual_change.to_numpy()*1000., y,
            xerr=1.96*poststop.dual_change_standard_error.to_numpy()*1000.,
            fmt="o", capsize=3, color="#268577")
        ax.axvline(0., color="black", linewidth=1.)
        ax.set_yticks(y, poststop["case"] if "case" in poststop else poststop.control)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", alpha=.2)
        ax.set(title="A post-stop-only control reduces the upper estimate",
            xlabel="Direct upper reduction vs unit ($ × 1,000); paired 95% interval")
        figures["poststop_upper_only"] = fig
    return figures


def _validated_tightening_design(trial_specs, seed):
    """Use fresh namespaces, with the fourth stream reserved for proxy fitting."""
    from ._pricing_rough_stopping_attribution import _validated_attribution_design
    from numbers import Integral
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(trial_specs, (tuple, list)) or not trial_specs:
        raise ValueError("trial_specs must be a nonempty declared case list")
    stripped, obstacles = [], []
    for spec in trial_specs:
        if not isinstance(spec, dict):
            raise TypeError("Each declared dual-tightening case must be a dictionary")
        item = dict(spec)
        obstacles.append(item.pop("obstacle_proxy", None))
        stripped.append(item)
    rows = _validated_attribution_design(stripped, int(seed)+160_000_000)
    for row, obstacle in zip(rows, obstacles):
        if row["policy_variants"]:
            raise ValueError("Dual tightening fixes the exercise policy; policy_variants must be empty")
        if "soft boundary intrinsic" not in row["dual_configs"]:
            raise ValueError("Each case must include the previous soft boundary control as its comparator")
        if any(fit.selection_objective != "error_budget" for fit in row["dual_configs"].values()):
            raise ValueError("Dual tightening selection uses the predeclared error-budget training score")
        if obstacle is not None:
            if not isinstance(obstacle, dict) or set(obstacle) != {"stock_knots", "controls"}:
                raise ValueError("obstacle_proxy declares stock_knots and fitted controls")
            knots = np.asarray(obstacle["stock_knots"])
            controls = obstacle["controls"]
            if (knots.dtype.kind not in "iuf" or knots.ndim != 1 or len(knots) < 2 or
                    not np.isfinite(knots).all() or np.any(knots <= 0) or np.any(np.diff(knots) <= 0)):
                raise ValueError("Obstacle proxy stock knots must increase strictly and be finite positive numbers")
            if (not isinstance(controls, (list, tuple)) or not controls or len(set(controls)) != len(controls)
                    or any(name not in row["dual_configs"] for name in controls)):
                raise ValueError("Obstacle proxy controls must identify distinct declared dual configurations")
            row["obstacle_proxy"] = {"stock_knots": tuple(float(x) for x in knots),
                "controls": tuple(controls)}
        else:
            row["obstacle_proxy"] = None
    return rows


def _select_proxy_controls(fits, unit_names):
    """Compare actual held-out budgets; each proxy retains its own baseline."""
    rows, units = [], {}
    for name, (family, fit) in fits.items():
        diagnostics = fit.diagnostics
        if diagnostics["selection_objective"] != "error_budget":
            raise ValueError("Cross-proxy selection requires unchanged projected error-budget scores")
        baseline = float(diagnostics["candidates"][0]["heldout_error_budget"])
        if family in units and not np.isclose(units[family], baseline, rtol=0., atol=1e-12):
            raise ValueError("Controls within one proxy must share their training holdout")
        if family not in units:
            rows.append({"control": unit_names[family], "proxy_family": family,
                "training_score": baseline, "proxy_unit_training_score": baseline,
                "selected_time_bins": "unit baseline", "selection_objective": "error_budget",
                "selection_source": diagnostics["selection_source"], "valuation_used_for_selection": False})
            units[family] = baseline
        rows.append({"control": name, "proxy_family": family,
            "training_score": float(diagnostics["selected_score"]),
            "proxy_unit_training_score": baseline,
            "selected_time_bins": diagnostics["selected_candidate"],
            "selection_objective": "error_budget", "selection_source": diagnostics["selection_source"],
            "valuation_used_for_selection": False})
    if not rows or not all(np.isfinite(row["training_score"]) for row in rows):
        raise ValueError("A predeclared finite training score is required for every proxy/control")
    selected = min(rows, key=lambda row: row["training_score"])["control"]
    for row in rows: row["selected_before_valuation"] = row["control"] == selected
    return selected, rows


def _assert_same_policy_traces(reference, candidate):
    """A proxy-family comparison must preserve paths and exercise cashflows."""
    for key in ("times", "eligible", "stocks", "factors", "discounted_payoffs", "stop_indices"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Control proxy changed the shared simulation or frozen exercise policy: {key}")


def _tightening_trial(case, params, market, kernel, folder):
    """Train proxy/control alternatives first, then value all on one unseen stream."""
    from dataclasses import asdict, replace
    from rvlab.pricing.rough_stopping_fourier import (
        train_fourier_stopping_policy, value_fourier_stopping_policy, fourier_policy_fingerprint)
    from rvlab.pricing.rough_dual import train_rough_dual_weights
    from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    from ._pricing_rough_stopping_attribution import (
        _freeze_json, _save_sample_archive, finite_reference_partition)
    config, contract = case["config"], case["contract"]
    base = train_fourier_stopping_policy(params, market, kernel,
        strike=contract["strike"], maturity=contract["maturity"], is_call=contract["is_call"],
        config=config, control_config=config.premium_control, premium_cap=config.premium_cap)
    policies = {"regression": base}
    if case["obstacle_proxy"]:
        policies["obstacle"] = build_rough_obstacle_proxy(base,
            stock_knots=case["obstacle_proxy"]["stock_knots"])
        if policies["obstacle"].continuations is not base.continuations:
            raise ValueError("Obstacle proxy replaced the frozen exercise policy")
    unit_names = {"regression": "unit baseline", "obstacle": "obstacle / unit"}
    controls, fits, traces, frozen_fits = {}, {}, {}, {}
    for family, policy in policies.items():
        count = next(iter(case["dual_configs"].values())).training_paths
        numerical = replace(policy.config, valuation_seed=case["dual_seed"], valuation_paths=count)
        trace = value_fourier_stopping_policy(policy, config=numerical, trace=True)["trace"]
        traces[family+"_dual_training"] = trace
        if family != "regression": _assert_same_policy_traces(traces["regression_dual_training"], trace)
        controls[unit_names[family]] = (family, (1., 1.))
        names = case["dual_configs"] if family == "regression" else case["obstacle_proxy"]["controls"]
        for local in names:
            settings = case["dual_configs"][local]
            fit = train_rough_dual_weights(policy, config=settings, training_trace=trace)
            name = local if family == "regression" else "obstacle / "+local
            fits[name], controls[name] = (family, fit), (family, fit)
            frozen_fits[name] = {"proxy_family": family, "time_edges": fit.time_edges.tolist(),
                "weights": fit.weights.tolist(), "config": asdict(fit.config),
                "selected_candidate": fit.diagnostics["selected_candidate"],
                "selected_score": fit.diagnostics["selected_score"],
                "policy_fingerprint": fit.policy_fingerprint}
    selected, selection = _select_proxy_controls(fits, unit_names)
    frozen = _freeze_json(Path(folder)/"selection_before_valuation.json", {
        "proxy_policy_fingerprints": {name: fourier_policy_fingerprint(policy) for name, policy in policies.items()},
        "base_policy_seed": config.training_seed, "dual_training_seed": case["dual_seed"],
        "reserved_proxy_training_seed": case["improvement_seed"], "proxy_training_stream_used": False,
        "reserved_valuation_seed": config.valuation_seed, "obstacle_proxy": case["obstacle_proxy"],
        "selected_control": selected, "selection": selection, "fits": frozen_fits,
        "valuation_used_for_selection": False})
    # No final valuation trace is generated until all proxy fits and selection
    # decisions have been sealed above.
    for family, policy in policies.items():
        trace = value_fourier_stopping_policy(policy, trace=True)["trace"]
        traces[family+"_valuation"] = trace
        if family != "regression": _assert_same_policy_traces(traces["regression_valuation"], trace)
    raw, lower, upper, stops = None, [], [], None
    for name, (family, weights) in controls.items():
        value = evaluate_trace_policy(traces[family+"_valuation"], martingale_weights=weights)
        if raw is None: raw, stops = value.raw_samples[:, 0], value.stop_indices[:, 0]
        elif not (np.array_equal(raw, value.raw_samples[:, 0]) and np.array_equal(stops, value.stop_indices[:, 0])):
            raise ValueError("A control changed raw stopping cashflows or exercise dates")
        lower.append(value.controlled_samples[:, 0]); upper.append(value.dual_samples[:, 0])
    lower, upper = np.column_stack(lower), np.column_stack(upper)
    names = list(controls)
    tables = paired_dual_comparison(raw, lower, upper, control_names=names,
        reference_control=names.index("soft boundary intrinsic"))
    for frame in tables.values(): frame["selected_before_valuation"] = frame.control.eq(selected)
    tables["training_selection"] = pd.DataFrame(selection)
    tables["optimizer_candidates"] = pd.DataFrame([{"control": name, "proxy_family": family, **row,
        "selected_time_bins": row["candidate"] == fit.diagnostics["selected_candidate"],
        "valuation_used_for_selection": False} for name, (family, fit) in fits.items()
        for row in fit.diagnostics["candidates"]])
    unit_changes = paired_dual_comparison(raw, lower, upper, control_names=names,
        reference_control=names.index("unit baseline"))["paired_changes"]
    unit_changes = unit_changes.loc[unit_changes.control.eq("micro poststop only")].copy()
    if not unit_changes.empty:
        index = names.index("micro poststop only")
        roundoff = float(np.max(np.abs(lower[:, index]-lower[:, names.index("unit baseline")])))
        trace = traces["regression_valuation"]
        component = trace["micro_poststop_martingale"][np.arange(len(raw)), stops, 0]
        if np.any(component != 0.): raise ValueError("Post-stop martingale changed its value at the policy stopping time")
        allowance = 64*np.finfo(float).eps*(1+float(np.max(np.abs(raw)))+float(np.max(np.abs(lower))))
        if roundoff > allowance: raise ValueError("Pure post-stop control changed the stopped lower samples beyond arithmetic roundoff")
        unit_changes["maximum_primal_roundoff"] = roundoff
        unit_changes["poststop_component_at_stop_exact_zero"] = True
    tables["poststop_vs_unit"] = unit_changes
    reference = None
    if params.vol_of_vol == 0 and params.v0 == params.theta:
        from rvlab.pricing.rough_stopping_reference import black_scholes_stopping_reference
        result = black_scholes_stopping_reference(market, volatility=np.sqrt(params.v0),
            strike=contract["strike"], exercise_times=base.exercise_times, is_call=contract["is_call"],
            spot_nodes=case["reference_nodes"], check_refinement=True)
        reference = {"price": result.price, "delta": result.delta,
            "settings": dict(result.settings), "diagnostics": dict(result.diagnostics)}
        tables["reference_partition"] = finite_reference_partition(tables["comparison"],
            reference_price=result.price, reference_change=result.diagnostics["reference_change_diagnostic"],
            reference_stable=result.diagnostics["price_stable"],
            reference_scope="Exact GBM transitions and the identical finite exercise calendar; no cash dividends")
    archives = [_save_sample_archive(Path(folder)/"paired_endpoints.npz",
        raw_payoffs=raw, controlled_primals=lower, dual_samples=upper, stop_indices=stops)]
    metadata = {}
    for name, trace in traces.items():
        archives.append(_save_sample_archive(Path(folder)/(name+"_trace.npz"),
            **{key: value for key, value in trace.items() if isinstance(value, np.ndarray)}))
        metadata[name] = {key: value for key, value in trace.items() if not isinstance(value, np.ndarray)}
    return {"status": "completed", "tables": {name: value.to_dict("records") for name, value in tables.items()},
        "control_names": names, "proxy_families": list(policies), "sample_archives": archives,
        "frozen_decisions": frozen, "trace_metadata": metadata, "reference": reference,
        "policy_fingerprint": fourier_policy_fingerprint(base), "selected_control": selected,
        "config": asdict(config), "exercise_policy_fixed": True,
        "same_raw_payoffs_across_controls": True, "same_paths_across_proxies": True,
        "reference_control": "soft boundary intrinsic", "reserved_proxy_training_seed": case["improvement_seed"],
        "proxy_training_stream_used": False,
        "dual_fit_diagnostics": {name: dict(fit.diagnostics) for name, (_, fit) in fits.items()}}


def _tightening_sources():
    import inspect
    from . import _pricing_rough_stopping_attribution as previous
    from ._pricing_rough_american import _core_source_hashes, stopping_accuracy
    functions = (_validated_tightening_design, _select_proxy_controls,
        _assert_same_policy_traces, _tightening_trial,
        paired_dual_comparison, stopping_accuracy)
    return {**_core_source_hashes(),
        "attribution_plumbing": hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),
        "study_numerics": hashlib.sha256("\n".join(inspect.getsource(f) for f in functions).encode()).hexdigest()}


def rough_dual_tightening_study(context, *, seed=42, trial_specs=None,
        trial_indices=None, reuse_checkpoints=True, compute_figures=True,
        probe_archives=()):
    """Bounded fresh-path dual comparisons; no policy or valuation selection."""
    from dataclasses import asdict
    from datetime import datetime, timezone
    from numbers import Integral
    import platform
    import scipy
    from rvlab.pricing import approximate_fractional_kernel, PricingError
    from ._pricing_rough_foundation import _checkpoint, _fingerprint, _jsonable, _market
    from ._pricing_rough_american import _american_models, _preserve_sources
    from ._pricing_rough_stopping_attribution import _verify_sample_archives
    for name, value in (("reuse_checkpoints", reuse_checkpoints), ("compute_figures", compute_figures)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    design = _validated_tightening_design(trial_specs, seed)
    if trial_indices is not None and (not isinstance(trial_indices, (tuple, list)) or
            any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 or v >= len(design)
                for v in trial_indices) or len(set(trial_indices)) != len(trial_indices)):
        raise ValueError("trial_indices must be distinct valid case indices")
    selected = set(range(len(design)) if trial_indices is None else trial_indices)
    probes, probe_records = _load_tightening_probes(probe_archives)
    models, market = _american_models(), _market()
    kernels = {n: approximate_fractional_kernel(.5 if n == 1 else .15, 1., n,
        **({} if n == 1 else {"min_time_ratio": .003 if n == 4 else .001, "max_nfev": 4000}))
        for n in sorted({row["factors"] for row in design})}
    sources = _tightening_sources()
    declared = [{**row, "config": asdict(row["config"]),
        "dual_configs": {name: asdict(value) for name, value in row["dual_configs"].items()}}
        for row in design]
    common = {"study_version": "rough_dual_tightening_v1", "seed": int(seed),
        "design": declared, "models": {name: asdict(value) for name, value in models.items()},
        "market": asdict(market), "kernels": {str(n): asdict(v) for n, v in kernels.items()},
        "sources": sources, "preserved_probes": probe_records,
        "historical_attribution_profile": HISTORICAL_ATTRIBUTION_PROFILE,
        "excluded_api_smoke_namespace": {"policy": 70_000_042, "dual": 80_000_042, "reserved_proxy": 90_000_042, "valuation": 65_000_042, "paths": 32, "steps": 4, "exercise_steps": 2, "purpose": "API pipeline smoke only; these namespaces are excluded from formal valuation"},
        "acceptance": {"price_atol": .001, "price_rtol": 1e-4,
            "delta_atol": .002, "delta_rtol": .005, "ci_multiplier": 1.96,
            "effective_half_spread_fraction": .1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = _fingerprint(common)
    folder = Path(context.output_dir)/"rough_dual_tightening_v1"/fingerprint[:16]
    if not reuse_checkpoints:
        folder /= "attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots = _preserve_sources(folder, {k: v for k, v in sources.items()
        if k not in ("study_numerics", "attribution_plumbing")})
    for path in (Path(__file__), Path(__file__).with_name("_pricing_rough_stopping_attribution.py")):
        payload = path.read_bytes()
        destination = folder/"source"/hashlib.sha256(payload).hexdigest()/path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError("Changed preserved dual-tightening source")
        if not destination.exists(): destination.write_bytes(payload)
        snapshots.append({"source": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
            "snapshot": str(destination), "bytes": len(payload)})
    provenance, failures, trials, collected, summaries = [], [], [], {}, []
    for index, case in enumerate(design):
        if index not in selected:
            continue
        context.progress(f"Dual tightening {index+1}/{len(design)}: {case['case']}")
        def calculate(case=case, index=index):
            before = _tightening_sources()
            if before != sources:
                raise PricingError("Numerical sources changed before the dual-tightening trial")
            result = _tightening_trial(case, models[case["contract"]["model"]], market,
                kernels[case["factors"]], folder/f"case_{index:02d}")
            after = _tightening_sources()
            if after != sources:
                raise PricingError("Numerical sources changed during the dual-tightening trial")
            result.update(source_after=after, source_changed_during_trial=False)
            return result
        record, origin = _checkpoint(folder, f"case_{index:02d}", {**common, "case_index": index},
            calculate, reuse=reuse_checkpoints)
        _verify_sample_archives(record)
        provenance.append(origin)
        trials.append({"case": case["case"], "case_index": index,
            "contract_id": case["contract"]["contract_id"], "factors": case["factors"],
            "status": record["status"], "policy_training_seed": case["config"].training_seed,
            "dual_training_seed": case["dual_seed"], "reserved_proxy_training_seed": case["improvement_seed"],
            "valuation_seed": case["config"].valuation_seed,
            "training_paths": case["config"].training_paths,
            "valuation_paths": case["config"].valuation_paths,
            "steps": case["config"].steps, "exercise_steps": case["config"].exercise_steps,
            "elapsed_seconds": origin["original_elapsed_seconds"], "reused": origin["reused"],
            "checkpoint": origin["checkpoint"],
            "source_changed_during_trial": record.get("source_changed_during_trial")})
        if record["status"] != "completed":
            failures.append({"case": case["case"], "status": record["status"], "error": record.get("error")})
            summaries.append({"case": case["case"], "status": record["status"],
                "american_pricing_ready": False, "stock_delta_ready": False})
            continue
        for name, values in record["tables"].items():
            table = pd.DataFrame(values)
            table.insert(0, "case", case["case"])
            collected.setdefault(name, []).append(table)
        comparison = pd.DataFrame(record["tables"]["comparison"])
        baseline = comparison.loc[comparison.control.eq("soft boundary intrinsic")].iloc[0]
        chosen = comparison.loc[comparison.control.eq(record["selected_control"])].iloc[0]
        summaries.append({"case": case["case"], "status": "completed",
            "training_selected_control": record["selected_control"],
            "previous_control_gap": baseline.paired_gap, "selected_control_gap": chosen.paired_gap,
            "gap_reduction_fraction": 1.-chosen.paired_gap/baseline.paired_gap if baseline.paired_gap else None,
            "selected_price_standard_error": chosen.price_standard_error,
            "selected_price_gate_passed": chosen.discrete_price_gate_passed,
            "completed_comparisons": len(comparison),
            "price_gate_passes": int(comparison.discrete_price_gate_passed.sum()),
            "american_pricing_ready": False, "stock_delta_ready": False})
        for row in record["tables"].get("optimizer_candidates", []):
            if row.get("status") in ("numerical_failure", "optimizer_failure"):
                failures.append({"case": case["case"], "control": row.get("control"),
                    "candidate": row.get("candidate"), "status": row["status"],
                    "error": row.get("error", row.get("message"))})
    tables = {name: pd.concat(values, ignore_index=True, sort=False) for name, values in collected.items()}
    for name in ("comparison", "paired_changes", "training_selection", "optimizer_candidates", "reference_partition"):
        tables.setdefault(name, pd.DataFrame())
    tables.update(summary=pd.DataFrame(summaries), trials=pd.DataFrame(trials),
        provenance=pd.DataFrame(provenance), failures=pd.DataFrame(failures), probes=probes,
        probe_inventory=pd.DataFrame([{"archive": Path(row["folder"]).name,
            "files": row["files"], "inventory_sha256": row["inventory_sha256"],
            "independent_acceptance_evidence": False} for row in probe_records]))
    manifest = {**_jsonable(common), "backend": "rough_heston_dual_tightening",
        "profile_fingerprint": fingerprint, "checkpoint_dir": str(folder), "source_snapshots": snapshots,
        "source_elapsed_seconds": sum(row["original_elapsed_seconds"] for row in provenance),
        "checkpoint_replay": bool(provenance) and all(row["reused"] for row in provenance),
        "planned_case_count": len(design), "selected_case_indices": sorted(selected),
        "partial_campaign": len(selected) != len(design), "current_engine_execution": True,
        "calibration_optimizer_calls": 0, "exercise_policy_fixed": True,
        "planned_dual_optimizer_calls": int(sum(len(fit.candidate_bins)*len(fit.dispersion_penalties)
            for index, case in enumerate(design) if index in selected
            for name, fit in case["dual_configs"].items()
            for _ in range(1+int(bool(case["obstacle_proxy"]) and name in case["obstacle_proxy"]["controls"])))),
        "recorded_dual_optimizer_calls": int(sum(row.get("candidate") != "unit baseline"
            for row in tables["optimizer_candidates"].to_dict("records"))),
        "optimizer_counts_incomplete_on_failed_cases": any(row["status"] != "completed" for row in trials),
        "four_independent_seed_namespaces": True, "fourth_stream_reserved_for_proxy_training": True,
        "executed_path_stream_count": 3, "reserved_proxy_stream_used": False,
        "valuation_used_for_selection": False, "american_pricing_ready": False, "stock_delta_ready": False,
        "timing_interpretation": "Original checkpoint wall runtimes include concurrent contention; campaign wall and cached replay are separate costs.",
        "limitations": ["Bounds concern the fixed finite kernel, transition law and exercise dates.",
            "Training-only representation selection does not use reported valuation paths.",
            "A tighter upper estimate does not certify optimal stock deltas or continuous-model convergence.",
            "All price/delta/spread tolerances remain unchanged; SPY input and historical-state blockers remain."]}
    return {"tables": tables, "figures": _tightening_figures(tables) if compute_figures else {},
        "manifest": manifest}


def add_dual_tightening_execution(result, *, output_dir):
    """Audit actual worker timing separately from checkpoint and replay cost."""
    from datetime import datetime
    profile = result["manifest"].get("profile_fingerprint")
    folder = Path(output_dir)/"rough_dual_tightening_v1/campaign_v1"
    records, starts, ends, case_counts = [], [], [], {}
    if folder.is_dir():
        for path in sorted(folder.rglob("manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest.get("profile_fingerprint") != profile or manifest.get("mode") != "worker":
                continue
            runner = path.parent/"executed_runner.py"
            digest = hashlib.sha256(runner.read_bytes()).hexdigest()
            if digest != manifest.get("runner_sha256"):
                raise ValueError(f"Changed dual-tightening runner evidence: {runner}")
            cold = not manifest.get("checkpoint_replay", False)
            records.append({"manifest": str(path), "profile_fingerprint": profile,
                "runner_sha256": digest, "started_at": manifest["started_at"],
                "completed_at": manifest["completed_at"], "elapsed_seconds": manifest["elapsed_seconds"],
                "checkpoint_replay": manifest.get("checkpoint_replay"),
                "selected_case_indices": manifest.get("selected_case_indices", []),
                "includes_new_numerical_work": cold})
            if cold:
                starts.append(datetime.fromisoformat(manifest["started_at"]))
                ends.append(datetime.fromisoformat(manifest["completed_at"]))
                for index in manifest.get("selected_case_indices", []):
                    case_counts[index] = case_counts.get(index, 0)+1
    result["tables"]["execution_provenance"] = pd.DataFrame(records)
    result["manifest"]["execution_timing"] = {
        "cold_worker_count": len(starts),
        "measured_worker_span_seconds": (max(ends)-min(starts)).total_seconds() if starts else None,
        "cold_case_indices": sorted(case_counts),
        "overlapping_cold_case_indices": sorted(index for index, count in case_counts.items() if count > 1),
        "unique_checkpoint_wall_seconds": result["manifest"].get("source_elapsed_seconds"),
        "interpretation": "Worker span includes startup and output; checkpoint sums include concurrent contention, and are neither measured serial runtime nor CPU time. Cached notebook replay is timed separately."}
    return result


def dual_tightening_probe_archives():
    """Sealed exploratory trials; historical valuation never selects final seeds."""
    root = Path(__file__).resolve().parents[3]
    output = root/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    names = ("rough_dual_dispersion_2026-10-18", "rough_dual_post_stop_2026-10-18",
        "rough_dual_post_switch_2026-10-18", "rough_dual_isolated_switch_2026-10-18",
        "rough_obstacle_control_probe_2026-10-18", "rough_micro_control_probe_2026-10-18",
        "rough_poststop_switch_probe_2026-10-18")
    paths = tuple(output/name for name in names)
    for path in paths:
        _probe_inventory_path(path)
    return paths


def _probe_inventory_path(archive):
    paths = [Path(archive)/name for name in
        ("preservation_manifest.json", "preservation_inventory.json") if (Path(archive)/name).is_file()]
    if len(paths) != 1:
        raise ValueError(f"Exactly one sealed dual-tightening probe inventory is required: {archive}")
    return paths[0]


def _load_tightening_probes(archives=()):
    """Verify both established inventory schemas without rewriting sealed files."""
    if not isinstance(archives, (list, tuple)):
        raise TypeError("probe_archives must be a list or tuple of preserved directories")
    tables, records = [], []
    for value in archives:
        archive = Path(value).resolve()
        manifest_path = _probe_inventory_path(archive)
        payload = manifest_path.read_bytes()
        inventory = json.loads(payload)
        if "inventory_sha256" in inventory:
            canonical = json.dumps(inventory["files"], sort_keys=True, separators=(',', ':')).encode()
            if hashlib.sha256(canonical).hexdigest() != inventory["inventory_sha256"]:
                raise ValueError("Changed dual-tightening probe inventory declaration")
        declared = []
        for row in inventory["files"]:
            relative = row.get("preserved", row.get("path"))
            if not isinstance(relative, str):
                raise ValueError("Probe inventory requires a declared relative file path")
            path = (archive/relative).resolve()
            if not path.is_relative_to(archive):
                raise ValueError("Probe evidence must remain inside its archive")
            content = path.read_bytes()
            if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed preserved dual-tightening probe: {path}")
            declared.append(relative)
        digest = hashlib.sha256(payload).hexdigest()
        for name in sorted(declared):
            if not name.endswith('.csv'):
                continue
            try:
                frame = pd.read_csv(archive/name, float_precision="round_trip")
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            frame["probe"] = Path(name).stem
            frame["source_archive"] = str(archive)
            frame["source_archive_sha256"] = digest
            frame["exploratory_only"] = True
            frame["independent_acceptance_evidence"] = False
            tables.append(frame)
        records.append({"folder": str(archive), "inventory_sha256": digest,
            "files": len(inventory["files"]),
            "interpretation": "Preserved exploratory trials; already inspected historical paths; not independent acceptance"})
    return (pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame(), records)


def dual_tightening_trial_specs():
    """Six predeclared targets, eight controls, and one fixed dense proxy mesh."""
    from ._pricing_rough_stopping_attribution import stopping_attribution_trial_specs
    knots = (.7,.85,.875,.90,.91,.92,.93,.94,.95,.96,.97,.98,.99,1.,1.01,1.02,1.05,1.1,1.2,1.3)
    cases = stopping_attribution_trial_specs()
    for case in cases:
        old = case["dual_configs"]["soft boundary intrinsic"]
        common = {key: old[key] for key in ("training_paths", "candidate_bins", "validation_fraction",
            "european_bounds", "premium_bounds", "objective", "selection_objective", "target_valuation_paths")}
        case["dual_configs"] = {
            "soft boundary intrinsic": old,
            "micro boundary": {**common, "components": ("european", "premium", "micro_boundary"),
                "intrinsic_bounds": (0., 1.5)},
            "micro poststop only": {**common, "premium_bounds": (1., 1.),
                "components": ("european", "premium", "micro_poststop"), "poststop_bounds": (0., 1.5)},
            "joint micro": {**common, "components": ("european", "premium", "micro_boundary", "micro_poststop"),
                "intrinsic_bounds": (0., 1.5), "poststop_bounds": (0., 1.5)},
        }
        case["policy_variants"] = {}
        case["obstacle_proxy"] = {"stock_knots": knots,
            "controls": ("soft boundary intrinsic", "micro boundary")}
    return cases


def add_dual_tightening_extras(result, *, archives=None):
    """Attach late exploratory evidence without changing the formal profile."""
    if archives is None:
        archives = (Path(__file__).resolve().parents[3]/
            "notebooks/_output/pricing/research_v1/01_pricing_and_exercise/rough_tensor_control_probe_2026-10-18",)
    table, records = _load_tightening_probes(archives)
    result["tables"]["supplemental_probes"] = table
    result["tables"]["supplemental_probe_inventory"] = pd.DataFrame([
        {"archive": Path(row["folder"]).name, "files": row["files"],
         "inventory_sha256": row["inventory_sha256"], "formal_campaign_input": False}
        for row in records])
    result["manifest"]["experimental_extras"] = [{**row, "formal_campaign_input": False,
        "interpretation": "Display-only tensor-proxy failures; concurrent source changes retained; no formal selection or acceptance input"}
        for row in records]
    return result
