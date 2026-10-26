"""Controlled attribution of exercise-policy and dual-upper approximation.

Historical accuracy results keep their original source identity.  New studies
compare policies under one frozen martingale and controls under one fixed
policy, with all fitting and selection confined to training paths.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ATTRIBUTION_ARCHIVE = (Path(__file__).resolve().parents[3]/
    "notebooks/legacy/pricing/rough_stopping_attribution_2026-10-18")
HISTORICAL_ACCURACY_PROFILE = "0f8cf9ddc07014b240ad15f8bcd76226322975de4168ee244966d230a772b141"


def load_historical_accuracy(context=None, *, archive=None, compute_figures=True):
    """Replay the complete 14-trial campaign without current-source repricing."""
    from ._pricing_rough_precision import _read_verified_archive
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = ATTRIBUTION_ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive/"results/manifest.json").read_text())
    stage = recorded.get("rough_american_accuracy", {})
    if (recorded.get("evidence_version") != "pricing_research_v1" or
            recorded.get("notebook") != "01_pricing_and_exercise" or
            stage.get("backend") != "rough_heston_lifted_primal_dual" or
            stage.get("profile_fingerprint") != HISTORICAL_ACCURACY_PROFILE):
        raise ValueError("Preserved evidence is not the declared 14-trial rough stopping accuracy campaign")
    prefix, tables = "rough_american_accuracy_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        path = archive/"results"/(name+".csv")
        try:
            table = pd.read_csv(path, float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical accuracy table count: {path}")
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
        context.progress("Loading byte-verified 14-trial stopping accuracy campaign; no historical fit or price is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}


def _mean_se(samples):
    return float(np.mean(samples)), float(np.std(samples, ddof=1)/np.sqrt(len(samples)))


def paired_attribution_tables(raw_payoffs, controlled_primals, dual_samples, *,
                              policy_names, control_names):
    """Summarize a shared-path policy-by-control counterfactual matrix.

    ``raw_payoffs`` has shape (paths, policies), ``controlled_primals`` has
    shape (paths, policies, controls), and ``dual_samples`` has shape (paths,
    controls). Thus one control defines the *same* dual path maximum for every
    policy, while a policy defines identical raw payoffs across controls.
    Mean differences use their paired samples, retaining covariance.
    """
    from ._pricing_rough_american import stopping_accuracy
    raw = np.asarray(raw_payoffs, float)
    primal, upper = np.asarray(controlled_primals, float), np.asarray(dual_samples, float)
    if (raw.ndim != 2 or primal.ndim != 3 or upper.ndim != 2 or
            len(raw) < 2 or primal.shape[:2] != raw.shape or
            primal.shape[::2] != upper.shape or min(primal.shape[1:]) < 1 or
            not all(np.isfinite(value).all() for value in (raw, primal, upper))):
        raise ValueError("Attribution samples require aligned finite (path, policy, control) arrays with at least two paths")
    for names, count, label in ((policy_names, raw.shape[1], "policy"),
            (control_names, upper.shape[1], "control")):
        if (not isinstance(names, (tuple, list)) or len(names) != count or
                any(not isinstance(name, str) or not name for name in names) or
                len(set(names)) != count):
            raise ValueError(f"{label}_names must be distinct nonempty names aligned with the sample axes")
    gaps = upper[:, None, :]-primal
    if np.min(gaps) < -1e-10:
        raise ValueError("Dual samples must dominate every policy's controlled payoff under the same martingale")
    rows, policy_changes, control_changes = [], [], []
    for p, policy in enumerate(policy_names):
        raw_mean, raw_se = _mean_se(raw[:, p])
        for c, control in enumerate(control_names):
            lower, lower_se = _mean_se(primal[:, p, c])
            high, high_se = _mean_se(upper[:, c])
            gap, gap_se = _mean_se(gaps[:, p, c])
            checks = stopping_accuracy(lower, lower_se, high, gap, gap_se)
            rows.append({"policy": policy, "control": control, "paths": len(raw),
                "raw_price": raw_mean, "raw_standard_error": raw_se,
                "price": lower, "price_standard_error": lower_se,
                "dual_price": high, "dual_standard_error": high_se,
                "paired_gap": gap, "paired_gap_standard_error": gap_se,
                **{name: np.asarray(value).item() for name, value in checks.items()},
                "american_pricing_ready": False, "stock_delta_ready": False})
            if p:
                raw_change, raw_change_se = _mean_se(raw[:, p]-raw[:, 0])
                change, change_se = _mean_se(primal[:, p, c]-primal[:, 0, c])
                policy_changes.append({"policy": policy, "reference_policy": policy_names[0],
                    "control": control, "raw_lower_change": raw_change,
                    "raw_change_standard_error": raw_change_se,
                    "controlled_lower_change": change,
                    "controlled_change_standard_error": change_se,
                    "dual_change": 0., "same_martingale": True,
                    "interpretation": "Paired lower-policy difference under one frozen control; not total optimality error"})
            if c:
                lower_change, lower_change_se = _mean_se(primal[:, p, c]-primal[:, p, 0])
                upper_change, upper_change_se = _mean_se(upper[:, c]-upper[:, 0])
                gap_change, gap_change_se = _mean_se(gaps[:, p, c]-gaps[:, p, 0])
                control_changes.append({"policy": policy, "control": control,
                    "reference_control": control_names[0], "raw_payoff_change": 0.,
                    "controlled_lower_change": lower_change,
                    "lower_change_standard_error": lower_change_se,
                    "dual_change": upper_change, "dual_change_standard_error": upper_change_se,
                    "gap_change": gap_change, "gap_change_standard_error": gap_change_se,
                    "same_exercise_policy": True,
                    "interpretation": "Paired upper/control change with an unchanged stopping policy; lower expectation is unchanged"})
    return {"matrix": pd.DataFrame(rows), "policy_changes": pd.DataFrame(policy_changes),
        "control_changes": pd.DataFrame(control_changes)}


def finite_reference_partition(matrix, *, reference_price, reference_change,
                               reference_stable, reference_scope):
    """Split known-calendar gaps using a separately resolved reference.

    The numerical reference change is an empirical resolution diagnostic, not
    a rigorous error bound. Marginal normal intervals include that change
    conservatively. Negative point components are retained, not clipped or
    renormalized. This decomposition must not use a different stopping target.
    """
    from ._pricing_rough_american import CONFIDENCE_MULTIPLIER
    if (not isinstance(matrix, pd.DataFrame) or
            not np.isfinite(reference_price) or not np.isfinite(reference_change) or
            reference_change < 0 or not isinstance(reference_stable, (bool, np.bool_)) or
            not isinstance(reference_scope, str) or not reference_scope):
        raise ValueError("Reference partition requires a finite price, nonnegative change, explicit stability and target scope")
    rows = []
    for row in matrix.to_dict("records"):
        lower_error = reference_price-row["price"]
        upper_error = row["dual_price"]-reference_price
        lower_margin = CONFIDENCE_MULTIPLIER*row["price_standard_error"]+reference_change
        upper_margin = CONFIDENCE_MULTIPLIER*row["dual_standard_error"]+reference_change
        rows.append({"policy": row["policy"], "control": row["control"],
            "reference_price": reference_price, "reference_change_diagnostic": reference_change,
            "reference_stable": bool(reference_stable), "reference_scope": reference_scope,
            "estimated_policy_shortfall": lower_error,
            "policy_shortfall_interval_low": lower_error-lower_margin,
            "policy_shortfall_interval_high": lower_error+lower_margin,
            "estimated_upper_excess": upper_error,
            "upper_excess_interval_low": upper_error-upper_margin,
            "upper_excess_interval_high": upper_error+upper_margin,
            "reconstructed_gap": lower_error+upper_error,
            "policy_point_share": lower_error/row["paired_gap"] if row["paired_gap"] > 0 else np.nan,
            "upper_point_share": upper_error/row["paired_gap"] if row["paired_gap"] > 0 else np.nan,
            "interpretation": "Empirical same-calendar reference split; shares are point diagnostics, not confidence intervals",
            "american_pricing_ready": False, "stock_delta_ready": False})
    return pd.DataFrame(rows)


def _attribution_figures(tables):
    """Compact figures for paired comparisons; every table remains available."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    matrix = tables.get("matrix", pd.DataFrame())
    if matrix.empty:
        return {}
    figures = {}
    case_column = "case" if "case" in matrix else None
    base_policy = matrix.policy.iloc[0]
    controls = matrix.loc[matrix.policy.eq(base_policy)].copy()
    groups = list(controls.groupby("case", sort=False)) if case_column else [("Declared finite stopping problem", controls)]
    columns = min(2, len(groups))
    rows = int(np.ceil(len(groups)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.1*columns, max(3.4*rows, 4.)),
        squeeze=False, constrained_layout=True)
    for axis, (case, selected) in zip(axes.ravel(), groups):
        y = np.arange(len(selected))
        for index, row in enumerate(selected.to_dict("records")):
            axis.plot([row["price"], row["dual_price"]], [index, index], color="#90A4B1", lw=4, alpha=.65)
        axis.errorbar(selected.price, y-.08, xerr=1.96*selected.price_standard_error,
            fmt="o", color="#2878A5", capsize=3, label="Frozen-policy lower ± 95% interval")
        axis.errorbar(selected.dual_price, y+.08, xerr=1.96*selected.dual_standard_error,
            fmt="s", color="#B76B32", capsize=3, label="Dual upper ± 95% interval")
        axis.set_yticks(y, selected.control)
        axis.invert_yaxis()
        axis.set(xlabel="Discounted price per underlying unit ($)", title=str(case))
        axis.ticklabel_format(axis="x", useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    for axis in axes.ravel()[len(groups):]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="outside upper center", ncol=2)
    figures["fixed_policy_bounds"] = fig
    policy = tables.get("policy_changes", pd.DataFrame())
    control = tables.get("control_changes", pd.DataFrame())
    panels = []
    if not policy.empty:
        # A single declared control per case makes lower gains easy to compare;
        # choosing the first control is design order, never outcome selection.
        first = matrix.control.iloc[0]
        selected = policy.loc[policy.control.eq(first)]
        if selected.policy.eq("positive policy refit").any():
            selected = selected.loc[selected.policy.eq("positive policy refit")]
        if not selected.empty:
            panels.append((selected, "controlled_lower_change", "controlled_change_standard_error",
                "Positive policy refit under the original martingale", "policy", 1.))
    if not control.empty:
        selected = control.loc[control.policy.eq(base_policy)]
        if selected.control.eq("soft boundary intrinsic").any():
            selected = selected.loc[selected.control.eq("soft boundary intrinsic")]
        if not selected.empty:
            panels.append((selected, "dual_change", "dual_change_standard_error",
                "Soft boundary control: gap and upper changes", "control", -1.))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6.2*len(panels),
            max(4., .43*max(len(panel[0]) for panel in panels)+1.1)),
            squeeze=False, constrained_layout=True)
        for axis, (selected, value, se, title, name, sign) in zip(axes[0], panels):
            names = [str(row["case"]) if "case" in row else row[name]
                     for row in selected.to_dict("records")]
            y = np.arange(len(selected))
            if name == "control":
                axis.errorbar(-selected.gap_change, y-.09, xerr=1.96*selected.gap_change_standard_error,
                    fmt="o", color="#268577", capsize=3, label="Paired gap reduction")
                axis.errorbar(-selected.dual_change, y+.09, xerr=1.96*selected.dual_change_standard_error,
                    fmt="s", color="#B76B32", capsize=3, label="Direct upper reduction")
                axis.legend(fontsize=8)
            else:
                axis.errorbar(sign*selected[value], y, xerr=1.96*selected[se],
                    fmt="o", color="#268577", capsize=3)
            axis.axvline(0., color="#89949B", linestyle="--", lw=1)
            axis.set_yticks(y, names)
            axis.invert_yaxis()
            axis.set(title=title, xlabel="Paired improvement ($), positive is better")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        figures["paired_attribution"] = fig
    partition = tables.get("reference_partition", pd.DataFrame())
    if not partition.empty:
        selected = partition.loc[partition.policy.eq(base_policy)]
        fig, axes = plt.subplots(1, 2, figsize=(11., max(3.8, .45*len(selected)+1.2)),
            constrained_layout=True)
        y = np.arange(len(selected))
        labels = selected.control.tolist()
        for axis, center, low, high, title, color in (
            (axes[0], "estimated_policy_shortfall", "policy_shortfall_interval_low",
             "policy_shortfall_interval_high", "Same-calendar BS reference − lower policy", "#2878A5"),
            (axes[1], "estimated_upper_excess", "upper_excess_interval_low",
             "upper_excess_interval_high", "Dual upper − same-calendar BS reference", "#B76B32")):
            axis.errorbar(selected[center], y,
                xerr=np.vstack((selected[center]-selected[low], selected[high]-selected[center])),
                fmt="o", color=color, capsize=3)
            axis.axvline(0., color="#89949B", linestyle="--", lw=1)
            axis.set_yticks(y, labels)
            axis.invert_yaxis()
            axis.set(title=title, xlabel="Estimated component ($)")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        figures["known_limit_partition"] = fig
    return figures


def _save_sample_archive(path, **arrays):
    """Preserve the actual paired samples used for attribution, atomically."""
    import tempfile
    path = Path(path)
    if not arrays or any(np.asarray(value).dtype.kind not in "biuf" for value in arrays.values()):
        raise ValueError("Sample archives require named numeric arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if (set(previous.files) != set(arrays) or any(
                    previous[name].dtype != np.asarray(value).dtype or
                    not np.array_equal(previous[name], value, equal_nan=True)
                    for name, value in arrays.items())):
                raise ValueError(f"Refusing to replace different paired samples: {path}")
        payload = path.read_bytes()
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        payload = temporary.read_bytes()
        temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload), "arrays": {name: {"shape": list(np.shape(value)),
            "dtype": str(np.asarray(value).dtype)} for name, value in arrays.items()}}


def _verify_sample_archives(record):
    """A checkpoint cannot silently lose or change the paired evidence."""
    for item in record.get("sample_archives", []):
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired-sample evidence: {path}")
        payload = path.read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Changed paired-sample evidence: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(item["arrays"]):
                raise ValueError(f"Changed paired-sample array names: {path}")
            for name, declaration in item["arrays"].items():
                if (list(arrays[name].shape) != declaration["shape"] or
                        str(arrays[name].dtype) != declaration["dtype"]):
                    raise ValueError(f"Changed paired-sample array schema: {path}/{name}")
    decision = record.get("frozen_decisions")
    if decision:
        path = Path(decision["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen training decisions: {path}")
        payload = path.read_bytes()
        if len(payload) != decision["bytes"] or hashlib.sha256(payload).hexdigest() != decision["sha256"]:
            raise ValueError(f"Changed frozen training decisions: {path}")
    return record


def _validated_attribution_design(trial_specs, seed):
    """Resolve all configurations and independent streams before simulation."""
    from dataclasses import replace
    from numbers import Integral
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    from rvlab.pricing.rough_dual import RoughDualConfig
    from ._pricing_rough_american import american_contracts
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(trial_specs, (list, tuple)) or not trial_specs:
        raise ValueError("trial_specs must be a nonempty declared case list")
    contracts = american_contracts().set_index("contract_id")
    design, labels = [], set()
    for index, spec in enumerate(trial_specs):
        if not isinstance(spec, dict) or set(spec)-{
                "case", "contract_id", "factors", "config", "dual_configs",
                "policy_variants", "improvement_paths", "reference_nodes"}:
            raise ValueError(f"Malformed attribution case {index}")
        label, contract_id = spec.get("case"), spec.get("contract_id")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"Attribution case {index} requires a distinct nonempty name")
        if contract_id not in contracts.index:
            raise ValueError(f"Attribution case {index} has an undeclared contract")
        contract = {"contract_id": contract_id, **contracts.loc[contract_id].to_dict()}
        if contract["scenario"] != "no dividends":
            raise ValueError("Stopping attribution currently requires an explicitly no-cash market")
        overrides = spec.get("config", {})
        if not isinstance(overrides, dict) or {"training_seed", "valuation_seed", "dual_config"} & set(overrides):
            raise ValueError("Case configuration must be a dictionary without assigned seeds or nested dual fitting")
        config = RoughAmericanConfig(training_seed=int(seed)+10_000_000+1000*index,
            valuation_seed=int(seed)+5_000_000+1000*index, **overrides)
        if config.training_method != "fourier_premium":
            raise ValueError("Attribution cases currently require fourier_premium training")
        factors = spec.get("factors", 1 if contract["model"] == "Hurst .5" else 4)
        if isinstance(factors, bool) or not isinstance(factors, Integral) or factors not in (1, 4, 8):
            raise ValueError("Attribution factors must be 1, 4 or 8")
        if (factors == 1) != (contract["model"] == "Hurst .5"):
            raise ValueError("One-factor attribution is reserved for Hurst .5")
        dual_seed, improvement_seed = int(seed)+20_000_000+1000*index, int(seed)+30_000_000+1000*index
        dual_specs = spec.get("dual_configs", {})
        if not isinstance(dual_specs, dict):
            raise TypeError("dual_configs must map distinct control names to explicit configurations")
        dual_configs = {}
        for name, options in dual_specs.items():
            if not isinstance(name, str) or not name or name == "unit baseline" or not isinstance(options, dict):
                raise ValueError("Dual controls require distinct nonempty names and configuration dictionaries")
            if "training_seed" in options:
                raise ValueError("Dual seeds are assigned from a separate campaign stream")
            dual_configs[name] = RoughDualConfig(training_seed=dual_seed, **options)
        if len({value.training_paths for value in dual_configs.values()}) > 1:
            raise ValueError("Controls compared on one training trace must declare the same path count")
        variants = spec.get("policy_variants", {})
        if not isinstance(variants, dict):
            raise TypeError("policy_variants must map distinct names to floor/cap dictionaries")
        for name, options in variants.items():
            if (not isinstance(name, str) or not name or name == "base policy" or
                    not isinstance(options, dict) or set(options)-{"premium_floor", "premium_cap"}):
                raise ValueError("Policy variants require names and explicit premium floor/cap settings")
            floor, cap = options.get("premium_floor", 0.), options.get("premium_cap", config.premium_cap)
            if ((floor is not None and (isinstance(floor, bool) or not np.isfinite(floor))) or
                    (cap is not None and (isinstance(cap, bool) or not np.isfinite(cap))) or
                    (floor is not None and cap is not None and floor > cap)):
                raise ValueError("Policy premium limits must be finite ordered levels or None")
        improvement_paths = spec.get("improvement_paths", config.training_paths)
        if isinstance(improvement_paths, bool) or not isinstance(improvement_paths, Integral) or improvement_paths < 2:
            raise ValueError("improvement_paths must be an integer >= 2")
        reference_nodes = spec.get("reference_nodes", 32769)
        if isinstance(reference_nodes, bool) or not isinstance(reference_nodes, Integral) or reference_nodes < 65:
            raise ValueError("reference_nodes must be an integer >= 65")
        design.append({"case": label, "contract": contract, "factors": int(factors),
            "config": config, "dual_configs": dual_configs,
            "policy_variants": deepcopy(variants), "dual_seed": dual_seed,
            "improvement_seed": improvement_seed, "improvement_paths": int(improvement_paths),
            "reference_nodes": int(reference_nodes)})
        labels.add(label)
    streams = [value for row in design for value in (row["config"].training_seed,
        row["config"].valuation_seed, row["dual_seed"], row["improvement_seed"])]
    if len(set(streams)) != len(streams):
        raise ValueError("Policy, dual, improvement and valuation streams must be disjoint across the campaign")
    return design


def _freeze_json(path, value):
    """Write a declaration once; a replay may verify but cannot replace it."""
    from ._pricing_rough_foundation import _jsonable
    payload = (json.dumps(_jsonable(value), indent=2, allow_nan=False)+"\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Refusing to replace frozen training decisions: {path}")
    if not path.exists():
        path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _select_control_from_training(fits):
    """Choose representation on held-out training score, before valuation."""
    if not fits:
        return "unit baseline", []
    rows, baseline_score = [], None
    for name, fit in fits.items():
        description = fit.diagnostics
        if description["selection_objective"] != "error_budget":
            raise ValueError("Cross-representation selection requires the declared error-budget training score")
        baseline = description["candidates"][0]["heldout_error_budget"]
        if baseline_score is None:
            baseline_score = baseline
        elif not np.isclose(baseline, baseline_score, rtol=0., atol=1e-12):
            raise ValueError("Controls must share the same held-out training sample and target path budget")
        score = description["selected_score"]
        if not np.isfinite(score):
            raise ValueError("Training selection score must be finite")
        rows.append({"control": name, "training_score": score,
            "selected_time_bins": description["selected_candidate"],
            "selection_source": description["selection_source"],
            "selection_objective": description["selection_objective"],
            "valuation_used_for_selection": False})
    rows.insert(0, {"control": "unit baseline", "training_score": baseline_score,
        "selected_time_bins": "unit baseline", "selection_source": "held-out dual-training paths only",
        "selection_objective": "error_budget", "valuation_used_for_selection": False})
    selected = min(rows, key=lambda row: row["training_score"])["control"]
    for row in rows:
        row["selected_before_valuation"] = row["control"] == selected
    return selected, rows


def _attribution_trial(case, params, market, kernel, folder):
    """One predeclared target with shared-path policy/control counterfactuals."""
    from dataclasses import asdict, replace
    from rvlab.pricing.rough_stopping_fourier import (
        train_fourier_stopping_policy, value_fourier_stopping_policy,
        fourier_policy_fingerprint,
    )
    from rvlab.pricing.rough_dual import train_rough_dual_weights
    from rvlab.pricing.rough_policy_diagnostics import (
        fit_controlled_policy_iteration, evaluate_trace_policy,
    )
    config, contract = case["config"], case["contract"]
    policy = train_fourier_stopping_policy(params, market, kernel,
        strike=contract["strike"], maturity=contract["maturity"],
        is_call=contract["is_call"], config=config,
        control_config=config.premium_control, premium_cap=config.premium_cap)
    traces = {}
    fits, policies, frozen_fits = {}, {"base policy": None}, {}
    if case["dual_configs"]:
        count = next(iter(case["dual_configs"].values())).training_paths
        numerical = replace(config, valuation_seed=case["dual_seed"], valuation_paths=count)
        traces["dual_training"] = value_fourier_stopping_policy(policy,
            config=numerical, trace=True)["trace"]
        for name, settings in case["dual_configs"].items():
            fit = train_rough_dual_weights(policy, config=settings,
                training_trace=traces["dual_training"])
            fits[name] = fit
            frozen_fits[name] = {"time_edges": fit.time_edges.tolist(),
                "weights": fit.weights.tolist(), "config": asdict(fit.config),
                "selected_candidate": fit.diagnostics["selected_candidate"],
                "selected_score": fit.diagnostics["selected_score"],
                "policy_fingerprint": fit.policy_fingerprint}
    selected, selection_rows = _select_control_from_training(fits)
    if case["policy_variants"]:
        numerical = replace(config, valuation_seed=case["improvement_seed"],
            valuation_paths=case["improvement_paths"])
        traces["policy_training"] = value_fourier_stopping_policy(policy,
            config=numerical, trace=True)["trace"]
        for name, settings in case["policy_variants"].items():
            policies[name] = fit_controlled_policy_iteration(policy,
                traces["policy_training"], martingale_weights=(1., 1.), **settings)
    frozen = _freeze_json(Path(folder)/"selection_before_valuation.json", {
        "policy_fingerprint": fourier_policy_fingerprint(policy),
        "base_policy_seed": config.training_seed, "dual_training_seed": case["dual_seed"],
        "policy_improvement_seed": case["improvement_seed"],
        "reserved_valuation_seed": config.valuation_seed,
        "selected_control": selected, "selection": selection_rows,
        "fits": frozen_fits, "policy_variants": case["policy_variants"],
        "valuation_used_for_selection": False})
    # Every optimization and policy fit has completed before this independent
    # trace is generated or inspected.
    traces["valuation"] = value_fourier_stopping_policy(policy, trace=True)["trace"]
    controls = {"unit baseline": (1., 1.), **fits}
    raw = np.empty((config.valuation_paths, len(policies)))
    primal = np.empty((config.valuation_paths, len(policies), len(controls)))
    dual = np.empty((config.valuation_paths, len(controls)))
    stops = np.empty((config.valuation_paths, len(policies)), dtype=int)
    for p, (name, candidate) in enumerate(policies.items()):
        for c, (control, weights) in enumerate(controls.items()):
            value = evaluate_trace_policy(traces["valuation"], policy=candidate,
                martingale_weights=weights)
            if c == 0:
                raw[:, p], stops[:, p] = value.raw_samples[:, 0], value.stop_indices[:, 0]
            elif not (np.array_equal(raw[:, p], value.raw_samples[:, 0]) and
                    np.array_equal(stops[:, p], value.stop_indices[:, 0])):
                raise ValueError("Control attribution changed the frozen policy or raw cashflows")
            primal[:, p, c] = value.controlled_samples[:, 0]
            if p == 0:
                dual[:, c] = value.dual_samples[:, 0]
            elif not np.array_equal(dual[:, c], value.dual_samples[:, 0]):
                raise ValueError("Policy attribution changed the shared upper samples")
    tables = paired_attribution_tables(raw, primal, dual,
        policy_names=list(policies), control_names=list(controls))
    tables["matrix"]["selected_before_valuation"] = tables["matrix"].control.eq(selected)
    if not tables["policy_changes"].empty:
        fractions = {name: float(np.mean(stops[:, p] != stops[:, 0]))
            for p, name in enumerate(policies) if p}
        tables["policy_changes"]["exercise_decision_change_fraction"] = tables["policy_changes"].policy.map(fractions)
    tables["training_selection"] = pd.DataFrame(selection_rows)
    candidate_rows = []
    for name, fit in fits.items():
        for row in fit.diagnostics["candidates"]:
            candidate_rows.append({"control": name, **row,
                "selected_time_bins": row["candidate"] == fit.diagnostics["selected_candidate"],
                "selection_source": fit.diagnostics["selection_source"],
                "valuation_used_for_selection": False})
    tables["optimizer_candidates"] = pd.DataFrame(candidate_rows)
    reference = None
    if params.vol_of_vol == 0 and params.v0 == params.theta:
        from rvlab.pricing.rough_stopping_reference import black_scholes_stopping_reference
        result = black_scholes_stopping_reference(market, volatility=np.sqrt(params.v0),
            strike=contract["strike"], exercise_times=policy.exercise_times,
            is_call=contract["is_call"], spot_nodes=case["reference_nodes"],
            check_refinement=True)
        reference = {"price": result.price, "delta": result.delta,
            "settings": dict(result.settings), "diagnostics": dict(result.diagnostics)}
        tables["reference_partition"] = finite_reference_partition(tables["matrix"],
            reference_price=result.price,
            reference_change=result.diagnostics["reference_change_diagnostic"],
            reference_stable=result.diagnostics["price_stable"],
            reference_scope="Exact GBM transitions and the identical finite exercise calendar; no cash dividends")
    archives = [_save_sample_archive(Path(folder)/"paired_endpoints.npz",
        raw_payoffs=raw, controlled_primals=primal, dual_samples=dual, stop_indices=stops)]
    metadata = {}
    for name, trace in traces.items():
        archives.append(_save_sample_archive(Path(folder)/(name+"_trace.npz"),
            **{key: value for key, value in trace.items() if isinstance(value, np.ndarray)}))
        metadata[name] = {key: value for key, value in trace.items() if not isinstance(value, np.ndarray)}
    return {"status": "completed", "tables": {name: value.to_dict("records") for name, value in tables.items()},
        "policy_names": list(policies), "control_names": list(controls),
        "sample_archives": archives, "frozen_decisions": frozen,
        "trace_metadata": metadata, "reference": reference,
        "policy_fingerprint": fourier_policy_fingerprint(policy),
        "selected_control": selected, "config": asdict(config),
        "same_raw_payoffs_across_controls": True, "same_upper_samples_across_policies": True,
        "policy_variants_selected_on_valuation": False,
        "dual_fit_diagnostics": {name: dict(value.diagnostics) for name, value in fits.items()}}


def stopping_attribution_trial_specs():
    """Six targets and controls declared before independent formal valuation."""
    base = {"training_paths": 4096, "valuation_paths": 4096, "steps": 64,
        "exercise_steps": 32, "batch_size": 512, "memory_budget_mb": 1024.,
        "training_method": "fourier_premium", "premium_cap": .7}
    def case(name, contract="rough_put_60_100", **changes):
        config = {**base, **changes}
        count = config["training_paths"]
        common = {"training_paths": count, "candidate_bins": (1, 4),
            "validation_fraction": .5, "european_bounds": (1., 1.),
            "premium_bounds": (0., 2.), "objective": "gap",
            "selection_objective": "error_budget",
            "target_valuation_paths": config["valuation_paths"]}
        soft = {**common, "components": ("european", "premium", "intrinsic_minus_european"),
            "predictable_features": ("soft_exercise_region",),
            "intrinsic_bounds": (0., 0.), "intrinsic_feature_bounds": (0., 1.5),
            "european_feature_bounds": (0., 0.), "premium_feature_bounds": (0., 0.)}
        return {"case": name, "contract_id": contract, "factors": 4,
            "config": config, "dual_configs": {"static weights": common,
                "soft boundary intrinsic": soft},
            "policy_variants": {"positive policy refit": {"premium_floor": 0., "premium_cap": .7},
                "signed policy refit": {"premium_floor": -.7, "premium_cap": .7}},
            "improvement_paths": count, "reference_nodes": 32769}
    return [case("central rough"), case("independent rough repeat"),
        case("curve specification", "curve_put_60_100"),
        case("constant variance", "constant_variance_put"),
        case("larger path budgets", training_paths=8192, valuation_paths=8192),
        case("joint time and exercise", steps=128, exercise_steps=64)]


def stopping_attribution_probe_archives():
    """Exploratory provenance; its outcomes never select formal valuation seeds."""
    root = Path(__file__).resolve().parents[3]
    output = root/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    names = ("rough_dual_accuracy_2026-10-18", "rough_policy_attribution_2026-10-18", "rough_value_proxy_refit_2026-10-18",
        "rough_predictable_dual_2026-10-18", "rough_intrinsic_dual_2026-10-18",
        "rough_dual_budget_selection_2026-10-18")
    paths = tuple(output/name for name in names)
    for path in paths:
        if not (path/"preservation_manifest.json").is_file():
            raise FileNotFoundError(f"Missing stopping-attribution probe inventory: {path}")
    return paths


def _attribution_sources():
    import inspect
    from ._pricing_rough_american import _core_source_hashes, stopping_accuracy
    functions = (_validated_attribution_design, _select_control_from_training,
        _attribution_trial, paired_attribution_tables, finite_reference_partition,
        stopping_accuracy, _mean_se)
    return {**_core_source_hashes(), "study_numerics": hashlib.sha256("\n".join(
        inspect.getsource(function) for function in functions).encode()).hexdigest()}


def rough_stopping_attribution_study(context, *, seed=42, trial_specs=None,
        trial_indices=None, reuse_checkpoints=True, compute_figures=True,
        probe_archives=()):
    """Bounded shared-path attribution, with training-only optimizer selection."""
    from dataclasses import asdict
    from datetime import datetime, timezone
    from numbers import Integral
    import platform
    import scipy
    from rvlab.pricing import approximate_fractional_kernel, PricingError
    from ._pricing_rough_foundation import _checkpoint, _fingerprint, _jsonable, _market
    from ._pricing_rough_american import _american_models, _core_source_hashes, _preserve_sources
    from ._pricing_rough_american_accuracy import load_accuracy_probes
    for name, value in (("reuse_checkpoints", reuse_checkpoints), ("compute_figures", compute_figures)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    design = _validated_attribution_design(trial_specs, seed)
    if trial_indices is not None and (not isinstance(trial_indices, (tuple, list)) or
            any(isinstance(value, bool) or not isinstance(value, Integral) or
                value < 0 or value >= len(design) for value in trial_indices) or
            len(set(trial_indices)) != len(trial_indices)):
        raise ValueError("trial_indices must be distinct valid case indices")
    selected = set(range(len(design)) if trial_indices is None else trial_indices)
    probes, probe_records = load_accuracy_probes(probe_archives)
    models, market = _american_models(), _market()
    factors = sorted({row["factors"] for row in design})
    kernels = {n: approximate_fractional_kernel(.5 if n == 1 else .15, 1., n,
        **({} if n == 1 else {"min_time_ratio": .003 if n == 4 else .001, "max_nfev": 4000}))
        for n in factors}
    sources = _attribution_sources()
    declared = [{**row, "config": asdict(row["config"]),
        "dual_configs": {name: asdict(value) for name, value in row["dual_configs"].items()}}
        for row in design]
    common = {"study_version": "rough_stopping_attribution_v1", "seed": int(seed),
        "design": declared, "models": {name: asdict(value) for name, value in models.items()},
        "market": asdict(market), "kernels": {str(n): asdict(value) for n, value in kernels.items()},
        "sources": sources, "preserved_probes": probe_records,
        "historical_accuracy_profile": HISTORICAL_ACCURACY_PROFILE,
        "acceptance": {"price_atol": .001, "price_rtol": 1e-4,
            "delta_atol": .002, "delta_rtol": .005, "ci_multiplier": 1.96,
            "effective_half_spread_fraction": .1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = _fingerprint(common)
    folder = Path(context.output_dir)/"rough_stopping_attribution_v1"/fingerprint[:16]
    if not reuse_checkpoints:
        folder /= "attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots = _preserve_sources(folder, sources)
    payload = Path(__file__).read_bytes()
    destination = folder/"source"/hashlib.sha256(payload).hexdigest()/Path(__file__).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError("Changed preserved stopping-attribution study source")
    if not destination.exists():
        destination.write_bytes(payload)
    snapshots.append({"source": Path(__file__).name, "sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot": str(destination), "bytes": len(payload)})
    provenance, failures, trials, collected, summaries, references = [], [], [], {}, [], []
    for index, case in enumerate(design):
        if index not in selected:
            continue
        context.progress(f"Stopping attribution {index+1}/{len(design)}: {case['case']}")
        def calculate(case=case, index=index):
            expected = {key: value for key, value in sources.items() if key != "study_numerics"}
            before = _core_source_hashes()
            if before != expected:
                raise PricingError("Numerical source changed before stopping attribution")
            try:
                result = _attribution_trial(case, models[case["contract"]["model"]], market,
                    kernels[case["factors"]], folder/(f"case_{index:02d}"))
            except MemoryError as error:
                result = {"status": "numerical_failure", "error": str(error)}
            after = _core_source_hashes()
            if before != after:
                raise PricingError("Numerical source changed during stopping attribution")
            result.update(source_after=after, source_changed_during_trial=False)
            return result
        record, origin = _checkpoint(folder, f"case_{index:02d}", {**common, "case_index": index},
            calculate, reuse=reuse_checkpoints)
        _verify_sample_archives(record)
        provenance.append(origin)
        trial = {"case": case["case"], "case_index": index,
            "contract_id": case["contract"]["contract_id"], "factors": case["factors"],
            "status": record["status"], "policy_training_seed": case["config"].training_seed,
            "dual_training_seed": case["dual_seed"], "policy_improvement_seed": case["improvement_seed"],
            "valuation_seed": case["config"].valuation_seed,
            "training_paths": case["config"].training_paths,
            "valuation_paths": case["config"].valuation_paths,
            "steps": case["config"].steps, "exercise_steps": case["config"].exercise_steps,
            "elapsed_seconds": origin["original_elapsed_seconds"], "reused": origin["reused"],
            "checkpoint": origin["checkpoint"],
            "source_changed_during_trial": record.get("source_changed_during_trial")}
        trials.append(trial)
        if record["status"] != "completed":
            failures.append({"case": case["case"], "status": record["status"], "error": record.get("error")})
            summaries.append({"case": case["case"], "status": record["status"],
                "american_pricing_ready": False, "stock_delta_ready": False})
            continue
        for name, values in record["tables"].items():
            table = pd.DataFrame(values)
            table.insert(0, "case", case["case"])
            collected.setdefault(name, []).append(table)
        matrix = pd.DataFrame(record["tables"]["matrix"])
        unit = matrix.loc[matrix.policy.eq("base policy") & matrix.control.eq("unit baseline")].iloc[0]
        chosen = matrix.loc[matrix.policy.eq("base policy") & matrix.control.eq(record["selected_control"])].iloc[0]
        summaries.append({"case": case["case"], "status": "completed",
            "training_selected_control": record["selected_control"],
            "original_gap": unit.paired_gap, "selected_control_gap": chosen.paired_gap,
            "selected_price_standard_error": chosen.price_standard_error,
            "selected_price_gate_passed": chosen.discrete_price_gate_passed,
            "completed_comparisons": len(matrix), "price_gate_passes": int(matrix.discrete_price_gate_passed.sum()),
            "american_pricing_ready": False, "stock_delta_ready": False})
        for row in record["tables"].get("optimizer_candidates", []):
            if row.get("status") in ("numerical_failure", "optimizer_failure"):
                failures.append({"case": case["case"], "control": row.get("control"),
                    "candidate": row.get("candidate"), "status": row["status"],
                    "error": row.get("error", row.get("message"))})
        if record.get("reference"):
            references.append({"case": case["case"], **record["reference"]})
    tables = {name: pd.concat(values, ignore_index=True, sort=False) for name, values in collected.items()}
    for name in ("matrix", "policy_changes", "control_changes", "training_selection",
                 "optimizer_candidates", "reference_partition"):
        tables.setdefault(name, pd.DataFrame())
    tables.update(summary=pd.DataFrame(summaries), trials=pd.DataFrame(trials),
        provenance=pd.DataFrame(provenance), failures=pd.DataFrame(failures),
        references=pd.DataFrame(references), probes=probes)
    manifest = {**_jsonable(common), "backend": "rough_heston_stopping_attribution",
        "profile_fingerprint": fingerprint, "checkpoint_dir": str(folder), "source_snapshots": snapshots,
        "source_elapsed_seconds": sum(row["original_elapsed_seconds"] for row in provenance),
        "checkpoint_replay": bool(provenance) and all(row["reused"] for row in provenance),
        "planned_case_count": len(design), "selected_case_indices": sorted(selected),
        "partial_campaign": len(selected) != len(design), "current_engine_execution": True,
        "calibration_optimizer_calls": 0,
        "planned_dual_optimizer_calls": int(sum(len(case["dual_configs"][name].candidate_bins)
            for index, case in enumerate(design) if index in selected for name in case["dual_configs"])),
        "recorded_dual_optimizer_calls": int(sum(not row.get("candidate", "").startswith("unit")
            for row in tables["optimizer_candidates"].to_dict("records"))),
        "optimizer_counts_incomplete_on_failed_cases": any(row["status"] != "completed" for row in trials),
        "four_independent_streams": True, "valuation_used_for_selection": False,
        "american_pricing_ready": False, "stock_delta_ready": False,
        "timing_interpretation": "Original checkpoint wall runtimes include concurrent contention; campaign wall and cached replay are separate costs.",
        "limitations": [
            "Attribution is conditional on the fixed finite kernel, transition law and exercise calendar.",
            "Rough policy/control swaps identify removable components among tested alternatives, not absolute optimal-policy bias.",
            "Only the no-cash constant-variance limit has an independent same-calendar value reference.",
            "Reference spatial/domain changes are empirical numerical diagnostics, not rigorous error bounds.",
            "Price endpoint bounds do not certify optimal-value stock deltas; this scalar study computes no new deltas.",
            "All price/delta/spread tolerances are unchanged; synthetic inception studies do not resolve SPY inputs or historical states."]}
    return {"tables": tables, "figures": _attribution_figures(tables) if compute_figures else {},
        "manifest": manifest}


def add_attribution_extras(result):
    """Attach independently preserved exploratory evidence, outside the profile."""
    from ._pricing_rough_american_accuracy import load_accuracy_probes
    path = (Path(__file__).resolve().parents[3]/
        "notebooks/_output/pricing/research_v1/01_pricing_and_exercise/rough_bs_premium_proxy_2026-10-18")
    tables, records = load_accuracy_probes((path,))
    result["tables"]["supplemental_probes"] = tables
    result["manifest"]["experimental_extras"] = [{**row,
        "formal_campaign_input": False,
        "provenance_limit": "Exploratory legacy cache identity excludes only newly added dual_config=None; initial identity failure is preserved.",
        "finding": "Same-policy BS-premium proxy gives about10% paired gap reduction; direct upper change is statistically inconclusive."}
        for row in records]
    return result


def add_attribution_execution(result, *, output_dir):
    """Audit actual worker timing separately from checkpoint and replay cost."""
    from datetime import datetime
    profile = result["manifest"].get("profile_fingerprint")
    folder = Path(output_dir)/"rough_stopping_attribution_v1/campaign_v1"
    records, starts, ends, case_counts = [], [], [], {}
    if folder.is_dir():
        for path in sorted(folder.rglob("manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest.get("profile_fingerprint") != profile or manifest.get("mode") != "worker":
                continue
            runner = path.parent/"executed_runner.py"
            digest = hashlib.sha256(runner.read_bytes()).hexdigest()
            if digest != manifest.get("runner_sha256"):
                raise ValueError(f"Changed attribution runner evidence: {runner}")
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
