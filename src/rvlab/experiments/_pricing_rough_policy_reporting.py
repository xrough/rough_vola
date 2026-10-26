"""Display-only provenance and timing for the frozen policy campaign."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np


def add_policy_execution(result, *, output_dir):
    """Keep worker span, summed wall time and notebook replay distinct."""
    manifest = dict(result["manifest"])
    rows = []
    root = Path(output_dir)/"rough_policy_improvement_v1/campaign_v1"
    for path in sorted(root.glob("*/manifest.json")):
        record = json.loads(path.read_text())
        if record.get("mode") != "worker" or record.get("profile_fingerprint") != manifest["profile_fingerprint"]:
            continue
        runner = path.with_name("executed_runner.py")
        if hashlib.sha256(runner.read_bytes()).hexdigest() != record["runner_sha256"]:
            raise ValueError(f"Changed executed policy runner: {runner}")
        rows.append({"worker": path.parent.name, "started_at": record["started_at"],
            "completed_at": record["completed_at"], "worker_wall_seconds": record["elapsed_seconds"],
            "runner_sha256": record["runner_sha256"], "manifest": str(path.resolve())})
    tables = dict(result["tables"])
    tables["execution_workers"] = pd.DataFrame(rows)
    if rows:
        start = min(datetime.fromisoformat(row["started_at"]) for row in rows)
        end = max(datetime.fromisoformat(row["completed_at"]) for row in rows)
        manifest["campaign_worker_span_seconds"] = (end-start).total_seconds()
        manifest["worker_manifest_count"] = len(rows)
        manifest["worker_wall_seconds_sum"] = sum(row["worker_wall_seconds"] for row in rows)
    manifest["timing_interpretation"] = "Worker span and summed concurrent checkpoint wall times are different costs; cached notebook replay is recorded separately. Neither sum is measured serial runtime or CPU time."
    return {**result, "tables": tables, "manifest": manifest}


def add_policy_timing_figure(result):
    """Describe stopping times from sealed final paths, without selecting rules."""
    import matplotlib.pyplot as plt
    from ._pricing_rough_stopping_attribution import _verify_sample_archives
    trials = result["tables"].get("trials", pd.DataFrame())
    selected = trials.loc[trials.case.isin(("central rough", "constant variance")) & trials.status.eq("completed")] if not trials.empty else trials
    if selected.empty:
        return result
    fig, axes = plt.subplots(1, len(selected), figsize=(11.5, 4.2), squeeze=False)
    for ax, row in zip(axes[0], selected.itertuples()):
        record = json.loads(Path(row.checkpoint).read_text())["payload"]
        _verify_sample_archives(record)
        endpoints = next(a for a in record["sample_archives"] if Path(a["path"]).name == "paired_policy_endpoints.npz")
        trace = next(a for a in record["sample_archives"] if Path(a["path"]).name == "final_valuation_trace.npz")
        with np.load(endpoints["path"], allow_pickle=False) as arrays:
            stops = arrays["stop_indices"]
        with np.load(trace["path"], allow_pickle=False) as arrays:
            times = arrays["times"]
        names = ["original", record["selected_policy"]]
        if "independent BS reference policy" in record["policy_names"]:
            names.append("independent BS reference policy")
        aliases = {"original": "Original policy", record["selected_policy"]: "Training-selected policy", "independent BS reference policy": "Independent BS reference"}
        for name in dict.fromkeys(names):
            values = stops[:, record["policy_names"].index(name)]
            fraction = np.mean(values[:, None] <= np.arange(len(times))[None, :], axis=0)
            ax.step(365*times, fraction, where="post", label=aliases[name])
        ax.set(title="Rough Heston" if row.case == "central rough" else "Constant-variance limit",
            xlabel="Days from inception", ylabel="Fraction stopped or expired", ylim=(-.02, 1.02))
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return {**result, "figures": {**result["figures"], "exercise_timing": fig}}


def add_policy_reference_pairs(result):
    """Describe paired differences to the independent approximate BS rule.

    This reads sealed endpoint samples; it never reprices, changes a stopping
    rule, selects a candidate or changes an acceptance decision. The intervals
    describe Monte Carlo uncertainty for the *approximate DP rule*. The
    independently measured mesh change is displayed separately and is not a
    rigorous bound on that rule's distance from the optimal stopping value.
    """
    from ._pricing_rough_stopping_attribution import _verify_sample_archives
    trials = result["tables"].get("trials", pd.DataFrame())
    if trials.empty or not {"case", "status", "checkpoint"}.issubset(trials.columns):
        return result
    completed = trials.loc[trials.case.eq("constant variance") & trials.status.eq("completed")]
    if completed.empty:
        return result
    reference_name = "independent BS reference policy"
    scope = ("Paired differences to an independent approximate same-calendar BS DP decision rule; "
        "not an exact optimal-value bias decomposition. Mesh change is an empirical diagnostic.")
    rows = []
    for trial in completed.itertuples():
        checkpoint = Path(trial.checkpoint)
        record = json.loads(checkpoint.read_text())["payload"]
        if record.get("status") != "completed":
            raise ValueError("Completed BS trial points to an incomplete checkpoint")
        if not record.get("reference") or reference_name not in record.get("policy_names", ()):
            continue
        if record.get("same_upper_samples") is not True:
            raise ValueError("Reference pairs require an identical frozen upper estimator")
        _verify_sample_archives(record)
        archives = [a for a in record["sample_archives"]
            if Path(a["path"]).name == "paired_policy_endpoints.npz"]
        if len(archives) != 1:
            raise ValueError("Reference pairs require exactly one sealed endpoint archive")
        with np.load(archives[0]["path"], allow_pickle=False) as arrays:
            lower = np.asarray(arrays["controlled_primals"], float)
            upper = np.asarray(arrays["dual_samples"], float)
        names = record["policy_names"]
        if (lower.ndim != 2 or len(lower) < 2 or lower.shape[1] != len(names)
                or upper.shape != (len(lower),) or len(set(names)) != len(names)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or "original" not in names or record.get("selected_policy") not in names):
            raise ValueError("Reference pairs require finite aligned policy and common-upper samples")
        reference_lower = lower[:, names.index(reference_name)]
        mesh_change = float(record["reference"]["diagnostics"]["reference_change_diagnostic"])
        if not np.isfinite(mesh_change) or mesh_change < 0:
            raise ValueError("Reference mesh change must be finite and nonnegative")
        for name in dict.fromkeys(("original", record["selected_policy"])):
            gain = reference_lower-lower[:, names.index(name)]
            gap = upper-reference_lower
            for quantity, samples in (("dp_minus_policy", gain),
                    ("upper_minus_dp", gap), ("upper_gap_minus_policy_gain", gap-gain)):
                estimate = float(np.mean(samples))
                se = float(np.std(samples, ddof=1)/np.sqrt(len(samples)))
                rows.append({"case": trial.case, "policy": name,
                    "reference_policy": reference_name, "quantity": quantity,
                    "estimate": estimate, "standard_error": se,
                    "confidence_low": estimate-1.96*se, "confidence_high": estimate+1.96*se,
                    "paths": len(samples), "confidence_multiplier": 1.96,
                    "reference_mesh_change": mesh_change, "approximate_dp_rule": True,
                    "reference_change_is_empirical": True, "acceptance_input": False,
                    "american_pricing_ready": False, "stock_delta_ready": False,
                    "same_upper_samples": True, "valuation_used_for_selection": False,
                    "checkpoint": str(checkpoint), "scope": scope})
    if not rows:
        return result
    tables, manifest = dict(result["tables"]), dict(result["manifest"])
    tables["reference_policy_pairs"] = pd.DataFrame(rows)
    manifest["reference_policy_pair_diagnostics"] = {
        "display_only": True, "acceptance_input": False, "quantities":
            ["dp_minus_policy", "upper_minus_dp", "upper_gap_minus_policy_gain"],
        "confidence_multiplier": 1.96, "mesh_change_included_in_intervals": False,
        "interval_scope": "Paired Monte Carlo uncertainty for the approximate DP decision rule",
        "reference_change_is_empirical": True, "optimal_bias_decomposition": False,
        "scope": scope, "reporting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {**result, "tables": tables, "manifest": manifest}


def add_policy_control_neutrality(result):
    """Retain selected-policy raw/control differences as marginal diagnostics.

    A centered adapted martingale leaves expected stopping-policy gains
    unchanged. This diagnostic measures the *paired* raw gain minus controlled
    gain on the original final paths. A pointwise interval excluding zero is
    neither proof of estimator bias nor confirmation of sampling noise. No new
    paths, selection, acceptance decisions or tolerance changes follow here.
    """
    from ._pricing_rough_stopping_attribution import _verify_sample_archives
    trials = result["tables"].get("trials", pd.DataFrame())
    if trials.empty or not {"case", "status", "checkpoint"}.issubset(trials.columns):
        return result
    completed = trials.loc[trials.status.eq("completed")]
    if completed.empty:
        return result
    scope = ("Pointwise paired raw-gain minus controlled-gain diagnostic on the original sealed final paths; "
        "neither proof of estimator bias nor confirmation of sampling noise. Multiple case comparisons are not simultaneous intervals.")
    rows = []
    for trial in completed.itertuples():
        checkpoint = Path(trial.checkpoint)
        record = json.loads(checkpoint.read_text())["payload"]
        if record.get("status") != "completed":
            raise ValueError("Completed policy trial points to an incomplete checkpoint")
        if record.get("same_upper_samples") is not True:
            raise ValueError("Control neutrality requires the identical frozen martingale comparison")
        _verify_sample_archives(record)
        archives = [a for a in record.get("sample_archives", ())
            if Path(a["path"]).name == "paired_policy_endpoints.npz"]
        if len(archives) != 1:
            raise ValueError("Control neutrality requires exactly one sealed endpoint archive")
        with np.load(archives[0]["path"], allow_pickle=False) as arrays:
            raw = np.asarray(arrays["raw_payoffs"], float)
            controlled = np.asarray(arrays["controlled_primals"], float)
        names = record.get("policy_names", ())
        selected = record.get("selected_policy")
        if (raw.ndim != 2 or len(raw) < 2 or controlled.shape != raw.shape
                or raw.shape[1] != len(names) or len(set(names)) != len(names)
                or "original" not in names or selected not in names
                or not np.isfinite(raw).all() or not np.isfinite(controlled).all()):
            raise ValueError("Control neutrality requires finite aligned raw and controlled policy samples")
        original_index, selected_index = names.index("original"), names.index(selected)
        raw_gain = raw[:, selected_index]-raw[:, original_index]
        controlled_gain = controlled[:, selected_index]-controlled[:, original_index]
        contrast = raw_gain-controlled_gain
        estimate = float(np.mean(contrast))
        se = float(np.std(contrast, ddof=1)/np.sqrt(len(contrast)))
        low, high = estimate-1.96*se, estimate+1.96*se
        z_score = estimate/se if se > 0 else (0. if estimate == 0 else None)
        rows.append({"case": trial.case, "selected_policy": selected,
            "selected_control": record.get("selected_control"), "paths": len(raw),
            "raw_gain": float(np.mean(raw_gain)),
            "raw_gain_standard_error": float(np.std(raw_gain, ddof=1)/np.sqrt(len(raw))),
            "controlled_gain": float(np.mean(controlled_gain)),
            "controlled_gain_standard_error": float(np.std(controlled_gain, ddof=1)/np.sqrt(len(raw))),
            "raw_minus_controlled_gain": estimate,
            "raw_minus_controlled_standard_error": se,
            "confidence_low": low, "confidence_high": high,
            "z_score": z_score, "zero_consistent": bool(low <= 0 <= high),
            "confidence_multiplier": 1.96, "marginal_inference_only": True,
            "bias_identified": False, "sampling_noise_confirmed": False,
            "acceptance_input": False, "same_upper_samples": True,
            "valuation_used_for_selection": False,
            "american_pricing_ready": False, "stock_delta_ready": False,
            "checkpoint": str(checkpoint), "scope": scope})
    tables, manifest = dict(result["tables"]), dict(result["manifest"])
    table = pd.DataFrame(rows)
    tables["control_neutrality"] = table
    manifest["policy_control_neutrality"] = {
        "display_only": True, "acceptance_input": False, "comparisons": len(table),
        "pointwise_intervals_excluding_zero": int((~table.zero_consistent).sum()),
        "confidence_multiplier": 1.96, "marginal_inference_only": True,
        "bias_identified": False, "sampling_noise_confirmed": False,
        "new_paths_generated": False, "selection_changed": False,
        "scope": scope, "reporting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {**result, "tables": tables, "manifest": manifest}
