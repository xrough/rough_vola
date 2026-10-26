"""Read-only figures for the independent raw/control centering investigation."""
from __future__ import annotations

import numpy as np


SUPPLEMENT_INVENTORY_SHA256 = "33979547aee6c2945b9ad7d9d12164bd6bc9b2ba778ac763f7fbc1e4cdbf2249"


def add_centering_supplement(result, *, output_dir):
    """Read a sealed, late local audit without changing primary inference."""
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd

    manifest, tables = dict(result["manifest"]), dict(result["tables"])
    profile = manifest["profile_fingerprint"]
    directory = (Path(output_dir)/"rough_control_centering_v1"/profile[:16]/
        "supplemental_disagreement_states_v1")
    inventory = json.loads((directory/"preservation_inventory.json").read_text())
    digest = hashlib.sha256(json.dumps(inventory["files"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if digest != inventory.get("inventory_sha256") or digest != SUPPLEMENT_INVENTORY_SHA256:
        raise ValueError("Changed supplementary disagreement-audit inventory")
    for record in inventory["files"]:
        path = (directory/record["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Supplementary inventory path escapes its sealed directory")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed supplementary disagreement-audit evidence: {path}")
    checked = json.loads((directory/"result.json").read_text())
    seal_path = directory/"selection_before_conditional_residuals.json"
    seal_data = seal_path.read_bytes()
    seal_digest = hashlib.sha256(seal_data).hexdigest()
    selection = json.loads(seal_data)
    if (checked.get("status") != "completed" or checked.get("late_exploratory") is not True
            or checked.get("primary_design_input") is not False
            or checked.get("primary_pool_input") is not False
            or checked.get("new_paths_generated") is not False
            or checked.get("tolerances_changed") is not False
            or selection.get("primary_profile") != profile[:16]
            or selection.get("selection_uses_conditional_residuals") is not False
            or selection.get("selection_uses_frozen_decisions_only") is not True
            or selection.get("policy_or_control_selection") is not False
            or checked.get("selection_sha256") != seal_digest):
        raise ValueError("Supplementary audit changed its declared exploratory scope or selection seal")
    source_trace = selection["source_trace"]
    trace_data = Path(source_trace["path"]).read_bytes()
    if (len(trace_data) != source_trace["bytes"] or
            hashlib.sha256(trace_data).hexdigest() != source_trace["sha256"]):
        raise ValueError("Changed existing independent trace used by supplementary audit")
    for name, rows in checked["tables"].items():
        table = pd.DataFrame(rows)
        table["late_exploratory"] = True
        table["primary_pool_input"] = False
        table["acceptance_input"] = False
        tables["supplementary_conditional_"+name] = table
    description = {"directory": str(directory.resolve()), "files": len(inventory["files"]),
        "inventory_sha256": digest, "selection_seal_sha256": seal_digest,
        "source_trace_sha256": source_trace["sha256"],
        "states": len(checked["tables"]["states"]),
        "branches": len(checked["tables"]["branches"]),
        "components": len(checked["tables"]["components"]),
        "late_exploratory": True, "formal_campaign_input": False,
        "primary_pool_input": False, "acceptance_input": False,
        "new_paths_generated": False, "valuation_used_for_selection": False,
        "selection_rule": selection["selection_rule"],
        "elapsed_seconds": checked["elapsed_seconds"]}
    tables["supplementary_inventory"] = pd.DataFrame([description])
    manifest["supplementary_disagreement_audit"] = description
    return {**result, "tables": tables, "manifest": manifest}


def add_centering_execution(result, *, output_dir):
    """Verify frozen worker runners and describe their wall-clock costs."""
    from datetime import datetime
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd

    manifest, tables = dict(result["manifest"]), dict(result["tables"])
    reconstruction = tables.get("reconstruction")
    if reconstruction is not None and len(reconstruction) == 1:
        rows = reconstruction.iloc[0].tolist()
        if rows and all(isinstance(row, dict) and
                {"policy", "quantity", "bit_identical", "max_absolute_change"}.issubset(row)
                for row in rows):
            tables["reconstruction"] = pd.DataFrame(rows)
    directory = Path(output_dir)/"rough_control_centering_v1/campaign_v1"
    workers, stages = [], []
    for path in sorted(directory.glob("*/manifest.json")):
        record = json.loads(path.read_text())
        if record.get("profile_fingerprint") != manifest["profile_fingerprint"]:
            continue
        runner = path.with_name("executed_runner.py")
        if hashlib.sha256(runner.read_bytes()).hexdigest() != record.get("runner_sha256"):
            raise ValueError(f"Changed executed centering runner: {runner}")
        row = {"stage": path.parent.name, "mode": record.get("mode"),
            "started_at": record["started_at"], "completed_at": record["completed_at"],
            "wall_seconds": record["elapsed_seconds"],
            "runner_sha256": record["runner_sha256"], "manifest": str(path.resolve())}
        stages.append(row)
        if record.get("mode") == "worker":
            workers.append(row)
    tables["execution_stages"] = pd.DataFrame(stages)
    tables["execution_workers"] = pd.DataFrame(workers)
    if workers:
        start = min(datetime.fromisoformat(row["started_at"]) for row in workers)
        end = max(datetime.fromisoformat(row["completed_at"]) for row in workers)
        manifest["campaign_worker_span_seconds"] = (end-start).total_seconds()
        manifest["worker_wall_seconds_sum"] = sum(row["wall_seconds"] for row in workers)
        manifest["worker_manifest_count"] = len(workers)
    manifest["timing_interpretation"] = (
        "Cold worker span, summed concurrent worker wall times, frozen-bundle "
        "reconstruction, conditional-audit preparation, cached collection and fresh-kernel "
        "notebook replay are separate costs. Summed wall times include contention and "
        "are neither CPU time nor measured serial runtime.")
    return {**result, "tables": tables, "manifest": manifest}


def add_centering_figures(result):
    """Plot sealed diagnostics; never select streams, policies or tolerances."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    tables, figures = result["tables"], dict(result.get("figures", {}))
    per_seed, pooled = tables.get("per_seed"), tables.get("pooled")
    if (per_seed is not None and pooled is not None and
            not per_seed.empty and not pooled.empty and
            {"quantity", "seed", "estimate", "standard_error"}.issubset(per_seed.columns)):
        rows = per_seed.loc[per_seed.quantity.eq("contrast")].copy()
        primary = pooled.loc[pooled.quantity.eq("contrast")]
        if not rows.empty and len(primary) == 1:
            # Keep the declared seed order; do not sort by an observed outcome.
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True,
                gridspec_kw={"width_ratios": (1.35, 1.)})
            ax = axes[0]
            y = np.arange(len(rows))
            ax.errorbar(rows.estimate, y, xerr=1.96*rows.standard_error,
                fmt="o", capsize=3, color="#3465a4", label="Fresh independent stream")
            ax.axvline(0., color="#555555", linewidth=1.)
            ax.set(yticks=y, yticklabels=[f"{int(seed):,}" for seed in rows.seed],
                xlabel="Raw gain minus controlled gain ($)",
                title="All predeclared fresh streams")
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(MaxNLocator(5))
            row = primary.iloc[0]
            axes[1].errorbar([row.estimate], [0],
                xerr=[1.96*row.standard_error], fmt="o", capsize=4,
                color="#238443", label="Fresh-only pool")
            labels = ["Fresh-only pool"]
            trigger = tables.get("exploratory_trigger")
            if trigger is not None and not trigger.empty:
                old = trigger.iloc[0]
                mean = old.get("estimate", old.get("raw_minus_controlled_gain"))
                se = old.get("standard_error", old.get("raw_minus_controlled_standard_error"))
                if mean is not None and se is not None and np.isfinite([mean, se]).all():
                    axes[1].errorbar([mean], [1], xerr=[1.96*se], fmt="D",
                        capsize=4, color="#a6611a", label="Historical trigger (excluded)")
                    labels.append("Historical trigger\nexcluded from pool")
            axes[1].axvline(0., color="#555555", linewidth=1.)
            axes[1].set(yticks=range(len(labels)), yticklabels=labels,
                xlabel="Raw gain minus controlled gain ($)",
                title="Keep the trigger separate")
            axes[1].invert_yaxis()
            axes[1].set_ylim(len(labels)-.5, -.5)
            axes[1].xaxis.set_major_locator(MaxNLocator(4))
            fig.suptitle("Fixed stopping policies and a fixed martingale on independent paths", y=1.02)
            fig.tight_layout()
            figures["independent_repeats"] = fig

    conditional = tables.get("conditional_components")
    if (conditional is not None and not conditional.empty and
            {"state_id", "component", "conditional_increment_mean",
             "quadrature_error_estimate"}.issubset(conditional.columns)):
        groups = list(conditional.groupby("component", sort=False))
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1),
            gridspec_kw={"width_ratios": (1.2, 1.)})
        palette = plt.get_cmap("tab10")
        for index, (name, frame) in enumerate(groups):
            axes[0].plot(frame.state_id.astype(str),
                frame.conditional_increment_mean, "o", markersize=4,
                label=str(name).replace("_", " "), color=palette(index % 10))
        axes[0].axhline(0., color="#555555", linewidth=1.)
        axes[0].set(xlabel="Predeclared fresh left state", ylabel="Conditional increment mean ($)",
            title="Independent scalar Gaussian integration")
        axes[0].tick_params(axis="x", labelrotation=45, labelsize=8)
        axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axes[0].legend(frameon=False, fontsize=8)
        labels = [str(name).replace("_", " ") for name, _ in groups]
        residuals = [float(frame.conditional_increment_mean.abs().max()) for _, frame in groups]
        reported = [float(frame.quadrature_error_estimate.abs().max()) for _, frame in groups]
        y = np.arange(len(groups))
        # Use a positive display floor only for a logarithmic axis. Original
        # signed values, including exact zero, remain in the evidence table.
        positive = np.asarray(residuals+reported)
        positive = positive[positive > 0]
        floor = float(positive.min()/10.) if len(positive) else 1e-16
        axes[1].scatter(np.maximum(residuals, floor), y, marker="o", color="#3465a4",
            label="Largest absolute conditional mean")
        axes[1].scatter(np.maximum(reported, floor), y, marker="x", color="#a6611a",
            label="Largest reported integration error")
        axes[1].set(xscale="log", yticks=y, yticklabels=labels,
            xlabel="Dollars, logarithmic scale", title="Numerical residual scale")
        axes[1].invert_yaxis()
        axes[1].legend(frameon=False, fontsize=8, loc="upper center",
            bbox_to_anchor=(.5, -.21))
        fig.tight_layout(rect=(0., .07, 1., 1.))
        figures["conditional_residuals"] = fig
    return {**result, "figures": figures}


__all__ = ["add_centering_figures", "add_centering_execution", "add_centering_supplement"]
