"""Read-only presentation of the independent conditional-centering campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


_BUDGET_MANIFEST_SHA = "4aa090d558ac88e732a0a5a35b1d6eea929042fb58325d97ae3e9c6bf51d3f04"
_SMOKE_FILES = {
    "manifest.json": (606, "dc50e642823bc734f4b57d76128904a0f202fb6321c910fe389d365d25b3508c"),
    "gated_states.csv": (1319, "cbf8f0d8a1d94a02198001e77032915405b9a1585f9d01ea8bfe68d7a725370a"),
}


def _verified(path, *, size=None, digest):
    data = Path(path).read_bytes()
    if (size is not None and len(data) != size) or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Changed precision-study display evidence: {path}")
    return data


def _pilot_tables(output_dir):
    """Read excluded development diagnostics; never feed the primary estimator."""
    directory = Path(output_dir) / "rough_centering_precision_v1/pilots"
    tables, descriptions = {}, []
    budget = directory / "quadrature_budget"
    if budget.exists():
        raw = _verified(budget / "manifest.json", digest=_BUDGET_MANIFEST_SHA)
        manifest = json.loads(raw)
        if (manifest.get("formal_new_precision_campaign_input") is not False or
                manifest.get("thresholds_changed") is not False or
                manifest.get("no_new_paths") is not True):
            raise ValueError("Quadrature pilot changed its excluded historical scope")
        for name, row in manifest["files"].items():
            path = (budget / name).resolve()
            if not path.is_relative_to(budget.resolve()):
                raise ValueError("Pilot file escapes its declared directory")
            _verified(path, size=row["bytes"], digest=row["sha256"])
        tables["pilot_budget_trials"] = pd.read_csv(budget / "budget_trials.csv", float_precision="round_trip")
        tables["pilot_tail_robustness"] = pd.read_csv(budget / "tail_robustness.csv", float_precision="round_trip")
        descriptions.append({"trial": "conditional quadrature budget", "seed": manifest["trace_seed"],
            "primary_pool_input": False, "acceptance_input": False,
            "already_spent_paths": True, "new_paths": False,
            "manifest_sha256": _BUDGET_MANIFEST_SHA,
            "files_verified": len(manifest["files"]) + 1})
    smoke = directory / "microstate_verification"
    if smoke.exists():
        for name, (size, digest) in _SMOKE_FILES.items():
            _verified(smoke / name, size=size, digest=digest)
        manifest = json.loads((smoke / "manifest.json").read_text())
        if (manifest.get("excluded_from_primary_inference") is not True or
                manifest.get("no_new_exercise_dates") is not True):
            raise ValueError("Microstate verification changed its excluded diagnostic scope")
        tables["pilot_microstate_verification"] = pd.DataFrame([manifest])
        tables["pilot_microstate_states"] = pd.read_csv(smoke / "gated_states.csv", float_precision="round_trip")
        descriptions.append({"trial": "microstate and realized-contrast identity", "seed": manifest["seed"],
            "primary_pool_input": False, "acceptance_input": False,
            "already_spent_paths": False, "new_paths": True,
            "manifest_sha256": _SMOKE_FILES["manifest.json"][1],
            "files_verified": len(_SMOKE_FILES)})
    if descriptions:
        tables["excluded_trials"] = pd.DataFrame(descriptions)
    return tables


def _execution_tables(output_dir, profile):
    """Separate measured worker span, concurrent wall time and cached replay."""
    from datetime import datetime

    directory = Path(output_dir) / "rough_centering_precision_v1/campaign_v1"
    rows = []
    for path in sorted(directory.rglob("manifest.json")) if profile else ():
        record = json.loads(path.read_text())
        if record.get("profile_fingerprint") != profile or record.get("mode") not in {"prepare", "worker", "collect"}:
            continue
        _verified(path.parent / "executed_runner.py", digest=record["runner_sha256"])
        rows.append({"stage": record["mode"], "run": str(path.parent.relative_to(directory)),
            "started_at": record["started_at"], "completed_at": record["completed_at"],
            "elapsed_seconds": record["elapsed_seconds"], "primary_complete": record.get("primary_complete", False),
            "checkpoint_replay": record.get("checkpoint_replay", False),
            "executed_runner_sha256": record["runner_sha256"]})
    descriptions = {}
    workers = [row for row in rows if row["stage"] == "worker" and not row["checkpoint_replay"]]
    if workers:
        descriptions["campaign_worker_span_seconds"] = (max(datetime.fromisoformat(row["completed_at"]) for row in workers) -
            min(datetime.fromisoformat(row["started_at"]) for row in workers)).total_seconds()
        descriptions["worker_wall_seconds_sum"] = sum(row["elapsed_seconds"] for row in workers)
        descriptions["worker_manifest_count"] = len(workers)
    descriptions["timing_interpretation"] = ("Worker span and summed concurrent wall times include contention; "
        "neither is CPU time or measured serial runtime. Cached collection and fresh-kernel notebook replay are separate.")
    return ({"execution_stages": pd.DataFrame(rows)} if rows else {}), descriptions


def add_precision_reporting(result, *, output_dir):
    """Attach verified excluded trials and figures without changing primary data."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    tables = dict(result["tables"])
    tables.update(_pilot_tables(output_dir))
    manifest = dict(result["manifest"])
    execution_tables, execution = _execution_tables(output_dir, manifest.get("profile_fingerprint"))
    tables.update(execution_tables)
    manifest.update(execution)
    if "excluded_trials" in tables:
        manifest["display_only_excluded_trials"] = tables["excluded_trials"].to_dict("records")
    figures = dict(result.get("figures", {}))
    pooled, budgets = tables.get("pooled"), tables.get("numerical_budget")
    if (pooled is not None and budgets is not None and not pooled.empty and not budgets.empty
            and {"quantity", "estimate", "standard_error"}.issubset(pooled.columns)):
        fine = pooled.loc[pooled.quantity.eq("projected_fine")]
        raw = pooled.loc[pooled.quantity.eq("raw_contrast")]
        total = budgets.loc[budgets.scope.eq("pooled")]
        if len(fine) == len(raw) == len(total) == 1:
            fine, raw, total = fine.iloc[0], raw.iloc[0], total.iloc[0]
            allowance = float(total.price_allowance)
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3),
                gridspec_kw={"width_ratios": (1., 1.3)})
            previous = tables.get("previous_comparison")
            points = [("New realized contrast", raw.estimate, 1.96*raw.standard_error, "#2878a5")]
            if previous is not None and not previous.empty:
                p = previous.iloc[0]
                points.insert(0, ("Historical raw pool\nkept separate", p.estimate,
                    1.96*p.standard_error, "#a6611a"))
            axes[0].axvspan(-allowance, allowance, color="#ddeedd", label="Unchanged price allowance")
            for j, (label, mean, error, color) in enumerate(points):
                axes[0].errorbar([mean], [j], xerr=[error], fmt="o", color=color, capsize=4)
            axes[0].axvline(0., color="#666666", lw=1.)
            axes[0].set(yticks=np.arange(len(points)), yticklabels=[p[0] for p in points],
                xlabel="Expected raw/control contrast ($)", title="The raw interval stays wide")
            axes[0].set_ylim(len(points)-.5, -.5)
            axes[0].xaxis.set_major_locator(MaxNLocator(5))
            axes[0].legend(loc="lower center", bbox_to_anchor=(.5, -.28), frameon=False)
            streams = budgets.loc[budgets.scope.eq("stream")]
            seed_rows = tables.get("per_seed", pd.DataFrame())
            seed_rows = seed_rows.loc[seed_rows.quantity.eq("projected_fine")] if not seed_rows.empty else seed_rows
            rows = []
            for b in streams.itertuples():
                selected = seed_rows.loc[seed_rows.seed.eq(b.seed)]
                if len(selected) == 1:
                    rows.append((f"{int(b.seed):,}", float(selected.iloc[0].estimate),
                        float(b.uncertainty_endpoint_low), float(b.uncertainty_endpoint_high), "#2878a5"))
            rows.append(("Fresh-only conditional pool", float(fine.estimate),
                float(total.uncertainty_endpoint_low), float(total.uncertainty_endpoint_high), "#268577"))
            axes[1].axvspan(-1., 1., color="#ddeedd")
            axes[1].axvline(0., color="#666666", lw=1.)
            for j, (_, mean, low, high, color) in enumerate(rows):
                axes[1].errorbar([mean/allowance], [j],
                    xerr=[[max(0., mean-low)/allowance], [max(0., high-mean)/allowance]],
                    fmt="D" if j == len(rows)-1 else "o", color=color, capsize=3)
            axes[1].set(yticks=np.arange(len(rows)), yticklabels=[row[0] for row in rows],
                xlabel="Conditional estimate / price allowance", title="Same expectation; less shock noise", xlim=(-1.2, 1.2))
            axes[1].set_ylim(len(rows)-.5, -.5)
            fig.suptitle("Independent paths test one frozen stopped-control contrast", y=1.01)
            fig.tight_layout()
            figures["tolerance_scale_comparison"] = fig

            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
            pilot = tables.get("pilot_budget_trials")
            if pilot is not None and not pilot.empty:
                low = pilot.loc[pilot.quadrature_nodes.eq("16/32/64")]
                high = pilot.loc[pilot.quadrature_nodes.eq("32/64/128")]
                values = [float(low.max_abs_first_level_selected_drift.max()),
                    float(low.max_abs_middle_level_selected_drift.max()),
                    float(low.max_abs_final_selected_drift.max()),
                    float(high.max_abs_final_selected_drift.max())]
                axes[0].plot([16, 32, 64, 128], values, "o-", color="#a6611a")
                axes[0].set(yscale="log", xticks=[16, 32, 64, 128],
                    xlabel="Gaussian quadrature nodes per smooth interval",
                    ylabel="Largest absolute conditional residual ($)", title="A coarse quadrature trial failed")
            else:
                axes[0].text(.5, .5, "No excluded quadrature trial recorded", ha="center", transform=axes[0].transAxes)
            axes[0].axhline(allowance, color="#268577", ls="--", label="Unchanged price allowance")
            axes[0].legend(frameon=False, fontsize=8)
            names = ["sampling_half_width", "absolute_refinement_change", "quadrature_error", "tail_error", "arithmetic_error"]
            labels = ["Sampling\n95% half-width", "Absolute\nrefinement", "Quadrature\nestimate", "Tail\nenvelope", "Arithmetic\nallowance"]
            values = np.asarray([total[name] for name in names], dtype=float)
            positive = values > 0
            axes[1].bar(np.arange(len(names))[positive], values[positive], color="#2878a5", width=.65)
            axes[1].axhline(allowance, color="#268577", ls="--")
            axes[1].set(yscale="log", xticks=np.arange(len(names)), xticklabels=labels,
                ylabel="Pooled whole-path uncertainty contribution ($)", title="Retain the numerical budget")
            for j in np.flatnonzero(~positive):
                axes[1].text(j, .02, "exact zero", ha="center", transform=axes[1].get_xaxis_transform(), rotation=90)
            fig.suptitle("Integration refinement is separate from Monte Carlo precision", y=1.01)
            fig.tight_layout()
            figures["integration_and_uncertainty"] = fig
    return {**result, "tables": tables, "figures": figures, "manifest": manifest}
