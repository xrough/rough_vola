"""Descriptive tail timing on already spent paths; no simulation or fitting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from rvlab.experiments._pricing_rough_local_control import correction_basis
from rvlab.experiments._pricing_rough_tail_policy import spent_gap_timing, margin_policy_specs


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True))


def cohort_rows(frame):
    ordered = frame.sort_values(["gap", "seed", "path_index"], ascending=[False, True, True])
    whole_gap = float(frame.gap.sum())
    rows = []
    for name, count in (("all", len(frame)), ("largest decile", int(np.ceil(.1 * len(frame)))),
            ("largest percentile", int(np.ceil(.01 * len(frame))))):
        subset = ordered.iloc[:count]; positive = subset.loc[subset.gap > 1e-12]
        mean = lambda column: float(positive[column].mean()) if len(positive) else None
        rows.append({"cohort": name, "paths": len(subset), "positive_gap_paths": len(positive),
            "gap_mean": float(subset.gap.mean()), "gap_sum_share": float(subset.gap.sum() / whole_gap),
            "peak_before_stop_positive_fraction": mean("peak_before_stop"),
            "peak_after_stop_positive_fraction": mean("peak_after_stop"),
            "stop_at_maturity_positive_fraction": mean("stop_at_maturity"),
            "peak_at_maturity_positive_fraction": mean("peak_at_maturity"),
            "zero_payoff_at_peak_positive_fraction": float(np.mean(positive.payoff_at_peak <= 1e-12)) if len(positive) else None,
            "gap_mean_raw_accounting": float((subset.payoff_at_peak - subset.payoff_at_stop).mean()),
            "gap_mean_martingale_accounting": float((subset.martingale_at_stop - subset.martingale_at_peak).mean()),
            "acceptance_input": False, "used_for_selection": False,
            "causal_policy_or_dual_error_identified": False})
    return rows


def main():
    output = ROOT / "notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    source = output / "rough_gap_time_refinement_v1/stock_grid_control_v2"
    folder = output / "rough_tail_attribution_v1/spent_policy_timing"
    if folder.exists(): raise FileExistsError("Preserve the existing diagnostic instead of replacing it")
    folder.mkdir(parents=True)
    inventory = json.loads((source / "preservation_inventory.json").read_text())
    canonical = json.dumps(inventory["files"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != inventory["inventory_sha256"]:
        raise ValueError("Changed spent experiment inventory")
    for entry in inventory["files"]:
        path = source / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Changed spent numerical evidence")
    paths = [Path(__file__).resolve(), ROOT / "src/rvlab/experiments/_pricing_rough_tail_policy.py",
        ROOT / "src/rvlab/experiments/_pricing_rough_local_control.py"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    for path in paths:
        target = folder / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    controls = json.loads((source / "controls_before_final.json").read_text())
    selected = controls["selected"]
    design = json.loads((source / "design_before_pilot.json").read_text())
    model = json.loads((output / "rough_larger_policy_v1/32768_paths_two_starts/frozen_policy_before_final.json").read_text())
    with np.load(output / "rough_larger_policy_v1/32768_paths_two_starts/frozen_policy_coefficients.npz", allow_pickle=False) as z:
        times = z["times"].copy()
    declaration = {"version": "rough_tail_spent_policy_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sources, "source_inventory_sha256": sha(source / "preservation_inventory.json"),
        "source_control_seal_sha256": sha(source / "controls_before_final.json"),
        "spent_seeds": [954005042, 955005042], "paths_per_seed": 4096,
        "selected_control": selected, "selected_policy": model["selected_name"],
        "margin_challenger_family_declared_before_analysis": margin_policy_specs(),
        "new_paths": 0, "new_fits": 0, "acceptance_input": False,
        "used_for_selection": False, "causal_policy_or_dual_error_identified": False,
        "interpretation": "Ex-post peak timing and raw/martingale accounting are descriptive; no deployable hindsight stopping rule."}
    write(folder / "design_before_analysis.json", declaration)
    frames = []
    for seed in declaration["spent_seeds"]:
        with np.load(source / f"final_{seed}_samples.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        for index, (name, control) in enumerate(controls["controls"].items()):
            basis = correction_basis(arrays["grid_corrections"], times, control["bins"])
            martingale = arrays["baseline_martingale"] + np.einsum("ntd,d->nt", basis, control["weights"])
            rows = spent_gap_timing(arrays["payoff"], martingale, arrays["stops"], times=times, seed=seed)
            np.testing.assert_allclose(rows.lower, arrays[f"control{index}_lower"], rtol=0., atol=2e-11)
            np.testing.assert_allclose(rows.upper, arrays[f"control{index}_upper"], rtol=0., atol=2e-11)
            rows["control"] = name; rows["selected_before_final"] = name == selected
            frames.append(rows)
    all_rows = pd.concat(frames, ignore_index=True)
    selected_rows = all_rows.loc[all_rows.control == selected].copy()
    selected_rows.to_csv(folder / "selected_path_timing.csv", index=False)
    summaries = []
    for name, frame in all_rows.groupby("control", sort=False):
        summaries.extend({"control": name, "selected_before_final": name == selected, **row}
            for row in cohort_rows(frame))
    pd.DataFrame(summaries).to_csv(folder / "tail_cohort_summary.csv", index=False)
    sensitivity = []
    ref = selected_rows.sort_values(["seed", "path_index"]).reset_index(drop=True)
    for name, frame in all_rows.groupby("control", sort=False):
        other = frame.sort_values(["seed", "path_index"]).reset_index(drop=True)
        for fraction in (.1, .01):
            n = int(np.ceil(fraction * len(ref)))
            a = set(ref.sort_values("gap", ascending=False, kind="stable").index[:n])
            b = set(other.sort_values("gap", ascending=False, kind="stable").index[:n])
            sensitivity.append({"control": name, "tail_fraction": fraction, "tail_paths": n,
                "overlap_fraction_with_selected": len(a & b) / n,
                "all_path_dual_peak_change_fraction": float(np.mean(ref.dual_peak_index != other.dual_peak_index)),
                "all_path_lower_change": float((other.lower - ref.lower).mean()),
                "all_path_upper_change": float((other.upper - ref.upper).mean()),
                "acceptance_input": False, "used_for_selection": False,
                "policy_fixed": True, "causal_policy_or_dual_error_identified": False})
    pd.DataFrame(sensitivity).to_csv(folder / "control_tail_sensitivity.csv", index=False)
    if sources != {str(path.relative_to(ROOT)): sha(path) for path in paths}:
        raise ValueError("Analysis source changed during spent diagnostics")
    write(folder / "manifest.json", {**declaration, "complete": True,
        "selected_paths": len(selected_rows), "control_path_rows_examined": len(all_rows),
        "verified_source_files": len(inventory["files"]), "all_original_paths_retained": True})
    entries = [{"path": str(path.relative_to(folder)), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(folder.rglob("*")) if path.is_file()]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(folder / "preservation_inventory.json", {"files": entries, "inventory_sha256": digest})
    print(pd.DataFrame(summaries).loc[lambda t: t.selected_before_final].to_string(index=False))
    print(pd.DataFrame(sensitivity).to_string(index=False))


if __name__ == "__main__": main()
