"""All-path post-stop capacity on sealed spent streams; no simulation or fit."""
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
from rvlab.experiments._pricing_rough_tail_capacity import poststop_capacity, summarize_poststop_capacity


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True))


def main():
    output = ROOT / "notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    source = output / "rough_gap_time_refinement_v1/stock_grid_control_v2"
    folder = output / "rough_tail_attribution_v1/spent_poststop_capacity_v1"
    if folder.exists():
        raise FileExistsError("Preserve the existing capacity diagnostic rather than replace it")
    inventory = json.loads((source / "preservation_inventory.json").read_text())
    canonical = json.dumps(inventory["files"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != inventory["inventory_sha256"]:
        raise ValueError("Changed spent experiment inventory")
    for entry in inventory["files"]:
        path = source / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Changed spent numerical evidence")
    folder.mkdir(parents=True)
    paths = [Path(__file__).resolve(), ROOT / "src/rvlab/experiments/_pricing_rough_tail_capacity.py",
        ROOT / "src/rvlab/experiments/_pricing_rough_local_control.py"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    for path in paths:
        target = folder / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    controls = json.loads((source / "controls_before_final.json").read_text())
    with np.load(output / "rough_larger_policy_v1/32768_paths_two_starts/frozen_policy_coefficients.npz", allow_pickle=False) as z:
        times = z["times"].copy()
    declaration = {"version": "rough_tail_spent_capacity_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sources, "source_inventory_sha256": sha(source / "preservation_inventory.json"),
        "source_control_seal_sha256": sha(source / "controls_before_final.json"),
        "spent_seeds": [954005042, 955005042], "paths_per_seed": 4096,
        "selected_control": controls["selected"], "new_paths": 0, "new_fits": 0,
        "acceptance_input": False, "used_for_selection": False,
        "causal_policy_or_dual_error_identified": False,
        "scope": "For each fixed baseline M, bound possible improvement from controls unchanged through its fixed policy stop.",
        "contrast": "capacity minus half the random baseline gap; paired complete paths, stratified stream SE",
        "interpretation": "A negative 95% upper contrast bound limits only the strictly post-stop family. It does not identify policy error or total dual error."}
    write(folder / "design_before_analysis.json", declaration)
    chunks = {name: [] for name in controls["controls"]}
    path_arrays = {}
    for seed in declaration["spent_seeds"]:
        with np.load(source / f"final_{seed}_samples.npz", allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        for index, (name, control) in enumerate(controls["controls"].items()):
            basis = correction_basis(arrays["grid_corrections"], times, control["bins"])
            martingale = arrays["baseline_martingale"] + np.einsum("ntd,d->nt", basis, control["weights"])
            capacity = poststop_capacity(arrays["payoff"], martingale, arrays["stops"])
            for suffix in ("lower", "upper"):
                np.testing.assert_allclose(capacity[f"baseline_{suffix}"], arrays[f"control{index}_{suffix}"], rtol=0., atol=2e-11)
            chunks[name].append(capacity)
            for key, value in capacity.items():
                path_arrays[f"seed{seed}_control{index}_{key}"] = value
    rows = []
    for name, values in chunks.items():
        row = summarize_poststop_capacity(values).iloc[0].to_dict()
        rows.append({"control": name, "selected_before_final": name == controls["selected"], **row})
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "capacity_comparison.csv", index=False)
    np.savez_compressed(folder / "capacity_samples.npz", **path_arrays)
    if sources != {str(path.relative_to(ROOT)): sha(path) for path in paths}:
        raise ValueError("Analysis source changed during spent diagnostics")
    write(folder / "manifest.json", {**declaration, "complete": True,
        "control_order": list(controls["controls"]), "controls": len(chunks),
        "verified_source_files": len(inventory["files"]), "all_original_paths_retained": True,
        "poststop_family_ceiling_verified_pathwise": True})
    entries = [{"path": str(path.relative_to(folder)), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(folder.rglob("*")) if path.is_file()]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(folder / "preservation_inventory.json", {"files": entries, "inventory_sha256": digest})
    print(frame.to_string(index=False))
    print(f"Preserved {len(entries)} files: {digest}")


if __name__ == "__main__": main()
