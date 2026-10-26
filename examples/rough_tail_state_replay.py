"""Recover omitted states on spent final paths; never acceptance or selection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.experiments._pricing_rough_state_replay import replay_inception_states
from rvlab.experiments._pricing_rough_policy_validation_stopping import _basis


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()


def main():
    output = ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder = output/"rough_tail_attribution_v1/spent_state_replay_v1"
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    bundle_folder = output/"rough_larger_policy_v1/32768_paths_two_starts"
    bundle = load_larger_policy_bundle(bundle_folder)
    proxy, policy = bundle["proxy"], bundle["selected_policy"]
    historical = output/"rough_gap_time_refinement_v1/stock_grid_control_v2"
    seal_path = historical/"controls_before_final.json"
    seal = json.loads(seal_path.read_text())
    control = seal["controls"][seal["selected"]]
    index = list(seal["controls"]).index(seal["selected"])
    sources = [Path(__file__), ROOT/"src/rvlab/experiments/_pricing_rough_state_replay.py"]
    sources += sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for p in sources:
        target = folder/"source"/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    inputs = [seal_path, bundle_folder/"frozen_policy_before_final.json", bundle_folder/"frozen_policy_coefficients.npz"]
    inputs += [historical/f"final_{seed}_samples.npz" for seed in (954005042, 955005042)]
    design = {"version": "rough_spent_state_replay_v1", "created_at": now(), "source_sha256": hashes,
        "seeds": [954005042, 955005042], "paths_per_seed": 4096, "steps": proxy.config.steps,
        "input_sha256": {str(p.resolve()): sha(p) for p in inputs}, "selected_control": seal["selected"],
        "used_for_acceptance": False, "used_for_formal_selection": False,
        "purpose": "retrospective reconstruction and diagnosis of already-spent synthetic inception paths"}
    (folder/"design_before_replay.json").write_text(json.dumps(design, indent=2)+"\n")
    rows, streams = [], []
    started = perf_counter()
    for seed in design["seeds"]:
        with np.load(historical/f"final_{seed}_samples.npz", allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        replay = replay_inception_states(proxy, seed=seed, paths=4096, expected_payoffs=arrays["payoff"])
        state = replay["factors"]
        stock = replay["stocks"][:, :, 0]
        np.savez_compressed(folder/f"spent_{seed}_states.npz", stocks=replay["stocks"], factors=state,
            times=replay["times"], actual_time_grid=replay["actual_time_grid"])
        m = arrays["baseline_martingale"]+np.einsum("ntd,d->nt", _basis(arrays["grid_corrections"], control["bins"]), control["weights"])
        adjusted = arrays["payoff"]-m
        winner, stops = np.argmax(adjusted, axis=1), arrays["stops"]
        gap = arrays[f"control{index}_upper"]-arrays[f"control{index}_lower"]
        norms = np.empty(stock.shape)
        q = np.empty(stock.shape)
        for j, model in enumerate(policy.continuations):
            norms[:, j] = np.max(abs((state[:, j]-model.center)/model.scale), axis=1)
            q[:, j] = np.clip(model.value(stock[:, j, None], state[:, j], proxy.strike)[:, 0], 0., policy.premium_cap)
        variance = proxy.params.v0+np.sum(state*np.asarray(proxy.kernel.weights), axis=-1)
        for path, (j, tau) in enumerate(zip(winner, stops)):
            rows.append({"seed": seed, "path_index": path, "gap": gap[path], "winning_date": int(j),
                "stop_date": int(tau), "winning_time": replay["times"][j], "stop_time": replay["times"][tau],
                "winner_before_stop": bool(j < tau), "winner_after_stop": bool(j > tau),
                "winning_stock": stock[path, j], "stop_stock": stock[path, tau],
                "winning_variance": variance[path, j], "stop_variance": variance[path, tau],
                "winning_standardized_factor_max": norms[path, j], "stop_standardized_factor_max": norms[path, tau],
                "winning_right_reliability_R3": float(np.exp(-(norms[path, j]/3.)**8)),
                "winning_selected_premium": q[path, j], "winning_discounted_payoff": arrays["payoff"][path, j],
                "stopped_discounted_payoff": arrays["raw"][path], "max_discounted_payoff": float(arrays["payoff"][path].max()),
                "all_payoffs_zero": bool(np.all(arrays["payoff"][path] == 0)),
                "acceptance_input": False, "used_for_selection": False})
        streams.append({"seed": seed, "paths": 4096, **replay["diagnostics"], "used_for_acceptance": False})
    table = pd.DataFrame(rows)
    table.to_csv(folder/"tail_states.csv", index=False)
    pd.DataFrame(streams).to_csv(folder/"streams.csv", index=False)
    summary = []
    conditions = {"winner before stop": table.winner_before_stop, "winner after stop": table.winner_after_stop,
        "winning stock >= strike": table.winning_stock >= proxy.strike,
        "winning factors beyond3sd": table.winning_standardized_factor_max > 3.,
        "winning factors beyond4sd": table.winning_standardized_factor_max > 4.,
        "all realized payoffs zero": table.all_payoffs_zero}
    for name, condition in conditions.items():
        summary.append({"group": name, "paths": int(condition.sum()), "path_fraction": float(condition.mean()),
            "gap_share": float(table.loc[condition, "gap"].sum()/table.gap.sum()),
            "mean_gap": float(table.loc[condition, "gap"].mean()), "acceptance_input": False})
    pd.DataFrame(summary).to_csv(folder/"state_summary.csv", index=False)
    manifest = {**design, "completed_at": now(), "elapsed_seconds": perf_counter()-started,
        "all_payoff_streams_bit_exact": True, "all_paths_retained": True,
        "source_changes": [n for n, h in hashes.items() if sha(ROOT/n) != h],
        "sources_unchanged": all(sha(ROOT/n) == h for n, h in hashes.items()), "rows": len(table)}
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    entries = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(folder.rglob("*")) if p.is_file()]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (folder/"preservation_inventory.json").write_text(json.dumps({"files": entries, "inventory_sha256": digest}, indent=2)+"\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print(table.nlargest(8, "gap").to_string(index=False))
    print("SECONDS", manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
