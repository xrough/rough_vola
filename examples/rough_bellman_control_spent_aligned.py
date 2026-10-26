"""Fit and benchmark a fixed control proxy on already spent Bellman labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from rvlab.experiments._pricing_rough_bellman_control_aligned import fit_bellman_correction, bellman_target_martingales, TARGETS
from rvlab.experiments._pricing_rough_frozen_policy import _encode_models
from rvlab.experiments._pricing_rough_large_policy import load_larger_policy_bundle
from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True))


def main():
    output = ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    source = output/"rough_bellman_policy_v1/one_step_v1"
    model_folder = output/"rough_larger_policy_v1/32768_paths_two_starts"
    folder = output/"rough_bellman_validation_v1/spent_proxy_aligned_v2"
    if folder.exists(): raise FileExistsError("Preserve the existing spent proxy trial rather than replace it")
    folder.mkdir(parents=True)
    sources = list((ROOT/"src/rvlab/pricing").glob("*.py"))+[Path(__file__).resolve()]
    sources += [ROOT/"src/rvlab/experiments"/name for name in ("_pricing_rough_bellman_control.py", "_pricing_rough_bellman_control_aligned.py",
        "_pricing_rough_exercise_block.py", "_pricing_rough_node_refinement_fast.py", "_pricing_rough_local_control.py",
        "_pricing_rough_large_policy.py", "_pricing_rough_frozen_policy.py")]
    inputs = [model_folder/"frozen_policy_before_final.json", model_folder/"frozen_policy_coefficients.npz",
        source/"selection_before_final.json", source/"manifest.json"]
    inputs += [source/f"final_{i}_{suffix}" for i in range(2) for suffix in ("samples.npz", "manifest.json")]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    input_hashes = {str(p.relative_to(output)): sha(p) for p in inputs}
    for path in sources:
        target = folder/"source"/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    design = {"version": "rough_bellman_control_spent_proxy_aligned_v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "fit_seed": 1077005042, "diagnostic_seed": 1078005042, "paths_per_spent_stream": 4096,
        "fit_target": "GH16 continuation minus old Euro+clipped-q continuation; all saved ITM labels",
        "ridge": .001, "degree": 1, "factor_clip": 3., "correction_bound": .05,
        "targets": list(TARGETS), "regression_row_clip_or_deletion": False,
        "nodal_european_grid": "original129 physical grid, same as actual/expected Euro",
        "previous_mixed_grid_trial_preserved": "spent_proxy_v1", "new_outer_paths": 0, "new_fits_of_exercise_rule": 0,
        "benchmark_source_seed": 1078005042, "benchmark_path_indices": list(range(32)),
        "benchmark_uses_all_original_exercise_dates": True,
        "used_for_fresh_acceptance": False, "acceptance_tolerances_changed": False}
    write(folder/"design_before_fit.json", design)
    bundle = load_larger_policy_bundle(model_folder); proxy = bundle["proxy"]
    arrays = []
    for i in range(2):
        record = json.loads((source/f"final_{i}_manifest.json").read_text())
        path = source/f"final_{i}_samples.npz"
        if record["sample_sha256"] != sha(path): raise ValueError("Changed spent Bellman labels")
        with np.load(path, allow_pickle=False) as z:
            arrays.append({name: z[name].copy() for name in ("stocks", "factors", "european_proxy", "times", "discounts",
                "actual_time_grid", "bellman_computed_mask", "bellman_continuations", "baseline_continuation")})
    start = perf_counter(); models, diagnostics = fit_bellman_correction(proxy, *arrays)
    fit_seconds = perf_counter()-start
    coefficients = {}; declarations = _encode_models(models[1:-1], "correction", coefficients)
    np.savez_compressed(folder/"proxy_coefficients.npz", **coefficients)
    diagnostics.to_csv(folder/"proxy_fit_diagnostics.csv", index=False)
    write(folder/"proxy_before_fresh_paths.json", {**design, "model_dates": list(range(1, 64)),
        "models": declarations, "coefficient_archive": "proxy_coefficients.npz",
        "coefficient_archive_sha256": sha(folder/"proxy_coefficients.npz"), "fit_seconds": fit_seconds,
        "no_fresh_paths_generated": True})
    source_arrays = arrays[1]
    trace = {name: source_arrays[name][:32].copy() for name in ("stocks", "factors", "european_proxy")}
    trace.update({name: source_arrays[name] for name in ("times", "discounts", "actual_time_grid")})
    trace.update(seed=1078005042, policy_fingerprint=fourier_policy_fingerprint(proxy),
        kernel_fingerprint=proxy.kernel.fingerprint, strike=proxy.strike, initial_spots=np.array([proxy.market.spot]))
    start = perf_counter()
    targets, benchmark_diagnostics, primitives = bellman_target_martingales(proxy, bundle["selected_policy"], models,
        trace, require_independent=False, batch_size=32, return_components=True)
    elapsed = perf_counter()-start
    np.savez_compressed(folder/"spent_target_samples.npz", targets=targets, **primitives)
    from dataclasses import replace
    from rvlab.experiments._pricing_rough_bellman_control_aligned import aligned_node_proxy
    from rvlab.experiments._pricing_rough_node_refinement_fast import refined_stock_knots
    from rvlab.pricing.rough_estimators import _AffineSurrogate
    from rvlab.experiments._pricing_rough_bellman_control import target_nodal_premiums as mixed_nodes
    from rvlab.experiments._pricing_rough_bellman_control_aligned import target_nodal_premiums as aligned_nodes
    import pandas as pd
    affine = _AffineSurrogate(proxy.params, proxy.market, proxy.kernel, np.array([proxy.strike]),
        np.array([proxy.maturity]), np.array([proxy.is_call]), trace["actual_time_grid"], proxy.control_config)
    aligned = aligned_node_proxy(proxy, affine); nodes = np.r_[0., refined_stock_knots(proxy.config.stock_knots, 8)]
    alignment_rows = []
    for j in (16, 32, 48, 63):
        memory = trace["factors"][:, j]
        old_euro = replace(proxy.value_premiums[j], stock_nodes=nodes).european_values(memory)
        new_euro = replace(aligned.value_premiums[j], stock_nodes=nodes).european_values(memory)
        old_nodes = mixed_nodes(proxy, bundle["selected_policy"], models, j, memory, fine_nodes=nodes)
        new_nodes = aligned_nodes(proxy, bundle["selected_policy"], models, j, memory, fine_nodes=nodes, node_proxy=aligned)
        for i in range(32):
            difference = new_euro[i]-old_euro[i]
            premium_difference = new_nodes[i]-old_nodes[i]
            alignment_rows.append({"exercise_index": j, "spent_path_index": i, "seed": 1078005042,
                "european_max_abs_change": float(np.max(abs(difference))), "european_mean_abs_change": float(np.mean(abs(difference))),
                "premium_max_abs_change": float(np.max(abs(premium_difference))),
                "used_for_fresh_acceptance": False, "previous_trial_validly_centered": True,
                "previous_trial_same_european_Snell_specification": False, "new_nodal_physical_grid_aligned": True})
    pd.DataFrame(alignment_rows).to_csv(folder/"mixed_grid_alignment.csv", index=False)
    prior_seconds = max(json.loads((source/f"final_{i}_manifest.json").read_text())["elapsed_seconds"] for i in range(2))
    target_projection = elapsed*4096/32
    benchmark = {"paths": 32, "elapsed_seconds": elapsed, "fit_seconds": fit_seconds,
        "target_projection_4096_seconds": target_projection,
        "previous_full_single_stock_4096_worker_seconds": prior_seconds,
        "conservative_worker_4096_projection_seconds": target_projection+prior_seconds,
        "final_budget_recommendation": 4096 if target_projection+prior_seconds <= 1200 else 2048,
        "projection_is_not_runtime_guarantee": True, "all_paths_and_dates_retained": True,
        "diagnostics": benchmark_diagnostics}
    write(folder/"spent_benchmark.json", benchmark)
    if source_hashes != {str(p.relative_to(ROOT)): sha(p) for p in sources}:
        raise ValueError("Spent proxy source changed during fitting/benchmarking")
    if input_hashes != {str(p.relative_to(output)): sha(p) for p in inputs}: raise ValueError("Spent inputs changed")
    write(folder/"manifest.json", {**design, "complete": True, "fit_seconds": fit_seconds,
        "benchmark": benchmark, "fit_rows": int(diagnostics.loc[diagnostics.role.eq("spent fit1077"), "rows"].sum()),
        "diagnostic_rows": int(diagnostics.loc[diagnostics.role.eq("spent diagnostic1078"), "rows"].sum())})
    entries = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(folder.rglob("*")) if p.is_file()]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(folder/"preservation_inventory.json", {"files": entries, "inventory_sha256": digest})
    print(json.dumps(benchmark, indent=2)); print(f"Preserved {len(entries)} files: {digest}")


if __name__ == "__main__": main()
