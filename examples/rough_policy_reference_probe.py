"""Independent-path BS-limit policy attribution with one frozen dual control.

The deterministic reference policies are declared before valuation. This pilot
does not select a rough-Heston policy or identify an absolute rough error split.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from rvlab.pricing import approximate_fractional_kernel
from rvlab.pricing.rough_dual import RoughDualConfig, train_rough_dual_weights
from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
from rvlab.pricing.rough_policy_diagnostics import evaluate_trace_policy
from rvlab.pricing.rough_policy_reference import build_black_scholes_reference_policy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    fourier_policy_fingerprint, train_fourier_stopping_policy, value_fourier_stopping_policy,
)
from rvlab.pricing.rough_stopping_reference import black_scholes_stopping_reference
from rvlab.experiments._pricing_rough_american import _american_models
from rvlab.experiments._pricing_rough_foundation import _market
from rvlab.experiments.pricing import _jsonable


def _summary(samples):
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1)/np.sqrt(len(samples)))
    return {"mean": mean, "standard_error": se,
        "confidence_low": mean-1.96*se, "confidence_high": mean+1.96*se}


def main():
    started = time.perf_counter()
    destination = ROOT/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise/rough_policy_reference_probe_2026-10-18"
    if destination.exists():
        raise FileExistsError(f"Preserved probe already exists: {destination}")
    destination.mkdir(parents=True)
    sources = [Path(__file__)]+sorted((ROOT/"src/rvlab/pricing").glob("*.py"))
    hashes = {}
    for source in sources:
        payload = source.read_bytes()
        relative = str(source.relative_to(ROOT))
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        snapshot = destination/"sources"/relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(payload)
    params, market = _american_models()["constant variance"], _market()
    kernel = approximate_fractional_kernel(.15, 1., 4,
        min_time_ratio=.003, max_nfev=4000)
    config = RoughAmericanConfig(training_paths=2048, valuation_paths=8192,
        steps=64, exercise_steps=32, batch_size=512,
        training_seed=262000042, valuation_seed=264000042,
        stock_knots=(.85, .9, .95, 1., 1.05, 1.1, 1.2), premium_cap=.7,
        training_method="fourier_premium")
    base = train_fourier_stopping_policy(params, market, kernel,
        strike=100., maturity=60/365, config=config, premium_cap=.7)
    proxy = build_rough_obstacle_proxy(base, stock_knots=(.7,.85,.875,.90,.91,
        .92,.93,.94,.95,.96,.97,.98,.99,1.,1.01,1.02,1.05,1.1,1.2,1.3))
    settings = RoughDualConfig(training_paths=2048, training_seed=263000042,
        candidate_bins=(1, 4), validation_fraction=.5,
        european_bounds=(1., 1.), premium_bounds=(0., 2.),
        components=("european", "premium", "micro_boundary"),
        intrinsic_bounds=(0., 1.5), target_valuation_paths=config.valuation_paths)
    training = value_fourier_stopping_policy(proxy,
        config=replace(proxy.config, valuation_paths=2048, valuation_seed=settings.training_seed),
        trace=True)["trace"]
    weights = train_rough_dual_weights(proxy, config=settings, training_trace=training)
    references = {name: build_black_scholes_reference_policy(base,
        spot_nodes=nodes, log_half_width=width) for name, nodes, width in (
            ("DP 32769", 32769, 2.), ("DP 65537", 65537, 2.),
            ("DP expanded", 131073, 4.))}
    reference = black_scholes_stopping_reference(market, volatility=.2,
        strike=100., exercise_times=base.exercise_times, spot_nodes=32769)
    sealed = {"started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "dual_config": asdict(settings),
        "policy_fingerprint": fourier_policy_fingerprint(proxy),
        "source_hashes": hashes, "params": asdict(params),
        "reference": {"price": reference.price, "diagnostics": dict(reference.diagnostics)},
        "reference_policies": {name: dict(policy.diagnostics) for name, policy in references.items()},
        "selected_dual": dict(weights.diagnostics),
        "decisions_frozen_before_valuation": True, "valuation_used_for_selection": False,
        "same_original_policy_poststop_gate": True}
    (destination/"selection_before_valuation.json").write_text(
        json.dumps(_jsonable(sealed), indent=2, allow_nan=False)+"\n")
    trace = value_fourier_stopping_policy(proxy, trace=True)["trace"]
    baseline = evaluate_trace_policy(trace, martingale_weights=weights)
    values = {"base": baseline, **{name: evaluate_trace_policy(trace, policy=policy,
        martingale_weights=weights) for name, policy in references.items()}}
    rows, samples = [], {}
    for name, value in values.items():
        if not np.array_equal(value.dual_samples, baseline.dual_samples):
            raise RuntimeError("Policy changed the common upper samples")
        lower, upper = value.controlled_samples[:, 0], value.dual_samples[:, 0]
        gap = upper-lower
        gain = lower-baseline.controlled_samples[:, 0]
        raw_gain = value.raw_samples[:, 0]-baseline.raw_samples[:, 0]
        lower_stats, gap_stats = _summary(lower), _summary(gap)
        allowance = .001+1e-4*abs(lower_stats["mean"])
        budget = gap_stats["mean"]+1.96*gap_stats["standard_error"]+1.96*lower_stats["standard_error"]
        rows.append({"policy": name, "lower": lower_stats["mean"],
            "lower_se": lower_stats["standard_error"], "upper": float(upper.mean()),
            "gap": gap_stats["mean"], "gap_se": gap_stats["standard_error"],
            "paired_gain": _summary(gain)["mean"], "paired_gain_se": _summary(gain)["standard_error"],
            "paired_gain_ci_low": _summary(gain)["confidence_low"],
            "paired_gain_ci_high": _summary(gain)["confidence_high"],
            "paired_raw_gain": _summary(raw_gain)["mean"],
            "paired_raw_gain_se": _summary(raw_gain)["standard_error"],
            "decision_change_fraction": float(np.mean(value.stop_indices != baseline.stop_indices)),
            "reference_shortfall": reference.price-lower_stats["mean"],
            "price_allowance": allowance, "price_error_budget": budget,
            "price_budget_pass": budget <= allowance, "same_upper_samples": True})
        key = name.lower().replace(" ", "_")
        samples[key+"_lower"] = lower
        samples[key+"_raw"] = value.raw_samples[:, 0]
        samples[key+"_stops"] = value.stop_indices[:, 0]
    samples["common_upper"] = baseline.dual_samples[:, 0]
    samples["times"] = np.asarray(trace["times"])
    samples["stocks"] = np.asarray(trace["stocks"])[..., 0]
    pd.DataFrame(rows).to_csv(destination/"comparison.csv", index=False)
    np.savez_compressed(destination/"paired_samples.npz", **samples)
    changed = [str(source.relative_to(ROOT)) for source in sources if
        hashlib.sha256(source.read_bytes()).hexdigest() != hashes[str(source.relative_to(ROOT))]]
    manifest = {**sealed, "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-started,
        "source_files_changed_during_probe": changed,
        "scope": "independent BS-limit same-calendar reference; no absolute rough decomposition",
        "same_upper_samples": True, "tolerances_unchanged": True,
        "american_pricing_ready": False, "optimal_stock_delta_ready": False}
    (destination/"manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, allow_nan=False)+"\n")
    inventory = [{"path": str(path.relative_to(destination)), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(destination.rglob("*")) if path.is_file()]
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    (destination/"preservation_inventory.json").write_text(json.dumps({
        "files": inventory, "inventory_sha256": hashlib.sha256(canonical).hexdigest()}, indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("SECONDS", manifest["elapsed_seconds"], "SOURCE CHANGES", changed, flush=True)


if __name__ == "__main__":
    main()
