"""Fingerprint-controlled follow-up to the unsuccessful rough stopping study.

The finite-kernel stopping problem, conditional sampling error and continuous
American-model accuracy remain separate.  Every trial is declared before any
valuation result is inspected; no favorable seed selects a reported estimator.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from ._pricing_rough_american import (
    PRICE_ATOL, PRICE_RTOL, DELTA_ATOL, DELTA_RTOL, CONFIDENCE_MULTIPLIER,
    _american_models, _core_source_hashes, _fd_reference, _preserve_sources,
    _stopping_rows, _stopping_trial, american_contracts, load_historical_american,
    stopping_accuracy, delta_accuracy,
)


def american_accuracy_trial_specs(*, training_paths=4096, valuation_paths=4096,
                                  steps=128, exercise_steps=64):
    """The bounded default campaign, declared before inspecting its outcomes.

    The first two rough trials have independent training and valuation streams;
    all refinements retain their own outcomes. More paths, a premium cap, or a
    Fourier surrogate are numerical budgets rather than relaxed tolerances.
    """
    from rvlab.pricing import RoughEuropeanControlConfig
    base = {"training_paths": training_paths, "valuation_paths": valuation_paths,
        "steps": steps, "exercise_steps": exercise_steps, "batch_size": 512,
        "memory_budget_mb": 1024., "training_method": "fourier_premium",
        "premium_cap": .7, "surrogate_interpolation": "value_bridge"}
    def trial(label, contract="rough_put_60_100", *, factors=4, **changes):
        return {"trial": label, "contract_id": contract, "factors": factors,
            "stage": label, "config": {**base, **changes}}
    return [trial("rough base"), trial("independent policy"),
        trial("curve specification", "curve_put_60_100"),
        trial("more training and paths", training_paths=4*training_paths,
            valuation_paths=2*valuation_paths),
        trial("time refinement", steps=2*steps),
        trial("exercise refinement", exercise_steps=2*exercise_steps),
        trial("joint and retrained bumps", steps=2*steps,
            exercise_steps=2*exercise_steps, retrain_bumps=True),
        trial("Heston limit", "heston_limit_put", factors=1,
            steps=2*steps, exercise_steps=2*exercise_steps),
        trial("constant variance", "constant_variance_put",
            steps=2*steps, exercise_steps=2*exercise_steps),
        trial("cash event bridge", "cash_call", training_method="conditional"),
        trial("eight factor probe", factors=8),
        trial("Fourier control refinement", premium_control=RoughEuropeanControlConfig(48, 200., 1024)),
        trial("half stock bump", bump_relative=.0025),
        trial("premium cap sensitivity", premium_cap=1.4)]


def american_accuracy_probe_archives():
    """Fixed exploratory provenance inputs for the executed default notebook."""
    root = Path(__file__).resolve().parents[3]
    output = root/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    paths = (root/"notebooks/_output/01_pricing_and_exercise/bridge_accuracy_2026-10-18",
        output/"rough_policy_accuracy_2026-10-18", output/"rough_fourier_accuracy_2026-10-18",
        output/"rough_signed_residual_probe_2026-10-18")
    for path in paths:
        if not (path/"preservation_manifest.json").is_file():
            raise FileNotFoundError(f"Missing preserved American accuracy trial inventory: {path}")
    return paths


def _validated_design(trial_specs, seed):
    """Create complete numerical configurations before invoking any simulator."""
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(trial_specs, (tuple, list)) or not trial_specs:
        raise ValueError("trial_specs must be a nonempty predeclared list of trial specifications")
    contracts = american_contracts().set_index("contract_id")
    rows, labels = [], set()
    for index, specification in enumerate(trial_specs):
        if not isinstance(specification, dict):
            raise TypeError("each trial specification must be a dictionary")
        allowed = {"trial", "stage", "contract_id", "factors", "config"}
        if set(specification)-allowed:
            raise ValueError(f"unknown trial fields at row {index}: {sorted(set(specification)-allowed)}")
        label, contract_id = specification.get("trial"), specification.get("contract_id")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"trial row {index} requires a distinct nonempty trial name")
        if contract_id not in contracts.index:
            raise ValueError(f"trial row {index} has an undeclared synthetic contract")
        stage = specification.get("stage", label)
        if not isinstance(stage, str) or not stage:
            raise ValueError(f"trial row {index} requires a nonempty stage")
        overrides = specification.get("config", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"trial row {index} config must be a dictionary")
        if {"training_seed", "valuation_seed"} & set(overrides):
            raise ValueError("trial seeds are assigned from the declared independent campaign streams")
        contract = {"contract_id": contract_id, **contracts.loc[contract_id].to_dict()}
        factors = specification.get("factors", 1 if contract["model"] == "Hurst .5" else 4)
        if isinstance(factors, bool) or not isinstance(factors, Integral) or factors not in (1, 4, 8):
            raise ValueError("trial factors must be 1, 4 or 8")
        if (factors == 1) != (contract["model"] == "Hurst .5"):
            raise ValueError("one-factor trials are reserved for the exact Hurst .5 kernel")
        config = RoughAmericanConfig(training_seed=int(seed)+1000*index,
            valuation_seed=int(seed)+1_000_000+1000*index, **overrides)
        rows.append((contract, label, stage, int(factors), config))
        labels.add(label)
    train = {row[4].training_seed for row in rows}
    valuation = {row[4].valuation_seed for row in rows}
    if train & valuation:
        raise ValueError("the campaign training and valuation streams overlap")
    return rows


def load_accuracy_probes(archives=()):
    """Load exploratory tables only after verifying their complete byte inventory.

    Exploratory ablations retain their own model/source/settings columns and are
    never promoted to the current production estimator or accepted calibrations.
    """
    from ._pricing_rough_precision import _read_verified_archive
    if not isinstance(archives, (list, tuple)):
        raise TypeError("probe_archives must be a list or tuple of preserved directories")
    tables, records = [], []
    for value in archives:
        archive = Path(value).resolve()
        inventory, digest = _read_verified_archive(archive)
        declared = {row["preserved"] for row in inventory["files"]}
        for name in sorted(declared):
            if not name.endswith(".csv"):
                continue
            try:
                table = pd.read_csv(archive/name)
            except pd.errors.EmptyDataError:
                continue
            if table.empty:
                continue
            table = table.copy()
            # Retain every original column; add common display coordinates for
            # independently authored exploratory runners only when absent.
            for alias, original in (("mode", "method"), ("se", "price_se"),
                    ("dual_gap", "gap")):
                if alias not in table and original in table:
                    table[alias] = table[original]
            table["probe"] = Path(name).stem
            table["historical_replay"] = True
            table["source_archive"] = str(archive)
            table["source_archive_sha256"] = digest
            tables.append(table)
        records.append({"folder": str(archive), "inventory_sha256": digest,
            "files": len(inventory["files"]), "interpretation": "preserved exploratory trials, not production acceptance"})
    return (pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame(), records)


def _accuracy_sources():
    functions = (_validated_design, _accuracy_sources, rough_american_accuracy_study,
        _stopping_trial, _stopping_rows, _fd_reference, _accuracy_comparison,
        stopping_accuracy, delta_accuracy)
    return {**_core_source_hashes(), "study_numerics": hashlib.sha256("\n".join(
        inspect.getsource(function) for function in functions).encode()).hexdigest()}


def _accuracy_comparison(prices, baseline):
    """Describe progress against the historical contract, never select a seed."""
    rows = []
    old = baseline.loc[baseline.stage.eq("initial")].set_index("contract_id")
    for row in prices.to_dict("records"):
        if row["contract_id"] not in old.index:
            continue
        previous = old.loc[row["contract_id"]]
        rows.append({"trial": row["trial"], "contract_id": row["contract_id"],
            "stage": row["stage"], "baseline_price": previous.price,
            "baseline_gap": previous.paired_gap, "paired_gap": row["paired_gap"],
            "gap_reduction_factor": previous.paired_gap/max(row["paired_gap"], np.finfo(float).tiny),
            "baseline_price_standard_error": previous.price_standard_error,
            "price_standard_error": row["price_standard_error"],
            "price_se_reduction_factor": previous.price_standard_error/max(row["price_standard_error"], np.finfo(float).tiny),
            "baseline_delta_standard_error": previous.delta_standard_error,
            "delta_standard_error": row["delta_standard_error"],
            "delta_se_reduction_factor": previous.delta_standard_error/max(row["delta_standard_error"], np.finfo(float).tiny),
            "price_gate_passed": row["discrete_price_gate_passed"],
            "delta_sampling_gate_passed": row["delta_sampling_precision_passed"],
            "baseline_source": "byte-verified unsuccessful rough_american_v1 campaign"})
    return pd.DataFrame(rows)


def rough_american_accuracy_study(context, *, seed=42, trial_specs=None,
                                  trial_indices=None, reuse_checkpoints=True,
                                  compute_figures=True, probe_archives=()):
    """Execute explicit independent train/value trials and retain all outcomes.

    Trial configurations must be supplied explicitly. This makes numerical
    budgets, estimators and ablations reviewable before the campaign starts.
    Conditional intervals do not cover policy construction, kernel, integration
    or exercise-calendar errors. None of these diagnostics certifies readiness.
    """
    import platform
    import scipy
    from rvlab.pricing import approximate_fractional_kernel, PricingError
    from ._pricing_rough_foundation import _checkpoint, _fingerprint, _jsonable, _market
    for name, value in (("reuse_checkpoints", reuse_checkpoints), ("compute_figures", compute_figures)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    design = _validated_design(trial_specs, seed)
    if trial_indices is not None and (not isinstance(trial_indices, (tuple, list)) or
            any(isinstance(value, bool) or not isinstance(value, Integral) or
                value < 0 or value >= len(design) for value in trial_indices) or
            len(set(trial_indices)) != len(trial_indices)):
        raise ValueError("trial_indices must be distinct valid predeclared indices")
    selected = set(range(len(design)) if trial_indices is None else trial_indices)
    historical = load_historical_american(compute_figures=False)
    probes, probe_records = load_accuracy_probes(probe_archives)
    models = _american_models()
    markets = {"no dividends": _market(), "cash dividend": _market(cash=True)}
    factors = sorted({row[3] for row in design})
    kernels = {n: approximate_fractional_kernel(.5 if n == 1 else .15, 1., n,
        **({} if n == 1 else {"min_time_ratio": .003 if n == 4 else .001, "max_nfev": 4000}))
        for n in factors}
    sources = _accuracy_sources()
    common = {"study_version": "rough_american_accuracy_v1", "seed": int(seed),
        "design": [{"contract": row[0], "trial": row[1], "stage": row[2],
            "factors": row[3], "config": asdict(row[4])} for row in design],
        "models": {name: asdict(value) for name, value in models.items()},
        "markets": {name: asdict(value) for name, value in markets.items()},
        "kernels": {str(n): asdict(value) for n, value in kernels.items()},
        "sources": sources, "preserved_probes": probe_records,
        "historical_archive_manifest_sha256": historical["manifest"]["historical_archive_manifest_sha256"],
        "acceptance": {"price_atol": PRICE_ATOL, "price_rtol": PRICE_RTOL,
            "delta_atol": DELTA_ATOL, "delta_rtol": DELTA_RTOL,
            "ci_multiplier": CONFIDENCE_MULTIPLIER, "effective_half_spread_fraction": .1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = _fingerprint(common)
    folder = Path(context.output_dir)/"rough_american_accuracy_v1"/fingerprint[:16]
    if not reuse_checkpoints:
        from datetime import datetime, timezone
        folder /= "attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots = _preserve_sources(folder, sources)
    # Preserve the follow-up's own executable numerical design as well.
    source = Path(__file__)
    payload = source.read_bytes()
    destination = folder/"source"/hashlib.sha256(payload).hexdigest()/source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError("Changed preserved rough American accuracy source")
    if not destination.exists():
        destination.write_bytes(payload)
    snapshots.append({"source": source.name, "sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot": str(destination), "bytes": len(payload)})
    provenance, failures, trials, rows, references = [], [], [], [], {}
    def checkpoint(label, specification, calculate):
        def guarded():
            expected = {k: v for k, v in sources.items() if k != "study_numerics"}
            before = _core_source_hashes()
            if before != expected:
                raise PricingError("Numerical source changed after the accuracy profile was constructed")
            try:
                result = calculate()
            except MemoryError as exc:
                result = {"status": "numerical_failure", "error": str(exc)}
            after = _core_source_hashes()
            if before != after:
                raise PricingError("Numerical source changed during the accuracy trial")
            result.update(source_after=after, source_changed_during_trial=False)
            return result
        record, origin = _checkpoint(folder, label, {**common, **specification}, guarded,
            reuse=reuse_checkpoints)
        provenance.append(origin)
        if record.get("status") == "numerical_failure":
            failures.append({"trial": label, "error": record["error"],
                "elapsed_seconds": origin["original_elapsed_seconds"]})
        return record
    unique_contracts = {row[0]["contract_id"]: row[0] for row in design}
    context.progress("American rough accuracy: independent limit references and predeclared trials")
    for contract_id, contract in unique_contracts.items():
        if contract["model"] not in ("constant variance", "Hurst .5"):
            continue
        record = checkpoint("reference_"+contract_id, {"reference": contract},
            lambda c=contract: _fd_reference(models[c["model"]], markets[c["scenario"]], c))
        if record.get("status") == "completed":
            references[contract_id] = record
    for index, (contract, label, stage, n, config) in enumerate(design):
        if index not in selected:
            continue
        context.progress(f"American rough accuracy {index+1}/{len(design)}: {label}")
        record = checkpoint(label, {"trial_index": index, "contract": contract, "config": asdict(config)},
            lambda c=contract, cfg=config, factors=n: _stopping_trial(models[c["model"]],
                markets[c["scenario"]], kernels[factors], c, cfg))
        origin = provenance[-1]
        trials.append({"trial": label, "stage": stage, "contract_id": contract["contract_id"],
            "status": record.get("status"), "factors": n, **asdict(config),
            "reused": origin["reused"], "elapsed_seconds": origin["original_elapsed_seconds"],
            "checkpoint": origin["checkpoint"]})
        if record.get("status") == "completed":
            rows.append(_stopping_rows(record, contract, label=label, stage=stage, factors=n,
                elapsed_seconds=origin["original_elapsed_seconds"],
                reference=references.get(contract["contract_id"])))
    prices = pd.DataFrame(rows)
    if prices.empty:
        prices = pd.DataFrame(columns=["contract_id", "stage", "price", "delta",
            "paired_gap", "price_standard_error", "delta_standard_error",
            "discrete_price_gate_passed", "delta_sampling_precision_passed"])
    baseline = historical["tables"]["prices"]
    summary = pd.DataFrame([{"completed_trials": len(prices), "planned_trials": len(design),
        "selected_trials": len(selected), "failures": len(failures),
        "discrete_price_passes": int(prices.discrete_price_gate_passed.sum()),
        "delta_sampling_passes": int(prices.delta_sampling_precision_passed.sum()),
        "american_pricing_ready": False, "stock_delta_ready": False}])
    tables = {"prices": prices, "trials": pd.DataFrame(trials), "probes": probes,
        "comparison": _accuracy_comparison(prices, baseline), "summary": summary,
        "references": pd.DataFrame([{"contract_id": key, **value} for key, value in references.items()]),
        "failures": pd.DataFrame(failures), "provenance": pd.DataFrame(provenance)}
    manifest = {**_jsonable(common), "backend": "rough_heston_lifted_primal_dual",
        "profile_fingerprint": fingerprint, "checkpoint_dir": str(folder),
        "source_snapshots": snapshots, "current_engine_execution": True,
        "source_elapsed_seconds": sum(row["original_elapsed_seconds"] for row in provenance),
        "timing_interpretation": "Cumulative original checkpoint wall runtimes; concurrent work and replay have separate costs.",
        "checkpoint_replay": bool(provenance) and all(row["reused"] for row in provenance),
        "planned_trial_count": len(design), "selected_trial_indices": sorted(selected),
        "partial_campaign": len(selected) != len(design),
        "independent_training_valuation": True, "american_pricing_ready": False, "stock_delta_ready": False,
        "limitations": [
            "Primal/dual expectations concern the finite kernel, discrete transition and declared exercise dates.",
            "Conditional sampling intervals exclude policy-training, kernel, time and exercise-calendar error.",
            "Fourier-premium surrogates clip residuals to [0, premium_cap]; this stabilizes a function, not sampled prices.",
            "The continuous-European surrogate can differ from the discrete European expectation; signed-residual exploratory trials did not remove the stopping-gap floor.",
            "Price bounds do not bound optimal-value stock deltas; independent sensitivity checks remain necessary.",
            "Synthetic inception studies do not supply historical states, independent SPY stock or cash expectations."]}
    return {"tables": tables, "figures": _accuracy_figures(tables) if compute_figures else {},
        "manifest": manifest}


def _accuracy_figures(tables):
    import matplotlib.pyplot as plt
    prices = tables["prices"]
    if prices.empty:
        return {}
    figures = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(4.4, .3*len(prices)+1.3)), constrained_layout=True)
    y = np.arange(len(prices))
    axes[0].barh(y-.17, prices.price_ci_half_width/prices.price_allowance,
        height=.3, label="Primal sampling half-width")
    axes[0].barh(y+.17, prices.paired_gap_upper/prices.price_allowance,
        height=.3, label="Stopping-gap upper diagnostic")
    axes[0].set_yticks(y, prices.trial)
    axes[0].set(xlabel="Unchanged price allowance units", title="Stopping gap and sampling are separate")
    axes[1].barh(y, prices.delta_ci_half_width/prices.delta_allowance, color="#268577")
    axes[1].set_yticks(y, prices.trial)
    axes[1].set(xlabel="Unchanged delta allowance units", title="Paired stock-delta uncertainty")
    for axis in axes:
        axis.axvline(1., color="#B64949", linestyle="--")
        axis.set_xscale("symlog", linthresh=1.)
        axis.invert_yaxis()
    axes[0].legend(fontsize=8)
    figures["accuracy_budgets"] = fig
    comparison = tables["comparison"]
    if not comparison.empty:
        central = comparison.loc[comparison.contract_id.eq("rough_put_60_100")]
        if not central.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11.5, max(3.8, .3*len(central)+1.3)), constrained_layout=True)
            y = np.arange(len(central))
            axes[0].barh(y-.17, central.baseline_gap, height=.3, color="#A4ADB4", label="Preserved baseline")
            axes[0].barh(y+.17, central.paired_gap, height=.3, color="#2878A5", label="Follow-up")
            axes[1].barh(y-.17, CONFIDENCE_MULTIPLIER*central.baseline_price_standard_error,
                height=.3, color="#A4ADB4", label="Preserved baseline")
            axes[1].barh(y+.17, CONFIDENCE_MULTIPLIER*central.price_standard_error,
                height=.3, color="#268577", label="Follow-up")
            for axis in axes:
                axis.set_yticks(y, central.trial)
                axis.invert_yaxis()
                axis.set_xscale("symlog", linthresh=.001)
            axes[0].set(title="60-day rough put: stopping gap", xlabel="Paired gap ($ per unit)")
            axes[1].set(title="60-day rough put: valuation uncertainty", xlabel="Primal 95% half-width ($ per unit)")
            axes[0].legend(fontsize=8)
            figures["progress_from_baseline"] = fig
    if "price_reference" in prices:
        limits = prices.loc[np.isfinite(prices.price_reference)]
        if not limits.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, max(3.7, .35*len(limits)+1.3)), constrained_layout=True)
            y = np.arange(len(limits))
            axes[0].barh(y, (limits.price_reference_absolute_error+limits.price_ci_half_width)/(PRICE_ATOL+PRICE_RTOL*abs(limits.price_reference)),
                color="#2878A5")
            axes[1].barh(y, (limits.delta_reference_absolute_error+limits.delta_ci_half_width)/limits.delta_allowance,
                color="#268577")
            for axis in axes:
                axis.set_yticks(y, limits.trial)
                axis.invert_yaxis()
                axis.axvline(1., color="#B64949", linestyle="--")
                axis.set_xscale("symlog", linthresh=1.)
                axis.set_xlabel("Independent-reference error + uncertainty / allowance")
            axes[0].set_title("BS/Heston controls: price")
            axes[1].set_title("BS/Heston controls: stock delta")
            figures["independent_limits"] = fig
    return figures
