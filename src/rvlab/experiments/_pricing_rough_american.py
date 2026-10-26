"""Independent-policy rough-Heston stopping experiments and honest diagnostics.

The historical European precision controls remain byte-verified evidence. They
are not recomputed or relabelled as American results when this study runs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd


PRICE_ATOL = .001
PRICE_RTOL = 1e-4
DELTA_ATOL = .002
DELTA_RTOL = .005
CONFIDENCE_MULTIPLIER = 1.96


PRECISION_ARCHIVE = (Path(__file__).resolve().parents[3] /
    "notebooks/legacy/pricing/rough_american_2026-10-18")
AMERICAN_BASELINE_ARCHIVE = (Path(__file__).resolve().parents[3] /
    "notebooks/legacy/pricing/rough_american_accuracy_2026-10-18")


def _load_historical_stage(stage, backend, context=None, *, archive=None, compute_figures=True):
    from ._pricing_rough_precision import _read_verified_archive
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = PRECISION_ARCHIVE if archive is None else Path(archive)
    inventory, fingerprint = _read_verified_archive(archive)
    recorded = json.loads((archive / "results/manifest.json").read_text())
    if (recorded.get("evidence_version") != "pricing_research_v1" or
            recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get(stage, {}).get("backend") != backend):
        raise ValueError(f"Preserved results are not the {'European ' if stage != 'rough_american' else ''}{stage} study")
    prefix = stage + "_"
    tables = {}
    for name, rows in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        path = archive / "results" / f"{name}.csv"
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != rows:
            raise ValueError(f"Historical table row count changed: {path}")
        tables[name.removeprefix(prefix)] = table
    figures = {name.removeprefix(prefix): archive / "results" / f"{name}.png"
        for name in recorded["figures"] if name.startswith(prefix)} if compute_figures else {}
    manifest = deepcopy(recorded[stage])
    manifest.update({"historical_replay": True,
        "historical_archive": str(archive.resolve()),
        "historical_archive_manifest_sha256": fingerprint,
        "historical_archive_files": len(inventory["files"]),
        "historical_execution_created_at": recorded["created_at"],
        "current_engine_execution": False,
        "american_pricing_ready": False, "stock_delta_ready": False})
    if context is not None:
        context.progress(f"Loading byte-verified {stage} controls; no historical trial is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}


def load_historical_precision(context=None, *, archive=None, compute_figures=True):
    """Read the preserved European precision campaign with its original identity.

    Every archived source, raw trial and figure is checked before any result is
    exposed. Loading recorded prices is not a new numerical execution.
    """
    return _load_historical_stage("rough_precision", "rough_heston_simulation_precision",
        context, archive=archive, compute_figures=compute_figures)


def load_historical_foundation(context=None, *, archive=None, compute_figures=True):
    """Replay the original foundation through the pre-American preservation set."""
    return _load_historical_stage("rough_foundation", "rough_heston_lifted_simulation_foundation",
        context, archive=archive, compute_figures=compute_figures)


def load_historical_american(context=None, *, archive=None, compute_figures=True):
    """Read the unsuccessful stopping baseline under its original source identity.

    The entire preserved notebook/result/source bundle is verified before loading
    any trial. No baseline price is repriced or promoted by a subsequent repair.
    """
    return _load_historical_stage("rough_american", "rough_heston_lifted_primal_dual",
        context, archive=AMERICAN_BASELINE_ARCHIVE if archive is None else archive,
        compute_figures=compute_figures)


def stopping_accuracy(prices, standard_errors, dual_prices,
                      paired_gaps, paired_gap_standard_errors):
    """Separate sampling precision and the estimated discrete stopping gap.

    Primal and dual share valuation paths, so the paired gap standard error is
    used rather than treating them as independent estimates. These are normal
    marginal sampling diagnostics, not simultaneous confidence bounds or a
    certificate for the continuous-exercise fractional model. Time, exercise
    dates, kernel and policy/delta checks remain separate.
    """
    import numpy as np
    arrays = np.broadcast_arrays(*[np.asarray(value, float) for value in
        (prices, standard_errors, dual_prices, paired_gaps, paired_gap_standard_errors)])
    price, se, dual, gap, gap_se = arrays
    if (not all(np.isfinite(a).all() for a in arrays) or
            np.any(se < 0) or np.any(gap_se < 0)):
        raise ValueError("Stopping estimates must be finite with nonnegative standard errors")
    if not np.allclose(dual-price, gap, rtol=1e-8, atol=1e-10):
        raise ValueError("Paired gaps must equal dual minus primal prices")
    allowance = PRICE_ATOL + PRICE_RTOL*np.abs(price)
    sampling = CONFIDENCE_MULTIPLIER*se
    gap_upper = gap + CONFIDENCE_MULTIPLIER*gap_se
    conservative = sampling + np.maximum(gap_upper, 0.)
    consistent = gap_upper >= 0.
    return {"price_allowance": allowance,
        "price_ci_half_width": sampling,
        "paired_gap_ci_half_width": CONFIDENCE_MULTIPLIER*gap_se,
        "paired_gap_upper": gap_upper,
        "dual_consistency_passed": consistent,
        "sampling_precision_passed": sampling <= allowance,
        "discrete_stopping_gap_passed": (gap_upper <= allowance) & consistent,
        "discrete_price_error_budget": conservative,
        "discrete_price_gate_passed": (conservative <= allowance) & consistent}


def delta_accuracy(deltas, standard_errors, *, reference=None,
                   reference_stable=False):
    """Keep stock-delta sampling and independent-reference checks explicit."""
    import numpy as np
    values, errors = np.broadcast_arrays(np.asarray(deltas, float),
        np.asarray(standard_errors, float))
    if not np.isfinite(values+errors).all() or np.any(errors < 0):
        raise ValueError("Delta estimates must be finite with nonnegative standard errors")
    target = values if reference is None else np.broadcast_to(np.asarray(reference, float), values.shape)
    if not np.isfinite(target).all():
        raise ValueError("Delta references must be finite")
    allowance = DELTA_ATOL + DELTA_RTOL*np.abs(target)
    half_width = CONFIDENCE_MULTIPLIER*errors
    discrepancy = np.abs(values-target)
    return {"delta_allowance": allowance, "delta_ci_half_width": half_width,
        "delta_sampling_precision_passed": half_width <= allowance,
        "delta_reference_absolute_error": discrepancy if reference is not None else np.full_like(values, np.nan),
        "delta_reference_gate_passed": (discrepancy+half_width <= allowance) &
            bool(reference is not None and reference_stable)}


def american_contracts():
    """Predeclared synthetic payoffs; market cases never substitute SPY inputs."""
    rows = [{"contract_id": f"rough_put_{days}_{int(strike)}", "model": "flat",
        "scenario": "no dividends", "days": days, "maturity": days/365,
        "strike": strike, "is_call": False}
        for days in (14, 30, 60) for strike in (95., 100., 105.)]
    rows += [{"contract_id": f"curve_put_{days}_100", "model": "curve",
        "scenario": "no dividends", "days": days, "maturity": days/365,
        "strike": 100., "is_call": False} for days in (14, 30, 60)]
    rows += [{"contract_id": label, "model": model,
        "scenario": "cash dividend" if model == "cash" else "no dividends",
        "days": 60, "maturity": 60/365, "strike": strike,
        "is_call": model == "cash"}
        for label, model, strike in (("heston_limit_put", "Hurst .5", 100.),
            ("constant_variance_put", "constant variance", 100.),
            ("cash_call", "cash", 95.))]
    return pd.DataFrame(rows)


def _american_models():
    from dataclasses import replace
    from ._pricing_rough_foundation import _truths
    models = _truths()
    models["Hurst .5"] = replace(models["flat"], hurst=.5)
    models["constant variance"] = replace(models["flat"], theta=.04, vol_of_vol=0.)
    models["cash"] = models["flat"]
    return models


def _stopping_trial(params, market, kernel, contract, config):
    from dataclasses import asdict
    import numpy as np
    from rvlab.pricing.rough_stopping import estimate_rough_american
    result = estimate_rough_american(params, market, kernel,
        strike=contract["strike"], maturity=contract["maturity"],
        is_call=contract["is_call"], config=config)
    names = ("prices", "primal_standard_errors", "dual_prices", "dual_standard_errors",
        "paired_gaps", "paired_gap_standard_errors", "deltas", "delta_standard_errors",
        "one_sided_deltas", "one_sided_delta_standard_errors", "bump_deltas",
        "bump_delta_standard_errors", "retrained_deltas", "retrained_delta_standard_errors",
        "raw_prices", "raw_standard_errors", "exercise_probabilities")
    return {"status": "completed", **{name: np.asarray(getattr(result, name)).reshape(-1).tolist()
        for name in names}, "settings": asdict(config),
        "diagnostics": dict(result.diagnostics)}


def _fd_reference(params, market, contract):
    """Independent BS/Heston PDE controls with unchanged per-axis checks."""
    import numpy as np
    from dataclasses import asdict
    from rvlab.pricing import BlackScholesParams, HestonParams, FDConfig, evaluate_vanilla
    if params.vol_of_vol == 0 and params.v0 == params.theta:
        reference_params = BlackScholesParams(np.sqrt(params.v0))
    elif params.hurst == .5:
        reference_params = HestonParams(params.v0, params.theta, params.kappa,
            params.vol_of_vol, params.rho)
    else:
        raise ValueError("An independent American PDE reference is only available for the declared limits")
    result = evaluate_vanilla(reference_params, market=market,
        strike=contract["strike"], maturity=contract["maturity"],
        is_call=contract["is_call"], exercise_style="american",
        config=FDConfig(240, 80, 300), check_refinement=True)
    return {"status": "completed", "price": float(np.asarray(result.prices)),
        "delta": float(np.asarray(result.deltas)), "price_stable": bool(result.price_stable),
        "delta_stable": bool(result.delta_stable), "settings": asdict(result.config),
        "diagnostics": _table_records(dict(result.diagnostics))}


def _table_records(value):
    """Make nested PDE diagnostics durable without dropping refinement rows."""
    from collections.abc import Mapping
    if isinstance(value, pd.DataFrame):
        return _table_records(value.to_dict("records"))
    if isinstance(value, Mapping):
        return {key: _table_records(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_table_records(item) for item in value]
    return value


def _stopping_rows(record, contract, *, label, stage, factors, elapsed_seconds,
                   reference=None):
    import numpy as np
    price, se, dual, dual_se, gap, gap_se, delta, delta_se = (
        float(np.asarray(record[name]).item()) for name in
        ("prices", "primal_standard_errors", "dual_prices", "dual_standard_errors",
         "paired_gaps", "paired_gap_standard_errors", "deltas", "delta_standard_errors"))
    price_checks = stopping_accuracy(price, se, dual, gap, gap_se)
    delta_checks = delta_accuracy(delta, delta_se,
        reference=None if reference is None else reference["delta"],
        reference_stable=False if reference is None else reference["delta_stable"])
    settings = record["settings"]
    row = {**contract, "trial": label, "stage": stage, "factors": factors,
        "price": price, "price_standard_error": se, "dual_price": dual,
        "dual_standard_error": dual_se, "paired_gap": gap,
        "paired_gap_standard_error": gap_se, "delta": delta,
        "delta_standard_error": delta_se,
        "elapsed_seconds": elapsed_seconds, **settings,
        **{k: float(v) if np.asarray(v).dtype.kind == "f" else bool(v)
           for k, v in {**price_checks, **delta_checks}.items()},
        "price_reference": np.nan if reference is None else reference["price"],
        "price_reference_stable": False if reference is None else reference["price_stable"],
        "american_accuracy_certified": False, "stock_delta_ready": False}
    for name in ("raw_prices", "raw_standard_errors", "exercise_probabilities",
                 "retrained_deltas", "retrained_delta_standard_errors"):
        value = np.asarray(record[name]).item()
        row[name] = np.nan if value is None else float(value)
    for name in ("bump_deltas", "bump_delta_standard_errors", "one_sided_deltas",
                 "one_sided_delta_standard_errors"):
        for suffix, value in zip(("first", "second"), record[name]):
            row[f"{name}_{suffix}"] = value
    row["bump_delta_change"] = abs(record["bump_deltas"][0]-record["bump_deltas"][1])
    row["bump_change_passed"] = row["bump_delta_change"] <= row["delta_allowance"]
    if reference is not None:
        from ._pricing_rough_foundation import price_gate
        gate = price_gate(price, reference["price"], standard_error=se)
        row["price_reference_absolute_error"] = float(gate["absolute_error"])
        row["price_reference_gate_passed"] = bool(gate["accepted"] and reference["price_stable"])
    else:
        row["price_reference_absolute_error"] = np.nan
        row["price_reference_gate_passed"] = False
    return row


def _american_sources():
    import inspect
    folder = Path(__file__).parents[1] / "pricing"
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.glob("*.py")) if p.name != "__init__.py"}
    functions = (stopping_accuracy, delta_accuracy, american_contracts,
        _american_models, _stopping_trial, _fd_reference, _stopping_rows,
        rough_american_study, _american_design, _refinement_rows, _table_records)
    sources["study_numerics"] = hashlib.sha256("\n".join(
        inspect.getsource(function) for function in functions).encode()).hexdigest()
    return sources


def _core_source_hashes():
    folder = Path(__file__).parents[1] / "pricing"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.glob("*.py")) if p.name != "__init__.py"}


def _preserve_sources(folder, sources):
    root = Path(__file__).parents[3]
    paths = [Path(__file__), *sorted((Path(__file__).parents[1] / "pricing").glob("*.py"))]
    records = []
    for source in paths:
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if source.name in sources and digest != sources[source.name]:
            raise ValueError(f"Numerical source changed during American rough-Heston startup: {source}")
        target = Path(folder) / "source" / digest / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Changed preserved American rough-Heston source: {target}")
        if not target.exists():
            import tempfile
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            temporary.replace(target)
        records.append({"source": str(source.relative_to(root)), "sha256": digest,
            "snapshot": str(target), "bytes": len(payload)})
    return records


def _preserved_trials(output_dir):
    """Load failed naive investigations only after checking every saved byte."""
    from ._pricing_rough_precision import _read_verified_archive
    output_dir = Path(output_dir)
    tables, records = {}, []
    for directory, table_name, description in (
            ("initial_64", "naive_pilot", "naive least-squares pilot"),
            ("conditional_64", "unbounded_conditional_pilot", "failed unbounded conditional Bellman pilot"),
            ("bounded_64", "bounded_pilot", "superseded initial cap-integrator Bellman pilot")):
        pilot = output_dir / "rough_american_v1/pilots" / directory
        if not (pilot / "preservation_manifest.json").is_file():
            continue
        inventory, digest = _read_verified_archive(pilot)
        manifest = json.loads((pilot / "manifest.json").read_text())
        if manifest.get("backend") != "rough_heston_lifted_primal_dual":
            raise ValueError("Preserved naive pilot has the wrong backend")
        prices = pd.read_csv(pilot / "prices.csv")
        if len(prices) != manifest["tables"]["prices"]:
            raise ValueError("Preserved naive pilot row count changed")
        prices["training_method"] = manifest["base_config"].get("training_method", "least_squares")
        prices["historical_replay"] = True
        tables[table_name] = prices
        records.append({"kind": description, "folder": str(pilot),
            "inventory_sha256": digest, "files": len(inventory["files"]),
            "source_elapsed_seconds": manifest["source_elapsed_seconds"],
            "profile_fingerprint": manifest["profile_fingerprint"]})
    for directory, table_name, description in (
            ("rough_american_controls_v1", "naive_training_trials", "naive training/ridge/ITM trials"),
            ("rough_american_controls_v2", "unbounded_ablation", "same-loaded-source unbounded Bellman ablation"),
            ("rough_american_controls_v3", "bounded_ablation", "superseded initial cap-integrator training ablation")):
        controls = output_dir / directory
        if not (controls / "preservation_manifest.json").is_file():
            continue
        inventory, digest = _read_verified_archive(controls)
        raw = json.loads((controls / "controls_audit.json").read_text())
        table = pd.read_csv(controls / "controls_audit.csv")
        if len(table) != len(raw["trials"]):
            raise ValueError("Preserved training controls are incomplete")
        if "reference_archive_sha256" in raw:
            reference = output_dir / "rough_american_controls_v1/preservation_manifest.json"
            if hashlib.sha256(reference.read_bytes()).hexdigest() != raw["reference_archive_sha256"]:
                raise ValueError("Preserved ablation's independent reference archive changed")
        table["historical_replay"] = True
        tables[table_name] = table
        records.append({"kind": description, "folder": str(controls),
            "inventory_sha256": digest, "files": len(inventory["files"]),
            "on_disk_source_change_flag": bool(table.source_changed_during_trial.any()),
            "source_elapsed_seconds": float(table.elapsed_seconds.sum())})
    trajectory = []
    for name, label in (("naive_pilot", "Naive cash-flow least squares"),
            ("unbounded_conditional_pilot", "Failed unbounded Bellman"),
            ("bounded_pilot", "Superseded initial bounded Bellman")):
        if name in tables:
            selected = tables[name].loc[tables[name].contract_id.eq("rough_put_60_100")].copy()
            selected["training_trial"] = label
            trajectory.append(selected)
    if trajectory:
        tables["pilot_trajectory"] = pd.concat(trajectory, ignore_index=True, sort=False)
    guard = output_dir / "rough_american_v1/superseded_precision_guard"
    if (guard / "preservation_manifest.json").is_file():
        inventory, digest = _read_verified_archive(guard)
        manifest = json.loads((guard / "aggregate/manifest.json").read_text())
        if manifest.get("backend") != "rough_heston_lifted_primal_dual":
            raise ValueError("Preserved precision-guard baseline has the wrong backend")
        for name in ("prices", "trials", "failures"):
            table = pd.read_csv(guard / "aggregate" / f"{name}.csv", float_precision="round_trip")
            if len(table) != manifest["tables"][name]:
                raise ValueError("Preserved precision-guard baseline has changed table counts")
            table["historical_replay"] = True
            tables["precision_guard_" + name] = table
        records.append({"kind": "superseded moment-cancellation guard baseline",
            "folder": str(guard), "files": len(inventory["files"]),
            "inventory_sha256": digest, "profile_fingerprint": manifest["profile_fingerprint"],
            "source_elapsed_seconds": manifest["source_elapsed_seconds"]})
    replay = output_dir / "rough_american_precision_repair_2026-10-18"
    if (replay / "preservation_manifest.json").is_file():
        inventory, digest = _read_verified_archive(replay)
        tables["precision_repair_replay"] = pd.read_csv(replay / "failed_case_replay.csv", float_precision="round_trip")
        raw = json.loads((replay / "failed_case_replay.json").read_text())
        records.append({"kind": "previously failed ordinary contract after analytic precision repair",
            "folder": str(replay), "files": len(inventory["files"]),
            "inventory_sha256": digest, "decimal_fallback_calls": raw["decimal_fallback_calls"],
            "maximum_certified_integral_error": raw["maximum_certified_integral_error"],
            "source_elapsed_seconds": raw["elapsed_seconds"]})
    return tables, records


def arithmetic_repair_comparison(current, previous):
    """A display-only same-trial comparison, after independent full repricing."""
    import numpy as np
    required = {"trial", "training_seed", "valuation_seed", "training_paths",
        "valuation_paths", "steps", "exercise_steps", "factors", "training_method",
        "bounded_surrogate", "ridge", "bump_relative", "retrain_bumps", "stock_integral",
        "model", "scenario", "maturity", "strike", "is_call", "raw_prices", "price", "delta"}
    columns = ["trial", "same_settings", "raw_payoff_bit_identical",
        "price_change", "delta_change"]
    if current.empty or previous.empty or not required.issubset(current) or not required.issubset(previous):
        return pd.DataFrame(columns=columns), pd.DataFrame()
    settings = sorted(required - {"trial", "raw_prices", "price", "delta"})
    matched = current.merge(previous, on="trial", suffixes=("_current", "_previous"), validate="one_to_one")
    rows = []
    for item in matched.to_dict("records"):
        same = all(item[key+"_current"] == item[key+"_previous"] for key in settings)
        left = np.asarray(item["raw_prices_current"], dtype=np.float64)
        right = np.asarray(item["raw_prices_previous"], dtype=np.float64)
        rows.append({"trial": item["trial"], "same_settings": same,
            "raw_payoff_bit_identical": bool(left.view(np.uint64) == right.view(np.uint64)),
            "price_change": abs(item["price_current"]-item["price_previous"]),
            "delta_change": abs(item["delta_current"]-item["delta_previous"])})
    detailed = pd.DataFrame(rows, columns=columns)
    summary = pd.DataFrame([{"matched_trials": len(detailed),
        "same_settings": bool(detailed.same_settings.all()),
        "raw_payoff_bit_identical": bool(detailed.raw_payoff_bit_identical.all()),
        "max_price_change": detailed.price_change.max(),
        "max_delta_change": detailed.delta_change.max(),
        "newly_finite_trials": len(current)-len(detailed),
        "comparison_scope": "Matched training/valuation seeds, paths, numerical settings and model contracts; no model parameters or tolerances changed."}])
    return detailed, summary


def _american_design(contracts, base, *, replications, run_refinements):
    """A fixed design shared by sequential runs and disjoint cache prefills."""
    from dataclasses import replace
    design = []
    for contract in contracts.to_dict("records"):
        design.append((contract, "initial", 0, base, 1 if contract["model"] == "Hurst .5" else 4))
    central_ids = {"rough_put_60_100", "heston_limit_put", "constant_variance_put"}
    for contract in contracts.loc[contracts.contract_id.isin(central_ids)].to_dict("records"):
        for repeat in range(1, replications):
            design.append((contract, "independent policy", repeat, base,
                1 if contract["model"] == "Hurst .5" else 4))
    central = contracts.loc[contracts.contract_id.eq("rough_put_60_100")]
    if run_refinements and not central.empty:
        contract = central.iloc[0].to_dict()
        for axis in ("time", "exercise", "training", "sampling", "bump", "joint"):
            design.append((contract, axis + " refinement", 0, base.refined(axis), 4))
        design.append((contract, "kernel refinement", 0, base.refined("time"), 8))
        design.append((contract, "retrained bump policy", 0, replace(base, retrain_bumps=True), 4))
    return design


def _campaign_execution(output_dir, profile):
    """Verify durable execution runners and report partial workers as partial."""
    folder = Path(output_dir) / "rough_american_v1"
    runner = folder / "run_main_campaign.py"
    if not runner.is_file():
        return []
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    rows = []
    for path in sorted((folder / "campaign_v1").glob("*/execution.json")):
        record = json.loads(path.read_text())
        if record.get("runner_sha256") != digest:
            raise ValueError(f"Campaign runner changed after execution: {path}")
        recorded_profile = record.get("profile_fingerprint")
        rows.append({"mode": record["mode"], "status": record["status"],
            "started_at": record.get("started_at"), "completed_at": record.get("created_at"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "selected_trials": None if record.get("selected_trial_indices") is None else len(record["selected_trial_indices"]),
            "same_profile": recorded_profile == profile,
            "profile_fingerprint": recorded_profile, "runner": str(runner),
            "runner_sha256": digest, "record": str(path)})
    return rows


def rough_american_study(context, *, seed=42, training_paths=2048,
                         valuation_paths=1024, steps=128, exercise_steps=32,
                         replications=2, run_refinements=True, contract_ids=None,
                         training_method="least_squares", bounded_surrogate=True,
                         trial_indices=None,
                         reuse_checkpoints=True, compute_figures=True):
    """Run bounded independent-policy stopping trials, retaining every outcome.

    All configured trials enter the ledger. Independent training replications
    are displayed individually: conditional valuation standard errors do not
    include uncertainty in the fitted policy. No favorable seed or held-out
    option price selects a policy. Refinement acceptance is diagnostic, not a
    rigorous continuous-time fractional-model error bound.
    """
    from dataclasses import asdict, replace
    from numbers import Integral
    import numpy as np
    import platform
    import scipy
    from rvlab.pricing import approximate_fractional_kernel
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    from ._pricing_rough_foundation import _checkpoint, _fingerprint, _jsonable, _market
    for name, value, minimum in (("seed", seed, 0), ("training_paths", training_paths, 2),
            ("valuation_paths", valuation_paths, 2), ("steps", steps, 1),
            ("exercise_steps", exercise_steps, 1), ("replications", replications, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("run_refinements", run_refinements),
            ("reuse_checkpoints", reuse_checkpoints), ("compute_figures", compute_figures),
            ("bounded_surrogate", bounded_surrogate)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if training_method not in ("least_squares", "conditional"):
        raise ValueError("training_method must be least_squares or conditional")
    contracts = american_contracts()
    if contract_ids is not None:
        if (not isinstance(contract_ids, (tuple, list)) or not contract_ids or
                any(not isinstance(value, str) for value in contract_ids) or
                len(set(contract_ids)) != len(contract_ids) or
                not set(contract_ids).issubset(set(contracts.contract_id))):
            raise ValueError("contract_ids must contain distinct declared contract identifiers")
        contracts = contracts.set_index("contract_id").loc[list(contract_ids)].reset_index()
    models = _american_models()
    markets = {"no dividends": _market(), "cash dividend": _market(cash=True)}
    kernels = {4: approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003),
        1: approximate_fractional_kernel(.5, 1., 1)}
    if run_refinements:
        kernels[8] = approximate_fractional_kernel(.15, 1., 8, min_time_ratio=.001, max_nfev=4000)
    base = RoughAmericanConfig(training_paths=training_paths, valuation_paths=valuation_paths,
        steps=steps, exercise_steps=exercise_steps, training_seed=seed,
        valuation_seed=seed+1_000_000, batch_size=min(1024, valuation_paths),
        training_method=training_method, bounded_surrogate=bounded_surrogate)
    design = _american_design(contracts, base, replications=replications,
        run_refinements=run_refinements)
    if trial_indices is not None and (not isinstance(trial_indices, (tuple, list)) or
            any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or value < 0 or value >= len(design) for value in trial_indices) or
            len(set(trial_indices)) != len(trial_indices)):
        raise ValueError("trial_indices must be distinct valid predeclared trial indices")
    selected = set(range(len(design)) if trial_indices is None else trial_indices)
    sources = _american_sources()
    common = {"study_version": "rough_american_v1", "seed": int(seed),
        "contracts": contracts.to_dict("records"), "base_config": asdict(base),
        "replications": int(replications), "run_refinements": run_refinements,
        "models": {name: asdict(value) for name, value in models.items()},
        "markets": {name: asdict(value) for name, value in markets.items()},
        "kernels": {str(n): asdict(value) for n, value in kernels.items()},
        "sources": sources,
        "acceptance": {"price_atol": PRICE_ATOL, "price_rtol": PRICE_RTOL,
            "delta_atol": DELTA_ATOL, "delta_rtol": DELTA_RTOL,
            "ci_multiplier": CONFIDENCE_MULTIPLIER, "effective_half_spread_fraction": .1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}}
    fingerprint = _fingerprint(common)
    folder = Path(context.output_dir) / "rough_american_v1" / fingerprint[:16]
    if not reuse_checkpoints:
        from datetime import datetime, timezone
        folder = folder / ("attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    snapshots = _preserve_sources(folder, sources)
    provenance, failures, trial_rows, rows, references = [], [], [], [], {}
    def checkpoint(label, specification, calculate):
        def guarded():
            from rvlab.pricing import PricingError
            before = _core_source_hashes()
            expected = {key: value for key, value in sources.items() if key != "study_numerics"}
            if before != expected:
                raise PricingError("Numerical source changed after the study profile was constructed; start a new fingerprinted attempt")
            try:
                result = calculate()
            except MemoryError as exc:
                result = {"status": "numerical_failure", "error": str(exc)}
            after = _core_source_hashes()
            result["source_after"] = after
            result["source_changed_during_trial"] = before != after
            return result
        record, origin = _checkpoint(folder, label, {**common, **specification},
            guarded, reuse=reuse_checkpoints)
        provenance.append(origin)
        if record.get("status") == "numerical_failure":
            failures.append({"trial": label, "error": record["error"],
                "elapsed_seconds": origin["original_elapsed_seconds"]})
        return record
    context.progress("American rough Heston: independent finite-difference limit references")
    for contract in contracts.loc[contracts.model.isin(["constant variance", "Hurst .5"])].to_dict("records"):
        label = "reference_" + contract["contract_id"]
        record = checkpoint(label, {"stage": "reference", "contract": contract},
            lambda c=contract: _fd_reference(models[c["model"]], markets[c["scenario"]], c))
        if record.get("status") == "completed":
            references[contract["contract_id"]] = record
    context.progress(f"American rough Heston: {len(selected)} selected of {len(design)} predeclared independent train/valuation trials")
    for index, (contract, stage, repeat, template, factors) in enumerate(design):
        if index not in selected:
            continue
        config = replace(template, training_seed=seed+1000*index,
            valuation_seed=seed+1_000_000+1000*index)
        label = f"{contract['contract_id']}_{stage.replace(' ', '_')}_{repeat}"
        specification = {"contract": contract, "stage": stage, "replication": repeat,
            "config": asdict(config), "factors": factors}
        record = checkpoint(label, specification,
            lambda c=contract, cfg=config, n=factors: _stopping_trial(
                models[c["model"]], markets[c["scenario"]], kernels[n], c, cfg))
        origin = provenance[-1]
        trial_rows.append({"trial": label, "contract_id": contract["contract_id"],
            "stage": stage, "replication": repeat, "status": record.get("status"),
            "factors": factors, **asdict(config), "reused": origin["reused"],
            "source_changed_during_trial": record.get("source_changed_during_trial"),
            "elapsed_seconds": origin["original_elapsed_seconds"], "checkpoint": origin["checkpoint"]})
        if record.get("status") == "completed":
            rows.append(_stopping_rows(record, contract, label=label, stage=stage,
                factors=factors, elapsed_seconds=origin["original_elapsed_seconds"],
                reference=references.get(contract["contract_id"])))
    prices = pd.DataFrame(rows)
    if prices.empty:
        prices = pd.DataFrame(columns=[*contracts.columns, "stage", "price", "dual_price",
            "paired_gap", "price_ci_half_width", "exercise_probabilities", "delta",
            "delta_ci_half_width", "bump_delta_change", "bump_change_passed", "retrained_deltas",
            "price_reference", "price_reference_stable", "price_reference_gate_passed",
            "delta_reference_gate_passed", "discrete_price_gate_passed"])
    refinement = _refinement_rows(prices)
    tables = {"prices": prices, "trials": pd.DataFrame(trial_rows),
        "contracts": contracts, "refinement": refinement,
        "references": pd.DataFrame([{"contract_id": key, **value} for key, value in references.items()]),
        "failures": pd.DataFrame(failures), "provenance": pd.DataFrame(provenance)}
    preserved_tables, preserved_provenance = _preserved_trials(context.output_dir)
    tables.update(preserved_tables)
    execution = _campaign_execution(context.output_dir, fingerprint)
    tables["execution"] = pd.DataFrame(execution)
    manifest = {**_jsonable(common), "backend": "rough_heston_lifted_primal_dual",
        "profile_fingerprint": fingerprint, "checkpoint_dir": str(folder),
        "source_snapshots": snapshots,
        "source_elapsed_seconds": sum(row["original_elapsed_seconds"] for row in provenance),
        "timing_interpretation": "Sum of original per-checkpoint wall runtimes. Concurrent prefills make this neither campaign wall time nor CPU time; cached replay is a separate execution.",
        "execution_evidence": execution,
        "checkpoint_replay": bool(provenance) and all(row["reused"] for row in provenance),
        "provenance": provenance, "independent_training_valuation": True,
        "planned_trial_count": len(design), "selected_trial_indices": sorted(selected),
        "partial_campaign": len(selected) != len(design),
        "completed_selected_trials": int(sum(row["status"] == "completed" for row in trial_rows)),
        "preserved_trials": preserved_provenance,
        "american_pricing_ready": False, "stock_delta_ready": False,
        "limitations": [
            "Primal and dual bound expectations only for the finite kernel, declared discrete transition and exercise dates.",
            "Conditional valuation standard errors exclude uncertainty in policy training; independent training trials are shown separately.",
            "Normal marginal sampling intervals are not simultaneous confidence bands and omit kernel/time/exercise-date error.",
            "Frozen-policy stock sensitivities and retrained-policy checks are not certified optimal-value deltas by the price dual bound.",
            "The inception-state synthetic study does not establish historical conditional-state, calibration or portfolio-replay readiness.",
            "Missing independent SPY stock and historically available dividend expectations remain calibration and hedging blockers."]}
    return {"tables": tables, "figures": _american_figures(tables) if compute_figures else {},
        "manifest": manifest}


def _refinement_rows(prices):
    """Independent-grid changes include both conditional sampling errors."""
    import numpy as np
    rows = []
    columns = ("trial", "stage", "price_change", "delta_change", "price_change_ci_half_width",
        "delta_change_ci_half_width", "price_allowance", "delta_allowance",
        "price_change_gate_passed", "delta_change_gate_passed", "independent_training_valuation",
        "paired_between_trials", "american_accuracy_certified", "stock_delta_ready")
    if prices.empty:
        return pd.DataFrame(columns=columns)
    base = prices.loc[prices.contract_id.eq("rough_put_60_100") & prices.stage.eq("initial")]
    if base.empty:
        return pd.DataFrame(columns=columns)
    base = base.iloc[0]
    candidates = prices.loc[prices.contract_id.eq("rough_put_60_100") & ~prices.stage.eq("initial")]
    for row in candidates.to_dict("records"):
        price_change = abs(row["price"]-base.price)
        delta_change = abs(row["delta"]-base.delta)
        price_uncertainty = CONFIDENCE_MULTIPLIER*np.hypot(row["price_standard_error"], base.price_standard_error)
        delta_uncertainty = CONFIDENCE_MULTIPLIER*np.hypot(row["delta_standard_error"], base.delta_standard_error)
        rows.append({"trial": row["trial"], "stage": row["stage"],
            "price_change": price_change, "delta_change": delta_change,
            "price_change_ci_half_width": price_uncertainty,
            "delta_change_ci_half_width": delta_uncertainty,
            "price_allowance": base.price_allowance, "delta_allowance": base.delta_allowance,
            "price_change_gate_passed": price_change+price_uncertainty <= base.price_allowance,
            "delta_change_gate_passed": delta_change+delta_uncertainty <= base.delta_allowance,
            "independent_training_valuation": True, "paired_between_trials": False,
            "american_accuracy_certified": False, "stock_delta_ready": False})
    return pd.DataFrame(rows, columns=columns)


def _american_figures(tables):
    import matplotlib.pyplot as plt
    import numpy as np
    figures = {}
    prices = tables["prices"]
    if prices.empty:
        return figures
    initial = prices.loc[prices.stage.eq("initial")]
    flat = initial.loc[initial.model.eq("flat")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    colors = {14: "#2878A5", 30: "#D17B32", 60: "#268577"}
    for days, group in flat.groupby("days", sort=True):
        group = group.sort_values("strike")
        color = colors.get(int(days), "#737B83")
        axes[0].errorbar(group.strike, group.price, yerr=group.price_ci_half_width,
            fmt="o-", capsize=3, color=color, label=f"{days}d primal")
        axes[0].errorbar(group.strike, group.dual_price,
            yerr=CONFIDENCE_MULTIPLIER*group.dual_standard_error,
            fmt="x--", capsize=3, color=color, alpha=.7, label=f"{days}d dual")
        axes[1].errorbar(group.strike, group.delta,
            yerr=group.delta_ci_half_width, fmt="o-", capsize=3, color=color, label=f"{days}d")
    axes[0].set(title="Stopping-policy values and dual upper estimates", xlabel="Put strike", ylabel="Price per unit ($)")
    axes[1].set(title="Paired stock-bump sensitivities", xlabel="Put strike", ylabel="Frozen-policy stock delta")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8)
    figures["prices_and_stock_deltas"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    central = prices.loc[prices.contract_id.eq("rough_put_60_100")]
    naive = tables.get("naive_pilot", pd.DataFrame())
    if not naive.empty:
        old = naive.loc[naive.contract_id.eq("rough_put_60_100")].assign(stage="naive LSM pilot (64 steps)")
        central = pd.concat([old, central], ignore_index=True, sort=False)
    unbounded = tables.get("unbounded_conditional_pilot", pd.DataFrame())
    if not unbounded.empty:
        old = unbounded.loc[unbounded.contract_id.eq("rough_put_60_100")].assign(stage="unbounded Bellman pilot (64 steps)")
        central = pd.concat([old, central], ignore_index=True, sort=False)
    bounded = tables.get("bounded_pilot", pd.DataFrame())
    if not bounded.empty:
        old = bounded.loc[bounded.contract_id.eq("rough_put_60_100")].assign(stage="initial bounded pilot (superseded)")
        central = pd.concat([old, central], ignore_index=True, sort=False)
    x = np.arange(len(central))
    axes[0].barh(x-.16, central.price_ci_half_width/central.price_allowance,
        height=.3, label="Primal sampling interval")
    axes[0].barh(x+.16, central.paired_gap_upper/central.price_allowance,
        height=.3, label="Paired stopping-gap upper diagnostic")
    axes[0].set_yticks(x, central.stage)
    axes[0].set(xlabel="Unchanged price allowance units", title="Sampling precision and policy quality are separate")
    axes[0].set_xscale("symlog", linthresh=1.)
    axes[0].axvline(1., color="#B64949", linestyle="--")
    refinements = tables["refinement"]
    if not refinements.empty:
        y = np.arange(len(refinements))
        axes[1].barh(y-.16, (refinements.price_change+refinements.price_change_ci_half_width)/refinements.price_allowance,
            height=.3, label="Price change + uncertainty")
        axes[1].barh(y+.16, (refinements.delta_change+refinements.delta_change_ci_half_width)/refinements.delta_allowance,
            height=.3, label="Delta change + uncertainty")
        axes[1].set_yticks(y, refinements.stage)
    axes[1].axvline(1., color="#B64949", linestyle="--")
    axes[1].set(xlabel="Unchanged numerical allowance units", title="Independent refinements can still fail")
    handles, labels = [], []
    for axis in axes:
        h, names = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(names)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.01),
            ncol=2, frameon=False, fontsize=8)
    figures["stopping_and_refinement"] = fig
    return figures
