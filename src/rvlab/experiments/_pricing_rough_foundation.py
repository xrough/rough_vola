"""Synthetic finite-factor rough-Heston foundations, before optimal stopping.

The study separates kernel approximation, fixed-lift time discretization and
Monte Carlo uncertainty.  A sampling interval overlapping a reference is not
an accuracy certificate.  All results are inception-state European controls.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from numbers import Integral
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.special import gamma

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DividendSchedule, DiscountCurve,
    HestonParams, MeanReversionCurve, PricingConfig, PricingError,
    RoughHestonCurveParams, RoughHestonParams, SpotMarket, price,
)

STUDY_VERSION = "rough_foundation_v1"
PRICE_ATOL = .001
PRICE_RTOL = 1e-4
CONFIDENCE_MULTIPLIER = 1.96
TENORS = (14/365, 60/365, 1.)


def price_gate(estimate, reference, *, standard_error=0.):
    """Conservative fixed-budget gate; never accept by CI overlap alone."""
    estimate, reference, se = np.broadcast_arrays(
        np.asarray(estimate, float), np.asarray(reference, float),
        np.asarray(standard_error, float))
    if not np.isfinite(estimate+reference+se).all() or (se < 0).any():
        raise ValueError("price estimates/references must be finite and standard errors nonnegative")
    allowance = PRICE_ATOL+PRICE_RTOL*np.abs(reference)
    error = np.abs(estimate-reference)
    uncertainty = CONFIDENCE_MULTIPLIER*se
    return {"absolute_error": error, "ci_half_width": uncertainty,
            "price_allowance": allowance,
            "conservative_error": error+uncertainty,
            "accepted": error+uncertainty <= allowance}


def sample_summary(samples):
    """Pathwise sampling uncertainty; independent replications are separate."""
    a = np.asarray(samples, float)
    if a.ndim < 1 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("samples require at least two finite independent paths")
    return np.mean(a, axis=0), np.std(a, axis=0, ddof=1)/np.sqrt(a.shape[0])


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fingerprint(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _engine_sources():
    folder = Path(__file__).parents[1]/"pricing"
    paths = [p for p in sorted(folder.glob("*.py")) if p.name != "__init__.py"]
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    functions = (price_gate, sample_summary, foundation_contracts, _market, _truths,
        _price_args, _price_records, _reference_prices, _lift_prices, _simulate_trial,
        _limit_controls)
    sources["study_numerics"] = hashlib.sha256("\n".join(
        inspect.getsource(fn) for fn in functions).encode()).hexdigest()
    return sources


def _checkpoint(folder, label, specification, calculate, *, reuse=True):
    """Reuse only completed, fingerprint-matched records; preserve originals."""
    fingerprint = _fingerprint(specification)
    path = Path(folder)/f"{label}_{fingerprint[:16]}.json"
    if reuse and path.is_file():
        record = json.loads(path.read_text())
        if (record.get("fingerprint") != fingerprint or
                _fingerprint(record.get("specification")) != fingerprint):
            raise ValueError(f"Corrupt foundation checkpoint: {path}")
        return record["payload"], {"label": label, "checkpoint": str(path),
            "fingerprint": fingerprint, "reused": True,
            "original_elapsed_seconds": record["elapsed_seconds"]}
    started = time.perf_counter()
    try:
        payload = calculate()
    except PricingError as exc:
        payload = {"status": "numerical_failure", "error": str(exc)}
    elapsed = time.perf_counter()-started
    record = {"fingerprint": fingerprint, "specification": _jsonable(specification),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed, "payload": _jsonable(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)
    return record["payload"], {"label": label, "checkpoint": str(path),
        "fingerprint": fingerprint, "reused": False, "original_elapsed_seconds": elapsed}


def foundation_contracts():
    return pd.DataFrame([{"maturity": t, "days": int(round(365*t)),
        "strike": k, "is_call": k >= 100.}
        for t in TENORS for k in (95., 100., 105.)])


def _market(*, cash=False):
    return SpotMarket(100., DiscountCurve.flat(.04, 1.),
        DividendSchedule((CashDividend(.08, 1.5),) if cash else (), 1.))


def _truths():
    return {"flat": RoughHestonParams(.04, .045, 1.5, .3, -.6, .15),
        "curve": RoughHestonCurveParams(.04, 1.5, .3, -.6,
            MeanReversionCurve((20/365, 45/365), (.035, .065, .045)), .15)}


def _price_args(contracts, market):
    if any(event.amount > 0 for event in market.dividends.events):
        raise ValueError("Fourier controls require an explicitly no-cash market; cash dividends use pathwise gains/payoffs")
    t = contracts.maturity.to_numpy()
    discount = market.discount_curve.discount(t)
    return {"forward": market.spot/discount, "discount": discount,
        "strike": contracts.strike.to_numpy(), "maturity": t,
        "is_call": contracts.is_call.to_numpy()}


def _price_records(contracts, values, reference, *, stage, model, factors,
                   steps, seed=None, standard_error=0., reference_stable=False):
    values = np.asarray(values, float)
    gate = price_gate(values, reference, standard_error=standard_error)
    result = contracts.assign(stage=stage, model=model, factors=factors,
        steps=steps, seed=seed, estimate=values, reference=np.asarray(reference),
        standard_error=np.broadcast_to(standard_error, values.shape),
        reference_stable=bool(reference_stable))
    for name, val in gate.items():
        result[name] = val
    result["accepted"] &= bool(reference_stable)
    return result.to_dict("records")


def _reference_prices(params, contracts, market, config):
    return {"status": "completed", "prices": np.asarray(price(
        params, **_price_args(contracts, market), config=config)).tolist()}


def _lift_prices(params, contracts, market, kernel, config):
    from rvlab.pricing.rough_lift import lifted_heston_price
    return {"status": "completed", "prices": np.asarray(lifted_heston_price(
        params, **_price_args(contracts, market), kernel=kernel,
        config=config)).tolist()}


def _simulate_trial(params, market, kernel, contracts, *, paths, steps, seed):
    from rvlab.pricing.rough_simulation import (
        RoughSimulationConfig, simulate_rough_heston, european_payoff_samples,
    )
    result = simulate_rough_heston(params, market, kernel,
        observation_times=TENORS,
        config=RoughSimulationConfig(paths=paths, steps=steps, seed=seed,
            stock_integral="trapezoidal"))
    means, errors = [], []
    conditional = not bool(market.dividends.events)
    for row in contracts.itertuples():
        samples = european_payoff_samples(result, strike=row.strike,
            maturity=row.maturity, is_call=row.is_call, conditional=conditional)
        mean, error = sample_summary(np.asarray(samples).reshape(paths))
        means.append(float(mean)); errors.append(float(error))
    gains, gains_se = sample_summary(result.discounted_gains)
    variances, variance_se = sample_summary(result.variances)
    return {"status": "completed", "prices": means, "standard_errors": errors,
        "gains_mean": gains.tolist(), "gains_standard_error": gains_se.tolist(),
        "variance_mean": variances.tolist(), "variance_standard_error": variance_se.tolist(),
        "diagnostics": dict(result.diagnostics),
        "payoff_estimator": "conditional Gaussian" if conditional else "raw payoff"}


def rough_foundation_study(context, *, seed=42, paths=16384, reuse_checkpoints=True,
                           compute_figures=True):
    """Run bounded kernel/European/simulation controls; no American stopping."""
    from rvlab.pricing.rough_lift import (
        approximate_fractional_kernel, lifted_forward_variance,
    )
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(paths, bool) or not isinstance(paths, Integral) or paths < 2:
        raise ValueError("paths must be an integer >= 2")
    if not isinstance(reuse_checkpoints, bool):
        raise TypeError("reuse_checkpoints must be boolean")
    contracts, truths, market = foundation_contracts(), _truths(), _market()
    config = PricingConfig(192, 160., 512)
    finest = PricingConfig(384, 320., 1024)
    common = {"study_version": STUDY_VERSION, "seed": int(seed), "paths": int(paths),
        "contracts": contracts.to_dict("records"), "truths": {k: asdict(p) for k,p in truths.items()},
        "market": asdict(market), "base_fourier": asdict(config),
        "reference_fourier": asdict(finest), "sources": _engine_sources(),
        "acceptance": {"price_atol": PRICE_ATOL, "price_rtol": PRICE_RTOL,
            "delta_atol": .002, "delta_rtol": .005, "effective_half_spread_fraction": .1,
            "ci_multiplier": CONFIDENCE_MULTIPLIER},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}}
    profile = _fingerprint(common)
    folder = Path(context.output_dir)/STUDY_VERSION/profile[:16]
    provenance, failures, prices, kernels_rows, limits, gains_rows, variance_rows = [], [], [], [], [], [], []
    def checkpoint(label, specification, fn):
        result, record = _checkpoint(folder, label, {**common, **specification}, fn,
            reuse=reuse_checkpoints)
        provenance.append(record)
        if result.get("status") == "numerical_failure":
            failures.append({"trial": label, "error": result["error"],
                "elapsed_seconds": record["original_elapsed_seconds"]})
        return result
    context.progress("Rough-Heston foundations: positive exponential kernels and independent European references")
    kernels = {}
    for n in (2, 4, 8):
        kernel = approximate_fractional_kernel(.15, 1., n)
        kernels[n] = kernel
        times = np.geomspace(1e-6, 1., 1000)
        exact = times**.65/gamma(1.65)
        approx = np.asarray(kernel.integrated(times))
        for t,a,e in zip(times, approx, exact):
            kernels_rows.append({"factors": n, "time": t, "integrated_kernel": a,
                "exact_integrated_kernel": e, "absolute_error": abs(a-e),
                "relative_error": abs(a-e)/e})
    reference, reference_stability = {}, {}
    references = [("base", config),
        ("quadrature", replace(config, quadrature_nodes=384)),
        ("cutoff_1_5x", replace(config, fourier_cutoff=240.)),
        ("cutoff_2x", replace(config, fourier_cutoff=320.)),
        ("riccati", replace(config, riccati_steps=1024)),
        ("joint", finest)]
    reference_rows = []
    for name, params in truths.items():
        trials = {}
        for axis, settings in references:
            rec = checkpoint(f"fractional_{name}_{axis}", {"model": name, "settings": asdict(settings)},
                lambda p=params, c=settings: _reference_prices(p, contracts, market, c))
            if rec.get("status") == "completed":
                trials[axis] = np.asarray(rec["prices"])
        if "joint" not in trials:
            continue
        reference[name] = trials["joint"]
        stable = len(trials) == len(references)
        for axis, value in trials.items():
            for target in (trials["joint"], trials.get("base", trials["joint"])):
                stable &= bool(np.all(price_gate(value, target)["accepted"]))
        for axis, value in trials.items():
            for target_name, target in (("joint",trials["joint"]),("base",trials.get("base",trials["joint"]))):
                reference_rows.extend(_price_records(contracts, value, target,
                    stage=f"fractional {axis} vs {target_name}", model=name, factors=0,
                    steps=dict(references)[axis].riccati_steps, reference_stable=stable))
        reference_stability[name] = bool(stable)
    lifted, lift_stability = {}, {}
    for name, params in truths.items():
        if name not in reference:
            continue
        for n, kernel in kernels.items():
            records = {}
            for label, settings in references:
                rec = checkpoint(f"lift_{name}_{n}_{label}",
                    {"model": name, "kernel": asdict(kernel), "settings": asdict(settings)},
                    lambda p=params,k=kernel,c=settings: _lift_prices(p,contracts,market,k,c))
                if rec.get("status") == "completed":
                    records[label] = np.asarray(rec["prices"])
            if "joint" not in records:
                continue
            fine = records["joint"]
            fixed_lift_stable = len(records) == len(references)
            for values in records.values():
                for target in (fine, records.get("base",fine)):
                    fixed_lift_stable &= bool(np.all(price_gate(values, target)["accepted"]))
            for label, values in records.items():
                for target_name, target in (("joint",fine),("base",records.get("base",fine))):
                    reference_rows.extend(_price_records(contracts, values, target,
                        stage=f"fixed-lift {label} vs {target_name}", model=name, factors=n,
                        steps=dict(references)[label].riccati_steps, reference_stable=fixed_lift_stable))
            lifted[name,n], lift_stability[name,n] = fine, fixed_lift_stable
            prices.extend(_price_records(contracts, fine, reference[name],
                stage="kernel approximation", model=name, factors=n,
                steps=finest.riccati_steps, reference_stable=reference_stability[name] and fixed_lift_stable))
    context.progress("Rough-Heston foundations: bounded simulation trials; sampling error remains explicit")
    simulation_design = [("flat",n,256,seed) for n in (2,4,8)]
    simulation_design += [("flat",8,steps,seed) for steps in (512,1024)]
    simulation_design += [("flat",8,1024,seed+i) for i in (1,2)]
    simulation_design += [("curve",8,1024,seed)]
    for name,n,steps,trial_seed in simulation_design:
        if (name,n) not in lifted:
            continue
        rec = checkpoint(f"simulation_{name}_{n}_{steps}_{trial_seed}",
            {"model": name, "kernel": asdict(kernels[n]), "steps": steps, "trial_seed": trial_seed},
            lambda p=truths[name], k=kernels[n], s=steps, q=trial_seed: _simulate_trial(
                p,market,k,contracts,paths=paths,steps=s,seed=q))
        if rec.get("status") != "completed":
            continue
        for target, ref, stable in (("fixed-lift simulation",lifted[name,n],lift_stability[name,n]),
                ("full rough model",reference[name],reference_stability[name])):
            prices.extend(_price_records(contracts,rec["prices"],ref,stage=target,
                model=name,factors=n,steps=steps,seed=trial_seed,
                standard_error=rec["standard_errors"],reference_stable=stable))
        expected = np.asarray(lifted_forward_variance(truths[name], np.asarray(TENORS), kernels[n]))
        for i,t in enumerate(TENORS):
            gains_rows.append({"scenario": "no dividends", "model": name, "factors": n,
                "steps":steps,"seed":trial_seed,"days":int(round(t*365)),
                "estimate":rec["gains_mean"][i],"reference":100.,
                "standard_error":rec["gains_standard_error"][i]})
            variance_rows.append({"model":name,"factors":n,"steps":steps,"seed":trial_seed,
                "days":int(round(t*365)),"estimate":rec["variance_mean"][i],
                "reference":expected[i],"standard_error":rec["variance_standard_error"][i]})
    _limit_controls(checkpoint, contracts, market, kernels, paths, seed, config, finest,
        limits, gains_rows)
    tables = {"kernel": pd.DataFrame(kernels_rows), "prices": pd.DataFrame(prices),
        "reference_refinement": pd.DataFrame(reference_rows), "limits": pd.DataFrame(limits),
        "martingale": pd.DataFrame(gains_rows), "variance_means": pd.DataFrame(variance_rows),
        "failures": pd.DataFrame(failures), "provenance": pd.DataFrame(provenance)}
    tables["summary"] = _summary(tables)
    addendum_path = Path(context.output_dir)/STUDY_VERSION/"kernel_fit_budget_trials.json"
    budget_trials, budget_provenance = _kernel_budget_addendum(addendum_path)
    tables["kernel_fit_budgets"] = budget_trials
    extra_tables, extra_provenance = _extra_addenda(Path(context.output_dir)/STUDY_VERSION)
    tables.update(extra_tables)
    candidates = tables["kernel_stiffness"]
    tables["candidate_selection"] = (candidates.loc[
        candidates.factors.eq(4) & candidates.control.eq(.003)].copy()
        if not candidates.empty else pd.DataFrame())
    tables["configuration"] = pd.DataFrame([
        {"item":"Scope", "value":"Inception-state European foundations; no optimal stopping, stock deltas or conditional historical state"},
        {"item":"Seed / paths", "value":f"{seed} / {paths}; independent checks {seed+1}, {seed+2}"},
        {"item":"Factors", "value":"2 / 4 / 8; one fixed 1-year kernel per factor count"},
        {"item":"Time intervals", "value":"256 / 512 / 1024, plus exact event/observation/curve knots"},
        {"item":"Price acceptance", "value":"0.001 + 1e-4 |reference price|; absolute discrepancy + 1.96 standard errors must fit"},
        {"item":"Cash convention", "value":"Dividend min(1.5,S-) at t=.08; spot max(S--1.5,0); factors do not jump"}])
    manifest = {**common, "backend":"rough_heston_lifted_simulation_foundation", "profile_fingerprint": profile, "checkpoint_dir": str(folder),
        "kernel_fit_budget_addendum":budget_provenance,
        "additional_trial_provenance":extra_provenance,
        "practical_kernel_selection":{"factors":4,"min_time_ratio":.003,
            "basis":"Deterministic kernel/European price gates and state dimension/stiffness only; no simulation prices select the candidate",
            "public_default_changed":False,"american_ready":False,"delta_ready":False},
        "kernel_approximations":{str(n):asdict(k) for n,k in kernels.items()},
        "simulation_trials":[{"model":m,"factors":n,"time_steps":s,"seed":q} for m,n,s,q in simulation_design],
        "provenance": provenance, "source_elapsed_seconds":sum(p["original_elapsed_seconds"] for p in provenance),
        "checkpoint_replay":all(p["reused"] for p in provenance),
        "american_pricing_ready":False,"stock_delta_ready":False,
        "limitations":["One low-factor approximation is not the rough model itself.",
            "Normal 95% sampling intervals are not simultaneous confidence bands and omit kernel, time and reference errors.",
            "No continuation regression, exercise policy, American value, stock delta, calibration or hedge replay is computed.",
            "Initial factor state is model inception only; no historical conditional state is inferred.",
            "Budget failures are retained and acceptance tolerances are unchanged."]}
    return {"tables":tables,"figures":_figures(tables) if compute_figures else {},"manifest":manifest}


def _limit_controls(checkpoint, contracts, market, kernels, paths, seed, config, finest,
                    rows, gains_rows):
    from rvlab.pricing.rough_lift import approximate_fractional_kernel
    truth = _truths()["flat"]
    for label, params, kernel, reference_params in (
        ("Hurst .5 simulator control",replace(truth,hurst=.5),
            approximate_fractional_kernel(.5,1.,1),
            HestonParams(truth.v0,truth.theta,truth.kappa,truth.vol_of_vol,truth.rho)),
        ("Constant variance", replace(truth,theta=truth.v0,vol_of_vol=0.),kernels[8],BlackScholesParams(.2)),
        ("Deterministic fractional variance",replace(truth,vol_of_vol=0.),kernels[8],replace(truth,vol_of_vol=0.)),
    ):
        reference = checkpoint("limit_reference_"+str(len(rows)),
            {"label":label,"reference":asdict(reference_params),"config":asdict(finest)},
            lambda p=reference_params:_reference_prices(p,contracts,market,finest))
        lift = checkpoint("limit_lift_"+str(len(rows)),
            {"label":label,"params":asdict(params),"kernel":asdict(kernel),"config":asdict(finest)},
            lambda p=params,k=kernel:_lift_prices(p,contracts,market,k,finest))
        simulation = checkpoint("limit_simulation_"+str(len(rows)),
            {"label":label,"params":asdict(params),"kernel":asdict(kernel),"steps":1024},
            lambda p=params,k=kernel:_simulate_trial(p,market,k,contracts,paths=paths,steps=1024,seed=seed))
        if reference.get("status") != "completed" or lift.get("status") != "completed":
            continue
        rows.extend(_price_records(contracts,lift["prices"],reference["prices"],
            stage=label+" · Fourier",model="control",factors=len(kernel.nodes),
            steps=finest.riccati_steps,reference_stable=True))
        if simulation.get("status") == "completed":
            rows.extend(_price_records(contracts,simulation["prices"],lift["prices"],
                stage=label+" · simulation vs lift",model="control",factors=len(kernel.nodes),
                steps=1024,seed=seed,standard_error=simulation["standard_errors"],reference_stable=True))
    flat_curve = RoughHestonCurveParams(truth.v0,truth.kappa,truth.vol_of_vol,truth.rho,
        MeanReversionCurve((.08,), (truth.theta,truth.theta)),truth.hurst)
    flat_curve_prices = checkpoint("flat_curve_lift_identity",
        {"params":asdict(flat_curve),"kernel":asdict(kernels[8]),"config":asdict(finest)},
        lambda:_lift_prices(flat_curve,contracts,market,kernels[8],finest))
    flat_lift_prices = checkpoint("constant_curve_lift_identity",
        {"params":asdict(truth),"kernel":asdict(kernels[8]),"config":asdict(finest)},
        lambda:_lift_prices(truth,contracts,market,kernels[8],finest))
    if flat_curve_prices.get("status") == flat_lift_prices.get("status") == "completed":
        rows.extend(_price_records(contracts,flat_curve_prices["prices"],flat_lift_prices["prices"],
            stage="Flat theta curve identity · Fourier",model="control",factors=8,
            steps=finest.riccati_steps,reference_stable=True))
    cash_market = _market(cash=True)
    cash = checkpoint("cash_dividend_simulation", {"market":asdict(cash_market),"kernel":asdict(kernels[8]),"steps":1024},
        lambda:_simulate_trial(truth,cash_market,kernels[8],contracts,paths=paths,steps=1024,seed=seed))
    if cash.get("status") == "completed":
        for i,t in enumerate(TENORS):
            gains_rows.append({"scenario":"capped cash dividend","model":"flat","factors":8,
                "steps":1024,"seed":seed,"days":int(round(t*365)),
                "estimate":cash["gains_mean"][i],"reference":100.,
                "standard_error":cash["gains_standard_error"][i]})


def _summary(tables):
    rows=[]
    for source in ("prices","limits"):
        values=tables[source]
        if values.empty:
            continue
        for keys, part in values.groupby(["stage","model","factors","steps"],dropna=False,sort=False):
            rows.append(dict(zip(("stage","model","factors","steps"),keys)) | {
                "contracts":len(part),
                "unique_contracts":len(part[["maturity","strike","is_call"]].drop_duplicates()),
                "seed_count":int(part.seed.nunique(dropna=True)),
                "max_absolute_error":part.absolute_error.max(),
                "max_ci_half_width":part.ci_half_width.max(),
                "max_error_allowance_ratio":(part.conservative_error/part.price_allowance).max(),
                "reference_stable":bool(part.reference_stable.all()),
                "accepted":bool(part.accepted.all())})
    return pd.DataFrame(rows)


def _kernel_budget_addendum(path):
    """Load separately executed trials with verified durable source evidence."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(), {"available":False}
    record = json.loads(path.read_text())
    engine = Path(__file__).parents[1]/"pricing/rough_lift.py"
    runner = path.parent/"raw"/Path(record["runner_source_path"]).name
    for label, source in (("engine",engine),("runner",runner)):
        if (not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()
                != record[f"{label}_source_sha256"]):
            raise ValueError(f"Kernel budget addendum {label} source does not match: {source}")
    return pd.DataFrame(record["trials"]), {"available":True,"path":str(path),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope":record["scope"],"created_at":record["created_at"],
        "separate_historical_execution":True,
        "elapsed_seconds":sum(row["elapsed_seconds"] for row in record["trials"])}


def _verified_extra(path, *, simulation=False):
    """Check local durable runners and numerical sources before reporting trials."""
    path = Path(path)
    record = json.loads(path.read_text())
    if simulation:
        runner = path.parent/Path(record["source_path"]).name
        expected = record["source_sha256"]
        sources = record["pricing_sources"]
    else:
        runner = path.parent/"raw"/Path(record["runner_source_path"]).name
        expected = record["runner_source_sha256"]
        sources = {Path(record["source_path"]).name:record["source_sha256"]}
    if not runner.is_file() or hashlib.sha256(runner.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Additional rough-foundation trial runner has changed: {runner}")
    matching_current, source_evidence = True, {}
    root = path.parent.parent if simulation else path.parent
    for name, expected_sha in sources.items():
        source = Path(__file__).parents[1]/"pricing"/name
        current_match = source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha
        if not current_match:
            source = root/"raw/pricing_sources"/expected_sha/name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Additional rough-foundation numerical source has changed: {source}")
        matching_current &= current_match
        source_evidence[name] = str(source)
    return record, {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256":sources,"source_evidence":source_evidence,
        "runner_sha256":expected,"matches_current_engine":matching_current,
        "separate_historical_execution":True}


def _extra_addenda(folder):
    """Separate stiffness/large-step follow-ups from the unchanged naive pilot."""
    folder = Path(folder)
    kernels, covariance, simulations, provenance = [], [], [], []
    for name in ("kernel_stiffness_trials.json","kernel_stiffness_followup.json"):
        path = folder/name
        if not path.is_file():
            continue
        record, origin = _verified_extra(path)
        provenance.append(origin)
        for trial in record["trials"]:
            common = {"source":name,"factors":trial["factors"],
                "kind":trial.get("kind","min_time_ratio"),
                "control":trial.get("control",trial.get("min_time_ratio")),
                "status":trial["status"],"maximum_node":trial.get("maximum_node"),
                "sum_weights":trial.get("sum_weights"),
                "elapsed_seconds":trial["elapsed_seconds"],
                "failure":trial.get("failure")}
            if trial.get("models"):
                for model, values in trial["models"].items():
                    kernels.append({**common,"model":model,
                        "max_absolute_price_error":values["max_absolute_price_error"],
                        "max_error_over_allowance":values["max_error_over_allowance"],
                        "price_only_gate":values["accepted"],"simulation_ready":False})
            else:
                kernels.append(common)
    path = folder/"split_covariance_audit.json"
    if path.is_file():
        record, origin = _verified_extra(path)
        provenance.append(origin)
        covariance = record["trials"]
    for path in sorted((folder/"simulation_audit").glob("*resolution*.json")):
        record, origin = _verified_extra(path,simulation=True)
        provenance.append(origin)
        config = record["configuration"]
        for row in record["contracts"]:
            gate = price_gate(row["estimate"],row["reference"],standard_error=row["standard_error"])
            simulations.append({"source":path.name,"factors":len(record["kernel"]["nodes"]),
                "maximum_node":max(record["kernel"]["nodes"]),
                "paths":config["paths"],"steps":config["steps"],"seed":config["seed"],
                "elapsed_seconds":record["elapsed_seconds"],**row,
                "conservative_error":float(gate["conservative_error"]),
                "price_allowance":float(gate["price_allowance"]),
                "sampling_gate":bool(gate["accepted"]),"overall_ready":False})
    return {"kernel_stiffness":pd.DataFrame(kernels),
        "split_covariance":pd.DataFrame(covariance),
        "simulation_resolution":pd.DataFrame(simulations)}, provenance


def _figures(tables):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4),constrained_layout=True)
    for n,part in tables["kernel"].groupby("factors"):
        axes[0].loglog(part.time,part.relative_error,label=f"{n} factors")
    axes[0].set(xlabel="Time lag (years)",ylabel="Integrated-kernel relative error",title="More memory factors resolve the fractional kernel")
    axes[0].legend()
    part=tables["prices"]
    if not part.empty:
        pure=part[part.stage.eq("kernel approximation") & part.model.eq("flat")]
        for n,p in pure.groupby("factors"):
            by=p.groupby("days").absolute_error.max()
            axes[1].semilogy(by.index,by.values,marker="o",label=f"{n} factors")
        if not pure.empty:
            tolerance = pure.groupby("days").price_allowance.agg(["min","max"])
            axes[1].fill_between(tolerance.index,tolerance["min"],tolerance["max"],
                color="#737B83",alpha=.15,label="Price-allowance range")
    axes[1].set(xlabel="Maturity (days)",ylabel="Largest price difference ($, log scale)",title="Kernel fit error and option error are separate")
    axes[1].legend()
    second,axes=plt.subplots(2,2,figsize=(11.5,7.6),constrained_layout=True)
    ax=axes.ravel()
    if not part.empty:
        sim=part[part.stage.eq("fixed-lift simulation") & part.model.eq("flat") & part.factors.eq(8)]
        for days,p in sim.groupby("days"):
            p=p[p.seed.eq(p.seed.min()) & p.strike.eq(100)].sort_values("steps")
            ax[0].errorbar(p.steps,p.estimate-p.reference,yerr=p.ci_half_width,marker="o",capsize=3,label=f"{days} days")
        ax[0].axhline(0,color="black",lw=.8)
        ax[0].legend()
    ax[0].set(xlabel="Simulation intervals",ylabel="ATM price − same-lift Fourier ($)",title="Original 8-factor kernel: substantial time error")
    extra=tables.get("simulation_resolution",pd.DataFrame())
    if not extra.empty:
        candidate=extra[extra.factors.eq(4) & extra.paths.eq(32768) & extra.strike.eq(100)]
        for days,p in candidate.groupby("days"):
            p=p.sort_values("steps")
            ax[1].errorbar(p.steps,p.estimate-p.reference,yerr=1.96*p.standard_error,
                marker="o",capsize=3,label=f"{days} days")
        ax[1].axhline(0,color="black",lw=.8)
        ax[1].legend()
    ax[1].set(xlabel="Simulation intervals · 32,768 paths",ylabel="ATM price − same-lift Fourier ($)",title="Practical 4-factor candidate: uncertainty remains")
    covariance=tables.get("split_covariance",pd.DataFrame())
    if not covariance.empty:
        for (n,ratio),p in covariance.groupby(["factors","min_time_ratio"],sort=False):
            label=f"N{n}, first cell {ratio:g}"
            ax[2].semilogx(p.steps,p.split_over_exact_covariance,marker="o",label=label)
        ax[2].axhline(1.,color="black",lw=.8)
        ax[2].legend(fontsize=8)
    ax[2].set(xlabel="Simulation intervals (log scale)",ylabel="Split / exact one-step covariance",title="Fast factors make splitting stiff · frozen variance")
    gains=tables["martingale"]
    if not gains.empty:
        for scenario,p in gains[(gains.factors.eq(8)) & gains.steps.eq(1024) & gains.seed.eq(gains.seed.min()) & gains.model.eq("flat")].groupby("scenario"):
            ax[3].errorbar(p.days,p.estimate-p.reference,yerr=1.96*p.standard_error,marker="o",capsize=3,label=scenario)
        ax[3].axhline(0,color="black",lw=.8)
        ax[3].legend()
    ax[3].set(xlabel="Maturity (days)",ylabel="Discounted stock + paid cash − initial spot ($)",title="Discounted gains under cash jumps")
    return {"kernel_and_prices":fig,"simulation_controls":second}
