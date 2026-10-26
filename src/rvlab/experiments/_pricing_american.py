"""American-only SPY experiments with a visible, point-in-time input gate.

These workflows never infer stock from options and never convert American
quotes to European targets. Numerical evidence is versioned separately from
the superseded European approximation studies.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from numbers import Integral
import time

import numpy as np
import pandas as pd


AMERICAN_EVIDENCE_VERSION = "american_vanilla_v1"
AMERICAN_BACKEND = "american_vanilla_fd"
SPY_NOTEBOOKS = {
    "02_spy_pricing", "04_spy_numerical_accuracy", "05_spy_fit_attribution",
    "06_spy_forward_variance", "07_spy_curve_convergence",
}
DATES = ("2025-08-07", "2025-08-08", "2025-08-11")
EXPIRIES = ("2025-08-22", "2025-08-29", "2025-09-05")
STUDIES = {
    "pricing": ("Direct American calibration", "BS and Heston; common observed American quotes"),
    "accuracy": ("Price and delta convergence", "Independent space, time and domain refinement"),
    "attribution": ("Bounds and input sensitivity", "Common exercise engine; declared Heston boxes"),
    "variance-curve": ("Maturity-dependent Heston", "Constant versus calendar-time mean-reversion curve"),
    "curve-convergence": ("Calibration convergence", "Multiple starts and optimizer coordinates; common objective"),
}


def _fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def load_american_spy_inputs(data_root=None, *, dates=DATES, dividend_coverage_through=None):
    """Rebuild snapshots from canonical observations; never trust old readiness.

    Missing reference tables become explicit blockers. Missing option archives
    also produce an audit row, allowing all SPY notebooks to explain the same
    data handoff without fabricating observations.
    """
    from .pricing import ROOT
    from rvlab.market_data.pipeline import read_option_window
    from rvlab.market_data.snapshots import build_calibration_snapshot
    root = Path(data_root) if data_root is not None else ROOT / "data/processed/spy"
    dates = tuple(pd.Timestamp(day).date().isoformat() for day in dates)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("dates must be nonempty and distinct")
    references, fingerprints = {}, []
    # Coverage is an explicit argument for these observations. Never inherit
    # it from readiness.json: the reference file may have changed since audit.
    for name in ("stock", "rates", "dividends"):
        path = root / "references" / f"{name}.parquet"
        # The canonical import name is stock_prices in some prepared archives.
        if name == "stock" and not path.is_file():
            path = root / "references/stock_prices.parquet"
        references[name] = pd.read_parquet(path) if path.is_file() else None
        fingerprints.append({"input": name, "present": path.is_file(),
                             "fingerprint": _fingerprint(path)})
    audits, snapshots, observed, exclusions = [], {}, [], []
    for day in dates:
        as_of = pd.Timestamp(f"{day} 15:45", tz="America/New_York").tz_convert("UTC")
        path = root / "options" / f"session_date={day}" / "quotes.parquet"
        if not path.is_file():
            audits.append({"date": day, "ready": False, "contracts": 0,
                "blocker": f"Missing canonical option observations: {path}",
                "stock_observed": False, "dividend_expectations_available": False})
            continue
        options = read_option_window(path, as_of)
        expiry_dates = pd.to_datetime(options.expiry_at, utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
        options = options.loc[expiry_dates.isin(EXPIRIES)].copy()
        options["exercise_style"] = "american"
        snapshot = build_calibration_snapshot(options, as_of, stock=references["stock"],
            rates=references["rates"], dividends=references["dividends"],
            min_dte=0, max_dte=365, max_relative_spread=.25,
            max_quote_age="1min", min_quote_size=1,
            dividend_coverage_through=dividend_coverage_through)
        snapshots[day] = snapshot
        quotes = snapshot.quotes.copy()
        quotes["snapshot_date"] = day
        observed.append(quotes)
        excluded = snapshot.exclusions.copy()
        excluded["snapshot_date"] = day
        exclusions.append(excluded)
        missing = tuple(snapshot.missing_inputs)
        audits.append({"date": day, "ready": snapshot.ready, "contracts": len(quotes),
            "blocker": "; ".join(missing) if missing else "None",
            "stock_observed": "stock" in snapshot.provenance,
            "dividend_expectations_available": not any("dividend" in item.lower() for item in missing)})
        manifest = path.with_name("manifest.json")
        fingerprints.append({"input": f"options/{day}", "present": True,
            "fingerprint": _fingerprint(manifest), "quotes_path": str(path.resolve()),
            "quotes_bytes": path.stat().st_size,
            "observable_window_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(options, index=False).values.tobytes()).hexdigest()})
    return {"audit": pd.DataFrame(audits), "snapshots": snapshots,
        "quotes": pd.concat(observed, ignore_index=True) if observed else pd.DataFrame(),
        "exclusions": pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame(),
        "fingerprints": fingerprints, "processed_root": str(root.resolve())}


def select_american_contracts(snapshot):
    """Six deterministic OTM contracts per maturity, selected from observed spot.

    No call/put matching or parity anchors are used. The ±2% contracts are held
    out; ties are resolved by strike then contract identity. Training and held
    out rows remain contract-disjoint, sharing the same maturity dates.
    """
    inputs = snapshot.calibration_inputs()
    spot = float(inputs["market"].spot)
    quotes = inputs["quotes"].copy()
    quotes["expiry_date"] = pd.to_datetime(quotes.expiry_at, utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    selected = []
    for expiry in EXPIRIES:
        group = quotes.loc[quotes.expiry_date.eq(expiry)]
        used = set()
        for target in (-.03, -.02, -.01, .01, .02, .03):
            otm = group.strike.ge(spot) if target > 0 else group.strike.le(spot)
            candidates = group.loc[group.is_call.eq(target > 0) & otm & ~group.contract_id.isin(used)].copy()
            candidates["distance"] = abs(np.log(candidates.strike.astype(float) / spot) - target)
            candidates = candidates.sort_values(["distance", "strike", "contract_id"], kind="stable")
            if candidates.empty:
                raise ValueError(f"{expiry}: no distinct American contract for log-moneyness target {target:+.2f}")
            row = candidates.iloc[[0]].drop(columns="distance").copy()
            row["target_log_moneyness"] = target
            row["split"] = "heldout" if abs(target) == .02 else "train"
            selected.append(row)
            used.add(row.contract_id.iloc[0])
    result = pd.concat(selected, ignore_index=True)
    result["exercise_style"] = "american"
    return result, inputs["market"]


def _audit_plot(audit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    x = np.arange(len(audit))
    axes[0].bar(x, audit.contracts, color="#2878A5")
    axes[0].set(xticks=x, xticklabels=audit.date, ylabel="Eligible observed option contracts",
                title="Option observations are available")
    columns = ["stock_observed", "dividend_expectations_available", "ready"]
    values = audit[columns].astype(int).to_numpy().T
    axes[1].imshow(values, vmin=0, vmax=1,
        cmap=ListedColormap(["#eef1f4", "#287d73"]), aspect="auto")
    axes[1].grid(False)
    axes[1].set(xticks=x, xticklabels=audit.date, yticks=range(3),
        yticklabels=["Independent stock", "Dated dividend expectations", "American calibration inputs"],
        title="Input gate: available / missing")
    for i in range(3):
        for j in range(len(audit)):
            axes[1].text(j, i, "Available" if values[i, j] else "Missing", ha="center", va="center",
                         fontsize=8, color="white" if values[i, j] else "#52616b")
    return figure


def _raw_price_plot(quotes):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, expiry in zip(axes, EXPIRIES):
        if not quotes.empty:
            dates = pd.to_datetime(quotes.expiry_at, utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
            for (day, call), part in quotes.loc[dates.eq(expiry)].groupby(["snapshot_date", "is_call"], sort=True):
                part = part.sort_values("strike")
                axis.plot(part.strike, part.market_price, ".-", markersize=2,
                    linestyle="-" if call else "--", label=f"{day[5:]} {'call' if call else 'put'}")
        axis.set(title=expiry, xlabel="Strike ($)", ylabel="Observed midpoint ($ per share)")
    axes[-1].legend(fontsize=6)
    figure.suptitle("Observed American quotes · no model or implied-volatility conversion")
    return figure


def american_spy_demo(context, *, study="pricing", data_root=None, seed=42,
                      config=None, max_nfev=40, dividend_coverage_through=None):
    """Audit first; calibrate only current, complete American market inputs."""
    if study not in STUDIES:
        raise ValueError(f"unknown American SPY study: {study}")
    from rvlab.pricing.vanilla_types import FDConfig
    if config is not None and not isinstance(config, FDConfig):
        raise TypeError("American SPY studies require FDConfig, not Fourier PricingConfig")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    inputs = load_american_spy_inputs(data_root, dividend_coverage_through=dividend_coverage_through)
    audit = inputs["audit"]
    default = inputs["snapshots"].get(DATES[0])
    ready = default is not None and default.ready
    tables = {"input_audit": audit, "observed_quotes": inputs["quotes"],
        "input_exclusions": inputs["exclusions"],
        "study_design": pd.DataFrame([{"study": STUDIES[study][0], "comparison": STUDIES[study][1],
            "fit_date": DATES[0], "exercise_style": "american", "status": "ready" if ready else "blocked"}])}
    figures = {"input_readiness": _audit_plot(audit)}
    if not inputs["quotes"].empty:
        figures["observed_prices"] = _raw_price_plot(inputs["quotes"])
    manifest = {"evidence_version": AMERICAN_EVIDENCE_VERSION,
        "backend": AMERICAN_BACKEND, "exercise_style": "american", "data_kind": "real_spy",
        "study": study, "status": "ready" if ready else "blocked_inputs",
        "seed": seed, "sample_dates": DATES, "expiries": EXPIRIES,
        "processed_root": inputs["processed_root"], "inputs": inputs["fingerprints"],
        "snapshot_provenance": {day: dict(snapshot.provenance)
            for day, snapshot in inputs["snapshots"].items()},
        "calibration_ready": bool(ready), "hedging_ready": False,
        "superseded_european_artifacts": "Retained separately; not loaded as American evidence.",
        "assumptions": ["American bid/ask quotes are fitted directly.",
            "Independent raw spot and dated dividend expectations are required.",
            "Treasury discounts remain a declared rate proxy, not historical OIS vintages.",
            "Option interval timestamps do not prove quote-change freshness.",
            "American rough Heston and hedge replay are outside this milestone."]}
    if not ready:
        context.progress("American SPY calibration blocked: independent stock/dividend inputs remain incomplete")
        return {"tables": tables, "figures": figures, "manifest": manifest}
    _fit_ready_surface(default, study, config, min(int(max_nfev), 80), tables, figures, manifest)
    return {"tables": tables, "figures": figures, "manifest": manifest}


def _fit_ready_surface(snapshot, study, config, budget, tables, figures, manifest):
    """The ready-data path exclusively calls the American calibrator."""
    from rvlab.pricing import (BlackScholesParams, HestonParams, HestonCurveParams,
        MeanReversionCurve, FDConfig, evaluate_vanilla, calibrate_vanilla_surface, PricingError,
        OptimizerConfig, parameter_values)
    quotes, market = select_american_contracts(snapshot)
    if config is None:
        config = FDConfig(240, 80, 300)
    if not isinstance(config, FDConfig):
        raise TypeError("American SPY studies require FDConfig, not Fourier PricingConfig")
    specs = [("Black–Scholes", BlackScholesParams(.2), None),
             ("Heston", HestonParams(.04, .045, 1.5, .3, -.6), None)]
    if study == "attribution":
        specs.append(("Heston expanded bounds", HestonParams(.04, .045, 2., .5, -.6),
            {"v0": (.0025, .25), "theta": (.0025, 2.), "kappa": (.01, 80.),
             "vol_of_vol": (.01, 2.), "rho": (-.95, 0.)}))
    if study in {"variance-curve", "curve-convergence"}:
        knots = tuple(sorted(quotes.maturity.unique())[:-1])
        curve = MeanReversionCurve(knots, (.045,) * (len(knots) + 1))
        specs.append(("Heston mean-reversion curve", HestonCurveParams(.04, 1.5, .3, -.6, curve), None))
        if study == "curve-convergence":
            specs.append(("Heston curve alternate start", HestonCurveParams(.025, 5., .6, -.4, curve), None))
    if study == "curve-convergence":
        specs = [(f"{label} · {coordinates}", initial, bounds, coordinates)
            for label, initial, bounds in specs if isinstance(initial, HestonCurveParams)
            for coordinates in ("physical", "asinh")]
    else:
        specs = [(*spec, "physical") for spec in specs]
    attempts, predictions, numerical_checks, parameter_rows = [], [], [], []
    for label, initial, bounds, coordinates in specs:
        start = time.perf_counter()
        row = {"model": label, "status": "numerical_failure", "ready": False,
               "train_rmse": None, "test_rmse": None, "error": None,
               "optimizer_success": False, "price_stable": False, "delta_stable": False,
               "nfev": 0, "pricing_evaluations": 0, "active_bounds": {},
               "initial_params": asdict(initial), "bounds": bounds,
               "coordinates": coordinates}
        try:
            fit = calibrate_vanilla_surface(quotes.loc[quotes.split.eq("train")], initial,
                market=market, exercise_style="american", config=config, bounds=bounds, max_nfev=budget,
                optimizer=OptimizerConfig(parameterization=coordinates))
            values = evaluate_vanilla(fit.params, market=market, strike=quotes.strike.to_numpy(),
                maturity=quotes.maturity.to_numpy(), is_call=quotes.is_call.to_numpy(),
                exercise_style="american", config=config, check_refinement=True)
            prediction = quotes.copy()
            prediction["model"], prediction["model_price"], prediction["delta"] = label, values.prices, values.deltas
            prediction["price_residual"] = prediction.model_price - prediction.market_price
            prediction["scaled_residual"] = prediction.price_residual / np.maximum((prediction.ask - prediction.bid) / 2, .005)
            price_ok, delta_ok = bool(np.all(values.price_stable)), bool(np.all(values.delta_stable))
            diagnostics = getattr(values, "diagnostics", {})
            scales = np.maximum((prediction.ask - prediction.bid).to_numpy() / 2, .005)
            price_ok = price_ok and all(np.all(np.asarray(change) <= .1*scales)
                for change in diagnostics.get("price_changes", {}).values())
            checks = diagnostics.get("checks", pd.DataFrame())
            if not checks.empty:
                numerical_checks.append(checks.assign(model=label, stage="all contracts"))
            for parameter, value in parameter_values(fit.params).items():
                parameter_rows.append({"model": label, "parameter": parameter,
                    "initial": parameter_values(initial)[parameter], "fitted": value,
                    "active_bound": dict(fit.active_bounds).get(parameter)})
            row.update(status="converged" if fit.optimizer_success and price_ok and delta_ok else
                "unstable" if fit.optimizer_success else "exhausted", ready=bool(fit.ready and price_ok and delta_ok),
                optimizer_success=fit.optimizer_success, price_stable=price_ok, delta_stable=delta_ok,
                nfev=fit.nfev, pricing_evaluations=fit.pricing_evaluations+diagnostics.get("pricing_evaluations", 0),
                optimizer_status=getattr(fit, "optimizer_status", None),
                optimizer_message=getattr(fit, "optimizer_message", None),
                optimizer_settings=dict(getattr(fit, "optimizer_settings", {})),
                physical_optimality=getattr(fit, "physical_optimality", None),
                active_bounds=dict(fit.active_bounds), params=asdict(fit.params), coordinates=coordinates)
            for split, prefix in (("train", "train"), ("heldout", "test")):
                subset = prediction.loc[prediction.split.eq(split)]
                row[f"{prefix}_rmse"] = float(np.sqrt(np.mean(subset.price_residual ** 2)))
                row[f"{prefix}_scaled_rmse"] = float(np.sqrt(np.mean(subset.scaled_residual ** 2)))
            predictions.append(prediction)
        except PricingError as exc:
            row["error"] = str(exc)
        row["elapsed_seconds"] = time.perf_counter() - start
        attempts.append(row)
    tables["selected_contracts"] = quotes
    tables["fit_summary"] = pd.DataFrame(attempts)
    tables["fitted_prices"] = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    tables["fitted_parameters"] = pd.DataFrame(parameter_rows)
    tables["numerical_checks"] = pd.concat(numerical_checks, ignore_index=True) if numerical_checks else pd.DataFrame()
    manifest.update(status="completed", numerical_config=asdict(config), max_nfev_per_attempt=budget,
        attempts=attempts, market=asdict(market),
        selected_contracts_sha256=hashlib.sha256(
            pd.util.hash_pandas_object(quotes, index=False).values.tobytes()).hexdigest())
    if predictions:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for label, group in tables["fitted_prices"].groupby("model", sort=False):
            axis.plot(np.arange(len(group)), group.price_residual, "o-", label=label)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel="Shared contract order", ylabel="Model − observed midpoint ($ per share)",
                 title="Direct American calibration · identical observed contracts")
        axis.legend()
        figures["calibration_residuals"] = figure
        checks = tables["numerical_checks"]
        if not checks.empty:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for label, group in checks.groupby("model", sort=False):
                axes[0].plot(group.axis, group.max_price_change, "o-", label=label)
                axes[1].plot(group.axis, group.max_delta_change, "o-", label=label)
            axes[0].set(ylabel="Maximum price change ($)", title="Independent price refinements")
            axes[1].set(ylabel="Maximum stock-delta change", title="Independent delta refinements")
            axes[-1].legend(fontsize=7)
            figures["numerical_refinement"] = figure


__all__ = ["AMERICAN_EVIDENCE_VERSION", "AMERICAN_BACKEND", "SPY_NOTEBOOKS",
           "american_spy_demo", "load_american_spy_inputs", "select_american_contracts"]
