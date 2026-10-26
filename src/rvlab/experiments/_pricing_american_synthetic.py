"""Bounded synthetic American/European pricing and stock-delta experiments."""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import pandas as pd

from ._pricing_american import AMERICAN_BACKEND, AMERICAN_EVIDENCE_VERSION


def american_synthetic_demo(context, *, seed=42, config=None, calibration_configs=None,
                             generation_configs=None, max_nfev=16):
    """Compare identical contracts; preserve refinement and calibration failures.

    The Heston fit deliberately estimates v0 only. Fixed parameters and every
    numerical budget are recorded; it is not a five-parameter recovery claim.
    """
    import matplotlib.pyplot as plt
    from rvlab.pricing import (BlackScholesParams, HestonParams, HestonCurveParams,
        MeanReversionCurve, CashDividend, DividendSchedule, DiscountCurve, SpotMarket,
        FDConfig, PricingError, evaluate_vanilla, calibrate_vanilla_surface)
    config = FDConfig(100, 32, 100) if config is None else config
    calibration_configs = ({"bs": FDConfig(400, 80, 400), "heston": FDConfig(240, 80, 300)}
        if calibration_configs is None else dict(calibration_configs))
    generation_configs = ({key: value.refined("joint") for key, value in calibration_configs.items()}
        if generation_configs is None else dict(generation_configs))
    if (not isinstance(config, FDConfig) or set(calibration_configs) != {"bs", "heston"}
            or set(generation_configs) != {"bs", "heston"}
            or any(not isinstance(value, FDConfig) for value in
                   (*calibration_configs.values(), *generation_configs.values()))):
        raise TypeError("synthetic American demonstrations require FDConfig")
    for key, fitted in calibration_configs.items():
        generated = generation_configs[key]
        if (generated.spot_steps <= fitted.spot_steps or generated.time_steps <= fitted.time_steps
                or (key == "heston" and generated.variance_steps <= fitted.variance_steps)
                or generated.spot_max_multiple < fitted.spot_max_multiple
                or generated.variance_max < fitted.variance_max):
            raise ValueError(f"{key}: generation must be strictly finer than calibration in space/time/variance with no smaller domains")
    budget = min(int(max_nfev), 24)
    if budget < 1:
        raise ValueError("max_nfev must be positive")
    truths = {"Black–Scholes": BlackScholesParams(.2),
              "Heston": HestonParams(.04, .045, 1.5, .3, -.6)}
    model_keys = {"Black–Scholes": "bs", "Heston": "heston"}
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((CashDividend(.08, 1.5),), 1.))
    no_dividend = SpotMarket(100., market.discount_curve, DividendSchedule((), 1.))
    # The seed is explicit for later noise studies; this clean experiment has no random errors.
    contracts = pd.DataFrame([{"contract_id": f"syn-{days}-{strike}", "maturity": days/365,
        "strike": float(strike), "is_call": strike >= 100,
        "split": "heldout" if strike in (97, 103) else "train"}
        for days in (14, 30, 60) for strike in (94, 97, 99, 101, 103, 106)])
    rows, checks, boundaries, failures = [], [], [], []
    references = {}
    for label, params in truths.items():
        context.progress(f"Pricing {label}: shared synthetic contracts, both exercise styles")
        for scenario, selected_market in (("cash dividend", market), ("no dividend", no_dividend)):
            prices = {}
            for exercise in ("european", "american"):
                start = time.perf_counter()
                try:
                    result = evaluate_vanilla(params, market=selected_market,
                        strike=contracts.strike.to_numpy(), maturity=contracts.maturity.to_numpy(),
                        is_call=contracts.is_call.to_numpy(), exercise_style=exercise, config=config)
                    part = contracts.copy()
                    part["model"], part["scenario"], part["exercise_style"] = label, scenario, exercise
                    part["price"], part["delta"] = result.prices, result.deltas
                    part["elapsed_seconds"] = time.perf_counter() - start
                    rows.append(part)
                    prices[exercise] = np.asarray(result.prices)
                except PricingError as exc:
                    failures.append({"model": label, "scenario": scenario, "exercise_style": exercise,
                                     "error": str(exc), "stage": "pricing"})
            if len(prices) == 2:
                checks.append({"model": label, "check": f"American dominates European · {scenario}",
                    "max_negative_premium": float(np.max(np.maximum(prices["european"] - prices["american"], 0.))),
                    "status": "passed" if np.all(prices["american"] >= prices["european"] - 1e-7) else "failed"})
        try:
            boundary_result = evaluate_vanilla(params, market=market,
                strike=np.array([99., 101.]), maturity=60/365,
                is_call=np.array([False, True]), exercise_style="american", config=config)
            boundaries.extend([{**dict(item), "model": label, "maturity": 60/365}
                               for item in boundary_result.exercise_boundaries])
            references[label] = evaluate_vanilla(params, market=market,
                strike=contracts.strike.to_numpy(), maturity=contracts.maturity.to_numpy(),
                is_call=contracts.is_call.to_numpy(), exercise_style="american",
                config=generation_configs[model_keys[label]])
        except PricingError as exc:
            failures.append({"model": label, "stage": "generation", "error": str(exc)})
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # Refinement axes are changed separately, with frozen physical parameters.
    refinement = []
    args = dict(market=market, strike=np.array([95., 100., 105.]), maturity=.25,
                is_call=np.array([True, False, False]), exercise_style="american")
    for label, params in truths.items():
        baseline = None
        for axis, settings in [("base", config), ("spot", config.refined("spot")),
                ("time", config.refined("time")), ("variance", config.refined("variance")),
                ("domain", config.refined("domain"))]:
            start = time.perf_counter()
            row = {"model": label, "axis": axis, "status": "numerical_failure", **asdict(settings)}
            try:
                result = evaluate_vanilla(params, config=settings, **args)
                if baseline is None:
                    baseline = result
                row.update(status="priced", price_change=float(np.max(abs(np.asarray(result.prices)-baseline.prices))),
                    delta_change=float(np.max(abs(np.asarray(result.deltas)-baseline.deltas))))
            except PricingError as exc:
                row["error"] = str(exc)
            row["elapsed_seconds"] = time.perf_counter() - start
            refinement.append(row)
    # Flat theta is the legacy Heston specification expressed as a curve.
    flat = HestonCurveParams(.04, 1.5, .3, -.6, MeanReversionCurve((.08,), (.045, .045)))
    try:
        ordinary = evaluate_vanilla(truths["Heston"], config=config, **args)
        curve = evaluate_vanilla(flat, config=config, **args)
        checks.append({"model": "Heston", "check": "Flat mean-reversion curve identity",
            "max_negative_premium": float(np.max(abs(np.asarray(curve.prices)-ordinary.prices))),
            "status": "passed" if np.allclose(curve.prices, ordinary.prices, atol=1e-7) else "failed"})
    except PricingError as exc:
        failures.append({"model": "Heston", "stage": "flat curve", "error": str(exc)})
    attempts, parameters, predictions, calibration_checks = [], [], [], []
    for label, truth in truths.items():
        if label not in references:
            continue
        quotes = contracts.copy()
        quotes["market_price"] = references[label].prices
        quotes["exercise_style"] = "american"
        for study, fit_config in (("deliberately coarse", config),
                ("requested starting budget", calibration_configs[model_keys[label]])):
            initial = BlackScholesParams(.25) if label == "Black–Scholes" else HestonParams(.06, .045, 1.5, .3, -.6)
            bounds = None if label == "Black–Scholes" else {key: (value, value) for key, value in asdict(truth).items() if key != "v0"}
            attempt = {"model": label, "study": study, "status": "numerical_failure", "free_parameters": "volatility" if label == "Black–Scholes" else "v0",
                   "train_rmse": None, "test_rmse": None, "optimizer_success": False,
                   "ready": False, "price_stable": False, "delta_stable": False, "hedge_delta_ready": False,
                   "initial_params": asdict(initial), "generating_params": asdict(truth),
                   "requested_bounds": bounds, "nfev": 0, "pricing_evaluations": 0,
                   "config": asdict(fit_config), "generation_config": asdict(generation_configs[model_keys[label]])}
            start = time.perf_counter()
            try:
                fit = calibrate_vanilla_surface(quotes.loc[quotes.split.eq("train")], initial,
                    market=market, exercise_style="american", config=fit_config, max_nfev=budget, bounds=bounds)
                values = evaluate_vanilla(fit.params, market=market, strike=quotes.strike.to_numpy(),
                    maturity=quotes.maturity.to_numpy(), is_call=quotes.is_call.to_numpy(),
                    exercise_style="american", config=fit_config, check_refinement=True)
                prediction = quotes.copy()
                prediction["model"] = label
                prediction["study"] = study
                prediction["model_price"] = values.prices
                prediction["price_residual"] = prediction.model_price - prediction.market_price
                predictions.append(prediction)
                price_stable = bool(fit.price_stable and values.price_stable)
                delta_stable = bool(fit.delta_stable and values.delta_stable)
                ready = bool(fit.optimizer_success and price_stable)
                attempt.update(status="converged" if ready else "unstable" if fit.optimizer_success else "exhausted",
                    ready=ready, optimizer_success=fit.optimizer_success,
                    price_stable=price_stable, delta_stable=delta_stable,
                    hedge_delta_ready=bool(ready and delta_stable),
                    nfev=fit.nfev, pricing_evaluations=fit.pricing_evaluations+values.diagnostics.get("pricing_evaluations", 0),
                    optimizer_status=getattr(fit, "optimizer_status", None),
                    optimizer_message=getattr(fit, "optimizer_message", None),
                    optimizer_settings=dict(getattr(fit, "optimizer_settings", {})),
                    active_bounds=dict(fit.active_bounds), max_price_change=fit.max_price_change,
                    max_delta_change=fit.max_delta_change)
                diagnostic = values.diagnostics.get("checks", pd.DataFrame())
                if not diagnostic.empty:
                    calibration_checks.append(diagnostic.assign(model=label, study=study,
                                                               coverage="training and held-out contracts"))
                    attempt["max_price_change"] = max(fit.max_price_change, float(diagnostic.max_price_change.max()))
                    attempt["max_delta_change"] = max(fit.max_delta_change, float(diagnostic.max_delta_change.max()))
                for split, prefix in (("train", "train"), ("heldout", "test")):
                    subset = prediction.loc[prediction.split.eq(split)]
                    attempt[f"{prefix}_rmse"] = float(np.sqrt(np.mean(subset.price_residual**2)))
                for name, value in asdict(truth).items():
                    parameters.append({"model": label, "study": study, "parameter": name, "generating": value,
                        "initial": getattr(initial, name), "fitted": getattr(fit.params, name),
                        "fixed": bounds is not None and name in bounds})
            except PricingError as exc:
                attempt["error"] = str(exc)
            attempt["elapsed_seconds"] = time.perf_counter() - start
            attempts.append(attempt)
    figures = {}
    if not table.empty:
        joined = table.pivot(index=["contract_id", "model", "scenario", "strike", "maturity", "is_call"],
                             columns="exercise_style", values="price").reset_index()
        joined["exercise_premium"] = joined.get("american", np.nan) - joined.get("european", np.nan)
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for axis, maturity in zip(axes, sorted(joined.maturity.unique())):
            for (label, scenario), part in joined.loc[joined.maturity.eq(maturity)].groupby(["model", "scenario"], sort=False):
                axis.plot(part.strike, part.exercise_premium, "o-", label=f"{label} · {scenario}")
            axis.set(title=f"{maturity*365:.0f} days", xlabel="Strike ($)", ylabel="American − European ($ per share)")
        axes[-1].legend(fontsize=6)
        figure.suptitle("Exercise value changes with dividends, maturity and volatility dynamics")
        figures["exercise_premiums"] = figure
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for (label, exercise), part in table.loc[table.scenario.eq("cash dividend") & table.maturity.eq(60/365)].groupby(["model", "exercise_style"], sort=False):
            axes[0].plot(part.strike, part.price, "o-", label=f"{label} · {exercise}")
            axes[1].plot(part.strike, part.delta, "o-", label=f"{label} · {exercise}")
        axes[0].set(xlabel="Strike ($)", ylabel="Price ($ per share)", title="Same 60-day positions")
        axes[1].set(xlabel="Strike ($)", ylabel="Stock delta (shares per option unit)", title="Sensitivity under fixed market inputs")
        axes[1].legend(fontsize=7)
        figures["prices_and_deltas"] = figure
    boundary_table = pd.DataFrame(boundaries)
    if not boundary_table.empty:
        figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
        boundary_col = next((col for col in ("spot_boundary", "boundary", "exercise_spot") if col in boundary_table), None)
        if boundary_col and "time" in boundary_table:
            groups = [col for col in ("model", "strike", "is_call") if col in boundary_table]
            for key, part in boundary_table.groupby(groups, sort=False):
                if "strike" in part and float(part.strike.iloc[0]) not in (99., 101.):
                    continue
                part = part.sort_values("time", kind="stable")
                model, strike, call = key
                axis.plot(part.time * 365, part[boundary_col],
                    marker=".", markersize=3,
                    label=f"{model} · K={strike:g} {'call' if call else 'put'}", alpha=.8)
            axis.axvline(.08*365, linestyle="--", color="black", label="Cash-dividend date")
            axis.set(xlabel="Calendar days from valuation", ylabel="Exercise boundary ($)",
                     title="Exercise decisions emerge from the stopping problem")
            missing = int(pd.to_numeric(boundary_table[boundary_col], errors="coerce").isna().sum())
            axis.text(.01, .02, f"{missing:,} unresolved/no-interior-frontier records omitted; boundary nodes are excluded.",
                      transform=axis.transAxes, fontsize=8, color="#52616b")
            axis.legend(fontsize=6)
        else:
            axis.text(.5, .5, "Boundary representation is retained in the exported diagnostics", ha="center", va="center", transform=axis.transAxes)
        figures["exercise_boundaries"] = figure
    refinement_table = pd.DataFrame(refinement)
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for label, group in refinement_table.groupby("model", sort=False):
        for axis, column in zip(axes, ("price_change", "delta_change")):
            if column in group:
                axis.plot(group.axis, group[column], "o-", label=label)
    axes[0].set(ylabel="Price change from base ($)", title="Space, time and domain checks")
    axes[1].set(ylabel="Delta change from base", title="Price accuracy does not imply delta accuracy")
    axes[-1].legend()
    figures["refinement"] = figure
    if predictions:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        summary = pd.DataFrame(attempts)
        for axis, column, title in ((axes[0], "test_rmse", "Held-out recovered-price error"),
                (axes[1], "max_price_change", "Independent fitted-price refinement")):
            for model, group in summary.groupby("model", sort=False):
                if column in group:
                    axis.plot(group.study, group[column], "o-", label=model)
            axis.set(title=title, ylabel="Dollars per underlying unit")
            axis.tick_params(axis="x", labelsize=8)
        axes[-1].legend()
        figures["calibration_recovery"] = figure
    return {"tables": {"contracts": contracts, "prices_and_deltas": table,
            "numerical_settings": pd.DataFrame([
                {"model": label, "study": stage, **asdict(settings)}
                for label, key in model_keys.items()
                for stage, settings in (("deliberately coarse", config),
                    ("requested starting budget", calibration_configs[key]),
                    ("clean observation generation", generation_configs[key]))]),
            "exercise_boundaries": boundary_table, "refinement": refinement_table,
            "calibration_refinement": pd.concat(calibration_checks, ignore_index=True) if calibration_checks else pd.DataFrame(),
            "limit_checks": pd.DataFrame(checks), "calibration_summary": pd.DataFrame(attempts),
            "fitted_parameters": pd.DataFrame(parameters),
            "calibration_predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
            "numerical_failures": pd.DataFrame(failures, columns=["model", "scenario", "exercise_style", "stage", "error"])},
        "figures": figures,
        "manifest": {"evidence_version": AMERICAN_EVIDENCE_VERSION, "backend": AMERICAN_BACKEND,
            "exercise_style": "american", "data_kind": "synthetic", "seed": seed,
            "observations": "Clean prices generated on a finer finite-difference grid",
            "market": asdict(market), "truths": {name: asdict(value) for name, value in truths.items()},
            "coarse_config": asdict(config),
            "calibration_configs": {key: asdict(value) for key, value in calibration_configs.items()},
            "generation_configs": {key: asdict(value) for key, value in generation_configs.items()},
            "max_nfev_per_attempt": budget, "fit_policy": "BS volatility; Heston v0 only; other Heston parameters fixed",
            "attempts": attempts,
            "limitations": ["Coarse demonstration grids may fail refinement; failures remain visible.",
                "Price recovery is assessed independently of parameter recovery and delta convergence.",
                "No real observations, trading claims or American rough-Heston pricing are used."]}}


__all__ = ["american_synthetic_demo"]
