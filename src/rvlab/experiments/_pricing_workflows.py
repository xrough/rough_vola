"""Synthetic notebook workflow; empirical and extended studies are separate."""

from __future__ import annotations

from dataclasses import asdict, replace
import time

import numpy as np
import pandas as pd
from scipy.special import gamma

from rvlab.pricing import (BlackScholesParams, HestonParams, RoughHestonParams,
    PricingError, price, solve_fractional_riccati)
from ._pricing_data import implied_volatility
from ._pricing_fits import (COARSE_CONFIG, REFERENCE_CONFIG, SYNTHETIC_INITIALS,
    SYNTHETIC_TRUTHS, synthetic_surface, fit_attempt, refinement_table)


def _ivs(frame, price_column="market_price"):
    """Retain rows whose approximate prices do not admit a European BS IV."""
    rows = frame.copy()
    values, errors = [], []
    for row in rows.itertuples(index=False):
        try:
            value = implied_volatility(getattr(row, price_column), forward=row.forward,
                discount=row.discount, strike=row.strike, maturity=row.maturity,
                is_call=bool(row.is_call))
            values.append(value)
            errors.append(None)
        except ValueError as exc:
            values.append(np.nan)
            errors.append(str(exc))
    rows["implied_volatility"], rows["iv_error"] = values, errors
    rows["log_moneyness"] = np.log(rows.strike / rows.forward)
    return rows


def _parameter_rows(attempts, truths):
    rows = []
    for attempt in attempts:
        summary = attempt.summary
        truth = truths.get(attempt.label, {})
        for name, initial in summary["initial_params"].items():
            fitted = (summary["params"] or {}).get(name)
            rows.append({"fit": attempt.label, "parameter": name,
                "generating_value": truth.get(name), "starting_value": initial,
                "fitted_value": fitted, "fixed": name in summary["fixed_parameters"],
                "active_bound": summary["active_bounds"].get(name),
                "status": summary["status"]})
    return pd.DataFrame(rows)


def _summary(attempts):
    keys = ("label", "status", "free_parameter_count", "fixed_parameter_count",
        "optimizer_success", "numerical_stable", "ready", "train_rmse", "test_rmse",
        "refined_train_rmse", "refined_test_rmse", "train_scaled_rmse", "test_scaled_rmse",
        "max_price_change", "nfev", "elapsed_seconds", "active_bounds", "error")
    return pd.DataFrame([{k: attempt.summary.get(k) for k in keys} for attempt in attempts])


def synthetic_demo(context, *, seed=42, coarse_config=None, fine_config=None, max_nfev=80):
    import matplotlib.pyplot as plt
    coarse = COARSE_CONFIG if coarse_config is None else coarse_config
    fine = REFERENCE_CONFIG if fine_config is None else fine_config
    max_nfev = min(int(max_nfev), 80)
    surfaces = {model: synthetic_surface(model, seed=seed, config=fine)
                for model in SYNTHETIC_TRUTHS}
    all_prices = pd.concat([frame.assign(generating_model=model)
        for model, frame in surfaces.items()], ignore_index=True)
    ivs = _ivs(all_prices)
    context.progress("Synthetic surfaces generated on the finer grid; fitting controlled cases")
    attempts = []
    truths = {}
    for model, surface in surfaces.items():
        initial = SYNTHETIC_INITIALS[model]
        bounds = None
        if model == "rough_heston":
            bounds = {name: (value, value) for name, value in asdict(initial).items() if name != "v0"}
        attempt = fit_attempt(model, surface, initial, bounds=bounds,
            config=coarse, max_nfev=max_nfev)
        attempts.append(attempt)
        truths[model] = asdict(SYNTHETIC_TRUTHS[model])
    flat = fit_attempt("flat_bs_on_heston", surfaces["heston"], BlackScholesParams(.2),
                       config=coarse, max_nfev=max_nfev)
    attempts.append(flat)

    # One contract's inverse differs from fitting one common sigma to a smile.
    inverse = ivs.loc[ivs.generating_model == "heston",
        ["contract_id", "strike", "maturity", "market_price", "implied_volatility", "split"]].copy()
    inverse["flat_surface_volatility"] = (flat.result.params.volatility if flat.result else np.nan)
    inverse["interpretation"] = "per-contract inverse versus one training-surface volatility"

    refinements = []
    for model, surface in surfaces.items():
        table = refinement_table(SYNTHETIC_TRUTHS[model], surface,
                                 [coarse, fine, fine.refined()])
        table.insert(0, "model", model)
        refinements.append(table)
    refinement = pd.concat(refinements, ignore_index=True)

    # Exact and limiting checks are displayed independently of fit statistics.
    args = dict(forward=100., discount=.99, strike=np.array([95., 100., 105.]),
                maturity=.1, is_call=np.array([False, True, True]))
    limits = []
    deterministic = HestonParams(.04, .06, 1.5, 0., -.6)
    integrated = deterministic.theta * .1 + (deterministic.v0 - deterministic.theta) * (-np.expm1(-deterministic.kappa * .1)) / deterministic.kappa
    lhs = price(deterministic, **args)
    rhs = price(BlackScholesParams(np.sqrt(integrated / .1)), **args)
    limits.append({"check": "Heston deterministic variance", "max_abs_price_difference": float(np.max(abs(lhs-rhs))), "note": "analytic integrated variance"})
    constant_variance = RoughHestonParams(.04, .04, 1., 0., -.6, .15)
    limits.append({"check": "Rough Heston constant variance", "max_abs_price_difference": float(np.max(abs(
        price(constant_variance, **args, config=fine) - price(BlackScholesParams(.2), **args)))),
        "note": "zero vol-of-vol and v0=theta give constant variance"})
    intrinsic = args["discount"] * np.maximum(np.where(args["is_call"],
        args["forward"]-args["strike"], args["strike"]-args["forward"]), 0.)
    limits.append({"check": "Black-Scholes zero volatility", "max_abs_price_difference": float(np.max(abs(
        price(BlackScholesParams(0.), **args) - intrinsic))), "note": "discounted deterministic payoff"})
    rough = SYNTHETIC_TRUTHS["rough_heston"]
    half = replace(rough, hurst=.5)
    ordinary = HestonParams(**{k: v for k, v in asdict(half).items() if k != "hurst"})
    limits.append({"check": "Hurst .5 delegates to Heston", "max_abs_price_difference": float(np.max(abs(price(half, **args, config=fine)-price(ordinary, **args, config=fine)))), "note": "dispatch identity; not independent solver validation"})
    constant = RoughHestonParams(.04, .04, 0., 0., -.6, .15)
    u = np.array([.2-.5j, 1.-.5j])
    times, solution = solve_fractional_riccati(constant, u, .1, n_steps=32)
    exact = -.5 * (u*u+1j*u) * .1**(.15+.5) / gamma(.15+.5+1)
    limits.append({"check": "Fractional constant forcing", "max_abs_price_difference": float(np.max(abs(solution[-1]-exact))), "note": "Riccati error, not a price; initial RHS is nonzero"})
    failure = "not triggered"
    try:
        explosive = RoughHestonParams(.04, .04, 1., 1e6, -.7, .1)
        solve_fractional_riccati(explosive, 100-.5j, 10., n_steps=16)
    except PricingError as exc:
        failure = str(exc)
    limits.append({"check": "Deliberately divergent solver", "max_abs_price_difference": None,
                   "note": failure})

    # Smooth curves for readable smile plots and one-parameter-at-a-time effects.
    grid = np.linspace(-.06, .06, 25)
    sensitivity_rows = []
    families = [
        ("Heston correlation", "rho", [-.85, -.6, -.25], SYNTHETIC_TRUTHS["heston"]),
        ("Heston vol of vol", "vol_of_vol", [.1, .3, .6], SYNTHETIC_TRUTHS["heston"]),
        ("Rough Hurst", "hurst", [.08, .15, .3, .5], rough),
    ]
    for family, parameter, settings, baseline in families:
        for value in settings:
            trial = replace(baseline, **{parameter: value})
            frame = pd.DataFrame({"forward": 100., "discount": np.exp(-.03*30/365),
                "strike": 100*np.exp(grid), "maturity": 30/365,
                "is_call": grid >= 0})
            try:
                frame["market_price"] = price(trial, forward=frame.forward, discount=frame.discount,
                    strike=frame.strike, maturity=frame.maturity, is_call=frame.is_call, config=fine)
                frame = _ivs(frame)
                frame["pricing_error"] = None
            except PricingError as exc:
                frame["market_price"] = np.nan
                frame["implied_volatility"] = np.nan
                frame["log_moneyness"] = grid
                frame["pricing_error"] = str(exc)
            frame["family"], frame["parameter"], frame["value"] = family, parameter, value
            sensitivity_rows.append(frame)
    sensitivities = pd.concat(sensitivity_rows, ignore_index=True)

    figures = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, constrained_layout=True)
    for ax, (tenor, group) in zip(axes, ivs.groupby("maturity", sort=True)):
        for model, part in group.groupby("generating_model", sort=False):
            ax.plot(part.log_moneyness, part.implied_volatility*100, "o-", label=model)
        ax.set(title=f"{tenor*365:.0f} days", xlabel="log(K/F)")
    axes[0].set_ylabel("BS implied volatility (%)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Known generating models: implied-volatility smiles")
    figures["generated_smiles"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, (family, part) in zip(axes, sensitivities.groupby("family", sort=False)):
        for value, curve in part.groupby("value", sort=False):
            ax.plot(curve.log_moneyness, curve.implied_volatility*100, label=f"{value:g}")
        ax.set(title=family, xlabel="log(K/F)", ylabel="Implied vol (%)")
        ax.legend(title=part.parameter.iloc[0], fontsize=8)
    fig.suptitle("Change one coefficient; keep the others fixed (30 days)")
    figures["parameter_effects"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for model, group in refinement.groupby("model", sort=False):
        axes[0].plot(range(len(group)), group.max_change_vs_finest, "o-", label=model)
    axes[0].set(xticks=[0, 1, 2], xticklabels=["coarse", "generation", "finest"],
        ylabel="Max price change versus finest ($)", title="Finest grid is a reference, not truth")
    axes[0].legend(fontsize=8)
    groups = inverse.groupby("maturity", sort=True)
    for tenor, part in groups:
        axes[1].plot(np.log(part.strike/(100*np.exp(.02*tenor))), part.implied_volatility*100,
                     "o-", label=f"{tenor*365:.0f}d individual IV")
    if flat.result:
        axes[1].axhline(flat.result.params.volatility*100, color="black", ls="--", label="one fitted BS volatility")
    axes[1].set(xlabel="log(K/F)", ylabel="Volatility (%)", title="Inversion versus flat-surface fitting")
    axes[1].legend(fontsize=7)
    figures["refinement_and_inverse"] = fig

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, attempt in zip(axes, attempts[:3]):
        predictions = attempt.predictions
        for split, group in predictions.groupby("split", sort=False):
            ax.scatter(group.target_price, group.price_residual,
                       marker="o" if split == "train" else "^", label=split)
        ax.axhline(0, color="black", lw=.7)
        ax.set(title=f"{attempt.label}: {attempt.summary['status']}",
               xlabel="Synthetic observed price ($)", ylabel="Model minus observed ($)")
        ax.legend(fontsize=8)
    fig.suptitle("Controlled calibration: coarse-price residuals")
    figures["controlled_fit_residuals"] = fig

    parameters = _parameter_rows(attempts, truths)
    details = _summary(attempts)
    comparison_columns = ["label", "status", "free_parameter_count", "optimizer_success",
        "numerical_stable", "ready", "train_rmse", "test_rmse", "refined_test_rmse",
        "max_price_change", "nfev", "elapsed_seconds", "active_bounds", "error"]
    tables = {"configuration": pd.DataFrame([
        {"item": "seed", "value": seed}, {"item": "generation settings", "value": str(asdict(fine))},
        {"item": "fit settings", "value": str(asdict(coarse))},
        {"item": "max optimizer evaluations", "value": max_nfev},
        {"item": "observation source", "value": "finer-resolution clean synthetic prices"}]),
        "parameter_setup": parameters.drop(columns=["fitted_value", "active_bound", "status"]),
        "prices": all_prices, "implied_vols": ivs,
        "bs_inverse": inverse, "calibration_summary": details[comparison_columns],
        "calibration_details": details,
        "fitted_parameters": parameters, "refinement": refinement,
        "limit_checks": pd.DataFrame(limits), "sensitivities": sensitivities}
    return {"tables": tables, "figures": figures, "manifest": {
        "scope": "synthetic pricing and controlled calibration", "seed": seed,
        "truths": {k: asdict(v) for k, v in SYNTHETIC_TRUTHS.items()},
        "generation_config": asdict(fine), "fit_config": asdict(coarse),
        "max_nfev": max_nfev, "noise_std": 0., "attempts": [a.summary for a in attempts],
        "interpretation": "Price recovery, parameter recovery, convergence and refinement are distinct diagnostics."}}


def spy_demo(*args, **kwargs):
    from ._pricing_spy import spy_demo as run
    return run(*args, **kwargs)


def extended_demo(*args, **kwargs):
    from ._pricing_extended import extended_demo as run
    return run(*args, **kwargs)
