"""Bounded rough-Heston calibration and identification demonstration.

All six parameters are free. Held-out observations describe generalisation;
they never select a start. This study is synthetic only: real SPY calibration
belongs to the American finite-difference workflows.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from numbers import Integral
import time

import numpy as np
import pandas as pd

from rvlab.pricing import PricingConfig, PricingError, price
from ._pricing_fits import (
    COARSE_CONFIG, DEMO_BOUNDS, EXTENDED_CONFIG, PRICE_SCALE_FLOOR,
    SPY_CONFIG, SYNTHETIC_TRUTHS, deterministic_starts, fit_attempt,
    refinement_table, select_attempt, synthetic_surface,
)


def _parameter_rows(dataset, attempts, truth):
    rows = []
    for attempt in attempts:
        summary = attempt.summary
        fitted = summary["params"] or {}
        for name, initial in summary["initial_params"].items():
            lo, hi = summary["bounds"][name]
            rows.append({
                "dataset": dataset, "label": attempt.label,
                "noise_std": (.005 if dataset == "synthetic_noisy" else
                              0. if dataset == "synthetic_clean" else None),
                "status": summary["status"], "parameter": name,
                "truth": truth.get(name), "initial": float(initial),
                "fitted": fitted.get(name), "lower_bound": lo, "upper_bound": hi,
                "is_free": name in summary["free_parameters"],
                "free_parameter_count": summary["free_parameter_count"],
                "fixed_parameter_count": summary["fixed_parameter_count"],
                "config": summary["config"],
                "active_bound": summary["active_bounds"].get(name),
            })
    return rows


def _objective_slices(selected, clean):
    """Training objectives on a slice; never a reoptimised profile likelihood."""
    rows = []
    training = clean.loc[clean.split == "train"]
    target = training.market_price.to_numpy(float)
    chosen = selected.result.params if selected is not None else None
    fixed = ({name: value for name, value in asdict(chosen).items()
              if name not in {"hurst", "vol_of_vol"}} if chosen is not None else {})
    for hurst in (.08, .15, .25, .35, .5):
        for eta in (.06, .12, .2, .3, .45):
            started = time.perf_counter()
            row = {"hurst": hurst, "vol_of_vol": eta,
                   "fixed_parameters": fixed, "config": asdict(COARSE_CONFIG),
                   "training_objective": None, "status": "no_selected_fit",
                   "error": "No finite clean-synthetic calibration result is available.",
                   "objective_kind": "training price MSE; other four parameters held fixed"}
            if chosen is not None:
                params = replace(chosen, hurst=hurst, vol_of_vol=eta)
                try:
                    fitted = np.asarray(price(
                        params, forward=training.forward.to_numpy(float),
                        discount=training.discount.to_numpy(float),
                        strike=training.strike.to_numpy(float),
                        maturity=training.maturity.to_numpy(float),
                        is_call=training.is_call.to_numpy(), config=COARSE_CONFIG,
                    ), dtype=float)
                    if fitted.shape != target.shape or not np.isfinite(fitted).all():
                        raise PricingError("objective slice returned nonfinite prices")
                    row.update(training_objective=float(np.mean((fitted - target) ** 2)),
                               status="ok", error=None)
                except PricingError as exc:
                    row.update(status="numerical_failure", error=str(exc))
            row["elapsed_seconds"] = time.perf_counter() - started
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_error_plot(summary):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(11.5, 5))
    x = np.arange(len(summary))
    for shift, column, label in ((-.18, "train_rmse", "Training"),
                                  (.18, "test_rmse", "Held out")):
        values = pd.to_numeric(summary[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        axis.bar(x[valid] + shift, np.maximum(values[valid], 1e-12), width=.34, label=label)
    labels = []
    for i, row in summary.reset_index(drop=True).iterrows():
        labels.append(f"{row.dataset.replace('synthetic_', 'syn ')}\n{row.label.rsplit('/', 1)[-1]}")
        finite = [value for value in (row.train_rmse, row.test_rmse)
                  if value is not None and np.isfinite(value)]
        if finite:
            axis.annotate(row.status, (i, max(max(finite), 1e-12)), xytext=(0, 5),
                          textcoords="offset points", ha="center", rotation=55, fontsize=7)
        else:
            axis.text(i, .02, "numerical failure", transform=axis.get_xaxis_transform(),
                      rotation=90, ha="center", va="bottom", fontsize=7)
    if np.isfinite(pd.to_numeric(summary.train_rmse, errors="coerce")).any():
        axis.set_yscale("log")
        finite_errors = summary[["train_rmse", "test_rmse"]].apply(
            pd.to_numeric, errors="coerce").to_numpy(float)
        finite_errors = finite_errors[np.isfinite(finite_errors) & (finite_errors > 0)]
        if len(finite_errors):
            axis.set_ylim(max(float(finite_errors.min()) * .5, 1e-13),
                          float(finite_errors.max()) * 10.)
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylabel("Fit-grid price RMSE per underlying unit (log scale)")
    axis.set_title("All starts retained; lower held-out error never selects a start")
    axis.legend()
    figure.tight_layout()
    return figure


def _parameter_plot(parameters):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    names = list(asdict(SYNTHETIC_TRUTHS["rough_heston"]))
    for axis, dataset in zip(axes, ("synthetic_clean", "synthetic_noisy")):
        subset = parameters.loc[parameters.dataset == dataset]
        for label, rows in subset.groupby("label", sort=False):
            rows = rows.set_index("parameter").reindex(names)
            fitted = pd.to_numeric(rows.fitted, errors="coerce").to_numpy(float)
            truth = pd.to_numeric(rows.truth, errors="coerce").to_numpy(float)
            if np.isfinite(fitted).any():
                axis.plot(names, fitted / truth, "o-", alpha=.8,
                          label=f"{label.rsplit('/', 1)[-1]} ({rows.status.iloc[0]})")
        axis.axhline(1., color="black", linestyle="--", linewidth=1, label="Generating value")
        axis.set_title(dataset.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=35)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Fitted parameter / generating parameter")
    figure.suptitle("Parameter agreement and price agreement are different questions")
    figure.tight_layout()
    return figure


def _slice_plot(slices):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    figure, axis = plt.subplots(figsize=(7.5, 5))
    pivot = slices.pivot(index="hurst", columns="vol_of_vol", values="training_objective")
    values = pivot.to_numpy(float)
    finite = values[np.isfinite(values)]
    if len(finite):
        lower = max(float(finite.min()), 1e-12)
        upper = max(float(finite.max()), lower * 1.01)
        image = axis.imshow(np.ma.masked_invalid(np.maximum(values, 1e-12)),
                            origin="lower", aspect="auto", norm=LogNorm(lower, upper))
        figure.colorbar(image, ax=axis, label="Training price MSE (log colour scale)")
    else:
        axis.text(.5, .5, "No numerical slice prices available", transform=axis.transAxes,
                  ha="center", va="center")
    axis.set_xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if not np.isfinite(value):
                axis.text(j, i, "failed", ha="center", va="center", fontsize=7, color="red")
    axis.set_xlabel("vol_of_vol")
    axis.set_ylabel("hurst")
    axis.set_title("Clean synthetic training objective slice\nOther four parameters fixed; no reoptimisation")
    figure.tight_layout()
    return figure


def _refinement_plot(refinements):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 4.5))
    good = refinements.loc[(refinements.status == "ok") &
                           refinements.max_change_vs_finest.notna()].copy()
    if len(good):
        x = np.arange(len(good))
        axis.bar(x, good.max_change_vs_finest.to_numpy(float))
        labels = [f"{row.dataset.replace('synthetic_', 'syn ')}\n"
                  f"N{int(row.quadrature_nodes)}, L{row.fourier_cutoff:g}, M{int(row.riccati_steps)}"
                  for row in good.itertuples()]
        axis.set_xticks(x, labels, fontsize=7, rotation=25, ha="right")
    else:
        axis.text(.5, .5, "No successful refinement comparison", transform=axis.transAxes,
                  ha="center", va="center")
    axis.axhline(.005, linestyle="--", color="black", linewidth=1,
                 label="$0.005 absolute component of stability tolerance")
    axis.set_ylabel("Maximum absolute price change\nversus final grid ($)")
    axis.set_title("Selected fits: all contracts repriced; failures remain in the table")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def extended_demo(context, *, seed=42, max_nfev=80, data_root=None, config=None):
    """Run six synthetic fits and identification/refinement diagnostics."""
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    requested_budget = int(max_nfev)
    budget = min(requested_budget, 80)
    if config is None:
        config = SPY_CONFIG
    elif not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    context.progress("Generating clean/noisy reference prices on the 256/240/1024 grid")
    clean = synthetic_surface("rough_heston", seed=seed, config=EXTENDED_CONFIG)
    noisy = clean.copy(deep=True)
    noisy["market_price"] = clean.reference_price.to_numpy(float) + np.random.default_rng(seed).normal(
        0., .005, len(clean))
    noisy.attrs["noise_std"] = .005
    data = {"synthetic_clean": clean, "synthetic_noisy": noisy}
    starts = deterministic_starts()
    all_attempts, attempts_rows, parameters, predictions, selected_rows, refinements = [], [], [], [], [], []
    selected_by_dataset = {}
    truth = asdict(SYNTHETIC_TRUTHS["rough_heston"])
    for dataset, quotes in data.items():
        attempts = []
        for i, start in enumerate(starts, 1):
            label = f"{dataset}/start_{i}"
            context.progress(f"Fitting all six parameters: {label}, max_nfev={budget}")
            attempt = fit_attempt(label, quotes, start, config=config, max_nfev=budget)
            attempts.append(attempt)
            noise_std = (.005 if dataset == "synthetic_noisy" else
                         0. if dataset == "synthetic_clean" else None)
            attempts_rows.append({"dataset": dataset, "noise_std": noise_std,
                                  **attempt.summary})
            prediction = attempt.predictions.copy(deep=True)
            prediction["dataset"], prediction["label"] = dataset, label
            prediction["fit_status"] = attempt.summary["status"]
            predictions.append(prediction)
            context.progress(f"{label}: {attempt.summary['status']}; "
                             f"training objective={attempt.summary['training_objective']}")
        all_attempts.extend(attempts)
        parameters.extend(_parameter_rows(dataset, attempts, truth if dataset.startswith("synthetic") else {}))
        selected = select_attempt(attempts)
        selected_by_dataset[dataset] = selected
        selected_row = {"dataset": dataset, "label": selected.label if selected else None,
                        "selection_rule": "minimum finite training objective; no held-out filtering",
                        "optimizer_success": selected.summary["optimizer_success"] if selected else False,
                        "original_numerical_stable": selected.summary["numerical_stable"] if selected else False,
                        "original_ready": selected.summary["ready"] if selected else False,
                        "extended_numerical_stable": False, "extended_ready": False,
                        "training_objective": selected.summary["training_objective"] if selected else None}
        if selected is not None:
            context.progress(f"Refining {dataset} selected fit through 512/300/2048")
            configs = list(dict.fromkeys((config, EXTENDED_CONFIG, EXTENDED_CONFIG.refined())))
            table = refinement_table(selected.result.params, quotes, configs)
            table["dataset"], table["label"] = dataset, selected.label
            selected_row["extended_numerical_stable"] = bool(table.stable_vs_finest.all())
            selected_row["extended_ready"] = bool(selected.summary["ready"] and table.stable_vs_finest.all())
            refinements.append(table)
        else:
            refinements.append(pd.DataFrame([{
                "dataset": dataset, "label": None, "status": "no_selected_fit",
                "error": "All calibration attempts failed numerically.",
                "quadrature_nodes": config.quadrature_nodes, "fourier_cutoff": config.fourier_cutoff,
                "riccati_steps": config.riccati_steps, "max_change_vs_finest": None,
                "stable_vs_finest": False, "elapsed_seconds": 0.,
            }]))
        selected_rows.append(selected_row)
    context.progress("Evaluating 25 training-only objective slice points with four parameters fixed")
    slices = _objective_slices(selected_by_dataset["synthetic_clean"], clean)
    attempts_table = pd.DataFrame(attempts_rows)
    selected_table = pd.DataFrame(selected_rows)
    attempts_table["selected_for_illustration"] = attempts_table.label.isin(selected_table.label.dropna())
    summary_columns = [
        "dataset", "label", "status", "selected_for_illustration", "optimizer_success",
        "numerical_stable", "ready", "training_objective", "refined_training_objective",
        "train_rmse", "test_rmse", "refined_train_rmse", "refined_test_rmse",
        "train_scaled_rmse", "test_scaled_rmse", "nfev", "max_price_change", "elapsed_seconds",
    ]
    summary = attempts_table[summary_columns].copy()
    prediction_table = pd.concat(predictions, ignore_index=True)
    parameter_table = pd.DataFrame(parameters)
    refinement_results = pd.concat(refinements, ignore_index=True)
    tables = {"attempts": attempts_table, "calibration_summary": summary,
              "fitted_parameters": parameter_table, "predictions": prediction_table,
              "heldout_predictions": prediction_table.loc[prediction_table.split != "train"].copy(),
              "selected_fits": selected_table, "refinement": refinement_results,
              "objective_slices": slices}
    figures = {"fit_errors": _fit_error_plot(summary),
               "parameter_comparison": _parameter_plot(parameter_table),
               "objective_slice": _slice_plot(slices),
               "refinement_changes": _refinement_plot(refinement_results)}
    manifest = {
        "experiment": "bounded full-parameter rough-Heston identification",
        "evidence_version": "synthetic_identification_v2", "data_kind": "synthetic",
        "exercise_style": "european",
        "seed": int(seed), "starts_seed": 42, "noise_std_per_underlying_unit": .005,
        "max_nfev_requested": requested_budget, "max_nfev_per_attempt": budget,
        "attempt_count": len(all_attempts), "truth": truth,
        "starts": [asdict(start) for start in starts],
        "bounds": {name: DEMO_BOUNDS[name] for name in truth},
        "fit_config": asdict(config), "generation_config": asdict(EXTENDED_CONFIG),
        "extended_check_config": asdict(EXTENDED_CONFIG),
        "extended_check_refined_config": asdict(EXTENDED_CONFIG.refined()),
        "slice_config": asdict(COARSE_CONFIG),
        "price_scale_floor": PRICE_SCALE_FLOOR,
        "synthetic_clean": clean.attrs, "synthetic_noisy": noisy.attrs,
        "attempts": attempts_rows, "selected_fits": selected_rows,
        "selection_rule": "lowest finite training objective; no held-out or ready filtering",
        "limitations": [
            "Six free parameters need not be uniquely identified by these sparse surfaces.",
            "The finite optimizer budget may exhaust; every attempt remains reported.",
            "Accurate synthetic price recovery does not establish parameter recovery.",
            "Objective slices hold four parameters fixed and are not reoptimised profiles.",
            "Only synthetic observations are used; no real SPY quotes enter European calibration.",
            "A final numerical grid is a comparison reference, not proof of exact prices.",
        ],
    }
    return {"tables": tables, "figures": figures, "manifest": manifest}


__all__ = ["extended_demo"]
