"""Recorded-result visual explanations for the synthetic pricing notebooks.

These figures only transform existing diagnostic tables. They do not call a
pricer, choose a new calibration start, or modify the experiment's evidence.
"""

from __future__ import annotations

import numpy as np


_BLUE = "#356a8a"
_TEAL = "#287d73"
_GRAY = "#6b7680"
_LABELS = {"v0": r"$v_0$", "theta": r"$\theta$", "kappa": r"$\kappa$",
           "vol_of_vol": r"$\nu$", "rho": r"$\rho$"}


def _clean_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.set_axisbelow(True)


def plot_heston_parameter_recovery(results):
    """Show the initial and fitted Heston vector against its known truth.

    Differences are divided by the absolute generating coefficient, so the
    negative correlation retains its intuitive direction of movement.
    """
    import matplotlib.pyplot as plt

    order = ["v0", "theta", "kappa", "vol_of_vol", "rho"]
    parameters = results["tables"]["fitted_parameters"]
    parameters = parameters.loc[parameters["fit"].eq("heston")].set_index("parameter").loc[order]
    truth = parameters["generating_value"].to_numpy(dtype=float)
    initial = parameters["starting_value"].to_numpy(dtype=float)
    fitted = parameters["fitted_value"].to_numpy(dtype=float)
    if not np.isfinite(np.r_[truth, initial, fitted]).all() or np.any(truth == 0):
        raise ValueError("Heston recovery plot requires finite coefficients and nonzero truth")
    initial_change = 100 * (initial - truth) / np.abs(truth)
    fitted_change = 100 * (fitted - truth) / np.abs(truth)
    summary = results["tables"]["calibration_summary"]
    fit = summary.loc[summary["label"].eq("heston")].iloc[0]

    figure, axis = plt.subplots(figsize=(9, 4.4))
    positions = np.arange(len(order))
    axis.axvline(0, color="#bcc6cd", linewidth=1.3, label="Generating value")
    for y, start, end in zip(positions, initial_change, fitted_change):
        axis.plot([start, end], [y, y], color="#b9c8d2", linewidth=2.5, zorder=2)
    axis.scatter(initial_change, positions, s=65, marker="o", facecolors="white",
                 edgecolors=_BLUE, linewidths=1.7, label="Initial value", zorder=3)
    axis.scatter(fitted_change, positions, s=65, color=_TEAL,
                 label=f"Fitted value · {fit['status']}", zorder=4)
    axis.set_yticks(positions, [_LABELS[item] for item in order])
    axis.invert_yaxis()
    axis.set_ylim(len(order) - .45, -.55)
    axis.set_xlabel("Difference from generating value / |generating value| (%)")
    axis.set_title("Price recovery can be stronger than parameter recovery", loc="left", pad=35)
    figure.text(.125, .88,
                f"Heston held-out RMSE: \\${float(fit['test_rmse']):.5f} per unit; "
                f"maximum refinement change: \\${float(fit['max_price_change']):.5f}",
                fontsize=10, color=_GRAY)
    axis.legend(loc="lower center", bbox_to_anchor=(.5, -.39), ncol=3,
                frameon=False, fontsize=9)
    _clean_axes(axis)
    axis.grid(axis="x", color="#e6ebee", linewidth=.8)
    figure.subplots_adjust(bottom=.25, top=.81, left=.11, right=.97)
    return figure


def plot_synthetic_noise_response(results):
    """Compare realized price noise with movement of the fitted surface.

    The selected starts come directly from the experiment's training-only
    selection table. Both training and held-out contracts remain visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    selected = results["tables"]["selected_fits"].set_index("dataset")
    predictions = results["tables"]["predictions"]
    frames = {}
    for dataset in ("synthetic_clean", "synthetic_noisy"):
        label = selected.loc[dataset, "label"]
        frames[dataset] = predictions.loc[predictions["dataset"].eq(dataset)
                                          & predictions["label"].eq(label)]
    merged = frames["synthetic_clean"][["contract_id", "strike", "forward", "maturity",
                                        "split", "market_price", "model_price"]].merge(
        frames["synthetic_noisy"][["contract_id", "market_price", "model_price"]],
        on="contract_id", suffixes=("_clean", "_noisy"), validate="one_to_one")
    if merged.empty:
        raise ValueError("No matched synthetic predictions for the recorded selected fits")
    merged["moneyness_pct"] = 100 * np.log(merged["strike"] / merged["forward"])
    merged["observed_noise_cents"] = 100 * (merged["market_price_noisy"] - merged["market_price_clean"])
    merged["fit_movement_cents"] = 100 * (merged["model_price_noisy"] - merged["model_price_clean"])
    if not np.isfinite(merged[["moneyness_pct", "observed_noise_cents", "fit_movement_cents"]].to_numpy()).all():
        raise ValueError("Synthetic noise plot requires finite prices and positive forwards/strikes")
    parameters = results["tables"]["fitted_parameters"]
    sigma = parameters.loc[parameters["dataset"].eq("synthetic_noisy"), "noise_std"].astype(float).iloc[0] * 100
    summary = results["tables"]["calibration_summary"]
    status_parts = []
    for dataset, name in (("synthetic_clean", "Clean"), ("synthetic_noisy", "Noisy")):
        row = summary.loc[summary["dataset"].eq(dataset)
                          & summary["label"].eq(selected.loc[dataset, "label"])].iloc[0]
        status_parts.append(f"{name}: {row['status']}")

    maturities = sorted(merged["maturity"].unique())
    figure, axes = plt.subplots(1, len(maturities), figsize=(11.1, 4.7), sharey=True, squeeze=False)
    amplitude = max(float(merged[["observed_noise_cents", "fit_movement_cents"]].abs().max().max()), sigma) * 1.3
    for axis, maturity in zip(axes.ravel(), maturities):
        data = merged.loc[merged["maturity"].eq(maturity)].sort_values("moneyness_pct")
        axis.axhspan(-sigma, sigma, color="#edf2f5", zorder=0)
        axis.axhline(0, color="#b9c4cc", linewidth=1)
        axis.plot(data["moneyness_pct"], data["fit_movement_cents"],
                  color=_TEAL, marker="s", markersize=4.5, linewidth=1.8, zorder=3)
        for split, marker, fill in (("train", "o", _BLUE), ("heldout", "D", "white")):
            points = data.loc[data["split"].eq(split)]
            axis.scatter(points["moneyness_pct"], points["observed_noise_cents"],
                         marker=marker, s=53, facecolors=fill, edgecolors=_BLUE,
                         linewidths=1.4, zorder=4)
        axis.set_title(f"{round(float(maturity) * 365)} days", loc="left", fontsize=11)
        axis.set_xlabel("Log-moneyness (%)")
        axis.set_ylim(-amplitude, amplitude)
        _clean_axes(axis)
        axis.grid(axis="y", color="#e5ebef", linewidth=.8)
    axes[0, 0].set_ylabel("Price change (cents per unit)")
    figure.suptitle("How a small price perturbation moves the calibrated surface", x=.08, ha="left", y=.99)
    figure.text(.08, .9, "; ".join(status_parts) + ". Selected by training objective only.",
                color=_GRAY, fontsize=10)
    handles = [Line2D([], [], marker="o", linestyle="none", color=_BLUE, label="Injected noise · training"),
               Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", color=_BLUE,
                      label="Injected noise · held out"),
               Line2D([], [], color=_TEAL, marker="s", label="Movement in fitted price"),
               Patch(facecolor="#edf2f5", label=f"± noise standard deviation ({sigma:g}¢)")]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                  bbox_to_anchor=(.5, .005), fontsize=9)
    figure.subplots_adjust(top=.79, bottom=.25, left=.08, right=.98, wspace=.12)
    return figure
