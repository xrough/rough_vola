"""Contract-level views of recorded curve fits, without pricing or calibration.

The functions accept the result dictionaries returned by the experiments or
``load_recorded_results``. Refinement outcomes are read independently of the
optimizer outcome; an exhausted fit is never silently removed from a chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_INPUTS = (
    ("raw_european_proxy", "Raw parity proxy"),
    ("american_bs_sensitivity", "Exercise sensitivity"),
)
_MODELS = (("heston", "Heston"), ("rough_heston", "Rough Heston"))
_SPECS = (("constant", "Constant"), ("curve_only", "Curve only"),
          ("joint_curve", "Joint curve"))


def _select(table, **values):
    selected = table
    for name, value in values.items():
        selected = selected.loc[selected[name] == value]
    return selected


def _numerics(checks):
    if checks.empty or "numerical_stable" not in checks:
        return "not checked"
    flags = checks["numerical_stable"]
    # CSV and live values have the same meaning, including explicit failure.
    passed = flags.map(lambda value: value is True or isinstance(value, np.bool_)
                       and bool(value) or str(value).lower() == "true")
    return "stable" if bool(passed.all()) else "unstable"


def _contract_order(prices):
    keys = ["contract_id", "expiry_date", "target_log_moneyness", "split", "is_call"]
    return (prices.loc[:, keys].drop_duplicates("contract_id")
            .sort_values(["expiry_date", "target_log_moneyness", "contract_id"]))


def curve_residual_map(results):
    """Compare every contract across the original three curve specifications.

    Color and annotations use signed *refined* residuals in the original
    calibration weighting scale, not standard deviations or dollar prices.
    Held-out rows have purple labels and boxes. Missing prices stay masked.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tables = results["tables"]
    prices, summary = tables["fitted_prices"], tables["fit_summary"]
    checks = tables["numerical_checks"]
    contracts = _contract_order(prices)
    observed = prices["refined_residual"].to_numpy(dtype=float)
    finite = observed[np.isfinite(observed)]
    limit = max(1., float(np.ceil(np.max(np.abs(finite))))) if len(finite) else 1.
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.5), layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e9edf0")
    image = None
    for input_index, (scenario, input_name) in enumerate(_INPUTS):
        for model_index, (model, model_name) in enumerate(_MODELS):
            ax = axes[model_index, input_index]
            matrix = np.full((len(contracts), len(_SPECS)), np.nan)
            headers = []
            for column, (specification, spec_name) in enumerate(_SPECS):
                filters = dict(input_scenario=scenario, model=model,
                               specification=specification)
                fitted = _select(prices, **filters)
                fit = _select(summary, **filters)
                state = str(fit.iloc[0]["status"]) if len(fit) else "not recorded"
                stable = _numerics(_select(checks, **filters))
                headers.append(f"{spec_name}\n{state}\nprices {stable}")
                if len(fitted):
                    values = fitted.set_index("contract_id")["refined_residual"]
                    if values.index.has_duplicates:
                        raise ValueError("expected one price per curve fit and contract")
                    matrix[:, column] = values.reindex(contracts["contract_id"]).to_numpy()
            image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                              vmin=-limit, vmax=limit, aspect="auto")
            labels = []
            for row, contract in enumerate(contracts.itertuples(index=False)):
                expiry = pd.Timestamp(contract.expiry_date).strftime("%d %b")
                option_type = "C" if bool(contract.is_call) else "P"
                heldout = contract.split == "heldout"
                labels.append(f"{expiry} · {contract.target_log_moneyness:+.0%} {option_type} · "
                              + ("H" if heldout else "T"))
                if heldout:
                    ax.add_patch(Rectangle((-.49, row - .47), 2.98, .94,
                                          fill=False, edgecolor="#7043a0", linewidth=1.35))
                for column in range(len(_SPECS)):
                    value = matrix[row, column]
                    ax.text(column, row, f"{value:+.1f}" if np.isfinite(value) else "—",
                            ha="center", va="center", fontsize=9,
                            color="white" if abs(value) > .6 * limit else "#162a3a")
            ax.set_yticks(range(len(contracts)), labels, fontsize=8.5)
            for tick, contract in zip(ax.get_yticklabels(), contracts.itertuples(index=False)):
                if contract.split == "heldout":
                    tick.set_color("#7043a0")
                    tick.set_fontweight("bold")
            ax.set_xticks(range(len(_SPECS)), headers, fontsize=9)
            ax.xaxis.tick_top()
            ax.tick_params(axis="both", length=0)
            for boundary in range(6, len(contracts), 6):
                ax.axhline(boundary - .5, color="white", linewidth=3)
            ax.grid(False)
            ax.set_title(f"{input_name} · {model_name}", fontsize=12, pad=57)
    colorbar = fig.colorbar(image, ax=axes, shrink=.65, pad=.02)
    colorbar.set_label("(Refined model price − target) / calibration price scale")
    fig.suptitle("Where does the variance curve improve the fit?\n"
                 "T = training · H = held out (purple box) · common color scale", fontsize=15)
    return fig


def rough_fit_residuals(results):
    """Expose option-level effects of physical versus asinh fine continuation.

    The ±1 band denotes the calibration price scale (half spread with floor).
    It remains a weighting diagnostic for the exercise-adjusted proxy targets.
    Both failed convergence controls and converged controls remain visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tables = results["tables"]
    prices = _select(tables["fitted_prices"], model="rough_heston", stage="fine")
    summary = _select(tables["fit_summary"], model="rough_heston", stage="fine")
    checks = tables["numerical_checks"]
    expiries = sorted(prices["expiry_date"].unique())
    if not len(expiries):
        raise ValueError("no fine rough-Heston prices recorded")
    fig, axes = plt.subplots(2, len(expiries), figsize=(12.4, 7.5),
                             squeeze=False, sharex=True, sharey=True,
                             layout="constrained")
    colors = {"physical": "#ce7c25", "asinh": "#087e8b"}
    for row, (scenario, input_name) in enumerate(_INPUTS):
        status_lines = []
        for coordinates in colors:
            fit = _select(summary, input_scenario=scenario, coordinates=coordinates)
            state = str(fit.iloc[0]["status"]) if len(fit) else "not recorded"
            labels = fit["label"].tolist()
            stable = _numerics(checks.loc[checks["label"].isin(labels)])
            status_lines.append(f"{coordinates}: {state}; prices {stable}")
        for column, expiry in enumerate(expiries):
            ax = axes[row, column]
            ax.axhspan(-1, 1, color="#dfe7ed", alpha=.7, zorder=0)
            ax.axhline(0, color="#687b88", linewidth=.8)
            for coordinates, color in colors.items():
                data = _select(prices, input_scenario=scenario,
                               coordinates=coordinates, expiry_date=expiry)
                # Connect puts and calls separately; no synthetic curve across ATM.
                for _, side in data.groupby("is_call"):
                    side = side.sort_values("log_moneyness")
                    ax.plot(100 * side["log_moneyness"], side["refined_residual"],
                            color=color, linewidth=1, alpha=.65)
                for split, marker in (("train", "o"), ("heldout", "D")):
                    selected = data.loc[data["split"] == split]
                    ax.scatter(100 * selected["log_moneyness"], selected["refined_residual"],
                               color=color, marker=marker, s=47,
                               facecolors=color if split == "train" else "white",
                               linewidths=1.5, zorder=3)
            ax.set_title(pd.Timestamp(expiry).strftime("Expiry %d %B"), fontsize=11)
            ax.set_xticks([-3, -2, -1, 1, 2, 3])
            if row == 1:
                ax.set_xlabel("OTM log-moneyness (%)")
            if column == 0:
                ax.set_ylabel(f"{input_name}\nSigned residual / price scale")
            if column == 1:
                ax.text(.5, 1.17, "\n".join(status_lines), transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9)
    legend = [Line2D([], [], color=color, linewidth=2, label=coordinates.title() + " coordinates")
              for coordinates, color in colors.items()]
    legend += [Line2D([], [], marker="o", color="#576b78", linestyle="", label="Training"),
               Line2D([], [], marker="D", color="#576b78", markerfacecolor="white",
                      linestyle="", label="Held out")]
    fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Calibration repair at the individual-contract level\n"
                 "Shaded band: ±1 calibration price scale; all points use refined prices", fontsize=15)
    return fig
