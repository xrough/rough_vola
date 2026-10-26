"""Input-audit figures for the pricing notebooks, without pricing or fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_anchor_intervals(results, *, snapshot_date="2025-08-07"):
    """Plot the observed call/put anchor intervals relative to each median.

    The same recorded table is available after a live experiment and through
    ``load_recorded_results``.  These European-parity intervals are input
    diagnostics for American SPY quotes, not confidence intervals. No table is
    modified and no proxy is promoted to an independent stock observation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, StrMethodFormatter

    anchors = results["tables"]["forward_anchors"]
    required = {
        "snapshot_date", "expiry_date", "strike", "forward",
        "forward_lower", "forward_upper",
    }
    missing = sorted(required.difference(anchors.columns))
    if missing:
        raise ValueError(f"Forward-anchor figure requires columns: {missing}")
    date = pd.Timestamp(snapshot_date).date()
    dates = pd.to_datetime(anchors["snapshot_date"]).dt.date
    selected = anchors.loc[dates.eq(date)].copy()
    if selected.empty:
        raise ValueError(f"No recorded forward anchors for {date}")
    selected["expiry_date"] = pd.to_datetime(selected["expiry_date"])
    values = selected[["strike", "forward", "forward_lower", "forward_upper"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forward-anchor figure requires finite strikes and intervals")
    if (selected["forward_lower"] > selected["forward_upper"]).any():
        raise ValueError("A forward anchor has reversed bid/ask interval endpoints")
    if ((selected["forward"] < selected["forward_lower"]) |
            (selected["forward"] > selected["forward_upper"])).any():
        raise ValueError("A forward midpoint lies outside its bid/ask interval")

    groups = list(selected.groupby("expiry_date", sort=True))
    figure, axes = plt.subplots(
        1, len(groups), figsize=(4.1 * len(groups), 5.3),
        squeeze=False, sharex=True,
    )
    blue, ink, amber = "#487da5", "#17354e", "#dca64c"
    relative_limits = []
    for axis, (expiry, group) in zip(axes.flat, groups):
        ordered = group.sort_values("strike", kind="stable")
        median = float(ordered["forward"].median())
        lower = ordered["forward_lower"].to_numpy() - median
        upper = ordered["forward_upper"].to_numpy() - median
        midpoint = ordered["forward"].to_numpy() - median
        positions = np.arange(len(ordered))
        axis.hlines(positions, lower, upper, color=blue, linewidth=2.6, alpha=.85)
        axis.scatter(midpoint, positions, color=ink, s=36, zorder=3)
        axis.axvline(0, color="#64717c", linestyle="--", linewidth=1.2)
        joint_lower, joint_upper = float(lower.max()), float(upper.min())
        if joint_lower > joint_upper:
            axis.axvspan(joint_upper, joint_lower, color=amber, alpha=.22, zorder=0)
            diagnosis = f"No common interval\nNon-overlap: ${joint_lower - joint_upper:.4f}"
        else:
            diagnosis = f"Common interval width: ${joint_upper - joint_lower:.4f}"
        axis.text(.04, .97, diagnosis, transform=axis.transAxes,
                  va="top", fontsize=9.5, color="#6e552b",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92,
                        "pad": 4})
        axis.set_title(f"{expiry:%b %d, %Y}\nMedian forward = ${median:.4f}",
                       fontsize=11, loc="left", pad=12)
        axis.set_yticks(positions, [f"{strike:g}" for strike in ordered["strike"]])
        axis.set_ylim(-.65, len(ordered) + 1.15)
        axis.set_ylabel("Anchor strike ($)", fontsize=10)
        axis.grid(axis="x", alpha=.22)
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(MultipleLocator(.05))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:+.2f}"))
        axis.tick_params(labelsize=9)
        relative_limits.extend((float(lower.min()), float(upper.max())))

    span = max(relative_limits) - min(relative_limits)
    padding = max(.01, .08 * span)
    axes[0, 0].set_xlim(min(relative_limits) - padding, max(relative_limits) + padding)
    figure.suptitle(f"A reproducible median does not reconcile every anchor · {date:%b %d, %Y}",
                   fontsize=14, x=.07, ha="left", y=.985)
    figure.supxlabel("Implied forward minus the expiry median ($ per underlying unit)",
                     fontsize=10.5, y=.11)
    figure.legend(handles=[
        Line2D([], [], color=ink, marker="o", linestyle="none", label="Midpoint estimate"),
        Line2D([], [], color=blue, linewidth=2.6, label="Bid/ask European-parity interval"),
        Line2D([], [], color="#64717c", linestyle="--", label="Median used for calibration"),
        Patch(facecolor=amber, alpha=.3, label="Gap preventing a common intersection"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(.5, -.015))
    figure.subplots_adjust(left=.07, right=.985, top=.79, bottom=.22, wspace=.28)
    return figure
