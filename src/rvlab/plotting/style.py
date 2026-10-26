"""
rvlab.plotting.style
====================
One visual identity for the whole notebook series, so eleven notebooks read as
one document. Import nothing else from matplotlib in a notebook — call
`set_rvlab_style()` once (setup_notebook does it for you) and use the helpers
in `rvlab.plotting.charts`.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical palette. Ordered so the first three are the ones used most:
# observed / baseline / model.
PALETTE = [
    "#2b6cb0",   # blue    - observed / data
    "#718096",   # grey    - baseline (carry, BS delta)
    "#c05621",   # orange  - the model under test
    "#2f855a",   # green   - a second model
    "#805ad5",   # purple  - third model
    "#b83280",   # magenta - highlight
]

# Semantic aliases, so charts.py never picks a colour by index.
C_OBSERVED = PALETTE[0]
C_BASELINE = PALETTE[1]
C_MODEL = PALETTE[2]
C_ALT = PALETTE[3]
C_WARN = "#c53030"
C_GRID = "#e2e8f0"

# Diverging map for skill/improvement heatmaps (negative = worse than baseline).
DIVERGING = "RdBu_r"
SEQUENTIAL = "viridis"


def set_rvlab_style() -> None:
    """Apply the series-wide matplotlib defaults. Idempotent."""
    mpl.rcParams.update({
        "figure.figsize": (8.0, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",

        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 10,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#a0aec0",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,

        "legend.frameon": False,
        "legend.fontsize": 9,

        "lines.linewidth": 1.6,
        "lines.markersize": 4,

        "xtick.color": "#4a5568",
        "ytick.color": "#4a5568",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def finish(ax, title: str = "", xlabel: str = "", ylabel: str = "", note: str = ""):
    """Apply the series' labelling convention to an axes and return it.

    `title` should state the finding ("BF25 curvature carries the signal"), not
    the mechanic ("feature importances"). `note` renders as a small caption
    under the axes for units, sample size or caveats.
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if note:
        ax.figure.text(0.005, -0.04, note, fontsize=8.5, color="#718096",
                       ha="left", va="top", transform=ax.transAxes)
    return ax


def savefig(fig, name: str, subdir: str = "figures"):
    """Save a figure into notebooks/_output/figures/ and return the path."""
    from ..config import OUTPUT_DIR
    out = OUTPUT_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    return path


__all__ = [
    "set_rvlab_style", "finish", "savefig", "PALETTE", "DIVERGING", "SEQUENTIAL",
    "C_OBSERVED", "C_BASELINE", "C_MODEL", "C_ALT", "C_WARN", "C_GRID", "plt",
]
