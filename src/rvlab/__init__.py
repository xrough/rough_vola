"""
rvlab — rough-volatility research toolkit
=========================================
The package behind the `notebooks/` series. Everything the notebooks compute
lives here; the notebooks themselves only import, call, plot and narrate.

Typical first cell of a notebook::

    try:
        import rvlab
    except ModuleNotFoundError:  # source checkout without an editable install
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
        import rvlab
    rvlab.setup_notebook()

Layout
------
data/       loading, schema contracts, synthetic fallback
features/   smile, realized-vol, time-series and roughness feature builders
models/     Black-Scholes, rough-vol closed form, forecasting baselines
pipelines/  scikit-learn transformers, leakage-safe splitters, pipeline factories
evaluate/   metrics, statistical tests, parameter sweeps, hedge backtest
plotting/   one house style for the whole series
competition.py  problem specs, validation routing, OOF evidence, submissions
"""

from __future__ import annotations

import dataclasses
import random
import sys
from dataclasses import dataclass, field

from . import compat, config

__version__ = "0.1.0"


@dataclass
class EnvReport:
    """What `setup_notebook()` found. Rendered as a table in a notebook."""

    python: str
    rvlab_version: str
    seed: int
    optional: dict[str, str] = field(default_factory=dict)   # name -> version | "MISSING"
    external: dict[str, str | None] = field(default_factory=dict)  # roughvol / research_shared
    datasets: list[tuple[str, bool, str]] = field(default_factory=list)
    forced_synthetic: bool = False

    # ── rendering ─────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        lines = [f"rvlab {self.rvlab_version} · python {self.python} · seed {self.seed}"]
        if self.forced_synthetic:
            lines.append("RVLAB_FORCE_SYNTHETIC is set — loaders will use synthetic data.")
        lines.append("packages : " + "  ".join(f"{k}={v}" for k, v in self.optional.items()))
        for name, where in self.external.items():
            lines.append(f"{name:9s}: {where or 'not found'}")
        lines.append("datasets :")
        for did, ok, note in self.datasets:
            lines.append(f"  {'ok     ' if ok else 'missing'} {did:12s} {note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        def row(cells, tag="td"):
            return "<tr>" + "".join(f"<{tag} style='text-align:left;padding:2px 10px'>{c}</{tag}>"
                                    for c in cells) + "</tr>"

        head = (f"<b>rvlab {self.rvlab_version}</b> &middot; python {self.python} "
                f"&middot; seed {self.seed}")
        if self.forced_synthetic:
            head += " &middot; <b>synthetic mode</b>"

        pkg = " &middot; ".join(
            f"{k} {v}" if v != "MISSING" else f"<span style='opacity:.45'>{k} missing</span>"
            for k, v in self.optional.items()
        )
        ext = "".join(row([n, w or "<i>not found</i>"]) for n, w in self.external.items())
        data = "".join(
            row([("&#9679;" if ok else "&#9675;") + " " + did, note]) for did, ok, note in self.datasets
        )
        return (f"<div style='font-family:system-ui;font-size:13px;line-height:1.5'>{head}"
                f"<div style='margin:4px 0'>{pkg}</div>"
                f"<table style='border-collapse:collapse'>{ext}{data}</table></div>")


def _probe(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except ImportError:
        return "MISSING"
    except Exception as exc:
        # This is an environment report, not a feature import.  Optional
        # compiled packages can be installed yet unusable (for example, a
        # missing shared library); that should be visible without preventing
        # notebooks that do not need the package from starting.
        return f"UNAVAILABLE ({type(exc).__name__})"


def setup_notebook(seed: int = config.DEFAULT_SEED, style: bool = True) -> EnvReport:
    """Configure the session and report what is available.

    Sets the plotting style, pandas/numpy display options and every RNG seed,
    puts `roughvol` and the research modules on sys.path, and returns an
    EnvReport that renders as a table so a notebook's first cell shows exactly
    which data and libraries this run had.
    """
    import warnings

    import numpy as np
    import pandas as pd

    random.seed(seed)
    np.random.seed(seed)

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 120)
    pd.set_option("display.precision", 4)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    if style:
        from .plotting.style import set_rvlab_style
        set_rvlab_style()

    compat.ensure_roughvol()
    compat.ensure_research_shared()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Registered last, on purpose: importing matplotlib/IPython prepends a
    # catch-all "always" filter for DeprecationWarning, so an earlier call here
    # would be shadowed by it. pandas 2.3 + numpy 2.5 emit this on every
    # pd.Timedelta(...) construction; nothing at the call site can avoid it, and
    # it would otherwise bury every notebook's output.
    warnings.filterwarnings(
        "ignore", message=".*generic.*unit for NumPy timedelta.*",
        category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore", message="Setting the shape on a NumPy array has been deprecated.*",
        category=DeprecationWarning, module="joblib.*")

    return EnvReport(
        python=".".join(str(v) for v in sys.version_info[:3]),
        rvlab_version=__version__,
        seed=seed,
        optional={n: _probe(n) for n in (
            "numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib",
            "seaborn", "pyarrow", "optuna", "tidyfinance", "xgboost", "lightgbm")},
        external=compat.paths(),
        datasets=[(d.dataset_id, d.exists(), d.tenor_note or d.description)
                  for d in config.DATASETS.values()],
        forced_synthetic=config.force_synthetic(),
    )


__all__ = ["setup_notebook", "EnvReport", "config", "compat", "__version__"]
