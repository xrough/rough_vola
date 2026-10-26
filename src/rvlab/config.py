"""
rvlab.config
============
Single source of truth for filesystem locations, dataset identities and the
defaults every notebook shares.

Nothing here does work. It only answers "where is X" and "what is the default",
so that a notebook never has to hard-code a path.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ── repository layout ──────────────────────────────────────────────────────────
# config.py lives at <repo>/src/rvlab/config.py
REPO_ROOT = Path(os.environ.get("RVLAB_REPO_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()

SRC_DIR = REPO_ROOT / "src"
DEMO_DIR = REPO_ROOT / "demo"
DATA_DIR = Path(os.environ.get("RVLAB_DATA_DIR", REPO_ROOT / "data")).expanduser().resolve()
DEMO_DATA_DIR = DEMO_DIR / "data"
RESEARCH_DIR = DEMO_DIR / "python" / "research"
RESEARCH_SHARED_DIR = Path(os.environ.get("RVLAB_RESEARCH_SHARED_DIR", RESEARCH_DIR / "shared")).expanduser().resolve()
RESEARCH_CACHE_DIR = RESEARCH_DIR / "cache"
LAB_DIR = REPO_ROOT / "lab"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def _resolve_output_dir() -> Path:
    """Choose a persistent writable location for notebook side-outputs.

    A source checkout keeps artifacts beside the notebooks. Installed packages
    may live in a read-only environment, while Kaggle guarantees a writable
    `/kaggle/working`; both cases need a fallback outside the package tree.
    `RVLAB_OUTPUT_DIR` always wins so automated runs can isolate their outputs.
    """
    configured = os.environ.get("RVLAB_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    if NOTEBOOK_DIR.is_dir() and os.access(NOTEBOOK_DIR, os.W_OK):
        return NOTEBOOK_DIR / "_output"

    for base in (Path("/kaggle/working"), Path.cwd()):
        if base.is_dir() and os.access(base, os.W_OK):
            return base / "rvlab_output"

    return Path(tempfile.gettempdir()) / "rvlab_output"


# Notebook side-outputs (figures, tidy result tables, run manifests).
OUTPUT_DIR = _resolve_output_dir()
# joblib.Memory location for expensive sweeps.
CACHE_DIR = OUTPUT_DIR / "cache"

# Kept as an empty compatibility constant. The selected roughvol modules now
# ship with this distribution; discovery never injects a sibling checkout.
ROUGHVOL_CANDIDATES = ()

# ── shared defaults ────────────────────────────────────────────────────────────
DEFAULT_SEED = 42

# Rough-volatility priors used throughout the series. H_PRIOR is the value the
# original project assumed; the notebooks test it rather than trust it.
H_PRIOR = 0.10
RATE = 0.053   # ~3m T-bill, 2025 — matches demo/python/research/shared/smile_pipeline.py
DIV = 0.013    # SPY annual dividend yield, same source
MIN_DTE = 7
MAX_DTE = 60


def force_synthetic() -> bool:
    """True when RVLAB_FORCE_SYNTHETIC is set — loaders then skip local caches.

    Used to prove the notebooks still run end-to-end on a machine that has none
    of the gitignored market data.
    """
    return os.environ.get("RVLAB_FORCE_SYNTHETIC", "").strip() not in ("", "0", "false", "False")


# ── dataset catalogue ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Dataset:
    """One addressable input to the notebook series."""

    dataset_id: str
    path: Path
    kind: str            # "pickle-records" | "csv"
    description: str
    tenor_note: str = ""

    def exists(self) -> bool:
        return self.path.exists()


DATASETS: dict[str, Dataset] = {
    "smile_90d": Dataset(
        dataset_id="smile_90d",
        path=RESEARCH_CACHE_DIR / "gate0b_days90_farF.pkl",
        kind="pickle-records",
        description="SPY smile features, 90 trading days, all expiries 7-60 DTE.",
        tenor_note="Multi-maturity: T in [0.019, 0.164]. The term-structure dataset.",
    ),
    "smile_127d": Dataset(
        dataset_id="smile_127d",
        path=RESEARCH_CACHE_DIR / "gate0_daysall_farF.pkl",
        kind="pickle-records",
        description="SPY smile features, 127 trading days, front expiry only.",
        tenor_note="Near-tenor only: T ~ 0.02-0.03. The long-history forecasting dataset.",
    ),
    "smile_5d": Dataset(
        dataset_id="smile_5d",
        path=RESEARCH_CACHE_DIR / "gate0_days5_farF.pkl",
        kind="pickle-records",
        description="SPY smile features, 5 trading days. Small and fast, for smoke runs.",
        tenor_note="Near-tenor only.",
    ),
    "chain_panel": Dataset(
        dataset_id="chain_panel",
        path=DEMO_DATA_DIR / "spy_chain_panel.csv",
        kind="csv",
        description="SPY 1-minute ATM chain panel, 26 columns incl. bid/ask, SSVI, realized vol.",
        tenor_note="2025-08-07 to 2025-08-13, one expiry per bar.",
    ),
}

DEFAULT_DATASET = "smile_90d"

__all__ = [
    "REPO_ROOT", "SRC_DIR", "DEMO_DIR", "DATA_DIR", "DEMO_DATA_DIR",
    "RESEARCH_DIR", "RESEARCH_SHARED_DIR", "RESEARCH_CACHE_DIR", "LAB_DIR",
    "NOTEBOOK_DIR", "OUTPUT_DIR", "CACHE_DIR", "ROUGHVOL_CANDIDATES",
    "DEFAULT_SEED", "H_PRIOR", "RATE", "DIV", "MIN_DTE", "MAX_DTE",
    "force_synthetic", "Dataset", "DATASETS", "DEFAULT_DATASET",
]
