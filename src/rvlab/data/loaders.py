"""
rvlab.data.loaders
==================
Every notebook gets its data from here, and every frame arrives carrying its own
provenance in `df.attrs["provenance"]`, so a notebook can print where its
numbers came from without any bookkeeping code.

    df = load_smile_panel("smile_90d")
    provenance(df)     # -> {'dataset_id': ..., 'path': ..., 'synthetic': False, ...}

When a local cache is missing (the market data is gitignored) or when
RVLAB_FORCE_SYNTHETIC is set, the loader silently falls back to
`rvlab.data.synthetic` and records `synthetic: True`. Notebooks therefore always
run; they just say so.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

from .. import config
from . import synthetic as _synthetic

_MAX_HASH_BYTES = 8 * 1024 * 1024          # hash a prefix; these files reach 54 MB


def _sha256_prefix(path: Path) -> str:
    """Hash the first few MB of a file — enough to detect a changed input."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(_MAX_HASH_BYTES))
    return h.hexdigest()[:16]


def _stamp(df: pd.DataFrame, **fields) -> pd.DataFrame:
    """Attach provenance to a frame and return it."""
    df.attrs["provenance"] = {"rows": len(df), **fields}
    return df


def provenance(df: pd.DataFrame) -> dict:
    """Read back the provenance a loader attached. Empty dict if none."""
    return dict(df.attrs.get("provenance", {}))


def describe_provenance(df: pd.DataFrame) -> str:
    """One human-readable line about where a frame came from."""
    p = provenance(df)
    if not p:
        return "no provenance recorded"
    origin = "SYNTHETIC" if p.get("synthetic") else p.get("path", "?")
    span = ""
    if "ts_min" in p:
        span = f" | {p['ts_min']} -> {p['ts_max']}"
    return f"{p.get('dataset_id', '?')}: {p['rows']:,} rows from {origin}{span}"


# ── catalogue ──────────────────────────────────────────────────────────────────
def available() -> pd.DataFrame:
    """What data this machine actually has. Show this early in a notebook."""
    rows = []
    for d in config.DATASETS.values():
        size = d.path.stat().st_size / 1e6 if d.exists() else 0.0
        rows.append({
            "dataset_id": d.dataset_id,
            "present": d.exists(),
            "size_mb": round(size, 1),
            "kind": d.kind,
            "description": d.description,
            "tenor": d.tenor_note,
        })
    out = pd.DataFrame(rows)
    out.attrs["forced_synthetic"] = config.force_synthetic()
    return out


# ── smile panels (the primary datasets) ────────────────────────────────────────
def load_smile_panel(
    dataset_id: str = config.DEFAULT_DATASET,
    min_dte: int | None = None,
    max_dte: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a SPY smile-feature panel, or synthesise one.

    Columns: ts (UTC), expiry (date), T (years), atm_iv, atm_total_var,
    rr25, bf25, alpha, gamma — plus `forward` on the multi-maturity datasets.

    min_dte / max_dte filter on T in calendar days, applied after loading.
    """
    ds = config.DATASETS.get(dataset_id)
    if ds is None:
        raise KeyError(f"unknown dataset {dataset_id!r}; choose from {sorted(config.DATASETS)}")

    if config.force_synthetic() or not ds.exists():
        df = _synthetic.synthetic_smile_panel()
        reason = "RVLAB_FORCE_SYNTHETIC" if config.force_synthetic() else "cache not present"
        _stamp(df, dataset_id=dataset_id, path=None, synthetic=True,
               generator=df.attrs.get("generator"), fallback_reason=reason,
               true_hurst=df.attrs.get("true_hurst"))
    else:
        with open(ds.path, "rb") as fh:
            records = pickle.load(fh)
        df = pd.DataFrame(records)
        _stamp(df, dataset_id=dataset_id, path=str(ds.path), synthetic=False,
               sha256=_sha256_prefix(ds.path))

    df = _normalise_smile_panel(df)

    if min_dte is not None:
        df = df[df["T"] >= min_dte / 365.0]
    if max_dte is not None:
        df = df[df["T"] <= max_dte / 365.0]
    if columns:
        df = df[[c for c in columns if c in df.columns]]

    df = df.reset_index(drop=True)
    prov = df.attrs.setdefault("provenance", {})
    prov["rows"] = len(df)
    if "ts" in df.columns and len(df):
        prov["ts_min"], prov["ts_max"] = str(df["ts"].min()), str(df["ts"].max())
        prov["n_timestamps"] = int(df["ts"].nunique())
        prov["n_expiries"] = int(df["expiry"].nunique()) if "expiry" in df else None
    return df


def _normalise_smile_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Make the real caches and the synthetic generator agree on dtypes/order."""
    prov = df.attrs.get("provenance", {})
    out = df.copy()
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col in ("T", "atm_iv", "atm_total_var", "rr25", "bf25", "alpha", "gamma", "forward"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    sort_cols = [c for c in ("ts", "expiry") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    out.attrs["provenance"] = prov
    return out


# ── the 1-minute chain panel ───────────────────────────────────────────────────
def load_chain_panel() -> pd.DataFrame:
    """SPY 1-minute ATM chain panel (26 columns: quotes, IV, SSVI, realized vol).

    This is the only dataset with bid/ask, so it is where quote-hygiene recipes
    live. Falls back to a synthetic panel projected onto the same column names.
    """
    ds = config.DATASETS["chain_panel"]
    if config.force_synthetic() or not ds.exists():
        df = _synthetic_chain_panel()
        reason = "RVLAB_FORCE_SYNTHETIC" if config.force_synthetic() else "csv not present"
        return _stamp(df, dataset_id="chain_panel", path=None, synthetic=True,
                      fallback_reason=reason)

    df = pd.read_csv(ds.path, parse_dates=["timestamp_utc"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return _stamp(df, dataset_id="chain_panel", path=str(ds.path), synthetic=False,
                  sha256=_sha256_prefix(ds.path),
                  ts_min=str(df["timestamp_utc"].min()), ts_max=str(df["timestamp_utc"].max()))


def _synthetic_chain_panel(n: int = 1_500, seed: int = config.DEFAULT_SEED) -> pd.DataFrame:
    """A stand-in for spy_chain_panel.csv with the same column names."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-08-07 13:31", periods=n, freq="1min", tz="UTC")
    spot = 636.0 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    atm_iv = np.clip(0.12 + np.cumsum(rng.normal(0, 0.0006, n)), 0.05, 0.9)
    T = np.linspace(0.030, 0.018, n)
    from datetime import timedelta
    expiry = [t.date() + timedelta(days=11) for t in ts]
    call_mid = spot * atm_iv * np.sqrt(T) * 0.4
    put_mid = call_mid * (1 + rng.normal(0, 0.01, n))
    half = 0.02 + 0.01 * rng.random(n)

    return pd.DataFrame({
        "timestamp_utc": ts,
        "underlying_price": spot,
        "atm_strike": np.round(spot),
        "expiry_date": expiry,
        "time_to_expiry": T,
        "call_mid": call_mid, "put_mid": put_mid,
        "call_bid": call_mid - half, "call_ask": call_mid + half,
        "put_bid": put_mid - half, "put_ask": put_mid + half,
        "atm_iv": atm_iv,
        "rr25_iv": -0.03 + rng.normal(0, 0.004, n),
        "bf25_iv": 0.0025 + rng.normal(0, 0.0006, n),
        "vix_varswap": atm_iv**2 * 1.1,
        "ssvi_rho": -0.5 + rng.normal(0, 0.05, n),
        "ssvi_phi": 42.0 + rng.normal(0, 3.0, n),
        "rv5_ann": np.abs(rng.normal(0.10, 0.03, n)),
    })


__all__ = [
    "available", "load_smile_panel", "load_chain_panel",
    "provenance", "describe_provenance",
]
