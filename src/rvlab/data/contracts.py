"""
rvlab.data.contracts
====================
Lightweight schema contracts for DataFrames.

The idea, borrowed from `demo/python/lab/validate_data_contracts.py`: state what
a frame must look like *before* you analyse it, so a malformed input fails with
a readable report instead of producing a plausible wrong number three notebooks
later.

    spec = SchemaSpec(
        name="smile_panel",
        columns={"ts": "datetime64[ns, UTC]", "T": "float", "atm_iv": "float"},
        non_null=["ts", "T", "atm_iv"],
        ranges={"T": (0.0, 5.0), "atm_iv": (0.01, 3.0)},
        sorted_by="ts",
    )
    report = spec.validate(df)
    report.raise_if_failed()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """The outcome of one `SchemaSpec.validate` call."""

    name: str
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "ValidationReport":
        """Raise ValueError listing every violation. Returns self when clean."""
        if self.errors:
            bullets = "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(f"schema '{self.name}' failed on {self.n_rows} rows:\n{bullets}")
        return self

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        out = [f"[{status}] {self.name} — {self.n_rows:,} rows"]
        out += [f"  error   : {e}" for e in self.errors]
        out += [f"  warning : {w}" for w in self.warnings]
        return "\n".join(out)

    def _repr_html_(self) -> str:
        colour = "#2f855a" if self.ok else "#c53030"
        items = "".join(f"<li style='color:#c53030'>{e}</li>" for e in self.errors)
        items += "".join(f"<li style='color:#b7791f'>{w}</li>" for w in self.warnings)
        body = f"<ul style='margin:4px 0 0 0'>{items}</ul>" if items else ""
        return (f"<div style='font-family:system-ui;font-size:13px'>"
                f"<b style='color:{colour}'>{'PASS' if self.ok else 'FAIL'}</b> "
                f"{self.name} — {self.n_rows:,} rows{body}</div>")


@dataclass(frozen=True)
class SchemaSpec:
    """A declarative contract for one DataFrame.

    columns    required column -> expected kind ("float", "int", "bool",
               "datetime", "date", "str", or "" to require presence only)
    non_null   columns that must contain no NaN/NaT
    ranges     column -> (lo, hi) inclusive bounds, checked on finite values
    unique     column subsets that must not contain duplicate rows
    sorted_by  column that must be non-decreasing
    """

    name: str
    columns: dict[str, str] = field(default_factory=dict)
    non_null: tuple[str, ...] = ()
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    unique: tuple[tuple[str, ...], ...] = ()
    sorted_by: str | None = None

    # ── kind checking ─────────────────────────────────────────────────────────
    @staticmethod
    def _kind_ok(series: pd.Series, kind: str) -> bool:
        if not kind:
            return True
        if kind == "float":
            return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
        if kind == "int":
            return pd.api.types.is_integer_dtype(series)
        if kind == "bool":
            return pd.api.types.is_bool_dtype(series)
        if kind == "datetime":
            return pd.api.types.is_datetime64_any_dtype(series)
        if kind == "date":
            # python date objects land in an object column
            return (pd.api.types.is_datetime64_any_dtype(series)
                    or series.dropna().map(lambda v: hasattr(v, "year")).all())
        if kind == "str":
            return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        raise ValueError(f"unknown kind {kind!r}")

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Check `df` against this contract. Never raises — inspect the report."""
        rep = ValidationReport(name=self.name, n_rows=len(df))

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            rep.errors.append(f"missing columns: {missing}")

        for col, kind in self.columns.items():
            if col in df.columns and not self._kind_ok(df[col], kind):
                rep.errors.append(f"column '{col}' has dtype {df[col].dtype}, expected {kind}")

        for col in self.non_null:
            if col in df.columns:
                n_bad = int(df[col].isna().sum())
                if n_bad:
                    rep.errors.append(f"column '{col}' has {n_bad:,} null values")

        for col, (lo, hi) in self.ranges.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            finite = vals[np.isfinite(vals)]
            n_out = int(((finite < lo) | (finite > hi)).sum())
            if n_out:
                rep.errors.append(
                    f"column '{col}': {n_out:,} values outside [{lo}, {hi}] "
                    f"(observed {finite.min():.4g}..{finite.max():.4g})")
            n_nonfinite = int(len(vals) - len(finite))
            if n_nonfinite:
                rep.warnings.append(f"column '{col}': {n_nonfinite:,} non-finite values")

        for subset in self.unique:
            cols = [c for c in subset if c in df.columns]
            if len(cols) == len(subset):
                n_dupe = int(df.duplicated(subset=cols).sum())
                if n_dupe:
                    rep.errors.append(f"{n_dupe:,} duplicate rows on {list(subset)}")

        if self.sorted_by and self.sorted_by in df.columns:
            if not df[self.sorted_by].is_monotonic_increasing:
                rep.warnings.append(f"'{self.sorted_by}' is not sorted ascending")

        return rep


# ── the contracts this series uses ─────────────────────────────────────────────
SMILE_PANEL = SchemaSpec(
    name="smile_panel",
    columns={"ts": "datetime", "expiry": "date", "T": "float", "atm_iv": "float",
             "atm_total_var": "float", "rr25": "float", "bf25": "float"},
    non_null=("ts", "expiry", "T"),
    ranges={"T": (0.0, 2.0), "atm_iv": (0.01, 3.0), "rr25": (-1.0, 1.0), "bf25": (-1.0, 1.0)},
    unique=(("ts", "expiry"),),
    sorted_by="ts",
)

CHAIN_PANEL = SchemaSpec(
    name="chain_panel",
    columns={"timestamp_utc": "datetime", "underlying_price": "float",
             "time_to_expiry": "float", "atm_iv": "float",
             "call_bid": "float", "call_ask": "float"},
    non_null=("timestamp_utc", "underlying_price"),
    ranges={"underlying_price": (1.0, 10_000.0), "time_to_expiry": (0.0, 2.0)},
    sorted_by="timestamp_utc",
)

__all__ = ["SchemaSpec", "ValidationReport", "SMILE_PANEL", "CHAIN_PANEL"]
