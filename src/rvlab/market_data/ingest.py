"""Lossless, bounded-memory normalization of the local SPY CBBO archive.

The CBBO ``ts_recv`` index denotes the end of the one-minute interval.  It is
also our earliest availability time.  ``ts_event`` describes the last trade;
it is deliberately *not* a quote timestamp or a quote-freshness measure.
Invalid and out-of-session observations are retained and labelled, never
silently discarded.  Prices are dollars per share, not contract notionals.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
from itertools import repeat
import json
import multiprocessing
from pathlib import Path
import re
import time
from typing import Callable, Iterator
from uuid import uuid4
import zipfile

import numpy as np
import pandas as pd

from . import calendar


SCHEMA_VERSION = 1
_MEMBER_RE = re.compile(r"(?:^|/)[^/]*-(\d{8})\.cbbo-1m\.dbn\.zst$")
_OSI_RE = re.compile(r"^([A-Z][A-Z0-9.]{0,5})(\d{6})([CP])(\d{8})$")
_REQUIRED = (
    "symbol", "instrument_id", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"
)


def option_schema():
    """Return the stable Arrow schema (Arrow is an optional data dependency)."""
    import pyarrow as pa

    utc = pa.timestamp("ns", tz="UTC")
    return pa.schema([
        ("timestamp", utc), ("available_at", utc), ("contract_id", pa.string()),
        ("symbol", pa.string()), ("instrument_id", pa.int64()),
        ("expiry_date", pa.date32()), ("expiry_at", utc), ("strike", pa.float64()),
        ("is_call", pa.bool_()), ("bid", pa.float64()), ("ask", pa.float64()),
        ("bid_size", pa.int64()), ("ask_size", pa.int64()), ("flags", pa.int64()),
        ("session_date", pa.date32()), ("in_regular_session", pa.bool_()),
        ("quote_valid", pa.bool_()), ("exclusion_reason", pa.string()),
        ("last_trade_ts", utc), ("last_trade_price", pa.float64()),
        ("last_trade_size", pa.int64()),
    ], metadata={
        b"rvlab.schema_version": str(SCHEMA_VERSION).encode(),
        b"rvlab.price_unit": b"USD_per_underlying_share",
        b"rvlab.timestamp": b"CBBO ts_recv: one-minute interval end; UTC",
        b"rvlab.last_trade_ts": b"ts_event; not quote age",
    })


@lru_cache(maxsize=150_000)
def _parse_symbol(symbol: str) -> tuple[str | None, date | None, bool | None, float, str | None]:
    """Parse padded or compact OSI identifiers without losing malformed rows."""
    compact = re.sub(r"\s+", "", symbol)
    match = _OSI_RE.fullmatch(compact)
    if match is None:
        return None, None, None, np.nan, "invalid_contract_symbol"
    root, expiry, kind, strike = match.groups()
    try:
        expiry_date = date(2000 + int(expiry[:2]), int(expiry[2:4]), int(expiry[4:]))
    except ValueError:
        return None, None, None, np.nan, "invalid_expiry_date"
    reason = None if root == "SPY" else "unexpected_root"
    return compact, expiry_date, kind == "C", int(strike) / 1000.0, reason


@lru_cache(maxsize=4096)
def _expiry_timestamp(expiry: date) -> pd.Timestamp:
    return pd.Timestamp(calendar.expiry_timestamp(expiry))


def normalize_quotes(frame: pd.DataFrame, session_date: date | str) -> pd.DataFrame:
    """Normalize one raw DBN chunk, preserving its row order and row count.

    ``quote_valid`` tests a parsed SPY contract and finite positive, uncrossed
    bid/ask prices.  The session flag is separate, so a valid extended-hours
    quote is retained without becoming an eligible regular-session quote.
    Zero displayed sizes remain visible for downstream liquidity screening.
    """
    missing = [name for name in _REQUIRED if name not in frame]
    if missing:
        raise ValueError(f"CBBO input is missing required columns: {', '.join(missing)}")
    session = date.fromisoformat(session_date) if isinstance(session_date, str) else session_date
    if not isinstance(session, date):
        raise TypeError("session_date must be a date or ISO date string")
    raw = frame.reset_index(drop=True)
    if "ts_recv" in frame:
        timestamps = pd.to_datetime(raw["ts_recv"], utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = pd.Series(pd.to_datetime(frame.index, utc=True), index=raw.index)
    else:
        raise ValueError("CBBO input requires a ts_recv column or DatetimeIndex")
    timestamps = timestamps.astype("datetime64[ns, UTC]")
    symbols = raw["symbol"].astype("string")
    unique_symbols = symbols.dropna().unique()
    parsed = {str(symbol): _parse_symbol(str(symbol)) for symbol in unique_symbols}
    missing_parse = (None, None, None, np.nan, "invalid_contract_symbol")
    records = [parsed.get(symbol, missing_parse) if pd.notna(symbol) else missing_parse for symbol in symbols]
    contract = pd.DataFrame(records, columns=["contract_id", "expiry_date", "is_call", "strike", "reason"])
    expiry_map = {value: _expiry_timestamp(value) for value in contract["expiry_date"].dropna().unique()}
    out = pd.DataFrame({
        "timestamp": timestamps,
        "available_at": timestamps,
        "contract_id": contract["contract_id"].astype("string"),
        "symbol": symbols,
        "instrument_id": pd.to_numeric(raw["instrument_id"], errors="raise").astype("Int64"),
        "expiry_date": contract["expiry_date"],
        "expiry_at": pd.to_datetime(contract["expiry_date"].map(expiry_map), utc=True).astype("datetime64[ns, UTC]"),
        "strike": contract["strike"].astype(float),
        "is_call": contract["is_call"].astype("boolean"),
        "bid": pd.to_numeric(raw["bid_px_00"], errors="coerce").astype(float),
        "ask": pd.to_numeric(raw["ask_px_00"], errors="coerce").astype(float),
        "bid_size": pd.to_numeric(raw["bid_sz_00"], errors="raise").astype("Int64"),
        "ask_size": pd.to_numeric(raw["ask_sz_00"], errors="raise").astype("Int64"),
        "flags": pd.to_numeric(raw.get("flags", pd.Series(0, index=raw.index)), errors="raise").astype("Int64"),
        "session_date": pd.Series([session] * len(raw), dtype=object),
    })
    bounds = calendar.session_bounds(session)
    out["in_regular_session"] = False if bounds is None else (
        (out["timestamp"] > bounds[0]) & (out["timestamp"] <= bounds[1])
    )
    reason = contract["reason"].astype("string")
    reason.loc[timestamps.isna()] = "invalid_timestamp"
    checks = (
        (~np.isfinite(out["bid"]) | ~np.isfinite(out["ask"]), "nonfinite_quote"),
        (out["bid"] <= 0, "nonpositive_bid"),
        (out["ask"] <= 0, "nonpositive_ask"),
        (out["bid"] > out["ask"], "crossed_market"),
        ((out["bid_size"] < 0) | (out["ask_size"] < 0), "negative_size"),
    )
    for invalid, label in checks:
        reason.loc[reason.isna() & invalid.fillna(False)] = label
    out["quote_valid"] = reason.isna()
    out["exclusion_reason"] = reason
    out["last_trade_ts"] = pd.to_datetime(
        raw.get("ts_event", pd.Series(pd.NaT, index=raw.index)), utc=True, errors="coerce"
    ).astype("datetime64[ns, UTC]")
    out["last_trade_price"] = pd.to_numeric(
        raw.get("price", pd.Series(np.nan, index=raw.index)), errors="coerce"
    ).astype(float)
    out["last_trade_size"] = pd.to_numeric(
        raw.get("size", pd.Series(pd.NA, index=raw.index, dtype="Int64")), errors="raise"
    ).astype("Int64")
    return out


def _member_date(name: str) -> date | None:
    match = _MEMBER_RE.search(name)
    return None if match is None else datetime.strptime(match.group(1), "%Y%m%d").date()


def inspect_archive(path: str | Path) -> dict:
    """Inventory ZIP members without extracting or decoding the daily DBNs."""
    archive = Path(path).resolve()
    days = []
    with zipfile.ZipFile(archive) as source:
        metadata = json.loads(source.read("metadata.json")) if "metadata.json" in source.namelist() else None
        for info in source.infolist():
            session = _member_date(info.filename)
            if session is not None:
                days.append({
                    "session_date": session.isoformat(), "member": info.filename,
                    "crc32": f"{info.CRC:08x}", "size_bytes": info.file_size,
                    "zip_compressed_bytes": info.compress_size,
                })
    days.sort(key=lambda item: (item["session_date"], item["member"]))
    dates = [item["session_date"] for item in days]
    if len(dates) != len(set(dates)):
        raise ValueError("Archive contains more than one CBBO member for the same session date")
    if not days:
        raise ValueError("Archive contains no daily CBBO-1m .dbn.zst members")
    return {
        "path": str(archive), "size_bytes": archive.stat().st_size,
        "day_count": len(days), "first_session": dates[0], "last_session": dates[-1],
        "dbn_bytes": sum(item["size_bytes"] for item in days),
        "metadata": metadata, "days": days,
    }


def _iter_dbn_frames(raw: bytes, chunk_size: int) -> Iterator[pd.DataFrame]:
    try:
        import databento as db
    except ImportError as exc:
        raise ImportError("Archive preparation requires the optional 'databento' dependency") from exc
    yield from db.DBNStore.from_bytes(raw).to_df(count=chunk_size)


class _DayStats:
    """Exact duplicate counts for time-sorted CBBO streams with bounded memory."""

    def __init__(self, session: date | str | None = None):
        self.raw_rows = self.valid_rows = self.regular_rows = self.valid_regular_rows = 0
        self.duplicates = 0
        self.contracts: set[str] = set()
        self.expiries: set[date] = set()
        self.exclusions: Counter = Counter()
        self.minimum = self.maximum = None
        self.boundary_time = None
        self.boundary_contracts: set[str] = set()
        self.regular_minutes: set[int] = set()
        self.valid_regular_minutes: set[int] = set()
        session = date.fromisoformat(session) if isinstance(session, str) else session
        bounds = None if session is None else calendar.session_bounds(session)
        self.expected_minutes = 0 if bounds is None else int((bounds[1] - bounds[0]).total_seconds() / 60)

    def update(self, frame: pd.DataFrame) -> None:
        timestamps = frame["timestamp"].dropna()
        if not timestamps.empty:
            if not timestamps.is_monotonic_increasing or (
                self.maximum is not None and timestamps.iloc[0] < self.maximum
            ):
                raise ValueError("CBBO ts_recv must be nondecreasing for exact streaming duplicate checks")
            self.minimum = timestamps.iloc[0] if self.minimum is None else self.minimum
            self.maximum = timestamps.iloc[-1]
        self.raw_rows += len(frame)
        self.valid_rows += int(frame["quote_valid"].sum())
        self.regular_rows += int(frame["in_regular_session"].sum())
        self.valid_regular_rows += int((frame["quote_valid"] & frame["in_regular_session"]).sum())
        self.regular_minutes.update(frame.loc[frame["in_regular_session"], "timestamp"].dropna().unique())
        self.valid_regular_minutes.update(frame.loc[frame["quote_valid"] & frame["in_regular_session"], "timestamp"].dropna().unique())
        self.contracts.update(frame["contract_id"].dropna())
        self.expiries.update(frame["expiry_date"].dropna())
        self.exclusions.update(frame["exclusion_reason"].dropna().value_counts().to_dict())
        keys = frame.loc[frame["timestamp"].notna() & frame["contract_id"].notna(), ["timestamp", "contract_id"]]
        if keys.empty:
            return
        self.duplicates += int(keys.duplicated().sum())
        unique = keys.drop_duplicates()
        if self.boundary_time is not None:
            overlapping = unique.loc[unique["timestamp"] == self.boundary_time, "contract_id"]
            self.duplicates += sum(contract in self.boundary_contracts for contract in overlapping)
        last_time = keys["timestamp"].iloc[-1]
        last_contracts = set(unique.loc[unique["timestamp"] == last_time, "contract_id"])
        if last_time == self.boundary_time:
            self.boundary_contracts.update(last_contracts)
        else:
            self.boundary_time = last_time
            self.boundary_contracts = last_contracts

    def report(self) -> dict:
        return {
            "raw_rows": self.raw_rows, "rows_written": self.raw_rows,
            "quote_valid_rows": self.valid_rows, "regular_session_rows": self.regular_rows,
            "valid_regular_session_rows": self.valid_regular_rows,
            "expected_regular_session_minutes": self.expected_minutes,
            "regular_session_minute_count": len(self.regular_minutes),
            "valid_regular_session_minute_count": len(self.valid_regular_minutes),
            "contract_count": len(self.contracts), "expiry_count": len(self.expiries),
            "expiry_dates": sorted(value.isoformat() for value in self.expiries),
            "timestamp_min": None if self.minimum is None else self.minimum.isoformat(),
            "timestamp_max": None if self.maximum is None else self.maximum.isoformat(),
            "duplicate_keys": self.duplicates,
            "duplicate_key_definition": "(timestamp, contract_id); non-null parsed keys only; extras beyond first",
            "exclusion_counts": dict(sorted(self.exclusions.items())),
        }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprints(chunk_size: int) -> tuple[str, str]:
    schema = hashlib.sha256(option_schema().serialize().to_pybytes()).hexdigest()
    config = {
        "schema_version": SCHEMA_VERSION, "chunk_size": chunk_size,
        "ingest_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calendar_source": hashlib.sha256(Path(calendar.__file__).read_bytes()).hexdigest(),
        "schema_fingerprint": schema,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), schema


def _can_resume(path: Path, report: dict, source: dict, config: str, schema: str) -> bool:
    if (report.get("status") != "complete" or report.get("source") != source
            or report.get("config_fingerprint") != config or report.get("schema_fingerprint") != schema
            or not path.is_file() or path.stat().st_size != report.get("output_bytes")):
        return False
    import pyarrow.parquet as pq

    try:
        existing = pq.ParquetFile(path)
        return (existing.metadata.num_rows == report.get("rows_written")
                and existing.schema_arrow.equals(option_schema(), check_metadata=True))
    except (OSError, ValueError):
        return False


def _prepare_day(
    archive_path: str, member: dict, output_dir: str, config: str, schema: str, chunk_size: int
) -> tuple[dict, str]:
    """Prepare one independent partition; safe to invoke in a spawned process."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    start = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(member["member"])
    source = {**member, "sha256": hashlib.sha256(raw).hexdigest()}
    partition = Path(output_dir) / "options" / f"session_date={member['session_date']}"
    partition.mkdir(parents=True, exist_ok=True)
    parquet_path = partition / "quotes.parquet"
    manifest_path = partition / "manifest.json"
    old = None
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            pass
    if old is not None and _can_resume(parquet_path, old, source, config, schema):
        return old, "resumed"
    temporary = partition / f".quotes.{uuid4().hex}.parquet.tmp"
    stats = _DayStats(member["session_date"])
    try:
        with pq.ParquetWriter(temporary, option_schema(), compression="zstd", compression_level=6) as writer:
            for frame in _iter_dbn_frames(raw, chunk_size):
                normalized = normalize_quotes(frame, member["session_date"])
                stats.update(normalized)
                writer.write_table(pa.Table.from_pandas(normalized, schema=option_schema(), preserve_index=False))
        temporary.replace(parquet_path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "session_date": member["session_date"], "source": source,
        "config_fingerprint": config, "schema_fingerprint": schema,
        "output_path": str(parquet_path), "output_bytes": parquet_path.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 3), **stats.report(),
    }
    _atomic_json(manifest_path, report)
    return report, "prepared"


def prepare_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    max_days: int | None = None,
    chunk_size: int = 250_000,
    workers: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Prepare every daily partition, or an explicitly requested pilot subset.

    Each worker holds one compressed DBN member and one dataframe chunk.
    Completed partitions resume only after source SHA256, normalizer/calendar
    configuration, schema, output existence, size and row count all match.
    Individual days and the aggregate manifest are written atomically.  Up to
    four spawned processes can prepare separate days; result aggregation and
    progress callbacks run only in the parent, in archive order.  Multiprocess
    callers must use Python's normal ``if __name__ == '__main__'`` guard.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if max_days is not None and (isinstance(max_days, bool) or not isinstance(max_days, int) or max_days <= 0):
        raise ValueError("max_days must be a positive integer or None")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("workers must be a positive integer between 1 and 4")

    inventory = inspect_archive(archive_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = inventory["days"] if max_days is None else inventory["days"][:max_days]
    config, schema = _fingerprints(chunk_size)
    reports = []
    processed = resumed = 0
    with ExitStack() as stack:
        runner = map
        if workers > 1:
            executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=min(workers, len(selected)),
                mp_context=multiprocessing.get_context("spawn"),
            ))
            runner = executor.map
        results = runner(
            _prepare_day, repeat(str(Path(archive_path).resolve())), selected,
            repeat(str(destination)), repeat(config), repeat(schema), repeat(chunk_size),
        )
        for report, status in results:
            reports.append(report)
            processed += status == "prepared"
            resumed += status == "resumed"
            if on_progress:
                on_progress({"session_date": report["session_date"], "status": status, "raw_rows": report["raw_rows"], "valid_regular_session_rows": report["valid_regular_session_rows"], "output_bytes": report["output_bytes"], "elapsed_seconds": report["elapsed_seconds"]})
    result = {
        "schema_version": SCHEMA_VERSION, "archive": inventory,
        "requested_day_count": len(selected), "completed_day_count": len(reports),
        "processed_day_count": processed, "resumed_day_count": resumed,
        "workers": workers,
        "is_full_archive_complete": len(reports) == inventory["day_count"],
        "raw_rows": sum(day["raw_rows"] for day in reports),
        "valid_regular_session_rows": sum(day["valid_regular_session_rows"] for day in reports),
        "duplicate_keys": sum(day["duplicate_keys"] for day in reports),
        "output_bytes": sum(day["output_bytes"] for day in reports), "days": reports,
    }
    _atomic_json(destination / "manifest.json", result)
    return result
