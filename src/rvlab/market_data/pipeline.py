"""Prepare auditable market inputs and report what is still missing."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .calendar import sessions
from .ingest import prepare_archive
from .references import (align_stock_prices, load_dividends, load_fred_rates,
                         load_rates, load_stock_prices, parse_ssga_dividends,
                         reference_coverage)
from .snapshots import (
    CALIBRATION_BACKEND, SNAPSHOT_SCHEMA_VERSION, build_calibration_snapshot,
)


def _json_default(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def _fingerprint(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    clean = frame.copy()
    clean.attrs = {}
    try:
        clean.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_option_window(path, as_of, *, lookback="1min") -> pd.DataFrame:
    """Read only recent, identifiable contracts, retaining bad latest quotes.

    Explicit file datasets avoid Hive's string/date partition-type inference.
    Do not filter quote_valid here: doing so could resurrect an older quote.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    instant = pd.Timestamp(as_of)
    if instant.tzinfo is None or pd.isna(instant):
        raise ValueError("as_of must be a timezone-aware timestamp")
    instant = instant.tz_convert("UTC")
    age = pd.Timedelta(lookback)
    if age <= pd.Timedelta(0):
        raise ValueError("lookback must be positive")
    scalar = lambda ts: pa.scalar(ts.to_pydatetime(), type=pa.timestamp("ns", tz="UTC"))
    expression = ((ds.field("timestamp") >= scalar(instant - age)) &
                  (ds.field("timestamp") <= scalar(instant)) &
                  ds.field("contract_id").is_valid() &
                  ds.field("expiry_at").is_valid() & ds.field("is_call").is_valid())
    result = ds.dataset(str(path), format="parquet", partitioning=None).to_table(
        filter=expression).to_pandas()
    result["exercise_style"] = "american"
    return result


def snapshot_manifest_current(manifest: Mapping) -> bool:
    """Reject superseded European-screened snapshot caches.

    A fresh audit always rebuilds snapshots. Consumers reading a saved snapshot
    must verify this marker before interpreting its readiness flag.
    """
    return bool(manifest.get("schema_version") == SNAPSHOT_SCHEMA_VERSION
                and manifest.get("calibration_backend") == CALIBRATION_BACKEND
                and manifest.get("exercise_style") == "american")


def _reference_inputs(raw_dir, stock_path, rates_path, dividends_path,
                      stock_timestamp_convention, stock_source, stock_price_basis):
    raw = Path(raw_dir) if raw_dir is not None else None
    paths = {"stock": stock_path, "rates": rates_path, "dividends": dividends_path}
    for name, stem in (("stock", "stock_prices"), ("rates", "rates"), ("dividends", "dividends")):
        if paths[name] is None and raw is not None:
            paths[name] = next((raw / f"{stem}{ext}" for ext in (".parquet", ".csv")
                                if (raw / f"{stem}{ext}").is_file()), None)
    result = {"stock": None, "rates": None, "dividends": None}
    errors, sources = {}, {}
    for name in result:
        path = None if paths[name] is None else Path(paths[name])
        try:
            if path is not None:
                if name == "stock":
                    result[name] = load_stock_prices(path, timestamp_convention=stock_timestamp_convention,
                        source=stock_source, price_basis=stock_price_basis)
                else:
                    result[name] = (load_rates if name == "rates" else load_dividends)(path)
                sources[name] = [_fingerprint(path)]
            elif name == "rates" and raw is not None and (raw / "SOFR.csv").is_file():
                result[name] = load_fred_rates(raw)
                sources[name] = [_fingerprint(raw / f"{s}.csv") for s in ("SOFR", "DGS1MO", "DGS3MO", "DGS6MO")]
            elif name == "dividends" and raw is not None and (raw / "ssga_historical_distributions.xlsx").is_file():
                result[name] = parse_ssga_dividends(raw / "ssga_historical_distributions.xlsx")
                sources[name] = [_fingerprint(raw / "ssga_historical_distributions.xlsx")]
        except (ValueError, TypeError, OSError, ImportError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return result, errors, sources


def _stock_coverage(calendar, stock):
    rows = []
    for day in calendar.itertuples(index=False):
        expected = pd.date_range(day.open + pd.Timedelta("1min"), day.close, freq="1min")
        if stock is None or stock.empty:
            valid, quoted = 0, 0
        else:
            aligned = align_stock_prices(expected, stock)
            valid = int(aligned.stock_valid.sum())
            quoted = int((aligned.stock_valid & aligned.bid.notna() & aligned.ask.notna()).sum()) if {"bid", "ask"} <= set(aligned) else 0
        rows.append({"session_date": day.session_date, "expected_stock_minutes": len(expected),
                     "covered_stock_minutes": valid, "covered_stock_quote_minutes": quoted,
                     "missing_stock_minutes": len(expected) - valid})
    return pd.DataFrame(rows)


def _partition_errors(days):
    """Check partition existence, footer row counts, size and schema before reuse."""
    import pyarrow.parquet as pq
    from .ingest import option_schema
    errors = {}
    for day, record in days.items():
        try:
            path = Path(record["output_path"])
            table = pq.ParquetFile(path)
            if path.stat().st_size != record["output_bytes"]:
                raise ValueError("partition size differs from manifest")
            if table.metadata.num_rows != record["rows_written"]:
                raise ValueError("partition row count differs from manifest")
            if not table.schema_arrow.equals(option_schema(), check_metadata=True):
                raise ValueError("partition schema differs from normalizer")
        except (OSError, ValueError, KeyError) as exc:
            errors[day] = f"{type(exc).__name__}: {exc}"
    return errors


def _financing_coverage(calendar, rates):
    if rates is None or rates.empty:
        return 0
    sofr = rates.loc[rates.series == "SOFR"]
    covered = 0
    for day in calendar.itertuples(index=False):
        as_of = day.close - pd.Timedelta("15min")
        usable = sofr.loc[sofr.available_at <= as_of]
        if not usable.empty:
            observation = pd.to_datetime(usable.observation_date, utc=True)
            covered += int(((as_of.normalize() - observation).between(
                pd.Timedelta(0), pd.Timedelta("7D"))).any())
    return covered


def _option_minute_gaps(calendar, days, partition_errors):
    """Locate source-wide minute gaps; never impute missing option quotes."""
    import pyarrow.parquet as pq
    result = []
    for day in calendar.itertuples(index=False):
        record = days.get(day.session_date)
        if record is None or record.get("valid_regular_session_minute_count", 0) >= record.get("expected_regular_session_minutes", 0):
            continue
        gap = {"session_date": day.session_date,
               "expected": record.get("expected_regular_session_minutes"),
               "observed": record.get("regular_session_minute_count"),
               "valid": record.get("valid_regular_session_minute_count")}
        if day.session_date not in partition_errors:
            frame = pq.ParquetFile(record["output_path"]).read(
                columns=["timestamp", "in_regular_session", "quote_valid"]).to_pandas()
            expected = pd.date_range(day.open + pd.Timedelta("1min"), day.close, freq="1min")
            regular = frame.loc[frame.in_regular_session]
            observed = pd.DatetimeIndex(regular.timestamp.unique())
            valid = pd.DatetimeIndex(regular.loc[regular.quote_valid, "timestamp"].unique())
            gap["missing_observation_endpoints"] = expected.difference(observed).tolist()
            gap["missing_valid_quote_endpoints"] = expected.difference(valid).tolist()
        result.append(gap)
    return result


def _markdown(report):
    refs = report["references"]
    lines = ["# SPY data readiness", "", f"Generated: {report['created_at']}", "",
        f"Archive normalized: **{report['archive_complete']}**. Option coverage checks passed: **{report['options_complete']}**. Calibration ready: **{report['calibration_ready']}**. Hedging ready: **{report['hedging_ready']}**.", "",
        "## Prepared archive", "",
        f"- Period: {report['start_session']} through {report['end_session']}.",
        f"- Sessions: {report['completed_sessions']} / {report['expected_sessions']} expected exchange sessions.",
        f"- Original rows preserved: {report['option_rows']:,}; valid regular-session rows: {report['valid_regular_session_rows']:,}.",
        f"- Regular-session minute endpoints with valid option quotes: {report['option_minutes_covered']:,} / {report['option_minutes_required']:,}.",
        f"- Duplicate contract/timestamp keys: {report['duplicate_keys']:,}.",
        f"- Normalized option storage: {report['option_output_bytes'] / 1e9:.3f} GB.",
        f"- Missing or unexpected sessions: {report['missing_sessions']} / {report['unexpected_sessions']}.", "",
        "## Supporting inputs", "", "| Input | Present | Rows |", "|---|---:|---:|"]
    if report["option_minute_gaps"]:
        position = lines.index("## Supporting inputs")
        gap_lines = ["## Option observation gaps", ""]
        for gap in report["option_minute_gaps"]:
            endpoints = ", ".join(str(t) for t in gap.get("missing_observation_endpoints", []))
            gap_lines.append(f"- {gap['session_date']}: {gap['valid']} / {gap['expected']} minute endpoints with valid quotes. Missing observations (UTC): {endpoints or 'see partition errors'}.")
        lines[position:position] = gap_lines + [""]
    for name in ("stock", "rates", "dividends"):
        lines.append(f"| {name} | {refs[name]['present']} | {refs[name]['rows']:,} |")
    lines.extend(["", f"Independent stock minutes available: {report['stock_minutes_covered']:,} / {report['stock_minutes_required']:,}.",
        f"Sessions with usable financing observations: {report['financing_sessions_covered']} / {report['expected_sessions']}.",
        f"Verified historical rate vintages: {report['point_in_time_rates_available']}.",
        f"Daily candidate snapshots: {report['snapshot_count']}; snapshots passing configured preparation: {report['ready_snapshot_count']}.",
        "", "## Missing inputs and remaining conditions", ""])
    lines.extend(f"- {item}" for item in report["missing_inputs"])
    if not report["missing_inputs"]:
        lines.append("- No blocking inputs under the recorded preparation assumptions.")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## Reproduce", "", "See the market_data package README for preparation, stock import, acquisition, and strict-readiness commands.",
        "The adjacent JSON report contains per-session coverage, source hashes, snapshot provenance, and all readiness flags.", ""])
    return "\n".join(lines)


def audit_prepared_dataset(output_dir, *, raw_references=None, stock_path=None,
        rates_path=None, dividends_path=None, stock_timestamp_convention=None,
        stock_source=None, stock_price_basis=None, allow_dividend_proxy=False,
        dividend_coverage_through=None, report_path=None, on_progress=None):
    """Refresh reference coverage and daily snapshots without decoding DBN again."""
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared archive manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    inventory = manifest["archive"]
    archive_dates = [item["session_date"] for item in inventory["days"]]
    if not archive_dates:
        raise ValueError("Prepared archive inventory has no sessions")
    calendar = sessions(min(archive_dates), max(archive_dates))
    days = {item["session_date"]: item for item in manifest["days"]}
    partition_errors = _partition_errors(days)
    expected = set(calendar.session_date)
    references, errors, sources = _reference_inputs(raw_references, stock_path,
        rates_path, dividends_path, stock_timestamp_convention, stock_source, stock_price_basis)
    reference_dir = output / "references"
    _parquet(reference_dir / "sessions.parquet", calendar)
    for name, frame in references.items():
        path = reference_dir / f"{name}.parquet"
        if frame is not None:
            _parquet(path, frame)
        else:
            path.unlink(missing_ok=True)
    coverage = _stock_coverage(calendar, references["stock"])
    _parquet(output / "stock_coverage.parquet", coverage)
    snapshot_reports, snapshot_missing = [], set()
    for day in calendar.itertuples(index=False):
        if day.session_date not in days or day.session_date in partition_errors:
            continue
        record = days[day.session_date]
        option_path = Path(record["output_path"])
        as_of = day.close - pd.Timedelta("15min")
        options = read_option_window(option_path, as_of)
        snapshot = build_calibration_snapshot(options, as_of, stock=references["stock"],
            rates=references["rates"], dividends=references["dividends"],
            allow_dividend_proxy=allow_dividend_proxy,
            dividend_coverage_through=dividend_coverage_through)
        snapshot_missing.update(snapshot.missing_inputs)
        partition = output / "snapshots" / f"session_date={day.session_date}"
        quotes = snapshot.quotes.copy()
        quotes["forward"], quotes["discount"] = snapshot.forward, snapshot.discount
        quotes["snapshot_as_of"] = as_of
        quotes["calibration_ready"] = snapshot.ready
        _parquet(partition / "quotes.parquet", quotes)
        _parquet(partition / "exclusions.parquet", snapshot.exclusions)
        item = {"schema_version": SNAPSHOT_SCHEMA_VERSION,
                "calibration_backend": CALIBRATION_BACKEND, "exercise_style": "american",
                "session_date": day.session_date, "as_of": as_of, "ready": snapshot.ready,
                "quote_count": len(quotes), "excluded_count": len(snapshot.exclusions),
                "missing_inputs": snapshot.missing_inputs, "provenance": dict(snapshot.provenance),
                "path": str(partition / "quotes.parquet")}
        _json(partition / "manifest.json", item)
        snapshot_reports.append(item)
        if on_progress:
            on_progress({"stage": "snapshot", "session_date": day.session_date,
                         "candidate_quotes": len(quotes), "ready": snapshot.ready})
    refs = reference_coverage(**references)
    expected_minutes = int(coverage.expected_stock_minutes.sum())
    stock_minutes = int(coverage.covered_stock_minutes.sum())
    stock_quote_minutes = int(coverage.covered_stock_quote_minutes.sum())
    minute_gaps = _option_minute_gaps(calendar, days, partition_errors)
    missing_sessions = sorted(expected - set(days))
    unexpected = sorted(set(days) - expected)
    archive_complete = bool(manifest["is_full_archive_complete"] and not missing_sessions
                            and not unexpected and not partition_errors)
    options_complete = bool(archive_complete and not manifest["duplicate_keys"] and not minute_gaps)
    ready_count = sum(item["ready"] for item in snapshot_reports)
    calibration_ready = options_complete and ready_count == len(calendar)
    missing = [f"Invalid {name} input: {error}" for name, error in errors.items()]
    if not options_complete:
        missing.append("Full option coverage/quality checks have not passed; inspect session and minute gaps.")
    missing.extend(f"Invalid partition {day}: {error}" for day, error in partition_errors.items())
    if stock_minutes < expected_minutes:
        missing.append(f"Independent unadjusted SPY stock prices missing for {expected_minutes - stock_minutes:,} regular-session minutes.")
    if stock_quote_minutes < expected_minutes:
        missing.append("Executable SPY stock bid/ask coverage is incomplete; stock trading costs remain an assumption.")
    missing.extend(sorted(snapshot_missing))
    financing_sessions = _financing_coverage(calendar, references["rates"])
    if financing_sessions < len(calendar):
        missing.append(f"SOFR financing observations are missing or stale for {len(calendar) - financing_sessions} sessions.")
    point_in_time_rates = bool(refs["rates"].get("point_in_time_vintages", False))
    if not point_in_time_rates:
        missing.append("Verified historical rate vintages are unavailable; latest FRED values remain research proxies for historical replay.")
    hedging_ready = bool(calibration_ready and stock_minutes == expected_minutes
                         and stock_quote_minutes == expected_minutes
                         and financing_sessions == len(calendar)
                         and point_in_time_rates)
    option_exclusions = Counter()
    for record in days.values():
        option_exclusions.update(record.get("exclusion_counts", {}))
    report = {"schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "calibration_backend": CALIBRATION_BACKEND, "exercise_style": "american",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output), "archive_manifest": str(manifest_path),
        "archive_complete": archive_complete,
        "options_complete": options_complete, "calibration_ready": bool(calibration_ready),
        "hedging_ready": hedging_ready, "expected_sessions": len(calendar),
        "start_session": min(archive_dates), "end_session": max(archive_dates),
        "completed_sessions": len(days), "missing_sessions": missing_sessions,
        "unexpected_sessions": unexpected, "option_minute_gaps": minute_gaps,
        "partition_errors": partition_errors,
        "option_rows": manifest["raw_rows"],
        "valid_regular_session_rows": manifest["valid_regular_session_rows"],
        "duplicate_keys": manifest["duplicate_keys"], "option_output_bytes": manifest["output_bytes"],
        "option_exclusion_counts": dict(option_exclusions),
        "option_minutes_required": expected_minutes,
        "option_minutes_covered": sum(d.get("valid_regular_session_minute_count", 0) for d in days.values()),
        "stock_minutes_required": expected_minutes, "stock_minutes_covered": stock_minutes,
        "stock_quote_minutes_covered": stock_quote_minutes, "references": refs,
        "financing_sessions_covered": financing_sessions,
        "point_in_time_rates_available": point_in_time_rates,
        "hindsight_dividend_exclusion_count": 0,
        "reference_errors": errors, "reference_sources": sources,
        "snapshot_count": len(snapshot_reports), "ready_snapshot_count": ready_count,
        "allow_dividend_proxy": allow_dividend_proxy,
        "dividend_coverage_through": dividend_coverage_through,
        "runtime_versions": {name: version(name) for name in (
            "numpy", "pandas", "pyarrow", "exchange-calendars")},
        "missing_inputs": list(dict.fromkeys(missing)),
        "limitations": [
            "CBBO timestamps identify interval-end observations, not the age of the last quote change; last-trade time is not quote freshness.",
            "The standard unadjusted SPY convention uses American physical-settlement options and a 100-share multiplier; non-SPY roots are ineligible. Expiry close is a pricing convention, not an exercise deadline.",
            "Treasury yields are explicitly labelled discount-curve proxies; SOFR is financing only. Latest FRED values and conservative availability lags are not historical vintages.",
            "Historical State Street distributions supply realised cash flows. Their payment-date availability proxy is not a declaration timestamp or a future dividend forecast.",
            "American calibration requires a historically available cash-dividend schedule with explicit completeness coverage; trailing distribution yields are unsupported. Missing expectations are not zero dividends.",
            "Observed American bid/ask quotes are passed directly to American BS/Heston pricing, including ITM and known dividend-crossing contracts. No parity-derived forward, European-bound filter, ex-post ex-date exclusion or exercise-premium conversion is used.",
            "Snapshot schema 2 supersedes prior European-screened snapshots; consumers must reject stale snapshot manifests before trusting readiness. Canonical option observations remain unchanged.",
            "A 15-minute-before-close snapshot is materialized for each session; the full interval-end option history remains available for future policies.",
            "Data readiness is distinct from successful model calibration and from economic or trading validation.",
        ], "stock_coverage": coverage.to_dict("records"), "snapshots": snapshot_reports}
    _json(output / "readiness.json", report)
    _atomic_text(output / "readiness.md", _markdown(report))
    if report_path is not None:
        destination = Path(report_path)
        _atomic_text(destination, _markdown(report))
        _json(destination.with_suffix(".json"), report)
    return report


def prepare_spy_dataset(archive_path, output_dir, *, max_days=None,
                        chunk_size=250_000, workers=2, **audit_options):
    """Normalize the archive, then refresh references, snapshots and readiness."""
    prepare_archive(archive_path, output_dir, max_days=max_days,
        chunk_size=chunk_size, workers=workers,
        on_progress=audit_options.get("on_progress"))
    return audit_prepared_dataset(output_dir, **audit_options)


__all__ = ["prepare_spy_dataset", "audit_prepared_dataset", "read_option_window",
           "snapshot_manifest_current"]
