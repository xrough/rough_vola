"""Prepared artifacts, missing inputs, and safe calibration handoff."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rvlab.market_data.calendar import session_bounds
from rvlab.market_data.ingest import normalize_quotes, option_schema
from rvlab.market_data.pipeline import (
    audit_prepared_dataset, read_option_window, snapshot_manifest_current,
)
from rvlab.market_data.__main__ import main


def _prepared(tmp_path):
    opening, closing = session_bounds("2025-08-07")
    times = pd.date_range(opening + pd.Timedelta("1min"), closing, freq="1min")
    raw = pd.DataFrame({"symbol": "SPY   250822C00100000", "instrument_id": 1,
        "bid_px_00": .9, "ask_px_00": 1.1, "bid_sz_00": 10, "ask_sz_00": 10}, index=times)
    normalized = normalize_quotes(raw, "2025-08-07")
    partition = tmp_path / "options/session_date=2025-08-07/quotes.parquet"
    partition.parent.mkdir(parents=True)
    table = pa.Table.from_pandas(normalized, schema=option_schema(), preserve_index=False)
    pq.write_table(table.replace_schema_metadata(option_schema().metadata), partition)
    day = {"session_date": "2025-08-07", "output_path": str(partition),
        "output_bytes": partition.stat().st_size, "rows_written": len(raw),
        "raw_rows": len(raw), "expected_regular_session_minutes": 390,
        "regular_session_minute_count": 390, "valid_regular_session_minute_count": 390}
    manifest = {"archive": {"days": [{"session_date": "2025-08-07"}]},
        "days": [day], "is_full_archive_complete": True, "raw_rows": len(raw),
        "valid_regular_session_rows": len(raw), "duplicate_keys": 0,
        "output_bytes": partition.stat().st_size}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return partition, opening, closing


def _references(tmp_path, opening, closing, *, stock_quotes=True):
    refs = tmp_path / "raw"
    refs.mkdir()
    times = pd.date_range(opening, closing - pd.Timedelta("1min"), freq="1min")
    stock = pd.DataFrame({"timestamp": times, "spot": 99., "source": "test",
        "price_basis": "raw", "timestamp_convention": "minute_bar_start"})
    if stock_quotes:
        stock["bid"], stock["ask"] = 98.99, 99.01
    stock.to_csv(refs / "stock_prices.csv", index=False)
    pd.DataFrame({"observation_date": ["2025-08-05"] * 4,
        "available_at": ["2025-08-06T21:00:00Z"] * 4,
        "tenor_days": [1, 30, 91, 182], "rate": [.04] * 4,
        "series": ["SOFR", "DGS1MO", "DGS3MO", "DGS6MO"], "source": "test",
        "vintage": "point_in_time"}).to_csv(refs / "rates.csv", index=False)
    pd.DataFrame({"ex_date": ["2024-09-20", "2024-12-20", "2025-03-21", "2025-06-20"],
        "pay_date": ["2024-10-31", "2025-01-31", "2025-04-30", "2025-07-31"],
        "amount": [1.] * 4, "source": "test",
        "available_at": ["2024-10-31T20:00:00Z", "2025-01-31T21:00:00Z", "2025-04-30T20:00:00Z", "2025-07-31T20:00:00Z"]}).to_csv(refs / "dividends.csv", index=False)
    return refs


def test_missing_inputs_are_reported_without_synthetic_fallback(tmp_path):
    _prepared(tmp_path)
    report = audit_prepared_dataset(tmp_path)
    assert report["options_complete"]
    assert not report["calibration_ready"] and not report["hedging_ready"]
    assert report["stock_minutes_required"] == 390
    assert report["stock_minutes_covered"] == 0
    assert report["ready_snapshot_count"] == 0
    assert any("stock" in item for item in report["missing_inputs"])
    assert (tmp_path / "readiness.json").is_file()
    snapshot = pq.ParquetFile(tmp_path / "snapshots/session_date=2025-08-07/quotes.parquet").read().to_pandas()
    assert len(snapshot) == 1 and not snapshot.calibration_ready.any()
    assert snapshot.forward.isna().all()
    assert snapshot.exercise_style.eq("american").all()
    manifest = json.loads((tmp_path / "snapshots/session_date=2025-08-07/manifest.json").read_text())
    assert snapshot_manifest_current(manifest)
    assert manifest["calibration_backend"] == "american_vanilla_fd"


def test_trailing_dividend_proxy_does_not_enable_american_calibration(tmp_path):
    _, opening, closing = _prepared(tmp_path)
    refs = _references(tmp_path, opening, closing)
    report = audit_prepared_dataset(tmp_path, raw_references=refs, allow_dividend_proxy=True)
    assert report["stock_minutes_covered"] == 390
    assert report["stock_quote_minutes_covered"] == 390
    assert report["financing_sessions_covered"] == 1
    assert not report["calibration_ready"] and not report["hedging_ready"]
    assert any("unsupported for American" in item for item in report["missing_inputs"])
    assert report["references"]["dividends"]["historical_expectations_available"] is False
    assert (tmp_path / "references/stock.parquet").is_file()


def test_price_only_stock_does_not_claim_executable_hedging_ready(tmp_path):
    _, opening, closing = _prepared(tmp_path)
    refs = _references(tmp_path, opening, closing, stock_quotes=False)
    report = audit_prepared_dataset(tmp_path, raw_references=refs,
        dividend_coverage_through="2025-08-22")
    assert report["calibration_ready"]
    assert not report["hedging_ready"]
    assert report["stock_quote_minutes_covered"] == 0


def test_latest_rate_values_do_not_claim_historical_hedging_ready(tmp_path):
    _, opening, closing = _prepared(tmp_path)
    refs = _references(tmp_path, opening, closing)
    path = refs / "rates.csv"
    pd.read_csv(path).assign(vintage="latest_not_point_in_time").to_csv(path, index=False)
    report = audit_prepared_dataset(tmp_path, raw_references=refs,
        dividend_coverage_through="2025-08-22")
    assert report["calibration_ready"] and not report["hedging_ready"]
    assert not report["point_in_time_rates_available"]
    assert any("vintages" in item for item in report["missing_inputs"])


def test_unknown_dividend_expectations_do_not_default_to_zero(tmp_path):
    _, opening, closing = _prepared(tmp_path)
    refs = _references(tmp_path, opening, closing)
    report = audit_prepared_dataset(tmp_path, raw_references=refs)
    assert not report["calibration_ready"]
    assert any("dividend" in item for item in report["missing_inputs"])


def test_cli_accepts_explicit_dividend_schedule_coverage(tmp_path):
    _, opening, closing = _prepared(tmp_path)
    refs = _references(tmp_path, opening, closing)
    destination = tmp_path / "report.md"
    # The caller explicitly asserts no additional expected event through expiry.
    assert main(["audit", "--output", str(tmp_path), "--raw-references", str(refs),
        "--dividend-coverage-through", "2025-08-22", "--require-ready",
        "--report", str(destination)]) == 0
    report = json.loads(destination.with_suffix(".json").read_text())
    assert report["calibration_ready"] and not report["allow_dividend_proxy"]
    assert report["dividend_coverage_through"] == "2025-08-22"
    assert report["schema_version"] == 2
    assert report["exercise_style"] == "american"


def test_known_empty_cash_schedule_and_complete_independent_inputs_allow_readiness(tmp_path):
    _, opening, closing = _prepared(tmp_path)
    refs = _references(tmp_path, opening, closing)
    report = audit_prepared_dataset(tmp_path, raw_references=refs,
        dividend_coverage_through="2025-08-22")
    assert report["calibration_ready"] and report["hedging_ready"]
    assert report["hindsight_dividend_exclusion_count"] == 0
    assert report["snapshots"][0]["provenance"]["exercise_policy"].startswith("direct American")


@pytest.mark.parametrize("manifest", [
    {"schema_version": 1, "ready": True},
    {"schema_version": 2, "calibration_backend": "european_fourier", "exercise_style": "american"},
    {"schema_version": 2, "calibration_backend": "american_vanilla_fd", "exercise_style": "european"},
])
def test_superseded_snapshot_cache_is_rejected_even_when_old_ready_flag_is_true(manifest):
    assert not snapshot_manifest_current(manifest)


def test_corrupted_partition_metadata_blocks_readiness(tmp_path):
    path, _, _ = _prepared(tmp_path)
    path.write_bytes(b"not parquet")
    report = audit_prepared_dataset(tmp_path)
    assert not report["options_complete"] and not report["calibration_ready"]
    assert "2025-08-07" in report["partition_errors"]
    assert report["snapshot_count"] == 0


def test_full_archive_can_contain_a_precisely_located_source_minute_gap(tmp_path):
    path, opening, _ = _prepared(tmp_path)
    missing = opening + pd.Timedelta("30min")
    frame = pq.ParquetFile(path).read().to_pandas()
    frame = frame.loc[frame.timestamp != missing]
    table = pa.Table.from_pandas(frame, schema=option_schema(), preserve_index=False)
    pq.write_table(table.replace_schema_metadata(option_schema().metadata), path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["days"][0].update(output_bytes=path.stat().st_size, rows_written=len(frame),
        regular_session_minute_count=389, valid_regular_session_minute_count=389)
    manifest_path.write_text(json.dumps(manifest))
    report = audit_prepared_dataset(tmp_path)
    assert report["archive_complete"] and not report["options_complete"]
    assert report["option_minute_gaps"][0]["missing_observation_endpoints"] == [missing]
    assert report["option_minute_gaps"][0]["missing_valid_quote_endpoints"] == [missing]


def test_recent_window_retains_bad_latest_quote_and_handles_hive_directory(tmp_path):
    path, _, closing = _prepared(tmp_path)
    frame = pq.ParquetFile(path).read().to_pandas()
    as_of = closing - pd.Timedelta("15min")
    frame.loc[frame.timestamp == as_of, "quote_valid"] = False
    frame.loc[frame.timestamp == as_of, "exclusion_reason"] = "crossed_market"
    table = pa.Table.from_pandas(frame, schema=option_schema(), preserve_index=False)
    pq.write_table(table.replace_schema_metadata(option_schema().metadata), path)
    selected = read_option_window(path, as_of)
    assert len(selected) == 2
    assert not selected.loc[selected.timestamp == as_of, "quote_valid"].iloc[0]
    with pytest.raises(ValueError, match="timezone-aware"):
        read_option_window(path, "2025-08-07 19:45")


def test_require_ready_cli_is_nonzero_but_writes_audit(tmp_path):
    _prepared(tmp_path)
    report = tmp_path / "report.md"
    assert main(["audit", "--output", str(tmp_path), "--raw-references", str(tmp_path / "absent"),
                 "--report", str(report), "--require-ready"]) == 2
    assert report.is_file() and report.with_suffix(".json").is_file()


def test_missing_archive_manifest_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        audit_prepared_dataset(tmp_path)
