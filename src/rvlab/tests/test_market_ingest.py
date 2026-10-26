"""SPY normalization and bounded, resumable archive ingestion."""

from datetime import date
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from rvlab.market_data import ingest


def raw_frame(timestamps=None, **overrides):
    if timestamps is None:
        timestamps = ["2025-08-07T13:31:00Z"]
    count = len(timestamps)
    frame = pd.DataFrame({
        "symbol": ["SPY   250808C00640000"] * count,
        "instrument_id": [123] * count,
        "bid_px_00": [1.0] * count, "ask_px_00": [1.2] * count,
        "bid_sz_00": [5] * count, "ask_sz_00": [8] * count,
        "flags": [192] * count,
    }, index=pd.to_datetime(timestamps, utc=True))
    frame.index.name = "ts_recv"
    for key, value in overrides.items():
        frame[key] = value
    return frame


def make_archive(path: Path, dates=("20250807", "20250808")):
    with zipfile.ZipFile(path, "w") as source:
        source.writestr("metadata.json", json.dumps({"query": {"schema": "cbbo-1m", "symbols": ["SPY.OPT"]}}))
        source.writestr("ignore.txt", "not a DBN")
        for value in dates:
            source.writestr(f"opra-pillar-{value}.cbbo-1m.dbn.zst", value.encode())


def test_normalize_preserves_full_contracts_and_trade_timestamp():
    frame = raw_frame(
        ["2025-08-07T13:31:00Z"] * 3,
        symbol=["SPY   250808C00640000", "SPY250919P00630000", "SPY   271217C00850000"],
        ts_event=pd.to_datetime(["2025-08-07T12:00:00Z", None, "2025-08-07T13:30:59Z"], utc=True),
        price=[2.0, np.nan, 3.0], size=[1, 0, 2],
    )
    result = ingest.normalize_quotes(frame, date(2025, 8, 7))
    assert len(result) == 3
    assert result.contract_id.tolist() == ["SPY250808C00640000", "SPY250919P00630000", "SPY271217C00850000"]
    assert result.strike.tolist() == [640.0, 630.0, 850.0]
    assert result.is_call.tolist() == [True, False, True]
    assert result.expiry_date.tolist() == [date(2025, 8, 8), date(2025, 9, 19), date(2027, 12, 17)]
    assert result.expiry_at.iloc[0] == pd.Timestamp("2025-08-08T20:00:00Z")
    assert result.timestamp.equals(result.available_at)
    assert result.last_trade_ts.iloc[0] == pd.Timestamp("2025-08-07T12:00:00Z")
    assert result.timestamp.iloc[0] != result.last_trade_ts.iloc[0]
    assert result.quote_valid.all()
    assert result.in_regular_session.all()


@pytest.mark.parametrize("symbol,reason", [
    ("bad", "invalid_contract_symbol"),
    (None, "invalid_contract_symbol"),
    ("SPY   250230C00640000", "invalid_expiry_date"),
    ("SPX   250808C00640000", "unexpected_root"),
    ("SPY1  250808C00640000", "unexpected_root"),
])
def test_bad_contracts_remain_visible(symbol, reason):
    result = ingest.normalize_quotes(raw_frame(symbol=[symbol]), "2025-08-07")
    assert len(result) == 1
    assert not result.quote_valid.iloc[0]
    assert result.exclusion_reason.iloc[0] == reason
    if reason != "unexpected_root":
        assert pd.isna(result.contract_id.iloc[0])
        assert pd.isna(result.expiry_at.iloc[0])


@pytest.mark.parametrize("overrides,reason", [
    ({"bid_px_00": [np.nan]}, "nonfinite_quote"),
    ({"ask_px_00": [np.inf]}, "nonfinite_quote"),
    ({"bid_px_00": [0]}, "nonpositive_bid"),
    ({"ask_px_00": [-1]}, "nonpositive_ask"),
    ({"bid_px_00": [2]}, "crossed_market"),
    ({"bid_sz_00": [-1]}, "negative_size"),
])
def test_invalid_quotes_are_labelled_not_discarded(overrides, reason):
    result = ingest.normalize_quotes(raw_frame(**overrides), "2025-08-07")
    assert len(result) == 1
    assert result.exclusion_reason.iloc[0] == reason
    assert not result.quote_valid.iloc[0]


def test_zero_sizes_and_locked_quotes_remain_available_for_liquidity_screening():
    result = ingest.normalize_quotes(raw_frame(bid_sz_00=[0], ask_px_00=[1]), "2025-08-07")
    assert result.quote_valid.iloc[0]
    assert result.bid_size.iloc[0] == 0


@pytest.mark.parametrize("session,timestamps,expected", [
    ("2025-08-07", ["2025-08-07T13:30:00Z", "2025-08-07T13:31:00Z", "2025-08-07T20:00:00Z", "2025-08-07T20:01:00Z"], [False, True, True, False]),
    ("2025-12-01", ["2025-12-01T14:30:00Z", "2025-12-01T14:31:00Z", "2025-12-01T21:00:00Z"], [False, True, True]),
    ("2025-11-28", ["2025-11-28T18:00:00Z", "2025-11-28T18:01:00Z"], [True, False]),
    ("2025-09-01", ["2025-09-01T15:00:00Z"], [False]),
])
def test_session_end_timestamp_convention(session, timestamps, expected):
    result = ingest.normalize_quotes(raw_frame(timestamps), session)
    assert result.in_regular_session.tolist() == expected


def test_timestamp_column_and_invalid_timestamp():
    frame = raw_frame().reset_index()
    frame["ts_recv"] = "invalid"
    result = ingest.normalize_quotes(frame, "2025-08-07")
    assert pd.isna(result.timestamp.iloc[0])
    assert result.exclusion_reason.iloc[0] == "invalid_timestamp"


def test_missing_required_columns_and_timestamp_fail_explicitly():
    with pytest.raises(ValueError, match="ask_px_00"):
        ingest.normalize_quotes(raw_frame().drop(columns="ask_px_00"), "2025-08-07")
    with pytest.raises(ValueError, match="ts_recv"):
        ingest.normalize_quotes(raw_frame().reset_index(drop=True), "2025-08-07")


def test_stable_arrow_schema_supports_nullable_contract_metadata():
    pa = pytest.importorskip("pyarrow")
    normalized = ingest.normalize_quotes(raw_frame(symbol=["bad"]), "2025-08-07")
    table = pa.Table.from_pandas(normalized, schema=ingest.option_schema(), preserve_index=False)
    assert table.schema.equals(ingest.option_schema(), check_metadata=False)
    assert table["expiry_date"].null_count == 1
    assert table["is_call"].null_count == 1
    assert table["last_trade_ts"].null_count == 1


def test_archive_inventory_is_complete_sorted_and_not_decoded(tmp_path):
    path = tmp_path / "source.zip"
    make_archive(path, dates=("20250808", "20250807"))
    result = ingest.inspect_archive(path)
    assert result["day_count"] == 2
    assert result["first_session"] == "2025-08-07"
    assert result["last_session"] == "2025-08-08"
    assert result["days"][0]["size_bytes"] == 8
    assert len(result["days"][0]["crc32"]) == 8


def test_empty_and_duplicate_session_archives_rejected(tmp_path):
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as source:
        source.writestr("a.txt", "empty")
    with pytest.raises(ValueError, match="no daily"):
        ingest.inspect_archive(empty)
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as source:
        source.writestr("opra-pillar-20250807.cbbo-1m.dbn.zst", "one")
        source.writestr("sub/opra-pillar-20250807.cbbo-1m.dbn.zst", "two")
    with pytest.raises(ValueError, match="same session"):
        ingest.inspect_archive(duplicate)


def test_duplicate_and_minute_counts_cross_chunk_boundaries():
    stats = ingest._DayStats("2025-08-07")
    first = raw_frame(["2025-08-07T13:31:00Z"] * 2)
    second = raw_frame(["2025-08-07T13:31:00Z"] * 2 + ["2025-08-07T13:32:00Z"])
    stats.update(ingest.normalize_quotes(first, "2025-08-07"))
    stats.update(ingest.normalize_quotes(second, "2025-08-07"))
    report = stats.report()
    assert report["raw_rows"] == 5
    assert report["duplicate_keys"] == 3
    assert report["contract_count"] == 1
    assert report["expiry_count"] == 1
    assert report["expected_regular_session_minutes"] == 390
    assert report["regular_session_minute_count"] == 2
    assert report["valid_regular_session_minute_count"] == 2


def test_unsorted_stream_does_not_silently_miscount_duplicates():
    stats = ingest._DayStats("2025-08-07")
    frame = raw_frame(["2025-08-07T13:32:00Z", "2025-08-07T13:31:00Z"])
    with pytest.raises(ValueError, match="nondecreasing"):
        stats.update(ingest.normalize_quotes(frame, "2025-08-07"))


def test_prepare_all_days_resume_and_revalidate_source(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "source.zip"
    make_archive(path)
    calls = []
    def decode(raw, chunk_size):
        calls.append((raw, chunk_size))
        day = pd.Timestamp(raw.decode()).strftime("%Y-%m-%d")
        yield raw_frame([f"{day}T13:31:00Z"], symbol=["SPY250919C00640000"])
        yield raw_frame([f"{day}T13:31:00Z"], symbol=["SPY250919C00640000"])
    monkeypatch.setattr(ingest, "_iter_dbn_frames", decode)
    events = []
    output = tmp_path / "output"
    result = ingest.prepare_archive(path, output, chunk_size=1, on_progress=events.append)
    assert result["is_full_archive_complete"]
    assert result["completed_day_count"] == result["processed_day_count"] == 2
    assert result["raw_rows"] == 4
    assert result["duplicate_keys"] == 2
    assert len(calls) == 2
    assert all(event["status"] == "prepared" for event in events)
    first_file = output / "options/session_date=2025-08-07/quotes.parquet"
    table = pq.ParquetFile(first_file).read()
    assert table.num_rows == 2
    assert table.schema.equals(ingest.option_schema(), check_metadata=True)
    resumed = ingest.prepare_archive(path, output, chunk_size=1, on_progress=events.append)
    assert resumed["resumed_day_count"] == 2
    assert resumed["processed_day_count"] == 0
    assert len(calls) == 2
    assert events[-1]["status"] == "resumed"
    report = json.loads((output / "manifest.json").read_text())
    assert report["days"][0]["source"]["sha256"]
    assert report["days"][0]["source"]["crc32"]
    first_file.unlink()
    missing = ingest.prepare_archive(path, output, chunk_size=1)
    assert missing["processed_day_count"] == 1
    assert missing["resumed_day_count"] == 1
    # A changed configuration cannot reuse an otherwise identical partition.
    changed = ingest.prepare_archive(path, output, chunk_size=2)
    assert changed["processed_day_count"] == 2
    # A replacement member with a different exact source cannot resume.
    with zipfile.ZipFile(path, "w") as source:
        source.writestr("opra-pillar-20250807.cbbo-1m.dbn.zst", b"20250808")
    changed_source = ingest.prepare_archive(path, output, chunk_size=2)
    assert changed_source["processed_day_count"] == 1


def test_pilot_does_not_claim_full_archive_completion(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    path = tmp_path / "source.zip"
    make_archive(path)
    monkeypatch.setattr(ingest, "_iter_dbn_frames", lambda raw, chunk_size: iter([raw_frame()]))
    result = ingest.prepare_archive(path, tmp_path / "output", max_days=1)
    assert result["requested_day_count"] == result["completed_day_count"] == 1
    assert result["archive"]["day_count"] == 2
    assert not result["is_full_archive_complete"]


def test_failure_leaves_no_partial_partition_or_manifest(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    path = tmp_path / "source.zip"
    make_archive(path, dates=("20250807",))
    def decode(raw, chunk_size):
        yield raw_frame()
        raise ValueError("broken source chunk")
    monkeypatch.setattr(ingest, "_iter_dbn_frames", decode)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="broken source"):
        ingest.prepare_archive(path, output)
    assert not list(output.rglob("*.tmp"))
    assert not list(output.rglob("quotes.parquet"))
    assert not list(output.rglob("manifest.json"))


def test_parallel_workers_keep_archive_and_callback_order(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    path = tmp_path / "source.zip"
    make_archive(path)
    monkeypatch.setattr(ingest, "_iter_dbn_frames", lambda raw, chunk_size: iter([raw_frame()]))
    executions = []
    class OutOfOrderExecutor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 2
            assert mp_context.get_start_method() == "spawn"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def map(self, function, *iterables):
            tasks = list(zip(*iterables))
            results = {}
            for index in reversed(range(len(tasks))):
                executions.append(tasks[index][1]["session_date"])
                results[index] = function(*tasks[index])
            return iter(results[index] for index in range(len(tasks)))
    monkeypatch.setattr(ingest, "ProcessPoolExecutor", OutOfOrderExecutor)
    events = []
    result = ingest.prepare_archive(path, tmp_path / "output", workers=2, on_progress=events.append)
    assert executions == ["2025-08-08", "2025-08-07"]
    assert [day["session_date"] for day in result["days"]] == ["2025-08-07", "2025-08-08"]
    assert [event["session_date"] for event in events] == ["2025-08-07", "2025-08-08"]
    assert result["workers"] == result["processed_day_count"] == 2


def test_spawned_workers_resume_existing_partitions(tmp_path, monkeypatch):
    """Run actual spawned processes; they must resume without the parent mock."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "source.zip"
    make_archive(path)
    monkeypatch.setattr(ingest, "_iter_dbn_frames", lambda raw, chunk_size: iter([raw_frame()]))
    output = tmp_path / "output"
    ingest.prepare_archive(path, output)
    # The ZIP bytes are intentionally not DBN.  Child processes import an
    # unpatched module; reaching its decoder would therefore fail this test.
    result = ingest.prepare_archive(path, output, workers=2)
    assert result["resumed_day_count"] == 2
    assert result["processed_day_count"] == 0
    assert result["is_full_archive_complete"]


@pytest.mark.parametrize("kwargs", [
    {"chunk_size": 0}, {"chunk_size": True}, {"max_days": 0}, {"max_days": 1.5},
    {"workers": 0}, {"workers": 5}, {"workers": True}, {"workers": 1.5},
])
def test_invalid_preparation_limits_rejected_before_io(tmp_path, kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        ingest.prepare_archive(tmp_path / "missing.zip", tmp_path / "out", **kwargs)
