"""Reference conventions matter more than provider-shaped happy paths."""

import io
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvlab.market_data.calendar import expiry_timestamp, session_bounds, sessions
from rvlab.market_data.references import (
    FRED_SERIES,
    ReferenceValidationError,
    acquire_alpaca_stock,
    align_stock_prices,
    load_dividends,
    load_fred_rates,
    load_rates,
    load_stock_prices,
    parse_ssga_dividends,
    reference_coverage,
)


def stock_frame(**updates):
    frame = pd.DataFrame({
        "timestamp": ["2025-08-07T13:30:00Z", "2025-08-07T13:31:00Z"],
        "spot": [630., 631.], "price_basis": ["raw", "raw"],
        "source": ["independent test feed"] * 2,
    })
    for key, value in updates.items():
        frame[key] = value
    return frame


def rate_frame(**updates):
    frame = pd.DataFrame({
        "observation_date": ["2025-08-07"],
        "available_at": ["2025-08-08T21:00:00Z"],
        "rate": [.04], "tenor_days": [30], "series": ["DGS1MO"], "source": ["FRED"],
    })
    for key, value in updates.items():
        frame[key] = value
    return frame


def dividend_frame(**updates):
    frame = pd.DataFrame({
        "ex_date": ["2025-09-19"], "pay_date": ["2025-10-31"],
        "amount": [1.831], "source": ["official distribution history"],
    })
    for key, value in updates.items():
        frame[key] = value
    return frame


def test_exchange_calendar_dst_holidays_and_short_sessions():
    assert session_bounds("2025-08-07") == (pd.Timestamp("2025-08-07T13:30Z"), pd.Timestamp("2025-08-07T20:00Z"))
    assert session_bounds("2026-02-06") == (pd.Timestamp("2026-02-06T14:30Z"), pd.Timestamp("2026-02-06T21:00Z"))
    assert session_bounds("2025-11-27") is None
    assert session_bounds("2025-08-09") is None
    assert session_bounds("2025-11-28")[1] == pd.Timestamp("2025-11-28T18:00Z")
    assert session_bounds("2025-12-24")[1] == pd.Timestamp("2025-12-24T18:00Z")


def test_full_archive_calendar_and_expiry():
    frame = sessions("2025-08-07", "2026-02-06")
    assert len(frame) == 127
    assert frame.session_date.iloc[0] == "2025-08-07"
    assert frame.session_date.iloc[-1] == "2026-02-06"
    assert str(frame.open.dtype).endswith(", UTC]")
    assert expiry_timestamp("2025-11-29") == pd.Timestamp("2025-11-28T18:00Z")
    assert expiry_timestamp("2028-01-21") == pd.Timestamp("2028-01-21T21:00Z")
    with pytest.raises(ValueError, match="start"):
        sessions("2025-08-08", "2025-08-07")


def test_minute_bar_close_is_only_available_at_minute_end():
    out = load_stock_prices(stock_frame(), timestamp_convention="minute_bar_start")
    assert out.timestamp.iloc[0] == pd.Timestamp("2025-08-07T13:30Z")
    assert out.price_time.iloc[0] == pd.Timestamp("2025-08-07T13:31Z")
    assert out.available_at.iloc[0] == out.price_time.iloc[0]
    aligned = align_stock_prices(pd.to_datetime(["2025-08-07T13:30:59Z", "2025-08-07T13:31:00Z", "2025-08-07T13:33:01Z"]), out)
    assert aligned.stock_valid.tolist() == [False, True, False]
    assert np.isnan(aligned.spot.iloc[0])
    assert aligned.spot.iloc[1] == 630


def test_stock_alignment_preserves_order_and_respects_publication_latency():
    frame = stock_frame(available_at=["2025-08-07T13:30:05Z", "2025-08-07T13:31:05Z"])
    out = load_stock_prices(frame)
    stamps = pd.to_datetime(["2025-08-07T13:31:05Z", "2025-08-07T13:30:04Z", "2025-08-07T13:31:05Z"])
    aligned = align_stock_prices(stamps, out)
    assert aligned.as_of.tolist() == stamps.tolist()
    assert aligned.stock_valid.tolist() == [True, False, True]
    assert aligned.spot.iloc[0] == 631.


def test_stock_alignment_normalizes_nanosecond_and_microsecond_resolutions():
    frame = load_stock_prices(stock_frame(), timestamp_convention="minute_bar_start")
    for column in ["timestamp", "available_at", "price_time"]:
        frame[column] = frame[column].astype("datetime64[us, UTC]")
    expected = pd.date_range("2025-08-07T13:31Z", periods=2, freq="1min").astype("datetime64[ns, UTC]")
    aligned = align_stock_prices(expected, frame)
    assert aligned.stock_valid.tolist() == [True, True]
    assert aligned.spot.tolist() == [630., 631.]
    assert str(aligned.available_at.dtype) == "datetime64[ns, UTC]"
    assert str(load_stock_prices(frame).available_at.dtype) == "datetime64[ns, UTC]"


def test_stock_does_not_carry_prior_or_extended_session_prices():
    frame = stock_frame(timestamp=["2025-08-07T13:29:59Z", "2025-08-07T20:00:00Z"])
    out = load_stock_prices(frame, timestamp_convention="observation")
    aligned = align_stock_prices(pd.to_datetime(["2025-08-07T13:30:00Z", "2025-08-08T13:30:00Z", "2025-08-07T20:00:01Z"]), out)
    assert not aligned.stock_valid.any()
    with pytest.raises(ValueError, match="one minute"):
        align_stock_prices([], out, max_age="2min")


@pytest.mark.parametrize("updates,kwargs,error", [
    ({}, {}, "timestamp_convention"),
    ({"price_basis": ["adjusted", "adjusted"]}, {"timestamp_convention": "observation"}, "raw/unadjusted"),
    ({"spot": [0., 1.]}, {"timestamp_convention": "observation"}, "row positions \\[0\\]"),
    ({"symbol": ["QQQ", "SPY"]}, {"timestamp_convention": "observation"}, "symbol"),
    ({"timestamp": ["2025-08-07 09:30", "2025-08-07 09:31"]}, {"timestamp_convention": "observation"}, "timezone-naive"),
    ({"available_at": ["2025-08-07T13:30Z", "2025-08-07T13:31Z"]}, {"timestamp_convention": "minute_bar_start"}, "bar end"),
    ({"timestamp": ["2025-08-07T13:30Z"] * 2}, {"timestamp_convention": "observation"}, "Duplicate"),
    ({"bid": [631, 631], "ask": [630, 632]}, {"timestamp_convention": "observation"}, "Crossed"),
])
def test_invalid_stock_rows_are_explicit(updates, kwargs, error):
    with pytest.raises(ReferenceValidationError, match=error):
        load_stock_prices(stock_frame(**updates), **kwargs)


def test_stock_import_requires_raw_basis_and_handles_explicit_timezone():
    frame = stock_frame(timestamp=["2025-08-07 09:30", "2025-08-07 09:31"]).drop(columns="price_basis")
    with pytest.raises(ReferenceValidationError, match="price_basis"):
        load_stock_prices(frame, timezone="America/New_York", timestamp_convention="observation")
    out = load_stock_prices(frame, timezone="America/New_York", timestamp_convention="observation", price_basis="raw")
    assert out.timestamp.iloc[0] == pd.Timestamp("2025-08-07T13:30Z")


@pytest.mark.parametrize("updates,error", [
    ({"rate": [4.]}, "decimal units"),
    ({"rate": [float("nan")]}, "Invalid rate"),
    ({"tenor_days": [30.5]}, "integer"),
    ({"available_at": ["2025-08-06T21:00Z"]}, "precedes"),
    ({"available_at": ["2025-08-08 17:00"]}, "timezone-naive"),
])
def test_invalid_rates(updates, error):
    with pytest.raises(ReferenceValidationError, match=error):
        load_rates(rate_frame(**updates))


def test_rates_allow_negative_decimal_rates_without_implicit_vintage():
    out = load_rates(rate_frame(rate=[-.005]))
    assert out.rate.iloc[0] == -.005
    assert out.vintage.iloc[0] == "unspecified"


def test_fred_local_normalization_skips_missing_and_lags_federal_holiday(tmp_path):
    for series in FRED_SERIES:
        (tmp_path / f"{series}.csv").write_text(f"observation_date,{series}\n2025-08-29,4.5\n2025-09-01,.\n2025-09-02,4.4\n")
    out = load_fred_rates(tmp_path)
    friday = out[out.observation_date.eq(date(2025, 8, 29))]
    assert len(out) == 8
    assert len(friday) == 4
    # Labor Day is Monday; conservative availability is Tuesday at 17:00 ET.
    assert friday.available_at.eq(pd.Timestamp("2025-09-02T21:00Z")).all()
    assert friday.rate.eq(.045).all()
    assert out.vintage.eq("latest_not_point_in_time").all()
    assert load_fred_rates(tmp_path, start="2025-09-02").shape[0] == 4
    (tmp_path / "SOFR.csv").unlink()
    with pytest.raises(FileNotFoundError, match="SOFR"):
        load_fred_rates(tmp_path)


def test_dividends_do_not_invent_announcement_dates():
    unknown = load_dividends(dividend_frame())
    assert pd.isna(unknown.available_at.iloc[0])
    assert unknown.availability_basis.iloc[0] == "unknown"
    conservative = load_dividends(dividend_frame(), payment_date_availability=True)
    assert conservative.available_at.iloc[0] == pd.Timestamp("2025-10-31T20:00Z")
    assert conservative.availability_basis.iloc[0] == "payment_date_conservative"
    assert conservative.ex_date.iloc[0] == date(2025, 9, 19)
    explicit = load_dividends(dividend_frame(available_at=["2025-09-18T20:15Z"]), payment_date_availability=True)
    assert explicit.available_at.iloc[0] == pd.Timestamp("2025-09-18T20:15Z")
    assert explicit.availability_basis.iloc[0] == "declared"


@pytest.mark.parametrize("updates,error", [
    ({"amount": [-1.]}, "Invalid amount"),
    ({"pay_date": ["2025-09-18"]}, "precedes"),
    ({"ex_date": [None]}, "Invalid ex_date"),
    ({"is_forecast": [True]}, "forecasts require"),
    ({"is_forecast": ["maybe"]}, "boolean"),
])
def test_invalid_dividends(updates, error):
    with pytest.raises(ReferenceValidationError, match=error):
        load_dividends(dividend_frame(**updates))


def test_official_workbook_parser_filters_spy_and_preserves_cash_flow():
    pytest.importorskip("openpyxl")
    columns = ["FUND NAME", "TICKER", "CUSIP", "EX-DATE", "RECORD DATE", "PAYABLE DATE", "DIVIDEND ($)", "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)", "FREQUENCY"]
    frame = pd.DataFrame([
        ["SPDR SPY", "SPY", "78462F103", "09/19/2025", "09/19/2025", "10/31/2025", 1.831, None, .02, "Quarterly"],
        ["Other fund", "BIL", "other", "09/19/2025", "09/19/2025", "10/31/2025", 4., None, None, "Monthly"],
    ], columns=columns)
    workbook = io.BytesIO()
    frame.to_excel(workbook, index=False, startrow=2)
    out = parse_ssga_dividends(workbook.getvalue())
    assert len(out) == 1
    assert out.symbol.iloc[0] == "SPY"
    assert out.amount.iloc[0] == pytest.approx(1.851)
    assert out.availability_basis.iloc[0] == "payment_date_conservative"


def test_reference_roundtrip_and_coverage(tmp_path):
    frame = load_stock_prices(stock_frame(), timestamp_convention="minute_bar_start")
    path = tmp_path / "stock.parquet"
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(load_stock_prices(path), frame)
    report = reference_coverage(stock=frame, dividends=load_dividends(dividend_frame()))
    assert report["stock"]["present"]
    assert not report["rates"]["present"]
    assert report["dividends"]["unknown_availability"] == 1
    assert not report["dividends"]["historical_expectations_available"]


def test_alpaca_missing_credentials_are_actionable(monkeypatch, tmp_path):
    for name in ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ReferenceValidationError, match="credentials are missing"):
        acquire_alpaca_stock("2025-08-07", "2026-02-06", tmp_path / "stock.parquet")
