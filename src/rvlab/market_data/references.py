"""Validated, independently sourced underlying, rate and dividend references.

No loader supplies synthetic observations or silently adjusted equity prices.
Latest FRED downloads are useful rate proxies, not historical data vintages.
Historical distributions are realised cash flows, not dividend expectations.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .calendar import session_bounds


SSGA_DISTRIBUTIONS_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "spdr-etf-historical-distributions.xlsx"
)
FRED_SERIES = {"SOFR": 1, "DGS1MO": 30, "DGS3MO": 91, "DGS6MO": 182}
NY = "America/New_York"


class ReferenceValidationError(ValueError):
    """A reference file cannot be used without correcting its observations."""


def _read(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Required market reference does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ReferenceValidationError(f"Expected CSV or Parquet reference, got {path.suffix}")


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReferenceValidationError(f"Missing reference columns: {', '.join(missing)}")


def _fail(frame: pd.DataFrame, bad, reason: str) -> None:
    mask = np.asarray(bad, dtype=bool)
    if mask.any():
        positions = np.flatnonzero(mask)[:10].tolist()
        raise ReferenceValidationError(f"{reason}; row positions {positions}")


def _utc(frame, column: str, *, timezone=None, nullable=False) -> pd.Series:
    # Pandas 3 may infer microsecond timestamps while calendars retain
    # nanoseconds. merge_asof requires exactly matching datetime resolutions.
    if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
        if not nullable:
            _fail(frame, frame[column].isna(), f"Missing {column}")
        return frame[column].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    values = []
    for position, value in enumerate(frame[column]):
        if pd.isna(value) or value == "":
            if nullable:
                values.append(pd.NaT)
                continue
            raise ReferenceValidationError(f"Missing {column}; row position {position}")
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                if timezone is None:
                    raise ValueError("timezone is missing")
                stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
            values.append(stamp.tz_convert("UTC"))
        except (TypeError, ValueError) as exc:
            raise ReferenceValidationError(
                f"Invalid or timezone-naive {column}; row position {position}: {value!r}"
            ) from exc
    return pd.Series(pd.to_datetime(values, utc=True), index=frame.index).astype("datetime64[ns, UTC]")


def _dates(frame, column, *, nullable=False) -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="coerce")
    invalid = parsed.isna()
    if nullable:
        invalid &= ~frame[column].isna()
    _fail(frame, invalid, f"Invalid {column}")
    return parsed.dt.date


def _number(frame, column, *, lower=None, strict=False) -> pd.Series:
    result = pd.to_numeric(frame[column], errors="coerce")
    invalid = ~np.isfinite(result)
    if lower is not None:
        invalid |= result <= lower if strict else result < lower
    _fail(frame, invalid, f"Invalid {column}")
    return result


def _source(frame, source):
    if "source" not in frame:
        if not source:
            raise ReferenceValidationError("A source column or explicit source is required")
        frame["source"] = source
    _fail(frame, frame.source.isna() | frame.source.astype(str).str.strip().eq(""), "Missing source")


def load_stock_prices(
    path: str | Path | pd.DataFrame,
    *,
    timestamp_convention: Literal["observation", "minute_bar_start", "minute_bar_end"] | None = None,
    timezone: str | None = None,
    price_basis: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Validate raw SPY observations, retaining bar time and availability.

    A canonical file includes ``timestamp``, ``available_at``, ``spot``,
    ``price_basis='raw'`` and ``source``. A caller importing one-minute bars
    must declare ``timestamp_convention='minute_bar_start'``; the close then
    cannot be used before the next minute. ``price_time`` records that bar end.
    Naive timestamps require an explicit timezone. Daily bars are not accepted
    as substitutes for an intraday stock feed.
    """
    out = _read(path)
    _require(out, ["timestamp", "spot"])
    _source(out, source)
    out["timestamp"] = _utc(out, "timestamp", timezone=timezone)
    out["spot"] = _number(out, "spot", lower=0, strict=True)
    if "symbol" not in out:
        out["symbol"] = "SPY"
    _fail(out, out.symbol.ne("SPY"), "Underlying symbol must be SPY")
    if "price_basis" not in out:
        if price_basis is None:
            raise ReferenceValidationError("Declare price_basis='raw'; adjusted stock prices are not usable")
        out["price_basis"] = price_basis
    _fail(out, out.price_basis.ne("raw"), "Stock prices must be raw/unadjusted")
    if timestamp_convention is not None:
        if timestamp_convention not in {"observation", "minute_bar_start", "minute_bar_end"}:
            raise ReferenceValidationError(f"Unsupported timestamp_convention: {timestamp_convention}")
        if "timestamp_convention" in out:
            _fail(out, out.timestamp_convention.ne(timestamp_convention), "Conflicting timestamp convention")
        out["timestamp_convention"] = timestamp_convention
    elif "timestamp_convention" not in out:
        if "available_at" not in out:
            raise ReferenceValidationError("Declare timestamp_convention or supply explicit available_at")
        out["timestamp_convention"] = "observation"
    _fail(
        out,
        ~out.timestamp_convention.isin(["observation", "minute_bar_start", "minute_bar_end"]),
        "Unsupported timestamp_convention",
    )
    end = out.timestamp + pd.to_timedelta(out.timestamp_convention.eq("minute_bar_start").astype(int), unit="min")
    if "price_time" in out:
        out["price_time"] = _utc(out, "price_time", timezone=timezone)
        _fail(out, out.price_time.ne(end), "price_time conflicts with bar timestamp convention")
    else:
        out["price_time"] = end
    if "available_at" in out:
        out["available_at"] = _utc(out, "available_at", timezone=timezone)
    else:
        out["available_at"] = out.price_time
    _fail(out, out.available_at < out.price_time, "Stock available_at precedes observation/bar end")
    out["session_date"] = out.price_time.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    bidask = [name in out for name in ("bid", "ask")]
    if any(bidask) and not all(bidask):
        raise ReferenceValidationError("Stock bid and ask must both be present")
    if all(bidask):
        out["bid"] = _number(out, "bid", lower=0, strict=True)
        out["ask"] = _number(out, "ask", lower=0, strict=True)
        _fail(out, out.bid > out.ask, "Crossed stock quote")
    _fail(out, out.duplicated(["symbol", "price_time"]), "Duplicate stock observation")
    return out.sort_values(["available_at", "price_time"], kind="stable").reset_index(drop=True)


def align_stock_prices(
    timestamps, stock: pd.DataFrame, *, max_age: str | pd.Timedelta = "1min"
) -> pd.DataFrame:
    """As-of align stock, disallowing future information and overnight carries.

    Returns the requested ``as_of`` plus the matched canonical columns and a
    ``stock_valid`` flag. Missing, stale or cross-session observations have
    missing ``spot``. The input order and duplicate requests are preserved.
    """
    age_limit = pd.Timedelta(max_age)
    if age_limit <= pd.Timedelta(0) or age_limit > pd.Timedelta("1min"):
        raise ValueError("max_age must be positive and no greater than one minute")
    left = pd.DataFrame({"as_of": list(timestamps)})
    left["as_of"] = _utc(left, "as_of")
    left["_order"] = np.arange(len(left))
    if stock.empty:
        left["spot"] = np.nan
        left["stock_valid"] = False
        return left.drop(columns="_order")
    _require(stock, ["available_at", "price_time", "spot", "session_date"])
    right = stock.copy()
    for column in ("available_at", "price_time"):
        right[column] = _utc(right, column)
    right = right.sort_values(["available_at", "price_time"], kind="stable")
    out = pd.merge_asof(left.sort_values("as_of"), right, left_on="as_of", right_on="available_at", direction="backward")
    age = out.as_of - out.price_time
    same_session = out.as_of.dt.tz_convert(NY).dt.strftime("%Y-%m-%d").eq(out.session_date)
    valid = out.spot.notna() & (age >= pd.Timedelta(0)) & (age <= age_limit) & same_session
    # Extended-hours quotes cannot be carried into regular-session snapshots.
    for day, group in out.groupby(out.as_of.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")):
        bounds = session_bounds(day)
        if bounds is None:
            valid.loc[group.index] = False
        else:
            opening, closing = bounds
            valid.loc[group.index] &= group.as_of.between(opening, closing) & group.price_time.between(opening, closing)
    out["stock_age_seconds"] = age.dt.total_seconds()
    out["stock_valid"] = valid
    out.loc[~valid, "spot"] = np.nan
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def load_rates(path: str | Path | pd.DataFrame, *, source: str | None = None) -> pd.DataFrame:
    """Validate annual decimal rates; negative rates remain allowed.

    ``tenor_days`` identifies a proxy tenor, not a bootstrapped zero curve.
    Availability must be supplied, and the observation date is never used as
    an implicit publication timestamp.
    """
    out = _read(path)
    _require(out, ["observation_date", "available_at", "tenor_days", "rate", "series"])
    _source(out, source)
    out["observation_date"] = _dates(out, "observation_date")
    out["available_at"] = _utc(out, "available_at")
    out["tenor_days"] = _number(out, "tenor_days", lower=0, strict=True)
    _fail(out, out.tenor_days.mod(1).ne(0), "tenor_days must be integer")
    out["tenor_days"] = out.tenor_days.astype(int)
    out["rate"] = _number(out, "rate")
    _fail(out, out.rate.abs() > 1, "Rate must be annual decimal units, not percentages")
    local_date = out.available_at.dt.tz_convert(NY).dt.date
    _fail(out, local_date < out.observation_date, "Rate available_at precedes observation date")
    _fail(out, out.series.isna() | out.series.astype(str).str.strip().eq(""), "Missing rate series")
    _fail(out, out.duplicated(["observation_date", "series", "available_at"]), "Duplicate rate observation")
    if "rate_convention" not in out:
        out["rate_convention"] = "unspecified"
    if "vintage" not in out:
        out["vintage"] = "unspecified"
    return out.sort_values(["available_at", "tenor_days"], kind="stable").reset_index(drop=True)


def load_dividends(
    path: str | Path | pd.DataFrame,
    *,
    source: str | None = None,
    payment_date_availability: bool = False,
) -> pd.DataFrame:
    """Validate realised distributions separately from historical expectations.

    Unknown declaration times stay ``NaT``. With explicit
    ``payment_date_availability=True``, realised amounts become available only
    at 16:00 New York on their payment date. That conservative proxy is labelled
    and is never an invented declaration timestamp. It enables a lagged
    trailing-distribution estimate, not foresight of a future distribution.
    """
    out = _read(path)
    _require(out, ["ex_date", "pay_date", "amount"])
    _source(out, source)
    out["ex_date"] = _dates(out, "ex_date")
    out["pay_date"] = _dates(out, "pay_date")
    out["amount"] = _number(out, "amount", lower=0)
    _fail(out, out.pay_date < out.ex_date, "Dividend pay_date precedes ex_date")
    if "symbol" not in out:
        out["symbol"] = "SPY"
    _fail(out, out.symbol.ne("SPY"), "Dividend symbol must be SPY")
    if "is_forecast" not in out:
        out["is_forecast"] = False
    mapped = out.is_forecast.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False, 1: True, 0: False})
    _fail(out, mapped.isna(), "is_forecast must be boolean")
    out["is_forecast"] = mapped.astype(bool)
    if "available_at" not in out:
        out["available_at"] = pd.NaT
    out["available_at"] = _utc(out, "available_at", nullable=True)
    if "availability_basis" not in out:
        out["availability_basis"] = np.where(out.available_at.isna(), "unknown", "declared")
    if payment_date_availability:
        mask = out.available_at.isna() & ~out.is_forecast
        conservative = pd.to_datetime(out.loc[mask, "pay_date"].astype(str) + " 16:00").dt.tz_localize(NY).dt.tz_convert("UTC")
        out.loc[mask, "available_at"] = conservative
        out.loc[mask, "availability_basis"] = "payment_date_conservative"
    _fail(out, out.is_forecast & out.available_at.isna(), "Dividend forecasts require their historical available_at")
    _fail(out, out.duplicated(["symbol", "ex_date", "is_forecast", "available_at"]), "Duplicate dividend observation")
    return out.sort_values(["ex_date", "available_at"], kind="stable").reset_index(drop=True)


def _write_reference(frame: pd.DataFrame, destination: str | Path, metadata: dict) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".csv":
        frame.to_csv(dest, index=False)
    elif dest.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(dest, index=False)
    else:
        raise ValueError("Reference destination must be .csv or .parquet")
    metadata = {**metadata, "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": len(frame)}
    dest.with_suffix(dest.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def _parse_fred_series(raw: pd.DataFrame, series: str, start=None, end=None) -> pd.DataFrame:
    if series not in raw or len(raw.columns) < 2:
        raise ReferenceValidationError(f"Unexpected FRED response for {series}")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(raw[series], errors="coerce")
    mask = dates.notna() & values.notna()
    if start is not None:
        mask &= dates.dt.date >= pd.Timestamp(start).date()
    if end is not None:
        mask &= dates.dt.date <= pd.Timestamp(end).date()
    dates, values = dates[mask], values[mask]
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    available = pd.Series(
        [(stamp + business_day).normalize() + pd.Timedelta(hours=17) for stamp in dates],
        index=dates.index, dtype="datetime64[ns]",
    ).dt.tz_localize(NY).dt.tz_convert("UTC")
    return pd.DataFrame({
        "observation_date": dates.dt.date,
        "available_at": available,
        "tenor_days": FRED_SERIES[series],
        "rate": values / 100.0,
        "series": series,
        "source": f"FRED/{series}",
        "source_url": f"https://fred.stlouisfed.org/series/{series}",
        "rate_convention": "simple_act360" if series == "SOFR" else "annualized_yield_proxy",
        "availability_basis": "next_federal_business_day_17_et_conservative",
        "vintage": "latest_not_point_in_time",
    })


def load_fred_rates(raw_dir: str | Path, *, start=None, end=None) -> pd.DataFrame:
    """Normalize previously downloaded ``{SERIES}.csv`` files without network access.

    All four required series must be present. Missing daily holiday values
    (FRED's ``.`` marker) are retained as gaps rather than fabricated rates.
    """
    root = Path(raw_dir)
    frames = []
    for series in FRED_SERIES:
        path = root / f"{series}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required FRED source is missing: {path}")
        frames.append(_parse_fred_series(pd.read_csv(path, na_values=["."]), series, start, end))
    out = load_rates(pd.concat(frames, ignore_index=True))
    if out.empty:
        raise ReferenceValidationError("Downloaded FRED files contain no usable rates")
    return out


def fetch_public_rates(start, end, destination: str | Path) -> pd.DataFrame:
    """Download SOFR and short Treasury yields from public FRED CSV endpoints.

    All four use a deliberately lagged availability proxy: 17:00 New York on
    the following US federal business day. SOFR is normally published the next
    business morning and can be revised that afternoon. Treasury yields are
    not zero rates. Latest downloaded values are not guaranteed point-in-time
    vintages; metadata and per-row ``vintage`` retain that limitation.
    """
    import requests

    start, end = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if start > end:
        raise ValueError("start must be on or before end")
    frames, urls = [], []
    for series in FRED_SERIES:
        # fredgraph is a public CSV endpoint and does not require an API key.
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
        response = requests.get(csv_url, timeout=60)
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text), na_values=["."])
        frames.append(_parse_fred_series(raw, series, start, end))
        urls.append(csv_url)
    out = load_rates(pd.concat(frames, ignore_index=True))
    if out.empty:
        raise ReferenceValidationError("Public FRED responses contain no usable rates")
    _write_reference(out, destination, {
        "provider": "FRED", "urls": urls, "start": str(start), "end": str(end),
        "availability_policy": "next US federal business day at 17:00 America/New_York (proxy)",
        "limitations": ["Latest observations, not archived first-release vintages", "Treasury par yields are a discount-curve proxy, not bootstrapped zero rates"],
    })
    return out


def parse_ssga_dividends(workbook: bytes | str | Path) -> pd.DataFrame:
    """Parse the official historical workbook and select unadjusted SPY cash flows."""
    source = io.BytesIO(workbook) if isinstance(workbook, bytes) else workbook
    sheets = pd.read_excel(source, sheet_name=None, header=None)
    matches = []
    for frame in sheets.values():
        for row in range(min(30, len(frame))):
            labels = frame.iloc[row].astype(str).str.strip().str.upper()
            if labels.eq("TICKER").any() and labels.str.contains("EX.DATE", regex=True).any():
                data = frame.iloc[row + 1:].copy()
                data.columns = labels
                data = data.loc[data["TICKER"].astype(str).str.strip().eq("SPY")]
                if data.empty:
                    break
                def column(*terms):
                    found = [name for name in data.columns if all(term in name for term in terms)]
                    if len(found) != 1:
                        raise ReferenceValidationError(f"Unexpected State Street workbook columns: {list(data.columns)}")
                    return data[found[0]]
                amount_names = [name for name in data.columns if name in {"DISTRIBUTION ($)", "DIVIDEND ($)", "DISTRIBUTION", "DIVIDEND"}]
                if len(amount_names) != 1:
                    raise ReferenceValidationError(f"Cannot identify distribution amount in workbook: {list(data.columns)}")
                amount = pd.to_numeric(data[amount_names[0]], errors="coerce")
                # Capital gains distributions are cash flows too if present.
                for name in data.columns:
                    if "CAPITAL GAIN" in name:
                        amount = amount + pd.to_numeric(data[name], errors="coerce").fillna(0)
                matches.append(pd.DataFrame({
                    "ex_date": column("EX", "DATE"),
                    "pay_date": column("PAY", "DATE"),
                    "amount": amount,
                    "symbol": "SPY",
                    "source": "State Street SPDR historical distributions",
                    "source_url": SSGA_DISTRIBUTIONS_URL,
                    "is_forecast": False,
                }))
                break
    if not matches:
        raise ReferenceValidationError("Official distribution workbook contains no identifiable SPY distribution table")
    return load_dividends(pd.concat(matches, ignore_index=True), payment_date_availability=True)


def fetch_public_dividends(start, end, destination: str | Path) -> pd.DataFrame:
    """Fetch realised SPY distributions; historical declaration times stay unknown.

    ``start``/``end`` filter ex-dates; callers needing a trailing estimate should
    request at least the previous year. Per-row availability is conservatively
    delayed until payment, not used as an announcement-date substitute.
    """
    import requests

    first, last = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if first > last:
        raise ValueError("start must be on or before end")
    response = requests.get(SSGA_DISTRIBUTIONS_URL, timeout=90)
    response.raise_for_status()
    out = parse_ssga_dividends(response.content)
    out = out.loc[out.ex_date.between(first, last)].reset_index(drop=True)
    if out.empty:
        raise ReferenceValidationError("Official workbook has no SPY distributions in the requested interval")
    _write_reference(out, destination, {
        "provider": "State Street", "url": SSGA_DISTRIBUTIONS_URL,
        "start": str(first), "end": str(last),
        "availability_policy": "16:00 New York on payment date (conservative proxy, not announcement)",
        "limitations": ["Realised distributions do not establish historical dividend expectations", "Workbook does not identify declaration timestamps"],
    })
    return out


def acquire_alpaca_stock(start, end, destination: str | Path) -> pd.DataFrame:
    """Download raw consolidated SIP one-minute SPY bars with existing API access.

    Uses credentials only from ``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY`` (or
    ``ALPACA_API_KEY``/``ALPACA_SECRET_KEY``). No account, subscription or purchase
    is created. Provider authorization failures are propagated explicitly.
    """
    import requests

    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise ReferenceValidationError("Alpaca credentials are missing; configure APCA_API_KEY_ID and APCA_API_SECRET_KEY for historical SIP access")
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    first = first.tz_localize(NY) if first.tzinfo is None else first
    last = last.tz_localize(NY) if last.tzinfo is None else last
    # Inclusive research date range, exclusive provider end timestamp.
    if last == last.normalize():
        last += pd.DateOffset(days=1)
    if first >= last:
        raise ValueError("start must precede end")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"timeframe": "1Min", "start": first.tz_convert("UTC").isoformat(), "end": last.tz_convert("UTC").isoformat(), "feed": "sip", "adjustment": "raw", "limit": 10000, "sort": "asc"}
    url = "https://data.alpaca.markets/v2/stocks/SPY/bars"
    rows, tokens = [], set()
    while True:
        response = requests.get(url, params=params, headers=headers, timeout=90)
        if response.status_code in {401, 403}:
            raise ReferenceValidationError(f"Alpaca historical SIP access denied ({response.status_code}); independent SPY prices remain missing")
        response.raise_for_status()
        payload = response.json()
        for bar in payload.get("bars", []) or []:
            rows.append({"timestamp": bar["t"], "spot": bar["c"], "volume": bar.get("v"), "source": "Alpaca SIP 1-minute raw bars", "symbol": "SPY"})
        token = payload.get("next_page_token")
        if not token:
            break
        if token in tokens:
            raise ReferenceValidationError("Alpaca returned a repeated page token")
        tokens.add(token)
        params["page_token"] = token
    if not rows:
        raise ReferenceValidationError("Alpaca returned no historical SPY bars")
    out = load_stock_prices(pd.DataFrame(rows), timestamp_convention="minute_bar_start", price_basis="raw")
    _write_reference(out, destination, {"provider": "Alpaca", "url": url, "feed": "sip", "adjustment": "raw", "timestamp_convention": "minute_bar_start; close becomes available at minute end", "start": str(start), "end": str(end), "limitations": ["Bar close is an observed underlying proxy, not an executable bid/ask quote"]})
    return out


def reference_coverage(stock=None, rates=None, dividends=None) -> dict:
    """Machine-readable presence and timing summary, without inferred coverage."""
    result = {}
    for name, frame, time in [("stock", stock, "price_time"), ("rates", rates, "available_at"), ("dividends", dividends, "ex_date")]:
        present = frame is not None and not frame.empty
        entry = {"present": present, "rows": len(frame) if frame is not None else 0}
        if present:
            entry.update(start=str(frame[time].min()), end=str(frame[time].max()))
            if name == "rates":
                entry["series"] = sorted(frame.series.unique().tolist())
                entry["point_in_time_vintages"] = bool(frame.vintage.eq("point_in_time").all())
            if name == "dividends":
                entry["unknown_availability"] = int(frame.available_at.isna().sum())
                entry["historical_expectations_available"] = bool(frame.is_forecast.any())
        result[name] = entry
    return result
