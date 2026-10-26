"""Exchange sessions used by SPY research (all instants are UTC).

SPY's regular underlying session and the standard SPY option expirations use
the US equity trading calendar.  Holidays and early closes must come from the
exchange calendar, rather than a fixed UTC offset or a weekday approximation.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=16)
def _calendar(start_year: int, end_year: int):
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise ImportError(
            "SPY sessions require exchange_calendars; install rvlab[market-data]."
        ) from exc
    return xcals.get_calendar(
        "XNYS", start=f"{start_year}-01-01", end=f"{end_year}-12-31"
    )


def _date(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("Session date cannot be missing")
    if value.tzinfo is not None:
        value = value.tz_convert("America/New_York").tz_localize(None)
    return value.normalize()


def sessions(start, end) -> pd.DataFrame:
    """Return inclusive XNYS sessions with ``session_date``, ``open``, ``close``.

    Session labels are ISO date strings. Open and close columns are timezone
    aware UTC timestamps, including daylight-saving and shortened sessions.
    """
    first, last = _date(start), _date(end)
    if first > last:
        raise ValueError("start must be on or before end")
    cal = _calendar(first.year, last.year)
    labels = cal.sessions_in_range(first, last)
    return pd.DataFrame(
        {
            "session_date": [str(label.date()) for label in labels],
            "open": pd.to_datetime([cal.session_open(label) for label in labels], utc=True),
            "close": pd.to_datetime([cal.session_close(label) for label in labels], utc=True),
        }
    )


def session_bounds(date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the regular session bounds, or ``None`` on a non-session date."""
    day = _date(date)
    cal = _calendar(day.year, day.year)
    if not cal.is_session(day):
        return None
    return cal.session_open(day), cal.session_close(day)


def expiry_timestamp(expirydate) -> pd.Timestamp:
    """Standard PM SPY expiry close; a non-session date uses its prior session.

    This is the pricing convention for standard, unadjusted SPY options. It is
    not an OCC exercise deadline or an AM-settled index option convention.
    """
    day = _date(expirydate)
    cal = _calendar(day.year - 1, day.year)
    label = cal.date_to_session(day, direction="previous")
    return cal.session_close(label)
