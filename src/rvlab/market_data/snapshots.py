"""Point-in-time market inputs for American SPY calibration.

This module never infers spot from option parity. Option CBBO intervals and
stock bars must have ended, references must have become available, and all
row arrays remain positionally aligned. Rates are Treasury yield *proxies*,
not a bootstrapped OIS curve. Observed American quotes are passed directly to
American pricers with independent spot and a dated cash-dividend schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .calendar import session_bounds
from rvlab.pricing.vanilla_types import (
    CashDividend, DiscountCurve, DividendSchedule, SpotMarket,
)


YEAR_SECONDS = 365.0 * 24 * 60 * 60
_RATE_SERIES = frozenset({"DGS1MO", "DGS3MO", "DGS6MO"})
SNAPSHOT_SCHEMA_VERSION = 2
CALIBRATION_BACKEND = "american_vanilla_fd"


class MarketDataNotReady(ValueError):
    """A snapshot lacks observable inputs or eligible contracts."""


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Detached quotes and aligned market arrays, with readiness diagnostics.

    ``ready`` means inputs passed the configured preparation screens. It does
    not certify the model, trading usefulness, or numerical convergence.
    A non-ready result may retain candidates with NaN market arrays for audit;
    use :meth:`calibration_inputs` to obtain safe calibration keyword inputs.
    """

    as_of: pd.Timestamp
    quotes: pd.DataFrame
    forward: np.ndarray
    discount: np.ndarray
    ready: bool
    missing_inputs: tuple[str, ...]
    exclusions: pd.DataFrame
    provenance: Mapping
    market: SpotMarket | None = None

    def require_ready(self) -> CalibrationSnapshot:
        if not self.ready:
            detail = "; ".join(self.missing_inputs) or "no eligible contracts"
            raise MarketDataNotReady(f"calibration snapshot is not ready: {detail}")
        return self

    def calibration_inputs(self) -> dict:
        """Return ``calibrate_vanilla_surface`` inputs for American contracts.

        ``forward`` and ``discount`` on this object are carry diagnostics only;
        they deliberately are not part of the calibration handoff.
        """
        self.require_ready()
        if self.market is None:
            raise MarketDataNotReady("American snapshot has no spot-market inputs")
        return {"quotes": self.quotes.copy(), "market": self.market,
                "exercise_style": "american"}


def _utc(value, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return result.tz_convert("UTC")


def _times(values: pd.Series, name: str, *, allow_missing=False) -> pd.Series:
    # Do not let pandas silently interpret naive input as UTC.
    out = []
    for pos, value in enumerate(values):
        if pd.isna(value) and allow_missing:
            out.append(pd.NaT)
        else:
            try:
                out.append(_utc(value, name))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} at row position {pos}: {exc}") from exc
    return pd.Series(pd.to_datetime(out, utc=True), index=values.index)


def _table(value, name: str, required: set[str]) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not value.columns.is_unique:
        raise ValueError(f"{name} must have unique columns")
    missing = required - set(value)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    return value.copy().reset_index(drop=True)


def _positive_duration(value, name: str) -> pd.Timedelta:
    result = pd.Timedelta(value)
    if pd.isna(result) or result <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive duration")
    return result


def _candidate_selection(options, as_of, min_dte, max_dte, max_quote_age):
    quotes = _table(options, "options", {"contract_id", "timestamp", "available_at",
        "expiry_at", "strike", "is_call", "bid", "ask", "quote_valid",
        "in_regular_session"})
    for column in ("timestamp", "available_at", "expiry_at"):
        quotes[column] = _times(quotes[column], f"options.{column}")
    if quotes["contract_id"].isna().any():
        raise ValueError("options.contract_id cannot be missing")
    for column in ("quote_valid", "in_regular_session", "is_call"):
        invalid = ~quotes[column].map(lambda item: isinstance(item, (bool, np.bool_)))
        if invalid.any():
            raise ValueError(f"options.{column} must contain booleans at row positions "
                             f"{np.flatnonzero(invalid).tolist()[:8]}")
    # Latest observable quote first, then quality: a bad latest quote must not
    # cause an older, apparently better quote to be resurrected.
    quotes = quotes.loc[(quotes["timestamp"] <= as_of) &
                        (quotes["available_at"] <= as_of)].copy()
    quotes = quotes.sort_values(["timestamp", "available_at"], kind="stable")
    quotes = quotes.drop_duplicates("contract_id", keep="last")
    quotes = quotes.sort_values("contract_id", kind="stable").reset_index(drop=True)
    quotes["maturity"] = (quotes["expiry_at"] - as_of).dt.total_seconds() / YEAR_SECONDS
    quotes["market_price"] = (pd.to_numeric(quotes["bid"], errors="coerce") +
                              pd.to_numeric(quotes["ask"], errors="coerce")) / 2
    quotes["exercise_style"] = "american"
    quotes["quote_age_seconds"] = (as_of - quotes["timestamp"]).dt.total_seconds()
    reasons = pd.Series("", index=quotes.index, dtype=object)
    checks = [
        (~quotes["quote_valid"], "invalid_quote"),
        (~quotes["in_regular_session"], "outside_regular_session"),
        (as_of - quotes["timestamp"] > max_quote_age, "stale_option_quote"),
        (quotes["maturity"] * 365 < min_dte, "below_min_dte"),
        (quotes["maturity"] * 365 > max_dte, "above_max_dte"),
        (quotes["maturity"] <= 0, "expired"),
    ]
    strike = pd.to_numeric(quotes["strike"], errors="coerce")
    bid = pd.to_numeric(quotes["bid"], errors="coerce")
    ask = pd.to_numeric(quotes["ask"], errors="coerce")
    checks.append((~np.isfinite(strike) | (strike <= 0) | ~np.isfinite(bid) |
                   ~np.isfinite(ask) | (bid <= 0) | (ask <= 0) | (ask < bid), "invalid_quote_values"))
    for mask, reason in checks:
        reasons.loc[mask] = reasons.loc[mask].map(lambda old: f"{old};{reason}".strip(";"))
    excluded = quotes.loc[reasons != ""].copy()
    excluded["snapshot_exclusion_reason"] = reasons.loc[reasons != ""]
    return quotes.loc[reasons == ""].reset_index(drop=True), excluded.reset_index(drop=True)


def candidate_snapshot(options: pd.DataFrame, as_of, *, min_dte=7.0,
                       max_dte=60.0, max_quote_age="1min") -> pd.DataFrame:
    """Select observable latest contracts without assuming missing references.

    Used for coverage audits. This is not a ready calibration surface. The
    ``attrs['exclusion_counts']`` summary counts all applicable screen reasons.
    """
    as_of = _utc(as_of, "as_of")
    if not (math.isfinite(min_dte) and math.isfinite(max_dte) and 0 <= min_dte <= max_dte):
        raise ValueError("require finite 0 <= min_dte <= max_dte")
    selected, excluded = _candidate_selection(options, as_of, min_dte, max_dte,
        _positive_duration(max_quote_age, "max_quote_age"))
    reasons = excluded["snapshot_exclusion_reason"].str.split(";").explode()
    selected.attrs["exclusion_counts"] = reasons.value_counts().to_dict()
    return selected


def _stock_observation(stock, as_of, max_age):
    if stock is None or len(stock) == 0:
        return None, "missing independent stock prices"
    rows = _table(stock, "stock", {"timestamp", "available_at", "spot"})
    for field in ("timestamp", "available_at"):
        rows[field] = _times(rows[field], f"stock.{field}")
    rows["price_time"] = (_times(rows["price_time"], "stock.price_time")
                          if "price_time" in rows else rows["timestamp"])
    if "symbol" in rows:
        rows = rows.loc[rows["symbol"] == "SPY"]
    rows = rows.loc[(rows["price_time"] <= as_of) & (rows["available_at"] <= as_of)]
    if rows.empty:
        return None, "no stock observation available at as_of"
    row = rows.sort_values(["price_time", "available_at"], kind="stable").iloc[-1]
    bounds = session_bounds(as_of)
    if bounds is None or not bounds[0] <= as_of <= bounds[1]:
        return None, "as_of is outside the XNYS regular stock session"
    session_open, session_close = bounds
    if not (session_open <= row["timestamp"] <= session_close and
            session_open <= row["price_time"] <= session_close):
        return None, "stock observation must belong to the same XNYS regular session as as_of"
    if as_of - row["price_time"] > max_age:
        return None, "independent stock observation is stale"
    if "price_basis" not in row or row["price_basis"] not in {"raw", "unadjusted"}:
        return None, "stock price must have an explicit raw/unadjusted price_basis"
    try:
        spot = float(row["spot"])
    except (TypeError, ValueError):
        return None, "stock price is not finite and positive"
    if not math.isfinite(spot) or spot <= 0:
        return None, "stock price is not finite and positive"
    return row, None


def _discount_curve(rates, as_of, maturities, max_rate_age):
    if rates is None or len(rates) == 0:
        return None, None, "missing dated Treasury rate proxies"
    rows = _table(rates, "rates", {"observation_date", "available_at", "tenor_days", "rate", "series"})
    rows["available_at"] = _times(rows["available_at"], "rates.available_at", allow_missing=True)
    rows = rows.loc[rows["series"].isin(_RATE_SERIES) & (rows["available_at"] <= as_of)]
    if rows.empty:
        return None, None, "no Treasury rate proxy available at as_of (SOFR is financing only)"
    rows["observation_date"] = pd.to_datetime(rows["observation_date"], errors="coerce").dt.date
    rows = rows.loc[rows["observation_date"].notna() &
                    (rows["observation_date"] <= as_of.tz_convert("America/New_York").date())]
    if rows.empty:
        return None, None, "no valid Treasury observation date available at as_of"
    rows = rows.sort_values(["observation_date", "available_at"], kind="stable").drop_duplicates("series", keep="last")
    observed = pd.to_datetime(rows["observation_date"], utc=True)
    rows = rows.loc[(as_of.normalize() - observed) <= max_rate_age].copy()
    tenor = pd.to_numeric(rows["tenor_days"], errors="coerce").to_numpy(float)
    rate = pd.to_numeric(rows["rate"], errors="coerce").to_numpy(float)
    if rows.empty or np.any(~np.isfinite(tenor) | (tenor <= 0) | ~np.isfinite(rate)):
        return None, None, "Treasury rate proxies are stale or invalid"
    order = np.argsort(tenor, kind="stable")
    years = tenor[order] / 365
    if np.any(np.diff(years) <= 0):
        return None, None, "Treasury rate proxies have duplicate tenors"
    # Annualized yield quote is deliberately treated as a continuous zero-rate
    # proxy, not described as a Treasury bootstrap. Origin log(D(0))=0.
    # Requested expiries must remain within the supplied tenor coverage.
    if len(maturities) and max(maturities) > years[-1] + 1e-12:
        return None, None, "Treasury rate curve does not cover requested expiries"
    log_discount = np.interp(maturities, np.r_[0, years], np.r_[0, -rate[order] * years])
    result = np.exp(log_discount)
    if not np.all(np.isfinite(result) & (result > 0)):
        return None, None, "Treasury proxy curve produces invalid discounts"
    return result, rows.iloc[order].copy(), None


def _dividend_inputs(dividends, as_of, horizon, spot, allow_proxy, coverage_through):
    # ``spot`` is retained for callers of this internal helper; a trailing
    # annual yield is not a dated American-exercise input.
    if allow_proxy:
        return None, None, ("trailing dividend-yield proxy is unsupported for American "
                            "pricing; supply a historically available cash-dividend schedule")
    if dividends is None:
        return None, None, "missing dividend observations and expectations"
    attrs = dividends.attrs if isinstance(dividends, pd.DataFrame) else {}
    rows = _table(dividends, "dividends", {"ex_date", "amount", "available_at"})
    rows["ex_date"] = pd.to_datetime(rows["ex_date"], errors="coerce").dt.date
    rows["available_at"] = _times(rows["available_at"], "dividends.available_at", allow_missing=True)
    amounts = pd.to_numeric(rows["amount"], errors="coerce")
    if amounts.isna().any() or (~np.isfinite(amounts) | (amounts < 0)).any() or rows["ex_date"].isna().any():
        raise ValueError("dividends require finite nonnegative amounts and valid ex_date values")
    rows["amount"] = amounts
    if "symbol" in rows:
        rows = rows.loc[rows["symbol"] == "SPY"]
    if "ex_at" in rows:
        rows["ex_at"] = _times(rows["ex_at"], "dividends.ex_at")
        local_days = rows["ex_at"].dt.tz_convert("America/New_York").dt.date
        if not local_days.eq(rows["ex_date"]).all():
            raise ValueError("dividends.ex_at must fall on its stated New York ex_date")
    else:
        instants = []
        for day in rows["ex_date"]:
            bounds = session_bounds(day)
            if bounds is None:
                raise ValueError(f"dividend ex_date {day} is not an XNYS session")
            instants.append(bounds[0])
        rows["ex_at"] = pd.Series(instants, index=rows.index, dtype="datetime64[ns, UTC]")
    # A later forecast, declaration, or realized record revises one economic
    # distribution; revisions are not separate cash payments. Filter first so
    # an unavailable future revision cannot overwrite the known amount.
    # Retain ``rows`` for provenance only. Unavailable event dates never select
    # or exclude contracts and never enter a historical valuation.
    available_rows = rows.loc[rows["available_at"] <= as_of]
    # When publication instants tie, an actual declaration/distribution takes
    # precedence over its superseded forecast, independent of input ordering.
    known = available_rows.assign(_realized_priority=(
        ~available_rows["is_forecast"].fillna(True).astype(bool)
        if "is_forecast" in available_rows else True))
    known = (known.sort_values(["ex_date", "available_at", "_realized_priority"], kind="stable")
        .drop_duplicates("ex_date", keep="last").drop(columns="_realized_priority"))
    coverage = coverage_through if coverage_through is not None else attrs.get("coverage_through")
    horizon_date = horizon.tz_convert("America/New_York").date()
    if coverage is not None and pd.Timestamp(coverage).date() >= horizon_date:
        # An explicit, dated expectation schedule may be empty (no expected
        # events). Its asserted horizon must cover every calibration expiry.
        future = known.loc[(known["ex_at"] > as_of) & (known["ex_at"] <= horizon)]
        return {"mode": "explicit_cash_dividend_schedule", "events": future,
                "coverage_through": str(pd.Timestamp(coverage).date()),
                "event_time_convention": "explicit ex_at or XNYS opening on ex_date"}, rows, None
    return None, rows, "dividend expectation coverage missing through calibration expiry; explicit cash-dividend schedule coverage required"


def _black_price(forward, discount, strike, maturity, is_call, volatility):
    if volatility <= 0:
        return discount * max((forward - strike) if is_call else (strike - forward), 0)
    sd = volatility * math.sqrt(maturity)
    d1 = math.log(forward / strike) / sd + 0.5 * sd
    d2 = d1 - sd
    sign = 1 if is_call else -1
    return discount * sign * (forward * ndtr(sign * d1) - strike * ndtr(sign * d2))


def _american_premium(spot, strike, maturity, rate, dividend_yield, volatility, is_call, steps):
    """CRR American minus European price on the *same* continuous-yield tree."""
    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1 / up
    probability = (math.exp((rate - dividend_yield) * dt) - down) / (up - down)
    if not 0 <= probability <= 1:
        raise ValueError("CRR probability outside [0,1]")
    step_discount = math.exp(-rate * dt)
    terminal = spot * np.exp((2 * np.arange(steps + 1) - steps) * math.log(up))
    sign = 1 if is_call else -1
    european = np.maximum(sign * (terminal - strike), 0)
    american = european.copy()
    for level in range(steps - 1, -1, -1):
        european = step_discount * ((1 - probability) * european[:-1] + probability * european[1:])
        continuation = step_discount * ((1 - probability) * american[:-1] + probability * american[1:])
        nodes = spot * np.exp((2 * np.arange(level + 1) - level) * math.log(up))
        american = np.maximum(continuation, sign * (nodes - strike))
    return max(float(american[0] - european[0]), 0.0)


def build_calibration_snapshot(
    options: pd.DataFrame, as_of, *, stock: pd.DataFrame | None = None,
    rates: pd.DataFrame | None = None, dividends: pd.DataFrame | None = None,
    min_dte=7.0, max_dte=60.0, max_quote_age="1min", max_stock_age="1min",
    max_rate_age="7days", allow_dividend_proxy=False, dividend_coverage_through=None,
    max_relative_spread=0.5, max_absolute_spread=None, min_quote_size=1,
) -> CalibrationSnapshot:
    """Build an auditable surface using observations available by ``as_of``.

    Dividend expectations require an explicit coverage horizon, either via
    ``dividend_coverage_through`` or ``dividends.attrs['coverage_through']``.
    The horizon is the caller's point-in-time completeness assertion, including
    a justified absence of events when the schedule is empty. A trailing-yield
    proxy cannot supply American cash-dividend inputs and never unlocks readiness.
    Stock requires an explicit raw/unadjusted price basis and an observation
    within the same XNYS regular session; its allowed age cannot exceed 1 min.

    ITM contracts and contracts spanning known dividends remain eligible. No
    European-price bound, CRR premium screen, or future event-calendar exclusion
    is applied. Quotes remain observed American prices, without conversion.
    """
    as_of = _utc(as_of, "as_of")
    stock_age = _positive_duration(max_stock_age, "max_stock_age")
    if stock_age > pd.Timedelta(minutes=1):
        raise ValueError("max_stock_age must not exceed 1 minute")
    if not (math.isfinite(min_dte) and math.isfinite(max_dte) and 0 <= min_dte <= max_dte):
        raise ValueError("require finite 0 <= min_dte <= max_dte")
    for name, value in (("max_relative_spread", max_relative_spread),):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if max_absolute_spread is not None and (not math.isfinite(max_absolute_spread) or max_absolute_spread < 0):
        raise ValueError("max_absolute_spread must be finite and nonnegative")
    if isinstance(min_quote_size, bool) or not isinstance(min_quote_size, (int, float)) or not math.isfinite(min_quote_size) or min_quote_size < 0:
        raise ValueError("min_quote_size must be finite and nonnegative")
    quotes, excluded = _candidate_selection(options, as_of, min_dte, max_dte,
        _positive_duration(max_quote_age, "max_quote_age"))
    provenance = {"schema_version": SNAPSHOT_SCHEMA_VERSION,
        "calibration_backend": CALIBRATION_BACKEND, "exercise_style": "american",
        "time_basis": "ACT/365 using actual UTC expiry instant",
        "option_quote_basis": "latest observable contract quote; no carry beyond max_quote_age",
        "exercise_policy": "direct American calibration; no exercise-premium or ex-date exclusions",
        "exercise_model_limitation": "American BS/Heston calibration only; rough calibration and historical factor-state estimation are unsupported",
        "forward_usage": "descriptive prepaid-cash carry reference only; never a calibration input",
        "candidate_count": len(quotes), "max_quote_age": str(pd.Timedelta(max_quote_age))}
    reasons = pd.Series("", index=quotes.index, dtype=object)

    def reject(mask, reason):
        reasons.loc[mask] = reasons.loc[mask].map(lambda old: f"{old};{reason}".strip(";"))

    midpoint = quotes["market_price"].to_numpy(float)
    spread = quotes["ask"].to_numpy(float) - quotes["bid"].to_numpy(float)
    reject(spread > max_relative_spread * midpoint, "wide_relative_spread")
    if max_absolute_spread is not None:
        reject(spread > max_absolute_spread, "wide_absolute_spread")
    for field in ("bid_size", "ask_size"):
        if field not in quotes:
            reject(np.ones(len(quotes), dtype=bool), f"missing_{field}")
        else:
            sizes = pd.to_numeric(quotes[field], errors="coerce").to_numpy(float)
            reject(~np.isfinite(sizes) | (sizes <= 0) | (sizes < min_quote_size),
                   f"insufficient_{field}")
    rejected = quotes.loc[reasons != ""].copy()
    rejected["snapshot_exclusion_reason"] = reasons.loc[reasons != ""]
    excluded = pd.concat([excluded, rejected], ignore_index=True)
    quotes = quotes.loc[reasons == ""].reset_index(drop=True)
    missing = []
    stock_row, error = _stock_observation(stock, as_of, stock_age)
    if error:
        missing.append(error)
    else:
        provenance["stock"] = stock_row.to_dict()
    discount, curve_rows, error = _discount_curve(rates, as_of, quotes["maturity"].to_numpy(),
        _positive_duration(max_rate_age, "max_rate_age"))
    if error:
        missing.append(error)
    else:
        provenance["rates"] = {"construction": "linear interpolation of log discounts from annualized Treasury yield proxies; ACT/365",
            "observations": curve_rows.to_dict("records"), "financing_series": "SOFR is not used as a term curve"}
    spot = float(stock_row["spot"]) if stock_row is not None else None
    horizon = quotes["expiry_at"].max() if not quotes.empty else as_of
    div_input = None
    # A missing spot cannot prevent the dividend coverage gap being reported.
    div_input, _, error = _dividend_inputs(dividends, as_of, horizon,
        spot if spot is not None else 1.0, allow_dividend_proxy, dividend_coverage_through)
    if error:
        missing.append(error)
    else:
        provenance["dividends"] = {key: value for key, value in div_input.items() if key != "events"}
        provenance["dividends"]["observations"] = div_input["events"].to_dict("records")
    if quotes.empty:
        missing.append("no eligible observable option contracts")
    if missing:
        nan = np.full(len(quotes), np.nan)
        return CalibrationSnapshot(as_of, quotes, nan.copy(), nan.copy(), False,
            tuple(missing), excluded, MappingProxyType(provenance))

    years = curve_rows["tenor_days"].to_numpy(float) / 365
    zero_rates = curve_rows["rate"].to_numpy(float)
    curve = DiscountCurve(times=tuple(np.r_[0.0, years]),
        discounts=tuple(np.r_[1.0, np.exp(-zero_rates * years)]))
    events = tuple(CashDividend(
        time=(event["ex_at"] - as_of).total_seconds() / YEAR_SECONDS,
        amount=float(event["amount"]))
        for _, event in div_input["events"].iterrows())
    # The caller's dated assertion establishes coverage of the whole snapshot.
    # Restrict the numerical schedule to that snapshot's horizon because events
    # after its last expiry have deliberately not been included above.
    schedule = DividendSchedule(events=events,
        coverage=(horizon - as_of).total_seconds() / YEAR_SECONDS)
    market = SpotMarket(spot=spot, discount_curve=curve, dividends=schedule)
    pv_dividends = np.zeros(len(quotes))
    for _, event in div_input["events"].iterrows():
        event_time = (event["ex_at"] - as_of).total_seconds() / YEAR_SECONDS
        pv_dividends += np.where(event["ex_at"] <= quotes["expiry_at"],
                                  float(event["amount"]) * curve.discount(event_time), 0)
    forward = (spot - pv_dividends) / discount
    quotes["forward"] = forward
    quotes["discount"] = discount
    quotes["spot"] = spot
    provenance["ex_post_dividend_exclusion_count"] = 0
    return CalibrationSnapshot(as_of, quotes, forward.copy(), discount.copy(),
        True, (), excluded, MappingProxyType(provenance), market)


__all__ = ["CalibrationSnapshot", "MarketDataNotReady", "candidate_snapshot",
           "build_calibration_snapshot", "SNAPSHOT_SCHEMA_VERSION", "CALIBRATION_BACKEND"]
