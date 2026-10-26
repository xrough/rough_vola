"""Small, explicit option-implied-forward samples for pricing demonstrations.

These research inputs are deliberately separate from strict market-data
readiness. European parity applied to American SPY quotes is an approximation;
an inferred forward is never an independent stock observation or hedge input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from rvlab.pricing import BlackScholesParams, black_scholes_price
from rvlab.pricing._validation import broadcast_inputs, scalar_or_array
from rvlab.market_data.snapshots import _candidate_selection, _discount_curve, _utc


DEFAULT_DATES = ("2025-08-07", "2025-08-08", "2025-08-11")
DEFAULT_EXPIRIES = ("2025-08-22", "2025-08-29", "2025-09-05")
INPUT_MODE = "option_implied_forward_proxy"
_TARGETS = (-.03, -.02, -.01, .01, .02, .03)


@dataclass(frozen=True)
class DemoSurface:
    """Detached demonstration tables; never a strict ready market snapshot."""

    quotes: pd.DataFrame
    anchors: pd.DataFrame
    diagnostics: pd.DataFrame
    exclusions: pd.DataFrame
    provenance: dict


def implied_volatility(prices, *, forward, discount, strike, maturity, is_call=True):
    """Invert forward Black prices, broadcasting all inputs.

    Intrinsic prices map to zero. Prices outside European bounds, prices at
    the infinite-volatility upper bound, and non-intrinsic expiry prices raise
    ``ValueError`` with the flattened row position. No clipping is applied.
    """
    f, d, k, t, call = broadcast_inputs(forward=forward, discount=discount,
        strike=strike, maturity=maturity, is_call=is_call)
    raw = np.asarray(prices)
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError("prices must contain real numeric values")
    try:
        observed, f, d, k, t, call = np.broadcast_arrays(
            np.asarray(prices, dtype=float), f, d, k, t, call)
    except (ValueError, TypeError) as exc:
        raise ValueError("prices and pricing inputs must broadcast") from exc
    result = np.empty(observed.shape)
    for i, (p, fi, di, ki, ti, ci) in enumerate(zip(
            observed.flat, f.flat, d.flat, k.flat, t.flat, call.flat)):
        intrinsic = di * max(fi - ki if ci else ki - fi, 0.)
        upper = di * (fi if ci else ki)
        if not np.isfinite(p) or p < intrinsic or p > upper:
            raise ValueError(f"prices at row position {i}: outside European bounds "
                             f"[{intrinsic}, {upper}]")
        if p == intrinsic:
            result.flat[i] = 0.
            continue
        if ti == 0:
            raise ValueError(f"prices at row position {i}: expiry price must equal intrinsic")
        if p == upper:
            raise ValueError(f"prices at row position {i}: upper bound has no finite implied volatility")
        # Solve total standard deviation, avoiding a fixed volatility bracket
        # that would unnecessarily restrict short maturities.
        root_t = np.sqrt(ti)
        def objective(stddev):
            return black_scholes_price(BlackScholesParams(stddev / root_t),
                forward=fi, discount=di, strike=ki, maturity=ti, is_call=bool(ci)) - p
        bracket = 1.
        while objective(bracket) < 0 and bracket < 64:
            bracket *= 2
        if objective(bracket) < 0:
            raise ValueError(f"prices at row position {i}: finite implied-volatility bracket failed")
        result.flat[i] = brentq(objective, 0., bracket, xtol=1e-13, rtol=1e-13) / root_t
    return scalar_or_array(result)


def parity_forward(call, put, strike, discount):
    """Return ``K + (C-P)/D`` with explicit European-proxy conventions."""
    try:
        c, p, k, d = np.broadcast_arrays(*[np.asarray(value, dtype=float)
            for value in (call, put, strike, discount)])
    except (TypeError, ValueError) as exc:
        raise ValueError("parity inputs must be numeric and broadcast-compatible") from exc
    if not np.all(np.isfinite(c) & np.isfinite(p) & (c >= 0) & (p >= 0)):
        raise ValueError("call and put prices must be finite and nonnegative")
    if not np.all(np.isfinite(k) & np.isfinite(d) & (k > 0) & (d > 0)):
        raise ValueError("strike and discount must be finite and positive")
    value = k + (c - p) / d
    if not np.all(np.isfinite(value) & (value > 0)):
        raise ValueError("parity-implied forward must be finite and positive")
    return scalar_or_array(value)


def _exclude(frame, reason):
    frame = frame.copy()
    frame["snapshot_exclusion_reason"] = reason
    return frame


def select_spy_snapshot(options, as_of, rates, *, expiries=DEFAULT_EXPIRIES,
                        anchor_count=7) -> DemoSurface:
    """Select deterministic OTM train/held-out quotes and disjoint parity anchors.

    Latest observable contracts are selected *before* quality filtering. Each
    expiry requires ``anchor_count`` paired strikes and six distinct non-anchor
    OTM contracts. Treasury proxies follow the dated availability construction
    used by the strict market-data package; spot and dividends are not inferred.
    """
    as_of = _utc(as_of, "as_of")
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, (int, np.integer)) or anchor_count < 1:
        raise ValueError("anchor_count must be a positive integer")
    expiry_dates = tuple(pd.Timestamp(value).date().isoformat() for value in expiries)
    if not expiry_dates or len(set(expiry_dates)) != len(expiry_dates):
        raise ValueError("expiries must be nonempty and distinct")
    if not {"bid_size", "ask_size"} <= set(options):
        raise ValueError("options require bid_size and ask_size for the demonstration")
    # The shared selector retains its detailed exclusions, unlike the public
    # coverage-only wrapper, which exposes counts rather than excluded rows.
    candidates, excluded = _candidate_selection(options, as_of, 0., 3650., pd.Timedelta("1min"))
    exclusions = [excluded]
    candidates["expiry_date"] = candidates.expiry_at.dt.tz_convert("America/New_York").dt.date.astype(str)
    keep_expiry = candidates.expiry_date.isin(expiry_dates)
    exclusions.append(_exclude(candidates.loc[~keep_expiry], "outside_demo_expiries"))
    candidates = candidates.loc[keep_expiry].copy()
    for column in ("bid", "ask", "bid_size", "ask_size", "strike"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates["half_spread"] = (candidates.ask - candidates.bid) / 2
    candidates["relative_spread"] = (candidates.ask - candidates.bid) / candidates.market_price
    reasons = pd.Series("", index=candidates.index, dtype=object)
    for mask, reason in (
        ((candidates.bid <= 0) | ~np.isfinite(candidates.bid) | ~np.isfinite(candidates.ask), "nonpositive_bid_or_nonfinite_quote"),
        ((candidates.bid_size < 1) | (candidates.ask_size < 1) |
         ~np.isfinite(candidates.bid_size) | ~np.isfinite(candidates.ask_size), "insufficient_quoted_size"),
        ((candidates.relative_spread > .25) | ~np.isfinite(candidates.relative_spread), "relative_spread_above_25pct"),
    ):
        reasons.loc[mask] = reasons.loc[mask].map(lambda old: f"{old};{reason}".strip(";"))
    rejected = candidates.loc[reasons != ""].copy()
    rejected["snapshot_exclusion_reason"] = reasons.loc[reasons != ""]
    exclusions.append(rejected)
    candidates = candidates.loc[reasons == ""].copy()
    quotes, anchors, diagnostics, rate_observations = [], [], [], []
    for expiry in expiry_dates:
        rows = candidates.loc[candidates.expiry_date == expiry].copy()
        if rows.empty:
            raise ValueError(f"{expiry}: no eligible quotes available at {as_of}")
        if rows.maturity.nunique() != 1:
            raise ValueError(f"{expiry}: inconsistent expiry timestamps")
        maturity = float(rows.maturity.iloc[0])
        discounts, used_rates, error = _discount_curve(rates, as_of,
            np.asarray([maturity]), pd.Timedelta("7days"))
        if error:
            raise ValueError(f"{expiry}: {error}")
        discount = float(discounts[0])
        rate_observations = used_rates.to_dict("records")
        # Ambiguous distinct contracts at one strike/type must not be silently
        # paired (e.g. adjusted roots); canonical SPY contracts are unique.
        if rows.duplicated(["strike", "is_call"]).any():
            raise ValueError(f"{expiry}: multiple contracts at a strike and option type")
        calls = rows.loc[rows.is_call].set_index("strike")
        puts = rows.loc[~rows.is_call].set_index("strike")
        pairs = calls[["contract_id", "bid", "ask", "market_price"]].join(
            puts[["contract_id", "bid", "ask", "market_price"]],
            how="inner", lsuffix="_call", rsuffix="_put").reset_index()
        pairs["equality_distance"] = abs(pairs.market_price_call - pairs.market_price_put)
        pairs = pairs.sort_values(["equality_distance", "strike"], kind="stable").head(anchor_count).copy()
        if len(pairs) < anchor_count:
            raise ValueError(f"{expiry}: need {anchor_count} eligible call/put pairs; found {len(pairs)}")
        pairs["forward"] = parity_forward(pairs.market_price_call, pairs.market_price_put, pairs.strike, discount)
        pairs["forward_lower"] = pairs.strike + (pairs.bid_call - pairs.ask_put) / discount
        pairs["forward_upper"] = pairs.strike + (pairs.ask_call - pairs.bid_put) / discount
        forward = float(pairs.forward.median())
        fmin, fmax = float(pairs.forward.min()), float(pairs.forward.max())
        lower, upper = float(pairs.forward_lower.max()), float(pairs.forward_upper.min())
        pairs["expiry_date"], pairs["discount"], pairs["maturity"] = expiry, discount, maturity
        pairs["snapshot_date"] = as_of.tz_convert("America/New_York").date().isoformat()
        pairs["as_of"], pairs["input_mode"], pairs["eligible_for_hedging"] = as_of, INPUT_MODE, False
        anchors.append(pairs)
        remaining = rows.loc[~rows.strike.isin(pairs.strike)].copy()
        remaining["log_moneyness"] = np.log(remaining.strike / forward)
        remaining = remaining.loc[(remaining.is_call & (remaining.strike > forward)) |
                                  (~remaining.is_call & (remaining.strike < forward))].copy()
        selected_ids = []
        for target in _TARGETS:
            side = remaining.loc[(remaining.log_moneyness < 0) == (target < 0)].copy()
            side = side.loc[~side.contract_id.isin(selected_ids)]
            if side.empty:
                raise ValueError(f"{expiry}: insufficient distinct OTM contracts for target {target:+.0%}")
            side["target_distance"] = abs(side.log_moneyness - target)
            chosen = side.sort_values(["target_distance", "strike", "contract_id"], kind="stable").iloc[[0]].copy()
            chosen["target_log_moneyness"] = target
            chosen["split"] = "heldout" if abs(target) == .02 else "train"
            chosen["forward"], chosen["forward_min"], chosen["forward_max"] = forward, fmin, fmax
            chosen["discount"] = discount
            chosen["snapshot_date"] = pairs.snapshot_date.iloc[0]
            chosen["as_of"], chosen["input_mode"], chosen["eligible_for_hedging"] = as_of, INPUT_MODE, False
            selected_ids.append(chosen.contract_id.iloc[0])
            quotes.append(chosen)
        unselected = rows.loc[~rows.contract_id.isin(selected_ids)].copy()
        unselected["snapshot_exclusion_reason"] = np.where(unselected.strike.isin(pairs.strike),
            "forward_anchor_strike", "outside_selected_demo_contracts")
        exclusions.append(unselected)
        diagnostics.append({"snapshot_date": pairs.snapshot_date.iloc[0], "expiry_date": expiry,
            "maturity": maturity, "discount": discount, "anchor_count": len(pairs),
            "forward_min": fmin, "forward_median": forward, "forward_max": fmax,
            "anchor_dispersion": fmax - fmin, "parity_interval_lower": lower,
            "parity_interval_upper": upper, "parity_intersection_empty": lower > upper,
            "train_count": 4, "heldout_count": 2, "input_mode": INPUT_MODE,
            "eligible_for_hedging": False})
    provenance = {"as_of": as_of.isoformat(), "input_mode": INPUT_MODE,
        "eligible_for_hedging": False, "independent_stock_observation": False,
        "strict_market_data_ready": False, "rates": rate_observations,
        "selection": {"max_quote_age": "1min", "min_quoted_size": 1,
            "max_relative_spread": .25, "anchor_count": int(anchor_count),
            "targets": list(_TARGETS), "heldout_targets": [-.02, .02]},
        "limitations": ["European parity applied to American SPY options",
            "Option interval timestamps do not establish quote-change freshness",
            "Treasury yield proxies are not a bootstrapped OIS curve or historical vintages",
            "Expiry selection avoids realized dividend dates retrospectively, not a point-in-time forecast"]}
    quote_table = pd.concat(quotes, ignore_index=True)
    # Historical input diagnostics may still be inspected, but their American
    # observations must never pass the European calibrator as synthetic data.
    quote_table["exercise_style"] = "american"
    return DemoSurface(quote_table, pd.concat(anchors, ignore_index=True),
        pd.DataFrame(diagnostics), pd.concat(exclusions, ignore_index=True), provenance)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def load_spy_demo(processed_root=None, dates=DEFAULT_DATES) -> DemoSurface:
    """Read canonical local minute windows; no downloads or synthetic fallback.

    Source DBN hashes are taken from the existing daily manifests. Only the
    small manifest and rates files are hashed here, never the original archive.
    ``RVLAB_FORCE_SYNTHETIC`` intentionally has no effect on this explicit API.
    """
    from rvlab.market_data.pipeline import read_option_window
    root = (Path(processed_root) if processed_root is not None else
            Path(__file__).resolve().parents[3] / "data" / "processed" / "spy")
    rates_path = root / "references" / "rates.parquet"
    if not rates_path.is_file():
        raise FileNotFoundError(f"Missing prepared Treasury proxies: {rates_path}. Run SPY data preparation first.")
    dates = tuple(pd.Timestamp(day).date().isoformat() for day in dates)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("dates must be nonempty and distinct")
    rates = pd.read_parquet(rates_path)
    surfaces, source_days = [], []
    for day in dates:
        directory = root / "options" / f"session_date={day}"
        options_path, manifest_path = directory / "quotes.parquet", directory / "manifest.json"
        for path in (options_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared SPY input: {path}. Run SPY data preparation first.")
        as_of = pd.Timestamp(f"{day} 15:45", tz="America/New_York").tz_convert("UTC")
        options = read_option_window(options_path, as_of)
        surface = select_spy_snapshot(options, as_of, rates)
        surfaces.append(surface)
        manifest = json.loads(manifest_path.read_text())
        source_days.append({"session_date": day, "as_of": as_of.isoformat(),
            "manifest": _sha256(manifest_path), "source": manifest.get("source"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "schema_fingerprint": manifest.get("schema_fingerprint"),
            "options_path": str(options_path.resolve()), "options_bytes": options_path.stat().st_size})
    provenance = {"input_mode": INPUT_MODE, "eligible_for_hedging": False,
        "independent_stock_observation": False, "strict_market_data_ready": False,
        "processed_root": str(root.resolve()), "rates_file": _sha256(rates_path),
        "source_days": source_days, "snapshots": [surface.provenance for surface in surfaces]}
    return DemoSurface(*(pd.concat([getattr(surface, name) for surface in surfaces], ignore_index=True)
        for name in ("quotes", "anchors", "diagnostics", "exclusions")), provenance)


__all__ = ["DemoSurface", "DEFAULT_DATES", "DEFAULT_EXPIRIES", "INPUT_MODE",
    "implied_volatility", "parity_forward", "select_spy_snapshot", "load_spy_demo"]
