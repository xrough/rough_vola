"""Historical synthetic controls for zero-dividend exercise sensitivity.

These detached research tables do not estimate independent stock observations
or turn SPY into a European security.  The hypothetical spot is ``F * D`` and
the rate is ``-log(D) / T``.  No dividend before expiry is an explicit scenario,
not a historically available forecast.  A stochastic-volatility American
premium can differ from the Black--Scholes diagnostic used here. This module
is retired for real SPY workflows; explicitly American observations are
rejected instead of constructing de-Americanized calibration targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from rvlab.market_data.snapshots import _american_premium
from rvlab.pricing import BlackScholesParams, black_scholes_price
from ._pricing_data import implied_volatility, parity_forward


EXERCISE_PROXY_MODE = "american_bs_zero_dividend_sensitivity_proxy"


@dataclass(frozen=True)
class ExerciseProxySurface:
    """Detached sensitivity inputs; frame order matches the supplied tables."""

    quotes: pd.DataFrame
    anchors: pd.DataFrame
    diagnostics: pd.DataFrame
    provenance: dict
    tree_steps: int


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _steps(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 16:
        raise ValueError("steps must be an integer >= 16")
    return int(value)


def american_bs_premium(*, forward, discount, strike, maturity, volatility,
                        is_call, steps=1024):
    """Return the American minus European premium on the same CRR tree.

    Forward and discount imply the hypothetical no-dividend spot ``F*D``.
    Using the same tree for both exercise styles removes ordinary European
    lattice pricing error from the premium.  Zero-variance and expiration
    limits are handled without a degenerate tree.  The result is per share.
    """
    f = _real(forward, "forward", positive=True)
    d = _real(discount, "discount", positive=True)
    k = _real(strike, "strike", positive=True)
    t = _real(maturity, "maturity")
    vol = _real(volatility, "volatility")
    n = _steps(steps)
    if not isinstance(is_call, (bool, np.bool_)):
        raise ValueError("is_call must be a boolean")
    call = bool(is_call)
    spot = f * d
    if not math.isfinite(spot):
        raise ValueError("forward * discount must be finite")
    if t == 0 or d == 1 or (call and d < 1) or (not call and d > 1):
        return 0.
    sign = 1. if call else -1.
    if vol == 0:
        return max(max(sign * (spot - k), 0.) - d * max(sign * (f - k), 0.), 0.)
    rate = -math.log(d) / t
    try:
        premium = _american_premium(spot, k, t, rate, 0., vol, call, n)
    except (OverflowError, FloatingPointError, ZeroDivisionError) as exc:
        raise ValueError(f"American BS premium numerical failure: {exc}") from exc
    if not math.isfinite(premium):
        raise ValueError("American BS premium produced a nonfinite result")
    return premium


def _european(f, d, k, t, vol, call):
    return float(black_scholes_price(BlackScholesParams(vol), forward=f,
        discount=d, strike=k, maturity=t, is_call=call))


def _american_iv(observed, *, forward, discount, strike, maturity, steps):
    def value(vol):
        premium = american_bs_premium(forward=forward, discount=discount,
            strike=strike, maturity=maturity, volatility=vol, is_call=False, steps=steps)
        return _european(forward, discount, strike, maturity, vol, False) + premium
    lower = value(0.)
    if observed < lower - 1e-10:
        raise ValueError(f"American put price {observed} is below diagnostic intrinsic {lower}")
    if abs(observed - lower) <= 1e-10:
        return 0.
    upper = .5
    while value(upper) < observed and upper < 16.:
        upper *= 2
    if value(upper) < observed:
        raise ValueError("American BS implied-volatility bracket failed")
    return float(brentq(lambda vol: value(vol) - observed, 0., upper,
                       xtol=1e-10, rtol=1e-10))


def _anchor_forward(call, put, strike, discount, maturity, steps):
    raw = float(parity_forward(call, put, strike, discount))
    def diagnostic(f):
        vol = float(implied_volatility(call, forward=f, discount=discount,
            strike=strike, maturity=maturity, is_call=True))
        premium = american_bs_premium(forward=f, discount=discount, strike=strike,
            maturity=maturity, volatility=vol, is_call=False, steps=steps)
        return f - raw - premium / discount, vol, premium
    residual, vol, premium = diagnostic(raw)
    if abs(residual) <= 1e-11:
        return raw, vol, premium
    # A positive-rate, no-dividend put premium raises the parity forward.
    # The call intrinsic bound constrains the root's admissible domain.
    limit = strike + call / discount
    admissible = limit - max(1e-9, 1e-11 * limit)
    step = max(.05, 2 * premium / discount)
    upper = min(raw + step, admissible)
    for _ in range(32):
        if upper <= raw:
            raise ValueError("no admissible American-parity forward bracket")
        if diagnostic(upper)[0] >= 0:
            break
        if upper >= admissible:
            raise ValueError("American-parity forward bracket failed before call intrinsic bound")
        step *= 2
        upper = min(raw + step, admissible)
    else:
        raise ValueError("American-parity forward bracket failed")
    fitted = float(brentq(lambda f: diagnostic(f)[0], raw, upper, xtol=1e-9, rtol=1e-12))
    _, vol, premium = diagnostic(fitted)
    return fitted, vol, premium


def _table(frame, name, required):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{name} must be a DataFrame with unique columns")
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty; no synthetic fallback is available")
    if "input_mode" in frame and (frame.input_mode == EXERCISE_PROXY_MODE).any():
        raise ValueError(f"{name} already contains exercise-adjusted sensitivity inputs")
    out = frame.copy(deep=True).reset_index(drop=True)
    for pos, row in out.iterrows():
        try:
            for column in required:
                if column in {"snapshot_date", "expiry_date", "is_call"}:
                    if pd.isna(row[column]):
                        raise ValueError(f"{column} is missing")
                else:
                    _real(row[column], column, positive=column not in {"maturity"})
            if row.maturity <= 0:
                raise ValueError("maturity must be positive")
            if row.discount > 1:
                raise ValueError("exercise surface proxy requires nonnegative rates (discount <= 1)")
        except ValueError as exc:
            raise ValueError(f"{name} row position {pos}: {exc}") from exc
    return out


def exercise_adjusted_surface(quotes, anchors, *, steps=1024):
    """Construct an explicitly hypothetical BS/no-dividend exercise scenario.

    Each call/put anchor gets its own self-consistent forward: its call-implied
    BS volatility and same-tree put premium reproduce that pair. The median
    forward is then used to invert each selected put's *American* BS price.
    Subtracting that premium from both bid and ask preserves quoted widths.
    Calls receive zero premium under this nonnegative-rate, no-dividend case.

    Run independently at 512 and 1024 steps to assess the proxy's lattice
    sensitivity. This is not an American Heston or rough-Heston pricer and is
    never a correction certified by independent stock or dividend observations.
    """
    for name, frame in (("quotes", quotes), ("anchors", anchors)):
        if isinstance(frame, pd.DataFrame) and "exercise_style" in frame:
            bad = frame.exercise_style.astype(str).str.lower().eq("american").to_numpy()
            if bad.any():
                raise ValueError(f"{name} American observations at row positions {np.flatnonzero(bad).tolist()} "
                    "require direct American pricing; de-Americanized SPY targets are retired")
    n = _steps(steps)
    keys = ["snapshot_date", "expiry_date"]
    q = _table(quotes, "quotes", keys + ["strike", "maturity", "discount",
        "forward", "is_call", "bid", "ask"])
    a = _table(anchors, "anchors", keys + ["strike", "maturity", "discount",
        "market_price_call", "market_price_put"])
    if a.duplicated(keys + ["strike"]).any():
        raise ValueError("anchors contain duplicate date/expiry/strike rows")
    for pos, row in a.iterrows():
        for side in ("call", "put"):
            bid, ask = f"bid_{side}", f"ask_{side}"
            if (bid in a) != (ask in a):
                raise ValueError(f"anchors require both {bid} and {ask} when either is provided")
            if bid in a:
                try:
                    low, high = _real(row[bid], bid, positive=True), _real(row[ask], ask, positive=True)
                    if high < low:
                        raise ValueError(f"{ask} must be >= {bid}")
                    if not np.isclose(row[f"market_price_{side}"], (low+high)/2, atol=1e-10, rtol=0):
                        raise ValueError(f"market_price_{side} must equal bid/ask midpoint")
                except ValueError as exc:
                    raise ValueError(f"anchors row position {pos}: {exc}") from exc
    for pos, row in q.iterrows():
        if not isinstance(row.is_call, (bool, np.bool_)):
            raise ValueError(f"quotes row position {pos}: is_call must be boolean")
        if row.ask < row.bid:
            raise ValueError(f"quotes row position {pos}: ask must be >= bid")
        mid = (row.bid + row.ask) / 2
        if "market_price" in q:
            try:
                observed = _real(row.market_price, "market_price", positive=True)
                if not np.isclose(observed, mid, atol=1e-10, rtol=0):
                    raise ValueError("market_price must equal bid/ask midpoint")
            except ValueError as exc:
                raise ValueError(f"quotes row position {pos}: {exc}") from exc
    q["market_price"] = (q.bid + q.ask) / 2
    for frame, columns in ((q, ["forward", "bid", "ask", "market_price", "log_moneyness", "relative_spread"]),
                           (a, ["forward", "market_price_call", "market_price_put",
                                "bid_call", "ask_call", "bid_put", "ask_put",
                                "forward_lower", "forward_upper", "equality_distance"])):
        for column in columns:
            if column in frame:
                frame[f"raw_{column}"] = frame[column].copy()
        if "input_mode" in frame:
            frame["raw_input_mode"] = frame.input_mode.copy()
        frame["input_mode"] = EXERCISE_PROXY_MODE
        frame["eligible_for_hedging"] = False
        frame["independent_stock_observation"] = False
        frame["tree_steps"] = n
    q["raw_target_price"] = q.market_price
    diagnostics = []
    quote_groups = q.groupby(keys, sort=False).groups
    anchor_groups = a.groupby(keys, sort=False).groups
    if set(quote_groups) != set(anchor_groups):
        raise ValueError("quotes and anchors must cover identical date/expiry groups")
    for key, positions in anchor_groups.items():
        qi = quote_groups[key]
        group = a.loc[positions]
        aligned = pd.concat([group[["maturity", "discount"]], q.loc[qi, ["maturity", "discount"]]])
        if aligned.maturity.nunique() != 1 or aligned.discount.nunique() != 1:
            raise ValueError(f"{key}: quotes and anchors require one matching maturity and discount")
        d, t = float(group.discount.iloc[0]), float(group.maturity.iloc[0])
        raw_forwards, fitted_forwards = [], []
        for pos, row in group.iterrows():
            try:
                raw = float(parity_forward(row.market_price_call, row.market_price_put, row.strike, d))
                f, vol, premium = _anchor_forward(float(row.market_price_call),
                    float(row.market_price_put), float(row.strike), d, t, n)
            except ValueError as exc:
                raise ValueError(f"anchors row position {pos}: {exc}") from exc
            a.loc[pos, "raw_forward"] = raw
            a.loc[pos, "forward"] = f
            a.loc[pos, "exercise_diagnostic_volatility"] = vol
            a.loc[pos, "exercise_premium_put"] = premium
            a.loc[pos, "market_price_put"] = row.market_price_put - premium
            for field in ("bid_put", "ask_put"):
                if field in a:
                    a.loc[pos, field] = float(row[field]) - premium
            a.loc[pos, "hypothetical_spot"] = f * d
            if {"bid_call", "ask_call", "bid_put", "ask_put"} <= set(a):
                a.loc[pos, "forward_lower"] = row.strike + (a.loc[pos, "bid_call"] - a.loc[pos, "ask_put"]) / d
                a.loc[pos, "forward_upper"] = row.strike + (a.loc[pos, "ask_call"] - a.loc[pos, "bid_put"]) / d
            if "equality_distance" in a:
                a.loc[pos, "equality_distance"] = abs(a.loc[pos, "market_price_call"] - a.loc[pos, "market_price_put"])
            raw_forwards.append(raw)
            fitted_forwards.append(f)
        forward = float(np.median(fitted_forwards))
        q.loc[qi, "forward"] = forward
        q.loc[qi, "forward_min"] = min(fitted_forwards)
        q.loc[qi, "forward_max"] = max(fitted_forwards)
        q.loc[qi, "hypothetical_spot"] = forward * d
        q.loc[qi, "log_moneyness"] = np.log(q.loc[qi, "strike"] / forward)
        for pos, row in q.loc[qi].iterrows():
            try:
                vol = (float(implied_volatility(row.market_price, forward=forward, discount=d,
                    strike=row.strike, maturity=t, is_call=True)) if row.is_call else
                    _american_iv(float(row.market_price), forward=forward, discount=d,
                        strike=float(row.strike), maturity=t, steps=n))
                premium = american_bs_premium(forward=forward, discount=d, strike=row.strike,
                    maturity=t, volatility=vol, is_call=row.is_call, steps=n)
                if premium >= row.bid:
                    raise ValueError("exercise premium would produce a nonpositive adjusted bid")
            except ValueError as exc:
                raise ValueError(f"quotes row position {pos}: {exc}") from exc
            q.loc[pos, "exercise_diagnostic_volatility"] = vol
            q.loc[pos, "exercise_premium"] = premium
            q.loc[pos, ["bid", "ask", "market_price"]] = np.asarray([row.bid, row.ask, row.market_price]) - premium
            q.loc[pos, "exercise_premium_half_spreads"] = premium / max((row.ask-row.bid)/2, .005)
            if "price" in q:
                q.loc[pos, "price"] = row.market_price - premium
            if "relative_spread" in q:
                q.loc[pos, "relative_spread"] = (row.ask-row.bid) / (row.market_price-premium)
        diagnostics.append({**dict(zip(keys, key)), "tree_steps": n,
            "raw_forward_median": float(np.median(raw_forwards)),
            "forward_median": forward,
            "forward_change": forward-float(np.median(raw_forwards)),
            "raw_anchor_dispersion": max(raw_forwards)-min(raw_forwards),
            "anchor_dispersion": max(fitted_forwards)-min(fitted_forwards),
            "hypothetical_spot": forward*d,
            "max_selected_premium": float(q.loc[qi, "exercise_premium"].max()),
            "input_mode": EXERCISE_PROXY_MODE, "eligible_for_hedging": False})
    provenance = {"input_mode": EXERCISE_PROXY_MODE, "tree_steps": n,
        "independent_stock_observation": False, "strict_market_data_ready": False,
        "eligible_for_hedging": False, "dividend_yield_scenario": 0.,
        "limitations": ["Black-Scholes exercise sensitivity, not an American stochastic-model valuation",
            "Hypothetical spot equals forward times discount; no independent stock observation",
            "Zero dividend before expiry is a research scenario, not a historically available forecast",
            "Proxy uncertainty is not measured by quoted bid/ask widths or tree convergence",
            "Treasury proxy, quote timing, stock borrow and model dependence remain"]}
    return ExerciseProxySurface(q, a, pd.DataFrame(diagnostics), provenance, n)


__all__ = ["EXERCISE_PROXY_MODE", "ExerciseProxySurface", "american_bs_premium",
           "exercise_adjusted_surface"]
