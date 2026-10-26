"""Model-neutral, self-financing delta-hedge backtests.

The backtester consumes already-computed hedge policies.  It therefore makes no
assumption about the pricing model used to produce a delta: every policy is just
an array aligned with the supplied spot paths and time grid.

Accounting convention
---------------------
One option is sold at ``time_grid[0]`` and its premium is deposited in the cash
account.  The hedge owns the policy delta in shares.  Between rebalances, cash
accrues at the continuously compounded risk-free rate and the shares earn a
continuous dividend-yield cashflow, discretized on the interval's opening spot.
Trades and their costs are paid from cash.  At expiry, the pre-expiry hedge is
liquidated, transaction costs are charged, and the option payoff is paid.

The final column of each supplied delta matrix is retained in the ledger as the
policy's expiry signal, but is not traded: there is no hedge interval after
expiry.  The executed terminal target is zero because the stock hedge is closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class HedgeBacktestConfig:
    """Execution-cost assumptions for one option contract.

    ``transaction_cost_bps`` is a one-way proportional charge on absolute stock
    notional.  ``fixed_cost_per_trade`` is charged whenever the absolute share
    trade exceeds ``trade_tolerance``.  Opening and terminal-liquidation trades
    are both included.
    """

    transaction_cost_bps: float = 0.0
    fixed_cost_per_trade: float = 0.0
    trade_tolerance: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "transaction_cost_bps": self.transaction_cost_bps,
            "fixed_cost_per_trade": self.fixed_cost_per_trade,
            "trade_tolerance": self.trade_tolerance,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class HedgeBacktestResult:
    """Tidy pathwise accounting ledger and one-row-per-policy summary."""

    ledger: pd.DataFrame
    summary: pd.DataFrame


def run_delta_hedge_backtest(
    *,
    spot_paths: np.ndarray,
    time_grid: np.ndarray,
    policy_deltas: Mapping[str, np.ndarray],
    premium: float,
    strike: float,
    is_call: bool,
    rate: float = 0.0,
    div_yield: float = 0.0,
    config: HedgeBacktestConfig | None = None,
) -> HedgeBacktestResult:
    """Backtest pre-computed delta policies against common realized spot paths.

    Parameters
    ----------
    spot_paths:
        Positive finite spot paths with shape ``(n_paths, n_times)``.
    time_grid:
        Strictly increasing finite times, in years, with shape ``(n_times,)``.
    policy_deltas:
        Non-empty mapping from policy name to a finite delta matrix with exactly
        the same shape as ``spot_paths``.  The caller is responsible for making
        externally supplied policies causal; helpers in this module only use
        information available at or before each time.
    premium:
        Scalar premium received for selling one option at the initial time.
    strike, is_call:
        European option payoff definition.
    rate, div_yield:
        Flat continuously compounded annual rates.
    config:
        Proportional and fixed stock-trading costs.

    Returns
    -------
    HedgeBacktestResult
        ``ledger`` has one row per policy, path, and grid time.  ``summary`` has
        one row per policy with terminal P&L and trading diagnostics.
    """

    spots, times, deltas_by_policy = _validate_backtest_inputs(
        spot_paths=spot_paths,
        time_grid=time_grid,
        policy_deltas=policy_deltas,
        premium=premium,
        strike=strike,
        is_call=is_call,
        rate=rate,
        div_yield=div_yield,
    )
    execution = config if config is not None else HedgeBacktestConfig()
    if not isinstance(execution, HedgeBacktestConfig):
        raise TypeError("config must be a HedgeBacktestConfig or None")

    n_paths, n_times = spots.shape
    cost_rate = float(execution.transaction_cost_bps) * 1e-4
    ledger_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    terminal_spot = spots[:, -1]
    if is_call:
        payoff = np.maximum(terminal_spot - float(strike), 0.0)
    else:
        payoff = np.maximum(float(strike) - terminal_spot, 0.0)

    for policy_name, policy_delta in deltas_by_policy.items():
        cash = np.full(n_paths, float(premium), dtype=float)
        shares = np.zeros(n_paths, dtype=float)
        cumulative_turnover = np.zeros(n_paths, dtype=float)
        cumulative_costs = np.zeros(n_paths, dtype=float)
        policy_parts: list[pd.DataFrame] = []

        for time_index in range(n_times):
            spot = spots[:, time_index]
            cash_before = cash.copy()
            shares_before = shares.copy()

            if time_index == 0:
                interest_cashflow = np.zeros(n_paths, dtype=float)
                dividend_cashflow = np.zeros(n_paths, dtype=float)
                event = "initial_hedge"
            else:
                dt = float(times[time_index] - times[time_index - 1])
                interest_cashflow = cash_before * np.expm1(float(rate) * dt)
                dividend_cashflow = (
                    shares_before
                    * spots[:, time_index - 1]
                    * np.expm1(float(div_yield) * dt)
                )
                event = "expiry" if time_index == n_times - 1 else "rebalance"

            cash_before_trade = cash_before + interest_cashflow + dividend_cashflow
            policy_signal = policy_delta[:, time_index]
            target_delta = (
                np.zeros(n_paths, dtype=float)
                if time_index == n_times - 1
                else policy_signal
            )
            trade_shares = target_delta - shares_before
            trade_notional = np.abs(trade_shares) * spot
            has_trade = np.abs(trade_shares) > float(execution.trade_tolerance)
            transaction_cost = (
                cost_rate * trade_notional
                + float(execution.fixed_cost_per_trade) * has_trade.astype(float)
            )
            trade_cashflow = -trade_shares * spot
            option_payoff = payoff if time_index == n_times - 1 else np.zeros(n_paths)
            cash_after = (
                cash_before_trade
                + trade_cashflow
                - transaction_cost
                - option_payoff
            )
            shares_after = target_delta
            stock_value = shares_after * spot
            portfolio_value = cash_after + stock_value

            cumulative_turnover = cumulative_turnover + np.abs(trade_shares)
            cumulative_costs = cumulative_costs + transaction_cost
            terminal_pnl = (
                portfolio_value
                if time_index == n_times - 1
                else np.full(n_paths, np.nan, dtype=float)
            )

            policy_parts.append(
                pd.DataFrame(
                    {
                        "policy": policy_name,
                        "path_id": np.arange(n_paths, dtype=int),
                        "time_index": time_index,
                        "time": float(times[time_index]),
                        "event": event,
                        "spot": spot,
                        "policy_delta": policy_signal,
                        "target_delta": target_delta,
                        "shares_before": shares_before,
                        "trade_shares": trade_shares,
                        "shares_after": shares_after,
                        "trade_notional": trade_notional,
                        "transaction_cost": transaction_cost,
                        "cash_before": cash_before,
                        "interest_cashflow": interest_cashflow,
                        "dividend_cashflow": dividend_cashflow,
                        "cash_before_trade": cash_before_trade,
                        "trade_cashflow": trade_cashflow,
                        "option_payoff": option_payoff,
                        "cash_after": cash_after,
                        "stock_value": stock_value,
                        "portfolio_value": portfolio_value,
                        "cumulative_turnover": cumulative_turnover,
                        "cumulative_transaction_costs": cumulative_costs,
                        "pnl": terminal_pnl,
                    }
                )
            )

            cash = cash_after
            shares = shares_after

        policy_ledger = pd.concat(policy_parts, ignore_index=True)
        ledger_parts.append(policy_ledger)

        pnl = cash.copy()
        q05, q50, q95 = np.quantile(pnl, [0.05, 0.50, 0.95])
        left_tail = pnl[pnl <= q05]
        summary_rows.append(
            {
                "policy": policy_name,
                "n_paths": n_paths,
                "mean": float(np.mean(pnl)),
                "std": float(np.std(pnl, ddof=1)) if n_paths > 1 else 0.0,
                "rmse": float(np.sqrt(np.mean(np.square(pnl)))),
                "mae": float(np.mean(np.abs(pnl))),
                "q05": float(q05),
                "q50": float(q50),
                "q95": float(q95),
                "cvar_05": float(np.mean(left_tail)),
                "mean_turnover": float(np.mean(cumulative_turnover)),
                "mean_transaction_costs": float(np.mean(cumulative_costs)),
            }
        )

    ledger = pd.concat(ledger_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return HedgeBacktestResult(ledger=ledger, summary=summary)


def causal_rolling_realized_vol(
    *,
    spot_paths: np.ndarray,
    time_grid: np.ndarray,
    window: int,
    min_periods: int = 1,
    initial_vol: float | None = None,
) -> np.ndarray:
    """Estimate rolling annualized volatility without using future returns.

    At column ``i`` the estimate uses at most the ``window`` log returns ending
    at ``i``.  Since ``time_grid`` is measured in years, realized variance is
    ``sum(log_return**2) / sum(dt)``.  The first column, and any column before
    ``min_periods`` returns are available, is ``initial_vol`` when provided and
    ``NaN`` otherwise.
    """

    spots, times = _validate_spots_and_times(spot_paths, time_grid)
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")
    if (
        not isinstance(min_periods, (int, np.integer))
        or isinstance(min_periods, bool)
        or min_periods < 1
        or min_periods > window
    ):
        raise ValueError("min_periods must be an integer between 1 and window")
    if initial_vol is not None:
        if not math.isfinite(float(initial_vol)) or float(initial_vol) < 0.0:
            raise ValueError("initial_vol must be finite and non-negative")

    n_paths, n_times = spots.shape
    fill_value = np.nan if initial_vol is None else float(initial_vol)
    realized = np.full((n_paths, n_times), fill_value, dtype=float)
    log_returns = np.diff(np.log(spots), axis=1)
    dt = np.diff(times)

    for time_index in range(1, n_times):
        start_return = max(0, time_index - int(window))
        n_observations = time_index - start_return
        if n_observations < int(min_periods):
            continue
        squared_returns = np.square(log_returns[:, start_return:time_index])
        elapsed = float(np.sum(dt[start_return:time_index]))
        realized[:, time_index] = np.sqrt(np.sum(squared_returns, axis=1) / elapsed)

    return realized


def black_scholes_delta_matrix(
    *,
    spot_paths: np.ndarray,
    time_grid: np.ndarray,
    strike: float,
    volatility: float | np.ndarray,
    rate: float = 0.0,
    div_yield: float = 0.0,
    is_call: bool = True,
) -> np.ndarray:
    """Compute point-in-time Black--Scholes deltas on a path/time matrix.

    ``volatility`` may be scalar or broadcastable to ``spot_paths``.  Each
    output column depends only on the spot and volatility in that same column,
    plus deterministic time-to-expiry; the helper never reads a future spot.
    """

    spots, times = _validate_spots_and_times(spot_paths, time_grid)
    _validate_finite_scalar("strike", strike, strictly_positive=True)
    _validate_finite_scalar("rate", rate)
    _validate_finite_scalar("div_yield", div_yield)
    if not isinstance(is_call, (bool, np.bool_)):
        raise ValueError("is_call must be a boolean")

    raw_vol = np.asarray(volatility, dtype=float)
    try:
        vols = np.broadcast_to(raw_vol, spots.shape)
    except ValueError as exc:
        raise ValueError("volatility must be scalar or broadcastable to spot_paths") from exc
    if not np.all(np.isfinite(vols)):
        raise ValueError("volatility must contain only finite values")
    if np.any(vols < 0.0):
        raise ValueError("volatility must be non-negative")

    tau = np.broadcast_to(times[-1] - times, spots.shape)
    result = np.empty_like(spots, dtype=float)
    live = tau > 0.0
    positive_vol = live & (vols > 0.0)
    zero_vol = live & ~positive_vol

    if np.any(positive_vol):
        live_spot = spots[positive_vol]
        live_tau = tau[positive_vol]
        live_vol = vols[positive_vol]
        d1 = (
            np.log(live_spot / float(strike))
            + (float(rate) - float(div_yield) + 0.5 * np.square(live_vol)) * live_tau
        ) / (live_vol * np.sqrt(live_tau))
        discounted_cdf = np.exp(-float(div_yield) * live_tau) * ndtr(d1)
        if is_call:
            result[positive_vol] = discounted_cdf
        else:
            result[positive_vol] = discounted_cdf - np.exp(
                -float(div_yield) * live_tau
            )

    if np.any(zero_vol):
        forward = spots[zero_vol] * np.exp(
            (float(rate) - float(div_yield)) * tau[zero_vol]
        )
        discounted_one = np.exp(-float(div_yield) * tau[zero_vol])
        if is_call:
            result[zero_vol] = np.where(forward > float(strike), discounted_one, 0.0)
        else:
            result[zero_vol] = np.where(forward < float(strike), -discounted_one, 0.0)

    expired = ~live
    if is_call:
        result[expired] = (spots[expired] > float(strike)).astype(float)
    else:
        result[expired] = -(spots[expired] < float(strike)).astype(float)
    return result


def _validate_backtest_inputs(
    *,
    spot_paths: np.ndarray,
    time_grid: np.ndarray,
    policy_deltas: Mapping[str, np.ndarray],
    premium: float,
    strike: float,
    is_call: bool,
    rate: float,
    div_yield: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    spots, times = _validate_spots_and_times(spot_paths, time_grid)
    _validate_finite_scalar("premium", premium, non_negative=True)
    _validate_finite_scalar("strike", strike, strictly_positive=True)
    _validate_finite_scalar("rate", rate)
    _validate_finite_scalar("div_yield", div_yield)
    if not isinstance(is_call, (bool, np.bool_)):
        raise ValueError("is_call must be a boolean")

    if not isinstance(policy_deltas, Mapping) or not policy_deltas:
        raise ValueError("policy_deltas must be a non-empty mapping")

    validated: dict[str, np.ndarray] = {}
    for policy_name, raw_delta in policy_deltas.items():
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise ValueError("policy names must be non-empty strings")
        delta = np.asarray(raw_delta, dtype=float)
        if delta.shape != spots.shape:
            raise ValueError(
                f"policy {policy_name!r} deltas must have the same shape as "
                f"spot_paths; got {delta.shape} and {spots.shape}"
            )
        if not np.all(np.isfinite(delta)):
            raise ValueError(f"policy {policy_name!r} deltas must be finite")
        validated[policy_name] = delta
    return spots, times, validated


def _validate_spots_and_times(
    spot_paths: np.ndarray,
    time_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spots = np.asarray(spot_paths, dtype=float)
    times = np.asarray(time_grid, dtype=float)
    if spots.ndim != 2:
        raise ValueError("spot_paths must be 2D with shape (n_paths, n_times)")
    if spots.shape[0] < 1 or spots.shape[1] < 2:
        raise ValueError("spot_paths must contain at least one path and two times")
    if not np.all(np.isfinite(spots)):
        raise ValueError("spot_paths must contain only finite values")
    if np.any(spots <= 0.0):
        raise ValueError("spot_paths must be strictly positive")
    if times.ndim != 1 or times.shape[0] != spots.shape[1]:
        raise ValueError("time_grid must be 1D and align with spot_paths")
    if not np.all(np.isfinite(times)):
        raise ValueError("time_grid must contain only finite values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_grid must be strictly increasing")
    return spots, times


def _validate_finite_scalar(
    name: str,
    value: float,
    *,
    non_negative: bool = False,
    strictly_positive: bool = False,
) -> None:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    if strictly_positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")


__all__ = [
    "HedgeBacktestConfig",
    "HedgeBacktestResult",
    "black_scholes_delta_matrix",
    "causal_rolling_realized_vol",
    "run_delta_hedge_backtest",
]
