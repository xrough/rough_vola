from __future__ import annotations

import math

import numpy as np
import pytest

from roughvol.experiments.model_comparison.hedge_backtest import (
    HedgeBacktestConfig,
    HedgeBacktestResult,
    black_scholes_delta_matrix,
    causal_rolling_realized_vol,
    run_delta_hedge_backtest,
)


def test_self_financing_rebalance_accounting_and_tidy_summary():
    result = run_delta_hedge_backtest(
        spot_paths=np.array([[100.0, 110.0, 120.0]]),
        time_grid=np.array([0.0, 0.5, 1.0]),
        policy_deltas={"candidate": np.array([[0.50, 0.75, 0.90]])},
        premium=10.0,
        strike=100.0,
        is_call=True,
    )

    assert isinstance(result, HedgeBacktestResult)
    ledger = result.ledger
    assert ledger[["policy", "path_id", "time_index"]].to_dict("records") == [
        {"policy": "candidate", "path_id": 0, "time_index": 0},
        {"policy": "candidate", "path_id": 0, "time_index": 1},
        {"policy": "candidate", "path_id": 0, "time_index": 2},
    ]
    assert np.allclose(ledger["cash_after"], [-40.0, -67.5, 2.5])
    assert np.allclose(ledger["trade_shares"], [0.50, 0.25, -0.75])
    assert np.allclose(ledger["option_payoff"], [0.0, 0.0, 20.0])

    expiry = ledger.iloc[-1]
    assert expiry["event"] == "expiry"
    assert expiry["policy_delta"] == pytest.approx(0.90)
    assert expiry["target_delta"] == pytest.approx(0.0)
    assert expiry["shares_after"] == pytest.approx(0.0)
    assert expiry["pnl"] == pytest.approx(2.5)

    summary = result.summary.iloc[0]
    assert summary["policy"] == "candidate"
    assert summary["n_paths"] == 1
    for metric in ["mean", "rmse", "mae", "q05", "q50", "q95", "cvar_05"]:
        assert summary[metric] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(0.0)
    assert summary["mean_turnover"] == pytest.approx(1.5)
    assert summary["mean_transaction_costs"] == pytest.approx(0.0)


def test_cash_accrual_dividends_and_transaction_costs():
    risk_free = run_delta_hedge_backtest(
        spot_paths=np.array([[100.0, 100.0]]),
        time_grid=np.array([0.0, 1.0]),
        policy_deltas={"unhedged": np.zeros((1, 2))},
        premium=10.0,
        strike=200.0,
        is_call=True,
        rate=math.log(1.21),
    )
    assert risk_free.summary.loc[0, "mean"] == pytest.approx(12.1)
    assert risk_free.ledger.iloc[-1]["interest_cashflow"] == pytest.approx(2.1)

    with_costs = run_delta_hedge_backtest(
        spot_paths=np.array([[100.0, 100.0]]),
        time_grid=np.array([0.0, 1.0]),
        policy_deltas={"covered": np.array([[1.0, 99.0]])},
        premium=100.0,
        strike=200.0,
        is_call=True,
        div_yield=math.log(1.1),
        config=HedgeBacktestConfig(
            transaction_cost_bps=100.0,
            fixed_cost_per_trade=0.5,
        ),
    )

    expiry = with_costs.ledger.iloc[-1]
    assert expiry["dividend_cashflow"] == pytest.approx(10.0)
    assert expiry["transaction_cost"] == pytest.approx(1.5)
    assert expiry["pnl"] == pytest.approx(107.0)
    summary = with_costs.summary.iloc[0]
    assert summary["mean_turnover"] == pytest.approx(2.0)
    assert summary["mean_transaction_costs"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("is_call", "expected_payoff"),
    [
        (True, np.array([20.0, 0.0])),
        (False, np.array([0.0, 20.0])),
    ],
)
def test_terminal_payoff_for_call_and_put(is_call, expected_payoff):
    result = run_delta_hedge_backtest(
        spot_paths=np.array([[100.0, 120.0], [100.0, 80.0]]),
        time_grid=np.array([0.0, 1.0]),
        policy_deltas={"zero": np.zeros((2, 2))},
        premium=0.0,
        strike=100.0,
        is_call=is_call,
    )

    expiry = result.ledger[result.ledger["event"] == "expiry"].sort_values("path_id")
    assert np.allclose(expiry["option_payoff"], expected_payoff)
    assert np.allclose(expiry["pnl"], -expected_payoff)


def test_summary_reports_distribution_and_left_tail_cvar():
    terminal_spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    spots = np.column_stack([np.full(terminal_spots.size, 100.0), terminal_spots])
    result = run_delta_hedge_backtest(
        spot_paths=spots,
        time_grid=np.array([0.0, 1.0]),
        policy_deltas={"unhedged": np.zeros_like(spots)},
        premium=0.0,
        strike=100.0,
        is_call=True,
    )

    summary = result.summary.iloc[0]
    pnl = np.array([0.0, 0.0, 0.0, -10.0, -20.0])
    assert summary["n_paths"] == 5
    assert summary["mean"] == pytest.approx(np.mean(pnl))
    assert summary["std"] == pytest.approx(np.std(pnl, ddof=1))
    assert summary["rmse"] == pytest.approx(np.sqrt(np.mean(np.square(pnl))))
    assert summary["mae"] == pytest.approx(np.mean(np.abs(pnl)))
    assert summary["q05"] == pytest.approx(np.quantile(pnl, 0.05))
    assert summary["q50"] == pytest.approx(0.0)
    assert summary["q95"] == pytest.approx(0.0)
    assert summary["cvar_05"] == pytest.approx(-20.0)


def test_backtest_rejects_invalid_shapes_values_and_config():
    valid_spots = np.array([[100.0, 101.0, 102.0]])
    valid_times = np.array([0.0, 0.5, 1.0])
    valid_deltas = {"valid": np.zeros_like(valid_spots)}

    with pytest.raises(ValueError, match="same shape"):
        run_delta_hedge_backtest(
            spot_paths=valid_spots,
            time_grid=valid_times,
            policy_deltas={"bad": np.zeros((1, 2))},
            premium=1.0,
            strike=100.0,
            is_call=True,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        run_delta_hedge_backtest(
            spot_paths=valid_spots,
            time_grid=np.array([0.0, 0.5, 0.5]),
            policy_deltas=valid_deltas,
            premium=1.0,
            strike=100.0,
            is_call=True,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        run_delta_hedge_backtest(
            spot_paths=np.array([[100.0, 0.0, 102.0]]),
            time_grid=valid_times,
            policy_deltas=valid_deltas,
            premium=1.0,
            strike=100.0,
            is_call=True,
        )
    with pytest.raises(ValueError, match="finite"):
        run_delta_hedge_backtest(
            spot_paths=valid_spots,
            time_grid=valid_times,
            policy_deltas={"bad": np.array([[0.0, np.nan, 0.0]])},
            premium=1.0,
            strike=100.0,
            is_call=True,
        )
    with pytest.raises(ValueError, match="non-negative"):
        HedgeBacktestConfig(transaction_cost_bps=-1.0)


def test_rolling_realized_vol_and_bs_delta_are_causal():
    times = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    base = np.array([[100.0, 101.0, 103.0, 104.0, 105.0]])
    changed_future = np.array([[100.0, 101.0, 103.0, 160.0, 50.0]])

    base_vol = causal_rolling_realized_vol(
        spot_paths=base,
        time_grid=times,
        window=2,
        min_periods=1,
        initial_vol=0.20,
    )
    changed_vol = causal_rolling_realized_vol(
        spot_paths=changed_future,
        time_grid=times,
        window=2,
        min_periods=1,
        initial_vol=0.20,
    )
    assert base_vol[0, 0] == pytest.approx(0.20)
    assert base_vol[0, 1] == pytest.approx(abs(math.log(101.0 / 100.0)) / 0.5)
    assert np.allclose(base_vol[:, :3], changed_vol[:, :3])
    assert not np.allclose(base_vol[:, 3:], changed_vol[:, 3:])

    base_delta = black_scholes_delta_matrix(
        spot_paths=base,
        time_grid=times,
        strike=102.0,
        volatility=0.20,
        is_call=True,
    )
    changed_delta = black_scholes_delta_matrix(
        spot_paths=changed_future,
        time_grid=times,
        strike=102.0,
        volatility=0.20,
        is_call=True,
    )
    assert base_delta.shape == base.shape
    assert np.allclose(base_delta[:, :3], changed_delta[:, :3])
    assert base_delta[0, -1] == pytest.approx(1.0)
    assert changed_delta[0, -1] == pytest.approx(0.0)


def test_realized_vol_requires_explicit_initial_value_for_finite_first_column():
    realized = causal_rolling_realized_vol(
        spot_paths=np.array([[100.0, 101.0]]),
        time_grid=np.array([0.0, 1.0]),
        window=1,
    )
    assert np.isnan(realized[0, 0])
    assert np.isfinite(realized[0, 1])
