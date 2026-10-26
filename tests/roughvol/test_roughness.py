from __future__ import annotations

import numpy as np
import pandas as pd

from roughvol.analytics.roughness import (
    deseasonalize_intraday_returns,
    estimate_hurst_exponent,
    estimate_hurst_exponent_grouped,
    log_returns_from_close,
    realized_volatility_proxy,
    realized_variance_blocks,
    simulate_lognormal_vol_paths,
)
from roughvol.sim.volterra import simulate_exact
from roughvol.types import make_rng


def test_realized_volatility_proxy_is_zero_for_constant_returns():
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.full(80, 0.01))))
    realized = realized_volatility_proxy(prices, window=10)

    assert len(realized) == 70
    assert np.allclose(realized.values, 0.0)


def test_realized_volatility_proxy_excludes_overnight_jumps_for_intraday_data():
    index = pd.DatetimeIndex(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
            "2024-01-03 09:32:00",
        ]
    )
    prices = pd.Series([100.0, 101.0, 102.0, 200.0, 202.0, 204.0], index=index)

    realized = realized_volatility_proxy(prices, window=2, annualization=1.0, session_aware=True)

    assert len(realized) == 2
    assert np.all(realized.values < 0.01)


def test_log_returns_from_close_excludes_overnight_jump_when_session_aware():
    index = pd.DatetimeIndex(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
        ]
    )
    prices = pd.Series([100.0, 101.0, 300.0, 303.0], index=index)

    returns = log_returns_from_close(prices, session_aware=True)

    assert len(returns) == 2
    assert np.all(np.abs(returns.values) < 0.03)


def test_realized_variance_blocks_are_non_overlapping_and_session_aware():
    index = pd.DatetimeIndex(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-02 09:32:00",
            "2024-01-02 09:33:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
            "2024-01-03 09:32:00",
            "2024-01-03 09:33:00",
        ]
    )
    log_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.01, 0.04, -0.02])
    prices = pd.Series(100.0 * np.exp(np.cumsum(log_returns)), index=index)

    rv_blocks = realized_variance_blocks(
        prices,
        block_size=2,
        annualization=1.0,
        session_aware=True,
        deseasonalize_intraday=False,
    )

    expected_raw_rv = np.array([(-0.02) ** 2 + 0.03**2, (-0.01) ** 2 + 0.04**2])
    assert len(rv_blocks) == 2
    assert np.allclose(rv_blocks["raw_realized_variance"].values, expected_raw_rv)
    assert np.all(rv_blocks["n_returns"].values == 2)


def test_deseasonalized_intraday_returns_remove_repeatable_time_of_day_scale():
    index = pd.DatetimeIndex(
        [
            "2024-01-02 09:30:00",
            "2024-01-02 09:31:00",
            "2024-01-03 09:30:00",
            "2024-01-03 09:31:00",
            "2024-01-04 09:30:00",
            "2024-01-04 09:31:00",
        ]
    )
    returns = pd.Series([0.02, 0.01, -0.02, -0.01, 0.04, 0.02], index=index)

    normalized = deseasonalize_intraday_returns(returns)

    std_by_bucket = normalized.groupby(normalized.index.time).std(ddof=0)
    assert np.allclose(std_by_bucket.values, std_by_bucket.values[0])
    assert np.isclose(normalized.std(ddof=0), returns.std(ddof=0))


def test_grouped_hurst_estimator_excludes_cross_session_level_jumps():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    sessions = np.array(["day-1"] * 5 + ["day-2"] * 5)

    grouped = estimate_hurst_exponent_grouped(
        values,
        sessions,
        min_lag=1,
        max_lag=3,
    )
    compatibility_path = estimate_hurst_exponent(
        values,
        groups=sessions,
        min_lag=1,
        max_lag=3,
    )
    ungrouped = estimate_hurst_exponent(values, min_lag=1, max_lag=3)

    assert np.allclose(grouped.structure_function, [1.0, 2.0, 3.0])
    assert np.isclose(grouped.hurst, 1.0)
    assert np.allclose(
        compatibility_path.structure_function,
        grouped.structure_function,
    )
    assert not np.allclose(ungrouped.structure_function, grouped.structure_function)


def test_hurst_estimator_is_near_half_for_brownian_motion():
    rng = np.random.default_rng(123)
    increments = rng.normal(size=4096)
    series = np.concatenate([[0.0], np.cumsum(increments)])

    estimate = estimate_hurst_exponent(series, min_lag=1, max_lag=64)

    assert 0.35 <= estimate.hurst <= 0.65
    assert estimate.r_squared > 0.95


def test_hurst_estimator_detects_rough_volterra_driver():
    t = np.linspace(0.0, 1.0, 513)
    rough_driver, _ = simulate_exact(
        t=t,
        hurst=0.12,
        n_paths=1,
        antithetic=False,
        rng=make_rng(7),
    )

    estimate = estimate_hurst_exponent(rough_driver[0], min_lag=1, max_lag=32)

    assert estimate.hurst < 0.3


def test_simulated_lognormal_vol_paths_stay_positive():
    t, rough_vol, brownian_vol = simulate_lognormal_vol_paths(
        hurst=0.15,
        n_steps=128,
        horizon=1.0,
        initial_vol=0.2,
        vol_of_vol=1.1,
        seed=5,
    )

    assert t.shape == rough_vol.shape == brownian_vol.shape
    assert np.isclose(rough_vol[0], 0.2)
    assert np.isclose(brownian_vol[0], 0.2)
    assert np.all(rough_vol > 0.0)
    assert np.all(brownian_vol > 0.0)
