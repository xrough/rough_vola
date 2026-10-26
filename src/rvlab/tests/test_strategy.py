"""Signal-to-position, costs, P&L and strategy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.strategy import (
    apply_costs, break_even_cost, cost_sweep, pnl_series, signal_to_position,
    strategy_metrics, vol_target, walk_forward_strategy,
)


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2024-01-01", periods=120, freq="B"), 20)
    signal = rng.normal(size=len(dates))
    return pd.DataFrame({"date": dates, "signal": signal,
                         "ret": signal * 0.002 + rng.normal(0, 0.01, len(dates))})


class TestSignalToPosition:
    def test_rank_positions_are_bounded_and_centred(self, panel):
        pos = signal_to_position(panel["signal"], dates=panel["date"], method="rank")
        assert pos.between(-1, 1).all()
        assert abs(pos.groupby(panel["date"].to_numpy()).mean()).max() < 0.06

    def test_zscore_is_capped(self, panel):
        extreme = panel["signal"].copy()
        extreme.iloc[0] = 500.0
        pos = signal_to_position(extreme, dates=panel["date"], method="zscore", cap=3.0)
        assert pos.abs().max() <= 3.0

    def test_rank_is_immune_to_an_extreme_score(self, panel):
        """The reason rank is the default for a cross-sectional book."""
        extreme = panel["signal"].copy()
        extreme.iloc[0] = 1e6
        ranked = signal_to_position(extreme, dates=panel["date"], method="rank")
        clean = signal_to_position(panel["signal"], dates=panel["date"], method="rank")
        # only the one displaced observation changes rank materially
        assert (ranked - clean).abs().gt(0.2).sum() <= 2

    def test_gross_target_normalises_exposure_per_date(self, panel):
        pos = signal_to_position(panel["signal"], dates=panel["date"], method="rank",
                                 gross_target=1.0)
        gross = pos.abs().groupby(panel["date"].to_numpy()).sum()
        assert np.allclose(gross, 1.0)

    def test_cross_sectional_differs_from_time_series_standardisation(self, panel):
        """The most common error: standardising across entities *and* time at once."""
        cross = signal_to_position(panel["signal"], dates=panel["date"], method="zscore")
        pooled = signal_to_position(panel["signal"], method="zscore")
        assert not np.allclose(cross, pooled)

    def test_sign_method_is_plus_or_minus_one(self, panel):
        pos = signal_to_position(panel["signal"], dates=panel["date"], method="sign")
        assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}


class TestCosts:
    def test_cost_is_charged_on_the_change_not_the_level(self):
        """Holding is free; trading is not."""
        held = pd.Series([1.0] * 10)
        traded = pd.Series([1.0, -1.0] * 5)
        assert apply_costs(held, cost_bps=10)["cost"].sum() < \
               apply_costs(traded, cost_bps=10)["cost"].sum()

    def test_constant_position_costs_only_the_opening_trade(self):
        costs = apply_costs(pd.Series([1.0] * 5), cost_bps=10)
        assert costs["turnover"].iloc[0] == pytest.approx(1.0)
        assert costs["turnover"].iloc[1:].sum() == pytest.approx(0.0)

    def test_fixed_cost_applies_per_trade(self):
        costs = apply_costs(pd.Series([1.0, -1.0, 1.0]), cost_bps=0.0,
                            fixed_per_trade=0.5)
        assert costs["cost"].sum() == pytest.approx(1.5)


class TestPnl:
    def test_positions_are_lagged_into_the_return(self):
        """A position must be decided before the return it earns."""
        pos = pd.Series([0.0, 1.0, 1.0])
        ret = pd.Series([0.10, 0.10, 0.10])
        out = pnl_series(pos, ret)
        assert out["gross"].iloc[0] == pytest.approx(0.0)      # no position into ret[1]
        assert out["gross"].iloc[-1] == pytest.approx(0.10)

    def test_net_is_gross_minus_cost(self, panel):
        out = pnl_series(panel["signal"], panel["ret"], cost_bps=2.0, dates=panel["date"])
        assert np.allclose(out["net"], out["gross"] - out["cost"])

    def test_panel_pnl_aggregates_to_one_row_per_date(self, panel):
        out = pnl_series(panel["signal"], panel["ret"], dates=panel["date"])
        assert len(out) == panel["date"].nunique()

    def test_zero_cost_leaves_net_equal_to_gross(self, panel):
        out = pnl_series(panel["signal"], panel["ret"], cost_bps=0.0, dates=panel["date"])
        assert np.allclose(out["net"], out["gross"])


class TestVolTarget:
    def test_scaling_uses_only_past_information(self):
        """The scale at t must not depend on the return at t.

        Tested directly rather than by counting warm-up rows: perturb the final
        return and confirm every position is unchanged. If the scale peeked at
        the contemporaneous return, the last position would move.
        """
        rng = np.random.default_rng(0)
        pos = pd.Series(np.ones(400))
        ret = pd.Series(rng.normal(0, 0.01, 400))

        base = vol_target(pos, ret, target_vol=0.10, lookback=60)
        perturbed_ret = ret.copy()
        perturbed_ret.iloc[-1] *= 50.0
        perturbed = vol_target(pos, perturbed_ret, target_vol=0.10, lookback=60)

        assert np.allclose(base, perturbed), "the scale saw the contemporaneous return"

    def test_leverage_is_capped(self):
        rng = np.random.default_rng(1)
        quiet = pd.Series(rng.normal(0, 1e-6, 400))
        scaled = vol_target(pd.Series(np.ones(400)), quiet, target_vol=0.10,
                            max_leverage=3.0)
        assert scaled.abs().max() <= 3.0 + 1e-9

    def test_targeting_stabilises_realised_volatility(self):
        rng = np.random.default_rng(2)
        regime = np.r_[rng.normal(0, 0.004, 500), rng.normal(0, 0.03, 500)]
        ret = pd.Series(regime)
        pos = pd.Series(np.ones(len(ret)))
        scaled = vol_target(pos, ret, target_vol=0.10, lookback=60)
        plain = (pos.shift(1) * ret).dropna()
        targeted = (scaled.shift(1) * ret).dropna()
        ratio = lambda s: s.iloc[600:].std() / s.iloc[100:400].std()
        assert ratio(targeted) < ratio(plain)


class TestMetrics:
    def test_sharpe_matches_its_definition(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.01, 1000))
        m = strategy_metrics(r)
        assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))

    def test_sortino_exceeds_sharpe_when_upside_dominates(self):
        r = pd.Series(np.r_[np.full(80, 0.02), np.full(20, -0.005)])
        m = strategy_metrics(r)
        assert m["sortino"] > m["sharpe"]

    def test_calmar_penalises_a_single_deep_drawdown(self):
        """A series with no drawdown has an undefined Calmar, so both arms need one."""
        rng = np.random.default_rng(3)
        mild = pd.Series(rng.normal(0.001, 0.004, 400))
        shocked = mild.copy(); shocked.iloc[200] = -0.30

        mild_metrics, shocked_metrics = strategy_metrics(mild), strategy_metrics(shocked)
        assert np.isfinite(mild_metrics["calmar"]) and np.isfinite(shocked_metrics["calmar"])
        assert shocked_metrics["calmar"] < mild_metrics["calmar"]
        # Sharpe barely notices the single bad day; Calmar does. That is the point.
        assert shocked_metrics["max_drawdown"] > 10 * mild_metrics["max_drawdown"]

    def test_calmar_is_undefined_without_a_drawdown(self):
        assert np.isnan(strategy_metrics(pd.Series(np.full(300, 0.001)))["calmar"])

    def test_max_drawdown_is_positive_and_matches_the_path(self):
        r = pd.Series([0.1, -0.3, 0.05])
        assert strategy_metrics(r)["max_drawdown"] == pytest.approx(0.30)

    def test_metrics_on_too_short_a_series_returns_empty(self):
        assert strategy_metrics(pd.Series([0.1])).empty


class TestCostSweepAndBreakEven:
    def test_sharpe_falls_monotonically_with_cost(self, panel):
        pos = signal_to_position(panel["signal"], dates=panel["date"], method="rank")
        sweep = cost_sweep(pos, panel["ret"], dates=panel["date"],
                           cost_levels=(0, 1, 5, 20, 100))
        assert sweep["sharpe"].is_monotonic_decreasing

    def test_break_even_is_finite_when_the_strategy_eventually_loses(self, panel):
        pos = signal_to_position(panel["signal"], dates=panel["date"], method="rank")
        sweep = cost_sweep(pos, panel["ret"], dates=panel["date"],
                           cost_levels=(0, 1, 5, 20, 100, 500))
        be = break_even_cost(sweep)
        assert 0 < be < 500

    def test_break_even_is_zero_when_never_profitable(self):
        sweep = pd.DataFrame({"cost_bps": [0, 1, 2],
                              "annualised_return": [-0.1, -0.2, -0.3]})
        assert break_even_cost(sweep) == 0.0


class TestWalkForward:
    def test_each_fold_trades_only_its_own_test_block(self, panel):
        def signal_fn(train, test):
            return np.sign(test["signal"].to_numpy())

        out = walk_forward_strategy(panel, signal_fn, "ret", date_col="date",
                                    train_periods=40, test_periods=20)
        assert not out.empty
        assert out["fold_start"].nunique() >= 2
        assert len(out) == out.index.nunique()
