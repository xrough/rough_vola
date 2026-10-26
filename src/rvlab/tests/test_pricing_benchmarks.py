"""Input attribution benchmarks preserve contracts and fit training prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_benchmarks as bench
from rvlab.pricing import BlackScholesParams, HestonParams, PricingConfig, PricingError, price


@pytest.fixture
def surface():
    maturity = np.repeat([.05, .1], 6)
    strike = np.tile([94., 97., 99., 101., 103., 106.], 2)
    forward = 100. * np.exp(.02 * maturity)
    discount = np.exp(-.03 * maturity)
    calls = strike > forward
    log_k = np.log(strike / forward)
    variance = .04 * maturity - .01 * maturity * log_k + .08 * maturity * log_k**2
    observed = bench._black_total_variance(forward, discount, strike, variance, calls)
    return pd.DataFrame({"contract_id": [f"contract-{i}" for i in range(12)],
        "expiry_date": np.repeat(["expiry-a", "expiry-b"], 6),
        "forward": forward, "discount": discount, "strike": strike,
        "maturity": maturity, "is_call": calls,
        "bid": observed - .01, "ask": observed + .01,
        "split": np.where(np.isin(strike, [97., 103.]), "heldout", "train")})


def flat_surface(surface):
    out = surface.copy()
    mid = price(BlackScholesParams(.2), **{name: out[name].to_numpy() for name in
        ("forward", "discount", "strike", "maturity", "is_call")})
    out["bid"], out["ask"] = mid - .01, mid + .01
    return out


def anchors_for(quotes, shifts=(-.1, 0., .1)):
    rows = []
    for expiry, q in quotes.groupby("expiry_date"):
        f, d = q.forward.iloc[0], q.discount.iloc[0]
        for i, shift in enumerate(shifts):
            k = f + i - 1
            # Nonnegative paired prices with exact discounted parity.
            diff = d * (f + shift - k)
            rows.append({"expiry_date": expiry, "strike": k,
                "forward": f + shift, "market_price_call": 2. + .5 * diff,
                "market_price_put": 2. - .5 * diff})
    return pd.DataFrame(rows)


def test_european_benchmarks_reject_explicit_american_observations(surface):
    american = surface.assign(exercise_style="american")
    with pytest.raises(ValueError, match="American observations.*row positions"):
        bench.quadratic_smile_benchmark(american)
    with pytest.raises(ValueError, match="American observations.*row positions"):
        bench.forward_sensitivity(BlackScholesParams(.2), american,
            anchors_for(american), PricingConfig(32, 60., 32))


def test_quadratic_recovers_prices_and_keeps_input_order(surface):
    q = surface.iloc[[5, 1, 8, 0, 11, 2, 6, 4, 10, 3, 7, 9]].copy()
    q.index = [3, 3, 8, 2, 0, 4, 4, 9, 1, 6, 2, 7]
    original = q.copy(deep=True)
    result = bench.quadratic_smile_benchmark(q)
    pd.testing.assert_frame_equal(q, original)
    summary = result["summary"].iloc[0]
    assert summary.status == "converged"
    assert summary.train_rmse < 1e-8
    assert summary.test_rmse < 1e-8
    np.testing.assert_array_equal(result["predictions"].index, q.index)
    np.testing.assert_array_equal(result["predictions"].contract_id, q.contract_id)
    assert (result["shape_checks"].min_total_variance > 0).all()
    assert result["shape_checks"].passes_local_checks.all()
    assert set(result["shape_checks"].grid) == {"observed_strikes", "dense_grid"}
    json.dumps(result["metadata"], allow_nan=False)


def test_quadratic_never_uses_heldout_prices_to_choose_parameters(surface):
    first = bench.quadratic_smile_benchmark(surface)
    changed = surface.copy()
    heldout = changed.split == "heldout"
    changed.loc[heldout, ["bid", "ask"]] += .4
    second = bench.quadratic_smile_benchmark(changed)
    for column in ["a", "b", "c", "selected_start", "training_objective"]:
        np.testing.assert_array_equal(first["parameters"][column], second["parameters"][column])
    np.testing.assert_array_equal(first["predictions"].model_price, second["predictions"].model_price)
    assert second["summary"].iloc[0].test_rmse > .3


def test_quadratic_exhaustion_remains_a_result(surface):
    surface = surface.copy()
    surface.loc[surface.split == "train", "ask"] += np.tile([.03, .07, .04, .02], 2)
    result = bench.quadratic_smile_benchmark(surface, max_nfev=1)
    assert result["summary"].iloc[0].status == "exhausted"
    assert (result["attempts"].status == "exhausted").all()
    assert result["predictions"].model_price.notna().all()
    assert not result["metadata"]["dynamic_model"]
    assert not result["metadata"]["global_arbitrage_free_guarantee"]


def test_quadratic_insufficient_training_is_explicit(surface):
    surface = surface.copy()
    surface.loc[surface.strike.isin([94., 106.]), "split"] = "heldout"
    result = bench.quadratic_smile_benchmark(surface)
    assert result["summary"].iloc[0].status == "incomplete"
    assert result["summary"].iloc[0].train_rmse is None
    assert (result["parameters"].status == "insufficient_training").all()
    assert result["predictions"].model_price.isna().all()


def test_shape_checks_can_reject_a_positive_but_nonconvex_smile():
    rows = bench._shape_checks(np.array([.001, .05, .001]), -.05, .05,
        100., .99, 100. * np.exp(np.linspace(-.05, .05, 6)), "expiry", 201)
    assert all(row["min_total_variance"] > 0 for row in rows)
    assert not rows[1]["passes_local_checks"]
    assert rows[1]["min_normalized_call_convexity"] < 0


def test_quadratic_numerical_failures_are_preserved(monkeypatch, surface):
    def fail(*args, **kwargs):
        raise PricingError("controlled benchmark failure")
    monkeypatch.setattr(bench, "_black_total_variance", fail)
    result = bench.quadratic_smile_benchmark(surface)
    assert result["summary"].iloc[0].status == "incomplete"
    assert (result["attempts"].status == "numerical_failure").all()
    assert result["predictions"].model_price.isna().all()
    assert result["attempts"].error.str.contains("controlled benchmark failure").all()


def test_forward_stress_preserves_training_selection_and_contract_order(surface):
    q = flat_surface(surface).iloc[[5, 1, 8, 0, 11, 2, 6, 4, 10, 3, 7, 9]].copy()
    q.index = [3, 3, 8, 2, 0, 4, 4, 9, 1, 6, 2, 7]
    a = anchors_for(q)
    first = bench.forward_sensitivity(BlackScholesParams(.2), q, a, PricingConfig(24, 40., 32))
    changed = q.copy()
    changed.loc[changed.split == "heldout", ["bid", "ask"]] += .5
    second = bench.forward_sensitivity(BlackScholesParams(.2), changed, a, PricingConfig(24, 40., 32))
    np.testing.assert_array_equal(first["shifts"].forward_shift, second["shifts"].forward_shift)
    assert len(first["summary"]) == 7
    for _, rows in first["predictions"].groupby("scenario", sort=False):
        np.testing.assert_array_equal(rows.index, q.index)
        np.testing.assert_array_equal(rows.contract_id, q.contract_id)
    json.dumps(first["metadata"], allow_nan=False)
    json.dumps(first["summary_records"], allow_nan=False)


def test_rate_stress_reconstructs_discount_and_parity_forward(surface):
    q = flat_surface(surface)
    result = bench.forward_sensitivity(BlackScholesParams(.2), q, anchors_for(q), PricingConfig(24, 40., 32))
    prices = result["predictions"].query("scenario == 'rate_plus_100bp'")
    np.testing.assert_allclose(prices.discount, q.discount * np.exp(-.01 * q.maturity))
    for expiry, rows in prices.groupby("expiry_date"):
        original = q.loc[q.expiry_date == expiry]
        a = anchors_for(q).query("expiry_date == @expiry")
        expected = np.median(a.strike + (a.market_price_call - a.market_price_put) / rows.discount.iloc[0])
        np.testing.assert_allclose(rows.forward, expected)
        assert len(rows) == len(original)
    assert not result["metadata"]["anchor_ranges_are_confidence_intervals"]
    assert not result["metadata"]["independent_stock_observation"]


def test_cf_cache_is_shared_across_forward_optimizer_trials(monkeypatch, surface):
    q = flat_surface(surface)
    seen = []
    original = bench.heston_cf
    def observe(params, u, tenor):
        seen.append(tenor)
        return original(params, u, tenor)
    monkeypatch.setattr(bench, "heston_cf", observe)
    result = bench.forward_sensitivity(HestonParams(.04, .04, 1., .1, -.3), q,
        anchors_for(q), PricingConfig(48, 80., 32))
    assert len(seen) == 2
    assert result["metadata"]["characteristic_function_evaluations"] == 2
    assert result["metadata"]["pricing_evaluations"] > len(seen)


def test_degenerate_anchor_interval_fixes_the_shift(surface):
    q = flat_surface(surface)
    result = bench.forward_sensitivity(BlackScholesParams(.2), q,
        anchors_for(q, shifts=(0.,)), PricingConfig(24, 40., 32))
    shifts = result["shifts"].query("scenario == 'forward_fit_anchor_box'")
    assert (shifts.active_bound == "fixed").all()
    assert (shifts.forward_shift == 0.).all()
    summary = result["summary"].query("scenario == 'forward_fit_anchor_box'").iloc[0]
    assert summary.optimizer_success and summary.nfev == 0


def test_forward_numerical_failure_has_missing_prices_and_json_status(monkeypatch, surface):
    def fail(*args, **kwargs):
        raise PricingError("controlled CF failure")
    monkeypatch.setattr(bench, "heston_cf", fail)
    q = flat_surface(surface)
    result = bench.forward_sensitivity(HestonParams(.04, .04, 1., .1, -.3), q,
        anchors_for(q), PricingConfig(24, 40., 32))
    assert (result["summary"].status == "numerical_failure").all()
    assert result["predictions"].model_price.isna().all()
    assert result["summary"].train_rmse.isna().all()
    assert result["summary"].error.str.contains("controlled CF failure").all()


@pytest.mark.parametrize("kwargs", [{"max_nfev": 0}, {"max_nfev": True}, {"grid_size": 2}])
def test_invalid_quadratic_controls(surface, kwargs):
    with pytest.raises(ValueError):
        bench.quadratic_smile_benchmark(surface, **kwargs)


def test_duplicate_contracts_and_mixed_expiry_inputs_rejected(surface):
    q = surface.copy()
    q.loc[1, "contract_id"] = q.loc[0, "contract_id"]
    with pytest.raises(ValueError, match="unique"):
        bench.quadratic_smile_benchmark(q)
    q = surface.copy()
    q.loc[0, "discount"] = .9
    with pytest.raises(ValueError, match="constant within"):
        bench.quadratic_smile_benchmark(q)
