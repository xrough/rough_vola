"""A bounded generic American rough-Heston coverage benchmark.

This module deliberately uses a newly trained, generic Fourier-premium rule.
It does not generalize or load the historical GH16/shared193 case study.
The full exercise decision (including its original-grid European function)
is frozen before valuation.  Control expectations use the actual valuation
transition law, even when its physical time grid is refined.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig, _AffineSurrogate
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_stopping import RoughAmericanConfig, _Continuation
from rvlab.pricing.rough_stopping_fourier import (
    FourierStoppingPolicy, _affine_values, fourier_policy_fingerprint,
    train_fourier_stopping_policy, value_fourier_stopping_policy,
)
from rvlab.pricing.types import RoughHestonParams
from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket


VERSION = "rough_generic_benchmark_v1"
RELATIVE_SPOTS = (-.005, -.0025, 0., .0025, .005)
ESTIMATORS = ("raw", "european", "european_premium")
TOLERANCES = {"price_atol": .001, "price_rtol": 1e-4,
    "delta_atol": .002, "delta_rtol": .005, "confidence_multiplier": 1.96,
    "spread_fraction": .1, "tolerances_changed": False}
PARAMETERS = {"v0": .04, "theta": .045, "kappa": 1.5,
    "vol_of_vol": .3, "rho": -.6, "hurst": .15}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def benchmark_cases():
    """Six predeclared puts; no market selection or dividend screen."""
    return [{"case_id": f"put_{days}d_k{strike}", "strike": float(strike),
        "days": days, "maturity": days/365, "is_call": False,
        "moneyness": 100/strike, "spot": 100., "rate": .04,
        "dividend_assumption": "explicitly no cash dividends over horizon"}
        for days in (30, 90) for strike in (90, 100, 110)]


def benchmark_trials(*, include_joint=True):
    if not isinstance(include_joint, bool):
        raise TypeError("include_joint must be boolean")
    trials = [{"trial_id": case["case_id"], "case": case,
        "policy_id": case["case_id"], "factors": 4, "steps": 128,
        "exercise_steps": 64, "stage": "base", "reference_trial": None,
        "comparison_scope": "generic independent-policy coverage"}
        for case in benchmark_cases()]
    case = next(row for row in benchmark_cases() if row["case_id"] == "put_90d_k100")
    for name, factors, steps, exercise, scope in (
        ("physical", 4, 256, 64, "fixed full policy; physical transition refinement"),
        ("exercise", 4, 256, 128, "joint physical/calendar/trained-policy construction sensitivity"),
        ("kernel", 8, 128, 64, "joint kernel/trained-policy construction sensitivity"),
        ("joint", 8, 256, 128, "joint kernel/physical/calendar/trained-policy construction sensitivity")):
        if name == "joint" and not include_joint:
            continue
        key = "put_90d_k100_"+name
        trials.append({"trial_id": key, "case": dict(case),
            "policy_id": case["case_id"] if name == "physical" else key,
            "factors": factors, "steps": steps, "exercise_steps": exercise,
            "stage": name, "reference_trial": case["case_id"], "comparison_scope": scope})
    return trials


def benchmark_config(*, training_seed, valuation_seed, training_paths=2048,
                     valuation_paths=2048, steps=128, exercise_steps=64):
    return RoughAmericanConfig(training_paths=training_paths,
        valuation_paths=valuation_paths, steps=steps, exercise_steps=exercise_steps,
        training_seed=training_seed, valuation_seed=valuation_seed, batch_size=512,
        memory_budget_mb=1024., ridge=1e-5, bump_relative=.005,
        retrain_bumps=False, stock_integral="exact_drift",
        training_method="fourier_premium", bounded_surrogate=True,
        surrogate_interpolation="value_bridge", premium_cap=.7,
        premium_control=RoughEuropeanControlConfig(24, 100., 512),
        dual_config=None, dual_proxy="regression")


def train_benchmark_policy(trial, kernel, config):
    """Train once on the declared independent stream; no candidate selection."""
    case = trial["case"]
    if (len(kernel.nodes) != trial["factors"] or config.steps != trial["steps"]
            or config.exercise_steps != trial["exercise_steps"]):
        raise ValueError("training kernel/grid must match its declared trial")
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    return train_fourier_stopping_policy(RoughHestonParams(**PARAMETERS), market,
        kernel, strike=case["strike"], maturity=case["maturity"],
        is_call=case["is_call"], config=config, control_config=config.premium_control,
        premium_cap=.7, exercise_style="american")


def policy_operator_grid(policy):
    """The full physical training grid, not merely the exercise calendar."""
    from rvlab.pricing.rough_simulation import _time_grid
    return _time_grid(policy.params, policy.market, policy.continuation_times,
        policy.config.steps)


def decision_european(policy, trace, *, batch_size=512):
    """Evaluate ORIGINAL-grid European decision function at all saved states."""
    _integer(batch_size, "batch_size", 1)
    stocks, factors = np.asarray(trace["stocks"]), np.asarray(trace["factors"])
    times = np.asarray(trace["times"])
    if stocks.ndim != 3 or factors.shape[:2] != stocks.shape[:2]:
        raise ValueError("decision trace needs aligned path/date/stock and factor arrays")
    proxy = _AffineSurrogate(policy.params, policy.market, policy.kernel,
        np.array([policy.strike]), np.array([policy.maturity]),
        np.array([policy.is_call]), policy_operator_grid(policy), policy.control_config)
    values = np.empty_like(stocks)
    for j, time in enumerate(times):
        for begin in range(0, len(stocks), batch_size):
            end = min(begin+batch_size, len(stocks))
            values[begin:end, j] = _affine_values(proxy, float(time),
                factors[begin:end, j, None], np.log(stocks[begin:end, j])[:, None, None],
                np.zeros((end-begin, 1, 1, stocks.shape[-1])))[:, 0, 0]
    if not np.isfinite(values).all():
        raise ArithmeticError("frozen decision European values are nonfinite")
    return values


def replay_stopping_rule(policy, trace, euro):
    """Own nonanticipating stopping date at each spot, including inception."""
    stocks, factors = np.asarray(trace["stocks"]), np.asarray(trace["factors"])
    times = np.asarray(trace["times"])
    euro = np.asarray(euro)
    if euro.shape != stocks.shape or stocks.shape[1] != len(policy.continuations):
        raise ValueError("frozen policy calendar and European arrays must align")
    n, dates, bumps = stocks.shape
    stops = np.full((n, bumps), dates-1, dtype=np.int64)
    alive = np.ones((n, bumps), bool)
    continuation = np.empty_like(stocks)
    decisions = np.zeros_like(stocks, dtype=bool)
    for j, time in enumerate(times):
        payoff = np.maximum((1 if policy.is_call else -1)*(stocks[:, j]-policy.strike), 0.)
        q = np.clip(policy.continuations[j].value(stocks[:, j], factors[:, j],
            policy.strike), 0., policy.premium_cap)
        future = euro[:, j]/policy.market.discount_curve.discount(time)+q
        continuation[:, j] = future
        if j == 0:
            exercise = np.full((n, bumps), policy.exercise_at_inception, dtype=bool)
        elif j == dates-1:
            exercise = np.ones((n, bumps), bool)
        else:
            exercise = (payoff > 0.) & (payoff >= future)
        decisions[:, j] = exercise
        stop = alive & exercise
        stops = np.where(stop, j, stops)
        alive &= ~stop
    if np.any(alive):
        raise ArithmeticError("frozen stopping replay left live contracts")
    return stops, continuation, decisions


def estimator_endpoints(payoff, european, premium, stops, eligible):
    """Same-policy raw/EU/EU+premium lower and dual upper arrays."""
    payoff, european, premium = map(np.asarray, (payoff, european, premium))
    stops, eligible = np.asarray(stops), np.asarray(eligible)
    if (payoff.ndim != 3 or european.shape != payoff.shape or premium.shape != payoff.shape
            or stops.shape != (payoff.shape[0], payoff.shape[2])
            or stops.dtype.kind not in "iu" or eligible.shape != (payoff.shape[1],)
            or eligible.dtype.kind != "b" or not eligible.any()
            or np.any(stops < 0) or np.any(stops >= payoff.shape[1])
            or not np.all(eligible[stops])):
        raise ValueError("invalid aligned payoff/martingale/stopping arrays")
    if not all(np.isfinite(a).all() for a in (payoff, european, premium)):
        raise ValueError("payoff and martingales must be finite")
    path, bump = np.arange(len(payoff))[:, None], np.arange(payoff.shape[-1])[None, :]
    result = {}
    for name, martingale in (("raw", np.zeros_like(european)),
            ("european", european), ("european_premium", european+premium)):
        adjusted = payoff-martingale
        lower = adjusted[path, stops, bump]
        upper = np.max(adjusted[:, eligible], axis=1)
        if np.any(upper-lower < -2e-13):
            raise ArithmeticError("same-policy lower exceeded the sampled dual maximum")
        result[name+"_lower"] = lower
        result[name+"_upper"] = upper
    return result


def value_benchmark_policy(policy, config):
    """Use generic physical engine; separate fixed decisions from its controls."""
    if (config.training_seed != policy.config.training_seed
            or config.exercise_steps != policy.config.exercise_steps
            or config.premium_control != policy.config.premium_control
            or config.premium_cap != policy.config.premium_cap):
        raise ValueError("valuation must preserve the sealed policy and control specification")
    spots = policy.market.spot*(1+np.asarray(RELATIVE_SPOTS))
    valued = value_fourier_stopping_policy(policy, config=config, spots=spots, trace=True)
    trace = valued["trace"]
    euro = decision_european(policy, trace, batch_size=config.batch_size)
    stops, continuation, decisions = replay_stopping_rule(policy, trace, euro)
    arrays = {name: np.asarray(trace[name]).copy() for name in (
        "times", "eligible", "discounts", "discounted_payoffs", "stocks", "factors",
        "european_martingale", "premium_martingale", "european_proxy")}
    arrays.update({"decision_european": euro, "decision_continuation": continuation,
        "decision_rule": decisions, "stop_indices": stops,
        "initial_spots": spots, "relative_spots": np.asarray(RELATIVE_SPOTS),
        "actual_time_grid": np.asarray(trace["actual_time_grid"]),
        "operator_time_grid": policy_operator_grid(policy),
        "public_stop_indices": np.asarray(trace["stop_indices"]).copy()})
    arrays.update(estimator_endpoints(arrays["discounted_payoffs"],
        arrays["european_martingale"], arrays["premium_martingale"], stops,
        arrays["eligible"]))
    same_law = np.array_equal(arrays["actual_time_grid"], arrays["operator_time_grid"])
    if same_law and (not np.array_equal(stops, arrays["public_stop_indices"])
            or not np.allclose(arrays["raw_lower"], valued["raw_samples"], rtol=0, atol=2e-13)
            or not np.allclose(arrays["european_premium_lower"], valued["primal_samples"], rtol=0, atol=2e-13)):
        raise ArithmeticError("base fixed-decision replay differs from public backend")
    return arrays, {"valuation": dict(valued["diagnostics"]),
        "same_physical_and_operator_grid": same_law,
        "base_public_replay_checked": same_law,
        "frozen_full_decision_function": True,
        "new_fitting_or_selection": False,
        "american_pricing_ready": False, "optimal_american_delta_ready": False}


def save_policy(policy, folder):
    """Explicit JSON/NPZ model seal; no pickle or valuation-dependent fitting."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {"exercise_times": policy.exercise_times,
        "continuation_times": policy.continuation_times}
    declaration = []
    for group in (policy.continuations, policy.value_premiums):
        records = []
        for index, model in enumerate(group):
            if not isinstance(model, _Continuation):
                raise TypeError("generic benchmark seal supports polynomial continuations only")
            prefix = f"g{len(declaration)}_{index}_"
            record = {"knots": list(model.knots), "rank": model.rank,
                "condition": float(model.condition)}
            for name in ("center", "scale", "beta"):
                key = prefix+name
                arrays[key] = getattr(model, name)
                record[name] = key
            records.append(record)
        declaration.append(records)
    np.savez_compressed(folder/"coefficients.npz", **arrays)
    payload = {"version": VERSION, "params": asdict(policy.params),
        "market": asdict(policy.market), "kernel": asdict(policy.kernel),
        "strike": policy.strike, "maturity": policy.maturity,
        "is_call": policy.is_call, "exercise_style": policy.exercise_style,
        "config": asdict(policy.config), "control_config": asdict(policy.control_config),
        "exercise_at_inception": policy.exercise_at_inception,
        "diagnostics": dict(policy.diagnostics), "premium_cap": policy.premium_cap,
        "models": declaration, "policy_fingerprint": fourier_policy_fingerprint(policy),
        "coefficient_sha256": sha256(folder/"coefficients.npz")}
    (folder/"policy.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    return payload


def load_policy(folder):
    from rvlab.pricing.vanilla_types import CashDividend
    folder = Path(folder)
    payload = json.loads((folder/"policy.json").read_text())
    if payload["version"] != VERSION or sha256(folder/"coefficients.npz") != payload["coefficient_sha256"]:
        raise ValueError("generic benchmark policy seal changed")
    config = dict(payload["config"])
    config["stock_knots"] = tuple(config["stock_knots"])
    config["premium_control"] = RoughEuropeanControlConfig(**config["premium_control"])
    config = RoughAmericanConfig(**config)
    market = payload["market"]
    market = SpotMarket(market["spot"], DiscountCurve(**market["discount_curve"]),
        DividendSchedule(tuple(CashDividend(**row) for row in market["dividends"]["events"]),
            market["dividends"]["coverage"]))
    with np.load(folder/"coefficients.npz", allow_pickle=False) as arrays:
        groups = [tuple(_Continuation(arrays[row["center"]], arrays[row["scale"]],
            arrays[row["beta"]], tuple(row["knots"]), row["rank"], row["condition"])
            for row in group) for group in payload["models"]]
        policy = FourierStoppingPolicy(RoughHestonParams(**payload["params"]), market,
            ExponentialKernel(**payload["kernel"]), payload["strike"], payload["maturity"],
            payload["is_call"], payload["exercise_style"], arrays["exercise_times"],
            arrays["continuation_times"], groups[0], groups[1], config,
            RoughEuropeanControlConfig(**payload["control_config"]),
            payload["exercise_at_inception"], payload["diagnostics"], payload["premium_cap"])
    if fourier_policy_fingerprint(policy) != payload["policy_fingerprint"]:
        raise ValueError("loaded generic policy differs from sealed coefficients")
    return policy


def benchmark_study(context):
    """Read the independently audited sealed panel; never run notebook MC."""
    from ._pricing_rough_benchmark_collection import benchmark_study as collect
    return collect(context)


__all__ = ["VERSION", "RELATIVE_SPOTS", "ESTIMATORS", "TOLERANCES", "PARAMETERS",
    "benchmark_cases", "benchmark_trials", "benchmark_config", "train_benchmark_policy",
    "value_benchmark_policy", "decision_european", "replay_stopping_rule",
    "estimator_endpoints", "policy_operator_grid", "save_policy", "load_policy",
    "sha256", "canonical", "benchmark_study"]
