"""Three research narratives, with computation separated by research question.

The archived demonstrations remain reproducible, but these workflows never
load their results. Pricing galleries do not calibrate; calibration does not
rerun the gallery; the SPY input gate is evaluated exactly once per study.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from numbers import Integral
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.special import gamma

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DividendSchedule, DiscountCurve, FDConfig,
    HestonCurveParams, HestonParams, MeanReversionCurve, OptimizerConfig,
    PricingConfig, PricingError, RoughHestonParams, SpotMarket,
    calibrate_vanilla_surface, evaluate_vanilla, parameter_values, price,
    solve_fractional_riccati,
)
from ._pricing_fits import (
    COARSE_CONFIG, DEMO_BOUNDS, REFERENCE_CONFIG, SYNTHETIC_INITIALS,
    SYNTHETIC_TRUTHS, fit_attempt, refinement_table, synthetic_surface,
)
from ._pricing_workflows import _ivs, _parameter_rows, _summary
from ._pricing_american import (
    AMERICAN_BACKEND, DATES, EXPIRIES, _audit_plot, _raw_price_plot,
    load_american_spy_inputs, select_american_contracts,
)


EVIDENCE_VERSION = "pricing_research_v1"
MIXED_BACKEND = "mixed_european_fourier_american_fd"
MODEL_LABELS = {"bs": "Black–Scholes", "heston": "Heston", "rough_heston": "Rough Heston",
                "flat_bs_on_heston": "Flat BS on Heston"}


def _budget(value, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return min(int(value), maximum)


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("seed must be an integer")
    return int(value)


def _contracts():
    return pd.DataFrame([
        {"contract_id": f"syn-{days}-{strike}", "maturity": days/365,
         "strike": float(strike), "is_call": strike >= 100,
         "split": "heldout" if strike in (97, 103) else "train"}
        for days in (14, 30, 60) for strike in (94, 97, 99, 101, 103, 106)
    ])


def _markets():
    curve = DiscountCurve.flat(.04, 1.)
    return {
        "cash dividend": SpotMarket(100., curve, DividendSchedule((CashDividend(.08, 1.5),), 1.)),
        "no dividend": SpotMarket(100., curve, DividendSchedule((), 1.)),
    }


def _fd_truths():
    return {"Black–Scholes": SYNTHETIC_TRUTHS["bs"], "Heston": SYNTHETIC_TRUTHS["heston"]}


def _settings_table(rows):
    return pd.DataFrame(rows, columns=["section", "item", "value"])


def _failure(rows, model, stage, exc, **kwargs):
    rows.append({"model": model, "stage": stage, "error": str(exc), **kwargs})


def _european_limits(config):
    args = dict(forward=100., discount=.99, strike=np.array([95., 100., 105.]),
                maturity=.1, is_call=np.array([False, True, True]))
    rows = []
    def record(check, value, note, measure="maximum absolute price difference", model="European"):
        rows.append({"model": model, "check": check, "error_measure": measure,
                     "value": float(value), "status": "passed" if value < 1e-9 else "reported", "note": note})
    deterministic = HestonParams(.04, .06, 1.5, 0., -.6)
    integrated = .06*.1 + (.04-.06)*(-np.expm1(-1.5*.1))/1.5
    record("Heston deterministic variance", np.max(abs(price(deterministic, **args)-
        price(BlackScholesParams(np.sqrt(integrated/.1)), **args))), "Exact integrated variance; European terminal payoff only")
    constant = RoughHestonParams(.04, .04, 1., 0., -.6, .15)
    record("Rough Heston constant variance", np.max(abs(price(constant, **args, config=config)-
        price(BlackScholesParams(.2), **args))), "Zero vol-of-vol, v0 = theta")
    rough = replace(SYNTHETIC_TRUTHS["rough_heston"], hurst=.5)
    ordinary = HestonParams(**{k: v for k, v in asdict(rough).items() if k != "hurst"})
    record("Hurst .5 delegates to Heston", np.max(abs(price(rough, **args, config=config)-
        price(ordinary, **args, config=config))), "Dispatch identity, not an independent fractional-solver benchmark")
    u = np.array([.2-.5j, 1.-.5j])
    _, solution = solve_fractional_riccati(RoughHestonParams(.04, .04, 0., 0., -.6, .15), u, .1, n_steps=32)
    exact = -.5*(u*u+1j*u)*.1**.65/gamma(1.65)
    record("Fractional constant forcing", np.max(abs(solution[-1]-exact)),
           "Nonzero initial forcing is included", measure="maximum absolute Riccati error", model="Rough Heston")
    try:
        solve_fractional_riccati(RoughHestonParams(.04, .04, 1., 1e6, -.7, .1), 100-.5j, 10., n_steps=16)
    except PricingError as exc:
        rows.append({"model": "Rough Heston", "check": "Deliberately divergent solver",
            "error_measure": "identifiable failure", "value": np.nan, "status": "expected_failure", "note": str(exc)})
    else:
        rows.append({"model": "Rough Heston", "check": "Deliberately divergent solver",
            "error_measure": "identifiable failure", "value": np.nan, "status": "not_triggered", "note": "No failure occurred; inspect settings"})
    return rows


def pricing_research(context, *, seed=42, fourier_config=None, fd_config=None,
                     run_rough_foundation=False, rough_reuse_checkpoints=True,
                     rough_study_kwargs=None, run_rough_precision=False,
                     rough_precision_kwargs=None, run_rough_american=False,
                     rough_american_kwargs=None, run_rough_american_accuracy=False,
                     rough_american_accuracy_kwargs=None, run_rough_stopping_attribution=False,
                     rough_stopping_attribution_kwargs=None, run_rough_dual_tightening=False,
                     rough_dual_tightening_kwargs=None, run_rough_policy_improvement=False,
                     rough_policy_improvement_kwargs=None, run_rough_control_centering=False,
                     rough_control_centering_kwargs=None, run_rough_centering_precision=False,
                     rough_centering_precision_kwargs=None, run_rough_stopping_delta=False,
                     rough_stopping_delta_kwargs=None, run_rough_policy_validation=False,
                     rough_policy_validation_kwargs=None, run_rough_gap_time_refinement=False,
                     rough_gap_time_refinement_kwargs=None, run_rough_tail_attribution=False,
                     rough_tail_attribution_kwargs=None, run_rough_bellman_policy=False,
                     rough_bellman_policy_kwargs=None, run_rough_bellman_delta=False,
                     rough_bellman_delta_kwargs=None, run_rough_price_budget=False,
                     rough_price_budget_kwargs=None, run_rough_mc_precision=False,
                     rough_mc_precision_kwargs=None, run_rough_shared_delta=False,
                     rough_shared_delta_kwargs=None):
    """Prices, exercise and sensitivities without model calibration.

    Optional stopping attribution optimizes martingale weights on training
    paths; those calls are reported separately from calibration optimizers.
    """
    import matplotlib.pyplot as plt
    for name, value in (("run_rough_foundation", run_rough_foundation),
                        ("run_rough_precision", run_rough_precision),
                        ("run_rough_american", run_rough_american),
                        ("run_rough_american_accuracy", run_rough_american_accuracy),
                        ("run_rough_stopping_attribution", run_rough_stopping_attribution),
                        ("run_rough_dual_tightening", run_rough_dual_tightening),
                        ("run_rough_policy_improvement", run_rough_policy_improvement),
                        ("run_rough_control_centering", run_rough_control_centering),
                        ("run_rough_centering_precision", run_rough_centering_precision),
                        ("run_rough_stopping_delta", run_rough_stopping_delta),
                        ("run_rough_policy_validation", run_rough_policy_validation),
                        ("run_rough_gap_time_refinement", run_rough_gap_time_refinement),
                        ("run_rough_tail_attribution", run_rough_tail_attribution),
                        ("run_rough_bellman_policy", run_rough_bellman_policy),
                        ("run_rough_bellman_delta", run_rough_bellman_delta),
                        ("run_rough_price_budget", run_rough_price_budget),
                        ("run_rough_mc_precision", run_rough_mc_precision),
                        ("run_rough_shared_delta", run_rough_shared_delta),
                        ("rough_reuse_checkpoints", rough_reuse_checkpoints)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    mc_precision_or_later = run_rough_mc_precision or run_rough_shared_delta
    price_budget_or_later = run_rough_price_budget or mc_precision_or_later
    bellman_delta_or_later = run_rough_bellman_delta or price_budget_or_later
    bellman_or_later = run_rough_bellman_policy or bellman_delta_or_later
    tail_or_later = run_rough_tail_attribution or bellman_or_later
    gap_time_or_later = run_rough_gap_time_refinement or tail_or_later
    policy_validation_or_later = run_rough_policy_validation or gap_time_or_later
    stopping_delta_or_later = run_rough_stopping_delta or policy_validation_or_later
    if rough_study_kwargs is not None and not isinstance(rough_study_kwargs, Mapping):
        raise TypeError("rough_study_kwargs must be a mapping or None")
    rough_options = dict(rough_study_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & rough_options.keys():
        raise ValueError("set the rough-study seed and reuse through the named workflow arguments")
    if rough_precision_kwargs is not None and not isinstance(rough_precision_kwargs, Mapping):
        raise TypeError("rough_precision_kwargs must be a mapping or None")
    precision_options = dict(rough_precision_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & precision_options.keys():
        raise ValueError("set the rough-precision seed and reuse through the named workflow arguments")
    if run_rough_precision and run_rough_foundation and rough_options:
        raise ValueError("rough_study_kwargs must be empty when replaying the historical foundation with the precision study")
    if rough_american_kwargs is not None and not isinstance(rough_american_kwargs, Mapping):
        raise TypeError("rough_american_kwargs must be a mapping or None")
    american_options = dict(rough_american_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & american_options.keys():
        raise ValueError("set the rough-American seed and reuse through the named workflow arguments")
    if (run_rough_american or run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later) and (rough_options or precision_options):
        raise ValueError("historical rough-study and precision kwargs must be empty when replayed with American stopping")
    if rough_american_accuracy_kwargs is not None and not isinstance(rough_american_accuracy_kwargs, Mapping):
        raise TypeError("rough_american_accuracy_kwargs must be a mapping or None")
    accuracy_options = dict(rough_american_accuracy_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & accuracy_options.keys():
        raise ValueError("set the rough-American accuracy seed and reuse through the named workflow arguments")
    if (run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later) and american_options:
        raise ValueError("rough_american_kwargs must be empty when replaying the historical stopping baseline")
    if rough_stopping_attribution_kwargs is not None and not isinstance(rough_stopping_attribution_kwargs, Mapping):
        raise TypeError("rough_stopping_attribution_kwargs must be a mapping or None")
    attribution_options = dict(rough_stopping_attribution_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & attribution_options.keys():
        raise ValueError("set stopping-attribution seed and reuse through the named workflow arguments")
    if (run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later) and accuracy_options:
        raise ValueError("rough_american_accuracy_kwargs must be empty when replaying its historical campaign")
    if rough_dual_tightening_kwargs is not None and not isinstance(rough_dual_tightening_kwargs, Mapping):
        raise TypeError("rough_dual_tightening_kwargs must be a mapping or None")
    tightening_options = dict(rough_dual_tightening_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & tightening_options.keys():
        raise ValueError("set dual-tightening seed and reuse through the named workflow arguments")
    if (run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later) and attribution_options:
        raise ValueError("rough_stopping_attribution_kwargs must be empty when replaying its historical campaign")
    if rough_policy_improvement_kwargs is not None and not isinstance(rough_policy_improvement_kwargs, Mapping):
        raise TypeError("rough_policy_improvement_kwargs must be a mapping or None")
    policy_options = dict(rough_policy_improvement_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & policy_options.keys():
        raise ValueError("set policy-study seed and reuse through named workflow arguments")
    if (run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later) and tightening_options:
        raise ValueError("rough_dual_tightening_kwargs must be empty when replaying its historical campaign")
    if rough_control_centering_kwargs is not None and not isinstance(rough_control_centering_kwargs, Mapping):
        raise TypeError("rough_control_centering_kwargs must be a mapping or None")
    centering_options = dict(rough_control_centering_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & centering_options.keys():
        raise ValueError("set centering seed and reuse through named workflow arguments")
    if (run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later) and policy_options:
        raise ValueError("rough_policy_improvement_kwargs must be empty when replaying its historical campaign")
    if rough_centering_precision_kwargs is not None and not isinstance(rough_centering_precision_kwargs, Mapping):
        raise TypeError("rough_centering_precision_kwargs must be a mapping or None")
    centering_precision_options = dict(rough_centering_precision_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & centering_precision_options.keys():
        raise ValueError("set centering-precision seed and reuse through named workflow arguments")
    if (run_rough_centering_precision or stopping_delta_or_later) and centering_options:
        raise ValueError("rough_control_centering_kwargs must be empty when replaying its historical campaign")
    if rough_stopping_delta_kwargs is not None and not isinstance(rough_stopping_delta_kwargs, Mapping):
        raise TypeError("rough_stopping_delta_kwargs must be a mapping or None")
    stopping_delta_options = dict(rough_stopping_delta_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & stopping_delta_options.keys():
        raise ValueError("set stopping/delta seed and reuse through named workflow arguments")
    if stopping_delta_or_later and centering_precision_options:
        raise ValueError("rough_centering_precision_kwargs must be empty when replaying its historical campaign")
    if rough_policy_validation_kwargs is not None and not isinstance(rough_policy_validation_kwargs, Mapping):
        raise TypeError("rough_policy_validation_kwargs must be a mapping or None")
    policy_validation_options = dict(rough_policy_validation_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & policy_validation_options.keys():
        raise ValueError("set policy-validation seed and reuse through named workflow arguments")
    if policy_validation_or_later and stopping_delta_options:
        raise ValueError("rough_stopping_delta_kwargs must be empty when replaying its historical campaign")
    if rough_gap_time_refinement_kwargs is not None and not isinstance(rough_gap_time_refinement_kwargs, Mapping):
        raise TypeError("rough_gap_time_refinement_kwargs must be a mapping or None")
    gap_time_options = dict(rough_gap_time_refinement_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & gap_time_options.keys():
        raise ValueError("set gap/time-refinement seed and reuse through named workflow arguments")
    if gap_time_or_later and policy_validation_options:
        raise ValueError("rough_policy_validation_kwargs must be empty when replaying its historical campaign")
    if rough_tail_attribution_kwargs is not None and not isinstance(rough_tail_attribution_kwargs, Mapping):
        raise TypeError("rough_tail_attribution_kwargs must be a mapping or None")
    tail_options = dict(rough_tail_attribution_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & tail_options.keys():
        raise ValueError("set tail-attribution seed and reuse through named workflow arguments")
    if tail_or_later and gap_time_options:
        raise ValueError("rough_gap_time_refinement_kwargs must be empty when replaying its historical campaign")
    if rough_bellman_policy_kwargs is not None and not isinstance(rough_bellman_policy_kwargs, Mapping):
        raise TypeError("rough_bellman_policy_kwargs must be a mapping or None")
    bellman_options = dict(rough_bellman_policy_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & bellman_options.keys():
        raise ValueError("set Bellman-policy seed and reuse through named workflow arguments")
    if bellman_or_later and tail_options:
        raise ValueError("rough_tail_attribution_kwargs must be empty when replaying its historical campaign")
    if rough_bellman_delta_kwargs is not None and not isinstance(rough_bellman_delta_kwargs, Mapping):
        raise TypeError("rough_bellman_delta_kwargs must be a mapping or None")
    bellman_delta_options = dict(rough_bellman_delta_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & bellman_delta_options.keys():
        raise ValueError("set Bellman-delta seed and reuse through the named workflow arguments")
    if bellman_delta_or_later and bellman_options:
        raise ValueError("rough_bellman_policy_kwargs must be empty when replaying its historical campaign")
    if rough_price_budget_kwargs is not None and not isinstance(rough_price_budget_kwargs, Mapping):
        raise TypeError("rough_price_budget_kwargs must be a mapping or None")
    price_budget_options = dict(rough_price_budget_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & price_budget_options.keys():
        raise ValueError("set price-budget seed and reuse through the named workflow arguments")
    if price_budget_or_later and bellman_delta_options:
        raise ValueError("rough_bellman_delta_kwargs must be empty when replaying its historical campaign")
    if rough_mc_precision_kwargs is not None and not isinstance(rough_mc_precision_kwargs, Mapping):
        raise TypeError("rough_mc_precision_kwargs must be a mapping or None")
    mc_precision_options = dict(rough_mc_precision_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & mc_precision_options.keys():
        raise ValueError("set Monte Carlo precision seed and reuse through the named workflow arguments")
    if mc_precision_or_later and price_budget_options:
        raise ValueError("rough_price_budget_kwargs must be empty when replaying its historical campaign")
    if rough_shared_delta_kwargs is not None and not isinstance(rough_shared_delta_kwargs, Mapping):
        raise TypeError("rough_shared_delta_kwargs must be a mapping or None")
    shared_delta_options = dict(rough_shared_delta_kwargs or {})
    if {"seed", "reuse", "reuse_checkpoints"} & shared_delta_options.keys():
        raise ValueError("set shared-control delta seed and reuse through the named workflow arguments")
    if run_rough_shared_delta and mc_precision_options:
        raise ValueError("rough_mc_precision_kwargs must be empty when replaying its historical campaign")
    seed = _seed(seed)
    fourier = REFERENCE_CONFIG if fourier_config is None else fourier_config
    fd = FDConfig(100, 32, 100) if fd_config is None else fd_config
    if not isinstance(fourier, PricingConfig) or not isinstance(fd, FDConfig):
        raise TypeError("fourier_config must be PricingConfig and fd_config must be FDConfig")
    context.progress("Pricing European synthetic surfaces; no calibration is run in this notebook")
    surfaces = {model: synthetic_surface(model, seed=seed, config=fourier) for model in SYNTHETIC_TRUTHS}
    european = _ivs(pd.concat([v.assign(generating_model=k) for k, v in surfaces.items()], ignore_index=True))
    sensitivity_rows, failures = [], []
    for family, parameter, values, base in (
        ("Heston correlation", "rho", (-.85, -.6, -.25), SYNTHETIC_TRUTHS["heston"]),
        ("Heston vol of vol", "vol_of_vol", (.1, .3, .6), SYNTHETIC_TRUTHS["heston"]),
        ("Rough Hurst", "hurst", (.08, .15, .3, .5), SYNTHETIC_TRUTHS["rough_heston"]),
    ):
        grid = np.linspace(-.06, .06, 25)
        for value in values:
            part = pd.DataFrame({"forward": 100., "discount": np.exp(-.03*30/365),
                "strike": 100*np.exp(grid), "maturity": 30/365, "is_call": grid >= 0})
            try:
                part["market_price"] = price(replace(base, **{parameter: value}), forward=part.forward,
                    discount=part.discount, strike=part.strike, maturity=part.maturity,
                    is_call=part.is_call, config=fourier)
                part = _ivs(part)
            except PricingError as exc:
                part["market_price"], part["implied_volatility"], part["log_moneyness"] = np.nan, np.nan, grid
                _failure(failures, family, "parameter sensitivity", exc, parameter=parameter, value=value)
            sensitivity_rows.append(part.assign(family=family, parameter=parameter, value=value))
    sensitivities = pd.concat(sensitivity_rows, ignore_index=True)
    limits = _european_limits(fourier)
    refinements = []
    for model in ("heston", "rough_heston"):
        detail = refinement_table(SYNTHETIC_TRUTHS[model], surfaces[model], [COARSE_CONFIG, fourier, fourier.refined()])
        for item in detail.to_dict("records"):
            refinements.append({"model": model, "exercise_style": "european", "axis":
                f"N{item['quadrature_nodes']}/L{item['fourier_cutoff']:g}/M{item['riccati_steps']}",
                "status": item["status"], "price_change": item.get("max_change_vs_finest"),
                "delta_change": np.nan, "elapsed_seconds": item["elapsed_seconds"], **item})

    context.progress("Comparing identical European/American BS and Heston contracts")
    contracts, markets, rows, boundaries = _contracts(), _markets(), [], []
    for model, params in _fd_truths().items():
        for scenario, market in markets.items():
            results = {}
            for style in ("european", "american"):
                started = time.perf_counter()
                try:
                    result = evaluate_vanilla(params, market=market, strike=contracts.strike.to_numpy(),
                        maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
                        exercise_style=style, config=fd)
                    rows.append(contracts.assign(model=model, scenario=scenario, exercise_style=style,
                        price=result.prices, delta=result.deltas, elapsed_seconds=time.perf_counter()-started))
                    results[style] = result
                except PricingError as exc:
                    _failure(failures, model, "pricing", exc, scenario=scenario, exercise_style=style)
            if len(results) == 2:
                violation = float(np.max(np.maximum(results["european"].prices-results["american"].prices, 0.)))
                limits.append({"model": model, "check": f"American dominates European · {scenario}",
                    "error_measure": "maximum negative exercise premium", "value": violation,
                    "status": "passed" if violation <= 1e-7 else "failed", "note": "Identical inputs and FD mesh"})
        try:
            result = evaluate_vanilla(params, market=markets["cash dividend"], strike=np.array([99., 101.]),
                maturity=60/365, is_call=np.array([False, True]), exercise_style="american", config=fd)
            boundaries.extend([{**dict(item), "model": model} for item in result.exercise_boundaries])
            result = evaluate_vanilla(params, market=markets["cash dividend"], strike=np.array([95., 100., 105.]),
                maturity=.25, is_call=np.array([True, False, False]), exercise_style="american", config=fd,
                check_refinement=True)
            for item in result.diagnostics["checks"].to_dict("records"):
                refinements.append({**item, "model": model, "exercise_style": "american",
                    "price_change": item["max_price_change"], "delta_change": item["max_delta_change"]})
        except PricingError as exc:
            _failure(failures, model, "boundary/refinement", exc)
    args = dict(market=markets["cash dividend"], strike=np.array([95., 100., 105.]), maturity=.25,
                is_call=np.array([True, False, False]), exercise_style="american", config=fd)
    curve = HestonCurveParams(.04, 1.5, .3, -.6, MeanReversionCurve((.08,), (.045, .045)))
    try:
        difference = np.max(abs(evaluate_vanilla(curve, **args).prices-
                                evaluate_vanilla(_fd_truths()["Heston"], **args).prices))
        limits.append({"model": "Heston", "check": "Flat mean-reversion curve identity",
            "error_measure": "maximum absolute price difference", "value": difference,
            "status": "passed" if difference < 1e-7 else "failed", "note": "Calendar-time knot inserted in both rollbacks"})
    except PricingError as exc:
        _failure(failures, "Heston", "flat curve limit", exc)
    american = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    boundary_table, refinement = pd.DataFrame(boundaries), pd.DataFrame(refinements)
    figures = _pricing_figures(european, sensitivities, american, boundary_table, refinement)
    result = {"tables": {"configuration": _settings_table([
                ("shared", "seed", seed), ("European", "Fourier settings", str(asdict(fourier))),
                ("American", "FD settings", str(asdict(fd))), ("American", "market", str(asdict(markets["cash dividend"]))),
                ("scope", "calibration", "None; calibration is notebook 02")]),
            "european_prices": european, "european_sensitivities": sensitivities,
            "limit_checks": pd.DataFrame(limits), "american_prices": american,
            "exercise_boundaries": boundary_table, "refinement": refinement,
            "numerical_failures": pd.DataFrame(failures)},
        "figures": figures, "manifest": {"evidence_version": EVIDENCE_VERSION,
            "backend": MIXED_BACKEND, "data_kind": "synthetic", "exercise_style": "mixed",
            "seed": seed, "truths": {k: asdict(v) for k, v in SYNTHETIC_TRUTHS.items()},
            "fourier_config": asdict(fourier), "fd_config": asdict(fd), "markets": {k: asdict(v) for k, v in markets.items()},
            "optimizer_calls": 0, "scope": "Pricing, exercise and fixed-input stock sensitivity",
            "limitations": ["Coarse FD settings expose convergence failures; a finite final grid is not truth.",
                "Rough-Heston stopping is a separate finite-factor Monte Carlo study with explicit accuracy diagnostics.",
                "Cash dividends use limited liability; synthetic schedules are declared assumptions."]}}
    result["manifest"]["rough_foundation_executed"] = bool(run_rough_foundation)
    result["manifest"]["rough_foundation_included"] = bool(run_rough_foundation)
    result["manifest"]["rough_precision_executed"] = bool(run_rough_precision)
    result["manifest"]["rough_precision_included"] = bool(run_rough_precision)
    result["manifest"]["rough_american_executed"] = bool(run_rough_american and not run_rough_american_accuracy and not run_rough_stopping_attribution and not run_rough_dual_tightening and not run_rough_policy_improvement and not run_rough_control_centering and not run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_american_included"] = bool(run_rough_american or run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_american_accuracy_executed"] = bool(run_rough_american_accuracy and not run_rough_stopping_attribution and not run_rough_dual_tightening and not run_rough_policy_improvement and not run_rough_control_centering and not run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_american_accuracy_included"] = bool(run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_stopping_attribution_executed"] = bool(run_rough_stopping_attribution and not run_rough_dual_tightening and not run_rough_policy_improvement and not run_rough_control_centering and not run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_stopping_attribution_included"] = bool(run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_dual_tightening_executed"] = bool(run_rough_dual_tightening and not run_rough_policy_improvement and not run_rough_control_centering and not run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_dual_tightening_included"] = bool(run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_policy_improvement_executed"] = bool(run_rough_policy_improvement and not run_rough_control_centering and not run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_policy_improvement_included"] = bool(run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_control_centering_executed"] = bool(run_rough_control_centering and not run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_control_centering_included"] = bool(run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_centering_precision_executed"] = bool(run_rough_centering_precision and not stopping_delta_or_later)
    result["manifest"]["rough_centering_precision_included"] = bool(run_rough_centering_precision or stopping_delta_or_later)
    result["manifest"]["rough_stopping_delta_executed"] = bool(run_rough_stopping_delta and not policy_validation_or_later)
    result["manifest"]["rough_stopping_delta_included"] = bool(stopping_delta_or_later)
    result["manifest"]["rough_policy_validation_executed"] = bool(run_rough_policy_validation and not gap_time_or_later)
    result["manifest"]["rough_policy_validation_included"] = bool(policy_validation_or_later)
    result["manifest"]["rough_gap_time_refinement_executed"] = bool(run_rough_gap_time_refinement and not tail_or_later)
    result["manifest"]["rough_gap_time_refinement_included"] = bool(gap_time_or_later)
    result["manifest"]["rough_tail_attribution_executed"] = bool(run_rough_tail_attribution and not bellman_or_later)
    result["manifest"]["rough_tail_attribution_included"] = bool(tail_or_later)
    result["manifest"]["rough_bellman_policy_executed"] = bool(run_rough_bellman_policy and not bellman_delta_or_later)
    result["manifest"]["rough_bellman_policy_included"] = bool(bellman_or_later)
    result["manifest"]["rough_bellman_delta_executed"] = bool(run_rough_bellman_delta and not price_budget_or_later)
    result["manifest"]["rough_bellman_delta_included"] = bool(bellman_delta_or_later)
    result["manifest"]["rough_price_budget_executed"] = bool(run_rough_price_budget and not mc_precision_or_later)
    result["manifest"]["rough_price_budget_included"] = bool(price_budget_or_later)
    result["manifest"]["rough_mc_precision_executed"] = bool(run_rough_mc_precision and not run_rough_shared_delta)
    result["manifest"]["rough_mc_precision_included"] = bool(mc_precision_or_later)
    result["manifest"]["rough_shared_delta_executed"] = bool(run_rough_shared_delta)
    result["manifest"]["rough_shared_delta_included"] = bool(run_rough_shared_delta)
    if run_rough_foundation:
        if run_rough_american or run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_american import load_historical_foundation
            foundation = load_historical_foundation(context)
            result["manifest"]["rough_foundation_executed"] = False
            result["manifest"]["rough_foundation_execution_mode"] = "preserved_historical_records"
        elif run_rough_precision:
            from ._pricing_rough_precision import load_historical_foundation
            foundation = load_historical_foundation(context)
            result["manifest"]["rough_foundation_executed"] = False
            result["manifest"]["rough_foundation_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_foundation import rough_foundation_study
            foundation = rough_foundation_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **rough_options)
            result["manifest"]["rough_foundation_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_foundation_{name}": frame
                                for name, frame in foundation["tables"].items()})
        result["figures"].update({f"rough_foundation_{name}": figure
                                 for name, figure in foundation["figures"].items()})
        result["manifest"]["rough_foundation"] = foundation["manifest"]
        result["manifest"]["supplemental_backends"] = [foundation["manifest"]["backend"]]
    if run_rough_precision:
        if run_rough_american or run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_american import load_historical_precision
            precision = load_historical_precision(context)
            result["manifest"]["rough_precision_executed"] = False
            result["manifest"]["rough_precision_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_precision import rough_precision_study
            precision = rough_precision_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **precision_options)
            result["manifest"]["rough_precision_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_precision_{name}": frame
                                for name, frame in precision["tables"].items()})
        result["figures"].update({f"rough_precision_{name}": figure
                                 for name, figure in precision["figures"].items()})
        result["manifest"]["rough_precision"] = precision["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(precision["manifest"]["backend"])
    if run_rough_american or run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
        from ._pricing_rough_american import (
            rough_american_study, arithmetic_repair_comparison, load_historical_american,
        )
        if run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            american = load_historical_american(context)
            result["manifest"]["rough_american_execution_mode"] = "preserved_historical_records"
        else:
            american = rough_american_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **american_options)
            result["manifest"]["rough_american_execution_mode"] = "current_engine"
        comparison, comparison_summary = arithmetic_repair_comparison(
            american["tables"].get("prices", pd.DataFrame()),
            american["tables"].get("precision_guard_prices", pd.DataFrame()))
        american["tables"]["arithmetic_repair_comparison"] = comparison
        american["tables"]["arithmetic_repair_summary"] = comparison_summary
        result["tables"].update({f"rough_american_{name}": frame
                                for name, frame in american["tables"].items()})
        result["figures"].update({f"rough_american_{name}": figure
                                 for name, figure in american["figures"].items()})
        result["manifest"]["rough_american"] = american["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(american["manifest"]["backend"])
    if run_rough_american_accuracy or run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
        if run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_stopping_attribution import load_historical_accuracy
            accuracy = load_historical_accuracy(context)
            result["manifest"]["rough_american_accuracy_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_american_accuracy import rough_american_accuracy_study
            accuracy = rough_american_accuracy_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **accuracy_options)
            from ._pricing_rough_american_accuracy_diagnostics import add_accuracy_refinements, add_accuracy_extras, add_accuracy_execution
            accuracy = add_accuracy_execution(add_accuracy_extras(add_accuracy_refinements(accuracy)), output_dir=getattr(context, "output_dir", None))
            result["manifest"]["rough_american_accuracy_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_american_accuracy_{name}": frame
                                for name, frame in accuracy["tables"].items()})
        result["figures"].update({f"rough_american_accuracy_{name}": figure
                                 for name, figure in accuracy["figures"].items()})
        result["manifest"]["rough_american_accuracy"] = accuracy["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(accuracy["manifest"]["backend"])
    if run_rough_stopping_attribution or run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
        result["manifest"]["calibration_optimizer_calls"] = 0
        result["manifest"]["optimizer_calls_scope"] = "Model calibration; training-only martingale optimization is reported by the attribution study"
        if run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_dual_tightening import load_historical_attribution
            attribution = load_historical_attribution(context)
            result["manifest"]["rough_stopping_attribution_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_stopping_attribution import rough_stopping_attribution_study, add_attribution_extras, add_attribution_execution
            attribution = rough_stopping_attribution_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **attribution_options)
            attribution = add_attribution_execution(add_attribution_extras(attribution), output_dir=context.output_dir)
            result["manifest"]["rough_stopping_attribution_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_stopping_attribution_{name}": frame
                                for name, frame in attribution["tables"].items()})
        result["figures"].update({f"rough_stopping_attribution_{name}": figure
                                 for name, figure in attribution["figures"].items()})
        result["manifest"]["rough_stopping_attribution"] = attribution["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(attribution["manifest"]["backend"])
    if run_rough_dual_tightening or run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
        if run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_policy_improvement import load_historical_tightening
            tightening = load_historical_tightening(context)
            result["manifest"]["rough_dual_tightening_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_dual_tightening import rough_dual_tightening_study, add_dual_tightening_execution, add_dual_tightening_extras
            tightening = rough_dual_tightening_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **tightening_options)
            tightening = add_dual_tightening_execution(add_dual_tightening_extras(tightening), output_dir=context.output_dir)
            result["manifest"]["rough_dual_tightening_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_dual_tightening_{name}": frame
                                for name, frame in tightening["tables"].items()})
        result["figures"].update({f"rough_dual_tightening_{name}": figure
                                 for name, figure in tightening["figures"].items()})
        result["manifest"]["rough_dual_tightening"] = tightening["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(tightening["manifest"]["backend"])
    if run_rough_policy_improvement or run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
        if run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_control_centering import load_historical_policy
            policy = load_historical_policy(context)
            result["manifest"]["rough_policy_improvement_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_policy_improvement import rough_policy_improvement_study
            policy = rough_policy_improvement_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **policy_options)
            from ._pricing_rough_policy_reporting import add_policy_execution, add_policy_timing_figure, add_policy_reference_pairs, add_policy_control_neutrality
            policy = add_policy_control_neutrality(add_policy_reference_pairs(add_policy_timing_figure(add_policy_execution(policy, output_dir=context.output_dir))))
            result["manifest"]["rough_policy_improvement_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_policy_improvement_{name}": frame for name, frame in policy["tables"].items()})
        result["figures"].update({f"rough_policy_improvement_{name}": figure for name, figure in policy["figures"].items()})
        result["manifest"]["rough_policy_improvement"] = policy["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(policy["manifest"]["backend"])
    if run_rough_control_centering or run_rough_centering_precision or stopping_delta_or_later:
        if run_rough_centering_precision or stopping_delta_or_later:
            from ._pricing_rough_uncertainty_history import load_historical_centering
            centering = load_historical_centering(context)
            result["manifest"]["rough_control_centering_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_control_centering import control_centering_study
            centering = control_centering_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **centering_options)
            from ._pricing_rough_centering_reporting import add_centering_execution, add_centering_supplement
            centering = add_centering_supplement(add_centering_execution(centering, output_dir=context.output_dir), output_dir=context.output_dir)
            result["manifest"]["rough_control_centering_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_control_centering_{name}": frame for name, frame in centering["tables"].items()})
        result["figures"].update({f"rough_control_centering_{name}": figure for name, figure in centering["figures"].items()})
        result["manifest"]["rough_control_centering"] = centering["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(centering["manifest"]["backend"])
    if run_rough_centering_precision or stopping_delta_or_later:
        if stopping_delta_or_later:
            from ._pricing_rough_stopping_delta_history import load_historical_centering_precision
            precision = load_historical_centering_precision(context)
            result["manifest"]["rough_centering_precision_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_centering_precision import centering_precision_study
            precision = centering_precision_study(context, seed=seed,
                reuse_checkpoints=rough_reuse_checkpoints, **centering_precision_options)
            result["manifest"]["rough_centering_precision_execution_mode"] = "current_engine"
        result["tables"].update({f"rough_centering_precision_{name}": frame for name, frame in precision["tables"].items()})
        result["figures"].update({f"rough_centering_precision_{name}": figure for name, figure in precision["figures"].items()})
        result["manifest"]["rough_centering_precision"] = precision["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(precision["manifest"]["backend"])
    if stopping_delta_or_later:
        if policy_validation_or_later:
            from ._pricing_rough_policy_validation_history import load_historical_stopping_delta
            study = load_historical_stopping_delta(context)
            result["manifest"]["rough_stopping_delta_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_stopping_delta import stopping_delta_study
            study = stopping_delta_study(context, seed=seed, **stopping_delta_options)
            result["manifest"]["rough_stopping_delta_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_stopping_delta_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_stopping_delta_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_stopping_delta"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if policy_validation_or_later:
        if gap_time_or_later:
            from ._pricing_rough_gap_time_history import load_historical_policy_validation
            study = load_historical_policy_validation(context)
            result["manifest"]["rough_policy_validation_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_policy_validation import policy_validation_study
            study = policy_validation_study(context, seed=seed, **policy_validation_options)
            result["manifest"]["rough_policy_validation_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_policy_validation_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_policy_validation_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_policy_validation"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if gap_time_or_later:
        if tail_or_later:
            from ._pricing_rough_tail_history import load_historical_gap_time_refinement
            study = load_historical_gap_time_refinement(context)
            result["manifest"]["rough_gap_time_refinement_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_gap_time_refinement import gap_time_refinement_study
            study = gap_time_refinement_study(context, seed=seed, **gap_time_options)
            result["manifest"]["rough_gap_time_refinement_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_gap_time_refinement_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_gap_time_refinement_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_gap_time_refinement"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if tail_or_later:
        if bellman_or_later:
            from ._pricing_rough_bellman_history import load_historical_tail_attribution
            study = load_historical_tail_attribution(context)
            result["manifest"]["rough_tail_attribution_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_tail_attribution import tail_attribution_study
            study = tail_attribution_study(context, seed=seed, **tail_options)
            result["manifest"]["rough_tail_attribution_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_tail_attribution_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_tail_attribution_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_tail_attribution"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if bellman_or_later:
        if bellman_delta_or_later:
            from ._pricing_rough_bellman_delta_history import load_historical_bellman_policy
            study = load_historical_bellman_policy(context)
            result["manifest"]["rough_bellman_policy_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_bellman_study import bellman_policy_study
            study = bellman_policy_study(context, seed=seed, **bellman_options)
            result["manifest"]["rough_bellman_policy_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_bellman_policy_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_bellman_policy_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_bellman_policy"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if bellman_delta_or_later:
        if price_budget_or_later:
            from ._pricing_rough_price_budget_history import load_historical_bellman_delta
            study = load_historical_bellman_delta(context)
            result["manifest"]["rough_bellman_delta_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_bellman_delta_collection import bellman_delta_study
            study = bellman_delta_study(context, seed=seed, **bellman_delta_options)
            result["manifest"]["rough_bellman_delta_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_bellman_delta_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_bellman_delta_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_bellman_delta"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if price_budget_or_later:
        if mc_precision_or_later:
            from ._pricing_rough_mc_precision_history import load_historical_price_budget
            study = load_historical_price_budget(context)
            result["manifest"]["rough_price_budget_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_price_budget_collection import price_budget_study
            study = price_budget_study(context, seed=seed, **price_budget_options)
            result["manifest"]["rough_price_budget_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_price_budget_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_price_budget_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_price_budget"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if mc_precision_or_later:
        if run_rough_shared_delta:
            from ._pricing_rough_shared_delta_history import load_historical_mc_precision
            study = load_historical_mc_precision(context)
            result["manifest"]["rough_mc_precision_execution_mode"] = "preserved_historical_records"
        else:
            from ._pricing_rough_mc_precision_collection import mc_precision_study
            study = mc_precision_study(context, seed=seed, **mc_precision_options)
            result["manifest"]["rough_mc_precision_execution_mode"] = "verified_current_campaign_collection"
        result["tables"].update({f"rough_mc_precision_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_mc_precision_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_mc_precision"] = study["manifest"]
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    if run_rough_shared_delta:
        from ._pricing_rough_shared_delta_collection import shared_delta_study
        study = shared_delta_study(context, seed=seed, **shared_delta_options)
        result["tables"].update({f"rough_shared_delta_{name}": frame for name, frame in study["tables"].items()})
        result["figures"].update({f"rough_shared_delta_{name}": figure for name, figure in study["figures"].items()})
        result["manifest"]["rough_shared_delta"] = study["manifest"]
        result["manifest"]["rough_shared_delta_execution_mode"] = "verified_current_campaign_collection"
        result["manifest"].setdefault("supplemental_backends", []).append(study["manifest"]["backend"])
    return result


def _pricing_figures(european, sensitivities, american, boundaries, refinement):
    import matplotlib.pyplot as plt
    figures = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, constrained_layout=True)
    for ax, (tenor, group) in zip(axes, european.groupby("maturity", sort=True)):
        for model, part in group.groupby("generating_model", sort=False):
            ax.plot(part.log_moneyness, part.implied_volatility*100, "o-", label=MODEL_LABELS.get(model, model))
        ax.set(title=f"{tenor*365:.0f} days", xlabel="log(K/F)")
    axes[0].set_ylabel("European BS implied volatility (%)")
    axes[-1].legend(fontsize=8)
    figures["european_smiles"] = fig
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, (family, part) in zip(axes, sensitivities.groupby("family", sort=False)):
        for value, group in part.groupby("value", sort=False):
            ax.plot(group.log_moneyness, group.implied_volatility*100, label=f"{value:g}")
        ax.set(title=family, xlabel="log(K/F)", ylabel="European BS IV (%)")
        ax.legend(title=part.parameter.iloc[0], fontsize=8)
    figures["parameter_effects"] = fig
    if not american.empty:
        premiums = american.pivot(index=["contract_id", "model", "scenario", "strike", "maturity", "is_call"],
                                 columns="exercise_style", values="price").reset_index()
        premiums["premium"] = premiums.get("american", np.nan)-premiums.get("european", np.nan)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for ax, maturity in zip(axes, sorted(premiums.maturity.unique())):
            for (model, scenario), group in premiums.loc[premiums.maturity.eq(maturity)].groupby(["model", "scenario"], sort=False):
                ax.plot(group.strike, group.premium, "o-", label=f"{model} · {scenario}")
            ax.set(title=f"{maturity*365:.0f} days", xlabel="Strike ($)", ylabel="American − European ($)")
        axes[-1].legend(fontsize=6)
        figures["exercise_premiums"] = fig
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
        selected = american.loc[american.scenario.eq("cash dividend") & american.maturity.eq(60/365)]
        for (model, style), group in selected.groupby(["model", "exercise_style"], sort=False):
            axes[0].plot(group.strike, group.price, "o-", label=f"{model} · {style}")
            axes[1].plot(group.strike, group.delta, "o-", label=f"{model} · {style}")
        axes[0].set(xlabel="Strike ($)", ylabel="Price per underlying unit ($)", title="Identical 60-day positions")
        axes[1].set(xlabel="Strike ($)", ylabel="Stock delta", title="Parameters and market assumptions held fixed")
        axes[-1].legend(fontsize=7)
        figures["prices_and_deltas"] = fig
    if not boundaries.empty:
        fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
        for (model, strike, call), group in boundaries.groupby(["model", "strike", "is_call"], sort=False):
            group = group.sort_values("time", kind="stable")
            ax.plot(group.time*365, group.spot_boundary, ".-", markersize=2,
                    label=f"{model} · {strike:g} {'call' if call else 'put'}")
        ax.axvline(.08*365, color="black", ls="--", label="Cash dividend")
        ax.set(xlabel="Days from valuation", ylabel="Exercise boundary ($)", title="Exercise frontiers on the declared dividend schedule")
        ax.legend(fontsize=7)
        figures["exercise_boundaries"] = fig
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    european = refinement.loc[refinement.exercise_style.eq("european")]
    for model, group in european.groupby("model", sort=False):
        axes[0].plot(np.arange(len(group)), group.price_change, "o-", label=MODEL_LABELS.get(model, model))
    axes[0].set(xticks=[0, 1, 2], xticklabels=["coarse", "generation", "refined"],
                ylabel="Maximum price change ($)", title="European Fourier / Riccati checks")
    axes[0].legend(fontsize=7)
    selected = refinement.loc[refinement.exercise_style.eq("american")]
    order = ["spot", "variance", "time", "domain", "joint"]
    positions = np.arange(len(order))
    groups = list(selected.groupby("model", sort=False))
    width = .8/max(len(groups), 1)
    for index, (model, group) in enumerate(groups):
        group = group.set_index("axis").reindex(order)
        offset = (index-(len(groups)-1)/2)*width
        axes[1].bar(positions+offset, group.price_change, width=width, label=model)
        axes[2].bar(positions+offset, group.delta_change, width=width, label=model)
    for axis in axes[1:]:
        axis.set_xticks(positions, order)
    axes[1].set(ylabel="Maximum price change ($)", title="American finite-difference checks")
    axes[2].set(ylabel="Maximum delta change", title="Stock-delta refinement")
    axes[-1].legend(fontsize=8)
    figures["numerical_refinement"] = fig
    return figures


def _calibration_fd_settings(fd_config, calibration_configs, generation_configs):
    coarse = FDConfig(100, 32, 100) if fd_config is None else fd_config
    fits = ({"bs": FDConfig(400, 80, 400), "heston": FDConfig(240, 80, 300)}
            if calibration_configs is None else dict(calibration_configs))
    generated = ({k: v.refined("joint") for k, v in fits.items()}
                 if generation_configs is None else dict(generation_configs))
    if (not isinstance(coarse, FDConfig) or set(fits) != {"bs", "heston"}
            or set(generated) != {"bs", "heston"} or any(not isinstance(v, FDConfig) for v in (*fits.values(), *generated.values()))):
        raise TypeError("American calibration and generation settings require BS/Heston FDConfig mappings")
    for key, fit in fits.items():
        reference = generated[key]
        if any(reference.spot_steps <= candidate.spot_steps or reference.time_steps <= candidate.time_steps
            or key == "heston" and reference.variance_steps <= candidate.variance_steps
            or reference.spot_max_multiple < candidate.spot_max_multiple or reference.variance_max < candidate.variance_max
            for candidate in (fit, coarse)):
            raise ValueError(f"{key}: generation must be strictly finer than fitting in space/time/variance with no smaller domains")
    return coarse, fits, generated


def _american_attempt(label, study, quotes, market, initial, bounds, config, budget, truth=None, optimizer=None):
    """One fit, one all-contract validation; failure rows survive intact."""
    started = time.perf_counter()
    row = {"model": label, "study": study, "status": "numerical_failure", "ready": False,
        "price_stable": False, "delta_stable": False, "hedge_delta_ready": False,
        "optimizer_success": False, "train_rmse": None, "test_rmse": None,
        "train_scaled_rmse": None, "test_scaled_rmse": None, "nfev": 0, "pricing_evaluations": 0,
        "active_bounds": {}, "initial_params": parameter_values(initial), "requested_bounds": bounds,
        "generating_params": parameter_values(truth) if truth is not None else None,
        "config": asdict(config), "error": None,
        "free_parameters": [name for name in parameter_values(initial) if bounds is None or name not in bounds or bounds[name][0] != bounds[name][1]]}
    prediction, parameters, checks, fitted = pd.DataFrame(), [], pd.DataFrame(), None
    try:
        fit = calibrate_vanilla_surface(quotes.loc[quotes.split.eq("train")], initial,
            market=market, exercise_style="american", config=config, max_nfev=budget, bounds=bounds, optimizer=optimizer)
        fitted = fit.params
        values = evaluate_vanilla(fit.params, market=market, strike=quotes.strike.to_numpy(),
            maturity=quotes.maturity.to_numpy(), is_call=quotes.is_call.to_numpy(),
            exercise_style="american", config=config, check_refinement=True)
        prediction = quotes.assign(model=label, study=study, model_price=values.prices, delta=values.deltas)
        scales = np.maximum((quotes.ask-quotes.bid).to_numpy()/2, .005) if "bid" in quotes else np.ones(len(quotes))
        prediction["price_residual"] = prediction.model_price-prediction.market_price
        prediction["scaled_residual"] = prediction.price_residual/scales
        price_ok = bool(fit.price_stable and values.price_stable)
        if "bid" in quotes:
            price_ok = price_ok and all(np.all(np.asarray(change) <= .1*scales)
                for change in values.diagnostics.get("price_changes", {}).values())
        delta_ok = bool(fit.delta_stable and values.delta_stable)
        row.update(optimizer_success=fit.optimizer_success, price_stable=price_ok, delta_stable=delta_ok,
            ready=bool(fit.optimizer_success and price_ok), hedge_delta_ready=bool(fit.optimizer_success and price_ok and delta_ok),
            status="converged" if fit.optimizer_success and price_ok else "unstable" if fit.optimizer_success else "exhausted",
            nfev=fit.nfev, pricing_evaluations=fit.pricing_evaluations+values.diagnostics.get("pricing_evaluations", 0),
            optimizer_status=fit.optimizer_status, optimizer_message=fit.optimizer_message,
            optimizer_settings=dict(fit.optimizer_settings), active_bounds=dict(fit.active_bounds), params=parameter_values(fit.params),
            training_objective=fit.objective, physical_optimality=fit.physical_optimality,
            max_price_change=fit.max_price_change, max_delta_change=fit.max_delta_change)
        checks = values.diagnostics.get("checks", pd.DataFrame()).assign(model=label, study=study)
        if not checks.empty:
            row["max_price_change"] = max(fit.max_price_change, float(checks.max_price_change.max()))
            row["max_delta_change"] = max(fit.max_delta_change, float(checks.max_delta_change.max()))
        for split, prefix in (("train", "train"), ("heldout", "test")):
            group = prediction.loc[prediction.split.eq(split)]
            row[f"{prefix}_rmse"] = float(np.sqrt(np.mean(group.price_residual**2))) if len(group) else None
            row[f"{prefix}_scaled_rmse"] = float(np.sqrt(np.mean(group.scaled_residual**2))) if len(group) else None
        for name, value in parameter_values(fit.params).items():
            parameters.append({"model": label, "study": study, "parameter": name,
                "generating": parameter_values(truth).get(name) if truth is not None else None,
                "initial": parameter_values(initial)[name], "fitted": value,
                "fixed": bounds is not None and name in bounds and bounds[name][0] == bounds[name][1],
                "active_bound": dict(fit.active_bounds).get(name)})
    except PricingError as exc:
        row["error"] = str(exc)
        fitted = None
    row["elapsed_seconds"] = time.perf_counter()-started
    return row, prediction, parameters, checks, fitted


def calibration_research(context, *, seed=42, coarse_config=None, fine_config=None,
        fd_config=None, calibration_configs=None, generation_configs=None,
        max_nfev=80, american_max_nfev=16, run_extended=False, extended_max_nfev=80,
        run_controlled_american=False, run_american_joint=True,
        joint_fit_config=None, joint_generation_config=None, joint_continuation_config=None,
        joint_max_nfev=80, joint_continuation_max_nfev=160,
        joint_run_continuation=True, joint_reuse_checkpoints=True,
        run_curve_diagnosis=False, curve_diagnosis_max_nfev=40, curve_diagnosis_reuse=True,
        run_maturity_extension=False, maturity_fit_config=None, maturity_generation_config=None,
        maturity_max_nfev=80, maturity_reuse=True):
    """European controls and joint American identification on synthetic contracts."""
    import matplotlib.pyplot as plt
    seed, budget, american_budget = _seed(seed), _budget(max_nfev, 80, "max_nfev"), _budget(american_max_nfev, 24, "american_max_nfev")
    for name, flag in (("run_extended", run_extended), ("run_controlled_american", run_controlled_american),
                       ("run_american_joint", run_american_joint), ("run_curve_diagnosis", run_curve_diagnosis),
                       ("run_maturity_extension", run_maturity_extension)):
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    if run_curve_diagnosis and not run_american_joint:
        raise ValueError("Curve diagnosis requires the original joint American study")
    coarse = COARSE_CONFIG if coarse_config is None else coarse_config
    fine = REFERENCE_CONFIG if fine_config is None else fine_config
    if not isinstance(coarse, PricingConfig) or not isinstance(fine, PricingConfig):
        raise TypeError("coarse_config and fine_config must be PricingConfig")
    if fine.quadrature_nodes <= coarse.quadrature_nodes or fine.fourier_cutoff < coarse.fourier_cutoff or fine.riccati_steps <= coarse.riccati_steps:
        raise ValueError("European generation must have strictly finer quadrature/Riccati resolution and no smaller cutoff")
    fd, fits, generated = _calibration_fd_settings(fd_config, calibration_configs, generation_configs)
    context.progress("Fitting European synthetic surfaces; price recovery and parameter recovery are reported separately")
    surfaces = {model: synthetic_surface(model, seed=seed, config=fine) for model in SYNTHETIC_TRUTHS}
    attempts, truths = [], {}
    for model, surface in surfaces.items():
        initial = SYNTHETIC_INITIALS[model]
        bounds = ({name: (value, value) for name, value in asdict(initial).items() if name != "v0"}
                  if model == "rough_heston" else None)
        attempts.append(fit_attempt(model, surface, initial, bounds=bounds, config=coarse, max_nfev=budget))
        truths[model] = asdict(SYNTHETIC_TRUTHS[model])
    flat = fit_attempt("flat_bs_on_heston", surfaces["heston"], BlackScholesParams(.2), config=coarse, max_nfev=budget)
    attempts.append(flat)
    inverse = _ivs(surfaces["heston"])[["contract_id", "strike", "maturity", "market_price", "implied_volatility", "split", "forward"]].copy()
    inverse["flat_surface_volatility"] = flat.result.params.volatility if flat.result else np.nan
    europe_summary, europe_params = _summary(attempts), _parameter_rows(attempts, truths)
    europe_predictions = pd.concat([a.predictions.assign(model=a.label) for a in attempts], ignore_index=True)
    market, contracts = _markets()["cash dividend"], _contracts()
    american_rows, american_params, american_predictions, american_checks = [], [], [], []
    for label, truth in (_fd_truths().items() if run_controlled_american else []):
        key = "bs" if isinstance(truth, BlackScholesParams) else "heston"
        context.progress(f"Generating finer-grid American {label} observations, then fitting controlled parameters")
        try:
            observed = evaluate_vanilla(truth, market=market, strike=contracts.strike.to_numpy(),
                maturity=contracts.maturity.to_numpy(), is_call=contracts.is_call.to_numpy(),
                exercise_style="american", config=generated[key])
        except PricingError as exc:
            american_rows.append({"model": label, "study": "generation", "status": "numerical_failure",
                "error": str(exc), "ready": False, "price_stable": False, "delta_stable": False,
                "hedge_delta_ready": False, "optimizer_success": False, "train_rmse": None,
                "test_rmse": None, "free_parameters": ["volatility" if key == "bs" else "v0"],
                "nfev": 0, "pricing_evaluations": 0, "max_price_change": None,
                "max_delta_change": None, "elapsed_seconds": 0.})
            continue
        quotes = contracts.assign(market_price=observed.prices, exercise_style="american")
        initial = BlackScholesParams(.25) if key == "bs" else replace(truth, v0=.06)
        bounds = None if key == "bs" else {name: (value, value) for name, value in asdict(truth).items() if name != "v0"}
        for study, config in (("deliberately coarse", fd), ("requested starting budget", fits[key])):
            row, prediction, params, checks, _ = _american_attempt(label, study, quotes, market, initial, bounds,
                config, american_budget, truth=truth)
            row["generation_config"] = asdict(generated[key])
            american_rows.append(row)
            american_params.extend(params)
            if not prediction.empty:
                american_predictions.append(prediction)
            if not checks.empty:
                american_checks.append(checks)
    american_summary = pd.DataFrame(american_rows)
    tables = {"configuration": _settings_table([
            ("shared", "seed", seed), ("European", "generation settings", str(asdict(fine))),
            ("European", "fit settings", str(asdict(coarse))), ("European", "optimizer budget", budget),
            ("American", "coarse settings", str(asdict(fd))),
            *(("American", f"{key} fit settings", str(asdict(value))) for key, value in fits.items()),
            *(("American", f"{key} generation settings", str(asdict(value))) for key, value in generated.items()),
            ("American", "controlled study executed", bool(run_controlled_american)),
            ("American", "joint study executed", bool(run_american_joint)),
            ("American", "controlled optimizer budget", american_budget),
            ("extended", "six-parameter European study executed", bool(run_extended))]),
        "bs_inverse": inverse, "european_summary": europe_summary, "european_parameters": europe_params,
        "european_predictions": europe_predictions, "american_summary": american_summary,
        "american_parameters": pd.DataFrame(american_params),
        "american_predictions": pd.concat(american_predictions, ignore_index=True) if american_predictions else pd.DataFrame(),
        "american_refinement": pd.concat(american_checks, ignore_index=True) if american_checks else pd.DataFrame()}
    figures = {}
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    for tenor, group in inverse.groupby("maturity", sort=True):
        ax.plot(np.log(group.strike/group.forward), group.implied_volatility*100, "o-", label=f"{tenor*365:.0f}d individual IV")
    if flat.result:
        ax.axhline(flat.result.params.volatility*100, color="black", ls="--", label="One fitted flat volatility")
    ax.set(xlabel="log(K/F)", ylabel="European BS IV (%)", title="Per-contract inversion and flat-surface calibration answer different questions")
    ax.legend(fontsize=8)
    figures["inverse_and_flat_fit"] = fig
    figures["european_recovery"] = _recovery_plot(europe_summary.rename(columns={"label": "model"}), "European controlled price recovery")
    if not american_summary.empty:
        figures["american_recovery"] = _recovery_plot(american_summary, "American controlled price recovery and independent refinement")
    figures["parameter_recovery"] = _parameter_recovery_plot(europe_params)
    manifest = {"evidence_version": EVIDENCE_VERSION, "backend": MIXED_BACKEND, "data_kind": "synthetic",
        "exercise_style": "mixed", "seed": seed,
        "noise_std_by_study": {"european_controls": 0., "controlled_american": 0.,
            "american_joint_noisy": .005 if run_american_joint else None},
        "truths": {k: asdict(v) for k, v in SYNTHETIC_TRUTHS.items()},
        "european_generation_config": asdict(fine), "european_fit_config": asdict(coarse),
        "european_attempts": [a.summary for a in attempts], "american_attempts": american_rows,
        "american_market": asdict(market), "american_fit_configs": {k: asdict(v) for k, v in fits.items()},
        "american_generation_configs": {k: asdict(v) for k, v in generated.items()},
        "max_nfev": budget, "american_max_nfev": american_budget, "extended_executed": bool(run_extended),
        "controlled_american_executed": bool(run_controlled_american), "american_joint_executed": bool(run_american_joint),
        "selection_rule": "Training objective only; held-out errors never select starts or parameters",
        "limitations": ["The optional controlled American Heston comparison estimates v0 only; the joint study frees all coefficients.",
            "Rough-Heston controlled calibration estimates v0 only; six-parameter trials are optional.",
            "Optimizer convergence, price stability, delta stability and parameter identification are distinct.",
            "Matching current-study checkpoints may be reused explicitly; superseded fits are never substituted."]}
    if run_american_joint:
        from ._pricing_american_identification import american_identification_study
        joint = american_identification_study(context, seed=seed,
            fit_config=joint_fit_config, generation_config=joint_generation_config,
            continuation_config=joint_continuation_config, max_nfev=joint_max_nfev,
            continuation_max_nfev=joint_continuation_max_nfev,
            run_continuation=joint_run_continuation, reuse_checkpoints=joint_reuse_checkpoints)
        tables.update({f"american_joint_{k}": v for k, v in joint["tables"].items()})
        figures.update({f"american_joint_{k}": v for k, v in joint["figures"].items()})
        manifest["american_joint"] = joint["manifest"]
        if getattr(context, "output_dir", None) is not None:
            import hashlib
            from pathlib import Path
            folder = Path(context.output_dir)/"american_joint_identification"
            manifest["supplemental_numerical_review"] = [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "role": "Prior 80-node grid review, before the spot-mesh correction"}
                for name in ("jacobian_review.json", "jacobian_review.csv", "jacobian_review_method.md")
                if (path := folder/name).is_file()]
            current_review = Path(context.output_dir)/"spot_grid_trials_v2"
            manifest["spot_grid_investigation"] = [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name in ("manifest.json", "jacobian_192_review.json", "jacobian_192_review.csv",
                             "target_revision_comparison.json", "target_revision_comparison.csv")
                if (path := current_review/name).is_file()]
            checkpoint_directory = joint["manifest"].get("checkpoint_directory")
            if checkpoint_directory is not None:
                import json
                from datetime import datetime
                records = []
                for path in sorted(Path(checkpoint_directory).glob("worker_*.json")):
                    record = json.loads(path.read_text())
                    if record["manifest"]["run_fingerprint"] == joint["manifest"]["run_fingerprint"]:
                        records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            **{key: record[key] for key in ("worker", "started_at", "completed_at",
                                "elapsed_seconds", "thread_environment")}})
                if records:
                    starts = [datetime.fromisoformat(record["started_at"]) for record in records]
                    ends = [datetime.fromisoformat(record["completed_at"]) for record in records]
                    manifest["american_joint_parallel_execution"] = {"workers": records,
                        "wall_seconds": (max(ends)-min(starts)).total_seconds(),
                        "timing_scope": "Completed calibration workers; synthetic generation and accuracy preflight ran earlier"}
    if run_curve_diagnosis:
        from ._pricing_curve_diagnosis import curve_diagnosis
        diagnosis = curve_diagnosis(context, joint["manifest"], max_nfev=curve_diagnosis_max_nfev,
                                    reuse=curve_diagnosis_reuse)
        tables.update({f"curve_diagnosis_{key}": value for key, value in diagnosis["tables"].items()})
        figures.update({f"curve_diagnosis_{key}": value for key, value in diagnosis["figures"].items()})
        manifest["american_curve_diagnosis"] = diagnosis["manifest"]
    if run_maturity_extension:
        from ._pricing_maturity_study import maturity_extension
        extended_maturities = maturity_extension(context, seed=seed, fit_config=maturity_fit_config,
            generation_config=maturity_generation_config, max_nfev=maturity_max_nfev, reuse=maturity_reuse)
        tables.update({f"maturity_extension_{key}": value for key, value in extended_maturities["tables"].items()})
        figures.update({f"maturity_extension_{key}": value for key, value in extended_maturities["figures"].items()})
        manifest["maturity_extension"] = extended_maturities["manifest"]
    if run_extended:
        from ._pricing_extended import extended_demo
        extended = extended_demo(context, seed=seed, max_nfev=_budget(extended_max_nfev, 80, "extended_max_nfev"))
        tables.update({f"extended_{k}": v for k, v in extended["tables"].items()})
        figures.update({f"extended_{k}": v for k, v in extended["figures"].items()})
        manifest["extended"] = extended["manifest"]
    return {"tables": tables, "figures": figures, "manifest": manifest}


def _parameter_recovery_plot(parameters):
    """Show estimated coefficients; fixed controls are not recovered estimates."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    fixed = parameters.fixed.astype(str).str.lower().eq("true")
    visible = parameters.loc[parameters.generating_value.notna()
        & parameters.generating_value.ne(0) & ~fixed]
    for label, group in visible.groupby("fit", sort=False):
        name = MODEL_LABELS.get(label, label)
        if group.parameter.tolist() == ["v0"]:
            name += " (v0 only)"
        ax.plot(group.parameter, group.fitted_value/group.generating_value, "o-", label=name)
    ax.axhline(1., color="black", ls="--")
    ax.set(ylabel="Estimated / generating parameter",
        title="Price recovery does not establish unique parameter recovery")
    ax.legend(fontsize=8)
    return fig


def _recovery_plot(summary, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    summary = summary.copy()
    x = np.arange(len(summary))
    stages = {"deliberately coarse": "coarse", "requested starting budget": "starting budget"}
    labels = [f"{MODEL_LABELS.get(row.model, row.model)}\n{stages.get(getattr(row, 'study', ''), getattr(row, 'study', ''))}"
              for row in summary.itertuples()]
    for column, shift, label in (("train_rmse", -.15, "Training"), ("test_rmse", .15, "Held out")):
        values = pd.to_numeric(summary.get(column, pd.Series(np.nan, index=summary.index)), errors="coerce").to_numpy(float)
        axes[0].bar(x+shift, values, width=.28, label=label)
    changes = pd.to_numeric(summary.get("max_price_change", pd.Series(np.nan, index=summary.index)), errors="coerce").to_numpy(float)
    axes[1].bar(x, changes)
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_ylabel("Dollars per underlying unit")
    axes[0].set_title("Recovered prices on the fitting grid")
    axes[1].set_title("Maximum independent refinement change")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    return fig


def spy_research(context, *, seed=42, data_root=None, config=None, max_nfev=40,
                 dividend_coverage_through=None):
    """One American SPY input audit and a common, reusable fit comparison."""
    seed, budget = _seed(seed), _budget(max_nfev, 80, "max_nfev")
    if config is not None and not isinstance(config, FDConfig):
        raise TypeError("SPY calibration requires FDConfig")
    inputs = load_american_spy_inputs(data_root, dividend_coverage_through=dividend_coverage_through)
    audit, snapshot = inputs["audit"], inputs["snapshots"].get(DATES[0])
    ready = snapshot is not None and snapshot.ready
    studies = [
        ("Common calibration", "BS, Heston and calendar-time Heston curve; identical American contracts"),
        ("Numerical accuracy", "Independent spot, variance, time and domain price/delta checks"),
        ("Bounds", "Declared versus expanded Heston box; same training objective"),
        ("Convergence", "Curve alternate start and asinh coordinates; held-out prices never select a fit"),
        ("Input sensitivity", "Fixed fitted parameters; spot/rates and counterfactual declared cash amounts"),
    ]
    tables = {"input_audit": audit, "observed_quotes": inputs["quotes"], "input_exclusions": inputs["exclusions"],
        "study_design": pd.DataFrame([{"study": name, "comparison": comparison, "fit_date": DATES[0],
            "exercise_style": "american", "status": "ready" if ready else "blocked"} for name, comparison in studies])}
    figures = {"input_readiness": _audit_plot(audit)}
    if not inputs["quotes"].empty:
        figures["observed_prices"] = _raw_price_plot(inputs["quotes"])
    manifest = {"evidence_version": EVIDENCE_VERSION, "backend": AMERICAN_BACKEND,
        "exercise_style": "american", "data_kind": "real_spy", "seed": seed,
        "status": "ready" if ready else "blocked_inputs", "sample_dates": DATES, "expiries": EXPIRIES,
        "calibration_ready": bool(ready), "hedging_ready": False, "input_audit_count": 1,
        "processed_root": inputs["processed_root"], "inputs": inputs["fingerprints"],
        "snapshot_provenance": {day: dict(value.provenance) for day, value in inputs["snapshots"].items()},
        "assumptions": ["Observed American bid/ask quotes are fitted directly.",
            "Independent raw stock and historically available dated dividend expectations are required.",
            "Treasury discounts are an explicit rate proxy, not historical OIS vintages.",
            "Quote interval timestamps do not establish quote-change freshness.",
            "No parity forward, inferred stock, de-Americanization or European SPY calibration is used."],
        "superseded_results": "Archived results remain provenance; no result is loaded or relabelled."}
    if not ready:
        context.progress("SPY calibration is blocked by missing independent stock/dividend inputs; no fits are run")
        return {"tables": tables, "figures": figures, "manifest": manifest}
    _spy_fits(context, snapshot, config or FDConfig(240, 80, 300), budget, tables, figures, manifest)
    return {"tables": tables, "figures": figures, "manifest": manifest}


def _spy_fits(context, snapshot, config, budget, tables, figures, manifest):
    """Fit each declared attempt once, then reuse its parameters for sensitivity."""
    import matplotlib.pyplot as plt
    quotes, market = select_american_contracts(snapshot)
    curve = MeanReversionCurve(tuple(sorted(quotes.maturity.unique())[:-1]), (.045,)*len(quotes.maturity.unique()))
    heston, initial_curve = HestonParams(.04, .045, 1.5, .3, -.6), HestonCurveParams(.04, 1.5, .3, -.6, curve)
    def demo_bounds(params):
        return {name: DEMO_BOUNDS["theta" if name.startswith("theta_") else name] for name in parameter_values(params)}
    specs = [
        ("Black–Scholes", BlackScholesParams(.2), None, "physical", "common"),
        ("Heston", heston, demo_bounds(heston), "physical", "common"),
        ("Heston curve", initial_curve, demo_bounds(initial_curve), "physical", "common"),
        ("Heston expanded bounds", heston, {"v0": (.0025, .25), "theta": (.0025, 2.), "kappa": (.01, 80.), "vol_of_vol": (.01, 2.), "rho": (-.95, 0.)}, "physical", "bounds"),
        ("Heston curve alternate start", replace(initial_curve, v0=.025, kappa=4., vol_of_vol=.5, rho=-.4), demo_bounds(initial_curve), "physical", "convergence"),
        ("Heston curve asinh", initial_curve, demo_bounds(initial_curve), "asinh", "convergence"),
    ]
    attempts, predictions, params, checks, fits = [], [], [], [], {}
    for label, initial, bounds, coordinates, study in specs:
        context.progress(f"American SPY: {label}; one common training/held-out contract set")
        row, prediction, parameters, diagnostic, fitted = _american_attempt(label, study, quotes, market, initial,
            bounds, config, budget, optimizer=OptimizerConfig(parameterization=coordinates))
        attempts.append(row)
        params.extend(parameters)
        if not prediction.empty:
            predictions.append(prediction)
        if not diagnostic.empty:
            checks.append(diagnostic)
        if fitted is not None and not prediction.empty:
            fits[label] = fitted
    prediction_table = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    check_table = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    sensitivity = []
    for label in ("Black–Scholes", "Heston", "Heston curve"):
        if label not in fits:
            continue
        baseline = prediction_table.loc[prediction_table.model.eq(label), "model_price"].to_numpy()
        scenarios = [
            ("spot −0.1%", replace(market, spot=market.spot*.999)),
            ("spot +0.1%", replace(market, spot=market.spot*1.001)),
            ("rates −25bp", replace(market, discount_curve=DiscountCurve(market.discount_curve.times,
                tuple(np.asarray(market.discount_curve.discounts)*np.exp(.0025*np.asarray(market.discount_curve.times)))))),
            ("rates +25bp", replace(market, discount_curve=DiscountCurve(market.discount_curve.times,
                tuple(np.asarray(market.discount_curve.discounts)*np.exp(-.0025*np.asarray(market.discount_curve.times)))))),
        ]
        if market.dividends.events:
            scenarios.extend((f"cash dividends {percentage:+.0%} · counterfactual",
                replace(market, dividends=replace(market.dividends,
                    events=tuple(replace(event, amount=event.amount*(1+percentage))
                                 for event in market.dividends.events))))
                for percentage in (-.1, .1))
        else:
            sensitivity.append({"model": label, "scenario": "Cash-dividend amount sensitivity",
                "status": "not_applicable", "refitted": False, "max_price_change": None,
                "train_rmse": None, "test_rmse": None, "scaled_rmse": None,
                "note": "Explicit empty schedule: no dividend event is invented."})
        for name, candidate in scenarios:
            row = {"model": label, "scenario": name, "status": "numerical_failure", "refitted": False,
                "max_price_change": None, "train_rmse": None, "test_rmse": None, "scaled_rmse": None}
            try:
                values = evaluate_vanilla(fits[label], market=candidate, strike=quotes.strike.to_numpy(),
                    maturity=quotes.maturity.to_numpy(), is_call=quotes.is_call.to_numpy(), exercise_style="american", config=config)
                residual = np.asarray(values.prices)-quotes.market_price.to_numpy()
                scale = np.maximum((quotes.ask-quotes.bid).to_numpy()/2, .005)
                row.update(status="priced", max_price_change=float(np.max(abs(values.prices-baseline))),
                    train_rmse=float(np.sqrt(np.mean(residual[quotes.split.eq("train")]**2))),
                    test_rmse=float(np.sqrt(np.mean(residual[quotes.split.eq("heldout")]**2))),
                    scaled_rmse=float(np.sqrt(np.mean((residual/scale)**2))))
            except PricingError as exc:
                row["error"] = str(exc)
            sensitivity.append(row)
    tables.update(selected_contracts=quotes, fit_summary=pd.DataFrame(attempts), fitted_prices=prediction_table,
        fitted_parameters=pd.DataFrame(params), numerical_checks=check_table, input_sensitivity=pd.DataFrame(sensitivity))
    manifest.update(status="completed", numerical_config=asdict(config), max_nfev_per_attempt=budget,
        attempts=attempts, market=asdict(market), fit_attempt_count=len(attempts),
        selected_contracts_sha256=hashlib.sha256(pd.util.hash_pandas_object(quotes, index=False).values.tobytes()).hexdigest(),
        interpretation="Compare common models using training and held-out errors only after numerical checks; no fit is selected using held-out data.")
    if not prediction_table.empty:
        fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
        for label, group in prediction_table.groupby("model", sort=False):
            ax.plot(np.arange(len(group)), group.price_residual, "o-", label=label)
        ax.axhline(0, color="black", lw=.7)
        ax.set(xlabel="Identical contract order", ylabel="Model − observed midpoint ($)", title="Direct American calibration residuals")
        ax.legend(fontsize=7)
        figures["calibration_residuals"] = fig
    if not check_table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        for label, group in check_table.groupby("model", sort=False):
            axes[0].plot(group.axis, group.max_price_change, "o-", label=label)
            axes[1].plot(group.axis, group.max_delta_change, "o-", label=label)
        axes[0].set(ylabel="Maximum price change ($)", title="Independent price checks")
        axes[1].set(ylabel="Maximum delta change", title="Independent stock-delta checks")
        axes[-1].legend(fontsize=6)
        figures["numerical_refinement"] = fig
    if sensitivity:
        fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
        for label, group in tables["input_sensitivity"].groupby("model", sort=False):
            ax.plot(group.scenario, group.max_price_change, "o-", label=label)
        ax.set(ylabel="Maximum price change ($)", title="Input sensitivity with fitted parameters held fixed")
        ax.legend(fontsize=8)
        figures["input_sensitivity"] = fig


__all__ = ["pricing_research", "calibration_research", "spy_research", "EVIDENCE_VERSION"]
