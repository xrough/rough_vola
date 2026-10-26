"""Matched exercise-policy comparisons for one declared finite rough lift.

This module adds no pricing law.  It trains the existing all-state cashflow
LSM and Fourier-premium LSM implementations, then challenges the latter with
a state-only one-exercise-interval Bellman lookahead.  Every candidate is
replayed on the SAME independent physical trace and SAME centered martingale.
The common upper therefore does not change when a stopping rule changes.

The lookahead integrates an approximate next-date Snell envelope, not the
conditional cashflows of the candidate policy.  It is an admissible challenge,
not a policy-improvement theorem.  Its Gaussian quadrature is used only in an
exercise decision; dual martingales retain the existing exact discrete centers.
No historical sealed numerical source is modified by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from rvlab.pricing.rough_estimators import (
    RoughEuropeanControlConfig, _AffineSurrogate, _conditional_payoffs,
)
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_simulation import _time_grid
from rvlab.pricing.rough_stopping import (
    RoughAmericanConfig, RoughStoppingPolicy, _Continuation,
    _decision_continuation, train_rough_stopping_policy,
)
from rvlab.pricing.rough_stopping_fourier import (
    FourierStoppingPolicy, _affine_values, _expected_premium_values,
    fourier_policy_fingerprint, train_fourier_stopping_policy,
    value_fourier_stopping_policy,
)
from rvlab.pricing.types import RoughHestonParams
from rvlab.pricing.vanilla_types import (
    CashDividend, DiscountCurve, DividendSchedule, SpotMarket,
)
from ._pricing_rough_exercise_block import exercise_block_law


VERSION = "rough_matched_policy_numeric_v1"
POLICY_NAMES = ("cashflow_lsm", "fourier_premium_lsm", "bellman_lookahead")
CONTROL_NAMES = ("raw", "european", "european_premium")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def operator_grid(base):
    return _time_grid(base.params, base.market, base.continuation_times, base.config.steps)


@dataclass(frozen=True)
class ComparisonPolicy:
    """Explicit rule, including a frozen TRAINING-only inception decision."""
    kind: str
    base_policy: object
    exercise_at_inception: bool
    lookahead_order: int | None = None
    lookahead_blend: float = 1.
    inception_continuation: float | None = None

    def __post_init__(self):
        if self.kind not in POLICY_NAMES:
            raise ValueError("unknown matched-policy kind")
        expected = RoughStoppingPolicy if self.kind == "cashflow_lsm" else FourierStoppingPolicy
        if not isinstance(self.base_policy, expected):
            raise TypeError("policy kind and base continuation model differ")
        if not isinstance(self.exercise_at_inception, (bool, np.bool_)):
            raise TypeError("inception rule must be boolean")
        object.__setattr__(self, "exercise_at_inception", bool(self.exercise_at_inception))
        # Historical fingerprinting hashes repr(flag), which differs between a
        # NumPy Boolean and Python bool despite identical decisions. Normalize
        # metadata before ANY trace identity is created, not only at JSON save.
        if isinstance(self.base_policy.exercise_at_inception, np.bool_):
            object.__setattr__(self, "base_policy", replace(self.base_policy,
                exercise_at_inception=bool(self.base_policy.exercise_at_inception)))
        base = self.base_policy
        if (base.exercise_style != "american" or base.market.dividends.events
                or not np.array_equal(base.exercise_times, base.continuation_times)):
            raise ValueError("comparison requires American inception policies with no cash events")
        if self.kind == "bellman_lookahead":
            _integer(self.lookahead_order, "lookahead_order", 2)
            if (not np.isfinite(self.lookahead_blend) or not 0 < self.lookahead_blend <= 1
                    or self.inception_continuation is None
                    or not np.isfinite(self.inception_continuation)):
                raise ValueError("declare a finite lookahead blend and training-only inception value")
        elif self.lookahead_order is not None:
            raise ValueError("ordinary LSM policies cannot carry lookahead quadrature")


class StateBellmanLookahead:
    """One/two actual physical increments plus deterministic stock quadrature.

    Only (date, current stock, current full factor state) enters this operator.
    No simulated future path or held-out payoff is an input.
    """
    def __init__(self, base, *, order=16, batch_size=16):
        if not isinstance(base, FourierStoppingPolicy):
            raise TypeError("lookahead requires a Fourier-premium base policy")
        _integer(order, "order", 2)
        _integer(batch_size, "batch_size", 1)
        if (base.market.dividends.events or base.exercise_style != "american"
                or not np.array_equal(base.exercise_times, base.continuation_times)
                or base.premium_cap is None):
            raise ValueError("lookahead needs a no-dividend calendar and declared premium cap")
        self.base, self.order, self.batch_size = base, int(order), int(batch_size)
        self.times = base.exercise_times
        self.grid = operator_grid(base)
        self.discounts = base.market.discount_curve.discount(self.times)
        x, w = np.polynomial.hermite.hermgauss(int(order))
        self.nodes, self.weights = np.sqrt(2.)*x, w/np.sqrt(np.pi)
        self.affine = _AffineSurrogate(base.params, base.market, base.kernel,
            np.array([base.strike]), np.array([base.maturity]),
            np.array([base.is_call]), self.grid, base.control_config)
        self.maximum_mass_error = 0.
        self.evaluated_states = 0

    def conditional_values(self, index, stocks, factors):
        """Discounted-next-envelope expectation in current-date dollars."""
        _integer(index, "exercise index")
        s, u = np.asarray(stocks, float), np.asarray(factors, float)
        if (index >= len(self.times)-1 or s.ndim != 1 or len(s) == 0
                or u.shape != (len(s), len(self.base.kernel.nodes))
                or not np.isfinite(s).all() or np.any(s <= 0) or not np.isfinite(u).all()):
            raise ValueError("lookahead needs current positive stocks and all factor coordinates")
        left, right = self.times[index:index+2]
        a, b = np.searchsorted(self.grid, [left, right])
        if self.grid[a] != left or self.grid[b] != right or not 1 <= b-a <= 2:
            raise ValueError("lookahead supports one/two original physical increments per interval")
        result = np.empty(len(s))
        base = self.base
        ratio = self.discounts[index+1]/self.discounts[index]
        for begin in range(0, len(s), self.batch_size):
            end = min(begin+self.batch_size, len(s))
            law = exercise_block_law(base, u[begin:end], times=self.grid[a:b+1])
            new, p = law["new_factors"], law["probabilities"]
            self.maximum_mass_error = max(self.maximum_mass_error,
                float(np.max(np.abs(p.sum(axis=1)-1))))
            mean = np.log(s[begin:end])[:, None, None]+law["log_stock_shifts"]
            clock = law["orthogonal"]
            weighted = lambda v: np.sum(v.mean(axis=2)*p, axis=1)
            if index+1 == len(self.times)-1:
                result[begin:end] = ratio*weighted(
                    _conditional_payoffs(mean, clock, base.strike, base.is_call))
                continue
            model = base.continuations[index+1]
            coefficient = model.coefficients(new)
            full_coefficient = coefficient if model.knots else coefficient[..., None, :]
            euro_expected = _affine_values(self.affine, float(right), new,
                mean[..., None], clock[..., None])[..., 0]/self.discounts[index+1]
            premium_expected = _expected_premium_values(mean[..., None], clock[..., None],
                coefficient[:, :, None, None], strike=base.strike,
                stock_knots=model.knots, cap=base.premium_cap)[..., 0]
            logs = mean[..., None]+np.sqrt(clock)[..., None]*self.nodes
            n, branches, endpoints, q = logs.shape
            euro = _affine_values(self.affine, float(right), new,
                logs.reshape(n, branches, endpoints*q, 1),
                np.zeros((n, branches, endpoints*q, 1)))[..., 0].reshape(logs.shape)
            euro /= self.discounts[index+1]
            z = np.exp(logs)/base.strike
            knot_index = np.searchsorted(model.knots, z, side="right")
            c = full_coefficient[np.arange(n)[:, None, None, None],
                np.arange(branches)[None, :, None, None], knot_index]
            premium = np.clip(c[..., 0]+z*(c[..., 1]+z*c[..., 2]), 0., base.premium_cap)
            payoff = np.maximum((1 if base.is_call else -1)*(np.exp(logs)-base.strike), 0.)
            # Integrate smooth European and clipped quadratic terms analytically;
            # quadrature handles ONLY the remaining obstacle kink.
            excess = np.maximum(payoff-euro-premium, 0.)@self.weights
            result[begin:end] = ratio*weighted(euro_expected+premium_expected+excess)
        if not np.isfinite(result).all() or np.any(result < 0):
            raise FloatingPointError("Bellman conditional quadrature is nonfinite or negative")
        self.evaluated_states += len(s)
        return result


def build_lookahead_policy(base, *, order=16, blend=1., batch_size=16):
    """Freeze inception by deterministic integration BEFORE any selection paths."""
    evaluator = StateBellmanLookahead(base, order=order, batch_size=batch_size)
    if base.maturity == 0:
        initial = max((1 if base.is_call else -1)*(base.market.spot-base.strike), 0.)
        value = initial
    else:
        value = float(evaluator.conditional_values(0, np.array([base.market.spot]),
            np.zeros((1, len(base.kernel.nodes))))[0])
        initial = max((1 if base.is_call else -1)*(base.market.spot-base.strike), 0.)
    if not np.isfinite(blend) or not 0 < blend <= 1:
        raise ValueError("lookahead blend must be in (0,1]")
    # Inception is a distinct trained constant, not a decision from valuation data.
    # The full challenger uses this one-step value; half blends interpolate with
    # the base TRAINING estimate when that value is available.
    if blend != 1.:
        raise ValueError("this comparison declares one full Bellman challenger (blend=1)")
    return ComparisonPolicy("bellman_lookahead", base,
        bool(initial > 0 and initial >= value), int(order), float(blend), value)


def train_policy_set(params, market, kernel, *, strike, maturity, is_call=False,
                     config, lookahead_order=16, lookahead_batch_size=16,
                     cashflow_config=None, fourier_config=None, bellman_config=None):
    """Train the declared three rules with explicit training-stream identities.

    Cashflow LSM is the existing all-training-state regression, not an ITM-only
    refit. By default the fits share path count/seed and the challenger reuses
    the Fourier fit. Explicit per-method configs permit a PREDECLARED clock
    budget arm with different training path counts. Stock/factor basis, ridge,
    physical law and exercise dates must still match. Fourier fitting also
    computes the European residual proxy.
    """
    started = perf_counter()
    cashflow_config = config if cashflow_config is None else cashflow_config
    fourier_config = config if fourier_config is None else fourier_config
    bellman_config = fourier_config if bellman_config is None else bellman_config
    cash_started = perf_counter()
    cash = train_rough_stopping_policy(params, market, kernel, strike=strike,
        maturity=maturity, is_call=is_call, config=replace(cashflow_config, training_method="least_squares"))
    cash_seconds = perf_counter()-cash_started
    euro_started = perf_counter()
    fourier = train_fourier_stopping_policy(params, market, kernel, strike=strike,
        maturity=maturity, is_call=is_call, config=replace(fourier_config, training_method="fourier_premium"),
        control_config=fourier_config.premium_control, premium_cap=fourier_config.premium_cap)
    fourier_seconds = perf_counter()-euro_started
    challenger_started = perf_counter()
    bellman_base = fourier
    reused = replace(bellman_config, training_method="fourier_premium") == fourier.config
    if not reused:
        bellman_base = train_fourier_stopping_policy(params, market, kernel, strike=strike,
            maturity=maturity, is_call=is_call,
            config=replace(bellman_config, training_method="fourier_premium"),
            control_config=bellman_config.premium_control, premium_cap=bellman_config.premium_cap)
    challenger = build_lookahead_policy(bellman_base, order=lookahead_order,
        batch_size=lookahead_batch_size)
    challenger_seconds = perf_counter()-challenger_started
    policies = {"cashflow_lsm": ComparisonPolicy("cashflow_lsm", cash, cash.exercise_at_inception),
        "fourier_premium_lsm": ComparisonPolicy("fourier_premium_lsm", fourier,
            fourier.exercise_at_inception), "bellman_lookahead": challenger}
    validate_policy_set(policies)
    return policies, {"training_seconds": {"cashflow_lsm": cash_seconds,
        "fourier_premium_lsm": fourier_seconds,
        "bellman_lookahead_incremental": challenger_seconds,
        "bellman_lookahead_total": (fourier_seconds if reused else 0.)+challenger_seconds},
        "total_elapsed_seconds": perf_counter()-started,
        "training_paths": {name: p.base_policy.config.training_paths for name, p in policies.items()},
        "training_seeds": {name: p.base_policy.config.training_seed for name, p in policies.items()},
        "same_training_seed": len({p.base_policy.config.training_seed for p in policies.values()}) == 1,
        "same_training_random_stream": len({(p.base_policy.config.training_seed,
            p.base_policy.config.training_paths) for p in policies.values()}) == 1,
        "bellman_reuses_fourier_fit": reused, "same_training_factor_basis": True,
        "cashflow_regression_states": "all training states, not ITM-only",
        "cashflow_continuation_stabilization": "existing nonnegative vanilla-value surrogate cap",
        "fourier_continuation_stabilization": "European function plus premium clipped to [0,premium_cap]",
        "estimated_cashflow_memory_bytes": cash.diagnostics["estimated_memory_bytes"],
        "stored_coefficient_bytes": {name: _coefficient_bytes(p.base_policy)
            for name, p in policies.items()},
        "selection_or_final_paths_used": False}


def validate_policy_set(policies):
    """Reject accidental changes to economics, dynamics or exercise calendar."""
    if tuple(policies) != POLICY_NAMES or any(p.kind != k for k, p in policies.items()):
        raise ValueError("all three policies in declared order are required")
    reference = policies["fourier_premium_lsm"].base_policy
    for policy in policies.values():
        base = policy.base_policy
        for key in ("params", "market", "kernel", "strike", "maturity", "is_call", "exercise_style"):
            if getattr(base, key) != getattr(reference, key):
                raise ValueError("policy comparison must hold model/contract fixed: "+key)
        if (not np.array_equal(base.exercise_times, reference.exercise_times)
                or not np.array_equal(operator_grid(base), operator_grid(reference))
                or base.config.stock_integral != reference.config.stock_integral
                or base.config.stock_knots != reference.config.stock_knots
                or base.config.ridge != reference.config.ridge
                or base.config.bounded_surrogate != reference.config.bounded_surrogate):
            raise ValueError("policy comparison must hold physical law/calendar/basis/ridge fixed")
        if isinstance(base, FourierStoppingPolicy) and (base.control_config != reference.control_config
                or base.premium_cap != reference.premium_cap):
            raise ValueError("Fourier policies must share the European function and premium cap")
    return reference


def generate_shared_trace(policies, config, *, spots=None, forbidden_seeds=(),
                          control_policy=None, physical_steps=None):
    """Simulate ONCE with frozen decisions and actual-law control expectations.

    ``physical_steps`` changes only the valuation transition.  In particular,
    the European DECISION function remains on its original training operator
    grid; the control's European function and conditional centers use the
    actual valuation grid. The Bellman decision operator remains frozen too.
    """
    reference = validate_policy_set(policies)
    if control_policy is not None:
        if not isinstance(control_policy, FourierStoppingPolicy):
            raise TypeError("common control anchor must be a frozen FourierStoppingPolicy")
        _same_control_economics(reference, control_policy)
        reference = control_policy
    if (config.steps != reference.config.steps or config.exercise_steps != reference.config.exercise_steps
            or config.stock_integral != reference.config.stock_integral):
        raise ValueError("policy comparison cannot change the physical law or exercise calendar")
    banned = ({p.base_policy.config.training_seed for p in policies.values()}
        |{reference.config.training_seed}|set(forbidden_seeds))
    if config.valuation_seed in banned:
        raise ValueError("shared valuation stream overlaps training/selection")
    physical_steps = config.steps if physical_steps is None else _integer(
        physical_steps, "physical_steps", 1)
    actual_config = replace(config, steps=physical_steps)
    started = perf_counter()
    value = value_fourier_stopping_policy(reference, config=actual_config,
        spots=np.array([reference.market.spot]) if spots is None else spots, trace=True)
    trace = dict(value["trace"])
    trace["comparison_physical_steps"] = physical_steps
    trace["decision_operator_grid"] = operator_grid(reference)
    if np.array_equal(trace["actual_time_grid"], trace["decision_operator_grid"]):
        trace["decision_european"] = np.asarray(trace["european_proxy"]).copy()
    else:
        from ._pricing_rough_benchmark import decision_european
        trace["decision_european"] = decision_european(reference, trace,
            batch_size=config.batch_size)
    return trace, {"trace_generation_seconds": perf_counter()-started,
        "valuation_paths": config.valuation_paths, "valuation_seed": config.valuation_seed,
        "shared_physical_paths_for_all_policies": True,
        "stored_trace_array_bytes": sum(v.nbytes for v in trace.values()
            if isinstance(v, np.ndarray)),
        "control_reference_policy": "fourier_premium_lsm",
        "physical_steps": physical_steps,
        "decision_operator_steps": reference.config.steps,
        "frozen_entire_decision_function": True,
        "control_function_and_centers_use_actual_transition_grid": True,
        "martingale_centering": value["diagnostics"]["martingale_centering"]}


def _coefficient_bytes(base):
    groups = [base.continuations]
    if isinstance(base, FourierStoppingPolicy):
        groups.append(base.value_premiums)
    return sum(getattr(model, key).nbytes for group in groups for model in group
        for key in ("center", "scale", "beta"))


def _same_control_economics(reference, anchor):
    for key in ("params", "market", "kernel", "strike", "maturity", "is_call", "exercise_style", "control_config"):
        if getattr(reference, key) != getattr(anchor, key):
            raise ValueError("common control anchor changes the simulated economics/European function")
    if (not np.array_equal(operator_grid(reference), operator_grid(anchor))
            or not np.array_equal(reference.exercise_times, anchor.exercise_times)
            or reference.config.stock_integral != anchor.config.stock_integral):
        raise ValueError("common control anchor changes the dynamics/calendar")


def _validate_trace(trace, policies, forbidden_seeds, control_policy):
    reference = validate_policy_set(policies)
    seed = trace.get("seed")
    _integer(seed, "trace seed")
    forbidden = {_integer(s, "forbidden seed") for s in forbidden_seeds}
    anchor = reference if control_policy is None else control_policy
    _same_control_economics(reference, anchor)
    training = ({p.base_policy.config.training_seed for p in policies.values()}
        |{anchor.config.training_seed})
    if seed in training|forbidden:
        raise ValueError("valuation trace overlaps training/selection")
    times, eligible = np.asarray(trace["times"]), np.asarray(trace["eligible"])
    physical_steps = _integer(trace.get("comparison_physical_steps"), "trace physical steps", 1)
    actual_grid = _time_grid(reference.params, reference.market,
        reference.continuation_times, physical_steps)
    stocks, factors = np.asarray(trace["stocks"]), np.asarray(trace["factors"])
    payoff, euro = np.asarray(trace["discounted_payoffs"]), np.asarray(trace["european_proxy"])
    decision_euro = np.asarray(trace["decision_european"])
    if (not np.array_equal(times, reference.exercise_times) or eligible.dtype.kind != "b"
            or eligible.shape != times.shape or not eligible.all()
            or not np.array_equal(trace["actual_time_grid"], actual_grid)
            or not np.array_equal(trace["decision_operator_grid"], operator_grid(reference))
            or trace.get("policy_fingerprint") != fourier_policy_fingerprint(anchor)
            or trace.get("kernel_fingerprint") != reference.kernel.fingerprint
            or trace.get("strike") != reference.strike
            or stocks.ndim != 3 or stocks.shape[0] < 2 or stocks.shape[1] != len(times)
            or payoff.shape != stocks.shape or euro.shape != stocks.shape or decision_euro.shape != stocks.shape
            or factors.shape != stocks.shape[:2]+(len(reference.kernel.nodes),)
            or not all(np.isfinite(a).all() for a in (stocks, factors, payoff, euro, decision_euro))
            or np.any(stocks <= 0)):
        raise ValueError("shared trace must identify matching physical states/calendar/reference policy")
    discounts = reference.market.discount_curve.discount(times)
    expected = discounts[None, :, None]*np.maximum((1 if reference.is_call else -1)
        *(stocks-reference.strike), 0.)
    if (not np.array_equal(trace["discounts"], discounts)
            or not np.array_equal(trace["initial_spots"], stocks[0, 0])
            or not np.array_equal(stocks[:, 0], np.broadcast_to(trace["initial_spots"], stocks[:, 0].shape))
            or np.any(factors[:, 0] != 0)
            or not np.allclose(expected, payoff, rtol=0, atol=2e-12)):
        raise ValueError("trace payoff/discount/initial stock inputs differ")
    for name in ("european_martingale", "premium_martingale"):
        value = np.asarray(trace[name])
        if value.shape != stocks.shape or not np.isfinite(value).all() or np.any(value[:, 0] != 0):
            raise ValueError("shared control must be finite, aligned and start at zero")
    return reference


def stopping_rule(policy, trace, *, batch_size=16):
    """Own nonanticipating replay; realized future outcomes are never labels."""
    base = policy.base_policy
    stocks, factors = np.asarray(trace["stocks"]), np.asarray(trace["factors"])
    times, euro = np.asarray(trace["times"]), np.asarray(trace["decision_european"])
    n, dates, bumps = stocks.shape
    stops = np.full((n, bumps), dates-1, np.int64)
    decisions = np.zeros_like(stocks, dtype=bool)
    continuation = np.full_like(stocks, np.nan)
    if policy.exercise_at_inception:
        stops[:] = 0
        decisions[:, 0] = True
        return stops, decisions, continuation, {"lookahead_states": 0, "maximum_probability_error": 0.}
    alive = np.ones((n, bumps), bool)
    evaluator = (StateBellmanLookahead(base, order=policy.lookahead_order, batch_size=batch_size)
        if policy.kind == "bellman_lookahead" else None)
    discounts = base.market.discount_curve.discount(times)
    for j in range(1, dates-1):
        intrinsic = np.maximum((1 if base.is_call else -1)*(stocks[:, j]-base.strike), 0.)
        if policy.kind == "cashflow_lsm":
            future = _decision_continuation(base.continuations[j].value(stocks[:, j],
                factors[:, j], base.strike), stocks[:, j], base.market,
                strike=base.strike, maturity=base.maturity, time=float(times[j]),
                is_call=base.is_call, enabled=base.config.bounded_surrogate)
            continuation[:, j] = future
        else:
            q = np.clip(base.continuations[j].value(stocks[:, j], factors[:, j],
                base.strike), 0., base.premium_cap)
            future = euro[:, j]/discounts[j]+q
            if evaluator is not None:
                future = future.copy()
                for bump in range(bumps):
                    active = alive[:, bump] & (intrinsic[:, bump] > 0)
                    if np.any(active):
                        challenge = evaluator.conditional_values(j, stocks[active, j, bump],
                            factors[active, j])
                        future[active, bump] = ((1-policy.lookahead_blend)*future[active, bump]
                            +policy.lookahead_blend*challenge)
            continuation[:, j] = future
        exercise = (intrinsic > 0) & (intrinsic >= future)
        decisions[:, j] = exercise
        newly = alive & exercise
        stops[newly] = j
        alive &= ~newly
    decisions[:, -1] = True
    return stops, decisions, continuation, {"lookahead_states": 0 if evaluator is None else evaluator.evaluated_states,
        "maximum_probability_error": 0. if evaluator is None else evaluator.maximum_mass_error}


def evaluate_policy_set(trace, policies, *, forbidden_seeds=(), lookahead_batch_size=16,
                        control_policy=None):
    """Policy axis is fixed; retain all control variants without selecting one.

    Lower arrays have shape (paths, policies, stock bumps); upper arrays have
    shape (paths, stock bumps) and are bit-identical for every exercise policy.
    Raw/European/EP compare estimators of each FIXED policy separately.
    """
    _validate_trace(trace, policies, forbidden_seeds, control_policy)
    payoff = np.asarray(trace["discounted_payoffs"])
    n, _, bumps = payoff.shape
    path, bump = np.arange(n)[:, None], np.arange(bumps)[None]
    arrays = {"policy_names": np.asarray(POLICY_NAMES, dtype="U32")}
    elapsed, diagnostics, stop_values = {}, {}, []
    for name, policy in policies.items():
        started = perf_counter()
        stops, _, _, details = stopping_rule(policy, trace, batch_size=lookahead_batch_size)
        elapsed[name] = perf_counter()-started
        diagnostics[name] = details
        stop_values.append(stops)
    stops = np.stack(stop_values, axis=1)
    arrays["stop_indices"] = stops
    euro, premium = np.asarray(trace["european_martingale"]), np.asarray(trace["premium_martingale"])
    for name, martingale in (("raw", np.zeros_like(euro)), ("european", euro),
            ("european_premium", euro+premium)):
        adjusted = payoff-martingale
        upper = np.max(adjusted, axis=1)
        lowers = np.stack([adjusted[path, s, bump] for s in stop_values], axis=1)
        if np.any(lowers > upper[:, None]+2e-11):
            raise FloatingPointError("common dual maximum failed to dominate a policy endpoint")
        arrays[name+"_lower"], arrays[name+"_upper"] = lowers, upper
    return arrays, {"policy_evaluation_seconds": elapsed, "policy_diagnostics": diagnostics,
        "same_paths_same_upper_for_all_policies": True,
        "policy_selection_performed": False,
        "kernel_time_calendar_unchanged": np.array_equal(trace["actual_time_grid"], trace["decision_operator_grid"]),
        "kernel_and_exercise_calendar_unchanged": True,
        "frozen_entire_decision_function": True,
        "optimal_policy_or_price_accuracy_certified": False}


def _encode_base(base, prefix, arrays):
    for name in ("exercise_times", "continuation_times"):
        arrays[prefix+name] = np.asarray(getattr(base, name))
    groups = [base.continuations]
    if isinstance(base, FourierStoppingPolicy):
        groups.append(base.value_premiums)
    models = []
    for group_index, group in enumerate(groups):
        declaration = []
        for index, model in enumerate(group):
            if type(model) is not _Continuation:
                raise TypeError("comparison seals support only explicit polynomial continuations")
            item = {"knots": list(model.knots), "rank": int(model.rank),
                "condition": float(model.condition)}
            for name in ("center", "scale", "beta"):
                key = f"{prefix}g{group_index}_{index}_{name}"
                arrays[key] = np.asarray(getattr(model, name))
                item[name] = key
            declaration.append(item)
        models.append(declaration)
    result = {"class": type(base).__name__, "params": asdict(base.params),
        "market": asdict(base.market), "kernel": asdict(base.kernel),
        "strike": base.strike, "maturity": base.maturity, "is_call": bool(base.is_call),
        "exercise_style": base.exercise_style, "config": asdict(base.config),
        "exercise_at_inception": bool(base.exercise_at_inception),
        "exercise_times": prefix+"exercise_times", "continuation_times": prefix+"continuation_times",
        "models": models}
    if isinstance(base, FourierStoppingPolicy):
        result.update(control_config=asdict(base.control_config), premium_cap=base.premium_cap)
    return result


def _encode(policies):
    validate_policy_set(policies)
    arrays, records = {}, {}
    for index, (name, policy) in enumerate(policies.items()):
        records[name] = {"kind": policy.kind, "exercise_at_inception": bool(policy.exercise_at_inception),
            "lookahead_order": policy.lookahead_order, "lookahead_blend": policy.lookahead_blend,
            "inception_continuation": policy.inception_continuation,
            "base": _encode_base(policy.base_policy, f"p{index}_", arrays)}
    return {"version": VERSION, "policy_order": list(POLICY_NAMES), "policies": records}, arrays


def policy_set_fingerprint(policies):
    """Logical identity includes inception, all coefficient bytes and settings."""
    payload, arrays = _encode(policies)
    digest = hashlib.sha256(_canonical(payload).encode())
    for key, array in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_policy_set(policies, folder):
    """JSON plus numeric NPZ; supported classes are explicit, never pickled."""
    payload, arrays = _encode(policies)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/"coefficients.npz", **arrays)
    payload.update(policy_set_fingerprint=policy_set_fingerprint(policies),
        coefficients_sha256=_sha(folder/"coefficients.npz"))
    (folder/"policies.json").write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False)+"\n")
    return payload


def _decode_base(payload, arrays):
    options = dict(payload["config"])
    options["stock_knots"] = tuple(options["stock_knots"])
    if options.get("dual_proxy_knots") is not None:
        options["dual_proxy_knots"] = tuple(options["dual_proxy_knots"])
    options["premium_control"] = RoughEuropeanControlConfig(**options["premium_control"])
    if options.get("dual_config") is not None:
        raise ValueError("comparison seal does not support adaptive dual fits")
    config = RoughAmericanConfig(**options)
    m = payload["market"]
    market = SpotMarket(m["spot"], DiscountCurve(**m["discount_curve"]),
        DividendSchedule(tuple(CashDividend(**event) for event in m["dividends"]["events"]),
            m["dividends"]["coverage"]))
    groups = [tuple(_Continuation(arrays[row["center"]], arrays[row["scale"]],
        arrays[row["beta"]], tuple(row["knots"]), row["rank"], row["condition"])
        for row in group) for group in payload["models"]]
    args = (RoughHestonParams(**payload["params"]), market, ExponentialKernel(**payload["kernel"]),
        payload["strike"], payload["maturity"], payload["is_call"], payload["exercise_style"],
        arrays[payload["exercise_times"]], arrays[payload["continuation_times"]], groups[0])
    if payload["class"] == "RoughStoppingPolicy" and len(groups) == 1:
        return RoughStoppingPolicy(*args, payload["exercise_at_inception"], config,
            {"loaded_from_explicit_comparison_seal": True})
    if payload["class"] == "FourierStoppingPolicy" and len(groups) == 2:
        return FourierStoppingPolicy(*args, groups[1], config,
            RoughEuropeanControlConfig(**payload["control_config"]),
            payload["exercise_at_inception"], {"loaded_from_explicit_comparison_seal": True},
            payload["premium_cap"])
    raise ValueError("unknown explicit comparison policy class")


def load_policy_set(folder):
    folder = Path(folder)
    payload = json.loads((folder/"policies.json").read_text())
    if (payload["version"] != VERSION or payload["policy_order"] != list(POLICY_NAMES)
            or payload["coefficients_sha256"] != _sha(folder/"coefficients.npz")):
        raise ValueError("comparison policy seal changed")
    with np.load(folder/"coefficients.npz", allow_pickle=False) as arrays:
        policies = {}
        for name in POLICY_NAMES:
            record = payload["policies"][name]
            policies[name] = ComparisonPolicy(record["kind"], _decode_base(record["base"], arrays),
                record["exercise_at_inception"], record["lookahead_order"],
                record["lookahead_blend"], record["inception_continuation"])
    if policy_set_fingerprint(policies) != payload["policy_set_fingerprint"]:
        raise ValueError("loaded comparison policy coefficients/settings differ from seal")
    return policies


__all__ = ["VERSION", "POLICY_NAMES", "CONTROL_NAMES", "ComparisonPolicy",
    "StateBellmanLookahead", "build_lookahead_policy", "train_policy_set",
    "validate_policy_set", "generate_shared_trace", "stopping_rule", "evaluate_policy_set",
    "operator_grid", "policy_set_fingerprint", "save_policy_set", "load_policy_set"]
