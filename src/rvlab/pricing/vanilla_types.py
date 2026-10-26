"""Explicit exercise, spot-market inputs and finite-difference controls.

All times are ACT/365 years from valuation. Cash events are strictly future;
at an ex-dividend observation the caller supplies the observed post-event spot.
The numerical cash model pays min(declared cash, pre-event spot), preserving
limited liability. It does not assume ordinary fixed-cash European parity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from numbers import Integral, Real
import math

import numpy as np

from .types import PricingError


class UnsupportedModelError(PricingError, NotImplementedError):
    """This exercise/model combination has no implemented pricing backend."""


class NumericalConvergenceError(PricingError):
    """A finite-difference solve failed to converge at its declared budget."""


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class DiscountCurve:
    """Positive discount factors, log-linearly interpolated without extrapolation.

    Include time zero with discount one. Forward rates are piecewise constant;
    discount factors need not decrease, so negative interest rates are supported.
    """
    times: tuple[float, ...]
    discounts: tuple[float, ...]

    def __post_init__(self):
        times = tuple(_real(t, "curve time", nonnegative=True) for t in self.times)
        discounts = tuple(_real(d, "discount", positive=True) for d in self.discounts)
        if len(times) != len(discounts) or len(times) < 2:
            raise ValueError("discount curve requires at least two aligned times and discounts")
        if times[0] != 0 or discounts[0] != 1 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("discount curve must start at (0, 1) with strictly increasing times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "discounts", discounts)

    @classmethod
    def flat(cls, rate, horizon):
        rate = _real(rate, "rate")
        horizon = _real(horizon, "horizon", positive=True)
        return cls((0., horizon), (1., math.exp(-rate*horizon)))

    def _times(self, t):
        a = np.asarray(t, float)
        if not np.isfinite(a).all() or (a < 0).any() or (a > self.times[-1] + 1e-12).any():
            raise ValueError("time is outside the supplied discount-curve coverage")
        return np.minimum(a, self.times[-1])

    def discount(self, t):
        a = self._times(t)
        out = np.exp(np.interp(a, self.times, np.log(self.discounts)))
        return float(out) if a.ndim == 0 else out

    def rate(self, t):
        a = self._times(t)
        i = np.clip(np.searchsorted(self.times, a, side="right") - 1, 0, len(self.times)-2)
        slopes = -np.diff(np.log(self.discounts)) / np.diff(self.times)
        out = slopes[i]
        return float(out) if a.ndim == 0 else out


@dataclass(frozen=True)
class CashDividend:
    time: float
    amount: float

    def __post_init__(self):
        object.__setattr__(self, "time", _real(self.time, "dividend time", positive=True))
        object.__setattr__(self, "amount", _real(self.amount, "dividend amount", nonnegative=True))


@dataclass(frozen=True)
class DividendSchedule:
    """Explicit expected events and complete expectation coverage in years.

    ``DividendSchedule((), coverage)`` explicitly declares no events through
    coverage. An unavailable schedule is not a valid empty schedule.
    """
    events: tuple[CashDividend, ...]
    coverage: float

    def __post_init__(self):
        events = tuple(self.events)
        coverage = _real(self.coverage, "dividend coverage", nonnegative=True)
        if any(not isinstance(e, CashDividend) for e in events):
            raise TypeError("events must contain CashDividend objects")
        if any(b.time <= a.time for a, b in zip(events, events[1:])):
            raise ValueError("dividend event times must be unique and strictly increasing")
        if any(e.time > coverage for e in events):
            raise ValueError("dividend events exceed the declared schedule coverage")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "coverage", coverage)


@dataclass(frozen=True)
class SpotMarket:
    spot: float
    discount_curve: DiscountCurve
    dividends: DividendSchedule

    def __post_init__(self):
        object.__setattr__(self, "spot", _real(self.spot, "spot", positive=True))
        if not isinstance(self.discount_curve, DiscountCurve):
            raise TypeError("discount_curve must be DiscountCurve")
        if not isinstance(self.dividends, DividendSchedule):
            raise TypeError("dividends must be an explicit DividendSchedule, not missing information")


@dataclass(frozen=True)
class FDConfig:
    """Starting numerical budgets, not accuracy promises (interval counts).

    ``spot_concentration * observed_spot`` is the sinh-mesh concentration
    width. Smaller widths allocate more nodes near spot and fewer to wings;
    refine them independently when wing prices or exercise boundaries matter.
    """
    spot_steps: int = 240
    variance_steps: int = 80
    time_steps: int = 300
    spot_max_multiple: float = 8.
    variance_max: float = 5.
    lcp_tolerance: float = 1e-10
    max_lcp_iterations: int = 100
    spot_concentration: float = .1

    def __post_init__(self):
        for name in ("spot_steps", "variance_steps", "time_steps", "max_lcp_iterations"):
            value = getattr(self, name)
            minimum = 8 if name == "spot_steps" else 3 if name == "variance_steps" else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("spot_max_multiple", "variance_max", "lcp_tolerance", "spot_concentration"):
            _real(getattr(self, name), name, positive=True)
        if self.spot_max_multiple <= 1:
            raise ValueError("spot_max_multiple must exceed 1")

    def refined(self, axis="joint"):
        if axis not in {"spot", "variance", "time", "domain", "joint"}:
            raise ValueError("unknown refinement axis")
        changes = {}
        if axis in {"spot", "joint"}:
            changes["spot_steps"] = 2*self.spot_steps
        if axis in {"variance", "joint"}:
            changes["variance_steps"] = 2*self.variance_steps
        if axis in {"time", "joint"}:
            changes["time_steps"] = 2*self.time_steps
        if axis == "domain":
            # More outer nodes avoid mistaking a coarser central grid for a
            # domain check. The central mesh is preserved by build_grid.
            changes.update(spot_max_multiple=2*self.spot_max_multiple,
                           variance_max=2*self.variance_max)
        return replace(self, **changes)


@dataclass(frozen=True)
class FrozenFDGrid:
    spot: np.ndarray
    variance: np.ndarray | None = None

    def __post_init__(self):
        for name in ("spot", "variance"):
            values = getattr(self, name)
            if values is None:
                continue
            a = np.array(values, dtype=float, copy=True)
            if a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all() or a[0] != 0 or np.any(np.diff(a) <= 0):
                raise ValueError(f"{name} grid must start at zero and increase strictly")
            a.setflags(write=False)
            object.__setattr__(self, name, a)


@dataclass(frozen=True)
class VanillaResult:
    prices: object
    deltas: object
    delta_left: object
    delta_right: object
    exercise_style: str
    config: FDConfig
    diagnostics: Mapping
    exercise_boundaries: object
    price_stable: bool | None = None
    delta_stable: bool | None = None
    grid: FrozenFDGrid | None = field(default=None, repr=False)

    @property
    def settings(self):
        from dataclasses import asdict
        return {"backend": "finite_difference", "exercise_style": self.exercise_style,
                **asdict(self.config)}


__all__ = ["DiscountCurve", "CashDividend", "DividendSchedule", "SpotMarket", "FDConfig",
           "FrozenFDGrid", "VanillaResult", "UnsupportedModelError", "NumericalConvergenceError"]
