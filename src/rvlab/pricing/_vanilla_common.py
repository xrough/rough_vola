"""Shared finite-difference meshes and event clocks; no market-data inference."""
from __future__ import annotations

import numpy as np
from .vanilla_types import FrozenFDGrid


def _insert_spot_nodes(base, required):
    """Insert payoff/observation nodes without creating redundant tiny cells.

    A required strike next to an existing mesh node otherwise makes the
    terminal payoff cell almost zero-width.  Kink averaging then disappears
    and the mesh has an abrupt tiny/large spacing transition.  Remove only
    nearby *base* nodes; every distinct required spot/strike is retained.
    The quarter-cell exclusion leaves the outer domain endpoints intact.
    """
    base = np.asarray(base, float)
    required = np.unique(np.asarray(required, float))
    keep = np.ones(len(base), dtype=bool)
    for point in required:
        right = int(np.clip(np.searchsorted(base, point), 1, len(base)-1))
        spacing = base[right] - base[right-1]
        keep &= np.abs(base-point) > .25*spacing
    keep[0] = keep[-1] = True
    return np.unique(np.r_[base[keep], required])


def build_grid(market, strikes, config, *, variance=True, params=None):
    center = float(market.spot)
    maximum = config.spot_max_multiple * max(center, float(np.max(strikes)))
    width = config.spot_concentration*center
    # Keep the inner grid independent of outer-domain expansion. A domain
    # expansion appends outer points instead of stretching the old mesh.
    base_max = 8.*max(center, float(np.max(strikes)))
    inner_max = min(maximum, base_max)
    x = np.linspace(np.arcsinh(-center/width), np.arcsinh((inner_max-center)/width), config.spot_steps+1)
    spots = center + width*np.sinh(x)
    spots[0], spots[-1] = 0., inner_max
    # Apply the same insertion on the unchanged inner grid before extending
    # its outer domain, so a domain check preserves every existing node.
    spots = _insert_spot_nodes(spots, np.r_[center, np.asarray(strikes, float)])
    if maximum > inner_max:
        spacing = spots[-1]-spots[-2]
        spots = np.r_[spots, np.linspace(inner_max, maximum, int(np.ceil((maximum-inner_max)/spacing))+1)[1:]]
    variances = None
    if variance:
        inner_vmax = min(config.variance_max, 5.)
        variances = .02*np.sinh(np.linspace(0., np.arcsinh(inner_vmax/.02), config.variance_steps+1))
        variances[-1] = inner_vmax
        if config.variance_max > inner_vmax:
            n = int(np.ceil((config.variance_max-inner_vmax)/(variances[-1]-variances[-2])))
            variances = np.r_[variances, np.linspace(inner_vmax, config.variance_max, n+1)[1:]]
        # v0 is not inserted: calibration must not remesh around candidates.
    return FrozenFDGrid(spots, variances)


def time_grid(params, market, maturity, config):
    knots = list(market.discount_curve.times)
    knots += [e.time for e in market.dividends.events]
    if hasattr(params, "curve"):
        knots += list(params.curve.knot_times)
    exact = np.unique([0., maturity, *[t for t in knots if 0 < t < maturity]])
    base = np.linspace(0., maturity, config.time_steps+1)
    # Keep declared event values authoritative. linspace may land one ulp
    # beside an event, producing a spurious near-zero interval and an unstable
    # IT multiplier. Distinct declared events/knots are never merged here.
    indices = np.searchsorted(exact, base)
    distance = np.minimum(abs(base-exact[np.clip(indices, 0, len(exact)-1)]),
                          abs(base-exact[np.clip(indices-1, 0, len(exact)-1)]))
    tolerance = 4*np.finfo(float).eps*max(1., maturity)
    return np.sort(np.r_[exact, base[distance > tolerance]])


def zero_spot_put_value(market, t, maturity, exercise_style):
    d0 = market.discount_curve.discount(t)
    if exercise_style == "european":
        return market.discount_curve.discount(maturity)/d0
    times = [t, maturity, *[v for v in market.discount_curve.times if t < v < maturity]]
    return float(np.max(market.discount_curve.discount(times))/d0)


def derivative_coefficients(x):
    """Second-order central first/second derivatives on a nonuniform grid."""
    a, b = np.diff(x)[:-1], np.diff(x)[1:]
    first = (-b/(a*(a+b)), (b-a)/(a*b), a/(b*(a+b)))
    second = (2/(a*(a+b)), -2/(a*b), 2/(b*(a+b)))
    return first, second
