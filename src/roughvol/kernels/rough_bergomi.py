from __future__ import annotations

import numpy as np

from roughvol.types import ArrayF


def rough_bergomi_midpoint_weights(t: ArrayF, hurst: float) -> ArrayF:
    """Build midpoint-quadrature weights for the rBergomi Volterra driver."""
    H = float(hurst)
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5) for rough Bergomi.")

    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or t[0] != 0.:
        raise ValueError("t must be a 1D time grid with at least two points.")

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("Time grid must be strictly increasing.")

    mids = 0.5 * (t[:-1] + t[1:])
    end_times = t[1:]
    n_steps = len(dt)

    weights = np.zeros((n_steps, n_steps), dtype=float)
    scale = np.sqrt(2.0 * H)
    exponent = H - 0.5

    for j in range(n_steps):
        tau = end_times[j] - mids[: j + 1]
        weights[j, : j + 1] = scale * np.power(tau, exponent)

    return weights
