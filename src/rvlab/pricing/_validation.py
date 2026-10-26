"""Shared broadcasting and domain checks; no historical-data assumptions."""

from __future__ import annotations

import numpy as np


def broadcast_inputs(*, forward, discount, strike, maturity, is_call):
    """Return F, D, K, T, call flags broadcast to one shape."""
    flags = np.asarray(is_call)
    if flags.dtype.kind != "b":
        raise ValueError("is_call must contain boolean values")
    for name, values in (("forward", forward), ("discount", discount),
                         ("strike", strike), ("maturity", maturity)):
        array = np.asarray(values)
        if np.iscomplexobj(array) or array.dtype.kind == "b":
            raise ValueError(f"{name} must contain real numeric values, not complex numbers or booleans")
    try:
        f, d, k, t, flags = np.broadcast_arrays(
            np.asarray(forward, dtype=float), np.asarray(discount, dtype=float),
            np.asarray(strike, dtype=float), np.asarray(maturity, dtype=float), flags)
    except (ValueError, TypeError) as exc:
        raise ValueError("pricing inputs must be numeric and broadcast-compatible") from exc
    for name, values in (("forward", f), ("discount", d), ("strike", k)):
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"{name} must be finite and positive")
    if not np.all(np.isfinite(t) & (t >= 0)):
        raise ValueError("maturity must be finite and nonnegative")
    return f, d, k, t, flags


def scalar_or_array(value):
    """Return a float only when the broadcast result is zero-dimensional."""
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value
