"""Explicit bounded optimizer coordinates, independent of physical parameters.

Curve calibration can have a curved kappa/theta product ridge. Scaling Jacobian
columns does not straighten that ridge. Monotone ``asinh`` and ``log1p`` maps
are optional reparameterizations of exactly the same physical search box; zero
variance and forcing levels remain attainable. Neither changes price weights,
model coefficients, the objective, or numerical accuracy tolerances.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np

from .types import PricingError


@dataclass(frozen=True)
class OptimizerConfig:
    """Controls for the common SciPy trust-region reflective workflow.

    ``physical`` preserves the original behavior. ``asinh``/``log1p`` act on
    nonnegative parameters using declared characteristic scales: variance
    coordinates .0025, kappa .01, vol-of-vol .01, volatility .01 and Hurst .01.
    Rho remains linear. Scales do not restrict the physical parameter domain.
    Fixed coordinates are excluded. ``optimality`` is in optimizer coordinates;
    inspect ``physical_optimality`` to compare parameterizations of one box.
    No raw relative finite-difference step is exposed: it can erase derivatives
    at nearly-zero curve levels. Transformed fits differentiate in physical
    coordinates with absolute floors before applying the exact chain rule.
    """

    parameterization: str = "physical"
    jacobian: str = "2-point"
    ftol: float | None = None
    xtol: float | None = 1e-9
    gtol: float | None = 1e-9

    def __post_init__(self):
        if self.parameterization not in ("physical", "asinh", "log1p"):
            raise ValueError("parameterization must be physical, asinh or log1p")
        if self.jacobian not in ("2-point", "3-point"):
            raise ValueError("jacobian must be 2-point or 3-point")
        for name in ("ftol", "xtol", "gtol"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not math.isfinite(value) or value <= np.finfo(float).eps):
                raise ValueError(f"{name} must be None or finite and greater than machine epsilon")
        if self.ftol is self.xtol is self.gtol is None:
            raise ValueError("at least one optimizer tolerance must be enabled")


def _scale(name):
    return (.0025 if name == "v0" or name == "theta" or name.startswith("theta_")
            else .01 if name in ("kappa", "vol_of_vol", "volatility", "hurst") else 1.)


class _Coordinates:
    """One-to-one maps for free coordinates and their physical box."""

    def __init__(self, names, lower, upper, optimizer):
        self.names = tuple(names)
        self.optimizer = optimizer
        self.lower = np.asarray([lower[name] for name in names], float)
        self.upper = np.asarray([upper[name] for name in names], float)
        self.transformed = np.array([name != "rho" for name in names], bool)
        if optimizer.parameterization == "physical":
            self.transformed[:] = False
        self.scales = np.array([_scale(name) if transformed else 1.
                                for name, transformed in zip(names, self.transformed)])
        self.bounds = self.encode(self.lower), self.encode(self.upper)

    def encode(self, physical):
        values = np.asarray(physical, float).copy()
        selected = self.transformed
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.optimizer.parameterization == "asinh":
                values[selected] = np.arcsinh(values[selected]/self.scales[selected])
            elif self.optimizer.parameterization == "log1p":
                values[selected] = np.log1p(values[selected]/self.scales[selected])
        if not np.all(np.isfinite(values)):
            raise ValueError("parameter bounds exceed the finite optimizer-coordinate range")
        return values

    def decode(self, coordinates):
        values = np.asarray(coordinates, float).copy()
        selected = self.transformed
        with np.errstate(over="ignore", invalid="ignore"):
            if self.optimizer.parameterization == "asinh":
                values[selected] = self.scales[selected]*np.sinh(values[selected])
            elif self.optimizer.parameterization == "log1p":
                values[selected] = self.scales[selected]*np.expm1(values[selected])
        if not np.all(np.isfinite(values)):
            raise ValueError("optimizer coordinates produced nonfinite physical parameters")
        # Inverse transforms can exceed an exact endpoint by a few ulps.
        # SciPy has already constrained the optimizer coordinates to their box.
        return np.clip(values, self.lower, self.upper)

    def derivative(self, coordinates):
        derivative = np.ones(len(self.names))
        selected = self.transformed
        if self.optimizer.parameterization == "asinh":
            derivative[selected] = self.scales[selected]*np.cosh(np.asarray(coordinates)[selected])
        elif self.optimizer.parameterization == "log1p":
            derivative[selected] = self.scales[selected]*np.exp(np.asarray(coordinates)[selected])
        return derivative

    def settings(self):
        return {**asdict(self.optimizer), "method": "trf", "x_scale": "jac", "loss": "linear",
                "coordinate_scales": dict(zip(self.names, self.scales.tolist())),
                "coordinate_bounds": {name: (float(lo), float(hi)) for name, lo, hi
                                      in zip(self.names, *self.bounds)},
                "finite_difference_space": "physical",
                "finite_difference_step": "max(abs(parameter),1) * eps**(1/2 or 1/3)"}

    def physical_optimality(self, coordinates, jacobian, residuals):
        """TRF-style KKT measure in physical coordinates for one fixed box.

        The gradient of .5*sum(residual**2) is multiplied by the distance to
        the feasible bound in its descent direction, then its infinity norm
        is taken. This is zero for a constrained stationary point. The final
        optimizer Jacobian is converted by the chain rule; no held-out quotes
        or pricing refinements enter this measure.
        """
        physical = self.decode(coordinates)
        gradient = (np.asarray(jacobian).T @ np.asarray(residuals))/self.derivative(coordinates)
        distance = np.where(gradient < 0, self.upper-physical, physical-self.lower)
        return float(np.max(np.abs(gradient)*distance)) if len(gradient) else 0.


def _finite_difference_jacobian(function, values, *, lower, upper, method, base=None):
    """Differentiate a vector function inside an explicit finite parameter box.

    Two-point steps use sqrt(machine epsilon), and three-point steps use its
    cube root, multiplied by ``max(abs(x_j), 1)``. This deliberately avoids a
    vanishing step for a parameter close to zero. The returned details record
    actual representable steps and whether a bound required a one-sided rule.
    """
    if method not in {"2-point", "3-point"}:
        raise ValueError("method must be '2-point' or '3-point'")
    x = np.asarray(values, float)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (x.ndim != 1 or x.size == 0 or lo.shape != x.shape or hi.shape != x.shape
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi)) or np.any(lo >= hi)
            or np.any(x < lo) or np.any(x > hi)):
        raise ValueError("values require aligned, finite, nonempty, strictly ordered bounds")

    def evaluate(point):
        output = np.asarray(function(point), float)
        if output.ndim != 1 or output.size == 0 or not np.all(np.isfinite(output)):
            raise PricingError("Jacobian evaluation must return a finite nonempty residual vector")
        return output

    base = evaluate(x) if base is None else np.asarray(base, float)
    exponent = 0.5 if method == "2-point" else 1 / 3
    step_factor = np.finfo(float).eps ** exponent
    columns, details = [], []
    for index, value in enumerate(x):
        nominal = step_factor * max(abs(value), 1.0)
        forward_room, backward_room = hi[index] - value, value - lo[index]
        if method == "2-point":
            direction = 1 if value >= 0 else -1
            preferred_room = forward_room if direction > 0 else backward_room
            other_room = backward_room if direction > 0 else forward_room
            if preferred_room < nominal and other_room >= nominal:
                direction *= -1
                preferred_room = other_room
            elif preferred_room < nominal and other_room > preferred_room:
                direction *= -1
                preferred_room = other_room
            step = direction * min(nominal, preferred_room)
            shifted = x.copy()
            shifted[index] = np.clip(value + step, lo[index], hi[index])
            actual = shifted[index] - value
            if actual == 0:
                raise ValueError(f"parameter {index} has no representable finite-difference step")
            perturbed = evaluate(shifted)
            if perturbed.shape != base.shape:
                raise PricingError("Jacobian evaluations changed residual shape")
            column = (perturbed - base) / actual
            scheme = "forward" if actual > 0 else "backward"
            steps = (actual,)
        else:
            if forward_room >= nominal and backward_room >= nominal:
                offsets = (nominal, -nominal)
                scheme = "central"
            elif forward_room >= 2 * nominal:
                offsets = (nominal, 2 * nominal)
                scheme = "forward"
            elif backward_room >= 2 * nominal:
                offsets = (-nominal, -2 * nominal)
                scheme = "backward"
            elif forward_room >= backward_room:
                offsets = (forward_room / 2, forward_room)
                scheme = "forward"
            else:
                offsets = (-backward_room / 2, -backward_room)
                scheme = "backward"
            first, second = x.copy(), x.copy()
            first[index] = np.clip(value + offsets[0], lo[index], hi[index])
            second[index] = np.clip(value + offsets[1], lo[index], hi[index])
            a, b = first[index] - value, second[index] - value
            if a == 0 or b == 0 or a == b:
                raise ValueError(f"parameter {index} has no representable three-point stencil")
            fa, fb = evaluate(first), evaluate(second)
            if fa.shape != base.shape or fb.shape != base.shape:
                raise PricingError("Jacobian evaluations changed residual shape")
            # Nonuniform three-point formula uses differences from the base
            # to reduce cancellation of a large constant residual component.
            column = (b * b * (fa - base) - a * a * (fb - base)) / (a * b * (b - a))
            steps = (a, b)
        if not np.all(np.isfinite(column)):
            raise PricingError("Finite differences produced a nonfinite Jacobian column")
        columns.append(column)
        details.append({"scheme": scheme, "nominal_absolute_step": float(nominal),
                        "actual_steps": tuple(float(s) for s in steps)})
    return np.column_stack(columns), details


__all__ = ["OptimizerConfig"]
