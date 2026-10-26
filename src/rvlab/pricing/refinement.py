"""Bounded calibration continuation over caller-declared numerical grids.

Every stage optimizes on one fixed grid. A later stage starts from the last
returned parameters, so a coarse-grid optimum is corrected on the finer grid
rather than merely repriced. Failed stages remain explicit diagnostic records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time

import pandas as pd

from .calibration import CalibrationResult, calibrate_surface
from .types import PricingConfig, PricingError
from .parameters import ModelParams
from .optimization import OptimizerConfig


@dataclass(frozen=True)
class CalibrationStage:
    """One fixed-grid attempt, including its starting parameters and failure."""

    config: PricingConfig
    initial_params: ModelParams
    result: CalibrationResult | None
    error: PricingError | None
    elapsed_seconds: float


@dataclass(frozen=True)
class RefinedCalibrationResult:
    """Ordered continuation history with no claim beyond the final stage.

    ``result`` is the last returned calibration result, even when a later
    numerical failure prevents acceptance. ``ready`` requires the final stage
    to have returned a ready result. Evaluation counts are summed over stages
    that returned results; a raised numerical failure has no reported count.
    """

    stages: tuple[CalibrationStage, ...]
    elapsed_seconds: float

    @property
    def result(self) -> CalibrationResult | None:
        return next((stage.result for stage in reversed(self.stages)
                     if stage.result is not None), None)

    @property
    def ready(self) -> bool:
        return bool(self.stages and self.stages[-1].result is not None
                    and self.stages[-1].result.ready)

    @property
    def total_nfev(self) -> int:
        return sum(stage.result.nfev for stage in self.stages
                   if stage.result is not None)

    @property
    def total_pricing_evaluations(self) -> int:
        """Full-surface price calls in returned stages, including Jacobians."""
        return sum(stage.result.pricing_evaluations for stage in self.stages
                   if stage.result is not None)


def calibrate_surface_refined(
    quotes: pd.DataFrame,
    initial_params: ModelParams,
    *,
    configs: Sequence[PricingConfig],
    forward,
    discount,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    max_nfev: int = 200,
    price_scale_floor: float = 0.005,
    stability_atol: float = 0.005,
    stability_rtol: float = 1e-4,
    optimizer: OptimizerConfig | None = None,
) -> RefinedCalibrationResult:
    """Warm-start a finite sequence of fixed-grid surface calibrations.

    ``max_nfev`` is the optimizer budget for each stage, not the whole sequence.
    Stop at the first stage whose optimizer converges and numerical refinement
    passes. Exhaustion and instability proceed to the next declared grid.
    ``PricingError`` is retained and the next stage retries from the last
    returned parameters, or the original start when no stage returned prices.
    Input validation exceptions propagate normally. No grids are generated and
    no held-out contracts are consulted; pass training contracts only.
    """
    if (not isinstance(configs, Sequence) or isinstance(configs, (str, bytes))
            or not configs):
        raise ValueError("configs must be a nonempty sequence of PricingConfig")
    settings = tuple(configs)
    if any(not isinstance(config, PricingConfig) for config in settings):
        raise ValueError("configs must contain only PricingConfig instances")

    started = time.perf_counter()
    stages = []
    current_params = initial_params
    for config in settings:
        stage_started = time.perf_counter()
        try:
            result = calibrate_surface(
                quotes, current_params, forward=forward, discount=discount,
                bounds=bounds, config=config, max_nfev=max_nfev,
                price_scale_floor=price_scale_floor,
                stability_atol=stability_atol, stability_rtol=stability_rtol,
                optimizer=optimizer,
            )
        except PricingError as exc:
            stages.append(CalibrationStage(
                config, current_params, None, exc,
                time.perf_counter() - stage_started,
            ))
            continue
        stages.append(CalibrationStage(
            config, current_params, result, None,
            time.perf_counter() - stage_started,
        ))
        current_params = result.params
        if result.ready:
            break
    return RefinedCalibrationResult(tuple(stages), time.perf_counter() - started)


__all__ = ["CalibrationStage", "RefinedCalibrationResult", "calibrate_surface_refined"]
