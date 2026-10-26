"""European Fourier, BS/Heston finite differences and rough stopping prices.

The Fourier boundary retains explicit forwards/discounts. Finite differences
use observed spot, discount curves and declared cash events. Existing
``rvlab.models`` teaching APIs are preserved separately.
"""

from .api import price
from .black_scholes import black_scholes_price
from .types import (
    BlackScholesParams, HestonParams, RoughHestonParams,
    PricingConfig, PricingError,
)
from .heston import heston_cf, heston_price
from .variance_curve import (
    MeanReversionCurve, HestonCurveParams, RoughHestonCurveParams,
    heston_curve_cf, rough_heston_curve_cf, heston_curve_price, rough_heston_curve_price,
    forward_variance, integrated_forward_variance,
)
from .parameters import ModelParams, parameter_values, replace_parameters
from .optimization import OptimizerConfig
from .rough_heston import rough_heston_cf, rough_heston_price, solve_fractional_riccati
from .diagnostics import PricingDiagnostics, pricing_diagnostics
from .calibration import CalibrationResult, calibrate_surface
from .american_calls import price_american_calls, calibrate_american_call_surface
from .refinement import CalibrationStage, RefinedCalibrationResult, calibrate_surface_refined
from .vanilla_types import (DiscountCurve, CashDividend, DividendSchedule, SpotMarket,
    FDConfig, FrozenFDGrid, VanillaResult, UnsupportedModelError, NumericalConvergenceError)
from .vanilla import evaluate_vanilla
from .vanilla_calibration import VanillaCalibrationResult, calibrate_vanilla_surface
from .rough_lift import (ExponentialKernel, approximate_fractional_kernel,
    lifted_forward_variance, lifted_integrated_variance,
    lifted_heston_cf, lifted_heston_price)
from .rough_simulation import (RoughSimulationConfig, RoughSimulationResult,
    RoughSimulationError, simulate_rough_heston, european_payoff_samples)
from .rough_estimators import (RoughEuropeanControlConfig, RoughEuropeanEstimate,
    estimate_rough_european)
from .rough_stopping import (RoughAmericanConfig, RoughAmericanEstimate,
    RoughStoppingPolicy, train_rough_stopping_policy, value_rough_stopping_policy,
    estimate_rough_american)
from .rough_dual import (RoughDualConfig, RoughDualWeights, train_rough_dual_weights,
    apply_rough_dual_weights, value_rough_dual_policy)

__all__ = [
    "BlackScholesParams", "HestonParams", "RoughHestonParams", "ModelParams",
    "MeanReversionCurve", "HestonCurveParams", "RoughHestonCurveParams",
    "heston_curve_cf", "rough_heston_curve_cf", "heston_curve_price", "rough_heston_curve_price",
    "forward_variance", "integrated_forward_variance", "parameter_values", "replace_parameters",
    "OptimizerConfig",
    "PricingConfig", "PricingError", "price", "black_scholes_price",
    "heston_cf", "heston_price", "rough_heston_cf", "rough_heston_price",
    "solve_fractional_riccati", "CalibrationResult", "calibrate_surface",
    "price_american_calls", "calibrate_american_call_surface",
    "PricingDiagnostics", "pricing_diagnostics",
    "CalibrationStage", "RefinedCalibrationResult", "calibrate_surface_refined",
    "DiscountCurve", "CashDividend", "DividendSchedule", "SpotMarket", "FDConfig",
    "FrozenFDGrid", "VanillaResult", "UnsupportedModelError", "NumericalConvergenceError",
    "evaluate_vanilla", "VanillaCalibrationResult", "calibrate_vanilla_surface",
    "ExponentialKernel", "approximate_fractional_kernel",
    "lifted_forward_variance", "lifted_integrated_variance",
    "lifted_heston_cf", "lifted_heston_price", "RoughSimulationConfig",
    "RoughSimulationResult", "RoughSimulationError", "simulate_rough_heston",
    "RoughEuropeanControlConfig", "RoughEuropeanEstimate", "estimate_rough_european",
    "european_payoff_samples",
    "RoughAmericanConfig", "RoughAmericanEstimate", "RoughStoppingPolicy",
    "train_rough_stopping_policy", "value_rough_stopping_policy", "estimate_rough_american",
    "RoughDualConfig", "RoughDualWeights", "train_rough_dual_weights",
    "apply_rough_dual_weights", "value_rough_dual_policy",
]
