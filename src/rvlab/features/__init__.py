"""Feature builders: smile observables, realized vol, time-series, roughness."""

from .encoding import (
    CVTargetEncoder, CyclicalEncoder, FrequencyEncoder, RareCategoryGrouper,
)
from .joins import (
    aggregate_and_join, asof_join, dates_in_columns_to_long, flatten_multiindex,
    interval_join, long_to_wide, merge_report, reconcile, safe_merge,
)
from .panel import (
    add_calendar_features, add_cross_sectional, add_entity_rolling,
    add_group_relative, add_horizon_returns, add_price_geometry,
    build_panel_features, make_panel_target, winsorize_by_date,
)
from .realized import (
    add_har_components, add_vrp, log_returns, realized_variance,
    realized_variance_blocks, realized_vol, variance_risk_premium,
)
from .roughness import (
    HurstEstimate, hurst_structure_function, skew_scaling_panel,
    skew_term_structure_fit, summarise_skew_scaling,
)
from .smile import (
    add_moneyness_features, add_structural_coefficients, add_tenor_bucket,
    cross_section, extract_from_chain, panel_from_chain, summarise_panel,
    widest_cross_section,
)
from .timeseries import (
    add_diffs, add_expanding, add_lags, add_rolling, build_design_matrix,
    make_target, resample_panel, supervised_frame,
)

__all__ = [
    "add_structural_coefficients", "add_moneyness_features", "add_tenor_bucket",
    "cross_section", "widest_cross_section", "summarise_panel",
    "extract_from_chain", "panel_from_chain",
    "log_returns", "realized_variance", "realized_vol", "add_har_components",
    "variance_risk_premium", "add_vrp", "realized_variance_blocks",
    "add_lags", "add_diffs", "add_rolling", "add_expanding", "make_target",
    "supervised_frame", "resample_panel", "build_design_matrix",
    "HurstEstimate", "hurst_structure_function", "skew_term_structure_fit",
    "skew_scaling_panel", "summarise_skew_scaling",
    "add_calendar_features", "add_price_geometry", "add_horizon_returns",
    "add_entity_rolling", "add_cross_sectional", "add_group_relative",
    "make_panel_target", "winsorize_by_date", "build_panel_features",
    "merge_report", "safe_merge", "reconcile", "asof_join", "interval_join",
    "aggregate_and_join", "flatten_multiindex", "dates_in_columns_to_long",
    "long_to_wide",
    "CVTargetEncoder", "RareCategoryGrouper", "FrequencyEncoder", "CyclicalEncoder",
]
