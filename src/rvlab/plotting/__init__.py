"""Plotting helpers with a single house style. See `rvlab.plotting.charts`."""

from . import gallery
from .gallery import (
    acf_pacf, annotate_point, annotated_bar, calendar_heatmap, calibration_plot,
    cluster_map, colorblind_check, confusion_matrix_plot, correlation_contrast,
    decile_lift, drawdown_underwater, ecdf_plot, event_study, facet_grid,
    hexbin_plot, hist_kde, joint_plot, learning_curve_plot, line_with_reference,
    multi_series,
    pair_grid, palette_preview, pdp_grid, qq_plot, residual_panel, ridgeline,
    roc_pr_plot, rolling_band, save_publication, scatter_fit,
    validation_curve_plot, violin_strip,
)
from .charts import (
    correlation_heatmap, coverage_plot, cumulative_return_plot, distribution_plot,
    forecast_diagnostics, group_box_plot, ic_timeseries_plot, importance_plot,
    missingness_map, model_comparison_plot, path_fan, pnl_plot, scaling_fit_plot,
    smile_plot, sweep_heatmap, target_distribution_plot, term_structure_plot,
)
from .style import (
    C_ALT, C_BASELINE, C_MODEL, C_OBSERVED, C_WARN, DIVERGING, PALETTE,
    SEQUENTIAL, finish, savefig, set_rvlab_style,
)

__all__ = [
    "set_rvlab_style", "finish", "savefig", "PALETTE", "DIVERGING", "SEQUENTIAL",
    "C_OBSERVED", "C_BASELINE", "C_MODEL", "C_ALT", "C_WARN",
    "smile_plot", "term_structure_plot", "scaling_fit_plot", "distribution_plot",
    "forecast_diagnostics", "sweep_heatmap", "pnl_plot", "missingness_map",
    "correlation_heatmap", "path_fan", "coverage_plot", "target_distribution_plot",
    "group_box_plot", "ic_timeseries_plot", "cumulative_return_plot",
    "importance_plot", "model_comparison_plot", "gallery",
    "hist_kde", "ecdf_plot", "qq_plot", "violin_strip", "ridgeline",
    "scatter_fit", "hexbin_plot", "joint_plot", "pair_grid",
    "correlation_contrast", "cluster_map", "rolling_band", "calendar_heatmap",
    "acf_pacf", "drawdown_underwater", "event_study", "multi_series",
    "facet_grid", "annotated_bar", "decile_lift", "residual_panel",
    "learning_curve_plot", "validation_curve_plot", "calibration_plot",
    "roc_pr_plot", "confusion_matrix_plot", "pdp_grid", "palette_preview",
    "colorblind_check", "annotate_point", "save_publication",
    "line_with_reference",
]
