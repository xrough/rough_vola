"""Metrics, statistical tests, parameter sweeps and the hedge backtest."""

from .classification import (
    best_threshold, calibration_report, decile_table, imbalance_report,
    threshold_sweep,
)
from .importance import group_importance, pipeline_feature_importance, pipeline_feature_names
from .ranking import (
    add_prediction_rank, blend, cross_sectional_zscore, daily_rank_ic,
    information_ratio, ranking_report, spread_return_series, spread_sharpe,
)
from .backtest import (
    bs_delta_policy, compare_hedge_policies, cost_sensitivity, hedge_error_stats,
    rough_delta_policy, smile_sensitivity,
)
from .metrics import (
    bias, classification_report_table, directional_accuracy, mae,
    out_of_fold_predictions, r2, regression_report, rmse, skill_vs_baseline,
)
from .sweeps import clear_cache, grid_points, run_grid, summarise_sweep, sweep_matrix
from .tests import (
    DMResult, benjamini_hochberg, block_bootstrap_ci, bonferroni,
    diebold_mariano, paired_permutation_test,
)

__all__ = [
    "rmse", "mae", "bias", "r2", "directional_accuracy", "skill_vs_baseline",
    "regression_report", "classification_report_table", "out_of_fold_predictions",
    "diebold_mariano", "DMResult", "block_bootstrap_ci", "benjamini_hochberg",
    "bonferroni", "paired_permutation_test",
    "run_grid", "grid_points", "summarise_sweep", "sweep_matrix", "clear_cache",
    "bs_delta_policy", "rough_delta_policy", "compare_hedge_policies",
    "cost_sensitivity", "hedge_error_stats", "smile_sensitivity",
    "add_prediction_rank", "daily_rank_ic", "information_ratio", "ranking_report",
    "spread_return_series", "spread_sharpe", "cross_sectional_zscore", "blend",
    "pipeline_feature_importance", "pipeline_feature_names", "group_importance",
    "imbalance_report", "threshold_sweep", "best_threshold", "calibration_report",
    "decile_table",
]
