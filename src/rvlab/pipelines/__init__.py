"""scikit-learn transformers, leakage-safe splitters and pipeline factories."""

from .build import (
    available_regressors, make_classification_pipeline, make_column_transformer,
    make_panel_pipeline, make_regression_pipeline, rough_hurst_grid,
)
from .inference import (
    RollingHistoryBuffer, align_feature_columns, check_inference_schema,
    fill_missing_predictions, rank_for_submission,
)
from .splits import (
    GroupTimeSeriesSplit, PurgedTimeSeriesSplit, WalkForwardSplit,
    describe_date_folds, describe_folds, purged_date_splits,
)
from .transformers import (
    ColumnSelector, LagFeatures, LogTransform, StructuralCoefficients, Winsorizer,
)

__all__ = [
    "PurgedTimeSeriesSplit", "GroupTimeSeriesSplit", "WalkForwardSplit",
    "describe_folds", "purged_date_splits", "describe_date_folds",
    "align_feature_columns", "check_inference_schema", "RollingHistoryBuffer",
    "rank_for_submission", "fill_missing_predictions", "Winsorizer", "StructuralCoefficients", "LagFeatures",
    "LogTransform", "ColumnSelector", "make_regression_pipeline",
    "make_classification_pipeline", "make_column_transformer",
    "available_regressors", "rough_hurst_grid", "make_panel_pipeline",
]
