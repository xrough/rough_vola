"""Leakage-resistant building blocks for competition-style analyses.

The notebook-facing API in this module is intentionally small.  A
:class:`ProblemSpec` states the statistical design, :func:`make_folds` routes
that design to the appropriate cross-validation scheme, and every operation
after that consumes the same validated folds.  This makes the important
assumptions inspectable instead of scattering them across notebook cells.

Nothing here chooses a sophisticated model.  The module owns the less
glamorous contracts that make model comparisons trustworthy: fold integrity,
out-of-fold predictions, a fold-local baseline, train/test drift detection,
submission alignment, and reproducible run identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__ as rvlab_version


Geometry = Literal["iid", "grouped", "time", "panel"]
Task = Literal["regression", "binary", "ranking"]
RankingTargetOrder = Literal["higher_is_better", "lower_is_better"]
TestTimeRelation = Literal["future", "unrestricted"]
Fold = tuple[np.ndarray, np.ndarray]

_GEOMETRIES = frozenset({"iid", "grouped", "time", "panel"})
_TASKS = frozenset({"regression", "binary", "ranking"})
_RANKING_TARGET_ORDERS = frozenset({"higher_is_better", "lower_is_better"})
_TEST_TIME_RELATIONS = frozenset({"future", "unrestricted"})
_LEDGER_COLUMNS = frozenset({"row_position", "fold", "y_true", "prediction"})
_DEFAULT_METRICS = {
    "regression": "root_mean_squared_error",
    "binary": "roc_auc",
    "ranking": "spearman",
}
SUPPORTED_METRICS: dict[str, dict[str, str]] = {
    "regression": {
        "root_mean_squared_error": "minimize",
        "mean_absolute_error": "minimize",
        "r2": "maximize",
    },
    "binary": {
        "roc_auc": "maximize",
        "average_precision": "maximize",
        "log_loss": "minimize",
        "brier": "minimize",
    },
    "ranking": {"spearman": "maximize"},
}
_SUBMISSION_TRANSFORMS = frozenset({"identity", "rank_descending_zero"})


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The immutable contract that determines an analysis design.

    Parameters
    ----------
    geometry:
        ``iid`` for exchangeable rows, ``grouped`` when an entity may occur on
        several rows, ``time`` for chronological observations, or ``panel``
        for several entities observed at each timestamp.
    task:
        ``regression``, ``binary``, or ``ranking``.  Ranking requires grouped
        or panel geometry because the corresponding group/time role defines a
        multi-row query.
    target:
        Training-frame target column.
    id_columns:
        Columns that uniquely identify and order submission rows.  They are
        optional for modelling but required by :func:`validate_submission`.
    time_column / group_column:
        Required by the corresponding geometry.  A panel requires both.
    test_time_relation:
        Required for time and panel designs.  ``future`` asserts that every
        test timestamp follows every training timestamp; ``unrestricted``
        records that the competition permits overlap or an arbitrary test
        window instead of silently waiving that check.
    label_horizon / feature_lookback:
        The forward label span and the past-only history needed at inference,
        both in distinct timestamps.  The label horizon determines the
        automatic purge.  A causal lookback uses only information already
        available at each row, so it is recorded but does not discard training
        rows.
    gap:
        An optional explicit purge in distinct timestamps.  It may be larger,
        but never smaller, than the declared label horizon.
    metric / metric_direction:
        A supported metric and its direction.  The direction is checked rather
        than guessed so a manifest can never say one objective while scoring
        another.
    submission_transform:
        ``identity`` for ordinary regression/probability submissions, or
        ``rank_descending_zero`` for competitions requiring unique integer
        ranks within each ranking query (best oriented score receives rank
        zero).  Equal predictions are resolved deterministically in source-row
        order (pandas ``method='first'``), and validation uses that same rule.
    ranking_target_order:
        Required for ranking tasks.  It states whether larger or smaller
        target values mean a better item, and is used identically by
        validation scoring and zero-best submission postprocessing.

    Notes
    -----
    The class is frozen deliberately: changing the split contract halfway
    through a run must require constructing a new spec and therefore a new
    manifest.
    """

    geometry: Geometry
    task: Task
    target: str = "target"
    id_columns: tuple[str, ...] = ()
    time_column: str | None = None
    group_column: str | None = None
    test_time_relation: TestTimeRelation | None = None
    n_splits: int = 5
    random_state: int = 42
    label_horizon: int = 0
    feature_lookback: int = 0
    gap: int | None = None
    positive_label: str | int | float | bool = 1
    metric: str | None = None
    metric_direction: Literal["minimize", "maximize"] | None = None
    prediction_column: str | None = None
    submission_transform: Literal["identity", "rank_descending_zero"] = "identity"
    ranking_target_order: RankingTargetOrder | None = None

    def __post_init__(self) -> None:
        # Accept a list at a notebook boundary but freeze its value as a tuple.
        object.__setattr__(self, "id_columns", tuple(self.id_columns))

        if self.geometry not in _GEOMETRIES:
            raise ValueError(
                f"geometry must be one of {sorted(_GEOMETRIES)}, got {self.geometry!r}")
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {sorted(_TASKS)}, got {self.task!r}")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty column name")
        if any(not isinstance(col, str) or not col for col in self.id_columns):
            raise ValueError("every id column must be a non-empty string")
        if len(set(self.id_columns)) != len(self.id_columns):
            raise ValueError("id_columns contains duplicates")
        if not isinstance(self.n_splits, int) or isinstance(self.n_splits, bool):
            raise TypeError("n_splits must be an integer")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise TypeError("random_state must be an integer")

        for name in ("label_horizon", "feature_lookback"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.gap is not None and (
            not isinstance(self.gap, int) or isinstance(self.gap, bool) or self.gap < 0
        ):
            raise ValueError("gap must be None or a non-negative integer")

        if self.geometry == "iid":
            if self.time_column is not None or self.group_column is not None:
                raise ValueError("iid geometry must not declare time_column or group_column")
            if self.test_time_relation is not None:
                raise ValueError("iid geometry must not declare test_time_relation")
        elif self.geometry == "grouped":
            if not self.group_column:
                raise ValueError("grouped geometry requires group_column")
            if self.time_column is not None:
                raise ValueError(
                    "grouped geometry does not preserve time; use panel when time matters")
            if self.test_time_relation is not None:
                raise ValueError("grouped geometry must not declare test_time_relation")
        elif self.geometry == "time":
            if not self.time_column:
                raise ValueError("time geometry requires time_column")
            if self.group_column is not None:
                raise ValueError("time geometry with entities must use panel geometry")
        elif self.geometry == "panel":
            if not self.time_column or not self.group_column:
                raise ValueError("panel geometry requires time_column and group_column")
            if self.time_column == self.group_column:
                raise ValueError("panel time_column and group_column must be different")
        if self.geometry in {"time", "panel"}:
            if self.test_time_relation not in _TEST_TIME_RELATIONS:
                raise ValueError(
                    "time and panel geometry require test_time_relation to be "
                    f"one of {sorted(_TEST_TIME_RELATIONS)}")

        required_gap = self.label_horizon
        if self.geometry not in {"time", "panel"}:
            if self.label_horizon or self.feature_lookback or self.gap not in (None, 0):
                raise ValueError(
                    "horizon/lookback declarations only apply to time or panel geometry")
        elif self.gap is not None and self.gap < required_gap:
            raise ValueError(
                f"gap={self.gap} is unsafe: declared label horizon requires "
                f"at least {required_gap}")

        if self.task == "ranking" and self.geometry not in {"grouped", "panel"}:
            raise ValueError(
                "ranking requires grouped or panel geometry so each query has multiple rows")
        if self.task == "ranking":
            if self.ranking_target_order not in _RANKING_TARGET_ORDERS:
                raise ValueError(
                    "ranking requires ranking_target_order to be one of "
                    f"{sorted(_RANKING_TARGET_ORDERS)}")
        elif self.ranking_target_order is not None:
            raise ValueError("ranking_target_order applies only to task='ranking'")
        if pd.isna(self.positive_label):
            raise ValueError("positive_label cannot be missing")
        if (isinstance(self.positive_label, Number)
                and not _numeric_scalar_is_finite(self.positive_label)):
            raise ValueError("numeric positive_label must be finite")

        non_target_roles = [*self.id_columns]
        non_target_roles += [c for c in (self.time_column, self.group_column) if c]
        if len(set(non_target_roles)) != len(non_target_roles):
            raise ValueError("ID, time, and group role columns must be distinct")
        role_columns = [self.target, *non_target_roles]
        if self.target in non_target_roles:
            raise ValueError("target cannot also be an ID, time, or group column")
        if any(col in _LEDGER_COLUMNS for col in role_columns):
            raise ValueError(
                f"role columns cannot use reserved ledger names {sorted(_LEDGER_COLUMNS)}")

        if self.metric is None:
            object.__setattr__(self, "metric", _DEFAULT_METRICS[self.task])
        elif not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a non-empty string")
        supported = SUPPORTED_METRICS[self.task]
        if self.metric not in supported:
            raise ValueError(
                f"metric {self.metric!r} is unsupported for {self.task}; "
                f"choose from {sorted(supported)}")
        expected_direction = supported[self.metric]
        if self.metric_direction is None:
            object.__setattr__(self, "metric_direction", expected_direction)
        elif self.metric_direction != expected_direction:
            raise ValueError(
                f"metric {self.metric!r} must be {expected_direction}, not "
                f"{self.metric_direction!r}")
        if self.prediction_column is None:
            object.__setattr__(self, "prediction_column", self.target)
        elif not isinstance(self.prediction_column, str) or not self.prediction_column:
            raise ValueError("prediction_column must be a non-empty string")
        if self.prediction_column in {
                *self.id_columns, self.time_column, self.group_column}:
            raise ValueError(
                "prediction_column cannot also be an ID, time, or group column")
        if self.submission_transform not in _SUBMISSION_TRANSFORMS:
            raise ValueError(
                f"submission_transform must be one of {sorted(_SUBMISSION_TRANSFORMS)}")
        if self.submission_transform != "identity" and self.task != "ranking":
            raise ValueError("rank submission transforms require task='ranking'")

    @property
    def purge_gap(self) -> int:
        """Effective chronological gap in distinct timestamps."""
        required = self.label_horizon
        return required if self.gap is None else self.gap


def _require_frame(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name} has duplicate columns: {duplicates}")


def _numeric_scalar_is_finite(value: Number) -> bool:
    """Handle NumPy and Python numeric scalar families consistently."""
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return bool(np.isfinite(complex(value)))


def _validated_binary_truth(
    values: pd.Series,
    positive_label: str | int | float | bool,
    *,
    context: str,
) -> np.ndarray:
    """Validate a two-class truth vector before mapping its positive class."""
    if values.isna().any():
        raise ValueError(f"{context} contains missing values")
    numeric_values = [value for value in values.to_numpy()
                      if isinstance(value, Number)]
    if any(not _numeric_scalar_is_finite(value) for value in numeric_values):
        raise ValueError(f"{context} contains infinity")
    classes = list(pd.unique(values))
    if len(classes) != 2:
        raise ValueError(
            f"{context} must contain exactly two classes, found {classes!r}")
    if positive_label not in classes:
        raise ValueError(
            f"positive_label={positive_label!r} is not in {context} classes "
            f"{classes!r}")
    return (values.to_numpy() == positive_label).astype(int)


def _validate_problem_data(
    data: pd.DataFrame,
    spec: ProblemSpec,
    *,
    target_required: bool,
) -> None:
    _require_frame(data, "data")
    roles = [*spec.id_columns]
    roles += [col for col in (spec.time_column, spec.group_column) if col]
    if target_required:
        roles.append(spec.target)
    missing = [col for col in roles if col not in data.columns]
    if missing:
        raise KeyError(f"data is missing required columns: {missing}")

    if spec.id_columns:
        ids = data.loc[:, list(spec.id_columns)]
        if ids.isna().any(axis=None):
            raise ValueError("ID columns contain missing values")
        if ids.duplicated().any():
            raise ValueError("ID columns do not uniquely identify rows")
    if spec.time_column:
        if data[spec.time_column].isna().any():
            raise ValueError(f"time column {spec.time_column!r} contains missing values")
        _sorted_unique(data[spec.time_column])
    if spec.group_column and data[spec.group_column].isna().any():
        raise ValueError(f"group column {spec.group_column!r} contains missing values")

    if not target_required:
        return
    target = data[spec.target]
    if target.isna().any():
        raise ValueError(f"target {spec.target!r} contains missing values")
    if spec.task in {"regression", "ranking"}:
        if not pd.api.types.is_numeric_dtype(target):
            raise TypeError(f"{spec.task} target {spec.target!r} must be numeric")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"target {spec.target!r} contains infinity")
    if spec.task == "binary":
        _validated_binary_truth(
            target, spec.positive_label, context="binary target")


def _validate_test_data(test: pd.DataFrame, spec: ProblemSpec) -> None:
    """Validate a competition test frame and forbid target-bearing rows."""
    _validate_problem_data(test, spec, target_required=False)
    if spec.target in test.columns:
        raise ValueError(
            f"test contains target column {spec.target!r}; remove labels at "
            "the competition boundary")


def _time_axis_kind(values: pd.Series) -> Literal["datetime", "numeric"]:
    """Validate a time role without guessing chronology from lexical strings."""
    dtype = values.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if (pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_bool_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype)):
        observed = values.to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"time column {values.name!r} contains infinity")
        return "numeric"
    raise TypeError(
        f"time column {values.name!r} must use a datetime or numeric dtype; "
        f"got {dtype!s}. Parse string dates explicitly before validation")


def _sorted_unique(values: pd.Series) -> np.ndarray:
    _time_axis_kind(values)
    try:
        return pd.Index(values.drop_duplicates()).sort_values().to_numpy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{values.name!r} cannot be sorted consistently") from exc


def _validate_train_test_time_contract(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ProblemSpec,
) -> None:
    """Validate the explicitly declared train-to-test chronological relation."""
    if not spec.time_column:
        return
    train_kind = _time_axis_kind(train[spec.time_column])
    test_kind = _time_axis_kind(test[spec.time_column])
    if train_kind != test_kind:
        raise TypeError(
            f"time column {spec.time_column!r} changes kind between train "
            f"({train_kind}) and test ({test_kind})")
    if train_kind == "datetime":
        train_tz = getattr(train[spec.time_column].dtype, "tz", None)
        test_tz = getattr(test[spec.time_column].dtype, "tz", None)
        if (train_tz is None) != (test_tz is None):
            raise TypeError(
                f"time column {spec.time_column!r} mixes timezone-naive and "
                "timezone-aware train/test values")
    if spec.test_time_relation == "unrestricted":
        return
    try:
        strictly_future = train[spec.time_column].max() < test[spec.time_column].min()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"train/test time column {spec.time_column!r} cannot be compared") from exc
    if not bool(strictly_future):
        raise ValueError(
            f"test_time_relation='future' requires max(train[{spec.time_column!r}]) "
            f"< min(test[{spec.time_column!r}])")


def make_folds(data: pd.DataFrame, spec: ProblemSpec) -> tuple[Fold, ...]:
    """Route ``spec.geometry`` to a validated, deterministic fold plan.

    IID binary data is stratified; grouped binary data is stratified without
    splitting groups; chronological data is split on *distinct timestamps*
    with an expanding window and the declared purge.  Splitting on timestamps
    rather than rows prevents a panel date from straddling train and validation.
    """
    _validate_problem_data(data, spec, target_required=True)
    n_rows = len(data)
    y = data[spec.target].to_numpy()

    try:
        if spec.geometry == "iid":
            if spec.task == "binary":
                splitter = StratifiedKFold(
                    n_splits=spec.n_splits,
                    shuffle=True,
                    random_state=spec.random_state,
                )
                raw_folds = splitter.split(np.zeros(n_rows), y)
            else:
                splitter = KFold(
                    n_splits=spec.n_splits,
                    shuffle=True,
                    random_state=spec.random_state,
                )
                raw_folds = splitter.split(np.zeros(n_rows))
        elif spec.geometry == "grouped":
            groups = data[spec.group_column].to_numpy()
            n_groups = data[spec.group_column].nunique(dropna=False)
            if n_groups < spec.n_splits:
                raise ValueError(
                    f"{n_groups} groups is too few for {spec.n_splits} splits")
            if spec.task == "binary":
                splitter = StratifiedGroupKFold(
                    n_splits=spec.n_splits,
                    shuffle=True,
                    random_state=spec.random_state,
                )
                raw_folds = splitter.split(np.zeros(n_rows), y, groups)
            else:
                # GroupKFold before sklearn 1.6 has no shuffle/random_state;
                # its deterministic balancing is sufficient here.
                splitter = GroupKFold(n_splits=spec.n_splits)
                raw_folds = splitter.split(np.zeros(n_rows), y, groups)
        else:
            times = _sorted_unique(data[spec.time_column])
            if len(times) <= spec.n_splits:
                raise ValueError(
                    f"{len(times)} distinct timestamps is too few for "
                    f"{spec.n_splits} splits")
            splitter = TimeSeriesSplit(n_splits=spec.n_splits, gap=spec.purge_gap)
            time_of_row = data[spec.time_column].to_numpy()
            mapped: list[Fold] = []
            for train_time_pos, valid_time_pos in splitter.split(times):
                train_times = times[train_time_pos]
                valid_times = times[valid_time_pos]
                mapped.append((
                    np.flatnonzero(np.isin(time_of_row, train_times)),
                    np.flatnonzero(np.isin(time_of_row, valid_times)),
                ))
            raw_folds = mapped

        folds = tuple(
            (np.asarray(train, dtype=int), np.asarray(valid, dtype=int))
            for train, valid in raw_folds
        )
    except ValueError as exc:
        raise ValueError(f"could not build safe {spec.geometry} folds: {exc}") from exc

    validate_folds(data, folds, spec)
    return folds


def make_final_holdout(
    data: pd.DataFrame,
    spec: ProblemSpec,
    *,
    fraction: float = 0.20,
) -> Fold:
    """Reserve one geometry-safe final holdout before model selection.

    IID rows are optionally stratified, grouped data holds out whole groups,
    and time/panel data holds out the final block of distinct timestamps after
    the label-horizon purge.  The returned positions refer to ``data``; rows in
    a chronological purge belong to neither side.
    """
    _validate_problem_data(data, spec, target_required=True)
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise TypeError("fraction must be numeric")
    if not 0 < float(fraction) < 0.5:
        raise ValueError("fraction must lie strictly between 0 and 0.5")
    positions = np.arange(len(data))

    if spec.geometry == "iid":
        stratify = data[spec.target] if spec.task == "binary" else None
        development, holdout = train_test_split(
            positions,
            test_size=float(fraction),
            random_state=spec.random_state,
            shuffle=True,
            stratify=stratify,
        )
        development, holdout = np.sort(development), np.sort(holdout)
    elif spec.geometry == "grouped":
        splitter = GroupShuffleSplit(
            n_splits=20,
            test_size=float(fraction),
            random_state=spec.random_state,
        )
        selected = None
        for development, holdout in splitter.split(
            positions, data[spec.target], data[spec.group_column]):
            if spec.task != "binary" or (
                data.iloc[development][spec.target].nunique() == 2
                and data.iloc[holdout][spec.target].nunique() == 2
            ):
                selected = (np.sort(development), np.sort(holdout))
                break
        if selected is None:
            raise ValueError(
                "could not make a group-exclusive holdout containing both binary classes")
        development, holdout = selected
    else:
        times = _sorted_unique(data[spec.time_column])
        n_holdout = max(1, int(np.ceil(len(times) * float(fraction))))
        holdout_start = len(times) - n_holdout
        development_end = holdout_start - spec.purge_gap
        if development_end < 1:
            raise ValueError(
                "too few timestamps remain before the final holdout and purge")
        time_of_row = data[spec.time_column].to_numpy()
        development = np.flatnonzero(np.isin(time_of_row, times[:development_end]))
        holdout = np.flatnonzero(np.isin(time_of_row, times[holdout_start:]))

    if not len(development) or not len(holdout):
        raise ValueError("final holdout produced an empty side")
    if np.intersect1d(development, holdout).size:
        raise AssertionError("internal error: final holdout overlaps development rows")
    if spec.geometry == "grouped":
        shared = set(data.iloc[development][spec.group_column]).intersection(
            data.iloc[holdout][spec.group_column])
        if shared:
            raise AssertionError("internal error: final holdout shares groups")
    if spec.geometry in {"time", "panel"}:
        train_end = data.iloc[development][spec.time_column].max()
        holdout_start_value = data.iloc[holdout][spec.time_column].min()
        times = pd.Index(_sorted_unique(data[spec.time_column]))
        realised_gap = (times.get_loc(holdout_start_value)
                        - times.get_loc(train_end) - 1)
        if not train_end < holdout_start_value or realised_gap < spec.purge_gap:
            raise AssertionError("internal error: unsafe chronological final holdout")
    if spec.task == "binary":
        if (data.iloc[development][spec.target].nunique() != 2
                or data.iloc[holdout][spec.target].nunique() != 2):
            raise ValueError("final holdout must contain both binary classes on each side")
    return np.asarray(development, dtype=int), np.asarray(holdout, dtype=int)


def _coerce_fold_indices(indices, *, side: str, fold: int, n_rows: int) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"fold {fold} {side} indices must be one-dimensional")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise TypeError(f"fold {fold} {side} indices must be integers")
    array = array.astype(int, copy=False)
    if not len(array):
        raise ValueError(f"fold {fold} has an empty {side} set")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"fold {fold} repeats rows inside its {side} set")
    if array.min() < 0 or array.max() >= n_rows:
        raise IndexError(f"fold {fold} {side} indices are outside [0, {n_rows})")
    return array


def validate_folds(
    data: pd.DataFrame,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    spec: ProblemSpec,
) -> pd.DataFrame:
    """Assert leakage invariants and return one diagnostic row per fold.

    The validation rows of separate folds must be disjoint.  IID and grouped
    plans must cover every row exactly once.  Grouped plans may not share an
    entity across a boundary.  Time and panel plans may not share a timestamp,
    must train strictly before validation, and must realise the requested gap.
    Binary folds must contain both classes on both sides so that fold metrics
    and ``predict_proba`` have an unambiguous meaning.
    """
    _validate_problem_data(data, spec, target_required=True)
    materialised = tuple(folds)
    if len(materialised) != spec.n_splits:
        raise ValueError(
            f"expected {spec.n_splits} folds from the spec, got {len(materialised)}")

    n_rows = len(data)
    seen_validation: set[int] = set()
    diagnostic_rows: list[dict[str, object]] = []
    all_times = _sorted_unique(data[spec.time_column]) if spec.time_column else None
    time_index = pd.Index(all_times) if all_times is not None else None

    for fold_number, pair in enumerate(materialised):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"fold {fold_number} must be a (train, validation) pair")
        train = _coerce_fold_indices(
            pair[0], side="training", fold=fold_number, n_rows=n_rows)
        valid = _coerce_fold_indices(
            pair[1], side="validation", fold=fold_number, n_rows=n_rows)

        row_overlap = np.intersect1d(train, valid)
        if len(row_overlap):
            raise ValueError(
                f"fold {fold_number} leaks {len(row_overlap)} rows across train/validation")
        repeated_validation = seen_validation.intersection(valid.tolist())
        if repeated_validation:
            raise ValueError(
                f"validation rows occur in more than one fold; first repeats are "
                f"{sorted(repeated_validation)[:5]}")
        seen_validation.update(valid.tolist())

        row: dict[str, object] = {
            "fold": fold_number,
            "n_train": len(train),
            "n_valid": len(valid),
            "row_overlap": 0,
        }

        if spec.group_column:
            train_groups = set(pd.unique(data.iloc[train][spec.group_column]))
            valid_groups = set(pd.unique(data.iloc[valid][spec.group_column]))
            group_overlap = train_groups.intersection(valid_groups)
            row.update({
                "n_train_groups": len(train_groups),
                "n_valid_groups": len(valid_groups),
                "group_overlap": len(group_overlap),
            })
            if spec.geometry == "grouped" and group_overlap:
                raise ValueError(
                    f"fold {fold_number} leaks {len(group_overlap)} groups across the boundary")

        if spec.time_column:
            train_values = data.iloc[train][spec.time_column]
            valid_values = data.iloc[valid][spec.time_column]
            train_times = set(pd.unique(train_values))
            valid_times = set(pd.unique(valid_values))
            overlap = train_times.intersection(valid_times)
            if overlap:
                raise ValueError(
                    f"fold {fold_number} leaks {len(overlap)} timestamps across the boundary")
            train_end = train_values.max()
            valid_start = valid_values.min()
            if not train_end < valid_start:
                raise ValueError(
                    f"fold {fold_number} is not chronological: train ends at "
                    f"{train_end!r}, validation starts at {valid_start!r}")
            positions = time_index.get_indexer([train_end, valid_start])
            if (positions < 0).any():
                raise AssertionError("internal error: fold times are absent from the time axis")
            gap_observed = int(positions[1] - positions[0] - 1)
            if gap_observed < spec.purge_gap:
                raise ValueError(
                    f"fold {fold_number} realises gap {gap_observed}, smaller than "
                    f"required {spec.purge_gap}")
            row.update({
                "n_train_times": len(train_times),
                "n_valid_times": len(valid_times),
                "time_overlap": 0,
                "train_time_end": train_end,
                "valid_time_start": valid_start,
                "gap": gap_observed,
            })

        target_train = data.iloc[train][spec.target]
        target_valid = data.iloc[valid][spec.target]
        if spec.task == "binary":
            train_classes = set(pd.unique(target_train))
            valid_classes = set(pd.unique(target_valid))
            if len(train_classes) != 2 or len(valid_classes) != 2:
                raise ValueError(
                    f"fold {fold_number} must contain both binary classes in train and "
                    f"validation; found {train_classes!r} and {valid_classes!r}")
            row["train_target_mean"] = float((target_train == spec.positive_label).mean())
            row["valid_target_mean"] = float((target_valid == spec.positive_label).mean())
        else:
            row["train_target_mean"] = float(target_train.mean())
            row["valid_target_mean"] = float(target_valid.mean())
        diagnostic_rows.append(row)

    if spec.geometry in {"iid", "grouped"} and len(seen_validation) != n_rows:
        missing = n_rows - len(seen_validation)
        raise ValueError(
            f"{spec.geometry} validation must cover every row exactly once; "
            f"{missing} rows are uncovered")
    return pd.DataFrame(diagnostic_rows)


def fold_diagnostics(
    data: pd.DataFrame,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    spec: ProblemSpec,
) -> pd.DataFrame:
    """Readable alias for :func:`validate_folds` used in notebooks."""
    return validate_folds(data, folds, spec)


def _resolve_features(data: pd.DataFrame, feature_columns: Sequence[str], spec: ProblemSpec) -> tuple[str, ...]:
    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be a sequence, not a single string")
    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns is empty")
    if any(not isinstance(col, str) or not col for col in features):
        raise ValueError("every feature column must be a non-empty string")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns contains duplicates")
    missing = [col for col in features if col not in data.columns]
    if missing:
        raise KeyError(f"feature columns are missing from data: {missing}")
    if spec.target in features:
        raise ValueError(
            f"target {spec.target!r} appears in feature_columns (direct target leakage)")
    return features


def validate_competition_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: ProblemSpec,
    *,
    availability: Mapping[str, bool],
    allow_cross_split_duplicates: bool = False,
) -> pd.DataFrame:
    """Fail fast at the data boundary and return a missingness/schema report.

    Selected features must be distinct from declared roles, explicitly known at
    prediction time, present with identical dtypes in train and test, non-empty,
    and finite where observed.  The helper also validates the explicit
    train/test time contract and sample-submission identity/order, and rejects
    exact feature rows copied across train and test unless the caller explicitly
    allows them.  Label-aware proxy screening is deliberately separate; run
    :func:`validate_development_target_proxies` only after reserving a holdout.
    """
    _validate_problem_data(train, spec, target_required=True)
    _validate_test_data(test, spec)
    _validate_train_test_time_contract(train, test, spec)
    _require_frame(sample_submission, "sample_submission")
    features = _resolve_features(train, feature_columns, spec)
    missing_test = [col for col in features if col not in test]
    if missing_test:
        raise KeyError(f"test is missing feature columns: {missing_test}")
    role_columns = {spec.target, *spec.id_columns}
    role_columns.update(col for col in (spec.time_column, spec.group_column) if col)
    overlap = sorted(role_columns.intersection(features))
    if overlap:
        raise ValueError(f"declared roles cannot also be raw features: {overlap}")
    if not isinstance(availability, Mapping):
        raise TypeError("availability must explicitly map every feature to bool")
    missing_availability = [col for col in features if col not in availability]
    extra_availability = [col for col in availability if col not in features]
    if missing_availability or extra_availability:
        raise ValueError(
            f"availability keys must equal features; missing={missing_availability}, "
            f"extra={extra_availability}")
    invalid_availability = {
        col: value for col, value in availability.items()
        if not isinstance(value, (bool, np.bool_)) or not bool(value)
    }
    if invalid_availability:
        raise ValueError(
            f"features not explicitly available at prediction time: "
            f"{invalid_availability}")

    rows: list[dict[str, object]] = []
    for column in features:
        train_dtype, test_dtype = str(train[column].dtype), str(test[column].dtype)
        if train_dtype != test_dtype:
            raise TypeError(
                f"feature {column!r} dtype differs: train={train_dtype}, test={test_dtype}")
        if train[column].isna().all() or test[column].isna().all():
            raise ValueError(f"feature {column!r} is all-null in train or test")
        if pd.api.types.is_numeric_dtype(train[column]):
            for name, frame in (("train", train), ("test", test)):
                observed = frame[column].dropna().to_numpy(dtype=float)
                if not np.isfinite(observed).all():
                    raise ValueError(f"{name} feature {column!r} contains infinity")
        train_null = float(train[column].isna().mean())
        test_null = float(test[column].isna().mean())
        rows.append({
            "column": column,
            "dtype": train_dtype,
            "train_null_rate": train_null,
            "test_null_rate": test_null,
            "null_rate_delta": test_null - train_null,
            "available_at_prediction": True,
        })

    # Build-and-discard a dummy submission to validate sample IDs and schema at
    # the boundary, before modelling work can hide the mismatch.
    validate_submission(test, np.zeros(len(test)), sample_submission, spec)
    train_hash = pd.util.hash_pandas_object(
        train.loc[:, list(features)], index=False, categorize=True)
    test_hash = pd.util.hash_pandas_object(
        test.loc[:, list(features)], index=False, categorize=True)
    cross_matches = len(set(train_hash.to_numpy()).intersection(test_hash.to_numpy()))
    if cross_matches and not allow_cross_split_duplicates:
        raise ValueError(
            f"train/test share {cross_matches} exact feature rows; investigate or "
            "set allow_cross_split_duplicates=True explicitly")
    report = pd.DataFrame(rows)
    report.attrs["cross_split_exact_feature_rows"] = cross_matches
    return report


def validate_development_target_proxies(
    development: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: ProblemSpec,
) -> pd.DataFrame:
    """Reject obvious target proxies using development labels only.

    This check is intentionally not part of :func:`validate_competition_data`:
    target-driven screening must happen after a final holdout is reserved, or
    feature acceptance has already inspected the supposedly untouched labels.
    The returned empty, schema-stable report is useful as bundle evidence; a
    non-empty report raises with the suspect columns and reasons.
    """
    _validate_problem_data(development, spec, target_required=True)
    features = _resolve_features(development, feature_columns, spec)
    from .data.quality import leakage_scan

    suspects = leakage_scan(
        development, spec.target, feature_cols=list(features))
    if len(suspects):
        details = suspects[["column", "flags"]].to_dict("records")
        raise ValueError(
            f"possible target leakage in development features: {details}")
    suspects.attrs["screened_rows"] = len(development)
    suspects.attrs["scope"] = "development_only"
    return suspects


def _binary_probability(estimator, X: pd.DataFrame, positive_label) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("binary estimators must implement predict_proba")
    probabilities = np.asarray(estimator.predict_proba(X))
    classes = list(getattr(estimator, "classes_", ()))
    if positive_label not in classes:
        raise ValueError(
            f"fitted estimator classes {classes!r} do not contain "
            f"positive_label={positive_label!r}")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba returned a shape inconsistent with classes_")
    return probabilities[:, classes.index(positive_label)]


def _ledger_piece(
    data: pd.DataFrame,
    spec: ProblemSpec,
    positions: np.ndarray,
    fold_number: int,
    predictions: np.ndarray,
) -> pd.DataFrame:
    trace_columns: list[str] = []
    for column in (*spec.id_columns, spec.time_column, spec.group_column):
        if column and column not in trace_columns:
            trace_columns.append(column)
    piece = pd.DataFrame({
        "row_position": positions,
        "fold": np.full(len(positions), fold_number, dtype=int),
    })
    if trace_columns:
        trace = data.iloc[positions][trace_columns].reset_index(drop=True)
        piece = pd.concat([piece, trace], axis=1)
    piece["y_true"] = data.iloc[positions][spec.target].to_numpy()
    piece["prediction"] = predictions
    return piece


def _checked_predictions(values, expected: int, *, context: str) -> np.ndarray:
    try:
        predictions = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} predictions must be numeric") from exc
    if predictions.ndim != 1 or len(predictions) != expected:
        raise ValueError(
            f"{context} predictions must have shape ({expected},), got "
            f"{predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError(f"{context} predictions contain NaN or infinity")
    return predictions


def oof_ledger(
    estimator,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: ProblemSpec,
    *,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
) -> pd.DataFrame:
    """Fit one cloned estimator per fold and return an OOF prediction ledger.

    Binary tasks always use the probability of ``spec.positive_label`` from
    ``predict_proba``; class predictions are never silently substituted.  The
    returned frame is sorted back into source-row order and includes IDs,
    time/group roles, the fold number, truth, and prediction.
    """
    _validate_problem_data(data, spec, target_required=True)
    features = _resolve_features(data, feature_columns, spec)
    resolved_folds = make_folds(data, spec) if folds is None else tuple(folds)
    validate_folds(data, resolved_folds, spec)

    pieces: list[pd.DataFrame] = []
    for fold_number, (train, valid) in enumerate(resolved_folds):
        train = np.asarray(train, dtype=int)
        valid = np.asarray(valid, dtype=int)
        model = clone(estimator)
        model.fit(data.iloc[train][list(features)], data.iloc[train][spec.target])
        if spec.task == "binary":
            raw = _binary_probability(
                model, data.iloc[valid][list(features)], spec.positive_label)
        else:
            if not hasattr(model, "predict"):
                raise TypeError("regression/ranking estimators must implement predict")
            raw = model.predict(data.iloc[valid][list(features)])
        predictions = _checked_predictions(raw, len(valid), context=f"fold {fold_number}")
        pieces.append(_ledger_piece(data, spec, valid, fold_number, predictions))

    ledger = pd.concat(pieces, ignore_index=True)
    ledger = ledger.sort_values("row_position", kind="stable").reset_index(drop=True)
    if ledger["row_position"].duplicated().any():
        raise AssertionError("internal error: OOF ledger contains duplicate rows")
    return ledger


def baseline_ledger(
    data: pd.DataFrame,
    spec: ProblemSpec,
    *,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
) -> pd.DataFrame:
    """Return fold-local trivial predictions in the same schema as ``oof_ledger``.

    Regression uses the training-fold mean, binary classification uses the
    training-fold positive prevalence, and ranking uses a no-information zero.
    Validation targets never contribute to their own baseline.
    """
    _validate_problem_data(data, spec, target_required=True)
    resolved_folds = make_folds(data, spec) if folds is None else tuple(folds)
    validate_folds(data, resolved_folds, spec)
    pieces: list[pd.DataFrame] = []

    for fold_number, (train, valid) in enumerate(resolved_folds):
        train = np.asarray(train, dtype=int)
        valid = np.asarray(valid, dtype=int)
        train_target = data.iloc[train][spec.target]
        if spec.task == "binary":
            value = float((train_target == spec.positive_label).mean())
        elif spec.task == "regression":
            value = float(train_target.mean())
        else:
            value = 0.0
        predictions = np.full(len(valid), value, dtype=float)
        pieces.append(_ledger_piece(data, spec, valid, fold_number, predictions))

    ledger = pd.concat(pieces, ignore_index=True)
    return ledger.sort_values("row_position", kind="stable").reset_index(drop=True)


def score_predictions(frame: pd.DataFrame, spec: ProblemSpec) -> float:
    """Score one prediction frame with the exact metric declared by ``spec``.

    ``frame`` must contain ``y_true`` and ``prediction``.  Ranking frames must
    additionally carry the query column declared by the problem geometry.
    """
    _require_frame(frame, "prediction frame")
    missing = [col for col in ("y_true", "prediction") if col not in frame]
    if missing:
        raise KeyError(f"prediction frame is missing columns: {missing}")
    prediction = _checked_predictions(
        frame["prediction"], len(frame), context=spec.metric)

    if spec.task == "regression":
        try:
            truth = frame["y_true"].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("regression truth must be numeric") from exc
        if not np.isfinite(truth).all():
            raise ValueError("regression truth contains NaN or infinity")
        if spec.metric == "root_mean_squared_error":
            return float(mean_squared_error(truth, prediction) ** 0.5)
        if spec.metric == "mean_absolute_error":
            return float(mean_absolute_error(truth, prediction))
        if spec.metric == "r2":
            if len(truth) < 2:
                raise ValueError("r2 requires at least two rows")
            score = float(r2_score(truth, prediction))
            if not np.isfinite(score):
                raise ValueError("r2 produced a non-finite score")
            return score

    if spec.task == "binary":
        truth = _validated_binary_truth(
            frame["y_true"], spec.positive_label,
            context=f"{spec.metric} truth")
        if ((prediction < 0) | (prediction > 1)).any():
            raise ValueError("binary predictions must lie in [0, 1]")
        if spec.metric == "roc_auc":
            return float(roc_auc_score(truth, prediction))
        if spec.metric == "average_precision":
            return float(average_precision_score(truth, prediction))
        if spec.metric == "log_loss":
            return float(log_loss(truth, prediction, labels=[0, 1]))
        if spec.metric == "brier":
            return float(brier_score_loss(truth, prediction))

    if spec.task == "ranking":
        query = spec.time_column or spec.group_column
        if query not in frame:
            raise KeyError(f"ranking prediction frame is missing query column {query!r}")
        if frame[query].isna().any():
            raise ValueError(f"ranking query column {query!r} contains missing values")
        try:
            truth = frame["y_true"].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("ranking truth must be numeric") from exc
        if not np.isfinite(truth).all():
            raise ValueError("ranking truth contains NaN or infinity")
        scored = pd.DataFrame({
            "query": frame[query].to_numpy(),
            "truth": truth,
            "raw_prediction": prediction,
        })
        if spec.submission_transform == "rank_descending_zero":
            scored["truth_for_metric"] = _ranking_zero_best(
                scored["query"], truth, spec, tie_method="average")
            scored["prediction_for_metric"] = _ranking_zero_best(
                scored["query"], prediction, spec, tie_method="first")
        else:
            scored["truth_for_metric"] = truth
            scored["prediction_for_metric"] = prediction
        values: list[float] = []
        for _, group in scored.groupby("query", observed=True, sort=False):
            if len(group) < 2 or group["truth"].nunique() < 2:
                continue
            # A constant raw score contains no ranking information.  Do not
            # let deterministic submission tie-breaking turn row order into a
            # validation advantage.
            if group["raw_prediction"].nunique() < 2:
                values.append(0.0)
            else:
                values.append(float(group["truth_for_metric"].corr(
                    group["prediction_for_metric"], method="spearman")))
        if not values:
            raise ValueError(
                "spearman requires at least one multi-row query with varying truth")
        score = float(np.mean(values))
        if not np.isfinite(score):
            raise ValueError("spearman produced a non-finite score")
        return score
    raise AssertionError(f"unhandled supported metric {spec.metric!r}")


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """Fold-level CV evidence plus a separately labelled pooled OOF score."""

    metric: str
    direction: Literal["minimize", "maximize"]
    fold_scores: tuple[float, ...]
    cv_mean: float
    cv_std: float
    worst_fold: float
    pooled_oof: float

    def to_frame(self) -> pd.DataFrame:
        """Render the summary as one notebook-friendly row."""
        return pd.DataFrame([{
            "metric": self.metric,
            "direction": self.direction,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "worst_fold": self.worst_fold,
            "pooled_oof": self.pooled_oof,
        }])

    def folds_frame(self) -> pd.DataFrame:
        """Render the individual fold scores without losing their identity."""
        return pd.DataFrame({"fold": range(len(self.fold_scores)),
                             "score": self.fold_scores})


def score_ledger(ledger: pd.DataFrame, spec: ProblemSpec) -> ScoreReport:
    """Score identical OOF rows fold by fold, then pooled as a diagnostic.

    The CV mean is the primary model-selection estimate.  ``pooled_oof`` is
    deliberately separate because non-decomposable metrics such as AUC compare
    scores emitted by different fitted fold models and need not equal the fold
    mean.
    """
    _require_frame(ledger, "ledger")
    if "fold" not in ledger:
        raise KeyError("ledger is missing fold")
    fold_numbers = sorted(pd.unique(ledger["fold"]))
    if not fold_numbers:
        raise ValueError("ledger contains no folds")
    fold_scores = tuple(
        score_predictions(ledger.loc[ledger["fold"] == fold], spec)
        for fold in fold_numbers
    )
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    worst = (max(fold_scores) if spec.metric_direction == "minimize"
             else min(fold_scores))
    return ScoreReport(
        metric=spec.metric,
        direction=spec.metric_direction,
        fold_scores=fold_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        worst_fold=float(worst),
        pooled_oof=score_predictions(ledger, spec),
    )


def _geometry_preserving_permutation(
    data: pd.DataFrame,
    spec: ProblemSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a task-aware null without retaining structure-level signal.

    Ranking is evaluated only within query, so its labels are shuffled within
    query.  For every non-ranking structured design, rows are arranged in a
    seeded random order of contiguous structure blocks (and randomized within
    block), then labels are circularly shifted on that randomized order by a
    feasible offset.  This is an exact row permutation: the target multiset is
    unchanged and no destination receives a label from its own structure.
    """
    target = data[spec.target].to_numpy(copy=True)
    if spec.geometry == "iid":
        return rng.permutation(target)

    structure = (spec.time_column
                 if spec.geometry in {"time", "panel"}
                 else spec.group_column)
    if spec.task == "ranking":
        permuted = target.copy()
        for positions in data.groupby(
                structure, observed=True, sort=False).indices.values():
            positions = np.asarray(positions, dtype=int)
            permuted[positions] = rng.permutation(target[positions])
        return permuted

    if spec.geometry in {"time", "panel"}:
        keys = _sorted_unique(data[structure])
        values = data[structure].to_numpy()
        blocks = [np.flatnonzero(values == key) for key in keys]
    else:
        blocks = [np.asarray(positions, dtype=int) for positions in
                  data.groupby(structure, observed=True, sort=False).indices.values()]
    n_rows = len(target)
    max_block = max(map(len, blocks))
    if 2 * max_block > n_rows:
        raise ValueError(
            "an exact structure-breaking permutation is impossible: the "
            f"largest block has {max_block} of {n_rows} rows, so fewer than "
            "that many labels exist outside it")

    randomized_blocks = [
        rng.permutation(blocks[index])
        for index in rng.permutation(len(blocks))
    ]
    destination_order = np.concatenate(randomized_blocks)
    # Every block occupies a circular interval of length <= m.  A forward
    # displacement s >= m and backward displacement N-s >= m therefore cannot
    # remain inside that same interval.  The feasible integer interval is
    # exactly [m, N-m], which exists iff 2m <= N.
    shift = int(rng.integers(max_block, n_rows - max_block + 1))
    source_order = np.roll(destination_order, -shift)
    permuted = target.copy()
    permuted[destination_order] = target[source_order]
    return permuted


def _binary_fold_classes_present(
    target: np.ndarray,
    folds: Sequence[Fold],
) -> bool:
    return all(
        len(pd.unique(target[np.asarray(positions, dtype=int)])) == 2
        for train, valid in folds for positions in (train, valid)
    )


def null_control_report(
    estimator,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: ProblemSpec,
    *,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
    tolerance: float = 0.10,
    assert_baseline: bool = True,
) -> pd.DataFrame:
    """Run repeated task-aware label controls against the baseline.

    IID labels are permuted globally.  Ranking labels are permuted within query
    because only within-query order is scored.  Non-ranking structured labels
    use an exact constrained row permutation across group/time blocks so
    legitimate structure-level features do not survive the null while the
    target multiset remains unchanged.  ``improvement`` is positive only when
    the null model beats the fold-local baseline, regardless of metric
    direction.  The mean improvement must not exceed ``tolerance``.
    """
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 2:
        raise ValueError("n_repeats must be an integer of at least 2")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    resolved_folds = make_folds(data, spec) if folds is None else tuple(folds)
    validate_folds(data, resolved_folds, spec)
    seed = spec.random_state + 1 if random_state is None else random_state
    rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        attempts = 0
        while True:
            permutation_seed = seed + repeat * 1_009 + attempts
            permuted_target = _geometry_preserving_permutation(
                data, spec, np.random.default_rng(permutation_seed))
            attempts += 1
            if (spec.task != "binary" or _binary_fold_classes_present(
                    permuted_target, resolved_folds)):
                break
            if attempts >= 100:
                raise RuntimeError(
                    "could not construct a structured binary null with both "
                    "classes in every training and validation fold after 100 "
                    "deterministic attempts")
        null_data = data.copy()
        null_data[spec.target] = permuted_target
        baseline = score_ledger(
            baseline_ledger(null_data, spec, folds=resolved_folds), spec)
        ledger = oof_ledger(
            estimator, null_data, feature_columns, spec, folds=resolved_folds)
        report = score_ledger(ledger, spec)
        improvement = (baseline.cv_mean - report.cv_mean
                       if spec.metric_direction == "minimize"
                       else report.cv_mean - baseline.cv_mean)
        rows.append({
            "repeat": repeat,
            "seed": permutation_seed,
            "permutation_attempts": attempts,
            "baseline_cv_mean": baseline.cv_mean,
            "null_cv_mean": report.cv_mean,
            "null_cv_std": report.cv_std,
            "null_pooled_oof": report.pooled_oof,
            "improvement_over_baseline": float(improvement),
        })
    result = pd.DataFrame(rows)
    mean_improvement = float(result["improvement_over_baseline"].mean())
    result.attrs.update({
        "metric": spec.metric,
        "direction": spec.metric_direction,
        "strategy": ("within_query" if spec.task == "ranking" else
                     "global" if spec.geometry == "iid" else
                     "exact_structure_breaking"),
        "mean_improvement_over_baseline": mean_improvement,
        "tolerance": float(tolerance),
    })
    if assert_baseline and mean_improvement > tolerance:
        raise AssertionError(
            f"null control beats the fold-local baseline by {mean_improvement:.4g}, "
            f"above tolerance {tolerance:.4g}; inspect leakage")
    return result


def heldout_permutation_importance(
    estimator,
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: ProblemSpec,
    *,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
    fold: int = -1,
    n_repeats: int = 8,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Permutation importance on one validation fold using the declared metric.

    Ranking features are permuted *within query* so the perturbation preserves
    each cross-section.  Binary scoring extracts ``spec.positive_label`` rather
    than relying on sklearn's class ordering.  Positive importance always means
    worse performance after permutation, for either metric direction.
    """
    _validate_problem_data(data, spec, target_required=True)
    features = _resolve_features(data, feature_columns, spec)
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    resolved_folds = make_folds(data, spec) if folds is None else tuple(folds)
    validate_folds(data, resolved_folds, spec)
    try:
        train, valid = resolved_folds[fold]
    except IndexError as exc:
        raise IndexError(f"fold {fold} is outside {len(resolved_folds)} folds") from exc
    train, valid = np.asarray(train, dtype=int), np.asarray(valid, dtype=int)
    model = clone(estimator).fit(
        data.iloc[train][list(features)], data.iloc[train][spec.target])
    valid_X = data.iloc[valid][list(features)].copy()

    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = (_binary_probability(model, frame, spec.positive_label)
               if spec.task == "binary" else model.predict(frame))
        return _checked_predictions(raw, len(frame), context="held-out")

    base_frame = _ledger_piece(data, spec, valid, fold % len(resolved_folds), predict(valid_X))
    base_score = score_predictions(base_frame, spec)
    seed = spec.random_state if random_state is None else random_state
    rng = np.random.default_rng(seed)
    query = spec.time_column or spec.group_column
    query_positions = (
        list(data.iloc[valid].reset_index(drop=True)
             .groupby(query, observed=True, sort=False).indices.values())
        if spec.task == "ranking" else None
    )

    rows: list[dict[str, object]] = []
    for feature in features:
        degradations: list[float] = []
        for _ in range(n_repeats):
            permuted = valid_X.copy()
            values = permuted[feature].to_numpy(copy=True)
            if query_positions is None:
                values = rng.permutation(values)
            else:
                for positions in query_positions:
                    positions = np.asarray(positions, dtype=int)
                    values[positions] = rng.permutation(values[positions])
            permuted[feature] = values
            scored = base_frame.copy()
            scored["prediction"] = predict(permuted)
            permuted_score = score_predictions(scored, spec)
            degradation = (permuted_score - base_score
                           if spec.metric_direction == "minimize"
                           else base_score - permuted_score)
            degradations.append(float(degradation))
        rows.append({
            "feature": feature,
            "importance": float(np.mean(degradations)),
            "std": float(np.std(degradations, ddof=1)) if n_repeats > 1 else 0.0,
            "base_score": base_score,
            "fold": fold % len(resolved_folds),
            "n_valid": len(valid),
        })
    return (pd.DataFrame(rows)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


@dataclass(frozen=True, slots=True)
class AdversarialReport:
    """Cross-validated ability to distinguish reference from candidate rows."""

    auc: float
    fold_auc: tuple[float, ...]
    n_reference: int
    n_candidate: int
    feature_columns: tuple[str, ...]
    random_state: int

    @property
    def mean_fold_auc(self) -> float:
        return float(np.mean(self.fold_auc))

    @property
    def std_fold_auc(self) -> float:
        return float(np.std(self.fold_auc, ddof=1)) if len(self.fold_auc) > 1 else 0.0

    def to_frame(self) -> pd.DataFrame:
        """Render fold AUCs as a small notebook-friendly table."""
        return pd.DataFrame({"fold": range(len(self.fold_auc)), "auc": self.fold_auc})


def adversarial_validation_report(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> AdversarialReport:
    """Measure multivariate train/test drift with OOF domain-classifier AUC.

    ``0.5`` is indistinguishable; values approaching ``1`` mean the two samples
    are easy to tell apart.  Numeric imputation/scaling and categorical
    one-hot encoding are fitted inside every fold.
    """
    _require_frame(reference, "reference")
    _require_frame(candidate, "candidate")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be a sequence, not a single string")
    features = tuple(feature_columns)
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be non-empty and unique")
    missing_reference = [col for col in features if col not in reference.columns]
    missing_candidate = [col for col in features if col not in candidate.columns]
    if missing_reference or missing_candidate:
        raise KeyError(
            f"drift features missing; reference={missing_reference}, "
            f"candidate={missing_candidate}")
    if min(len(reference), len(candidate)) < n_splits:
        raise ValueError("each sample must contain at least n_splits rows")

    numeric: list[str] = []
    categorical: list[str] = []
    for column in features:
        left_numeric = pd.api.types.is_numeric_dtype(reference[column])
        right_numeric = pd.api.types.is_numeric_dtype(candidate[column])
        if left_numeric != right_numeric:
            raise TypeError(
                f"feature {column!r} changes kind between reference and candidate")
        (numeric if left_numeric else categorical).append(column)

    combined = pd.concat(
        [reference.loc[:, list(features)], candidate.loc[:, list(features)]],
        ignore_index=True,
    )
    for column in numeric:
        observed = combined[column].dropna().astype(float)
        if not np.isfinite(observed.to_numpy()).all():
            raise ValueError(f"numeric drift feature {column!r} contains infinity")
    for column in categorical:
        combined[column] = combined[column].astype("string").fillna("__MISSING__")

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    estimator = Pipeline([
        ("prepare", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
        )),
    ])

    domain = np.r_[np.zeros(len(reference), dtype=int), np.ones(len(candidate), dtype=int)]
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    probabilities = np.full(len(combined), np.nan, dtype=float)
    fold_scores: list[float] = []
    for train, valid in splitter.split(combined, domain):
        model = clone(estimator)
        model.fit(combined.iloc[train], domain[train])
        fold_probability = model.predict_proba(combined.iloc[valid])[:, 1]
        probabilities[valid] = fold_probability
        fold_scores.append(float(roc_auc_score(domain[valid], fold_probability)))
    if not np.isfinite(probabilities).all():
        raise AssertionError("internal error: adversarial OOF predictions are incomplete")

    return AdversarialReport(
        auc=float(roc_auc_score(domain, probabilities)),
        fold_auc=tuple(fold_scores),
        n_reference=len(reference),
        n_candidate=len(candidate),
        feature_columns=features,
        random_state=random_state,
    )


def _ranking_zero_best(
    queries,
    values,
    spec: ProblemSpec,
    *,
    tie_method: Literal["average", "first"],
) -> np.ndarray:
    """Convert ranking values to zero-best ranks under the declared order."""
    if spec.ranking_target_order not in _RANKING_TARGET_ORDERS:
        raise ValueError("ranking_target_order is required for ranking postprocessing")
    scores = _checked_predictions(values, len(queries), context="ranking")
    frame = pd.DataFrame({"query": np.asarray(queries), "score": scores})
    if frame["query"].isna().any():
        raise ValueError("ranking queries contain missing values")
    ascending = spec.ranking_target_order == "lower_is_better"
    ranks = (frame.groupby("query", observed=True, sort=False)["score"]
             .rank(method=tie_method, ascending=ascending)
             .sub(1))
    return ranks.to_numpy(dtype=float)


def postprocess_predictions(
    test: pd.DataFrame,
    predictions,
    spec: ProblemSpec,
) -> np.ndarray:
    """Apply the submission transform declared by ``spec`` in source order."""
    _validate_problem_data(test, spec, target_required=False)
    checked = _checked_predictions(predictions, len(test), context="submission")
    if spec.task == "binary" and ((checked < 0) | (checked > 1)).any():
        raise ValueError("binary submission probabilities must lie in [0, 1]")
    if spec.submission_transform == "identity":
        return checked
    query = spec.time_column or spec.group_column
    if not query or query not in test:
        raise KeyError("rank submission transform requires its query column in test")
    return _ranking_zero_best(
        test[query].to_numpy(), checked, spec, tie_method="first").astype(int)


def validate_submission(
    test: pd.DataFrame,
    predictions,
    sample_submission: pd.DataFrame,
    spec: ProblemSpec,
) -> pd.DataFrame:
    """Build a submission only when IDs, order, shape, and values are valid.

    The sample submission is the schema authority.  Test IDs must match it in
    the same order; the function never silently sorts or merges rows.  A copy
    of the sample is returned with only ``spec.prediction_column`` replaced.
    """
    _validate_problem_data(test, spec, target_required=False)
    _require_frame(sample_submission, "sample_submission")
    if not spec.id_columns:
        raise ValueError("submission validation requires at least one id_column")
    required = [*spec.id_columns, spec.prediction_column]
    missing = [col for col in required if col not in sample_submission.columns]
    if missing:
        raise KeyError(f"sample_submission is missing required columns: {missing}")
    unexpected = [col for col in sample_submission.columns if col not in required]
    if unexpected:
        raise ValueError(
            f"sample_submission has unsupported extra columns: {unexpected}")
    if len(test) != len(sample_submission):
        raise ValueError(
            f"test has {len(test)} rows but sample_submission has "
            f"{len(sample_submission)}")

    test_ids = test.loc[:, list(spec.id_columns)].reset_index(drop=True)
    sample_ids = sample_submission.loc[:, list(spec.id_columns)].reset_index(drop=True)
    if sample_ids.isna().any(axis=None) or sample_ids.duplicated().any():
        raise ValueError("sample submission IDs must be complete and unique")
    if not test_ids.equals(sample_ids):
        same_values = set(map(tuple, test_ids.to_numpy())) == set(
            map(tuple, sample_ids.to_numpy()))
        detail = "same IDs but different order" if same_values else "different IDs"
        raise ValueError(f"test and sample submission have {detail}")

    checked = postprocess_predictions(test, predictions, spec)

    submission = sample_submission.copy(deep=True)
    original_columns = list(submission.columns)
    submission[spec.prediction_column] = checked
    if list(submission.columns) != original_columns:
        raise AssertionError("internal error: submission column order changed")
    if not submission.loc[:, list(spec.id_columns)].reset_index(drop=True).equals(test_ids):
        raise AssertionError("internal error: submission ID order changed")
    return submission


def data_fingerprint(frame: pd.DataFrame, *, include_index: bool = True) -> str:
    """Return a SHA-256 fingerprint of schema, row order, values, and index.

    The hash is deterministic within the declared pandas environment and is
    sensitive to column order and row order.  It is intended to identify the
    exact frame used by a run, not to serve as a cross-language file format.
    """
    _require_frame(frame, "frame")
    schema: list[dict[str, object]] = []
    for column, dtype in frame.dtypes.items():
        entry: dict[str, object] = {
            "name_type": type(column).__name__,
            "name": repr(column),
            "dtype": str(dtype),
        }
        if isinstance(dtype, pd.CategoricalDtype):
            entry["categories"] = [repr(value) for value in dtype.categories]
            entry["ordered"] = dtype.ordered
        schema.append(entry)
    index_dtypes = (list(frame.index.dtypes)
                    if isinstance(frame.index, pd.MultiIndex)
                    else [frame.index.dtype])
    metadata = {
        "shape": list(frame.shape),
        "schema": schema,
        "include_index": include_index,
        "index_names": [repr(name) for name in frame.index.names],
        "index_dtypes": [str(dtype) for dtype in index_dtypes],
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame, index=include_index, categorize=True).to_numpy(dtype="uint64")
    except (TypeError, ValueError) as exc:
        raise TypeError("frame contains values pandas cannot fingerprint") from exc
    digest.update(row_hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def code_fingerprint(path: os.PathLike[str] | str) -> str:
    """Hash source identity while ignoring notebook execution noise.

    Ordinary files are hashed byte for byte.  For ``.ipynb`` files, only cell
    type, stable ID, source, and sorted cell tags plus the notebook format
    version are hashed.  Tags are semantic here because they select setup,
    parameters and data-context behavior; outputs, execution counts and other
    transient metadata cannot change code identity after a successful run.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"code identity path does not exist: {resolved}")
    if resolved.suffix.lower() != ".ipynb":
        return hashlib.sha256(resolved.read_bytes()).hexdigest()
    try:
        notebook = json.loads(resolved.read_text(encoding="utf-8"))
        canonical = {
            "nbformat": notebook["nbformat"],
            "nbformat_minor": notebook["nbformat_minor"],
            "cells": [{
                **{key: cell.get(key) for key in ("cell_type", "id", "source")},
                "tags": sorted(cell.get("metadata", {}).get("tags", [])),
            } for cell in notebook["cells"]],
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid notebook for code identity: {resolved}") from exc
    payload = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_notebook_source(
    filename: str,
    configured_path: os.PathLike[str] | str | None = None,
) -> Path | None:
    """Resolve a running notebook without assuming an editable source tree.

    An explicit path is authoritative and must exist.  Otherwise the common
    launch locations ``cwd/filename``, ``cwd/notebooks/filename`` and
    ``cwd/notebooks/legacy/filename`` are checked, in that order. ``None`` is a
    valid result for hosted environments that do not expose the notebook file;
    callers can still fingerprint installed modules.
    """
    if (not isinstance(filename, str) or not filename
            or Path(filename).name != filename or not filename.endswith(".ipynb")):
        raise ValueError("filename must be a local .ipynb basename")
    if configured_path is not None:
        resolved = Path(configured_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"configured notebook source does not exist: {resolved}")
        return resolved
    for candidate in (
            Path.cwd() / filename,
            Path.cwd() / "notebooks" / filename,
            Path.cwd() / "notebooks" / "legacy" / filename):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _dataset_manifest(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "fingerprint": data_fingerprint(frame),
    }
    for attr in ("synthetic", "profile", "task"):
        if attr in frame.attrs:
            value = frame.attrs[attr]
            if isinstance(value, np.generic):
                value = value.item()
            result[attr] = value
    return result


def build_run_manifest(
    spec: ProblemSpec,
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    feature_columns: Sequence[str] | None = None,
    git_commit: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a deterministic, JSON-serialisable identity for an analysis run.

    Wall-clock time is intentionally absent: calling this twice on unchanged
    inputs returns the same manifest and ``manifest_id``.  A caller that needs
    an audit timestamp can put it in ``extra`` without confusing run identity
    with reproducibility.
    """
    _validate_problem_data(train, spec, target_required=True)
    if test is not None:
        _validate_test_data(test, spec)
        _validate_train_test_time_contract(train, test, spec)
    if feature_columns is None:
        excluded = {spec.target, *spec.id_columns, spec.time_column, spec.group_column}
        features = tuple(col for col in train.columns if col not in excluded)
        if not features:
            raise ValueError("no feature columns remain after excluding problem roles")
    else:
        features = _resolve_features(train, feature_columns, spec)
    if test is not None:
        missing_test = [col for col in features if col not in test.columns]
        if missing_test:
            raise KeyError(f"test is missing manifest features: {missing_test}")
    if git_commit is not None and (not isinstance(git_commit, str) or not git_commit):
        raise ValueError("git_commit must be None or a non-empty string")
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping")

    try:
        detached_extra = json.loads(json.dumps(
            dict(extra or {}), sort_keys=True, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise TypeError("extra must contain only finite JSON-serialisable values") from exc
    manifest: dict[str, object] = {
        "spec": dataclasses.asdict(spec),
        "features": list(features),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rvlab": rvlab_version,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "data": {
            "train": _dataset_manifest(train),
            "test": _dataset_manifest(test) if test is not None else None,
        },
        "git_commit": git_commit,
        "extra": detached_extra,
    }
    try:
        canonical = json.dumps(
            manifest, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("manifest contains a value that is not JSON-serialisable") from exc
    manifest["manifest_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def materialize_folds(
    data: pd.DataFrame,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    spec: ProblemSpec,
    *,
    source_positions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Expand a fold plan into a durable row-level assignment table.

    ``source_positions`` maps positions in a development frame back to the
    original training frame.  IDs and geometry roles are copied alongside the
    positional mapping so an exported split can be audited without rebuilding
    it from a random seed.  The mapping is required: silently treating a sliced
    development frame as positions ``0..n`` creates false provenance.
    """
    resolved_folds = tuple(folds)
    validate_folds(data, resolved_folds, spec)
    if source_positions is None:
        raise ValueError(
            "source_positions is required; pass an explicit mapping to the "
            "original training frame")
    raw = np.asarray(source_positions)
    if raw.ndim != 1 or len(raw) != len(data):
        raise ValueError(
            "source_positions must be one-dimensional and match data rows")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError("source_positions must contain integers")
    source = raw.astype(int, copy=False)
    if len(np.unique(source)) != len(source):
        raise ValueError("source_positions must be unique")
    if source.min() < 0:
        raise IndexError("source_positions cannot be negative")

    trace_columns: list[str] = []
    for column in (*spec.id_columns, spec.time_column, spec.group_column):
        if column and column not in trace_columns:
            trace_columns.append(column)
    pieces: list[pd.DataFrame] = []
    for fold_number, (fit, valid) in enumerate(resolved_folds):
        for role, positions in (("train", fit), ("validation", valid)):
            positions = np.asarray(positions, dtype=int)
            piece = pd.DataFrame({
                "fold": fold_number,
                "role": role,
                "development_position": positions,
                "source_position": source[positions],
            })
            if trace_columns:
                trace = data.iloc[positions][trace_columns].reset_index(drop=True)
                piece = pd.concat([piece, trace], axis=1)
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _resolve_source_positions(
    train: pd.DataFrame,
    development: pd.DataFrame,
    spec: ProblemSpec,
    source_positions: Sequence[int] | None,
) -> np.ndarray:
    """Validate an explicit source map or derive one from immutable row IDs."""
    if source_positions is None:
        if development.reset_index(drop=True).equals(train.reset_index(drop=True)):
            source = np.arange(len(train), dtype=int)
        elif spec.id_columns:
            train_ids = pd.MultiIndex.from_frame(
                train.loc[:, list(spec.id_columns)], names=list(spec.id_columns))
            development_ids = pd.MultiIndex.from_frame(
                development.loc[:, list(spec.id_columns)], names=list(spec.id_columns))
            source = train_ids.get_indexer(development_ids)
            if (source < 0).any():
                missing = development.loc[
                    source < 0, list(spec.id_columns)].head(5).to_dict("records")
                raise ValueError(
                    f"cannot derive source_positions: development IDs are absent "
                    f"from train; first missing IDs are {missing}")
        else:
            raise ValueError(
                "source_positions is required when development is not the full "
                "training frame and no unique id_columns are declared")
    else:
        raw = np.asarray(source_positions)
        if raw.ndim != 1 or len(raw) != len(development):
            raise ValueError(
                "source_positions must be one-dimensional and match development rows")
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
            raise TypeError("source_positions must contain integers")
        source = raw.astype(int, copy=False)
        if len(np.unique(source)) != len(source):
            raise ValueError("source_positions must be unique")
    if not len(source):
        raise ValueError("source_positions is empty")
    if source.min() < 0 or source.max() >= len(train):
        raise IndexError("source_positions point outside the training frame")
    try:
        pd.testing.assert_frame_equal(
            development.reset_index(drop=True),
            train.iloc[source].reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(
            "development rows do not match train at resolved source_positions") from exc
    return np.asarray(source, dtype=int)


def _predict_fitted(estimator, frame: pd.DataFrame, spec: ProblemSpec) -> np.ndarray:
    if spec.task == "binary":
        raw = _binary_probability(estimator, frame, spec.positive_label)
    else:
        if not hasattr(estimator, "predict"):
            raise TypeError("fitted regression/ranking estimator must implement predict")
        raw = estimator.predict(frame)
    return _checked_predictions(raw, len(frame), context="fitted model")


def _git_identity(repo_root: os.PathLike[str] | str | None) -> tuple[str | None, bool | None]:
    if repo_root is None:
        return None, None
    root = Path(repo_root)
    if not root.is_dir():
        raise FileNotFoundError(f"repo_root does not exist: {root}")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=False)
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=False)
    except OSError as exc:
        raise RuntimeError("could not inspect git identity") from exc
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return commit, dirty


def _final_manifest_id(manifest: Mapping[str, object]) -> str:
    """Hash the complete manifest payload, excluding only its own ID field."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
    canonical = json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _directories_byte_identical(left: Path, right: Path) -> bool:
    """Return whether two bundle directories have the same complete contents."""
    if not left.is_dir() or not right.is_dir():
        return False
    left_entries = {
        str(path.relative_to(left)): "file" if path.is_file() else "dir"
        for path in left.rglob("*")
    }
    right_entries = {
        str(path.relative_to(right)): "file" if path.is_file() else "dir"
        for path in right.rglob("*")
    }
    if left_entries != right_entries:
        return False
    return all(
        (left / relative).read_bytes() == (right / relative).read_bytes()
        for relative, kind in left_entries.items() if kind == "file"
    )


def _csv_token_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact text tokens pandas serializes for a CSV frame."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, dtype=str, keep_default_na=False)


def _schema_value(value: object) -> dict[str, object]:
    """Describe an axis label without silently changing its Python type."""
    if isinstance(value, np.generic):
        value = value.item()
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if value is None or isinstance(value, (str, int, bool)):
        return {"type": type_name, "value": value}
    if isinstance(value, float) and np.isfinite(value):
        return {"type": type_name, "value": value}
    return {"type": type_name, "repr": repr(value)}


def _categorical_schema(dtype: object) -> dict[str, object] | None:
    """Record category order and unused levels omitted by CSV itself."""
    if not isinstance(dtype, pd.CategoricalDtype):
        return None
    return {
        "categories": [_schema_value(value) for value in dtype.categories],
        "ordered": bool(dtype.ordered),
    }


def _normalise_evidence_table(
    frame: pd.DataFrame,
    filename: str,
    *,
    materialise_index: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Make CSV schema explicit and return lossless sidecar metadata."""
    if isinstance(frame.columns, pd.MultiIndex):
        flat_columns = [
            " | ".join("" if value is None else str(value) for value in label)
            for label in frame.columns.tolist()
        ]
        column_labels = [
            [_schema_value(value) for value in label]
            for label in frame.columns.tolist()
        ]
    else:
        flat_columns = [str(label) for label in frame.columns]
        column_labels = [_schema_value(label) for label in frame.columns]
    if len(set(flat_columns)) != len(flat_columns):
        raise ValueError(
            f"evidence table {filename!r} has columns that collide after "
            "deterministic CSV flattening")

    normalised = frame.copy(deep=False)
    normalised.columns = flat_columns
    default_index = (
        isinstance(frame.index, pd.RangeIndex)
        and frame.index.start == 0
        and frame.index.stop == len(frame)
        and frame.index.step == 1
        and frame.index.name is None
    )
    exported_index_columns: list[str] = []
    preserve_index = materialise_index and not default_index
    if preserve_index:
        occupied = set(flat_columns)
        index_parts: dict[str, object] = {}
        for level in range(frame.index.nlevels):
            name = frame.index.names[level]
            base = (
                str(name)
                if name is not None and not (isinstance(name, str) and not name)
                else "index"
            )
            candidate = base
            if candidate in occupied:
                candidate = f"__index_{level}_{base}__"
            suffix = 1
            while candidate in occupied:
                candidate = f"__index_{level}_{base}_{suffix}__"
                suffix += 1
            occupied.add(candidate)
            exported_index_columns.append(candidate)
            index_parts[candidate] = frame.index.get_level_values(level).to_numpy()
        normalised = pd.concat(
            [pd.DataFrame(index_parts), normalised.reset_index(drop=True)], axis=1)

    try:
        detached_attrs = json.loads(json.dumps(
            dict(frame.attrs), sort_keys=True, allow_nan=False,
            separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"evidence table {filename!r} attrs must contain only finite "
            "JSON-serialisable values") from exc
    metadata: dict[str, object] = {
        "attrs": detached_attrs,
        "rows": len(frame),
        "index": {
            "class": f"{type(frame.index).__module__}.{type(frame.index).__qualname__}",
            "names": [_schema_value(name) for name in frame.index.names],
            "dtypes": [
                str(frame.index.get_level_values(level).dtype)
                for level in range(frame.index.nlevels)
            ],
            "categoricals": [
                {"level": level, **categorical}
                for level in range(frame.index.nlevels)
                if (categorical := _categorical_schema(
                    frame.index.get_level_values(level).dtype)) is not None
            ],
            "preserved": preserve_index,
            "nondefault_index_excluded": not default_index and not materialise_index,
            "exported_columns": exported_index_columns,
            "range": ({
                "start": frame.index.start,
                "stop": frame.index.stop,
                "step": frame.index.step,
            } if isinstance(frame.index, pd.RangeIndex) else None),
        },
        "columns": {
            "class": f"{type(frame.columns).__module__}.{type(frame.columns).__qualname__}",
            "names": [_schema_value(name) for name in frame.columns.names],
            "labels": column_labels,
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "categoricals": [
                {"position": position, **categorical}
                for position, dtype in enumerate(frame.dtypes)
                if (categorical := _categorical_schema(dtype)) is not None
            ],
            "exported_columns": list(normalised.columns),
        },
    }
    return normalised, metadata


def _validate_fresh_joblib_load(model_path: Path) -> None:
    """Prove an artifact is loadable outside the exporting interpreter."""
    environment = os.environ.copy()
    python_paths = [
        str(Path.cwd()) if entry == "" else str(entry)
        for entry in sys.path if entry is not None
    ]
    inherited = environment.get("PYTHONPATH")
    if inherited:
        python_paths.extend(inherited.split(os.pathsep))
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    command = [
        sys.executable,
        "-c",
        "import joblib, pathlib, sys; joblib.load(pathlib.Path(sys.argv[1]))",
        str(model_path),
    ]
    message = (
        "model.joblib cannot load in a fresh interpreter; move notebook-local "
        "custom classes and functions into an importable module")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False,
            env=environment, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TypeError(message) from exc
    if result.returncode:
        detail = next(
            (line.strip() for line in reversed(result.stderr.splitlines())
             if line.strip()),
            f"fresh process exited {result.returncode}",
        )
        raise TypeError(f"{message}: {detail}")


def export_run_bundle(
    output_dir: os.PathLike[str] | str,
    run_name: str,
    spec: ProblemSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_columns: Sequence[str],
    fitted_model,
    submission: pd.DataFrame,
    *,
    development: pd.DataFrame,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    source_positions: Sequence[int] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    model_card: Mapping[str, object] | None = None,
    code_paths: Sequence[os.PathLike[str] | str] = (),
    repo_root: os.PathLike[str] | str | None = None,
) -> Path:
    """Save and revalidate a deterministic competition evidence bundle.

    Split diagnostics, materialised row assignments, a fitted model, model
    card, validated submission, manifest, data/sample/code identities, and
    SHA-256 checksums are always written.  A checksummed metadata sidecar keeps
    each table's attrs and original index/column schema; semantic evidence
    indexes are explicit CSV columns and MultiIndex columns are flattened
    deterministically.  A submission index is recorded but never serialized,
    because only the sample-submission columns define that contract.
    Before returning, the model is loaded in a fresh interpreter and the saved
    submission repeats the entire raw-feature prediction, task-specific
    postprocessing, ID/order validation path.
    """
    if not isinstance(run_name, str) or not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be a non-empty directory basename")
    _validate_problem_data(train, spec, target_required=True)
    _validate_test_data(test, spec)
    _validate_train_test_time_contract(train, test, spec)
    features = _resolve_features(train, feature_columns, spec)
    missing_test = [column for column in features if column not in test]
    if missing_test:
        raise KeyError(f"test is missing export features: {missing_test}")
    train_types = train.loc[:, list(features)].dtypes.astype(str)
    test_types = test.loc[:, list(features)].dtypes.astype(str)
    if not train_types.equals(test_types):
        changed = [column for column in features
                   if train_types[column] != test_types[column]]
        raise TypeError(f"train/test feature dtypes differ at export: {changed}")

    resolved_folds = tuple(folds)
    split_report = fold_diagnostics(development, resolved_folds, spec)
    source = _resolve_source_positions(
        train, development, spec, source_positions)
    assignments = materialize_folds(
        development, resolved_folds, spec, source_positions=source)
    raw_prediction = _predict_fitted(
        fitted_model, test.loc[:, list(features)], spec)
    expected_submission = validate_submission(
        test, raw_prediction, sample_submission, spec)
    try:
        pd.testing.assert_frame_equal(
            expected_submission.reset_index(drop=True),
            submission.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        raise ValueError(
            "provided submission does not match fitted-model predictions") from exc

    commit, dirty = _git_identity(repo_root)
    code_identity: dict[str, str | bool | None] = {
        "git_commit": commit,
        "git_dirty": dirty,
    }
    identity_root = Path(repo_root).resolve() if repo_root is not None else None
    for item in code_paths:
        path = Path(item)
        resolved = path.resolve()
        try:
            label = str(resolved.relative_to(identity_root)) if identity_root else resolved.name
        except ValueError:
            label = resolved.name
        if label in code_identity:
            raise ValueError(f"code identity paths share the label {label!r}")
        code_identity[label] = code_fingerprint(path)
    try:
        card = json.loads(json.dumps(
            dict(model_card or {}), sort_keys=True, allow_nan=False,
            separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise TypeError("model_card must contain finite JSON-serialisable values") from exc
    card["code_identity"] = code_identity
    card.setdefault("metric", spec.metric)
    card.setdefault("metric_direction", spec.metric_direction)
    manifest = build_run_manifest(
        spec, train, test, feature_columns=features, git_commit=commit,
        extra={
            "model_card": card,
            "sample_submission_fingerprint": data_fingerprint(sample_submission),
            "code_identity": code_identity,
        },
    )
    reserved = {
        "split_diagnostics.csv", "fold_assignments.csv", "submission.csv",
        "model_card.json", "model.joblib", "manifest.json",
        "table_metadata.json",
    }
    evidence: dict[str, pd.DataFrame] = {
        "split_diagnostics.csv": split_report,
        "fold_assignments.csv": assignments,
    }
    for filename, frame in dict(tables or {}).items():
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith(".csv")):
            raise ValueError("evidence table names must be local .csv basenames")
        if filename in reserved:
            raise ValueError(f"evidence filename is reserved: {filename}")
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"evidence table {filename!r} must be a DataFrame")
        evidence[filename] = frame
    evidence["submission.csv"] = expected_submission
    normalised_evidence: dict[str, pd.DataFrame] = {}
    table_metadata: dict[str, object] = {}
    for filename, frame in evidence.items():
        normalised, metadata = _normalise_evidence_table(
            frame, filename, materialise_index=filename != "submission.csv")
        normalised_evidence[filename] = normalised
        table_metadata[filename] = metadata

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_name}.", dir=output_root))
    try:
        for filename, frame in normalised_evidence.items():
            frame.to_csv(staging / filename, index=False)
        table_metadata_path = staging / "table_metadata.json"
        table_metadata_path.write_text(
            json.dumps(
                table_metadata, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8")
        (staging / "model_card.json").write_text(
            json.dumps(card, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8")
        model_path = staging / "model.joblib"
        joblib.dump(fitted_model, model_path)
        _validate_fresh_joblib_load(model_path)

        # Validate the staged bytes, including the complete model -> raw score
        # -> task postprocess -> submission path, before a public run directory
        # exists.  Compare canonical serialized ID tokens rather than relying
        # on read_csv dtype inference; this preserves leading zeros, datetimes,
        # categoricals, and extension-backed identifiers alike.
        reloaded = joblib.load(model_path)
        reloaded_raw = _predict_fitted(
            reloaded, test.loc[:, list(features)], spec)
        reloaded_submission = validate_submission(
            test, reloaded_raw, sample_submission, spec)
        expected_tokens = _csv_token_frame(reloaded_submission)
        expected_id_tokens = expected_tokens.loc[:, list(spec.id_columns)]
        if expected_id_tokens.eq("").any(axis=None):
            raise ValueError(
                "declared IDs contain an empty token after CSV serialization")
        if expected_id_tokens.duplicated().any():
            raise ValueError(
                "declared IDs collide after CSV serialization; use a lossless "
                "canonical ID representation before export")
        saved_tokens = pd.read_csv(
            staging / "submission.csv", dtype=str, keep_default_na=False)
        if saved_tokens.columns.tolist() != expected_tokens.columns.tolist():
            raise AssertionError("staged submission CSV changed its column schema")
        saved_id_tokens = saved_tokens.loc[:, list(spec.id_columns)]
        if saved_id_tokens.eq("").any(axis=None):
            raise AssertionError("staged submission CSV contains an empty ID token")
        if saved_id_tokens.duplicated().any():
            raise AssertionError("staged submission CSV contains duplicate ID tokens")
        pd.testing.assert_frame_equal(
            expected_id_tokens,
            saved_id_tokens, check_exact=True)
        saved_prediction = pd.to_numeric(
            saved_tokens[spec.prediction_column], errors="raise").to_numpy(dtype=float)
        np.testing.assert_allclose(
            reloaded_submission[spec.prediction_column].to_numpy(dtype=float),
            saved_prediction, rtol=1e-12, atol=1e-12)

        artifacts = [*(staging / filename for filename in normalised_evidence),
                     staging / "model_card.json", model_path,
                     table_metadata_path]
        manifest["artifacts_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(artifacts, key=lambda item: item.name)
        }
        # The preliminary input-only ID from build_run_manifest is replaced by
        # the final identity of inputs, contract, code and exact artifact bytes.
        manifest["manifest_id"] = _final_manifest_id(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8")
        destination = output_root / f"{run_name}_{manifest['manifest_id'][:12]}"

        if destination.exists():
            if _directories_byte_identical(staging, destination):
                return destination
            raise FileExistsError(
                f"run destination already exists with different bytes: {destination}")
        try:
            staging.rename(destination)
        except OSError:
            # Handle a concurrent writer deterministically: identical exports
            # are idempotent; anything else is a collision and never mutates it.
            if destination.exists() and _directories_byte_identical(
                    staging, destination):
                return destination
            if destination.exists():
                raise FileExistsError(
                    f"run destination appeared with different bytes: {destination}")
            raise
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _smoke_pipeline(data: pd.DataFrame, features: Sequence[str], task: Task, seed: int):
    numeric = data.loc[:, list(features)].select_dtypes("number").columns.tolist()
    categorical = [column for column in features if column not in numeric]
    branches: list[tuple[str, object, list[str]]] = []
    if numeric:
        branches.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        branches.append(("categorical", Pipeline([
            ("impute", SimpleImputer(
                strategy="most_frequent", missing_values=pd.NA)),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    model = (LogisticRegression(
        class_weight="balanced", max_iter=1_000, random_state=seed)
        if task == "binary" else Ridge(alpha=1.0))
    return Pipeline([
        ("prepare", ColumnTransformer(branches, remainder="drop")),
        ("model", model),
    ])


def smoke_test_condition_matrix(
    *,
    random_state: int = 42,
    n_train: int = 180,
    n_test: int = 60,
    n_splits: int = 3,
    holdout_fraction: float = 0.20,
) -> pd.DataFrame:
    """Exercise every supported task/geometry path on deterministic demos.

    Each row runs the real data-boundary validator, geometry-safe final
    holdout, fold router, baseline and model OOF ledgers, exact scorer,
    holdout score, full-data refit, task-aware postprocessing, and
    sample-submission validation.  This is a plumbing control, never model
    evidence for a real dataset.
    """
    designs: dict[Geometry, tuple[Task, ...]] = {
        "iid": ("regression", "binary"),
        "grouped": ("regression", "binary", "ranking"),
        "time": ("regression", "binary"),
        "panel": ("regression", "binary", "ranking"),
    }
    roles: dict[Geometry, dict[str, str]] = {
        "iid": {},
        "grouped": {"group_column": "group"},
        "time": {"time_column": "time"},
        "panel": {"time_column": "time", "group_column": "group"},
    }
    features = ("x_num", "x_aux", "category")
    rows: list[dict[str, object]] = []
    for geometry, tasks in designs.items():
        for task in tasks:
            train, test, sample = make_competition_demo(
                geometry, task=task, n_train=n_train, n_test=n_test,
                random_state=random_state)
            chronological = geometry in {"time", "panel"}
            spec = ProblemSpec(
                geometry=geometry,
                task=task,
                id_columns=("row_id",),
                n_splits=n_splits,
                random_state=random_state,
                label_horizon=1 if chronological else 0,
                test_time_relation="future" if chronological else None,
                submission_transform=(
                    "rank_descending_zero" if task == "ranking" else "identity"),
                ranking_target_order=(
                    "higher_is_better" if task == "ranking" else None),
                **roles[geometry],
            )
            validate_competition_data(
                train, test, sample, features, spec,
                availability={feature: True for feature in features})
            development_positions, holdout_positions = make_final_holdout(
                train, spec, fraction=holdout_fraction)
            development = train.iloc[development_positions].reset_index(drop=True)
            holdout = train.iloc[holdout_positions]
            validate_development_target_proxies(development, features, spec)
            folds = make_folds(development, spec)
            diagnostics = fold_diagnostics(development, folds, spec)
            estimator = _smoke_pipeline(
                development, features, task, random_state)
            baseline_report = score_ledger(
                baseline_ledger(development, spec, folds=folds), spec)
            ledger = oof_ledger(
                estimator, development, features, spec, folds=folds)
            model_report = score_ledger(ledger, spec)

            holdout_model = clone(estimator).fit(
                development.loc[:, list(features)], development[spec.target])
            holdout_frame = pd.DataFrame({
                "y_true": holdout[spec.target].to_numpy(),
                "prediction": _predict_fitted(
                    holdout_model, holdout.loc[:, list(features)], spec),
            })
            query = spec.time_column or spec.group_column
            if task == "ranking":
                holdout_frame[query] = holdout[query].to_numpy()
            holdout_score = score_predictions(holdout_frame, spec)

            final_model = clone(estimator).fit(
                train.loc[:, list(features)], train[spec.target])
            submission = validate_submission(
                test,
                _predict_fitted(final_model, test.loc[:, list(features)], spec),
                sample,
                spec,
            )
            if not submission.loc[:, list(spec.id_columns)].equals(
                    sample.loc[:, list(spec.id_columns)]):
                raise AssertionError("smoke submission changed ID order")
            if not np.isfinite(submission[spec.prediction_column]).all():
                raise AssertionError("smoke submission contains non-finite values")
            if task == "ranking":
                ranked = submission.assign(_query=test[query].to_numpy())
                for _, group in ranked.groupby("_query", observed=True, sort=False):
                    expected = list(range(len(group)))
                    if sorted(group[spec.prediction_column].tolist()) != expected:
                        raise AssertionError("smoke ranking postprocessing is invalid")
            rows.append({
                "condition": f"{geometry}/{task}",
                "geometry": geometry,
                "task": task,
                "metric": spec.metric,
                "direction": spec.metric_direction,
                "folds": len(diagnostics),
                "max_row_overlap": int(diagnostics["row_overlap"].max()),
                "oof_rows": len(ledger),
                "baseline_cv_mean": baseline_report.cv_mean,
                "cv_mean": model_report.cv_mean,
                "cv_std": model_report.cv_std,
                "worst_fold": model_report.worst_fold,
                "pooled_oof": model_report.pooled_oof,
                "final_holdout_score": holdout_score,
                "submission_rows": len(submission),
                "postprocess": spec.submission_transform,
            })
    return pd.DataFrame(rows)


def _demo_layout(
    profile: Geometry,
    n_train: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_roles: dict[str, np.ndarray] = {}
    test_roles: dict[str, np.ndarray] = {}
    if profile == "grouped":
        n_groups = min(24, max(4, n_train // 8))
        train_groups = np.resize(np.array([f"g{i:02d}" for i in range(n_groups)]), n_train)
        test_groups = np.resize(np.array([f"g{i:02d}" for i in range(n_groups)]), n_test)
        rng.shuffle(train_groups)
        rng.shuffle(test_groups)
        train_roles["group"] = train_groups
        test_roles["group"] = test_groups
    elif profile == "time":
        all_times = pd.date_range("2020-01-01", periods=n_train + n_test, freq="D")
        train_roles["time"] = all_times[:n_train].to_numpy()
        test_roles["time"] = all_times[n_train:].to_numpy()
    elif profile == "panel":
        n_entities = min(10, max(4, n_train // 30))
        entities = np.array([f"e{i:02d}" for i in range(n_entities)])
        train_steps = np.arange(n_train)
        test_steps = np.arange(n_test)
        train_date_number = train_steps // n_entities
        first_test_date = int(train_date_number.max()) + 1
        test_date_number = first_test_date + test_steps // n_entities
        origin = pd.Timestamp("2020-01-01")
        train_roles["time"] = (
            origin + pd.to_timedelta(train_date_number, unit="D")).to_numpy()
        test_roles["time"] = (
            origin + pd.to_timedelta(test_date_number, unit="D")).to_numpy()
        train_roles["group"] = np.resize(entities, n_train)
        test_roles["group"] = np.resize(entities, n_test)
    return train_roles, test_roles


def make_competition_demo(
    profile: Geometry,
    *,
    task: Task | None = None,
    n_train: int = 240,
    n_test: int = 80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic train/test/sample frames for one design profile.

    Defaults intentionally exercise different paths: IID regression, grouped
    imbalanced binary classification, time-series regression, and panel
    ranking.  Every profile contains numeric and categorical features,
    missingness, a novel test category, and moderate planted covariate drift.
    Standard role names (``row_id``, ``time``, ``group``, ``target``) keep the
    notebook's swap-data cell to one parameter.
    """
    if profile not in _GEOMETRIES:
        raise ValueError(f"profile must be one of {sorted(_GEOMETRIES)}, got {profile!r}")
    default_tasks: dict[str, Task] = {
        "iid": "regression",
        "grouped": "binary",
        "time": "regression",
        "panel": "ranking",
    }
    resolved_task = default_tasks[profile] if task is None else task
    if resolved_task not in _TASKS:
        raise ValueError(f"task must be one of {sorted(_TASKS)}, got {resolved_task!r}")
    if resolved_task == "ranking" and profile not in {"grouped", "panel"}:
        raise ValueError("demo ranking requires grouped or panel profile")
    for name, value, minimum in (("n_train", n_train, 30), ("n_test", n_test, 10)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError("random_state must be an integer")

    rng = np.random.default_rng(random_state)
    train_roles, test_roles = _demo_layout(profile, n_train, n_test, rng)
    total = n_train + n_test
    latent = rng.normal(size=total)
    x_num = latent + rng.normal(scale=0.45, size=total)
    x_aux = -0.35 * latent + rng.normal(scale=0.9, size=total)
    category = np.where(latent > 0.7, "high", np.where(latent < -0.7, "low", "mid"))

    # Test rows are different but not trivially separable: enough drift for the
    # adversarial report to find, plus missingness/category novelty to audit.
    x_num[n_train:] += 0.85
    novel = rng.random(n_test) < 0.16
    category[n_train:][novel] = "new"

    complete_aux = x_aux.copy()
    category_effect = pd.Series(category).map(
        {"low": -0.5, "mid": 0.0, "high": 0.6, "new": 0.3}).to_numpy()
    score = 1.25 * x_num - 0.65 * complete_aux + category_effect

    all_groups = np.r_[
        train_roles.get("group", np.repeat("none", n_train)),
        test_roles.get("group", np.repeat("none", n_test)),
    ]
    if profile in {"grouped", "panel"}:
        group_levels = sorted(pd.unique(all_groups))
        group_effect = {group: rng.normal(scale=0.35) for group in group_levels}
        score += np.array([group_effect[group] for group in all_groups])
    if profile in {"time", "panel"}:
        train_time = train_roles["time"]
        test_time = test_roles["time"]
        all_time = np.r_[train_time, test_time]
        time_codes = pd.factorize(all_time, sort=True)[0]
        score += 0.25 * np.sin(time_codes / 5.0)

    if resolved_task == "regression":
        target = score[:n_train] + rng.normal(scale=0.75, size=n_train)
    elif resolved_task == "binary":
        probabilities = 1.0 / (1.0 + np.exp(-(score[:n_train] - 1.65)))
        target = rng.binomial(1, probabilities).astype(int)
        # A grouped demo must be valid for any group-exclusive fold: make sure
        # every group has at least one observation from each class.
        if profile == "grouped":
            groups = train_roles["group"]
            for group in pd.unique(groups):
                positions = np.flatnonzero(groups == group)
                if len(np.unique(target[positions])) < 2:
                    ordered = positions[np.argsort(score[positions])]
                    target[ordered[0]], target[ordered[-1]] = 0, 1
        if len(np.unique(target)) < 2:
            target[np.argmin(score[:n_train])] = 0
            target[np.argmax(score[:n_train])] = 1
    else:
        target = score[:n_train] + rng.normal(scale=0.45, size=n_train)
        query = train_roles.get("time", train_roles.get("group"))
        target = target - pd.Series(target).groupby(query, sort=False).transform("mean").to_numpy()

    x_aux[rng.random(total) < np.r_[
        np.full(n_train, 0.09), np.full(n_test, 0.20)]] = np.nan
    category = pd.array(category, dtype="string")
    category[rng.random(total) < np.r_[
        np.full(n_train, 0.05), np.full(n_test, 0.13)]] = pd.NA

    train = pd.DataFrame({
        "row_id": [f"train_{i:06d}" for i in range(n_train)],
        "x_num": x_num[:n_train],
        "x_aux": x_aux[:n_train],
        "category": category[:n_train],
    })
    test = pd.DataFrame({
        "row_id": [f"test_{i:06d}" for i in range(n_test)],
        "x_num": x_num[n_train:],
        "x_aux": x_aux[n_train:],
        "category": category[n_train:],
    })
    for column, values in train_roles.items():
        train.insert(1, column, values)
    for column, values in test_roles.items():
        test.insert(1, column, values)
    train["target"] = target
    sample_submission = test[["row_id"]].copy()
    sample_submission["target"] = 0.0

    attrs = {"synthetic": True, "profile": profile, "task": resolved_task}
    train.attrs.update(attrs)
    test.attrs.update(attrs)
    sample_submission.attrs.update(attrs)
    return train, test, sample_submission


__all__ = [
    "AdversarialReport",
    "ProblemSpec",
    "RankingTargetOrder",
    "SUPPORTED_METRICS",
    "ScoreReport",
    "TestTimeRelation",
    "adversarial_validation_report",
    "baseline_ledger",
    "build_run_manifest",
    "code_fingerprint",
    "data_fingerprint",
    "export_run_bundle",
    "fold_diagnostics",
    "heldout_permutation_importance",
    "make_competition_demo",
    "make_final_holdout",
    "make_folds",
    "materialize_folds",
    "null_control_report",
    "oof_ledger",
    "postprocess_predictions",
    "resolve_notebook_source",
    "score_ledger",
    "score_predictions",
    "smoke_test_condition_matrix",
    "validate_competition_data",
    "validate_development_target_proxies",
    "validate_folds",
    "validate_submission",
]
