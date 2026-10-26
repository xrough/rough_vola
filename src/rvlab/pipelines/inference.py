"""
rvlab.pipelines.inference
=========================
The half of a competition pipeline that runs *after* the model is chosen, and
where a validated model most often falls over.

Three failure modes, one helper each:

  * **Feature drift.** Test data arrives with columns missing, extra, or in a
    different order. `align_feature_columns` reconstructs exactly the training
    schema.
  * **Streaming inference.** Many competitions hand you one day at a time and
    expect a prediction before showing the next. Rolling features need history,
    so you must carry a buffer — and trim it, or memory grows without bound.
  * **Submission format.** Predictions have to become unique integer ranks in
    the row order the grader supplied.

None of this is modelling. All of it is where the score goes to zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_feature_columns(df: pd.DataFrame, columns, fill=np.nan) -> pd.DataFrame:
    """Return `df` with exactly `columns`, in order, adding missing ones as NaN.

    A fitted sklearn estimator matches features **by position**, and a
    `ColumnTransformer` by name — so a reordered test frame either raises or,
    worse, silently scores the wrong columns. Call this on every inference frame.

    Missing columns become NaN rather than 0 so the pipeline's imputer handles
    them the same way it handled gaps during training.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = fill
    return out[list(columns)]


def check_inference_schema(train_columns, inference_df: pd.DataFrame) -> pd.DataFrame:
    """Compare a training schema against an inference frame. Inspect before predicting.

    Reports which expected columns are absent (they will be imputed) and which
    extra columns will be dropped. A long "missing" list usually means the
    feature-engineering function was not applied to the test data.
    """
    train_cols = list(train_columns)
    have = set(inference_df.columns)
    missing = [c for c in train_cols if c not in have]
    extra = [c for c in inference_df.columns if c not in set(train_cols)]
    return pd.DataFrame({
        "check": ["expected features", "present", "missing (imputed)", "extra (dropped)"],
        "count": [len(train_cols), len(train_cols) - len(missing), len(missing), len(extra)],
        "examples": ["", "", ", ".join(missing[:6]), ", ".join(extra[:6])],
    })


class RollingHistoryBuffer:
    """Fixed-length history for streaming inference.

    Feeds one day at a time, keeps the last `max_days` *dates* (not rows, since
    the entity count varies), and de-duplicates on the key so a corrected
    re-send overwrites the original.

        buffer = RollingHistoryBuffer(history, max_days=80)
        for day in stream:
            frame = buffer.append(day)          # everything needed for features
            features = build_features(frame)

    `max_days` must exceed the longest rolling window, or the window silently
    computes on a truncated history and the first predictions of every session
    differ from the ones validation produced.
    """

    def __init__(self, history: pd.DataFrame, max_days: int = 90,
                 date_col: str = "Date", key_cols=("Date", "SecuritiesCode")):
        self.date_col = date_col
        self.key_cols = list(key_cols)
        self.max_days = max_days
        self.frame = self._trim(self._normalise(history))

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])
        return out

    def _trim(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = np.sort(df[self.date_col].unique())
        if len(dates) > self.max_days:
            df = df[df[self.date_col].isin(dates[-self.max_days:])]
        return df.sort_values(self.key_cols).reset_index(drop=True)

    def append(self, day: pd.DataFrame) -> pd.DataFrame:
        """Add one day's rows and return the current buffer."""
        new = self._normalise(day)
        columns = sorted(set(self.frame.columns) | set(new.columns))
        for frame in (self.frame, new):
            for col in columns:
                if col not in frame.columns:
                    frame[col] = np.nan

        combined = pd.concat([self.frame[columns], new[columns]], ignore_index=True)
        combined = combined.drop_duplicates(self.key_cols, keep="last")
        self.frame = self._trim(combined)
        return self.frame

    @property
    def n_dates(self) -> int:
        return int(self.frame[self.date_col].nunique())

    def __repr__(self) -> str:
        return (f"RollingHistoryBuffer({len(self.frame):,} rows, "
                f"{self.n_dates} dates, max_days={self.max_days})")


def rank_for_submission(sample: pd.DataFrame, predictions, date_col: str = "Date",
                        rank_col: str = "Rank") -> pd.DataFrame:
    """Attach unique integer ranks to a grader-supplied frame, preserving row order.

    Two details competitions actually reject submissions over: the rank must be
    a unique integer starting at 0 within each date (hence `method="first"`),
    and the row order must be the one you were handed — so this never sorts.
    """
    out = sample.copy()
    out["_prediction"] = np.asarray(predictions, dtype=float)
    out[rank_col] = (out.groupby(date_col, observed=True)["_prediction"]
                     .rank(ascending=False, method="first").astype(int) - 1)
    return out.drop(columns="_prediction")


def fill_missing_predictions(predictions: pd.Series, dates) -> pd.Series:
    """Replace NaN predictions with that date's median, then with 0.

    An entity absent from your feature frame — newly listed, or dropped by a
    join — must still receive a rank. Predicting the day's median places it in
    the middle of the book, which is the neutral choice; dropping the row breaks
    the submission.
    """
    p = pd.Series(np.asarray(predictions, dtype=float))
    by_date = p.groupby(pd.Series(np.asarray(dates)).to_numpy())
    return p.fillna(by_date.transform("median")).fillna(0.0)


__all__ = [
    "align_feature_columns", "check_inference_schema", "RollingHistoryBuffer",
    "rank_for_submission", "fill_missing_predictions",
]
