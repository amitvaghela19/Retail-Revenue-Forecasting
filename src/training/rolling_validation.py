from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics


@dataclass
class RollingFoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    metrics: dict


def make_rolling_date_folds(
    df: pd.DataFrame,
    date_col: str = "Date",
    n_folds: int = 3,
    val_days: int = 30,
    min_train_days: int = 365,
    step_days: int = 30,
):
    df = df.sort_values(date_col).copy()
    unique_dates = pd.Series(pd.to_datetime(df[date_col].unique())).sort_values().reset_index(drop=True)

    folds = []
    start_idx = min_train_days
    for i in range(n_folds):
        val_start_idx = start_idx + i * step_days
        val_end_idx = val_start_idx + val_days
        if val_end_idx > len(unique_dates):
            break

        train_end = unique_dates.iloc[val_start_idx - 1]
        val_start = unique_dates.iloc[val_start_idx]
        val_end = unique_dates.iloc[val_end_idx - 1]

        train_mask = df[date_col] <= train_end
        val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)

        folds.append(
            {
                "train_idx": df.index[train_mask].to_numpy(),
                "val_idx": df.index[val_mask].to_numpy(),
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
            }
        )

    return folds


def evaluate_rolling_model(
    df: pd.DataFrame,
    build_model_fn: Callable,
    feature_cols: list[str],
    target_col: str = "Revenue",
    date_col: str = "Date",
    n_folds: int = 3,
    val_days: int = 30,
    min_train_days: int = 365,
    step_days: int = 30,
    fit_kwargs: dict | None = None,
):
    fit_kwargs = fit_kwargs or {}
    folds = make_rolling_date_folds(
        df=df,
        date_col=date_col,
        n_folds=n_folds,
        val_days=val_days,
        min_train_days=min_train_days,
        step_days=step_days,
    )

    fold_results = []
    all_true = []
    all_pred = []

    for fold_i, fold in enumerate(folds, start=1):
        train_df = df.loc[fold["train_idx"]].copy()
        val_df = df.loc[fold["val_idx"]].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        model = build_model_fn()
        model.fit(X_train, y_train, **fit_kwargs)

        pred = model.predict(X_val)
        m = regression_metrics(y_val.values, pred)

        fold_results.append(
            RollingFoldResult(
                fold=fold_i,
                train_start=train_df[date_col].min(),
                train_end=train_df[date_col].max(),
                val_start=val_df[date_col].min(),
                val_end=val_df[date_col].max(),
                metrics=m,
            )
        )

        all_true.append(y_val.values)
        all_pred.append(pred)

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    overall_metrics = regression_metrics(y_true, y_pred) if len(y_true) else {}

    return fold_results, overall_metrics