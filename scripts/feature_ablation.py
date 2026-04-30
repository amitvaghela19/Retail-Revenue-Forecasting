import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.split_data import chronological_split
from src.features.calendar_features import add_calendar_features
from src.features.encoders import GroupLabelEncoder
from src.features.extra_features import (
    add_calendar_plus_features,
    add_difference_features,
    add_outlier_flag_features,
)
from src.features.lag_features import add_lag_features
from src.features.outlier_treatment import TrainOnlyWinsorizer
from src.features.rolling_features import add_rolling_features
from src.features.target_transform import add_log_target
from src.models.baselines import build_catboost_model
from src.training.rolling_validation import evaluate_rolling_model
from src.utils.io import load_parquet
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

FEATURE_DROP_COLS = [
    "Date",
    "Revenue",
    "Revenue_log1p",
    "Product_Category",
    "Sub_Category",
    "group_key",
]


def prepare_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in FEATURE_DROP_COLS]
    X = df[feature_cols].copy()
    object_cols = X.select_dtypes(include=["object"]).columns
    for col in object_cols:
        X[col] = X[col].astype("category")
    return X, feature_cols


def build_panel(feature_config, blocks):
    panel = load_parquet("data/processed/sales_daily_panel_full.parquet")
    panel["group_key"] = panel["Product_Category"].astype(str) + "__" + panel["Sub_Category"].astype(str)

    panel = add_calendar_features(panel, date_col="Date")
    panel = add_log_target(panel, target_col="Revenue")

    panel = add_lag_features(
        df=panel,
        group_cols=["Product_Category", "Sub_Category"],
        target_col="Revenue",
        lags=feature_config["lag_features"],
    )
    panel = add_rolling_features(
        df=panel,
        group_cols=["Product_Category", "Sub_Category"],
        target_col="Revenue",
        windows=feature_config["rolling_windows"],
    )

    if "calendar_plus" in blocks:
        panel = add_calendar_plus_features(panel, date_col="Date")

    if "diffs" in blocks:
        panel = add_difference_features(
            panel,
            group_cols=["Product_Category", "Sub_Category"],
            target_col="Revenue",
            diffs=[7, 28],
        )

    if "outlier_flag" in blocks:
        panel = add_outlier_flag_features(
            panel,
            group_cols=["Product_Category", "Sub_Category"],
            target_col="Revenue",
            window=28,
            z_thresh=3.0,
        )

    return panel


def main():
    set_seed(42)

    with open("configs/feature_config.yaml", "r", encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)

    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    experiments = {
        "base": [],
        "base_calendar_plus": ["calendar_plus"],
        "base_diffs": ["diffs"],
        "base_outlier_flag": ["outlier_flag"],
        "base_calendar_plus_diffs": ["calendar_plus", "diffs"],
        "base_calendar_plus_outlier_flag": ["calendar_plus", "outlier_flag"],
        "base_all": ["calendar_plus", "diffs", "outlier_flag"],
        "base_all_winsorized": ["calendar_plus", "diffs", "outlier_flag", "winsorize"],
    }

    results = []

    best_cat_params = {
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 600,
        "l2_leaf_reg": 3,
    }

    for exp_name, blocks in experiments.items():
        logger.info("Running experiment: %s", exp_name)

        panel = build_panel(feature_config, blocks)

        train_df, val_df, test_df = chronological_split(
            panel,
            date_col="Date",
            train_ratio=train_config["split"]["train_ratio"],
            val_ratio=train_config["split"]["val_ratio"],
        )

        encoder = GroupLabelEncoder().fit(train_df["group_key"])
        for part in (train_df, val_df, test_df):
            part["group_id"] = encoder.transform(part["group_key"])

        train_val = pd.concat([train_df, val_df], axis=0).sort_values("Date").copy()
        train_val = train_val.dropna().copy()

        if "winsorize" in blocks:
            wz = TrainOnlyWinsorizer(lower_q=0.01, upper_q=0.99)
            train_val["Revenue"] = wz.fit_transform(train_val["Revenue"])

        X_all, feature_cols = prepare_features(train_val)
        train_val[feature_cols] = X_all

        def build_cat():
            return build_catboost_model(random_state=42, **best_cat_params)

        fold_results, overall = evaluate_rolling_model(
            df=train_val,
            build_model_fn=build_cat,
            feature_cols=feature_cols,
            target_col="Revenue",
            date_col="Date",
            n_folds=3,
            val_days=30,
            min_train_days=365,
            step_days=30,
        )

        results.append(
            {
                "experiment": exp_name,
                "blocks": blocks,
                "overall": overall,
                "folds": [fr.metrics for fr in fold_results],
                "n_features": len(feature_cols),
            }
        )

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/feature_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = pd.DataFrame(
        [
            {
                "experiment": r["experiment"],
                "n_features": r["n_features"],
                "MAE": r["overall"].get("MAE"),
                "RMSE": r["overall"].get("RMSE"),
                "WAPE": r["overall"].get("WAPE"),
                "SMAPE": r["overall"].get("SMAPE"),
            }
            for r in results
        ]
    ).sort_values(["MAE", "WAPE"])

    summary.to_csv("outputs/metrics/feature_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()