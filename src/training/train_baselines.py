import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.split_data import chronological_split
from src.evaluation.metrics import regression_metrics
from src.features.calendar_features import add_calendar_features
from src.features.encoders import GroupLabelEncoder
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.target_transform import add_log_target
from src.models.baselines import build_catboost_model, build_lightgbm_model
from src.models.baselines import moving_average_predict, seasonal_naive_predict
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
    y = df["Revenue"].copy()
    return X, y, feature_cols


def group_metrics(df: pd.DataFrame, pred_col: str):
    rows = []
    for g, grp in df.groupby(["Product_Category", "Sub_Category"]):
        m = regression_metrics(grp["Revenue"].values, grp[pred_col].values)
        rows.append(
            {
                "Product_Category": g[0],
                "Sub_Category": g[1],
                **m,
                "n_rows": len(grp),
            }
        )
    return pd.DataFrame(rows)


def main():
    set_seed(42)

    with open("configs/feature_config.yaml", "r", encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)

    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

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

    train_df, val_df, test_df = chronological_split(
        panel,
        date_col="Date",
        train_ratio=train_config["split"]["train_ratio"],
        val_ratio=train_config["split"]["val_ratio"],
    )

    encoder = GroupLabelEncoder().fit(train_df["group_key"])
    train_df["group_id"] = encoder.transform(train_df["group_key"])
    val_df["group_id"] = encoder.transform(val_df["group_key"])
    test_df["group_id"] = encoder.transform(test_df["group_key"])

    train_df = train_df.dropna().copy()
    val_df = val_df.dropna().copy()
    test_df = test_df.dropna().copy()

    results = {}

    val_true = val_df["Revenue"].values
    test_true = test_df["Revenue"].values

    val_pred_sn = seasonal_naive_predict(val_df, "Revenue", season_lag=7)
    test_pred_sn = seasonal_naive_predict(test_df, "Revenue", season_lag=7)
    results["SeasonalNaive_val"] = regression_metrics(val_true, val_pred_sn)
    results["SeasonalNaive_test"] = regression_metrics(test_true, test_pred_sn)

    val_pred_ma = moving_average_predict(val_df, "Revenue", window=7)
    test_pred_ma = moving_average_predict(test_df, "Revenue", window=7)
    results["MovingAverage_val"] = regression_metrics(val_true, val_pred_ma)
    results["MovingAverage_test"] = regression_metrics(test_true, test_pred_ma)

    X_train, y_train, feature_cols = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    X_test, y_test, _ = prepare_features(test_df)

    categorical_cols = ["group_id", "dayofweek", "month", "quarter", "year", "is_weekend"]

    lgbm = build_lightgbm_model()
    lgbm.fit(
        X_train,
        y_train,
        categorical_feature=[c for c in categorical_cols if c in X_train.columns],
    )
    val_pred_lgbm = lgbm.predict(X_val)
    test_pred_lgbm = lgbm.predict(X_test)
    results["LightGBM_val"] = regression_metrics(y_val, val_pred_lgbm)
    results["LightGBM_test"] = regression_metrics(y_test, test_pred_lgbm)

    cat = build_catboost_model()
    cat.fit(
        X_train,
        y_train,
        cat_features=[X_train.columns.get_loc(c) for c in categorical_cols if c in X_train.columns],
    )
    val_pred_cat = cat.predict(X_val)
    test_pred_cat = cat.predict(X_test)
    results["CatBoost_val"] = regression_metrics(y_val, val_pred_cat)
    results["CatBoost_test"] = regression_metrics(y_test, test_pred_cat)

    test_predictions = test_df[["Date", "Product_Category", "Sub_Category", "Revenue"]].copy()
    test_predictions["pred_seasonal_naive"] = test_pred_sn
    test_predictions["pred_moving_average"] = test_pred_ma
    test_predictions["pred_lightgbm"] = test_pred_lgbm
    test_predictions["pred_catboost"] = test_pred_cat

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

    with open("outputs/metrics/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("outputs/metrics/baseline_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    test_predictions.to_csv("outputs/metrics/test_predictions.csv", index=False)

    group_metrics(test_predictions, "pred_lightgbm").to_csv(
        "outputs/metrics/lightgbm_group_metrics.csv", index=False
    )
    group_metrics(test_predictions, "pred_catboost").to_csv(
        "outputs/metrics/catboost_group_metrics.csv", index=False
    )

    logger.info("Saved upgraded baseline metrics and group metrics.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()