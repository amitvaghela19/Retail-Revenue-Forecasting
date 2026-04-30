import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.split_data import chronological_split
from src.evaluation.group_metrics import compute_group_metrics
from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_actual_vs_pred, plot_error_distribution, plot_group_metric_bar
from src.features.calendar_features import add_calendar_features
from src.features.encoders import GroupLabelEncoder
from src.features.extra_features import add_calendar_plus_features, add_difference_features, add_outlier_flag_features
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.target_transform import add_log_target
from src.models.baselines import build_catboost_model, build_lightgbm_model, moving_average_predict, seasonal_naive_predict
from src.utils.io import load_parquet
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

FEATURE_DROP_COLS = ["Date", "Revenue", "Revenue_log1p", "Product_Category", "Sub_Category", "group_key"]


def prepare_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in FEATURE_DROP_COLS]
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    y = df["Revenue"].copy()
    return X, y, feature_cols

def build_panel(feature_config):
    panel = load_parquet("data/processed/sales_daily_panel_full.parquet")
    panel["group_key"] = panel["Product_Category"].astype(str) + "__" + panel["Sub_Category"].astype(str)
    panel = add_calendar_features(panel, date_col="Date")
    panel = add_log_target(panel, target_col="Revenue")
    panel = add_lag_features(panel, ["Product_Category", "Sub_Category"], "Revenue", feature_config["lag_features"])
    panel = add_rolling_features(panel, ["Product_Category", "Sub_Category"], "Revenue", feature_config["rolling_windows"])
    panel = add_calendar_plus_features(panel, date_col="Date")
    panel = add_difference_features(panel, ["Product_Category", "Sub_Category"], "Revenue", [7, 28])
    panel = add_outlier_flag_features(panel, ["Product_Category", "Sub_Category"], "Revenue", window=28, z_thresh=3.0)
    return panel


def main():
    set_seed(42)
    with open("configs/feature_config.yaml", "r", encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)
    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    with open("outputs/metrics/rolling_tuning_results.json", "r", encoding="utf-8") as f:
        tuning = json.load(f)

    best_lgbm = tuning["best"]["LightGBM"]["params"]
    best_cat = tuning["best"]["CatBoost"]["params"]

    panel = build_panel(feature_config)
    train_df, val_df, test_df = chronological_split(
        panel,
        date_col="Date",
        train_ratio=train_config["split"]["train_ratio"],
        val_ratio=train_config["split"]["val_ratio"],
    )

    encoder = GroupLabelEncoder().fit(train_df["group_key"])
    for part in (train_df, val_df, test_df):
        part["group_id"] = encoder.transform(part["group_key"])

    full_train = pd.concat([train_df, val_df], axis=0).sort_values("Date").copy().dropna()
    test_df = test_df.dropna().copy()

    X_full_train, y_full_train, feature_cols = prepare_features(full_train)
    X_test, y_test, _ = prepare_features(test_df)

    lgbm = build_lightgbm_model(random_state=42, **best_lgbm)
    lgbm.fit(
        X_full_train,
        y_full_train,
        categorical_feature=[c for c in ["group_id", "dayofweek", "month", "quarter", "year", "is_weekend", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end", "outlier_flag_28"] if c in X_full_train.columns],
    )
    pred_lgbm = lgbm.predict(X_test)

    cat = build_catboost_model(random_state=42, **best_cat)
    cat.fit(
        X_full_train,
        y_full_train,
        cat_features=[X_full_train.columns.get_loc(c) for c in ["group_id", "dayofweek", "month", "quarter", "year", "is_weekend", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end", "outlier_flag_28"] if c in X_full_train.columns],
    )
    pred_cat = cat.predict(X_test)

    pred_sn = seasonal_naive_predict(test_df, "Revenue", season_lag=7)
    pred_ma = moving_average_predict(test_df, "Revenue", window=7)

    results = {
        "SeasonalNaive_test": regression_metrics(y_test.values, pred_sn),
        "MovingAverage_test": regression_metrics(y_test.values, pred_ma),
        "LightGBM_test": regression_metrics(y_test.values, pred_lgbm),
        "CatBoost_test": regression_metrics(y_test.values, pred_cat),
        "LightGBM_params": best_lgbm,
        "CatBoost_params": best_cat,
        "feature_cols": feature_cols,
    }

    test_predictions = test_df[["Date", "Product_Category", "Sub_Category", "Revenue"]].copy()
    test_predictions["pred_seasonal_naive"] = pred_sn
    test_predictions["pred_moving_average"] = pred_ma
    test_predictions["pred_lightgbm"] = pred_lgbm
    test_predictions["pred_catboost"] = pred_cat

    Path("outputs/final").mkdir(parents=True, exist_ok=True)
    Path("outputs/final/figures").mkdir(parents=True, exist_ok=True)

    with open("outputs/final/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    test_predictions.to_csv("outputs/final/test_predictions_final.csv", index=False)

    lgbm_group = compute_group_metrics(test_predictions, "pred_lightgbm")
    cat_group = compute_group_metrics(test_predictions, "pred_catboost")
    lgbm_group.to_csv("outputs/final/lightgbm_group_metrics_final.csv", index=False)
    cat_group.to_csv("outputs/final/catboost_group_metrics_final.csv", index=False)

    plot_actual_vs_pred(test_predictions.head(400), "pred_catboost", "CatBoost Actual vs Predicted", "outputs/final/figures/catboost_actual_vs_pred.png")
    plot_error_distribution(test_predictions, "pred_catboost", "CatBoost Error Distribution", "outputs/final/figures/catboost_error_distribution.png")
    plot_group_metric_bar(cat_group, "MAE", "Top Group MAE for CatBoost", "outputs/final/figures/catboost_group_mae.png")

    logger.info("Saved final train+validation retrain results.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()