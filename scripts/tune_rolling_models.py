import json
from pathlib import Path
 
import pandas as pd
import yaml

from src.data.split_data import chronological_split
from src.features.calendar_features import add_calendar_features
from src.features.encoders import GroupLabelEncoder
from src.features.extra_features import add_calendar_plus_features, add_difference_features, add_outlier_flag_features
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.target_transform import add_log_target
from src.models.baselines import build_catboost_model, build_lightgbm_model
from src.training.rolling_validation import evaluate_rolling_model
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
    return X, feature_cols


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

    panel = build_panel(feature_config).dropna().copy()
    train_df, val_df, test_df = chronological_split(
        panel,
        date_col="Date",
        train_ratio=train_config["split"]["train_ratio"],
        val_ratio=train_config["split"]["val_ratio"],
    )

    encoder = GroupLabelEncoder().fit(train_df["group_key"])
    for part in (train_df, val_df, test_df):
        part["group_id"] = encoder.transform(part["group_key"])

    train_val = pd.concat([train_df, val_df], axis=0).sort_values("Date").copy().dropna()
    X_all, feature_cols = prepare_features(train_val)
    train_val[feature_cols] = X_all

    search_spaces = {
        "LightGBM": [
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 500, "max_depth": -1, "subsample": 0.8, "colsample_bytree": 0.8},
            {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 800, "max_depth": -1, "subsample": 0.8, "colsample_bytree": 0.8},
            {"num_leaves": 127, "learning_rate": 0.02, "n_estimators": 1000, "max_depth": 8, "subsample": 0.9, "colsample_bytree": 0.9},
        ],
        "CatBoost": [
            {"depth": 6, "learning_rate": 0.05, "iterations": 600, "l2_leaf_reg": 3},
            {"depth": 8, "learning_rate": 0.03, "iterations": 900, "l2_leaf_reg": 5},
            {"depth": 10, "learning_rate": 0.02, "iterations": 1200, "l2_leaf_reg": 7},
        ],
    }

    results = {"LightGBM": [], "CatBoost": []}

    for params in search_spaces["LightGBM"]:
        def build_lgbm():
            return build_lightgbm_model(random_state=42, **params)

        folds, overall = evaluate_rolling_model(
            train_val,
            build_lgbm,
            feature_cols,
            target_col="Revenue",
            date_col="Date",
            n_folds=3,
            val_days=30,
            min_train_days=365,
            step_days=30,
                    #using verbose gives error in lightgbm, so we will not use it here
        )
        results["LightGBM"].append({"params": params, "overall": overall, "folds": [f.metrics for f in folds]})

    for params in search_spaces["CatBoost"]:
        def build_cat():
            return build_catboost_model(random_state=42, **params)

        folds, overall = evaluate_rolling_model(
            train_val,
            build_cat,
            feature_cols,
            target_col="Revenue",
            date_col="Date",
            n_folds=3,
            val_days=30,
            min_train_days=365,
            step_days=30,
        )
        results["CatBoost"].append({"params": params, "overall": overall, "folds": [f.metrics for f in folds]})

    best_lgbm = min(results["LightGBM"], key=lambda x: x["overall"]["MAE"])
    best_cat = min(results["CatBoost"], key=lambda x: x["overall"]["MAE"])

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/rolling_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump({"LightGBM": results["LightGBM"], "CatBoost": results["CatBoost"], "best": {"LightGBM": best_lgbm, "CatBoost": best_cat}}, f, indent=2)

    summary = pd.DataFrame([
        {"model": "LightGBM", "MAE": best_lgbm["overall"]["MAE"], "RMSE": best_lgbm["overall"]["RMSE"], "WAPE": best_lgbm["overall"]["WAPE"], "SMAPE": best_lgbm["overall"]["SMAPE"]},
        {"model": "CatBoost", "MAE": best_cat["overall"]["MAE"], "RMSE": best_cat["overall"]["RMSE"], "WAPE": best_cat["overall"]["WAPE"], "SMAPE": best_cat["overall"]["SMAPE"]},
    ]).sort_values(["MAE", "WAPE"])
    summary.to_csv("outputs/metrics/rolling_tuning_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
