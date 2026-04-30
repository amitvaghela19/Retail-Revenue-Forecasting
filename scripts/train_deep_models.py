import json
import sys
from pathlib import Path

import pandas as pd
import yaml
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_data import chronological_split
from src.evaluation.metrics import regression_metrics
from src.features.calendar_features import add_calendar_features
from src.features.encoders import GroupLabelEncoder
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.target_transform import add_log_target
from src.models.gru import build_gru_model
from src.models.lstm import build_lstm_model
from src.models.transformer import build_transformer_model
from src.training.datasets import make_train_val_test_sequences
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


def prepare_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in FEATURE_DROP_COLS]


def scale_numeric_cols(train_df, val_df, test_df, feature_cols):
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c]) and c != "group_id"
    ]
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, val_df, test_df


def fit_and_save_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    meta_val,
    meta_test,
    train_config,
):
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_config.get("deep_learning", {}).get("patience", 5),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=train_config.get("deep_learning", {}).get("lr_patience", 3),
            factor=0.5,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/models/{model_name.lower()}_best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=train_config.get("deep_learning", {}).get("epochs", 30),
        batch_size=train_config.get("deep_learning", {}).get("batch_size", 64),
        callbacks=callbacks,
        verbose=1,
    )

    val_pred = model.predict(X_val).ravel()
    test_pred = model.predict(X_test).ravel()

    results = {
        f"{model_name}_val": {k: float(v) for k, v in regression_metrics(y_val, val_pred).items()},
        f"{model_name}_test": {k: float(v) for k, v in regression_metrics(y_test, test_pred).items()},
    }

    with open(f"outputs/metrics/{model_name.lower()}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history.history).to_csv(
        f"outputs/metrics/{model_name.lower()}_training_history.csv", index=False
    )

    val_out = meta_val.copy()
    val_out["y_true"] = y_val
    val_out["y_pred"] = val_pred
    val_out.to_csv(f"outputs/metrics/{model_name.lower()}_val_predictions.csv", index=False)

    test_out = meta_test.copy()
    test_out["y_true"] = y_test
    test_out["y_pred"] = test_pred
    test_out.to_csv(f"outputs/metrics/{model_name.lower()}_test_predictions.csv", index=False)

    return results


def main():
    set_seed(42)

    with open("configs/feature_config.yaml", "r", encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)

    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    panel = load_parquet("data/processed/sales_daily_panel_full.parquet")
    panel["group_key"] = (
        panel["Product_Category"].astype(str) + "__" + panel["Sub_Category"].astype(str)
    )

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

    feature_cols = prepare_feature_columns(train_df)
    train_df, val_df, test_df = scale_numeric_cols(train_df, val_df, test_df, feature_cols)

    lookback = train_config.get("deep_learning", {}).get("lookback", 14)

    seqs = make_train_val_test_sequences(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col="Revenue",
        group_col="group_id",
        date_col="Date",
        lookback=lookback,
    )

    X_train = seqs["X_train"]
    y_train = seqs["y_train"]
    X_val = seqs["X_val"]
    y_val = seqs["y_val"]
    X_test = seqs["X_test"]
    y_test = seqs["y_test"]
    meta_val = seqs["meta_val"]
    meta_test = seqs["meta_test"]

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_val shape: %s", X_val.shape)
    logger.info("X_test shape: %s", X_test.shape)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty sequence arrays found: "
            f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}. "
            f"Try reducing lookback or checking split sizes."
        )

    lstm_model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=train_config.get("deep_learning", {}).get("lstm_units", 64),
        dropout=train_config.get("deep_learning", {}).get("dropout", 0.2),
        dense_units=train_config.get("deep_learning", {}).get("dense_units", 32),
        learning_rate=train_config.get("deep_learning", {}).get("learning_rate", 1e-3),
    )
    lstm_results = fit_and_save_model(
        model_name="LSTM",
        model=lstm_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        meta_val=meta_val,
        meta_test=meta_test,
        train_config=train_config,
    )

    gru_model = build_gru_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        gru_units=train_config.get("deep_learning", {}).get("gru_units", 64),
        dropout=train_config.get("deep_learning", {}).get("dropout", 0.2),
        dense_units=train_config.get("deep_learning", {}).get("dense_units", 32),
        learning_rate=train_config.get("deep_learning", {}).get("learning_rate", 1e-3),
    )
    gru_results = fit_and_save_model(
        model_name="GRU",
        model=gru_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        meta_val=meta_val,
        meta_test=meta_test,
        train_config=train_config,
    )

    transformer_model = build_transformer_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        d_model=train_config.get("deep_learning", {}).get("transformer_d_model", 64),
        num_heads=train_config.get("deep_learning", {}).get("transformer_num_heads", 4),
        ff_dim=train_config.get("deep_learning", {}).get("transformer_ff_dim", 128),
        num_blocks=train_config.get("deep_learning", {}).get("transformer_num_blocks", 2),
        dropout=train_config.get("deep_learning", {}).get("dropout", 0.15),
        dense_units=train_config.get("deep_learning", {}).get("dense_units", 32),
        learning_rate=train_config.get("deep_learning", {}).get("learning_rate", 1e-3),
    )
    transformer_results = fit_and_save_model(
        model_name="Transformer",
        model=transformer_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        meta_val=meta_val,
        meta_test=meta_test,
        train_config=train_config,
    )

    summary = {
        "LSTM": lstm_results,
        "GRU": gru_results,
        "Transformer": transformer_results,
    }

    with open("outputs/metrics/deep_learning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()