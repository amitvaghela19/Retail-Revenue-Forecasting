import numpy as np
import pandas as pd


def _to_numeric_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()

    for col in X.columns:
        if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def make_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Revenue",
    group_col: str = "group_id",
    date_col: str = "Date",
    lookback: int = 14,
):
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    X_list = []
    y_list = []
    meta_list = []

    for group_id, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)

        if len(g) <= lookback:
            continue

        Xg = _to_numeric_frame(g, feature_cols).to_numpy(dtype=np.float32)
        yg = g[target_col].to_numpy(dtype=np.float32)

        for i in range(lookback, len(g)):
            X_list.append(Xg[i - lookback:i])
            y_list.append(yg[i])
            meta_list.append(
                {
                    group_col: group_id,
                    date_col: g.loc[i, date_col],
                    "target_index": i,
                }
            )

    if not X_list:
        X = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        meta = pd.DataFrame(columns=[group_col, date_col, "target_index"])
        return X, y, meta

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)
    return X, y, meta


def make_train_val_test_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Revenue",
    group_col: str = "group_id",
    date_col: str = "Date",
    lookback: int = 14,
):
    X_train, y_train, meta_train = make_sequence_arrays(
        train_df, feature_cols, target_col, group_col, date_col, lookback
    )
    X_val, y_val, meta_val = make_sequence_arrays(
        val_df, feature_cols, target_col, group_col, date_col, lookback
    )
    X_test, y_test, meta_test = make_sequence_arrays(
        test_df, feature_cols, target_col, group_col, date_col, lookback
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }