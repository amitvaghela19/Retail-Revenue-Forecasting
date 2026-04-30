import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = (
            df.groupby(group_cols)[target_col].shift(lag)
        )

    return df