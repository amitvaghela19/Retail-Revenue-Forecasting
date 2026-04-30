import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)

    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(window=window).mean())
        )

        df[f"{target_col}_roll_std_{window}"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(window=window).std())
        )

        df[f"{target_col}_roll_min_{window}"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(window=window).min())
        )

        df[f"{target_col}_roll_max_{window}"] = (
            df.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(window=window).max())
        )

    return df