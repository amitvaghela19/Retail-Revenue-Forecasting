import numpy as np
import pandas as pd


def add_calendar_plus_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["is_quarter_start"] = df[date_col].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df[date_col].dt.is_quarter_end.astype(int)
    df["is_year_start"] = df[date_col].dt.is_year_start.astype(int)
    df["is_year_end"] = df[date_col].dt.is_year_end.astype(int)
    return df


def add_difference_features(df: pd.DataFrame, group_cols: list[str], target_col: str, diffs: list[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)
    for d in diffs:
        df[f"{target_col}_diff_{d}"] = df.groupby(group_cols)[target_col].transform(lambda s: s - s.shift(d))
    return df


def add_outlier_flag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    window: int = 28,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"]).reset_index(drop=True)
    rolling_mean = df.groupby(group_cols)[target_col].transform(lambda s: s.shift(1).rolling(window=window).mean())
    rolling_std = df.groupby(group_cols)[target_col].transform(lambda s: s.shift(1).rolling(window=window).std())
    z = (df[target_col] - rolling_mean) / rolling_std.replace(0, np.nan)
    df["outlier_flag_28"] = (z.abs() > z_thresh).fillna(False).astype(int)
    return df