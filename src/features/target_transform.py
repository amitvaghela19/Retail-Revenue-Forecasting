import numpy as np
import pandas as pd


def add_log_target(df: pd.DataFrame, target_col: str = "Revenue") -> pd.DataFrame:
    df = df.copy()
    df["Revenue_log1p"] = np.log1p(df[target_col].clip(lower=0))
    return df