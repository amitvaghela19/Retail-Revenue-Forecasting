from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, engine="pyarrow")


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")