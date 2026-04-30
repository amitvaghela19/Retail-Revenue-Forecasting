import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    unique_dates = sorted(df[date_col].unique())
    n_dates = len(unique_dates)

    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    train_cutoff = unique_dates[train_end - 1]
    val_cutoff = unique_dates[val_end - 1]

    train_df = df[df[date_col] <= train_cutoff].copy()
    val_df = df[(df[date_col] > train_cutoff) & (df[date_col] <= val_cutoff)].copy()
    test_df = df[df[date_col] > val_cutoff].copy()

    logger.info("Train cutoff: %s", train_cutoff)
    logger.info("Validation cutoff: %s", val_cutoff)
    logger.info("Train shape: %s", train_df.shape)
    logger.info("Validation shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    return train_df, val_df, test_df