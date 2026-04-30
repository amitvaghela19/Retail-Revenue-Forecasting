import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date"]).copy()

    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        logger.info("Dropping %d duplicate rows", duplicates)
        df = df.drop_duplicates().copy()

    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Rows before cleaning: %d", before)
    logger.info("Rows after cleaning: %d", len(df))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    required_cols = [
        "Date",
        "Product_Category",
        "Sub_Category",
        "Revenue",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")