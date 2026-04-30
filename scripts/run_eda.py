import pandas as pd

from src.data.clean_data import clean_sales_data
from src.data.load_data import load_sales_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    df = load_sales_data()
    df = clean_sales_data(df)

    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Date min: %s", df["Date"].min())
    logger.info("Date max: %s", df["Date"].max())
    logger.info("Unique products: %d", df["Product"].nunique())
    logger.info("Unique sub-categories: %d", df["Sub_Category"].nunique())
    logger.info("Missing values:\n%s", df.isna().sum())


if __name__ == "__main__":
    main()
    