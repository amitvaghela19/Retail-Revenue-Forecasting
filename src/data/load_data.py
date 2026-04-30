from pathlib import Path

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sales_data(config_path: str = "configs/data_config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    raw_path = Path(config["raw_data_path"])

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)

    logger.info("Loaded shape: %s", df.shape)
    return df