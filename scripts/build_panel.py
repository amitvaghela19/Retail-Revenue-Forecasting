import yaml

from src.data.build_panel import aggregate_daily_panel, reindex_full_daily_panel
from src.data.clean_data import clean_sales_data, validate_schema
from src.data.load_data import load_sales_data
from src.utils.io import ensure_dir, save_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    print("START: build_panel.py is running")

    with open("configs/data_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ensure_dir(config["interim_dir"])
    ensure_dir(config["processed_dir"])

    df = load_sales_data()
    validate_schema(df)
    df = clean_sales_data(df)

    save_parquet(df, f"{config['interim_dir']}/sales_cleaned.parquet")
    print("Saved cleaned parquet")

    panel = aggregate_daily_panel(
        df=df,
        group_cols=config["group_cols"],
        target_col=config["target_col"],
    )
    save_parquet(panel, f"{config['interim_dir']}/sales_daily_panel.parquet")
    print("Saved daily aggregated panel")

    full_panel = reindex_full_daily_panel(
        panel=panel,
        group_cols=config["group_cols"],
        target_col=config["target_col"],
    )
    save_parquet(full_panel, f"{config['processed_dir']}/sales_daily_panel_full.parquet")
    print("Saved full reindexed panel")

    logger.info("Panel build complete.")
    print("DONE: build_panel.py finished")


if __name__ == "__main__":
    main()