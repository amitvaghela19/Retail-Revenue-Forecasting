import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def aggregate_daily_panel(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "Revenue",
) -> pd.DataFrame:
    panel = (
        df.groupby(["Date"] + group_cols, as_index=False)[target_col]
        .sum()
        .sort_values(["Date"] + group_cols)
        .reset_index(drop=True)
    )
    logger.info("Aggregated panel shape: %s", panel.shape)
    return panel


def reindex_full_daily_panel(
    panel: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "Revenue",
) -> pd.DataFrame:
    min_date = panel["Date"].min()
    max_date = panel["Date"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")

    frames = []
    for keys, grp in panel.groupby(group_cols):
        grp = grp.copy()
        grp = grp.set_index("Date").reindex(full_dates)
        grp.index.name = "Date"
        grp[target_col] = grp[target_col].fillna(0)

        if not isinstance(keys, tuple):
            keys = (keys,)

        for col, val in zip(group_cols, keys):
            grp[col] = val

        frames.append(grp.reset_index())

    full_panel = pd.concat(frames, ignore_index=True)
    full_panel = full_panel.rename(columns={"index": "Date"})
    full_panel = full_panel.sort_values(group_cols + ["Date"]).reset_index(drop=True)

    logger.info("Reindexed full panel shape: %s", full_panel.shape)
    return full_panel