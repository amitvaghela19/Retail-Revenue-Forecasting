import pandas as pd

from src.evaluation.metrics import regression_metrics


def compute_group_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (pc, sc), grp in df.groupby(["Product_Category", "Sub_Category"]):
        m = regression_metrics(grp["Revenue"].values, grp[pred_col].values)
        rows.append(
            {
                "Product_Category": pc,
                "Sub_Category": sc,
                "n_rows": len(grp),
                **m,
            }
        )
    return pd.DataFrame(rows)