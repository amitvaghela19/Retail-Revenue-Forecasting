from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_actual_vs_pred(df: pd.DataFrame, pred_col: str, title: str, out_path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Revenue"], label="Actual", linewidth=2)
    plt.plot(df["Date"], df[pred_col], label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_error_distribution(df: pd.DataFrame, pred_col: str, title: str, out_path: str):
    err = df["Revenue"] - df[pred_col]
    plt.figure(figsize=(10, 5))
    sns.histplot(err, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Error")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_group_metric_bar(group_df: pd.DataFrame, metric: str, title: str, out_path: str):
    top = group_df.sort_values(metric, ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top, x=metric, y="Sub_Category", hue="Product_Category", dodge=False)
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()