import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def seasonal_naive_predict(df, target_col: str, season_lag: int = 7):
    return df[f"{target_col}_lag_{season_lag}"].values


def moving_average_predict(df, target_col: str, window: int = 7):
    return df[f"{target_col}_roll_mean_{window}"].values


def build_lightgbm_model(random_state: int = 42, **kwargs):
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    params.update(kwargs)
    return LGBMRegressor(**params)


def build_catboost_model(random_state: int = 42, **kwargs):
    params = dict(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        verbose=0,
        random_seed=random_state,
    )
    params.update(kwargs)
    return CatBoostRegressor(**params)