import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return np.mean(diff) * 100


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "SMAPE": smape(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }