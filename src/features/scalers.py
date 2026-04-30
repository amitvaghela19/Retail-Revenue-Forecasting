import numpy as np
from sklearn.preprocessing import StandardScaler


class TargetScaler:
    def __init__(self, use_log1p: bool = True):
        self.use_log1p = use_log1p
        self.scaler = StandardScaler()

    def fit(self, y):
        y = np.asarray(y).reshape(-1, 1)
        if self.use_log1p:
            y = np.log1p(y)
        self.scaler.fit(y)
        return self

    def transform(self, y):
        y = np.asarray(y).reshape(-1, 1)
        if self.use_log1p:
            y = np.log1p(y)
        return self.scaler.transform(y).ravel()

    def inverse_transform(self, y):
        y = np.asarray(y).reshape(-1, 1)
        y = self.scaler.inverse_transform(y).ravel()
        if self.use_log1p:
            y = np.expm1(y)
        return y