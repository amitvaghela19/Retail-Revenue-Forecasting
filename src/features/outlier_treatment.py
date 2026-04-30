import pandas as pd


class TrainOnlyWinsorizer:
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_ = None
        self.upper_ = None

    def fit(self, y: pd.Series):
        self.lower_ = y.quantile(self.lower_q)
        self.upper_ = y.quantile(self.upper_q)
        return self

    def transform(self, y: pd.Series):
        return y.clip(lower=self.lower_, upper=self.upper_)

    def fit_transform(self, y: pd.Series):
        self.fit(y)
        return self.transform(y)