import pandas as pd


class GroupLabelEncoder:
    def __init__(self):
        self.mapping = {}
        self.inverse_mapping = {}

    def fit(self, values: pd.Series):
        unique_vals = sorted(values.astype(str).unique())
        self.mapping = {v: i for i, v in enumerate(unique_vals)}
        self.inverse_mapping = {i: v for v, i in self.mapping.items()}
        return self

    def transform(self, values: pd.Series):
        values = values.astype(str)
        return values.map(self.mapping).astype(int)

    def fit_transform(self, values: pd.Series):
        self.fit(values)
        return self.transform(values)