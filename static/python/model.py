import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Model:
    def __init__(self):
        self.lr = None
        self.rf = None

    def initialize(self,
                   df: pd.DataFrame,
                   features: list,
                   target: str):

        X = df[features]
        y = df[target]

        self.lr = LinearRegression().fit(X, y)
        self.rf = RandomForestRegressor(n_estimators=100).fit(X, y)

    def lr_predict(self,
                   X: pd.DataFrame):
       return self.lr.predict(X)

    def rf_predict(self,
                   X: pd.DataFrame):
       return self.rf.predict(X)
