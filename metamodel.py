"""
    Scrypt for metamodel inference
"""

from catboost import CatBoostRegressor as CatBoost

class Meta():
    def __init__(self, path) -> None:
        self.model = CatBoost()
        self.model.load_model(path)

    def predict(self, X):
        return self.model.predict(X)