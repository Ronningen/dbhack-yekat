"""
    Scrypt for metamodel inference
"""

from catboost import CatBoostRegressor as CatBoost
import numpy as np


class Meta():
    def __init__(self, path) -> None:
        self.model = CatBoost()
        self.model.load_model(path)

    def predict(self, X):
        return self.model.predict(X)
    

def std_predict(X):
    return np.tanh((np.std(np.array(X), axis=0)/3).sum())


if __name__=='__main__':
    print(std_predict([[0,0,0],[0,1,2],[0,-1,-2]]))