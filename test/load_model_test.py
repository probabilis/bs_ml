from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import plotext

"""
x_min = 0
x_max = 2 * np.pi
N = 100

X = np.linspace(x_min,x_max,N)
X_seq = X.reshape(-1,1)
"""
import sys
from pathlib import Path
parentddir = Path(__file__).resolve().parent

from preprocessing.cross_validators import era_splitting
from utils import loading_dataset, repo_path

df, features, target, eras = loading_dataset()

print(df)

#cb = CatBoostRegressor()

#fn = "CatBoost_{'learning_rate': 0.01, 'max_depth': 1, 'n_estimators': 500, 'rsm': 0.1}.json"

lgbm = LGBMRegressor()

lgbm.train(df[features],df[target])

#cb.load_model(fn, "json")  # load model

Y_pred = lgbm.predict(df[features])
print(Y_pred)

plotext.scatter(Y_pred)
plotext.show()
