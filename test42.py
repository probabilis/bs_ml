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
parentdir = Path(__file__).resolve().parent
sys.path.append(parentdir)
print( parentdir )
from preprocessing.cross_validators import era_splitting
from utils import loading_dataset, repo_path
import gc
df, features, target, eras = loading_dataset()

print(df)

df_, eras_ = era_splitting(df, eras)

del df ; gc.collect()

model = LGBMRegressor()

print(df_)

model.fit(df_[features],df_[target])

#cb.load_model(fn, "json")  # load model

Y_pred = model.predict(df_[features])
print(Y_pred)

plotext.scatter(Y_pred)
plotext.show()
