from catboost import CatBoostRegressor
import numpy as np
import matplotlib.pyplot as plt
import plotext

x_min = 0
x_max = 2 * np.pi
N = 100

X = np.linspace(x_min,x_max,N)
X_seq = X.reshape(-1,1)

cb = CatBoostRegressor()

fn = "CatBoost_{'learning_rate': 0.01, 'max_depth': 1, 'n_estimators': 500, 'rsm': 0.1}.json"

cb.load_model(fn, "json")  # load model

Y_pred = cb.predict(X_seq)
print(Y_pred)

plotext.scatter(X,Y_pred)
plotext.show()
