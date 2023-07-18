from gbm_class import GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import numpy as np
from math import * 


x_min = 0
x_max = 2*pi
N = 100

X = np.linspace(x_min,x_max,N).reshape(-1,1)
Y = np.sin(X)
Y = np.ravel(Y)

##################################

n = 100
alpha = 0.2
deep = 1

sklearn_gbm = GradientBoostingRegressor(n_estimators = n, learning_rate = alpha, max_depth = deep)
self_gbm = GradientBoosting(n_trees = n, learning_rate = alpha, max_depth = deep)

models = [sklearn_gbm, self_gbm]

for model in models:
    model.fit(X,Y)
    mse = mean_squared_error(Y, model.predict(X))
    print('MSE of model ' + str(model) + str(' : '),round(mse,5))

################################

val_score = np.zeros(n, dtype=np.float64)

for i, Y_pred in enumerate(sklearn_gbm.staged_predict(X)):
    val_score[i] = sklearn_gbm.loss_(Y, Y.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.title('Deviance over boosting iterations from SKLEARN module / Scratch')
plt.plot(np.arange(n) + 1, sklearn_gbm.train_score_, color = 'salmon', linestyle = '--',label='Training Set Deviance')
plt.hlines(mse,0,n,color = 'mediumseagreen', linestyle = '--', label = 'Scratch Deviance at n = ' + str(n))
plt.text(80,0.25,'MSE = ' + str(round(mse,5)))
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations i')
plt.ylabel('Deviance d / 1')
plt.tight_layout()
plt.savefig("gbm_scratch_sklearn_loss_comparison.png")
plt.show()
