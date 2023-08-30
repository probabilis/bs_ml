"""
Author: Maximilian Gschaider
MN: 12030366
"""
########################################
#official open-source repositories
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
#own modules 
from gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X

##################################

Y = testfunction(X, noise = 1)

X = X.reshape(-1,1)
Y = np.ravel(Y)

##################################

def gbm_scratch_sklearn_comparison(plot_save) -> None:
    n = 100
    alpha = 0.2
    deep = 1

    sklearn_gbm = GradientBoostingRegressor(n_estimators = n, learning_rate = alpha, max_depth = deep)
    self_gbm = GradientBoosting(n_trees = n, learning_rate = alpha, max_depth = deep, X = X, Y = Y)

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
    plt.title('Deviance / MSE over boosting iterations $n$ from SKLEARN / Scratch module')
    plt.plot(np.arange(n) + 1, sklearn_gbm.train_score_, color = 'salmon', linestyle = '--',label='Training Set Deviance')
    plt.hlines(mse,0,n,color = 'mediumseagreen', linestyle = '--', label = 'Scratch Deviance at n = ' + str(n))
    plt.text(80,1,'MSE = ' + str(round(mse,5)))
    plt.legend(loc='upper right')
    plt.xlabel('Boosting iterations $n$')
    plt.ylabel('Deviance / MSE ')
    plt.tight_layout()
    if plot_save == True:
        plt.savefig("gbm_scratch_sklearn_loss_comparison.png")
    plt.show()

gbm_scratch_sklearn_comparison(plot_save = False)
