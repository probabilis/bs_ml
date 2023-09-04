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
import sys
#own modules 
from gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X
sys.path.append('../')
from bs_ml.utils import repo_path, fontsize_title, fontsize

##################################

Y = testfunction(X, noise = 0.5)

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
        if model == self_gbm:
            y_hat, _ = model.predict(X)
        else:
            y_hat = model.predict(X)
        mse = mean_squared_error(Y, y_hat)
        print('MSE of model ' + str(model) + str(' : '), round(mse,3))

    ################################

    val_score = np.zeros(n, dtype=np.float64)

    for i, Y_pred in enumerate(sklearn_gbm.staged_predict(X)):
        val_score[i] = sklearn_gbm.loss_(Y, Y.reshape(-1, 1))

    plt.figure(figsize=(12, 8))
    plt.title('Deviance / MSE over boosting iterations $m$ from SKLEARN / Scratch module', fontsize = fontsize_title)
    plt.plot(np.arange(n) + 1, sklearn_gbm.train_score_, color = 'salmon', linestyle = '--',label='SKLEARN MSE Deviance')
    plt.hlines(mse, 0, n, color = 'mediumseagreen', linestyle = '--', label = 'Scratch MSE Deviance at n = ' + str(n))
    plt.text(86, 1.1, 'MSE = ' + str(round(mse,3)), fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
    plt.legend(loc='upper right', fontsize = fontsize)
    plt.xlabel('Boosting iterations $m$ / number of trees $n_{trees}$', fontsize = fontsize)
    plt.ylabel('Deviance / MSE ', fontsize = fontsize)
    plt.tight_layout()
    if plot_save == True:
        plt.savefig(repo_path + "/figures/" + "gbm_scratch_sklearn_loss_comparison.png", dpi=300)

    plt.show()

gbm_scratch_sklearn_comparison(plot_save = True)
