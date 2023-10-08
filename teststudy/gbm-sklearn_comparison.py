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
import pandas as pd
import sys
#own modules 
sys.path.append('../')
from from_scratch.gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X
from repo_utils import repo_path, fontsize_title, fontsize

##################################

Y = testfunction(X, noise = 0.5)

X = pd.DataFrame(X.reshape(-1,1))
Y = np.ravel(Y)

##################################

def gbm_scratch_sklearn_mse_comparison(plot_save) -> None:

    fig, ax = plt.subplots(1)
    fig.set_size_inches(10,6)
    
    learning_rate = 0.2 ; max_depth = 1 ; n_estimators = 100

    val_score = np.zeros(n_estimators, dtype = np.float64)

    sklearn_gbm = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth)
    self_gbm = GradientBoosting(n_trees = n_estimators, learning_rate = learning_rate, max_depth = max_depth, X = X, Y = Y)

    models = [sklearn_gbm, self_gbm]

    for model in models:
        model.fit(X,Y)
        if model == self_gbm:
            y_hat, _ = model.predict(X)
            mse_scratch = mean_squared_error(Y, y_hat)
        else:
            y_hat = model.predict(X)
        
    #print('MSE of model ' + str(model) + str(' : '), round(mse_scratch,3))
    ################################

    for i, Y_pred in enumerate(sklearn_gbm.staged_predict(X)):
        val_score[i] = sklearn_gbm.loss_(Y, Y.reshape(-1, 1))

    ax.plot(np.arange(n_estimators) + 1, sklearn_gbm.train_score_, color = 'salmon', linestyle = '--',label='SKLEARN MSE Deviance')
    ax.hlines(mse_scratch, 0, n_estimators, color = 'cornflowerblue', linestyle = '--', label = 'Scratch MSE Deviance at $n_{trees}$ = ' + str(n_estimators))

    print("deviances from skrach : ", mse_scratch)
    print("deviances of sklearn : ", sklearn_gbm.train_score_[-1])

    ax.text(n_estimators - 10, mse_scratch + 0.1, 'MSE = ' + str(round(mse_scratch,3)), fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))
    ax.set_title('Deviance $\\Delta$ / MSE over boosting iterations $m$ from SKLEARN / Scratch module', fontsize = fontsize_title)
    ax.legend(loc='upper right', fontsize = fontsize)
    ax.set_xlabel('Boosting iterations $m$ / number of trees $n_{trees}$', fontsize = fontsize)
    ax.set_ylabel('Deviance $\Delta$ / MSE ', fontsize = fontsize)
    fig.tight_layout()

    if plot_save == True:
        plt.savefig(repo_path + "/figures/" + "gbm_scratch_sklearn_loss_comparison.png", dpi=300)
    
    plt.show()

gbm_scratch_sklearn_mse_comparison(plot_save = True)
