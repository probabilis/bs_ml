"""
Author: Maximilian Gschaider
MN: 12030366
"""
########################################
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../')
from from_scratch.gradient_boosting import GradientBoosting
from testfunction import testfunction, X
from repo_utils import repo_path, fontsize, fontsize_title

########################################
#testfunction
Y = testfunction(X, noise = 0.5)
Y_ = testfunction(X, noise = 0)

X_0 = X.reshape(-1,1)

X_0 = pd.DataFrame(X_0)

########################################
#hyperparameters

filename = "params_bayes_testfunction_ip=10_ni=200_2023-09-19.csv"
path = repo_path + "/models/" + filename

params_gbm = pd.read_csv(path).to_dict(orient = "list")
params_gbm.pop("Unnamed: 0")

max_depth = params_gbm['max_depth'][0]
learning_rate = params_gbm['learning_rate'][0]
colsample_bytree = params_gbm['colsample_bytree'][0]
n_trees = int(round(params_gbm['n_estimators'][0],1))

########################################

def plot_bo(save_plot) -> None:


    fig, ax = plt.subplots(1)
    fig.set_size_inches(12,8)
    fig.suptitle('Gradient Boosting with Decision Tree Regressor from Scratch with best hyperparameter configuration', fontsize = fontsize_title)

    colors = ['mediumseagreen','lightskyblue',"mediumpurple",'salmon',"palevioletred"]

    gbm = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
    gbm.fit(X_0, Y)
    y_hat, _ = gbm.predict(X_0)
    ax.plot(X, y_hat, color = colors[1], label = "$\hat{y}(x)$", linewidth = 5)
    ax.set_title('$d_{max}$ = ' + str(max_depth) + '/ $n_{trees}$ = ' + str(n_trees) + '/ $\\nu$ = ' + str(learning_rate) + '/ $\\epsilon$ = ' + str(colsample_bytree), loc = 'left', pad=10, fontsize = fontsize)
    ax.scatter(X,Y,color = 'gray', marker='o', edgecolors='k', s=18, label = "sample points")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.plot(X, Y_, color = colors[3], linestyle = "--", linewidth = 3, label = "F(x)")
    ax.legend(fontsize = fontsize)

    fig.tight_layout()
    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "gbm_with_decision_tree_best_parameters.png", dpi=300)
    plt.show()

plot_bo(save_plot = False)