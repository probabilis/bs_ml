"""
Author: Maximilian Gschaider
MN: 12030366
"""
########################################
#official open-source repositories
import sys
import matplotlib.pyplot as plt
import pandas as pd
#own modules
from gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X
sys.path.append('../')
from bs_ml.utils import repo_path, fontsize, fontsize_title

########################################
#testfunction
Y = testfunction(X, noise = 0.5)
Y_ = testfunction(X, noise = 0)

X_0 = X.reshape(-1,1)

########################################
#hyperparameters

filename = "params_bayes_testfunction_ip=10_ni=100_2023-09-18.csv"
path = repo_path + "/models/" + filename
params_gbm = pd.read_csv(path)

params = ['max_depth','learning_rate','colsample_bytree','n_estimators']

params_gbm = params_gbm.to_dict(orient = "list")
params_gbm.pop("Unnamed: 0")

max_depth = params_gbm[params[0]][0]
learning_rate = params_gbm[params[1]][0]
colsample_bytree = params_gbm[params[2]][0]
n_trees = int(round(params_gbm[params[3]][0],1))

########################################

def plot_bo(save_plot) -> None:


    fig, ax = plt.subplots(1)
    fig.set_size_inches(12,8)
    fig.suptitle('Gradient Boosting with Decision Tree Regressor from Scratch with best hyperparameter combination', fontsize = fontsize_title)

    colors = ['mediumseagreen','lightskyblue',"mediumpurple",'salmon',"palevioletred"]

    gbm = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
    gbm.fit(X_0,Y)
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

#plot_bo(save_plot = True)

bo_iterations = pd.read_csv(repo_path + "/from_scratch/" + "bo_iterations.csv")
target = bo_iterations["target"]
#dict = bo_iterations["params"]


cbt = []
lr = []
md = []
nt = []


for i in range(0,len(target)):
    dict_ = eval(bo_iterations["params"][i])
    cbt.append( dict_["colsample_bytree"] )
    lr.append( dict_["learning_rate"])
    md.append(dict_["max_depth"])
    nt.append(dict_["n_estimators"])

data = {"target":bo_iterations["target"], "colsample_bytree": cbt, "learning_rate": lr,"n_estimators": nt, "max_depth": md}
df = pd.DataFrame(data = data)
print(df)

import matplotlib.pyplot as plt
from matplotlib import cm

fig, axs = plt.subplots(4, 1)
fig.set_size_inches(16,12)

A = df["max_depth"]
X = df["learning_rate"]
Y = df["n_estimators"]
Z = df["colsample_bytree"]

Q = [A, X, Y, Z]

for i in range(len(Q)):
    axs[i].plot(Q[i], df["target"])
plt.show()