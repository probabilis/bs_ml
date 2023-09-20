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

plot_bo(save_plot = True)

filenames = [
    "bo_iterations_ip=10_ni=100_2023-09-18.csv",
    "bo_iterations_ip=10_ni=100_2023-09-19.csv",
    "bo_iterations_ip=10_ni=200_2023-09-19.csv"
]

def plot_hyperparameter_scatter_plot(save_plot) -> None:
    bo_iterations = pd.read_csv(repo_path + "/from_scratch/" + filenames[2])
    target = bo_iterations["target"]

    cbt = [] ; lr = [] ; md = [] ; nt = []

    for i in range(0,len(target)):
        dict_ = eval(bo_iterations["params"][i])
        cbt.append( dict_["colsample_bytree"] )
        lr.append( dict_["learning_rate"])
        md.append(dict_["max_depth"])
        nt.append(dict_["n_estimators"])

    data = {"target":bo_iterations["target"], "colsample_bytree": cbt, "learning_rate": lr,"n_estimators": nt, "max_depth": md}
    df = pd.DataFrame(data = data)
    #print(df)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12,6)

    fig.suptitle("scatter plot of objective function over the hyperparameter space", fontsize = fontsize_title)

    cm = plt.cm.get_cmap('Spectral')
    im_1 = axs[0].scatter(df["max_depth"], df["n_estimators"], c = df["target"], cmap = cm, s = 100)
    axs[0].set_xlabel("max. depth / $d_{max}$", fontsize = fontsize)
    axs[0].set_ylabel("nr. of trees / $n_{trees}$ = m", fontsize = fontsize)
    fig.colorbar(im_1, ax = axs[0])

    im_2 = axs[1].scatter(df["learning_rate"], df["colsample_bytree"], c= df["target"], cmap = cm, s = 100)
    axs[1].set_xlabel("learning rate / $\\nu$", fontsize = fontsize)
    axs[1].set_ylabel("colsample by tree / $\\epsilon$", fontsize = fontsize)
    fig.colorbar(im_2, ax = axs[1])

    fig.tight_layout()
    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "gbm_with_decision_tree_hyperparameters_scatter_plot_2.png", dpi=300)
    plt.show()


#plot_hyperparameter_scatter_plot(save_plot = True)
