"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
import sys
import matplotlib.pyplot as plt
#own modules
sys.path.append('../')
from from_scratch.decision_regression_tree_from_scratch import DecisionTreeRegressorScratch
from testfunction import testfunction, X
from repo_utils import repo_path, fontsize_title, fontsize

###################################

max_depth = 2

Y = testfunction(X, noise = 0.5)
X = X.reshape(-1,1)

###################################

def dt_scratch_sklearn_comparison(plot_save) -> None:

    fig, axs = plt.subplots(1)
    fig.set_size_inches(12,8)

    model = DecisionTreeRegressor(max_depth = max_depth)
    model.fit(X,Y)
    y_pred = model.predict(X)

    model_scratch = DecisionTreeRegressorScratch(pd.DataFrame(X), Y.tolist(), max_depth = max_depth)
    model_scratch.grow_tree()
    y_pred_scratch = model_scratch.fit(pd.DataFrame(X))

    ###################################



    fig.suptitle("Decision Tree Regressor from Scratch comparison with SKLEARN module / max. depth $d_{max}$ = " + str(max_depth), fontsize = fontsize_title)
    
    axs.scatter(X, Y, color = 'gray')
    axs.plot(X, y_pred, color = 'salmon', linestyle = ':', linewidth = 3, label = 'SKLEARN')
    axs.plot(X, y_pred_scratch, color = 'mediumseagreen', linestyle = '--',linewidth = 3, label = 'SCRATCH')
    axs.set_xlabel("$x$")
    axs.set_ylabel("$F(x)$")
    axs.legend(fontsize = fontsize)
    
    fig.tight_layout()
    if plot_save == True:
        plt.savefig(repo_path + "/figures/" + f"decision_tree_regressor_comparison_md={max_depth}.png", dpi=300)

    plt.show()

    #import graphviz
    #dot_data = export_graphviz(model, out_file = None)
    #graph = graphviz.Source(dot_data, filename = 'test.gv', format = 'png')
    #graph.view()

    fig, axs = plt.subplots(1)


    tree.plot_tree(model, fontsize = fontsize)
    fig = plt.gcf()
    fig.suptitle("Decision Tree splitting structure / max. depth $d_{max}$ = " + str(max_depth), fontsize = fontsize_title)
    fig.set_size_inches(12,8)
    fig.tight_layout()
    if plot_save == True:
        plt.savefig(repo_path + "/figures/" + f"decision_tree_splitting_structure_md={max_depth}.png", dpi=300)
    plt.show()

dt_scratch_sklearn_comparison(plot_save = True)