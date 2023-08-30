"""
Author: Maximilian Gschaider
MN: 12030366
"""
########################################
#official open-source repositories
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt

#own modules
from gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X

########################################

repo_path = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "bs_ml")

########################################
#testfunction
Y = testfunction(X, noise = 0.5)
Y_ = testfunction(X, noise = 0)

########################################
#hyperparameters
learning_rate = 0.1
n_trees = 1
max_depth = 1

X_0 = X.reshape(-1,1)

########################################

def gbm_iterations(save_plot) -> None:
    k = 5
    fig, axs = plt.subplots(k,2, sharex = True)
    fig.set_size_inches(12,18)
    #fig.suptitle('Gradient Boosting with Decision Tree Regressor from Scratch with max. depth of trees $d_{max}$, nr. of trees $n_{trees}$ and learning rate $\\alpha$ = ' + str(learning_rate))

    colors = ['mediumseagreen','lightskyblue',"mediumpurple",'salmon',"palevioletred"]

    Y_previous = np.zeros(k, dtype = object)

    k_ = [0, 0, 10 , 20, 30]

    for i in range(k):
        if i == 0:
            n_trees = 1
            y_hat = np.ones(len(X))
            y_hat[:] = np.mean(Y)
        else:
            n_trees = i * 10
            gbm = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
            gbm.fit(X_0,Y)
            y_hat, h_m = gbm.predict(X_0)

        Y_previous[i] = y_hat

        if i != 0:
            residuals = Y - Y_previous[i-1]
            markerline, stemlines, baseline = axs[i][0].stem(X, residuals, linefmt='grey', markerfmt='D', bottom=1.1, label = "y - $F_{%s}(x)$" %(k_[i]))
            markerline.set_markerfacecolor('none')
            axs[i][0].plot(X, h_m[-1],label = "$h_{%s}$(x)" %(n_trees), color = 'darkcyan', linewidth = 3)
            axs[i][0].set_ylim(-5,5)
        
        axs[i][1].plot(X, y_hat, color = colors[i], label = "$F_{%s}(x) = \hat{y}$" %(n_trees), linewidth = 5)
        axs[i][1].scatter(X, Y, color = 'gray', marker='o', edgecolors='k', s=18, label = "sample points")

        axs[i][0].set_title('$n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10)
        axs[i][1].set_title('$n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10)

        axs[i][0].set_ylabel("residuals")
        axs[i][1].set_ylabel("F(x)")
    
        axs[i][0].legend()
        axs[i][1].legend()
    
    axs[-1][0].set_xlabel("x")
    axs[-1][1].set_xlabel("x")

    fig.suptitle("Gradient Boosting with Decision Tree Regressor from Scratch with max. depth of trees $d_{max}$ = " + str(max_depth) + ", learning rate $\\alpha$ = " + str(learning_rate) + " and nr. of trees $n_{trees}$")
    fig.tight_layout()

    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "gbm_iterations.png")
    plt.show()

gbm_iterations(save_plot = True)
