"""
Author: Maximilian Gschaider
MN: 12030366
"""
########################################
#official open-source repositories
import sys
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt

#own modules
from gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X
sys.path.append('../')
from bs_ml.utils import repo_path, fontsize, fontsize_title

########################################
#testfunction
Y = testfunction(X, noise = 0.5)
Y_ = testfunction(X, noise = 0)

########################################
#hyperparameters
learning_rate = 0.01
n_trees = 1000
max_depth = 1

X_0 = X.reshape(-1,1)

########################################

def gbm_dt_comparison(plot_save) -> None:

	gbm_model = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
	gbm_model.fit(X_0, Y)
	y_hat_gbm, _ = gbm_model.predict(X_0)

	model = DecisionTreeRegressor(max_depth = max_depth)
	model.fit(X_0, Y)
	y_hat_dt = model.predict(X_0)

	plt.title('Gradient Boosting with Decision Tree Regressor from Scratch with max depth $d_{max}$ and number of trees $m$', fontsize = fontsize_title)
	plt.scatter(X,Y, color = 'gray', marker='o', edgecolors='k', s=18, label = 'sample points of F(x) with Gaussian noise')
	plt.plot(X, y_hat_dt, color = 'salmon', label = '$\hat{y}$ | GBM-DT with $d_{max}$ = ' + str(max_depth) + ' and $m$ = 1', linewidth = 5)
	plt.plot(X, y_hat_gbm, color = 'cornflowerblue', label = '$\hat{y}$ | GBM-DT with $d_{max}$ = ' + str(max_depth) + ' and $m$ = ' + str(n_trees), linewidth = 5)
	plt.plot(X, Y_, color = 'limegreen', linestyle = "--", linewidth = 5, label = "F(x)")
	plt.legend(fontsize = fontsize)
	fig = plt.gcf()
	fig.set_size_inches(12,10)
	fig.tight_layout()
	if plot_save == True:
		plt.savefig(repo_path + "/figures/" + "gbm_decision_tree_comparison.png", dpi=300)
	plt.show()

gbm_dt_comparison(plot_save = True)

########################################

def hyperparameters_matrix(save_plot) -> None:
	max_depth = 1
	k = 5
	fig, axs = plt.subplots(k,k)
	fig.set_size_inches(24,18)
	fig.suptitle('Gradient Boosting with Decision Tree Regressor from Scratch with max. depth of trees $d_{max}$, nr. of trees $n_{trees}$ and learning rate $\\alpha$ = ' + str(learning_rate), fontsize = fontsize_title)

	colors = ['mediumseagreen','lightskyblue',"mediumpurple",'salmon',"palevioletred"]

	for i in range(k):
		for j in range(k):
			n_trees = (i+1) * 100
			max_depth = j + 1
			gbm = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
			gbm.fit(X_0,Y)
			y_hat, _ = gbm.predict(X_0)
			axs[i][j].plot(X, y_hat, color = colors[j],label = '$\hat{y}$ | $d_{max}$ = ' + str(max_depth) + " / $n_{trees} = $" + str(n_trees), linewidth = 5)
			axs[i][j].set_title('$d_{max}$ = ' + str(max_depth) + ' and $n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10, fontsize = fontsize)
			axs[i][j].scatter(X,Y,color = 'gray', marker='o', edgecolors='k', s=18)
			axs[i][j].plot(X, Y_, color = 'limegreen', linestyle = "--", linewidth = 3, label = "F(x)")
			axs[i][j].legend(fontsize = fontsize)

	fig.tight_layout()
	if save_plot == True:
		plt.savefig(repo_path + "/figures/" + "gbm_with_decision_tree_various_parameters.png", dpi=300)
	plt.show()

hyperparameters_matrix(save_plot = True)
