"""
Author: Maximilian Gschaider
MN: 12030366
"""
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from math import * 
import os

from gbm_class import GradientBoosting

repo_path = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "bs_ml")

########################################

x_min = 0
x_max = pi * 2
N = 100

X = np.linspace(x_min,x_max,N)
X_0 = X.reshape(-1,1)

Y = np.sin(X)

learning_rate = 0.1
n_trees = 100
max_depth = 2

##############################
gbm_model = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
gbm_model.fit(X_0, Y)
y_hat = gbm_model.predict(X_0)
print(y_hat)

model = DecisionTreeRegressor(max_depth = max_depth)
model.fit(X_0, Y)

y_pred = model.predict(X_0)

##########################
x = np.linspace(x_min,x_max,N)
plt.title('Gradient Boosting / Decision Tree Regressor comparison from Scratch with max depth $d_{max}$ and number of trees $n$', fontsize = 16)
plt.scatter(X,Y, color = 'gray', label = 'sin(x) with x $\\in$ [0,2$\\pi$]')
plt.plot(X, y_pred, color = 'salmon', label = 'DT with d = ' + str(max_depth))
plt.plot(X, y_hat, color = 'mediumseagreen', label = 'GBM with $d_{max}$ = ' + str(max_depth) + ' and $n$ = ' + str(n_trees) )
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18,14)
fig.tight_layout()
#plt.savefig(repo_path + "/figures/" + "gbm_decision_tree_comparison.png")
plt.show()

#############################
max_depth = 1

k = 5
fig, axs = plt.subplots(k,k)
fig.set_size_inches(24,18)
fig.suptitle('Gradient Boosting with Decision Tree Regressor from Scratch with max. depth of trees $d_{max}$, nr. of trees $n_{trees}$ and learning rate $\\alpha$ = ' + str(learning_rate))

colors = ['mediumseagreen','lightskyblue',"mediumpurple",'salmon',"palevioletred"]

for i in range(k):
	for j in range(k):
		n_trees = i + 1
		max_depth = j + 1
		gbm = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
		gbm.fit(X_0,Y)
		y_hat = gbm.predict(X_0)
		axs[i][j].plot(X, y_hat, color = colors[j],label = '$d_{max}$ = ' + str(max_depth) + " / $n_{trees} = $" + str(n_trees) )
		axs[i][j].set_title('$d_{max}$ = ' + str(max_depth) + ' and $n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10)
		axs[i][j].scatter(X,Y,color = 'gray')
		axs[i][j].legend()

fig.tight_layout()
#plt.savefig(repo_path + "/figures/" + "gbm_with_decision_tree_various_parameters.png")
plt.show()
