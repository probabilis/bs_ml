import sys
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from math import * 

from gradient_boosting_from_scratch import gbm
from pathlib import Path

import os
repo_path = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "bs_ml")

########################################

x_min = 0
x_max = pi * 2
N = 100

X = np.linspace(x_min,x_max,100)
X_0 = X.reshape(-1,1)

Y = np.sin(X)

learning_rate = 0.1
n_trees = 100
max_depth = 2

##############################

y_hat = gbm(learning_rate, max_depth, n_trees, X_0, Y)
model = DecisionTreeRegressor(max_depth = max_depth)
model.fit(X_0,Y)

x_seq = np.linspace(x_min,x_max,N).reshape(-1,1)
y_pred = model.predict(x_seq)

##########################
x = np.linspace(x_min,x_max,N)
plt.title('Gradient Boosting / Decision Tree Regressor comparison from Scratch with max depth $d_{max}$ and number of trees $n$', fontsize = 16)
plt.scatter(X,Y, color = 'gray', label = 'sin(x) with x $\\in$ [0,2$\\pi$]')
plt.plot(x, y_pred, color = 'salmon', label = 'DT with d = ' + str(max_depth))
plt.plot(x, y_hat, color = 'mediumseagreen', label = 'GBM with $d_{max}$ = ' + str(max_depth) + ' and $n$ = ' + str(n_trees) )
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18,14)
fig.tight_layout()
plt.savefig(repo_path + "/figures/" + "gbm_decision_tree_comparison.png")
plt.show()
print("hi")
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
		y_hat = gbm(learning_rate, max_depth, n_trees, X_0, Y)
		axs[i][j].plot(x,y_hat, color = colors[j],label = '$d_{max}$ = ' + str(max_depth) + " / $n_{trees} = $" + str(n_trees) )
		axs[i][j].set_title('$d_{max}$ = ' + str(max_depth) + ' and $n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10)
		axs[i][j].scatter(X,Y,color = 'gray')
		axs[i][j].legend()

fig.tight_layout()
plt.savefig(repo_path + "/figures/" + "gbm_with_decision_tree_various_parameters.png")
plt.show()
