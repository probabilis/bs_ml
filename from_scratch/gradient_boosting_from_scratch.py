"""
Author: Maximilian Gschaider
MN: 12030366
"""
from sklearn.tree import DecisionTreeRegressor
import numpy as np

########################################

def gbm(learning_rate, max_depth, n_trees, x,y):
	F_0 = y.mean()
	F_m = F_0
	trees = []

	for _ in range(n_trees):
		y_tilde = y - F_m
		tree = DecisionTreeRegressor(max_depth = max_depth)
		tree.fit(x, y_tilde)
		F_m += learning_rate * tree.predict(x)
		trees.append(tree)

	y_hat = F_0 + learning_rate * np.sum([tree.predict(x) for tree in trees], axis = 0)
	return y_hat
