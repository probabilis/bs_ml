from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoosting():

	def __init__(self, n_trees, learning_rate, max_depth):
		self.n_trees = n_trees
		self.learning_rate = learning_rate
		self.max_depth = max_depth 

	def fit(self, x, y):
		self.trees = []
		self.F_0 = y.mean()
		F_m = self.F_0

		for _ in range(self.n_trees):
			tree = DecisionTreeRegressor(max_depth = self.max_depth)
			tree.fit(x, y - F_m)
			F_m += self.learning_rate * tree.predict(x)
			self.trees.append(tree)

	def predict(self, x):
		y_hat = self.F_0 + self.learning_rate * np.sum([tree.predict(x) for tree in self.trees], axis = 0)
		return y_hat