from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoosting():
	"""
	A class to apply the Gradient Boosting Framework proposed from Friedman et al.

    ...

    Attributes
    ----------
    n_trees : int
        number of weak/base learners 
    learning_rate : float
        scalar for the stagewise approximation
    max_depth : int
        maximimal depth of the tree

    Methods
    -------
    fit(X,Y):
        fits the regression trees to data deening on number of trees.
	predict(X,Y):
		predicts the function through the regression model built up on the training
	"""
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