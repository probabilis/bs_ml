"""
Author: Maximilian Gschaider
MN: 12030366
"""
from from_scratch.decision_regression_tree_from_scratch import DecisionTreeRegressorScratch
import numpy as np
import pandas as pd

########################################

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
	X : DataFrame
		input dataframe for training
	Y : DataFrame
		output dataframe for training
    Methods
    -------
    fit(X,Y):
        fits the regression trees to data deening on number of trees.
	predict(X,Y):
		predicts the function through the regression model built up on the training
	"""
	def __init__(self, learning_rate : float, max_depth : int ,n_trees : int, 
	      		X : pd.DataFrame, Y : pd.DataFrame):

		#hyperparameter asssignment
		self.learning_rate = learning_rate
		self.max_depth = max_depth 
		self.n_trees = n_trees
		#input and output dataframe assignment
		self.X = X
		self.Y = Y

	def fit(self, X : pd.DataFrame, Y : pd.DataFrame):
		"""
		fitting stagewise the regression tree through given data

		...

		Params
		------
		X : pd.DataFrame
			input df / vector over the features room
		Y : pd.DataFrame
			target variables to learn the model         
		---------------
		return : None
			None / only training the model
		"""
		self.trees = []
		self.F_0 = Y.mean()
		F_m = self.F_0
		self.X = X
		self.Y = Y

		#stagewise iteration for n < n_trees
		for _ in range(self.n_trees):
			y_tilde = Y - F_m
			tree = DecisionTreeRegressorScratch(X, y_tilde, max_depth = self.max_depth)
			tree.fit()

			F_m += self.learning_rate * tree.predict(X)
			self.trees.append(tree)

	def predict(self, X : pd.DataFrame):
		"""
		predicting stagewise the regression tree through given input and output data

		...

		Params
		------
		X : pd.DataFrame
			input df / vector over the features room
		---------------
		return : array
			y_hat / predicted output df
		"""
		y_hat = self.F_0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis = 0)
		h_m = [tree.predict(X) for tree in self.trees]
		return y_hat, h_m