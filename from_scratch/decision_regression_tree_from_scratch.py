"""
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from collections import Counter

####################################

class DecisionTreeRegressorScratch():
	"""
	Class for growing a regression Decision Tree by Scratch
	"""
	def __init__(self, X : pd.DataFrame, Y : list, min_samples_split = None, 
		max_depth = None, depth = None, node_type = None, rule = None):

		#data assignment for specific node
		self.X = X
		self.Y = Y

		#hyperparamter's assignment if not given
		self.min_samples_split = min_samples_split if min_samples_split else 10
		self.max_depth = max_depth if max_depth else 3
		self.depth = depth if depth else 0

		#extracting and assigning features from input dataframe X
		self.features = list(self.X.columns)

		#defining node_type if not given
		self.node_type = node_type if node_type else 'root'

		#defining rule for splitting 
		self.rule = rule if rule else ""

		#getting the mean of Y
		self.ymean = np.mean(Y)

		#calculating residuals of Y
		self.residuals = self.Y - self.ymean
		#calculating mean-squared-error of Y
		self.mse = self.get_mse(Y, self.ymean)
		#observations of the spefic node
		self.n = len(Y)

		#iniating left and right nodes as empty ones
		self.left = None ; self.right = None
		#iniating best feature and value as empty one
		self.best_feature = None ; self.best_value = None

		y_hat = None
		self.y_hat = self.ymean

	@staticmethod 
	def get_mse(y_true, y_hat):
		r = np.sum( (y_true - y_hat)**2 )
		return r / len(y_true)

	@staticmethod
	def ma(x, window):
		"""
		Calculating Moving Average through convulation with one position overlap
		"""
		return np.convolve(x, np.ones(window), 'valid') / window 

	def best_split(self):
		"""
		By giving the features as input dataframe X and targets Y the best splits
		are calculated for the decision tree
		"""
		df = self.X.copy()
		df['Y'] = self.Y

		#calculating the MSE deviance for the base input layer
		mse_base = self.mse
		#print(mse_base)

		best_feature = None ; best_value = None

		for feature in self.features:
			#dropping missing values and sorting by features
			df_X = df.dropna().sort_values(feature) 

			x_conv = self.ma(df_X[feature].unique(), 2)

			#iterating for all calculated xi's from the convulution / moving average
			for xi in x_conv:
				#subsampling the X-dataframe into left and right sub-space
				y_l = df_X[df_X[feature] < xi]['Y'].values 
				y_r = df_X[df_X[feature] > xi]['Y'].values
				
				#calculating the residuals from left and right sub-space
				res_l = (y_l - np.mean(y_l))
				res_r = (y_r - np.mean(y_r))
				#merging the residuals into 1D-array
				r = np.concatenate((res_l, res_r), axis = None)
				#calculating the MSE from the merged residuals array
				mse_split = np.sum(r**2) / len(r)

				#checking if the calculated split is the best so far
				if mse_split < mse_base:
					best_feature = feature
					best_value = xi

					mse_base = mse_split

		return (best_feature, best_value)

	def grow_tree(self):
		"""
		Recursive function for creating the decision tree
		"""
		df = self.X.copy()
		df['Y'] = self.Y

		#If there's further MSE momentum gain possible, we split further

		if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

			#getting the best split
			best_feature, best_value = self.best_split()

			if best_feature is not None:
				self.best_feature = best_feature
				self.best_value = best_value

				#splitting df into left and right sub-space

				left_df, right_df = df[df[best_feature] <= best_value].copy(), df[df[best_feature] > best_value].copy()

				left = DecisionTreeRegressorScratch(
					left_df[self.features],
					left_df['Y'].values.tolist(),
					depth = self.depth + 1,
					max_depth = self.max_depth,
					min_samples_split = self.min_samples_split,
					node_type = 'left_node',
					rule = f"{best_feature} > {round(best_value, 3)}" )

				self.left = left
				self.left.grow_tree()

				right = DecisionTreeRegressorScratch(
					right_df[self.features],
					right_df['Y'].values.tolist(),
					depth = self.depth + 1,
					max_depth = self.max_depth,
					min_samples_split = self.min_samples_split,
					node_type = 'right_node',
					rule = f"{best_feature}  {round(best_value, 3)}" )

				self.right = right
				self.right.grow_tree()

	def fit(self, X : pd.DataFrame):
		"""
		Fitting the input data
		"""
		predictions = []
		for _, x in X.iterrows():
			values = {}
			for feature in self.features:
				values.update({feature: x[feature]})

			predictions.append(self.predict(values))
		return predictions

	def predict(self, values : dict):
		"""
		Method for predicting the functional for the given input feature space
		"""
		xnode = self
		while xnode.depth < xnode.max_depth:
			best_feature = xnode.best_feature
			best_value = xnode.best_value

			if xnode.n < xnode.min_samples_split:
				break
			elif (values.get(best_feature) < best_value):
				if self.left is not None:
					xnode = xnode.left
			else:
				if self.right is not None:
					xnode = xnode.right

		return xnode.y_hat