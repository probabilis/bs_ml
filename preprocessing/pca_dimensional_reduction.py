import pandas as pd
import numpy as np
import time

from sklearn.decomposition import PCA


def dim_reduction(df, features, target, n):
	"""
	input df ...		dataframe
	input features ... 	feature vector
	input target ...	target vector
	param n ... 		reduction size
    ---------------
    return: df & array as dupel 
    pca_df, pca_features  
    """
	pca = PCA(n_components = n)
	st = time.time()
	pca_df = pca.fit_transform(df[features])
	et = time.time()

	print('time needed for reduction: ' + str(et-st) + ' sec')

	pca_features = pca.get_feature_names_out(features)
	pca_df = pd.DataFrame(data = pca_df, columns = pca_features, index = df.index)
	pca_df["target"] = df[target]
	pca_ls = pca.explained_variance_ratio_
	print('PCA / information lost: ', 1-np.sum(pca_ls))
	return pca_df, pca_features

