import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import time
import os
import sys
from preprocessing.pca_dimensional_reduction import dim_reduction

sys.path.append('../')
from utils import loading_dataset

####################################

df, features, target, eras = loading_dataset()

#############################################

df1 = df[eras<=5]
del df
print("loaded df sucessfully")

#################################
n = 5 
df_pca, features_pca  = dim_reduction(df1,features,target,n)
del df1
print("deledted df sucessfully")
##################################

def GridSearch(regressor, params, X, Y):
	"""
	params: module, dict, df, df 
	regressor ... 	type of regression module e.g.: CatBoost, LightGBM, XGBoost
	params ... 		given hyperparameters to analyze
	X ... 			input df / vector over the features room
	Y ... 			target variables to learn the model
	---------------
	return: dupel
	estimators & params from best model
	"""
	model = GridSearchCV(regressor, params)
	model.fit(X,Y)
	
	print(model.best_params_)
	return model.best_estimator_ , model.best_params_

params_cb = {"learning_rate":[0.01,0.05,0.1,0.15],"max_depth":[1,3,5,10],"n_estimators":[500,1000,2000], "rsm":[0.1,0.3,0.5,0.8]}
params_xb = {"learning_rate":[0.01,0.05,0.1,0.15],"max_depth":[1,3,5,10],"n_estimators":[500,1000,2000], "colsample_bytree":[0.1,0.3,0.5,0.8]}
params_lb = params_xb

params = [params_cb, params_lb, params_xb]
models = [CatBoostRegressor(),LGBMRegressor(),XGBRegressor()]
modeltypes = ["CatBoost","LGBM","XGB"]


def model_optimizer(models,params,X,Y,prefix):
	best_models = []
	time_ls = []

	for m, model in enumerate(models):
		st = time.time()
		mtype = modeltypes[m]
		
		model_ , params_ = GridSearch(model, params[m], X, Y)
		modelname = f"{prefix}_{mtype}_{params_}.json"

		if mtype == "LGBM":
			model_.booster_.save_model(modelname)
		if mtype == "XGB":
			model_.save_model(modelname)
		if mtype == "CatBoost":
			model_.save_model(modelname, format = "json")

		time_ = time.time() - st
		time_ls.append(time_)
		best_models.append(modelname)
	return best_models, time_ls

prefix = "pca_" + str(n)
st = time.time()
best_models, times = model_optimizer(models, params, df_pca[features_pca], df_pca[target],prefix)
time_ = st - time.time()
print(best_models)
with open(f"{prefix}_output.txt") as fp:
    print("total time needed in sec. :"f"{time_} ", fd = fp)

data = {"best_models": best_models, "times": times}
data = pd.DataFrame(data, index = modeltypes)
data.to_csv(f"{prefix}_round_infos.csv")
###################################
