"""
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import time
from datetime import date
import os
import gc
import sys
from sklearn.model_selection import cross_val_score
from pca_dimensional_reduction import dim_reduction
from preprocessing.cross_validators import era_splitting

sys.path.append('../')
from utils import loading_dataset

####################################

df, features, target, eras = loading_dataset()

#############################################

df_, eras_ = era_splitting(df, eras)

del df ; gc.collect()

##################################

n = 10
df_pca, features_pca = dim_reduction(df_,features,n)
del df_

##################################

X = df_pca[features_pca]
Y = df_pca[target]

def gbm_reg_bo(learning_rate,max_depth,n_estimators,colsample_bytree):
    """
    params: module, dict, df 
    regressor ...   type of regression module e.g.: CatBoost, LightGBM, XGBoost
    params ...      given hyperparameters to analyze
    X ...           input df / vector over the features room
    Y ...           target variables to learn the model
    ---------------
    return: scalar
    score from best model
    """
    params_gbm = {}
    params_gbm['max_depth'] = round(max_depth)
    params_gbm['learning_rate'] = learning_rate
    params_gbm['n_estimators'] = round(n_estimators)
    params_gbm['colsample_bytree'] = colsample_bytree
    scores = cross_val_score(LGBMRegressor(**params_gbm),X,Y)
    score = scores.mean()
    return score

st = time.time()

params_gbm = {"learning_rate":(0.01,0.15),"max_depth":(1,10),"n_estimators":(500,3000), "colsample_bytree":(0.1,0.8)}


gbm_bo = BayesianOptimization(gbm_reg_bo,params_gbm,random_state = 111) 
gbm_bo.maximize(init_points = 10, n_iter = 5) #
print('It takes %s minutes' %((time.time()-st)/60))

params_gbm = gbm_bo.max['params']
params_gbm['max_depth'] = round(params_gbm['max_depth'])
params_gbm['colsample_bytree'] = round(params_gbm['colsample_bytree'], 1)
print(params_gbm)

today = date.today()

name = f"round_infos_bayes_{today}_n={n}"

data = pd.DataFrame([params_gbm])
data.to_csv("outputs/" + name + ".csv")

