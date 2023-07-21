"""
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import time
import os
from sklearn.model_selection import cross_val_score
from pca_dimensional_reduction import dim_reduction

####################################

path = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "train.parquet")
print(path)

df = pd.read_parquet(path)

features = [f for f in df if f.startswith("feature")]
target = "target"
df["erano"] = df.era.astype(int)
eras = df.erano

df1 = df[eras<=eras.median()]

del df
print("deleted df sucessfully")

##################################

n = 10
df_pca, features_pca = dim_reduction(df1,features,n)
del df1

##################################

X = df_pca[features_pca]
Y = df_pca[target]

def gbm_reg_bo(learning_rate,max_depth,n_estimators,colsample_bytree):
    """
    params: module, dict, df, df 
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

data = pd.DataFrame([params_gbm])
data.to_csv("round_infos_bayes.csv")

