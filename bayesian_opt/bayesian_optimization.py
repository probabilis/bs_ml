"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import pandas as pd
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
import time
from datetime import date
import gc
import sys
from sklearn.model_selection import cross_val_score
#own modules
sys.path.append('../')
from preprocessing.cross_validators import era_splitting
from preprocessing.pca_dimensional_reduction import dim_reduction
from repo_utils import repo_path, loading

#############################################
#############################################
#loading all necassary data from the reposiroty utils file 
train, feature_cols, target_cols, targets_df, t20s, t60s = loading()

#############################################

train = era_splitting(train)

del targets_df ; gc.collect()

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
    scores = cross_val_score(LGBMRegressor(**params_gbm),train[feature_cols], train["target"])
    score = scores.mean()
    return score

st = time.time()

params_gbm = {"learning_rate":(0.01,0.2),"max_depth":(1,10),"n_estimators":(500,50000), "colsample_bytree":(0.1,1)}

#n_iter:  Number of estimators of Bayesian Optimization (the more steps the more likely to find a good maximum you are)
#init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
init_points = 2 ; n_iter = 5

pbounds = params_gbm
bounds_transformer = SequentialDomainReductionTransformer() #minimum_window=0.5

gbm_bo = BayesianOptimization(gbm_reg_bo, params_gbm) 

gc.collect()

gbm_bo.maximize(init_points = init_points, n_iter = n_iter)

print('It takes %s minutes' %((time.time()-st)/60))

params_gbm = gbm_bo.max['params']
params_gbm['max_depth'] = round(params_gbm['max_depth'])
params_gbm['learning_rate'] = round(params_gbm['learning_rate'], 2)
params_gbm['colsample_bytree'] = round(params_gbm['colsample_bytree'], 1)
params_gbm['n_estimators'] = round(params_gbm['n_estimators'], 1)
print(params_gbm)

name = f"params_bayes_ip={init_points}_ni={n_iter}_{date.today()}_n=full"

data = pd.DataFrame([params_gbm])
data.to_csv(repo_path + "/models/" + name + ".csv" )

filename = name + ".txt"

with open(filename, "w") as file:
    for i, res in enumerate(gbm_bo.res):
        iter_ = "Iteration {}: \n\t{}".format(i, res)
        print(iter_)
        file.write(iter_)

