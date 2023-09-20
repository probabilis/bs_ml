"""
Project: Bachelor Machine Learning 
Script: Main Program
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
import csv
from sklearn.model_selection import cross_val_score
from pathlib import Path

sys.path.append('../')

from bs_ml.preprocessing.cross_validators import era_splitting
from bs_ml.preprocessing.pca_dimensional_reduction import dim_reduction
from bs_ml.utils import loading_dataset, repo_path, path_val

#############################################
#############################################
#############################################

#loading dataset
df, features, target, eras = loading_dataset()

#############################################

#splitting the eras
train, eras_ = era_splitting(df, eras)

del df ; gc.collect()

#############################################

#n = 100
#df_pca, features_pca = dim_reduction(train,features,target,n)
#del df_

#############################################

#loading the specific hyperparameter configuration from bayesian optimization

filename = "params_bayes_ip=20_ni=300_2023-09-15_n=300.csv"
path = repo_path + "/models/" + filename

params_gbm = pd.read_csv(path).to_dict(orient = "list")
params_gbm.pop("Unnamed: 0")

max_depth = params_gbm['max_depth'][0]
learning_rate = params_gbm['learning_rate'][0]
colsample_bytree = params_gbm['colsample_bytree'][0]
n_trees = int(round(params_gbm['n_estimators'][0],1))

#############################################
#defining the target candidates for ensemble modeling

target_candidates = ["target_cyrus_v4_20", "target_waldo_v4_20", "target_victor_v4_20", "target_xerxes_v4_20"]

#############################################

models = {}
for target in target_candidates:
    model = LGBMRegressor(
        n_estimators = n_trees,
        learning_rate=learning_rate,
        max_depth = max_depth,
        colsample_bytree=colsample_bytree
    )
    model.fit(train[features], train[target]
    );
    models[target] = model

#############################################

validation = pd.read_parquet(path_val)

validation = validation[validation['data_type'].str.contains("validation")]
del validation["data_type"]

validation = era_splitting(validation, eras)

last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

for target in target_candidates:
    validation[f"prediction_{target}"] = models[target].predict(validation[features])
    
pred_cols = [f"prediction_{target}" for target in target_candidates]

print(validation[pred_cols])
