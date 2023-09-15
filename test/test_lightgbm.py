import pandas as pd
import gc, os, sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from lightgbm import LGBMRegressor
sys.path.append('../')
from preprocessing.cross_validators import era_splitting
from utils import loading_dataset, repo_path, numerai_score

#############################################

path_val = os.path.join(os.path.expanduser('~'), 'Documents', 'github_repos', "validation.parquet")

#############################################

df, features, target, eras = loading_dataset()

df_, eras_ = era_splitting(df, eras)
del df ; gc.collect()
print("deleted df from memory successfully")

#############################################

params_gbm = {"learning_rate":0.01,"max_depth":7,"n_estimators":1100, "colsample_bytree":0.2}

#############################################

print("data loading completed")

lgb = LGBMRegressor(**params_gbm)
lgb.fit(df_[features], df_[target])
print("lgbm created")

pred = pd.Series(lgb.predict(df_[features]), index = df_.index)

del df_; gc.collect()
print("deleted df_ from memory successfully")

df_val = pd.read_parquet(path_val)
df_val = df_val[df_val['data_type'].str.contains("validation")]



score = numerai_score(df_val[target],pred, eras_)
print("numer.ai score :", score)
r2 = lgb.score(df_val[target],pred)
print("R2 score :", r2)

