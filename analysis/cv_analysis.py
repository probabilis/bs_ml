import sys
import gc
import pandas as pd 
import numpy as np
from sklearn import model_selection, metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
sys.path.append('../')
from utils import loading_dataset, correlation_score

#############################################

df, features, target, eras = loading_dataset()

#############################################

df_, eras_ = era_splitting(df, eras)

del df ; gc.collect()

#############################################

params_gbm = {"learning_rate": 0.02,"max_depth": 2,"n_estimators":1550, "colsample_bytree": 0.8}

#############################################

crossvalidators = [
    model_selection.KFold(5),   #classical 5 kFold CV
    model_selection.KFold(5, shuffle = True),   #classical 5 kFold CV with shuffling (= mischen)
    model_selection.GroupKFold(5), #K-fold iterator with non-overlapping groups
    TimeSeriesSplitGroups(5) #classical time series split with eras + boundaries
]

#def cross_validation(X,Y,eras, crossvalidators):

for cv in crossvalidators:
    print(cv)
    score = cross_val_score(LGBMRegressor(**params_gbm),df_[features],df_[target], cv = cv, n_jobs = 1, groups = eras, scoring = metrics.make_scorer(correlation_score, greater_is_better = True))
    print(np.mean(score))
    print('-------')

#out = cross_validation(df_[features],df_[target],eras_, crossvalidators)



