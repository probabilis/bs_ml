import sys
import gc
import pandas as pd 
import numpy as np
from sklearn import model_selection, metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
sys.path.append('../')
from preprocessing.cross_validators import TimeSeriesSplitGroups 
from repo_utils import loading, correlation_score

#############################################

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()
eras = train.era.astype(int)

target = "target_cyrus_v4_20"
gc.collect()

#############################################

params_gbm = {"learning_rate": 0.02,"max_depth": 2,"n_estimators":1550, "colsample_bytree": 0.8}

crossvalidators = [
    model_selection.KFold(5),   #classical 5 kFold CV
    model_selection.KFold(5, shuffle = True),   #classical 5 kFold CV with shuffling (= mischen)
    model_selection.GroupKFold(5), #K-fold iterator with non-overlapping groups
    TimeSeriesSplitGroups(5) #classical time series split with eras + boundaries
]

for cv in crossvalidators:
    print(cv)
    score = cross_val_score(LGBMRegressor(**params_gbm),train[feature_cols],train[target], cv = cv, n_jobs = 1, groups = eras, scoring = metrics.make_scorer(correlation_score, greater_is_better = True))
    print(np.mean(score))
    print('-------')


