import sys
import pandas as pd 
import numpy as np
from sklearn import model_selection, metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

sys.path.append('/bs_ml/')
from utils import path, features, target, eras, correlation_score

sys.path.append('/bs_ml/preprocessing')
from preprocessing.cross_validators import era_splitting, TimeSeriesSplitGroups 


###########################

df = pd.read_parquet(path)

###########################

params_gbm = {"learning_rate":(0.01,0.15),"max_depth":(1,10),"n_estimators":(500,3000), "colsample_bytree":(0.1,0.8)}

crossvalidators = [
    model_selection.KFold(5),   #classical 5 kFold CV
    model_selection.KFold(5, shuffle = True),   #classical 5 kFold CV with shuffling (= mischen)
    model_selection.GroupKFold(5), #K-fold iterator with non-overlapping groups
    TimeSeriesSplitGroups(5) #classical time series split with eras + boundaries
]

def cross_validation(X,Y,crossvalidators):

    for cv in crossvalidators:
        print(cv)
        score = cross_val_score(LGBMRegressor(**params_gbm),X,Y, cv = cv, n_jobs = 1, groups = eras, scoring = metrics.make_scorer(correlation_score, greater_is_better = True))
        print(np.mean(score))
        print('-------')




