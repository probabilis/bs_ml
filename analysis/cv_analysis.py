import sys
import pandas as pd 
import numpy as np
from sklearn import model_selection, metrics
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

sys.path.append('../')
from utils import path, correlation_score
from preprocessing.cross_validators import era_splitting, TimeSeriesSplitGroups 

###########################

df = pd.read_parquet(path)
features = [f for f in df if f.startswith("feature")]
target = "target"
df["erano"] = df.era.astype(int)
eras = df.erano

###########################

df_, eras_ = era_splitting(df, eras)

del df

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
    return

cross_validation(df_[features],df_[target], crossvalidators)



