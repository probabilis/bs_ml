from numerapi import NumerAPI
import parquet
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sklearn

import catboost

from sklearn import (
    feature_extraction, feature_selection, decomposition, linear_model,
    model_selection, metrics, svm
)

from sklearn.decomposition import PCA
import time

"""
napi = NumerAPI()

current_round = napi.get_current_round()

napi.download_dataset("v4/train.parquet", "train.parquet")
"""


df = pd.read_parquet("train.parquet")
df.head()
print(df.head())

df["erano"] = df.era.astype(int)
eras = df.erano

features = [f for f in df if f.startswith("feature")]
#targets = [t for t in df if t.startswith("target")]
target = "target"

#features = features[:10]

print("data loading completed")

########################################

df1 = df[eras<= eras.median()]
print("df1 timehorizan to df1 < df | < eras.median")

del df
print("deleted df from memory successfully")

#lgb1 = lightgbm.LGBMRegressor(learning_rate = 0.01,n_estimators = 2000)
#lgb1.fit(df1[features], df1[target])

#del df1

cb = catboost.CatBoostRegressor(max_depth = 10) #
cb.fit(df1[features],df1[target])

print("cb created")


r2 = cb.score(df1[features],df1[target])
print("R2 / features on df1",r2)



def feature_importance(model):

    feature_importance = model.get_feature_importance()

    data = pd.DataFrame({'feature_importance': feature_importance, 
        'feature_names': features_new}).sort_values(by = ['feature_importance'],ascending = False)
    data.to_csv('feature_importance_td=10.csv')
    plt.figure(figsize = (12,8))
    data[:20].sort_values(by=['feature_importance'], ascending = True).plot.barh(x='feature_names',y='feature_importance')
    plt.show()
    return

#feature_importance(cb)
