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

#from numerai_functions import numerai_score


napi = NumerAPI()
"""
current_round = napi.get_current_round()
print(current_round)
napi.download_dataset("v4/train.parquet", "train.parquet")
print("successfully updated train parquet")
napi.download_dataset("v4/validation.parquet" , "validation.parquet")
print("sucessfully updated validation parquet")
"""

df = pd.read_parquet("train.parquet")
df.head()
print(df.head())

df["erano"] = df.era.astype(int)
eras = df.erano

features = [f for f in df if f.startswith("feature")]
#targets = [t for t in df if t.startswith("target")]
target = "target"

print("data loading completed")

########################################

#df1 = df[eras<= eras.median()]
#print("df1 timehorizan to df1 < df | < eras.median")
#df2 = df[eras> eras.median()]
#del df
#print("deleted df from memory successfully")

df1 = df

del df

cb = catboost.CatBoostRegressor(rsm = 0.6)
cb.fit(df1[features],df1[target])

print("cb created")

r2 = cb.score(df1[features],df1[target])
print("R2 df1",r2)

#r2 = cb.score(df2[features],df2[target])
#print("R2 / df2",r2)

def feature_importance(model):

    feature_importance = model.get_feature_importance()

    data = pd.DataFrame({'feature_importance': feature_importance, 
        'feature_names': features}).sort_values(by = ['feature_importance'],ascending = False)
    data.to_csv('feature_importance7.csv')
    data[:20].sort_values(by=['feature_importance'], ascending = True).plot.barh(x='feature_names',y='feature_importance')
    plt.show()
    return

def numerai_score(y, y_pred):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct = True, method = "first") )
    return np.corrcoef(y, rank_pred)[0,1]

#pred = pd.Series(cb.predict(df2[features]), index = df2.index)
#score = numerai_score(df2[target],pred)
#print("score: ", score)


del df1
#del df2
df = pd.read_parquet("validation.parquet")
print('validation file loaded')

df = df[df['data_type'].str.contains("validation")]

r2 = cb.score(df[features],df[target])
print("R2 df_val",r2)

pred = pd.Series(cb.predict(df[features]), index = df.index)
score = numerai_score(df[target],pred)
print("score: ", score)

#feature_importance(cb)

