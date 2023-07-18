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

from numerai_functions import numerai_score

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

print("data loading completed")

########################################
df = df[eras.isin(np.arange(1,575,4))]

df1 = df[eras<= eras.median()]
print("df1 timehorizan to df1 < df | < eras.median")

#df2 = df[eras> eras.median()]

del df
print("deleted df from memory successfully")

########################################

n_comp = 420

pca = PCA(n_components = n_comp)
st = time.time()
pc_df = pca.fit_transform(df1[features])
et = time.time()

print('time: ' + str(et-st) + ' sec')

features_new = pca.get_feature_names_out(features)
print(len(features_new))
print(features_new)

pca_df = pd.DataFrame(data = pc_df, columns = features_new, index = df1.index)
pca_df["target"] = df1[target]

pca_ls = pca.explained_variance_ratio_
print('PCA / information lost: ', 1-np.sum(pca_ls))

########################################

del df1

cb = catboost.CatBoostRegressor(n_estimators = 3000, max_depth = 12) #learning_rate = 0.1, 
#, 
cb.fit(pca_df[features_new],pca_df[target])

print("cb created")

r2 = cb.score(pca_df[features_new],pca_df[target])
print("R2 PCA",r2)

del pca_df

def feature_importance(model):

    feature_importance = model.get_feature_importance()

    data = pd.DataFrame({'feature_importance': feature_importance, 
        'feature_names': features_new}).sort_values(by = ['feature_importance'],ascending = False)
    data.to_csv('feature_importance6.csv')
    data[:20].sort_values(by=['feature_importance'], ascending = True).plot.barh(x='feature_names',y='feature_importance')
    plt.show()
    return

feature_importance(cb)

#pred = pd.Series(cb.predict(df2[features]), index = df2.index)
#score = numerai_score(df2[target],pred)
#print("score: ", score)