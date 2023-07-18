from numerapi import NumerAPI
import parquet
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

import time

#napi = NumerAPI()

#current_round = napi.get_current_round()

#napi.download_dataset("v4/train.parquet", "train.parquet")
df = pd.read_parquet("train.parquet")
#df.head()
#print(df.head())


features = [c for c in df if c.startswith("feature")]

df["erano"] = df.era.astype(int)
eras = df.erano

target = 'target'
#targets = [c for c in df if c.startswith("target")]

#corr_matrix = df[targets].corr()
#sn.heatmap(corr_matrix,annot = True)
#plt.show()

#st = time.time()
#plt.figure(figsize = (8,8))
#plt.imshow(df[df.era=='0001'][features].corr())
#et = time.time()
#print('time needed for calculation: ' + str(round(et-st,2)) + ' sec')
#plt.show()

"""

#f = features[0]

stock = df.iloc[0]
print(stock.name)

print(df.head())

s = df.loc[str(stock.name)]

print(s)

f1 = "feature_honoured_observational_balaamite"

y = df[f1]
x = np.arange(0,len(y),1)

plt.plot(x,y)
plt.show()
"""

#1df.groupby(eras).size().plot()
#plt.show()

"""
feature_scores = {feature: score for feature, score in zip(features, np.corrcoef(df[df.era=='0001'][[target]+features].T)[1:,0])}

by_era_correlation = pd.Series({
    era: np.corrcoef(tdf[target], tdf["feature_untidy_withdrawn_bargeman"])[0,1]
    for era, tdf in df.groupby(eras)
})
by_era_correlation.plot()
plt.show()
"""

#df1 = df[eras<= eras.median()]

df1 = df[eras<= 10]

del df

print(eras)

def correlation_features(df,features,target):

    eras = df.erano

    correlations = np.zeros((len(eras),len(features)) , dtype = object)

    for e,era in enumerate(eras):
        df_ = df[eras == era]
        for f, feature in enumerate(features):
            if feature in df_:
                corr_ = np.corrcoef(df_[feature],df_[target])
                print('corr. calculated for feature:',feature)
            
                correlations[e][f] = corr_


    return correlations
st = time.time()
correlations = correlation_features(df1,features,target)
et = time.time()
print('time: ' + str(et-st) + ' sec')

output = pd.DataFrame(correlations)
output.to_csv('correlations_target.csv')