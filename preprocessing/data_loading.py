from numerapi import NumerAPI
import parquet
import pandas as pd
import numpy as np

dft = pd.read_parquet("train.parquet")
print(dft.head())

print(dft["target"])

del dft

dfv = pd.read_parquet("validation.parquet")
print(dfv.head())

features = [f for f in dfv if f.startswith("feature")]
dfv = dfv[dfv['data_type'].str.contains("validation")]
print(dfv)
print(dfv["target"])


