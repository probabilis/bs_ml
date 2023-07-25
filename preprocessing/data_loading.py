import os 
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

###########

def loading_dataset(dir_path, filename):
    """
    params: path, filename 
    path ...        str / relative path folders for file 
    filename ...    str / filename of df
    ---------------
    return: df, features, target, eras
    df ...          Dataframe
    features ...    features vector
    target ....     target vector
    eras   ...      eras vector
    """
    path_ = os.path.join(os.path.expanduser('~'), dir_path, "train.parquet")
    df = pd.read_parquet(path_)
    print("Loaded dataframe from path : ", path_)

    features = [f for f in df if f.startswith("feature")]
    target = "target"
    df["erano"] = df.era.astype(int)
    eras = df.erano
    return df, features, target, eras


