import sys
import pandas as pd
sys.path.append('../')
from utils import loading_dataset, get_biggest_change_features

#############################################

df, features, target, eras = loading_dataset()

#############################################

def feature_corr(df, era_col, target_col):
    """
    params: df, era_col, target_col 
    era_col ...     era column
    target_col ...  target column
    ---------------
    return: all_feature_corr
    df ...          dataframe with correlation from all features  
    """
    all_feature_corrs = df.groupby(era_col).apply(
    lambda era: era[features].corrwith(era[target_col]))
    return all_feature_corrs

corrs_ = feature_corr(df, "era", "target")

n = 50
risky = get_biggest_change_features(corrs_, n)
print(risky)

