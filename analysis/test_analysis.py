import sys
import pandas as pd
sys.path.append('../')
from utils import loading_dataset, feature_corr, get_biggest_change_features

#############################################

df, features, target, eras = loading_dataset()

#############################################

corrs_ = feature_corr(df, "era", "target")

n = 50
risky = get_biggest_change_features(corrs_, n)
print(risky)

