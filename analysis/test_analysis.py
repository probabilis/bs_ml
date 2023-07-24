import sys
import pandas as pd
sys.path.append('../')
from utils import path, get_biggest_change_features


df = pd.read_parquet(path)
features = [f for f in df if f.startswith("feature")]
target = "target"
df["erano"] = df.era.astype(int)
eras = df.erano


def feature_corr(df, era_col, target_col):
    all_feature_corrs = df.groupby(era_col).apply(
    lambda era: era[features].corrwith(era[target_col]))
    return all_feature_corrs

corrs_ = feature_corr(df, "era", "target")

n = 50
risky = get_biggest_change_features(corrs_, n)
print(risky)

