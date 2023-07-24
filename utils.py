import os
from pathlib import Path
import pandas as pd
import numpy as np

#initialization

path = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "train.parquet")

#functions

def numerai_score(y, y_pred):
	rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct = True, method = "first") )
	return np.corrcoef(y, rank_pred)[0,1]

def correlation_score(y, y_pred):
	return np.corrcoef(y, y_pred)[0,1]


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n7



MODEL_FOLDER = "models"

def save_model(model, mtype, params): 
    if mtype == "LGBM":
        try:
            Path(MODEL_FOLDER).mkdir(exist_ok = True, parents = True)
        except Exception as ex:
            pass
        model.booster_.save_model(f"{MODEL_FOLDER}/{mtype}_{params}.json")
    if mtype == "XGB":
        try:
            Path(MODEL_FOLDER).mkdir(exist_ok = True, parents = True)
        except Exception as ex:
            pass
        model.booster_.save_model(f"{MODEL_FOLDER}/{mtype}_{params}.json")
    if mtype == "CatBoost":
        try:
            Path(MODEL_FOLDER).mkdir(exist_ok = True, parents = True)
        except Exception as ex:
            pass
        model.save_model(f"{MODEL_FOLDER}/{mtype}_{params}.json", format = "json")






