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





