import os
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