import numpy as np

def numerai_score(y, y_pred):
	rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct = True, method = "first") )
	return np.corrcoef(y, rank_pred)[0,1]

def correlation_score(y, y_pred):
	return np.corrcoef(y, y_pred)[0,1]