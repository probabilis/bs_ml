"""
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import time
import os
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

####################################

X = np.linspace(0,2*np.pi,1000).reshape(-1,1)
Y = np.sin(X).ravel()

##################################

def gbm_reg_bo(learning_rate,max_depth,n_estimators): #,colsample_bytree
	params_gbm = {}
	params_gbm['max_depth'] = round(max_depth)
	params_gbm['learning_rate'] = learning_rate
	params_gbm['n_estimators'] = round(n_estimators)
	#params_gbm['colsample_bytree'] = colsample_bytree
	scores = cross_val_score(LGBMRegressor(**params_gbm),X,Y)
	score = scores.mean()
	return score

st = time.time()

#params_gbm = {"learning_rate":[0.01,0.05,0.1,0.15],"max_depth":[1,3,5,10],"n_estimators":[500,1000,2000,3000], "colsample_bytree":[0.1,0.3,0.5,0.8]}

params_gbm = {"learning_rate":(0.01,0.15),"max_depth":(1,10),"n_estimators":(500,3000)} #, "colsample_bytree":(0.1,0.8)


gbm_bo = BayesianOptimization(gbm_reg_bo,params_gbm,random_state = 111) 
gbm_bo.maximize(init_points = 20, n_iter = 30)
print('It takes %s minutes' %((time.time()-st)/60))

params_gbm = gbm_bo.max['params']
params_gbm['max_depth'] = round(params_gbm['max_depth'])
params_gbm['n_estimators'] = round(params_gbm['n_estimators'])
#params_gbm['colsample_bytree'] = round(params_gbm['colsample_bytree'])
print(params_gbm)

lgbm = LGBMRegressor(**params_gbm)
lgbm.fit(X,Y)
Y_pred = lgbm.predict(X)

plt.plot(X,Y_pred)
plt.plot(X,Y)
plt.show()



#best_models, times = model_optimizer(models, params, df1[features], df1[target])
#print(best_models)

#data = {"best_models": best_models, "times": times}
#data = pd.DataFrame(data, index = modeltypes)
#data.to_csv("round_infos.csv")

