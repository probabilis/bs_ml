"""
Project: Bachelor Project / Supervised Machine Learning / Gradient Boosting Machine based on Decision Trees
Script: Main Program
Author: Maximilian Gschaider
Date: 15.11.2023
MN: 12030366
------------------
Ref.: www.numer.ai
#(some of the code from the scripts provided was used)
"""
#official open-source repositories
import pandas as pd
import sys
import numpy as np
from lightgbm import LGBMRegressor, plot_importance
import time
from datetime import date
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn as nn 
#############################################
sys.path.append("../")
from repo_utils import gh_repos_path, repo_path, loading, hyperparameter_loading, numerai_corr, neutralize, least_correlated

#############################################
#overall prefix for saving (directory management)
prefix = "00"

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()

target_cyrus = "target_cyrus_v4_20"

PATH = "nn_model_0"

ifa = len(feature_cols)

model_nn = nn.Sequential(
    nn.Linear(ifa, 1000),
    nn.ReLU(),
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 1)
)

model_nn.load_state_dict(torch.load(PATH))
model_nn.eval()

target_candidates = ["target_cyrus_v4_20","target_nomi_v4_20","target_victor_v4_20"]
print(target_candidates)
#############################################
#GBM MODEL training for the given targets

st = time.time()

models = {}
for target in target_candidates:
    model = LGBMRegressor(
        n_estimators = 10,
        learning_rate = 0.1,
        max_depth = 1,
        colsample_bytree = 0.8
    )
    model.fit(train[feature_cols], train[target])
    
    #plot_importance(model, title = f'Feature importance of GBDT model over target : {target}',max_num_features = 30, figsize = (16,8), dpi = 300)
    #plt.savefig(repo_path + "/rounds/" + f"{date.today()}{prefix}_feature_importance_{target}.png", dpi = 300)
    models[target] = model

print(f'It takes %s minutes for training all {len(target_candidates)} models :' %((time.time()-st)/60))

#############################################
validation = pd.read_parquet(gh_repos_path + "/validation.parquet", columns=["era", "data_type"] + feature_cols + target_cols) 

validation = validation[validation["data_type"] == "validation"]

del validation["data_type"]

validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]


#############################################

X_val = torch.tensor(validation[feature_cols].values, dtype = torch.float32, requires_grad = True)

#LGBM models
for target in target_candidates:
    print(target)
    validation[f"prediction_{target}"] = models[target].predict(validation[feature_cols])

#X_val = X_val.detach().numpy()

#NN model
    
y_pred = model_nn(X_val)
y_pred = y_pred.detach().numpy()

validation[f"prediction_{target_cyrus}_nn"] = y_pred


#############################################
#function for cumulative correlation score

def cumulative_correlation(target_candidates : list, plot_save : bool) -> dict:
    correlations = {}
    cumulative_correlations = {}
    #GBM
    for target in target_candidates:
        correlations[f"prediction_{target}"] = validation.groupby("era").apply(lambda d: numerai_corr(d[f"prediction_{target}"], d["target"]))
        cumulative_correlations[f"prediction_{target}"] = correlations[f"prediction_{target}"].cumsum()
    
    #NN
    correlations[f"prediction_{target_cyrus}_nn"] = validation.groupby("era").apply(lambda d: numerai_corr(d[f"prediction_{target_cyrus}_nn"], d["target"]))
    cumulative_correlations[f"prediction_{target_cyrus}_nn"] = correlations[f"prediction_{target_cyrus}_nn"].cumsum()


    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    cumulative_correlations.plot(title="Cumulative Correlation of validation predictions", figsize=(10, 6), xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')
    if plot_save == True:
        plt.savefig(repo_path + "/neuralnetwork/" + f"{date.today()}{prefix}_cumulative_correlation_of_validation_predicitions_nn.png", dpi = 300)
    return correlations, cumulative_correlations

correlations, cumulative_correlations = cumulative_correlation(target_candidates, plot_save = True)
