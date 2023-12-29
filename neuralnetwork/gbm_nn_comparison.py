"""
Author: Maximilian Gschaider
MN: 12030366
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

target = "target_cyrus_v4_20"

nn_model_name = "nn_model_n_epochs=10_batch_size=500"
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

model_nn.load_state_dict(torch.load(nn_model_name))
model_nn.eval()

#############################################
#GBM MODEL training for the given targets

st = time.time()

model_gbm = LGBMRegressor(
        n_estimators = 6352,
        learning_rate = 0.02,
        max_depth = 1,
        colsample_bytree = 0.9
    )
model_gbm.fit(train[feature_cols], train[target])

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

#LGBM model
validation[f"prediction_{target}_gbm"] = model_gbm.predict(validation[feature_cols])

#NN model
y_pred = model_nn(X_val) #.detach().numpy()
y_pred = y_pred.detach().numpy()
validation[f"prediction_{target}_nn"] = y_pred

#############################################
#function for cumulative correlation score
def cumulative_correlation_model_comparison(models : list, plot_save : bool) -> dict:    
    correlations = {}
    cumulative_correlations = {}
    for model in models:
        correlations[f"prediction_{target}_{model}"] = validation.groupby("era").apply(lambda d: numerai_corr(d[f"prediction_{target}_{model}"], d["target"]))
        cumulative_correlations[f"prediction_{target}_{model}"] = correlations[f"prediction_{target}_{model}"].cumsum()

    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    cumulative_correlations.plot(title="Cumulative Correlation of validation predictions / GBM and NN comparison", figsize=(10, 6), xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')
    if plot_save == True:
        plt.savefig(repo_path + "/neuralnetwork/" + f"{date.today()}{prefix}_cumulative_correlation_of_validation_predicitions_gbm_nn_0.png", dpi = 300)
    return correlations, cumulative_correlations


models = ["gbm", "nn"]
correlations, cumulative_correlations = cumulative_correlation_model_comparison(models, plot_save = True)
