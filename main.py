"""
Project: Bachelor Machine Learning 
Script: Main Program
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import time
from datetime import date
import os
import json
import gc
import sys
import csv
from sklearn.model_selection import cross_val_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from numerapi import NumerAPI
sys.path.append('../')

from preprocessing.cross_validators import era_splitting
from preprocessing.pca_dimensional_reduction import dim_reduction
from utils import loading_dataset, numerai_corr, gh_repos_path, path_val, repo_path

#############################################
#############################################
#############################################

#loading dataset
#for old v4.0 data framework
#df, features, target, eras = loading_dataset()

napi = NumerAPI()
napi.download_dataset("v4.2/train_int8.parquet", gh_repos_path + "/train.parquet");
napi.download_dataset("v4.2/features.json", gh_repos_path + "/features.json");

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]
train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

train = train[train["era"].isin(train["era"].unique()[::4])]

assert train["target"].equals(train["target_cyrus_v4_20"])
target_names = target_cols[1:]
targets_df = train[["era"] + target_names]

t20s = [t for t in target_names if t.endswith("_20")]
t60s = [t for t in target_names if t.endswith("_60")]

target_correlations = targets_df[target_names].corr()
sns.heatmap(target_correlations, cmap="coolwarm", xticklabels=False, yticklabels=False);
target_correlations.to_csv(repo_path + "/analysis/target_correlations.csv")
plt.savefig(repo_path + "/figures/" + "target_correlations", dpi=300)

#############################################

#splitting the eras
#train, eras_ = era_splitting(df, eras)

#del df ; gc.collect()
"""
#############################################

#n = 100
#df_pca, features_pca = dim_reduction(train,features,target,n)
#del df_

#############################################

#loading the specific hyperparameter configuration from bayesian optimization

filename = "params_bayes_ip=20_ni=300_2023-09-15_n=300.csv"
path = repo_path + "/models/" + filename

params_gbm = pd.read_csv(path).to_dict(orient = "list")
params_gbm.pop("Unnamed: 0")

max_depth = params_gbm['max_depth'][0]
learning_rate = params_gbm['learning_rate'][0]
colsample_bytree = params_gbm['colsample_bytree'][0]
n_trees = int(round(params_gbm['n_estimators'][0],1))

#############################################
#defining the target candidates for ensemble modeling

target_candidates = ["target_cyrus_v4_20", "target_waldo_v4_20", "target_victor_v4_20", "target_xerxes_v4_20"]

#############################################

models = {}
for target in target_candidates:
    model = LGBMRegressor(
        n_estimators = n_trees,
        learning_rate=learning_rate,
        max_depth = max_depth,
        colsample_bytree=colsample_bytree
    )
    model.fit(train[features], train[target]
    );
    models[target] = model

#############################################

validation = pd.read_parquet(path_val)

validation = validation[validation['data_type'].str.contains("validation")]
del validation["data_type"]

validation = era_splitting(validation, eras)

last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

for target in target_candidates:
    validation[f"prediction_{target}"] = models[target].predict(validation[features])
    
pred_cols = [f"prediction_{target}" for target in target_candidates]

print(validation[pred_cols])

#############################################

def cumulative_correlations() -> dict:
    correlations = {}
    cumulative_correlations = {}
    for target in target_candidates:
        correlations[f"prediction_{target}"] = validation.groupby("era").apply(lambda d: numerai_corr(d[f"prediction_{target}"], d["target"]))
        cumulative_correlations[f"prediction_{target}"] = correlations[f"prediction_{target}"].cumsum() 

    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    cumulative_correlations.plot(title="Cumulative Correlation of validation Predictions", figsize=(10, 6), xticks=[]);
    plt.savefig("cumulative_correlation_of_validation_predicitions.png", dpi = 300)
    return correlations

correlations = cumulative_correlations()

#############################################

def summary_metrics(correlations) -> pd.DataFrame:
    summary_metrics = {}
    for target in target_candidates:
        # per era correlation between this target and cyrus 
        mean_corr_with_cryus = validation.groupby("era").apply(lambda d: d[target].corr(d["target_cyrus_v4_20"])).mean()
        # per era correlation between predictions of the model trained on this target and cyrus
        mean = correlations[f"prediction_{target}"].mean()
        std = correlations[f"prediction_{target}"].std()
        sharpe = mean / std
        rolling_max = cumulative_correlations[f"prediction_{target}"].expanding(min_periods=1).max()
        max_drawdown = (rolling_max - cumulative_correlations[f"prediction_{target}"]).max()
        summary_metrics[f"prediction_{target}"] = {
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "mean_corr_with_cryus": mean_corr_with_cryus,
        }
    pd.set_option('display.float_format', lambda x: '%f' % x)
    summary = pd.DataFrame(summary_metrics).T
    return summary

#summary = summary_metrics(correlations)
"""