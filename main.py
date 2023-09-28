"""
Project: Bachelor Project / Supervised Machine Learning / Gradient Boosting Machine based on Decision Trees
Script: Main Program
Author: Maximilian Gschaider
Date: 22.09.2023
MN: 12030366
------------------
Ref.: www.numer.ai
"""
#official open-source repositories
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, plot_importance
import time
from datetime import date
import json
import gc
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from numerapi import NumerAPI
#own modules
sys.path.append('../')
from preprocessing.cross_validators import era_splitting
from repo_utils import numerai_corr, gh_repos_path, repo_path

#############################################
#############################################
#############################################
#prefix for saving
prefix = "_round2"

#numer.AI official API for retrieving and pushing data
napi = NumerAPI()
#train set
napi.download_dataset("v4.2/train_int8.parquet", gh_repos_path + "/train.parquet")
#validation set
napi.download_dataset("v4.2/validation_int8.parquet", gh_repos_path + "/validation.parquet" )
#live dataset 
napi.download_dataset("v4.2/live_int8.parquet", gh_repos_path + "/live.parquet")
#features metadata
napi.download_dataset("v4.2/features.json", gh_repos_path + "/features.json")

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

#loading training dataset v4.2
train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

#############################################
#perform subsampling / era splitting due to performance

train = era_splitting(train)

gc.collect()

#############################################

assert train["target"].equals(train["target_cyrus_v4_20"])
target_names = target_cols[1:]
targets_df = train[["era"] + target_names]

t20s = [t for t in target_names if t.endswith("_20")]
t60s = [t for t in target_names if t.endswith("_60")]

target_correlations = targets_df[target_names].corr()
def plot_target_correlations(plot_save) -> None:
    sns.heatmap(target_correlations, cmap="coolwarm", xticklabels=False, yticklabels=False);
    target_correlations.to_csv(repo_path + "/analysis/target_correlations.csv")
    if plot_save == True:
        plt.savefig(repo_path + "/rounds/" + "target_correlations", dpi=300)

#############################################
#loading the specific hyperparameter configuration from bayesian optimization

filename = "params_bayes_ip=10_ni=100_2023-09-23_n=300.csv"

def hyperparameter_loading(filename):
    path = repo_path + "/models/" + filename
    params_gbm = pd.read_csv(path).to_dict(orient = "list")
    params_gbm.pop("Unnamed: 0")
    return params_gbm

params_gbm = hyperparameter_loading(filename)

max_depth = params_gbm['max_depth'][0]
learning_rate = params_gbm['learning_rate'][0]
colsample_bytree = params_gbm['colsample_bytree'][0]
n_trees = int(round(params_gbm['n_estimators'][0],1))

#############################################
#defining the target candidates for ensemble modeling

target_correlations_20 = targets_df[t20s].corr()
target_correlations_20.to_csv(repo_path + "/rounds/" + f"{date.today()}_target_correlations_20{prefix}.csv")

def least_correlated(df_correlation, amount):
    min_correlation = df_correlation.mask(np.tril(np.ones(df_correlation.shape)).astype(bool)).min().min()
    least_correlated_pairs = np.where(np.abs(df_correlation) == min_correlation)

    variable_names = df_correlation.columns

    least_correlated_variables = []

    if amount > 0:
        for i in range(amount):

            least_correlated_variable = variable_names[least_correlated_pairs[i][0]]
            least_correlated_variables.append(least_correlated_variable)

    return least_correlated_variables

target_candidates = least_correlated(target_correlations_20, amount = 1)

#############################################
#least correlated targets plus cyrus and nomi

top_targets = ["target_cyrus_v4_20","target_nomi_v4_20","target_victor_v4_20"]

target_candidates.extend(top_targets)

print(target_candidates)

#############################################

st = time.time()

models = {}
for target in target_candidates:
    model = LGBMRegressor(
        n_estimators = n_trees,
        learning_rate=learning_rate,
        max_depth = max_depth,
        colsample_bytree=colsample_bytree
    )
    model.fit(train[feature_cols], train[target])
    
    plot_importance(model, title = f'Feature importance of model with target : {target}',max_num_features = 30, figsize = (12,8), dpi = 300)
    plt.savefig(repo_path + "/rounds/" + f"{date.today()}_feature_importance_{target}{prefix}.png", dpi = 300)
    models[target] = model

print('It takes %s minutes for training the models :' %((time.time()-st)/60))

#############################################
#getting validation data

#loading validation data v4.2
validation = pd.read_parquet(gh_repos_path + "/validation.parquet", columns=["era", "data_type"] + feature_cols + target_cols) 

validation = validation[validation["data_type"] == "validation"]

del validation["data_type"]

validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

for target in target_candidates:
    validation[f"prediction_{target}"] = models[target].predict(validation[feature_cols])
    
pred_cols = [f"prediction_{target}" for target in target_candidates]

#############################################
#function for cumulative correlation score

def cumulative_correlations_targets(plot_save) -> dict:
    correlations = {}
    cumulative_correlations = {}
    for target in target_candidates:
        correlations[f"prediction_{target}"] = validation.groupby("era").apply(lambda d: numerai_corr(d[f"prediction_{target}"], d["target"]))
        cumulative_correlations[f"prediction_{target}"] = correlations[f"prediction_{target}"].cumsum() 

    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    cumulative_correlations.plot(title="Cumulative Correlation of validation Predictions", figsize=(10, 6), xticks=[]);
    if plot_save == True:
        plt.savefig(repo_path + "/rounds/" + f"{date.today()}_cumulative_correlation_of_validation_predicitions{prefix}.png", dpi = 300)
    return correlations, cumulative_correlations

correlations, cumulative_correlations = cumulative_correlations_targets(plot_save = True)

#############################################
#defining function for summary metrics

def summary_metrics_targets() -> pd.DataFrame:
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

summary_metrics_targets_df = summary_metrics_targets()
summary_metrics_targets_df.to_csv(repo_path + "/rounds/" + f"{date.today()}_summary_metrics_targets{prefix}.csv")
print(summary_metrics_targets_df)

#############################################
#############################################
#############################################
#ENSEMBLE modeling

# Ensemble predictions together with a simple average
numerai_selected_targets = ["target_cyrus_v4_20", "target_victor_v4_20"]

#favorite_targets = [element for element in least_correlated_targets[0:2]]
favorite_targets = target_candidates
favorite_targets.extend(target for target in numerai_selected_targets if target not in favorite_targets)

print(favorite_targets)

ensemble_cols = [f"prediction_{target}" for target in favorite_targets]
#ensure that the ensemble score are ranked by percentile (pct = True)
validation["ensemble"] = validation.groupby("era")[ensemble_cols].rank(pct=True).mean(axis=1)
#PROBLEMO


# Print the ensemble predictions
pred_cols = ensemble_cols + ["ensemble"]
validation[pred_cols]

#############################################
#ENSEMBLE model performance

def cumulative_correlations_ensemble(save_plot):
    correlations= {}
    cumulative_correlations = {}
    for col in pred_cols:
        correlations[col] = validation.groupby("era").apply(lambda d: numerai_corr(d[col], d["target"]))
        cumulative_correlations[col] = correlations[col].cumsum() 

    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    cumulative_correlations.plot(title="Cumulative Correlation of validation Predictions", figsize=(10, 6), xticks=[])
    if save_plot == True:
        plt.savefig(repo_path + "/rounds/" + f"{date.today()}_cumulative_correlation_of_validation_predicitions_ensemble{prefix}.png", dpi = 300)
    return correlations, cumulative_correlations

correlations, cumulative_correlations = cumulative_correlations_ensemble(save_plot=True)

def summary_metrics_ensemble() -> pd.DataFrame:
    summary_metrics = {}
    for col in pred_cols:
        mean = correlations[col].mean()
        std = correlations[col].std()
        sharpe = mean / std
        rolling_max = cumulative_correlations[col].expanding(min_periods=1).max()
        max_drawdown = (rolling_max - cumulative_correlations[col]).max()
        summary_metrics[col] = {
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }
    pd.set_option('display.float_format', lambda x: '%f' % x)
    summary = pd.DataFrame(summary_metrics).T
    return summary

summary_metrics_ensemble_df = summary_metrics_ensemble()
summary_metrics_ensemble_df.to_csv(repo_path + "/rounds/" + f"{date.today()}_summary_metrics_ensemble{prefix}.csv")
print(summary_metrics_ensemble_df)

#############################################
#ENSEMBLE predicting 

def predict_ensemble(live_features: pd.DataFrame) -> pd.DataFrame:
    # generate predictions from each model
    predictions = pd.DataFrame(index=live_features.index)
    for target in favorite_targets:
        predictions[target] = models[target].predict(live_features[feature_cols])
    # ensemble predictions
    ensemble = predictions.rank(pct=True).mean(axis=1)
    # format submission
    submission = ensemble.rank(pct=True, method="first")
    return submission.to_frame("prediction")

live_features = pd.read_parquet(gh_repos_path + "/live.parquet", columns=feature_cols)
predictions = predict_ensemble(live_features)
print("----predictions-----")
print(predictions)
predictions.to_csv(repo_path + "/rounds/" + f"{date.today()}_predictions{prefix}.csv")

