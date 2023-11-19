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
import numpy as np
from lightgbm import LGBMRegressor, plot_importance
import time
from datetime import date
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from numerapi import NumerAPI
#own modules
from preprocessing.cross_validators import era_splitting
from repo_utils import numerai_corr, gh_repos_path, repo_path, neutralize

#############################################
#############################################
#############################################
#prefix for saving
prefix = "_round0_all_targets"

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

start = time.time()

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

#loading training dataset v4.2
train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

#############################################
#performing subsampling of the initial training dataset due to performance (era splitting)
train = era_splitting(train)
#start garbage collection interface / full collection
gc.collect()

#############################################

assert train["target"].equals(train["target_cyrus_v4_20"])
target_names = target_cols[1:]
targets_df = train[["era"] + target_names]

t20s = [t for t in target_names if t.endswith("_20")]
t60s = [t for t in target_names if t.endswith("_60")]

#############################################
#current best hyperparamter configuration for giving training dataframe determined through bayesian optimization
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
#using all target candidates 

target_candidates = t20s
target_candidates = target_candidates[:2]
print(target_candidates)

#############################################
#MODEL training for the given targets

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
    
    #plot_importance(model, title = f'Feature importance of GBDT model over target : {target}',max_num_features = 30, figsize = (16,8), dpi = 300)
    #plt.savefig(repo_path + "/rounds/" + f"{date.today()}{prefix}_feature_importance_{target}.png", dpi = 300)
    models[target] = model

print(f'It takes %s minutes for training all {len(target_candidates)} models :' %((time.time()-st)/60))

#############################################
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

def cumulative_correlation(target_candidates : list, plot_save : bool) -> dict:
    correlations = {}
    cumulative_correlations = {}
    for target in target_candidates:
        correlations[f"prediction_{target}"] = validation.groupby("era").apply(lambda d: numerai_corr(d[f"prediction_{target}"], d["target"]))
        cumulative_correlations[f"prediction_{target}"] = correlations[f"prediction_{target}"].cumsum()

    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    
    
    #cumulative_correlations.plot(title="Cumulative Correlation of validation predictions", figsize=(10, 6), xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')
    
    fig, [ax1,ax2] = plt.subplots(1,2, figsize = (14,6), width_ratios=[3, 1])

    cumulative_correlations.plot(ax = ax1,title="Cumulative Correlation of validation predictions",legend=False, xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')

    cumulative_correlations.plot(ax = ax2)
    h,l = ax2.get_legend_handles_labels()
    ax2.clear()
    ax2.legend(h,l,loc="upper right")
    ax2.axis("off")
    
    if plot_save == True:
        plt.savefig(repo_path + "/rounds/" + f"{date.today()}{prefix}_cumulative_correlation_of_validation_predicitions.png", dpi = 300)
    return correlations, cumulative_correlations

correlations, cumulative_correlations = cumulative_correlation(target_candidates, plot_save = True)

#############################################
#function for summary metrics statistics for all different targets

def summary_metrics(target_candidates : list, correlations : pd.DataFrame, cumulative_correlations : pd.DataFrame) -> pd.DataFrame:
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

summary_metrics_targets_df = summary_metrics(target_candidates, correlations, cumulative_correlations)
summary_metrics_targets_df.to_csv(repo_path + "/rounds/" + f"{date.today()}{prefix}_summary_metrics_targets.csv")
print(summary_metrics_targets_df)

"""
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

# Print the ensemble predictions
pred_cols = ensemble_cols + ["ensemble"]
validation[pred_cols]

#############################################
#ENSEMBLE model performance

def cumulative_correlations_ensemble(pred_cols, plot_save):
    correlations= {}
    cumulative_correlations = {}
    for col in pred_cols:
        correlations[col] = validation.groupby("era").apply(lambda d: numerai_corr(d[col], d["target"]))
        cumulative_correlations[col] = correlations[col].cumsum() 

    cumulative_correlations = pd.DataFrame(cumulative_correlations)
    cumulative_correlations.plot(title="Cumulative Correlation of validation predictions incl. ensemble model", figsize=(10, 6), xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')
    if plot_save == True:
        plt.savefig(repo_path + "/rounds/" + f"{date.today()}{prefix}_cumulative_correlation_of_validation_predicitions_ensemble.png", dpi = 300)
    return correlations, cumulative_correlations

correlations, cumulative_correlations = cumulative_correlations_ensemble(pred_cols, plot_save=True)

#############################################
#function for summary metrics statistics for all different targets

def summary_metrics_ensemble(pred_cols, correlations, cumulative_correlations) -> pd.DataFrame:
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

summary_metrics_ensemble_df = summary_metrics_ensemble(pred_cols, correlations, cumulative_correlations)
summary_metrics_ensemble_df.to_csv(repo_path + "/rounds/" + f"{date.today()}{prefix}_summary_metrics_ensemble.csv")
print(summary_metrics_ensemble_df)

#############################################
#feature neutralization
feature_sets = feature_metadata["feature_sets"]

sizes = ["small", "medium", "all"]
groups = ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution", "agility", "serenity", "all"]
subgroups = {}
for size in sizes:
    subgroups[size] = {}
    for group in groups:
        # intersection of feature sets
        subgroups[size][group] = set(feature_sets[size]).intersection(set(feature_sets[group]))

# as data frame
pd.DataFrame(subgroups).applymap(len).sort_values(by="all", ascending=False)

for group in groups:
    feature_subset = list(subgroups["medium"][group])
    neutralized = validation.groupby("era").apply(lambda d: neutralize(d["ensemble"], d[feature_subset]))
    validation[f"neutralized_{group}"] = neutralized.reset_index().set_index("id")["ensemble"] 

prediction_cols_groups = ["ensemble"] + [f"neutralized_{group}" for group in groups]
print(prediction_cols_groups)
correlations_neutral = {}
cumulative_correlations_neutral = {}
for col in prediction_cols_groups:
    correlations_neutral[col] = validation.groupby("era").apply(lambda d: numerai_corr(d[col], d["target"]))
    cumulative_correlations_neutral[col] = correlations_neutral[col].cumsum() 
pd.DataFrame(cumulative_correlations_neutral).plot(title="Cumulative Correlation of Neutralized Predictions", figsize=(10, 6), xticks=[])
plt.savefig(repo_path + "/rounds/" + f"{date.today()}{prefix}_cumulative_correlation_of_validation_predicitions_neutralization_ensemble.png", dpi = 300)

#############################################

pred_cols_neutral = ["ensemble"] + ["neutralized_serenity"]

def summary_metrics_neutralized_ensemble(pred_cols_neutral, correlations_neutral, cumulative_correlations_neutral) -> pd.DataFrame:
    summary_metrics = {}
    for col in pred_cols_neutral:
        mean = correlations_neutral[col].mean()
        std = correlations_neutral[col].std()
        sharpe = mean / std
        rolling_max = cumulative_correlations_neutral[col].expanding(min_periods=1).max()
        max_drawdown = (rolling_max - cumulative_correlations_neutral[col]).max()
        summary_metrics[col] = {
            "mean": mean,
            "std": std,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }
    pd.set_option('display.float_format', lambda x: '%f' % x)
    summary = pd.DataFrame(summary_metrics).T
    return summary

summary_metrics_neutralized_ensemble_df = summary_metrics_neutralized_ensemble(pred_cols_neutral, correlations_neutral, cumulative_correlations_neutral)
summary_metrics_neutralized_ensemble_df.to_csv(repo_path + "/rounds/" + f"{date.today()}{prefix}_summary_metrics_neutralized_ensemble.csv")
print(summary_metrics_ensemble_df)

#############################################
#ENSEMBLE predicting 

feature_subset = list(subgroups["medium"]["serenity"])

def predict_neutral(live_features: pd.DataFrame) -> pd.DataFrame:
    # make predictions using all features
    predictions = pd.DataFrame(index = live_features.index)

    for target in favorite_targets:
        predictions[target] = models[target].predict(live_features[feature_cols])
        
    # ensemble predictions
    ensemble = predictions.rank(pct=True).mean(axis=1)
    # neutralize predictions to a subset of features

    neutralized = neutralize(ensemble, live_features[feature_subset], 1.0)
    submission = pd.Series(neutralized).rank(pct=True, method="first")
    return submission.to_frame("prediction")

live_features = pd.read_parquet(gh_repos_path + "/live.parquet", columns=feature_cols)
predictions = predict_neutral(live_features)


print("----predictions-----")
print(predictions)
predictions.to_csv(repo_path + "/rounds/" + f"{date.today()}{prefix}_neutralized_predictions.csv")

print(f'It takes %s minutes in total to run main.py.' %((time.time()-start)/60))
"""