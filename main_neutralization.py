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
#############################################
from preprocessing.cross_validators import era_splitting
from repo_utils import gh_repos_path, repo_path, loading, hyperparameter_loading, numerai_corr, neutralize, least_correlated

#############################################
#overall prefix for saving (directory management)
prefix = "_round0_all_targets"

#############################################
#loading all necassary data from the reposiroty utils file 
train, feature_cols, target_cols, targets_df, t20s, t60s = loading()

#############################################
#current best hyperparamter configuration for giving training dataframe determined through bayesian optimization

#hyperparameter csv file
filename = "params_bayes_ip=10_ni=100_2023-09-23_n=300.csv"

max_depth, learning_rate, colsample_bytree, n_trees = hyperparameter_loading(filename)

print("loading check")
#############################################
#defining the target candidates for the ensemble model

target_correlations_20 = targets_df[t20s].corr()
target_correlations_20.to_csv(repo_path + "/rounds/" + f"{date.today()}{prefix}_target_correlations_20.csv")

least_correlated_targets = least_correlated(target_correlations_20, amount = 1)

#############################################
#least correlated targets plus cyrus and nomi

#top_targets = ["target_cyrus_v4_20","target_nomi_v4_20","target_victor_v4_20"]
top_targets = ["target_cyrus_v4_20","target_nomi_v4_20","target_victor_v4_20","target_ralph_v4_20","target_bravo_v4_20"]
target_candidates = least_correlated_targets.extend(top_targets)
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
    
    plot_importance(model, title = f'Feature importance of GBDT model over target : {target}',max_num_features = 30, figsize = (16,8), dpi = 300)
    plt.savefig(repo_path + "/rounds/" + f"{date.today()}{prefix}_feature_importance_{target}.png", dpi = 300)
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
    cumulative_correlations.plot(title="Cumulative Correlation of validation predictions", figsize=(10, 6), xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')
    #Scumulative_correlations.to_csv(repo_path + "/rounds/" + "val_pred.csv")
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
pd.DataFrame(cumulative_correlations_neutral).plot(title="Cumulative Correlation of Neutralized Predictions", figsize=(10, 6), xlabel='eras', ylabel='$\\Sigma_i$ corr($\\tilde{y}_i$, $y_i$)')
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
