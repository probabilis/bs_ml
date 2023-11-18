"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import gc
#own modules
sys.path.append('../')
from repo_utils import gh_repos_path
from statistical_analysis_tools import statistics, plot_statistics, histogram, overall_statistics, plot_correlations
from preprocessing.cross_validators import era_splitting

#############################################
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

feature_sets = feature_metadata["feature_sets"]

assert train["target"].equals(train["target_cyrus_v4_20"])
target_names = target_cols[1:]
targets_df = train[["era"] + target_names]

t20s = [t for t in target_names if t.endswith("_20")]

#############################################
#############################################
#############################################
#FEATURE statistical analysis tests
#mean, var
#correlation

print("feature statistics")
df_st = statistics(train, feature_cols)
print(df_st)

mean, var = overall_statistics(train, feature_cols)
print(mean, var)

#############################################

plot_statistics(df_st,'mean',"feature","train_df_features_mean")

plot_statistics(df_st,'variance',"feature", "train_df_features_variance")

#histogram(df_st['feature_mean'], "train_df_hist_mean")
#histogram(df_st['feature_variance'], "train_df_hist_var")

#############################################
#FEATURE correlations

feature_correlations = train[feature_cols].corr()
plot_correlations(feature_correlations, plot_save = True, name = "feature")
print("feature correlation plot successfully created")
#############################################

del df_st, mean, var

#############################################
#############################################
#############################################
#TARGET statistical analysis tests
#mean, var
#correlation

print("target statistics")
df_st = statistics(targets_df, t20s)
print(df_st)

mean, var = overall_statistics(targets_df, t20s)
print(mean, var)

#############################################

plot_statistics(df_st,'mean', "targets_df_targets_mean")
plot_statistics(df_st,'variance', "targets_df_targets_variance")

#histogram(df_st['feature_mean'], "targets_df_hist_mean")
#histogram(df_st['feature_variance'], "targets_df_hist_var")

target_correlations = targets_df[t20s].corr()
plot_correlations(target_correlations, plot_save = True, name = "target")
print("target correlation plot successfully created")

histogram(train['feature_wetter_unbaffled_loma'], 'feature_wetter_unbaffled_loma')


