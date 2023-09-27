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
#own modules
sys.path.append('../')
from repo_utils import gh_repos_path
from data_loading import loading_datasets
from statistical_analysis_tools import statistics, plot_statistics, histogram, overall_statistics, plot_correlations
from preprocessing.cross_validators import era_splitting

#############################################

train, feature_cols, target_cols = loading_datasets()

df = era_splitting(train)

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_sets = feature_metadata["feature_sets"]

assert train["target"].equals(train["target_cyrus_v4_20"])
target_names = target_cols[1:]
targets_df = train[["era"] + target_names]

t20s = [t for t in target_names if t.endswith("_20")]

#############################################
#############################################
#############################################
#overall statistics

df_st = statistics(df, feature_cols)
print(df_st)

mean, var = overall_statistics(df, feature_cols)
print(mean, var)

#############################################

plot_statistics(df_st,'mean', "train_df_features_mean")
plot_statistics(df_st,'variance', "train_df_features_variance")

histogram(df_st['feature_mean'], "train_df_hist_mean")
histogram(df_st['feature_variance'], "train_df_hist_var")


#############################################
#############################################
#############################################
#TARGET correlations

target_correlations = targets_df[t20s].corr()
plot_correlations(target_correlations, plot_save = True, name = "target")
print("target correlation plot successfully created")
#############################################
#FEATURE correlations

feature_correlations = train[feature_cols].corr()
plot_correlations(feature_correlations, plot_save = True, name = "feature")
print("feature correlation plot successfully created")
#############################################