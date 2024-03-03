"""
Author: Maximilian Gschaider
MN: 12030366
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import gc
sys.path.append('../')
from repo_utils import gh_repos_path, loading
from statistical_analysis_tools import statistics, plot_statistics, histogram, overall_statistics, plot_correlations

#############################################

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()
feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

gc.collect()

#############################################

feature_sets = feature_metadata["feature_sets"]

#############################################
#############################################
#FEATURE statistical analysis tests
#mean, var
#correlation

print("feature statistics")
df_st = statistics(train,"feature", feature_cols)
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
#TARGET statistical analysis tests
#mean, var
#correlation

print("target statistics")
df_st = statistics(targets_df,"target", t20s)
print(df_st)

mean, var = overall_statistics(targets_df, t20s)
print(mean, var)

#############################################

plot_statistics(df_st,'mean',"target", "train_df_targets_mean")
plot_statistics(df_st,'variance',"target", "train_df_targets_variance")

#histogram(df_st['feature_mean'], "targets_df_hist_mean")
#histogram(df_st['feature_variance'], "targets_df_hist_var")

target_correlations = targets_df[t20s].corr()
plot_correlations(target_correlations, plot_save = True, name = "target")
print("target correlation plot successfully created")

#############################################
#histogram plot of specific FEATURE
#histogram(train['feature_wetter_unbaffled_loma'], 'feature_wetter_unbaffled_loma')


