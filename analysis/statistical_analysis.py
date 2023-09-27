"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#own modules
sys.path.append('../')
from data_loading import data_loading
from statistical_analysis_tools import statistics, plot_statistics, histogram, overall_statistics
from preprocessing.cross_validators import era_splitting

#############################################

train, feature_cols, target_cols = data_loading()

df = era_splitting(train)

features = feature_cols

#############################################

df_st = statistics(df, features)
print(df_st)

mean, var = overall_statistics(df, features)
print(mean, var)

#############################################

plot_statistics(df_st,'mean', "train_df_features_mean", path_ = "/figures")
plot_statistics(df_st,'variance', "train_df_features_variance", path_ = "/figures")

histogram(df_st['feature_mean'], "train_df_hist_mean", path_ = "/figures")
histogram(df_st['feature_variance'], "train_df_hist_var", path_ = "/figures")


