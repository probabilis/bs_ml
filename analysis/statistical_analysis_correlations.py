"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from datetime import date
#from numerapi import NumerAPI
#own modules
sys.path.append('../')
from repo_utils import repo_path, gh_repos_path, fontsize, fontsize_title
from data_loading import loading_datasets
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
#function for plotting

def plot_correlations(df_correlations, plot_save, name = None) -> None:
    mask = np.triu(np.ones_like(df_correlations, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df_correlations, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=False, yticklabels=False)
    fig.suptitle(f'{df_correlations} matrix', fontsize=fontsize_title)
    #sns.heatmap(df_correlations, cmap="coolwarm", xticklabels=False, yticklabels=False);
    df_correlations.to_csv(repo_path + f"/analysis/{name}_correlations_matrix_{date.today()}.csv")

    if plot_save == True:
        plt.savefig(repo_path + f"/figures/{name}_correlations_matrix_{date.today()}.png", dpi=300)

#############################################
#############################################
#############################################
#TARGET correlations

target_correlations = targets_df[t20s].corr()

plot_correlations(target_correlations, plot_save = True, name = "target")

#############################################
#FEATURE correlations

feature_correlations = train[feature_cols].corr()

plot_correlations(feature_correlations, plot_save = True, name = "feature")


#############################################




