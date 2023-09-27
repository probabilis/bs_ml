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
#from numerapi import NumerAPI
#own modules
sys.path.append('../')
from repo_utils import repo_path, gh_repos_path, numerai_corr, fontsize, fontsize_title
from preprocessing.cross_validators import era_splitting
from data_loading import loading_datasets

#############################################

train, feature_cols, target_cols = loading_datasets()

df = era_splitting(train)

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_sets = feature_metadata["feature_sets"]

sizes = ["small", "medium", "all"]
groups = ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution", "agility", "serenity", "all"]
subgroups = {}

for size in sizes:
    subgroups[size] = {}
    for group in groups:
        # intersection of feature sets
        subgroups[size][group] = set(feature_sets[size]).intersection(set(feature_sets[group]))

subgroups_df = pd.DataFrame(subgroups).applymap(len).sort_values(by="all", ascending=False)

feature_cols = feature_metadata["feature_sets"]["medium"]

#print(feature_cols)
target_cols = feature_metadata["targets"]
train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)



def per_era_correlations(save_plot):
    
    fig, axs = plt.subplots((len(groups)/2, 2)) #, sharex=True
    fig.set_size_inches(12,16)

    for i, group in enumerate(groups[:-1]):
        j = 0
        feature_subset = list(subgroups["medium"][group])
        per_era_corrs = pd.DataFrame(index = train.era.unique())

        for feature_name in feature_subset:
            per_era_corrs[feature_name] = train.groupby("era").apply(lambda df: numerai_corr(df[feature_name], df["target"]))
        
        per_era_corrs *= np.sign(per_era_corrs.mean())

        if i >= 4:
            j = 1
            i = i - 4
        
        per_era_corrs.cumsum().plot(ax = axs[i,j], figsize=(15, 5), title= f"Cumulative sum of correlations of features group {group} to the target (w/ negative flipped)", legend=False, xlabel="eras")
    
    
    #axs.set_title("per era correlation", loc = 'left', pad=10, fontsize = fontsize)
    #axs.set_xlabel("eras")
    fig.tight_layout()
    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "per_era_correlations.png", dpi=300)
    plt.show()

df = per_era_correlations(save_plot = True)
