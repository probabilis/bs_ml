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
#own modules
sys.path.append('../')
from repo_utils import repo_path, gh_repos_path, numerai_corr, fontsize, fontsize_title
from preprocessing.cross_validators import era_splitting

#############################################

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

#loading training dataset v4.2
train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

#############################################

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
    
    fig, axs = plt.subplots( int(len(groups[:-1])/2), 2, sharey = True) #, sharex=True
    fig.set_size_inches(12,10)

    for i, group in enumerate(groups[:-1]):

        j = 0
        feature_subset = list(subgroups["medium"][group])
        per_era_corrs = pd.DataFrame(index = train.era.unique())

        for feature_name in feature_subset:
            per_era_corrs[feature_name] = train.groupby("era").apply(lambda df: numerai_corr(df[feature_name], df["target"]))
        
        per_era_corrs *= np.sign(per_era_corrs.mean())

        if i >= 4:
            #coordinates for plotting
            j = 1 ; i = i - 4
        
        per_era_corrs.cumsum().plot(ax = axs[i,j], title= f"Cumulative corr. of features group ${group}$ with main target", legend=False, xlabel="eras")
    
        del per_era_corrs

        axs[i][0].set_ylabel("$\\Sigma_i$ corr($x_i$, $y_i$)")
    
    #axs.set_title("per era correlation", loc = 'left', pad=10, fontsize = fontsize)
    fig.tight_layout()
    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "per_era_correlations.png", dpi=300)
    plt.show()

df = per_era_correlations(save_plot = True)
