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
from scipy import stats
#from numerapi import NumerAPI
#own modules
sys.path.append('../')
from repo_utils import gh_repos_path, numerai_corr

#############################################
"""
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
"""

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

print(feature_cols)
target_cols = feature_metadata["targets"]

train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

#creating unique per_erra_corss Df
per_era_corrs = pd.DataFrame(index = train.era.unique())

#print(subgroups["medium"]["serenity"])
#loading training dataset v4.2

#for feature_name in feature_cols[""]