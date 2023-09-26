import json
import pandas as pd
import numpy as np 
import gc
import sys
from repo_utils import gh_repos_path
#own modules
sys.path.append('../')
from preprocessing.cross_validators import era_splitting
from analysis.least_correlated import find_least_correlated_variables_pca, find_least_correlated_variables_pca_v2   
from repo_utils import numerai_corr, gh_repos_path, repo_path

#targets = ['target_tyler_v4_20', 'target_victor_v4_20', 'target_ralph_v4_20', 'target_waldo_v4_20', 'target_jerome_v4_20', 'target_janet_v4_20', 'target_ben_v4_20', 'target_alan_v4_20', 'target_paul_v4_20', 'target_george_v4_20', 'target_william_v4_20', 'target_arthur_v4_20', 'target_thomas_v4_20', 'target_cyrus_v4_20', 'target_caroline_v4_20', 'target_sam_v4_20', 'target_xerxes_v4_20', 'target_alpha_v4_20', 'target_bravo_v4_20', 'target_charlie_v4_20', 'target_delta_v4_20', 'target_echo_v4_20', 'target_jeremy_v4_20', 'target_cyrus_v4_20']
#print(targets)

feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

#loading training dataset v4.2
train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

#############################################
#perform subsampling / era splitting due to performance

train = era_splitting(train)

gc.collect()

assert train["target"].equals(train["target_cyrus_v4_20"])
target_names = target_cols[1:]
targets_df = train[["era"] + target_names]


t20s = [t for t in target_names if t.endswith("_20")]

target_correlations_20 = targets_df[t20s].corr()
#print(target_correlations_20.values[:, 1:])

least_correlated_targets = find_least_correlated_variables_pca(target_correlations_20.values[:, 1:], n_components = x)
print(least_correlated_targets)
least_correlated_targets_v2 = find_least_correlated_variables_pca_v2(target_correlations_20.values[:, 1:], n_components = x)
print(least_correlated_targets_v2)