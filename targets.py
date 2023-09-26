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

x = 10

#############################################
#############################################
#############################################

least_correlated_targets = find_least_correlated_variables_pca(target_correlations_20.values[:, 1:], n_components = x)
print(least_correlated_targets)

columns = list(target_correlations_20)[1::]
print(columns)

sorted_least_target_corr_20 = [columns[i] for i in least_correlated_targets]
print(sorted_least_target_corr_20)

#############################################
#############################################
#############################################

#least_correlated_targets_v2 = find_least_correlated_variables_pca_v2(target_correlations_20.values[:, 1:], n_components = x)
#print(least_correlated_targets_v2)

#############################################
#############################################
#############################################


correlation_matrix = target_correlations_20

num_variables = len(correlation_matrix.columns)
min_correlation = 1.0  # Initialize with a high value

for i in range(num_variables):
    for j in range(i + 1, num_variables):
        correlation = abs(correlation_matrix.iloc[i, j])
        if correlation < min_correlation:
            min_correlation = correlation
            least_correlated_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])

least_correlated_variables = least_correlated_pair
print(least_correlated_variables)

#############################################
#############################################
#############################################

df_correlation = correlation_matrix

min_correlation = df_correlation.mask(np.tril(np.ones(df_correlation.shape)).astype(bool)).min().min()
least_correlated_pairs = np.where(np.abs(correlation_matrix) == min_correlation)

variable_names = df_correlation.columns
least_correlated_variable1 = variable_names[least_correlated_pairs[0][0]]
least_correlated_variable2 = variable_names[least_correlated_pairs[1][0]]

print(f"The least correlated variables are: {least_correlated_variable1} and {least_correlated_variable2}")
