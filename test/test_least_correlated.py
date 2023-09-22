import numpy as np
import pandas as pd
import json
import itertools
from sklearn.decomposition import PCA
from utils import loading_dataset, numerai_corr, gh_repos_path, path_val, repo_path
# Example correlation matrix (replace with your data)
correlation_matrix = np.array([
                            [1.0, 0.2, 0.4, 0.1],
                               [0.2, 1.0, 0.3, 0.2],
                               [0.4, 0.3, 1.0, 0.5],
                               [0.1, 0.2, 0.5, 1.0]])

data = {'A': [45, 37, 42],
        'B': [38, 31, 26],
        'C': [10, 15, 17]
        }

df = pd.read_csv(repo_path + "/analysis/target_correlations.csv")
feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

target_cols = feature_metadata["targets"]
target_names = target_cols[1:]
t20s = [t for t in target_names if t.endswith("_20")]

def calculate_score(subset, correlation_matrix):
    # Calculate the score for a subset of variables
    return np.sum(np.abs(correlation_matrix[np.ix_(subset, subset)]))

def find_least_correlated_subset(correlation_matrix):
    num_vars = correlation_matrix.shape[0]
    all_vars = list(range(num_vars))
    current_subset = []
    
    while len(current_subset) < num_vars:
        best_score = float('inf')
        best_variable = None
        
        for var in all_vars:
            if var not in current_subset:
                subset_with_var = current_subset + [var]
                score = calculate_score(subset_with_var, correlation_matrix)
                if score < best_score:
                    best_score = score
                    best_variable = var
        
        current_subset.append(best_variable)
    
    return current_subset

def find_least_correlated_variables_v2(correlation_matrix):
    n = correlation_matrix.shape[0]
    
    # Create a list to store variable pairs and their correlation values
    variable_pairs = []

    for i in range(n):
        for j in range(i+1, n):
            variable_pairs.append((i, j, abs(correlation_matrix[i, j])))

    # Sort variable pairs based on correlation value in ascending order
    variable_pairs.sort(key=lambda x: x[2])

    # Initialize a set to store the least correlated variables
    least_correlated_variables = set()

    while len(least_correlated_variables) < n:
        # Get the pair with the smallest absolute correlation
        i, j, correlation = variable_pairs.pop(0)

        # Add the variables to the set of least correlated variables
        least_correlated_variables.add(i)
        least_correlated_variables.add(j)

        # Remove all pairs involving variables i or j
        variable_pairs = [(x, y, c) for x, y, c in variable_pairs if x != i and x != j and y != i and y != j]

    return least_correlated_variables

from sklearn.decomposition import PCA

def find_least_correlated_variables_pca(data, n_components):
    # Create a PCA model with the desired number of components
    pca = PCA(n_components=n_components)

    # Fit the PCA model to the data
    pca.fit(data)

    # Get the indices of the top n_components principal components
    top_components_indices = np.argsort(np.abs(pca.components_).sum(axis=0))[:n_components]

    return set(top_components_indices)

print(df)
columns = list(df.columns)[1::]
print(columns)
df_ = df.values[:, 1:]
print(df_.shape)

print("----------------------------------------")
least_correlated_subset = find_least_correlated_subset(df_)
print("Least correlated subset of variables:", least_correlated_subset)

sorted_list = [columns[i] for i in least_correlated_subset]
print(sorted_list)

print("----------------------------------------")

#least_correlated_subset_v2 = find_least_correlated_variables_v2(df_)
#print("Least correlated subset of variables:", least_correlated_subset_v2)

least_correlated_subset_pca = find_least_correlated_variables_pca(df_, 10)
print("Least correlated subset of variables using PCA:", least_correlated_subset_pca)

sorted_list = [columns[i] for i in least_correlated_subset_pca]
print(sorted_list)
"""
pca = PCA()
pca.fit(df)
print(pca.explained_variance_ratio_)
print(pca.score_samples(df))
"""




