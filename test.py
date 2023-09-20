import numpy as np
import pandas as pd
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


df_ = df.get_value()

print(df_)

least_correlated_subset = find_least_correlated_subset(df_)
print("Least correlated subset of variables:", least_correlated_subset)

"""
pca = PCA()
pca.fit(df_)
print(pca.explained_variance_ratio_)

print(pca.score_samples(df_))
"""





