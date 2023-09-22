"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import numpy as np
from sklearn.decomposition import PCA

########################################

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


def find_least_correlated_variables_pca(data, n_components):
    # Create a PCA model with the desired number of components
    pca = PCA(n_components=n_components)

    # Fit the PCA model to the data
    pca.fit(data)

    # Get the indices of the top n_components principal components
    top_components_indices = np.argsort(np.abs(pca.components_).sum(axis=0))[:n_components]

    return set(top_components_indices)