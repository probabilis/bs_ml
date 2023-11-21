"""
Author: Maximilian Gschaider
MN: 12030366
"""
import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

"""
Ref: 
https://github.com/numerai/example-scripts/blob/master/analysis_and_tips.ipynb
"""

def era_splitting(df, window = 4):
    """
    params: df 
    window ...       overlapping window length
    ---------------
    return: reduced df with splitted eras
    """
    return df[df["era"].isin(df["era"].unique()[::window])]

"""
Reference at Scikit:
https://scikit-learn.org/stable/modules/cross_validation.html

TimeSeriesSplitGroups Class is written by Numerai. All Rights reserved.

Because the TimeSeriesSplit class in sklearn does not use groups and won't respect era boundries, we implement
a version that will
"""

class TimeSeriesSplitGroups(_BaseKFold):

    def __init__(self, n_splits = 5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, Y = None, groups = None):
        X, Y, groups = indexable(X, Y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(("Cannot have number of folds = {0} greater than the number of samples : {1}.").format(n_folds,n_groups))
        
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:

            yield (indices[groups.isin(group_list[:test_start])] ,
                   indices[groups.isin(group_list[test_start : test_start + test_size])]
                   )

