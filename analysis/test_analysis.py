import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.append('../')
from repo_utils import repo_path, gh_repos_path

#############################################

#############################################

#sys.path.insert(0, '/home/Documents/github_repos/bs_ml')

"""

df = pd.read_csv(repo_path + "/rounds/" + "val_pred.csv")
print(len(df))
x = np.arange(0,len(df))

df = df.drop(columns = "era")
df.plot()

y = lambda x : x
#y_ = y(x)

plt.plot(x,y(x))
plt.show()

"""

x = pd.read_csv(repo_path + "/rounds/2023-11-19_round0_all_targets_summary_metrics_targets.csv")#.to_dict(orient = "list")

x = x.sort_values(by = ["sharpe"], ascending = False)
print(x)

#############################################
"""
df, features, target, eras = loading_dataset()

#############################################

corrs_ = feature_corr(df, "era", "target")

n = 50
risky = get_biggest_change_features(corrs_, n)
print(risky)
"""

"""
histogram(x, features[0])
x.plot(kind = 'box')
plt.show()


skews = get_skewed_columns(df[features])
first = skews.index[0]
print(first)
x = df[first]
#feature_suppressed_unremovable_telephone

from scipy.stats import shapiro, norm
#my_data = norm.rvs(size=500)
shap = shapiro(x)
print('shap :', shap)

histogram(x, first)
x.plot(kind = 'box')
plt.show()

"""