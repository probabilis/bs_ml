import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.append('../')
from repo_utils import loading_dataset, feature_corr, get_biggest_change_features, repo_path, gh_repos_path

#############################################

def neutralize(predictions: pd.DataFrame, features: pd.DataFrame, proportion: float = 1.0) -> pd.DataFrame:
    """
    newer version from v4.2 datasets / from sept. 2023 / Neutralize predictions to features
    params: df, features, proportion
    df ...          input df / vector over the features room
    features ...     array / columns of df
    proportion ...  scalar 
    ---------------
    return: new neutralized df
    """
    # add a constant term the features so we can fit the bias/offset term
    out_ = np.hstack((features, np.array([np.mean(predictions)] * len(features)).reshape(-1, 1)))
    print(features)
    # remove the component of the predictions that are linearly correlated with features
    return predictions - proportion * features @ (np.linalg.pinv(features, rcond=1e-6) @ predictions)

#############################################
#############################################
#############################################

def predict_neutral(live_features: pd.DataFrame) -> pd.DataFrame:
    # make predictions using all features
    predictions = pd.DataFrame(index = live_features.index)

    for target in favorite_targets:
        predictions[target] = models[target].predict(live_features[feature_cols])
        
    # ensemble predictions
    ensemble = predictions.rank(pct=True).mean(axis=1)
    # neutralize predictions to a subset of features

    neutralized = neutralize(ensemble, live_features[feature_subset], 1.0)
    submission = pd.Series(neutralized).rank(pct=True, method="first")
    return submission.to_frame("prediction")

#############################################

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

feature_subset = list(subgroups["medium"][group])

filename = "2023-09-28_round2_predictions.csv"
df = pd.read_csv(repo_path + "/rounds/" + filename)
#print(df["prediction"])

#print(df.iloc[:,1])

neutralized = df.apply(lambda d: neutralize(d.iloc[:,1], d[feature_subset]))


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