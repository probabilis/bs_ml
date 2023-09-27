import sys
import pandas as pd
sys.path.append('../')
from repo_utils import loading_dataset, feature_corr, get_biggest_change_features

#############################################

df, features, target, eras = loading_dataset()

#############################################

corrs_ = feature_corr(df, "era", "target")

n = 50
risky = get_biggest_change_features(corrs_, n)
print(risky)


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