import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import loading_dataset
from statistical_analysis_tools import statistics, plot_statistics, histogram, overall_statistics

#############################################

df, features, target, eras = loading_dataset()

#############################################

df_st = statistics(df, features)
print(df_st)

mean, var = overall_statistics(df, features)
print(mean, var)

x = df[features[0]]
var = np.var(x)
mean = np.mean(x)
print('var of ' + str(x),var)
print('mean of ' + str(x), mean)


#histogram(x, features[0])
#x.plot(kind = 'box')
#plt.show()

plot_statistics(df_st,'mean', "train_df_features_mean", path_ = "/figures")
plot_statistics(df_st,'variance', "train_df_features_variance", path_ = "/figures")

histogram(df_st['feature_mean'], "train_df_hist_mean", path_ = "/figures")
histogram(df_st['feature_variance'], "train_df_hist_var", path_ = "/figures")

"""
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

