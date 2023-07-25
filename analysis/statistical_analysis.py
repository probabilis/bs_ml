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

x = df[features[1]]


mean, var = overall_statistics(df, features)
print(mean, var)

var = np.var(x)
mean = np.mean(x)
print('var : ',var)
print('mean : ', mean)

histogram(x, features[1])
x.plot(kind = 'box')
plt.show()

plot_statistics(df_st,'mean')
plot_statistics(df_st,'variance')

histogram(df_st['feature_mean'], 'mean')
histogram(df_st['feature_variance'], 'variance')



