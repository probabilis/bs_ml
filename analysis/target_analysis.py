"""
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import time
import os
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from statistical_analysis_tools import statistics

####################################

path = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "train.parquet")
#print(path)

df = pd.read_parquet(path)

####################################

#features and target
features = [f for f in df if f.startswith("feature")]
targets = [t for t in df if t.startswith("target")]
target = "target"

targets_20 = [t for t in df if t.endswith("20")]

df_st = statistics(df,targets_20)

def histogram(x, name):
	q25, q75 = np.percentile(x, [25, 75])
	bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
	bins = round((x.max() - x.min()) / bin_width)
	plt.hist(x, bins = bins)
	plt.title(name)
	#fig = plt.gcf()
	#fig.set_size_inches(12,10)
	#fig.tight_layout()
	plt.show()

df_st = df_st.set_index('feature_names')
#print(df_st)
#histogram(df_st['feature_variance'], 'variance')
targets_20_sorted = df_st['feature_variance'].sort_values(ascending = True)

print(targets_20_sorted)

target_correlations = df[targets_20].corr()
print(round(target_correlations,1))

import seaborn as sns

sns.heatmap(target_correlations.corr())
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.savefig("target_correlations_heatmap.png")
plt.show()

#ta = target_correlations["target_nomi_v4_20"].sort_values(ascending = True)
#print(ta)