"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
#own modules
from statistical_analysis_tools import statistics
from repo_utils import gh_repos_path
####################################

#loading training dataset v4.2
feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

df = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

####################################

#features and target
features = [f for f in df if f.startswith("feature")]
targets = [t for t in df if t.startswith("target")]
target = "target"

targets_20 = [t for t in df if t.endswith("20")]





df_st = statistics(df,targets_20)

df_st = df_st.set_index('feature_names')
#print(df_st)
#histogram(df_st['feature_variance'], 'variance')
targets_20_sorted = df_st['feature_variance'].sort_values(ascending = True)

print(targets_20_sorted)

target_correlations = df[targets_20].corr()
print(round(target_correlations,1))

sns.heatmap(target_correlations.corr())
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.savefig("target_correlations_heatmap.png")
plt.show()

#ta = target_correlations["target_nomi_v4_20"].sort_values(ascending = True)
#print(ta)