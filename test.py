"""
Project: Bachelor Project / Supervised Machine Learning / Gradient Boosting Machine based on Decision Trees
Script: Main Program
Author: Maximilian Gschaider
Date: 15.11.2023
MN: 12030366
------------------
Ref.: www.numer.ai
#(some of the code from the scripts provided was used)
"""
#official open-source repositories
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, plot_importance
import time
from datetime import date
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
#############################################
from preprocessing.cross_validators import era_splitting
from repo_utils import gh_repos_path, repo_path, loading, hyperparameter_loading, numerai_corr, neutralize, least_correlated

#############################################
#overall prefix for saving (directory management)
prefix = "00"

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()

#last_train_era = int(train["era"].unique()[-1])
#print(last_train_era)


validation = pd.read_parquet(gh_repos_path + "/validation.parquet", columns=["era", "data_type"] + feature_cols + target_cols) 
validation = validation[validation["data_type"] == "validation"]

del validation["data_type"]

validation = validation[validation["era"].isin(validation["era"].unique()[::4])]


validation[f"prediction_{target}_nn"] = model(validation[feature_cols])