"""
Project: Bachelor Machine Learning 
Script: Main Program
Author: Maximilian Gschaider
MN: 12030366
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import time
from datetime import date
import os
import gc
import sys
import csv
from sklearn.model_selection import cross_val_score
from pathlib import Path

sys.path.append('../')

from bs_ml.preprocessing.cross_validators import era_splitting
from bs_ml.preprocessing.pca_dimensional_reduction import dim_reduction
from bs_ml.utils import loading_dataset, repo_path

#############################################

df, features, target, eras = loading_dataset()

#############################################

df_, eras_ = era_splitting(df, eras)

del df ; gc.collect()

#############################################

n = 100
df_pca, features_pca = dim_reduction(df_,features,target,n)
del df_

#############################################

init_points = 10 ; n_iter = 5

filename_params = f"params_bayes_ip={init_points}_ni={n_iter}_{date.today()}_n={n}"



filename = repo_path + "/models/" + filename_params + ".csv"

with open(filename, 'r') as data:
    for line in csv.DictReader(data):
        params = line

print(params)