import pandas as pd
import numpy as np
import time
import json
import gc
from numerapi import NumerAPI
#own modules
from preprocessing.cross_validators import era_splitting
from repo_utils import numerai_corr, gh_repos_path, repo_path, neutralize


def loading():
    #numer.AI official API for retrieving and pushing data
    napi = NumerAPI()
    #train set
    napi.download_dataset("v4.2/train_int8.parquet", gh_repos_path + "/train.parquet")
    #validation set
    napi.download_dataset("v4.2/validation_int8.parquet", gh_repos_path + "/validation.parquet" )
    #live dataset 
    napi.download_dataset("v4.2/live_int8.parquet", gh_repos_path + "/live.parquet")
    #features metadata
    napi.download_dataset("v4.2/features.json", gh_repos_path + "/features.json")

    start = time.time()

    feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

    feature_cols = feature_metadata["feature_sets"]["medium"]
    target_cols = feature_metadata["targets"]

    #loading training dataset v4.2
    train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

    #############################################
    #performing subsampling of the initial training dataset due to performance (era splitting)
    train = era_splitting(train)
    #start garbage collection interface / full collection
    gc.collect()

    #############################################

    assert train["target"].equals(train["target_cyrus_v4_20"])
    target_names = target_cols[1:]
    targets_df = train[["era"] + target_names]

    t20s = [t for t in target_names if t.endswith("_20")]
    t60s = [t for t in target_names if t.endswith("_60")]

    return train, feature_cols, target_cols, targets_df, t20s, t60s

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()