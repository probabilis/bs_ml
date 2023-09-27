import json
import pandas as pd
from numerapi import NumerAPI
from repo_utils import gh_repos_path

def loading_datasets():
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

    feature_metadata = json.load(open(gh_repos_path + "/features.json")) 

    feature_cols = feature_metadata["feature_sets"]["medium"]
    target_cols = feature_metadata["targets"]

    #loading training dataset v4.2
    train = pd.read_parquet(gh_repos_path + "/train.parquet", columns=["era"] + feature_cols + target_cols)

    return train, feature_cols, target_cols




