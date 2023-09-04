import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy


##########################
#initialization

path_ = os.path.join(os.path.expanduser('~'), 'Documents', 'bachelor', "train.parquet")
repo_path = os.path.join(os.path.expanduser('~'), 'Documents', 'github_repos', "bs_ml")

fontsize_title = 16
fontsize = 12

##########################
#functions

def loading_dataset():
    """
    params: path, filename 
    path ...        str / relative path folders for file 
    filename ...    str / filename of df
    ---------------
    return: df, features, target, eras
    df ...          Dataframe
    features ...    features vector
    target ....     target vector
    eras   ...      eras vector
    """
    df = pd.read_parquet(path_)
    print("Loaded dataframe from path : ", path_)

    features = [f for f in df if f.startswith("feature")]
    target = "target"
    df["erano"] = df.era.astype(int)
    eras = df.erano
    return df, features, target, eras


def numerai_score(y, y_pred, eras):
    """
    params: y, y_pred, eras 
    y ...           target vector as trainings data
    y_pred ...      predicted target vector from evaluating function over feature space
    eras ...        timeline in data
    ---------------
    return: array -> pearson correlation array
    """
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct = True, method = "first") )
    return np.corrcoef(y, rank_pred)[0,1]


def correlation_score(y, y_pred):
    """
    params: y, y_pred, eras 
    y ...           target vector as trainings data
    y_pred ...      predicted target vector from evaluating function over feature space
    ---------------
    return: array -> pearson correlation array
    """
    return np.corrcoef(y, y_pred)[0,1]


def get_biggest_change_features(corrs, n):
    """
    params: corrs, n 
    corrs ...       correlation vector
    n ...           amount of riskiest features
    ---------------
    return: array with feature names
    """
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n

def neutralize(df, columns, neutralizers = None, proportion = 1.0, normalize = True, era_col = "era", verbose = False):
    """
    params: df, columns, neutralizers, proportion, normalize, era_col, verbose
    df ...          input df / vector over the features room
    columns ...     array / columns of df
    neutralizers .. ls / features to neutralize
    proportion ...  scalar 
    normalize ...   boolean
    era_col ...     eras
    verbose ...     boolean
    ---------------
    return: new neutralized df
    """
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    if verbose:
        iterator = tqdm(unique_eras)
    else:
        iterator = unique_eras
    for i in iterator:
        df_era = df[df[era_col] == i]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for s in scores.T:
                s = (scipy.stats.rankdata(s, method = "ordinal") - 0.5) / len(s)
                s = scipy.stas.norm.ppf(s)
                scores2.append(s)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(np.linalg.pinv(exposures.astype(np.float32), rcond = 1e-6).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof = 0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed), columns=columns, index = df.index)

def feature_corr(df, era_col, target_col):
    """
    params: df, era_col, target_col 
    era_col ...     era column
    target_col ...  target column
    ---------------
    return: all_feature_corr
    df ...          dataframe with correlation from all features  
    """
    all_feature_corrs = df.groupby(era_col).apply(
    lambda era: era[features].corrwith(era[target_col]))
    return all_feature_corrs


##########################
#saving and loading models

MODEL_FOLDER = "models"

def save_model(model, mtype, params): 
    if mtype == "LGBM":
        try:
            Path(MODEL_FOLDER).mkdir(exist_ok = True, parents = True)
        except Exception as ex:
            pass
        model.booster_.save_model(f"{MODEL_FOLDER}/{mtype}_{params}.json")
    if mtype == "XGB":
        try:
            Path(MODEL_FOLDER).mkdir(exist_ok = True, parents = True)
        except Exception as ex:
            pass
        model.booster_.save_model(f"{MODEL_FOLDER}/{mtype}_{params}.json")
    if mtype == "CatBoost":
        try:
            Path(MODEL_FOLDER).mkdir(exist_ok = True, parents = True)
        except Exception as ex:
            pass
        model.save_model(f"{MODEL_FOLDER}/{mtype}_{params}.json", format = "json")






