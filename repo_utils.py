"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy
#############################################
#############################################
#############################################
#initialization

path_ = os.path.join(os.path.expanduser('~'), 'Documents', 'github_repos', "train.parquet")
gh_repos_path = os.path.join(os.path.expanduser('~'), 'Documents', 'github_repos')
repo_path = os.path.join(os.path.expanduser('~'), 'Documents', 'github_repos', "bs_ml")
path_val = os.path.join(os.path.expanduser('~'), 'Documents', 'github_repos', "validation.parquet")
fontsize_title = 16
fontsize = 12

#############################################
#functions and methods

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

##########################

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

##########################

def numerai_corr(preds, target):
    """
    #function from numer.ai
    #######################
    params: preds, target 
    preds ...pd.Series with predictions
    target ...pd.Series with targets
    ---------------
    return: array -> numer.ai corr array
    """
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
    gauss_ranked_preds = scipy.stats.norm.ppf(ranked_preds)
    centered_target = target - target.mean()
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    return np.corrcoef(preds_p15, target_p15)[0, 1]

##########################

def correlation_score(y, y_pred):
    """
    params: y, y_pred, eras 
    y ...           target vector as trainings data
    y_pred ...      predicted target vector from evaluating function over feature space
    ---------------
    return: array -> pearson correlation array
    """
    return np.corrcoef(y, y_pred)[0,1]

##########################

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

##########################

def neutralize_old(df, columns, neutralizers = None, proportion = 1.0, normalize = True, era_col = "era", verbose = False):
    """
    older version until v4.0 datasets / until sept. 2023
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

##########################

def neutralize(predictions: pd.DataFrame, features: pd.DataFrame, proportion: float = 1.0) -> pd.DataFrame:
    """
    newer version from v4.2 datasets / from sept. 2023 / Neutralize predictions to features
    params: df, features, proportion
    df ...          input df / vector over the features room
    features ...     array / columns of df
    proportion ...  scalar 
    ---------------
    return: new neutralized df
    """
    # add a constant term the features so we can fit the bias/offset term
    features = np.hstack((features, np.array([np.mean(predictions)] * len(features)).reshape(-1, 1)))
    print(features)
    # remove the component of the predictions that are linearly correlated with features
    return predictions - proportion * features @ (np.linalg.pinv(features, rcond=1e-6) @ predictions)

##########################

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

def feature_importance(model):

    feature_importance = model.get_feature_importance()

    data = pd.DataFrame({'feature_importance': feature_importance, 
        'feature_names': features_new}).sort_values(by = ['feature_importance'],ascending = False)
    data.to_csv('feature_importance6.csv')
    data[:20].sort_values(by=['feature_importance'], ascending = True).plot.barh(x='feature_names',y='feature_importance')
    plt.show()
    return

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






