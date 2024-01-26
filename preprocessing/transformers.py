"""
Author: Maximilian Gschaider
MN: 12030366
"""
import numpy as np

########################################

def era_splitting(df, window = 4):
    """
    params: df
    window ...       overlapping window length
    ---------------
    return: reduced df with splitted eras
    """
    return df[df["era"].isin(df["era"].unique()[::window])]
