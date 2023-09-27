"""
Author: Maximilian Gschaider
MN: 12030366
"""
#official open-source repositories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import date
import seaborn as sns
#own modules
sys.path.append('../')
from repo_utils import repo_path, fontsize, fontsize_title

def statistics(df, features):
	"""
    params: df, features
    df ...          input df / vector over the features room
    features ...	str / features of df
    ---------------
    return: df_statistics
    df ...          dataframe with statistics of   
    """
	variances = np.zeros(len(features), dtype = object)
	means = np.zeros(len(features), dtype = object)
	for f,feature in enumerate(features):
		variance = np.var(df[feature])
		mean = np.mean(df[feature])
		variances[f] = variance
		means[f] = mean
	df_statistics = pd.DataFrame({'feature_variance': variances, 'feature_mean':means, 'feature_names': features})
	return df_statistics

def plot_statistics(df, statistic, name):
	"""
    params: df, statistic
    df ...          input statistics df / vector over the features room
    statistic ...	str / type of statistical test
    ---------------
    return: plot of statistical test
    """
	df.sort_values(by = ['feature_' + str(statistic)], ascending = True).plot.barh(y='feature_' + str(statistic))
	plt.xlabel('$\\mu$ / 1')
	plt.yticks(ticks=[])
	plt.ylabel('features')
	plt.title('numer.ai dataframe v4.2 / features ' + statistic)
	fig = plt.gcf()
	fig.set_size_inches(12,10)
	fig.savefig(repo_path +f"{name}_{statistic}_horizontal_barplot_{date.today()}.png")
	fig.tight_layout()
	plt.show()
	return

def round_int(x):
    if x in [float("-inf"),float("inf")]: return int(10e+6)
    return int(round(x))

def histogram(x, name):
    """
    params: x, name
    x ...       input statistics array / e.g.: vector over one feature room
    name ...	str / name of statistical input variable
    ---------------
    return: plot of histogram diagram
    """
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    print(bin_width)
    if float(bin_width) < 10e-2:
        bin_width = 0.1
    bins = abs(round_int((x.max() - x.min()) / bin_width))
    if bins == 0:
        bins = 1
    print(bins)
    plt.hist(x, density=True, bins = bins)
    plt.title(name)
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    fig.tight_layout()
    fig.savefig(repo_path + f"{name}_histogram_plot_{date.today()}.png")
    plt.show()

def overall_statistics(df, features):
	"""
    params: df, statistic
    df ...          input statistics df / vector over the features room
    features ...	str / features vector
    ---------------
    return: scalar dupel 
    df_mean, df_var
    """
	variances = np.zeros(len(features), dtype = object)
	means = np.zeros(len(features), dtype = object)
	for f, feature in enumerate(features):
		variance = np.var(df[feature])
		mean = np.mean(df[feature])
		variances[f] = variance
		means[f] = mean
	df_mean = np.mean(means)
	df_var = np.var(variances)
	return df_mean, df_var

def get_similar_value_cols(df, percent=70):
    """
    :param df: input data in the form of a dataframe
    :param percent: integer value for the threshold for finding similar values in columns
    :return: sim_val_cols: list of columns where a singular value occurs more than the threshold
    """
    count = 0
    sim_val_cols = []
    for col in df.columns:
        percent_vals = (df[col].value_counts()/len(df)*100).values
        # filter columns where more than 90% values are same and leave out binary encoded columns
        if percent_vals[0] > percent and len(percent_vals) > 2:
            sim_val_cols.append(col)
            count += 1
    print("Total columns with majority singular value shares: ", count)
    return sim_val_cols

def get_skewed_columns(df,skew_limit = 0.2):
    """
    :param df: dataframe where the skewed columns need to determined
    :param skew_limit: scalar which defines a limit above which we will log transform
    :return: skew_cols: dataframe with the skewed columns
    """
    skew_vals = df.skew()
    # Showing the skewed columns
    skew_cols = (skew_vals
                 .sort_values(ascending=False)
                 .to_frame()
                 .rename(columns={0: 'Skew'})
                 .query('abs(Skew) > {}'.format(skew_limit)))
    return skew_cols

def correlation_features(df,features,target):

    eras = df.erano

    correlations = np.zeros((len(eras),len(features)) , dtype = object)

    for e,era in enumerate(eras):
        df_ = df[eras == era]
        for f, feature in enumerate(features):
            if feature in df_:
                corr_ = np.corrcoef(df_[feature],df_[target])
                print('corr. calculated for feature:',feature)
            
                correlations[e][f] = corr_

    return correlations


def plot_correlations(df_correlations, plot_save, name = None) -> None:
    """
    params: df, plot_save, name
    df ...          input correlation df
    plot_save ... boolean   
    name ... string
    ---------------
    return: plot and csv 
    """
    mask = np.triu(np.ones_like(df_correlations, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    #cmap = sns.diverging_palette(230, 20, as_cmap=True)
    #sns.heatmap(df_correlations, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=False, yticklabels=False)
    fig.suptitle(f'{df_correlations} matrix', fontsize=fontsize_title)
    sns.heatmap(df_correlations, cmap="coolwarm", mask = mask, xticklabels=False, yticklabels=False)
    df_correlations.to_csv(repo_path + f"/analysis/{name}_correlations_matrix_{date.today()}.csv")

    if plot_save == True:
        plt.savefig(repo_path + f"/figures/{name}_correlations_matrix_{date.today()}.png", dpi=300)