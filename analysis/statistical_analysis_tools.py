import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def statistics(df, features):
	variances = np.zeros(len(features), dtype = object)
	means = np.zeros(len(features), dtype = object)
	for f,feature in enumerate(features):
		variance = np.var(df[feature])
		mean = np.mean(df[feature])
		variances[f] = variance
		means[f] = mean
	df_statistics = pd.DataFrame({'feature_variance': variances, 'feature_mean':means, 'feature_names': features})
	return df_statistics

def plot_statistics(df,statistic):
	df.sort_values(by = ['feature_' + str(statistic)], ascending = True).plot.barh(y='feature_' + str(statistic))
	plt.xlabel('$\\mu$ / 1')
	plt.yticks(ticks=[])
	plt.ylabel('features')
	plt.title(statistic)
	fig = plt.gcf()
	fig.set_size_inches(12,10)
	plt.show()
	return

def histogram(x, name):
	q25, q75 = np.percentile(x, [25, 75])
	bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
	bins = round((x.max() - x.min()) / bin_width)
	plt.hist(x, density=True, bins = bins)
	plt.title(name)
	fig = plt.gcf()
	fig.set_size_inches(12,10)
	fig.tight_layout()
	plt.show()

def overall_statistics(df, features):
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