import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import matplotlib.pyplot as plt

n = 10000

pf = ParquetFile('419_v4_train.parquet')
first_ten_rows = next(pf.iter_batches(batch_size = n))
df = pa.Table.from_batches([first_ten_rows]).to_pandas()

#print(df)

features = [f for f in df if f.startswith("feature")]

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

df_st = statistics(df, features)



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

k = 1
x = df[features[k]]

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


mean, var = overall_statistics(df, features)
print(mean, var)

var = np.var(x)
mean = np.mean(x)
print('var : ',var)
print('mean : ', mean)

histogram(x, features[k])
x.plot(kind = 'box')
plt.show()

plot_statistics(df_st,'mean')
plot_statistics(df_st,'variance')

histogram(df_st['feature_mean'], 'mean')
histogram(df_st['feature_variance'], 'variance')



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

#get_similar_value_cols(df)

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
"""
skews = get_skewed_columns(df[features])
first = skews.index[0]
print(first)
x = df[first]
#feature_suppressed_unremovable_telephone

from scipy.stats import shapiro, norm
#my_data = norm.rvs(size=500)
shap = shapiro(x)
print('shap :', shap)

histogram(x, first)
x.plot(kind = 'box')
plt.show()

"""


