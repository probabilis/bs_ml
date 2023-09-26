import pandas as pd
import numpy as np



# Compute the per-era correlation of each feature to the target
per_era_corrs = pd.DataFrame(index=train.era.unique())
for feature_name in feature_subset:
    per_era_corrs[feature_name] = train.groupby("era").apply(lambda d: numerai_corr(d[feature_name], d["target"]))

# Flip the sign of correlations for the features with negative average correlations (since the sign of each feature is arbitrary)
per_era_corrs *= np.sign(per_era_corrs.mean())

# Plot the per-era correlations
per_era_corrs.cumsum().plot(figsize=(15, 5), title="Cumulative sum of correlations of the feature subsets to the target (w/ negative flipped)", legend=False, xlabel="eras ");