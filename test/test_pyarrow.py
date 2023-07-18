import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyarrow.parquet import ParquetFile
import pyarrow as pa

pf = ParquetFile('419_v4_train.parquet')
#first_ten_rows = next(pf.iter_batches(batch_size = 10))
#df = pa.Table.from_batches([first_ten_rows]).to_pandas()

df = '419_v4_train.parquet'

df = pd.read_parquet(df)

indices = list(df.index.values)

df.shape


features = [c for c in df if c.startswith("feature")]
df["erano"] = df.era.astype(int)
eras = df.erano




target = "target"
targets = [c for c in df if c.startswith("target")]


print(df[targets].corr())


#print(features)
#print(df[features])


#y = df.iloc[0][features]
#x = np.arange(0,len(y),1)

#plt.plot(x,y)
#plt.show()

#print(a[features])

