import pandas as pd

df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/ColorCue/data/sub-01/sub-01_col50-dir50_rawData.h5",
    "data",
)
# %%
df.columns
# %%
df.saccades
# %%
df.velocity_y
# %%
import matplotlib.pyplot as plt

plt.plot(df.velocity_x.iloc[0])
plt.show()
