import pandas as pd

df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/sub-006/session-04/rawData.h5"
)
# %%
df.columns
# %%
df.saccades[1]
# %%
import matplotlib.pyplot as plt

plt.plot(df.velocity_x.iloc[0])
plt.show()
