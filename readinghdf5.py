import pandas as pd

df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/sub-001/session-01/rawData.h5",
    "data",
)
# %%
df.session
# %%
df.columns
# %%
df.velocity_x[0]
# %%
import matplotlib.pyplot as plt

plt.plot(df.velocity_x[0])
plt.show()
