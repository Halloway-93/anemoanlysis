import pandas as pd
import h5py

import scipy
import matplotlib.pyplot as plt

df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue/sub-001/CP_s1c3_rawData.h5",
    "data",
)
# %%
# df.session
# %%
df.columns
# %%
df
# %%
df.velocity_x[0]
# %%

plt.plot(df.time[0], df.velocity_x[0])
plt.plot(df.time[0], df.velocity_y[0])
plt.show()
# %%
file = "/Users/mango/Downloads/s02_CP_rawData.h5"
# Open the HDF5 file
with h5py.File(file, "r") as f:
    # Get keys at the root level
    root_keys = list(f.keys())
    print("Root level keys:", root_keys)

    # To get all keys recursively (including nested groups)
    def get_all_keys(name):
        print(name)

    # Visit every node in the file hierarchy
    f.visit(get_all_keys)

# %%
df = pd.read_hdf(file, "rawFormatted")
df.columns
# %%
df["new_cond"].unique()
# %%
df
# %%
dd = (
    df.groupby(["subject", "condition", "trial", "target_dir"])["new_cond"]
    .mean()
    .reset_index()
)
dd
# %%
dd[dd["condition"] == 1]["target_dir"].value_counts()
# %%
filPath = (
    "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue/sub-004/sub-004_c2.tsv"
)

# %%
mat = pd.read_csv(filPath, sep="\t")
# %%
mat["firstSegmentMotion"].value_counts()
# %%
cond = filPath.split(
    ".mat",
)[
    0
][-2:]
# %%
mat
# %%
(mat["trialType"])[:, 1]
# %%
with h5py.File(filPath, "r") as f:
    data = f.read()

# %%
import numpy as np

with h5py.File(filPath, "r") as f:
    # H5py loads MATLAB variables differently
    # We need to get and transpose the dataset

    firstSeg = np.array(f.get("listFirstSeg"))[0]
    secondSeg = np.array(f.get("listSecondSeg"))[0]

    # Convert to numpy array and transpose (MATLAB stores arrays column-wise)
    data = np.array([firstSeg, secondSeg]).T
    df = pd.DataFrame(
        data,
        columns=["firstSegmentMotion", "secondSegmentMotion"],
    )
    print(df)
