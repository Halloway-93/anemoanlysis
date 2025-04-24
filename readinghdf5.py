import pandas as pd
import h5py

import matplotlib.pyplot as plt

# %%
# df.session
df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue/sub-001/CP_s1c3_rawData.h5",
    "data",
)
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
file = "/Volumes/hpc_home/oueld.h/MD/s10_CP_rawData.h5"
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
df[df.new_cond.isin([4,5,6])][ ["new_cond",'condition'] ].value_counts()
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
conds=[i for i in range(1,7)]
# %%
subs=[s for s in range(1,12)]
conds=[i for i in range(1,7)]
# %%
for s in subs:
    for i in conds:
        print(f"Subj {s}, condition {i}:")
        if s>9:
            filPath = f"/Users/mango/oueld.h/contextuaLearning/motionDirectionCue/sub-0{s}/sub-0{s}_c{i}.tsv"
        else:
            filPath = f"/Users/mango/oueld.h/contextuaLearning/motionDirectionCue/sub-00{s}/sub-00{s}_c{i}.tsv"

        mat = pd.read_csv(filPath, sep="\t")
        print( mat[mat.firstSegmentMotion==1]["firstSegmentMotion"].value_counts())
            
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
