# %% bibs
# run always
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from functions.utils import *
from ANEMO.ANEMO import ANEMO, read_edf
from functions.utils import *

import warnings

warnings.filterwarnings("ignore")
import traceback

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

import warnings

warnings.filterwarnings("ignore")
import traceback

# %% Parameters
# run always
main_dir = "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue"
os.chdir(main_dir)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


# run always
screen_width_px = 1920  # px
screen_height_px = 1080  # px
screen_width_cm = 70  # cm
viewingDistance = 57.0  # cm

tan = np.arctan((screen_width_cm / 2) / viewingDistance)
screen_width_deg = 2.0 * tan * 180 / np.pi
px_per_deg = screen_width_px / screen_width_deg


subjects = [
    "sub-001",
    "sub-002",
    "sub-003",
    "sub-004",
    "sub-005",
    "sub-006",
    "sub-007",
    "sub-008",
    "sub-009",
    "sub-010",
    "sub-011",
]

conditions = [
    "c1",
    "c2",
    "c3",
]


# %% Transform raw data
# (also save to the same file, under a different folder...
# if you already ran once and have the sXX_4C_rawData.h5 with a rawFormatted subfolder,
# then you can skip this cell)

keys2save = [
    "subject",
    "condition",
    "proba",
    "trial",
    "trialType",
    "direction",
    "time",
    "posDeg_x",
    "velocity_x",
    # "posDeg_y",
    # "velocity_y",
]

float_keys = [
    "posDeg_x",
    "velocity_x",
    # "posDeg_y",
    # "velocity_y",
]
int_keys = ["trial", "time"]

data = dict()
for sub in subjects:
    print("Subject:", sub)

    subNumber = int(sub.split("-")[1])
    temp = pd.DataFrame()
    for cond in conditions:
        print(cond)
        try:
            # read data
            h5_rawfile = "{sub}/CP_s{subNumber}{cond}_rawData.h5".format(
                sub=sub, subNumber=subNumber, cond=cond
            )
            temp = pd.read_hdf(h5_rawfile, "data")
            temp.reset_index(inplace=True)

            # transform data in a new dataframe
            for index, row in temp.iterrows():
                temp.loc[index]["posDeg_x"][
                    temp.loc[index]["posPxl_x"] < screen_width_px * 0.05
                ] = np.nan
                temp.loc[index]["posDeg_x"][
                    temp.loc[index]["posPxl_x"] > screen_width_px * 0.95
                ] = np.nan

                subj = np.array(np.arange(len(row["time"]))).astype(object)
                condi = np.array(np.arange(len(row["time"]))).astype(object)
                probi = np.array(np.arange(len(row["time"]))).astype(object)
                trial = np.array(np.arange(len(row["time"])))
                trialTp = np.array(np.arange(len(row["time"]))).astype(object)
                tgdir = np.array(np.arange(len(row["time"]))).astype(object)
                subj[:] = sub
                condi[:] = row["condition"]
                probi[:] = row["proba"]
                trial[:] = row["trial"]
                tgdir[:] = row["direction"]
                trialTp[:] = row["trialType"]

                newData = np.vstack(
                    (
                        subj,
                        condi,
                        probi,
                        trial,
                        trialTp,
                        tgdir,
                        row["time"],
                        row["posDeg_x"],
                        row["velocity_x"],
                    )
                ).T

                if index == 0:
                    data = pd.DataFrame(newData, columns=keys2save)
                else:
                    data = pd.concat(
                        [data, pd.DataFrame(newData, columns=keys2save)],
                        ignore_index=True,
                    )

            # cast data to correct format
            data[float_keys] = data[float_keys].astype(float)
            data[int_keys] = data[int_keys].astype(int)

            data.to_hdf(h5_rawfile, "rawFormatted")
        except Exception as e:
            print("Error! \n Couldn't process {}, condition {}".format(sub, cond))
            traceback.print_exc()

# %% read data
# the file for this analysis was generated within the preprocessing script
# run if you don't have the sXX_4C_smoothPursuitData.h5 file

keys = [
    "cond",
    "proba",
    "trial",
    "target_dir",
    "trialType",
    "aSPon",
    "aSPv",
    "SPacc",
    "SPss",
    "SPlat",
    "aSPoff",
]

for sub in subjects:
    print("Subject:", sub)

    subNumber = int(sub.split("-")[1])
    tempDF = pd.DataFrame()
    for cond in conditions:
        try:
            h5_file = "{sub}/CP_s{subNumber}{cond}_posFilter.h5".format(
                sub=sub, subNumber=subNumber, cond=cond
            )
            print(h5_file)
            temp_tmp = pd.read_hdf(h5_file, "data")
            print(temp_tmp)
            tempDF = pd.concat([tempDF, temp_tmp], ignore_index=True)
            # if you do a manual quality check, you should exclude the bad trials here

        except Exception as e:
            print("Error! \n Couldn't process {}, condition {}".format(sub, cond))
            traceback.print_exc()

    # print("\t", tempDF.shape)
    tempDF.dropna(how="all", subset=["t_0", "t_end", "fit_x"], inplace=True)
    try:
        tempDF.drop(["velocity_y", "time"], axis=1, inplace=True)
    except:
        pass

    # transform into a dataframe and save into sXX_4C_smoothPursuitData.h5

    temp = np.empty((len(tempDF), len(keys))).astype(object)

    temp[:, 0] = tempDF["condition"]
    temp[:, 1] = tempDF["proba"]
    temp[:, 2] = tempDF["trial"]
    temp[:, 3] = tempDF["target_dir"]
    temp[:, 4] = tempDF["trialType"]
    temp[:, 5] = tempDF["aSPon"]
    temp[:, 6] = tempDF["aSPv"]
    temp[:, 7] = tempDF["SPacc"]
    temp[:, 8] = tempDF["SPss"]
    temp[:, 9] = tempDF["SPlat"]
    temp[:, 10] = tempDF["aSPoff"]

    params = []
    params = pd.DataFrame(temp, columns=keys)

    float_keys = ["aSPv", "aSPon", "SPacc", "SPss", "SPlat"]
    params[float_keys] = params[float_keys].astype(float)

    h5_file = "".join([str(sub), "/", str(sub), "_smoothPursuitData.h5"])
    params.to_hdf(h5_file, "data")

    del tempDF, temp

# %%
