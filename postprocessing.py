#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita

creates and updates files after running preprocessing.py and qualityctrol.py
"""

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

main_dir = "/Users/mango/oueld.h/contextuaLearning/ColorCue/imposedColorData/"
os.chdir(main_dir)


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

import warnings

warnings.filterwarnings("ignore")
import traceback


# %% Parameters
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

main_dir = "/Users/mango/oueld.h/contextuaLearning/ColorCue/imposedColorData/"
os.chdir(main_dir)


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

import warnings

warnings.filterwarnings("ignore")
import traceback

# run always
screen_width_px = 1920  # px
screen_height_px = 1080  # px
screen_width_cm = 70  # cm
viewingDistance = 57.0  # cm

tan = np.arctan((screen_width_cm / 2) / viewingDistance)
screen_width_deg = 2.0 * tan * 180 / np.pi
px_per_deg = screen_width_px / screen_width_deg


subjects = [
    "sub-01",
    "sub-02",
    "sub-03",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-08",
    "sub-10",
    "sub-11",
    # "sub-12",
    # "sub-13",
    # "sub-14",
    # "sub-15",
    # "sub-16",
]

conditions = [
    "col50-dir25",
    "col50-dir50",
    "col50-dir75",
]


# %% Transform raw data
# (also save to the same file, under a different folder...
# if you already ran once and have the sXX_4C_rawData.h5 with a rawFormatted subfolder,
# then you can skip this cell)

keys2save = [
    "subject",
    "condition",
    "trial",
    "trialType",
    "direction",
    "time_x",
    "posDeg_x",
    "velocity_x",
]

float_keys = [
    "posDeg_x",
    "velocity_x",
]
int_keys = ["trial", "time_x"]

data = dict()
for sub in subjects:
    print("Subject:", sub)

    temp = pd.DataFrame()
    for cond in conditions:
        print(cond)
        try:
            # read data
            h5_rawfile = "{sub}/{sub}_{cond}_rawData.h5".format(sub=sub, cond=cond)
            temp = pd.read_hdf(h5_rawfile, "imposedColorData/")

            # get bad data
            # h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(sub=sub, cond=cond)
            # cq        = pd.read_hdf(h5_qcfile, 'data/')
            #
            # # if you do a manual quality check, you should exclude the bad trials here
            # for index, row in cq.iterrows():
            #     if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
            #         temp.drop(temp[temp['trial']==row['trial']].index, inplace=True)

            temp.reset_index(inplace=True)

            # transform data in a new dataframe
            for index, row in temp.iterrows():
                temp.loc[index]["posDeg_x"][
                    temp.loc[index]["posPxl_x"] < screen_width_px * 0.05
                ] = np.nan
                temp.loc[index]["posDeg_x"][
                    temp.loc[index]["posPxl_x"] > screen_width_px * 0.95
                ] = np.nan

                subj = np.array(np.arange(len(row["time_x"]))).astype(object)
                condi = np.array(np.arange(len(row["time_x"]))).astype(object)
                trial = np.array(np.arange(len(row["time_x"])))
                trialTp = np.array(np.arange(len(row["time_x"]))).astype(object)
                tgdir = np.array(np.arange(len(row["time_x"]))).astype(object)
                subj[:] = sub
                condi[:] = row["condition"]
                trial[:] = row["trial"]
                tgdir[:] = row["direction"]
                trialTp[:] = row["trialType"]

                newData = np.vstack(
                    (
                        subj,
                        condi,
                        trial,
                        trialTp,
                        tgdir,
                        row["time_x"],
                        row["posDeg_x"],
                        row["velocity_x"],
                    )
                ).T
                print(data)

                if index == 0:
                    data = pd.DataFrame(newData, columns=keys2save)
                else:
                    data = pd.concat([data,
                        pd.DataFrame(newData, columns=keys2save)], ignore_index=True
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
    "trial",
    "target_dir",
    "trialType",
    # 'trialTgUP',
    "aSPon",
    "aSPv",
    "SPacc",
    "SPss",
    "SPlat",
    "aSPoff",
]

for sub in subjects:
    print("Subject:", sub)

    tempDF = pd.DataFrame()
    for cond in conditions:
        try:
            h5_file = "{sub}/{sub}_{cond}_posFilter.h5".format(sub=sub, cond=cond)
            print(h5_file)
            temp_tmp = pd.read_hdf(h5_file, "imposedColorData/")
            print(temp_tmp)
            tempDF = pd.concat([tempDF, temp_tmp ], ignore_index=True)
            # if you do a manual quality check, you should exclude the bad trials here

        except Exception as e:
            print("Error! \n Couldn't process {}, condition {}".format(sub, cond))
            traceback.print_exc()

    print("\t", tempDF.shape)
    tempDF.dropna(how="all", subset=["t_0", "t_end", "fit_x"], inplace=True)
    try:
        tempDF.drop(["velocity_y", "time"], axis=1, inplace=True)
    except:
        pass

    # transform into a dataframe and save into sXX_4C_smoothPursuitData.h5

    temp = np.empty((len(tempDF), len(keys))).astype(object)

    temp[:, 0] = tempDF["condition"]
    temp[:, 1] = tempDF["trial"]
    temp[:, 2] = tempDF["target_dir"]
    temp[:, 3] = tempDF["trialType"]
    # temp[:,4]  = tempDF['trialTgUP']
    temp[:, 4] = tempDF["aSPon"]
    temp[:, 5] = tempDF["aSPv"]
    temp[:, 6] = tempDF["SPacc"]
    temp[:, 7] = tempDF["SPss"]
    temp[:, 8] = tempDF["SPlat"]
    temp[:, 9] = tempDF["aSPoff"]

    params = []
    params = pd.DataFrame(temp, columns=keys)

    float_keys = ["aSPv", "aSPon", "SPacc", "SPss", "SPlat"]
    params[float_keys] = params[float_keys].astype(float)

    h5_file = "".join([str(sub), "/", str(sub), "_smoothPursuitData.h5"])
    params.to_hdf(h5_file, "imposedColorData")

    del tempDF, temp

# %%
