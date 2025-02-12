"""
Created on Mon Feb 10 2025

@author: Hamza O.K. El Hallaoui

creates and updates files after running preprocessing.py and qualityctrol.py
"""

# %% bibs
# run always
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
from functions.utils import *
from ANEMO.ANEMO import ANEMO, read_edf
from functions.utils import *


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

import warnings

warnings.filterwarnings("ignore")
import traceback


# %% Parameters
# run always
screen_width_px = 1920  # px
screen_height_px = 1080  # px
screen_width_cm = 70  # cm
viewingDistance = 57.0  # cm

tan = np.arctan((screen_width_cm / 2) / viewingDistance)
screen_width_deg = 2.0 * tan * 180 / np.pi
px_per_deg = screen_width_px / screen_width_deg


def get_subjects_and_sessions(base_path):
    """
    Scans directory structure to get ordered subject and session information.

    Args:
        base_path (str): Base directory path containing subject folders

    Returns:
        dict: Dictionary with subjects as keys and list of sessions as values
    """
    base_dir = Path(base_path)
    subject_pattern = re.compile(r"sub-(\d+)")
    session_pattern = re.compile(r"session-(\d+)")

    # Get and sort subjects
    subjects = {}
    for item in base_dir.iterdir():
        if item.is_dir():
            subject_match = subject_pattern.match(item.name)
            if subject_match:
                subject_num = int(subject_match.group(1))
                # Get sessions for this subject
                sessions = []
                for session_dir in item.iterdir():
                    if session_dir.is_dir():
                        session_match = session_pattern.match(session_dir.name)
                        if session_match:
                            sessions.append(int(session_match.group(1)))
                if sessions:
                    subjects[subject_num] = sorted(sessions)

    return dict(sorted(subjects.items()))


# %%

# Example usage:
print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir())

dirVoluntary = (
    "/Users/mango/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/"
)
dirImposed = (
    "/Users/mango/oueld.h/contextuaLearning/directionCue/results_imposeDirection/"
)
main_dir = dirImposed

subject_sessions = get_subjects_and_sessions(main_dir)

# Convert to format matching your original code
subjects = [f"sub-{str(num).zfill(3)}" for num in subject_sessions.keys()]

# You can also get sessions per subject if needed
sessions_by_subject = {
    f"sub-{str(sub).zfill(3)}": [f"session-{str(sess).zfill(2)}" for sess in sessions]
    for sub, sessions in subject_sessions.items()
}

print(subjects)
# %%
print(sessions_by_subject)
# %% Transform raw data
# (also save to the same file, under a different folder...
# if you already ran once and have the sXX_4C_rawData.h5 with a rawFormatted subfolder,
# then you can skip this cell)

keys2save = [
    "sub",
    "session",
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
for idxSub, sub in enumerate(subjects):
    print("Subject:", sub)
    subNumber = sub.split("-")[1]  # Extract subject number

    # Get sessions for this subject
    sessions = sessions_by_subject[sub]

    temp = pd.DataFrame()
    for session in sessions:
        print("Session:", session)
        sessionNumber = session.split("-")[1]  # Extract session number

        # Construct the path to the session directory
        session_dir = os.path.join(main_dir, sub, session)

        print(session_dir)
        try:
            # read data
            h5_rawfile = os.path.join(session_dir, "rawData.h5")
            temp = pd.read_hdf(h5_rawfile, "data")

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
                sessioni = np.array(np.arange(len(row["time_x"]))).astype(object)
                trial = np.array(np.arange(len(row["time_x"])))
                trialTp = np.array(np.arange(len(row["time_x"]))).astype(object)
                tgdir = np.array(np.arange(len(row["time_x"]))).astype(object)
                subj[:] = int(subNumber)
                condi[:] = row["condition"]
                sessioni[:] = row["session"]
                trial[:] = row["trial"]
                tgdir[:] = row["direction"]
                trialTp[:] = row["trialType"]

                newData = np.vstack(
                    (
                        subj,
                        sessioni,
                        condi,
                        trial,
                        trialTp,
                        tgdir,
                        row["time_x"],
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

            print(data)
            data.to_hdf(h5_rawfile, "rawFormatted")
        except Exception as e:
            print("Error! \n Couldn't process {}, condition {}".format(sub, session))
            traceback.print_exc()

# %% read data
# the file for this analysis was generated within the preprocessing script
# run if you don't have the sXX_4C_smoothPursuitData.h5 file

keys = [
    "session",
    "cond",
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

for idxSub, sub in enumerate(subjects):
    print("Subject:", sub)

    # Get sessions for this subject
    sessions = sessions_by_subject[sub]

    tempDF = pd.DataFrame()
    for session in sessions:
        print("Session:", session)

        # Construct the path to the session directory
        session_dir = os.path.join(main_dir, sub, session)

        print(session_dir)
        try:
            # read data
            h5_file = os.path.join(session_dir, "posFilter.h5")
            print(h5_file)
            temp = pd.read_hdf(h5_file, "data")
            tempDF = pd.concat([tempDF, temp], ignore_index=True)
            # if you do a manual quality check, you should exclude the bad trials here

        except Exception as e:
            print("Error! \n Couldn't process {}, condition {}".format(sub, session))
            traceback.print_exc()

    print("\t", tempDF.shape)
    tempDF.dropna(how="all", subset=["t_0", "t_end", "fit_x"], inplace=True)
    try:
        tempDF.drop(["velocity_y", "time"], axis=1, inplace=True)
    except:
        pass

    # transform into a dataframe and save into sXX_4C_smoothPursuitData.h5

    temp = np.empty((len(tempDF), len(keys))).astype(object)

    temp[:, 0] = tempDF["session"]
    temp[:, 1] = tempDF["condition"]
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
    h5_file = os.path.join(main_dir, sub, "smoothPursuitData.h5")
    params.to_hdf(h5_file, "data")

    del tempDF, temp

# %%
