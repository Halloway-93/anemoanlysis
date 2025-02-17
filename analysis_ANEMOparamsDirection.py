#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Hamza O.K. El Hallaoui
inspired by:  Vanessa Morita

This script analyses the extracted anemo params
- generates the velocity schema for the experiment design figure
- plots the scatter with errorbars and distributions
- plots the distributions of the pooled data for each variable (violinplots) -> Paper
- exports data for ANOVA
- plots group boxplot (mean for each participant)
- exports data for LMM
"""


# %% bibs
# run always
import os
import numpy as np
import pandas as pd
from functions.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

SMALL_SIZE = 8  # points
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.spines.right"] = False
# plt.rcParams['axes.spines.top']    = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu"]
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

cm = 1 / 2.54  # centimeters in inches
single_col = 8.9 * cm
oneDot5_col = 12.7 * cm
two_col = 18.2 * cm


lmm_dir = "~/anemoanlysis/LMM"


# %% Parameters


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
main_dir = dirVoluntary

os.chdir(main_dir)
subject_sessions = get_subjects_and_sessions(main_dir)

subjects = [f"sub-{str(num).zfill(3)}" for num in subject_sessions.keys()]

# You can also get sessions per subject if needed
sessions_by_subject = {
    f"sub-{str(sub).zfill(3)}": [f"session-{str(sess).zfill(2)}" for sess in sessions]
    for sub, sessions in subject_sessions.items()
}


# colormap = {
#     "Red": np.array([255, 25, 7, 255]) / 255,
#     "Green": np.array([25, 255, 120, 255]) / 255,
# }

# # %% violinplots + export mean Velocities
print("Plotting parameters - violin plots")

# keys2plot: Defines which variables are going to be plotted
# for each row:
# - variable of the first subplot (x-axis)
# - value to scale the kde distribution (for the old plots,not used right now)
# - title of the plot
# - measure unity (to be added to the y-axis label)
# - ylim variable


# reads the data and prepares it for the plot
dataSub = pd.DataFrame()
for sub in subjects:
    try:
        h5_file = os.path.join(main_dir, sub, "smoothPursuitData.h5")

        data_tmp = pd.read_hdf(h5_file, "data")

        print(data_tmp)
        # if the start_anti = latency-1, them there's no anticipation... so changes the value to nan
        data_tmp["aSPon"][data_tmp["aSPoff"] == data_tmp["aSPon"] + 1] = np.nan
        data_tmp["sub"] = int(sub.split("-")[1])

        dataSub = pd.concat([dataSub, data_tmp], ignore_index=True)

    except Exception as e:
        print("Error! \n Couldn't process {}".format(sub))
        # traceback.print_exc()
# %%
dataSub
# %%

allEvents = pd.read_csv(os.path.join(main_dir, "allEvents.csv"))
allEvents
# %%
# To align anemo data that start at trial 0
allEvents["trial"] = allEvents["trial"].values - 1
allEvents.rename(columns={"proba": "cond"}, inplace=True)
# %%
dataSub.columns
# %%
dataSub = dataSub.merge(allEvents, on=["sub", "cond", "trial"], how="inner")
# %%
dataSub
# %%
keys2plot = [
    ["aSPon", 300, "Anticipation Onset", "ms"],
    ["aSPv", 7, "Anticipatory eye velocity", "deg/s", [-20, 20]],
]
# %%
for k in keys2plot:  # for each variable to plot
    data2plot = dataSub  # select the data to plot

    fig1 = plt.figure(figsize=(single_col, 6 * cm))  # width, height in inches
    plt.title("{0}".format(k[2]))
    sns.violinplot(
        x="cond",
        y=k[0],
        hue="chosen_arrow",
        # hue="arrow",
        data=data2plot,
        saturation=1,
        # bw=0.3,
        cut=0,
        # size=3,
        # aspect=1,
        linewidth=1,
        split=True,
        inner="quartile",
    )
    # ax.set_xticklabels(labels=list(condList.values()))
    plt.ylabel("{}".format(k[2]))
    plt.xlabel("P(Right|Up)")
    plt.tight_layout()
    # plt.ylim(k[3])
    plt.savefig("allSubs_mean_{}_violinplot".format(k[0]))  # variable
    # plt.savefig('allSubs_mean_{}_violinplot.pdf'.format(k[2]))
    # plt.savefig('allSubs_mean_{}_violinplot.svg'.format(k[2]))

    fig2 = plt.figure(figsize=(two_col, 15 * cm))
    plt.suptitle("{0}".format(k[2]))
    for sub in subjects:
        plt.subplot(4, 4, int(sub[-2:]))
        sns.violinplot(
            x="cond",
            y=k[0],
            # hue="arrow",  # for imposed condition
            hue="chosen_arrow",
            data=data2plot[data2plot["sub"] == sub],
            saturation=1,
            # bw=0.3,
            cut=0,
            # size=3,
            # aspect=1,
            linewidth=1,
            split=True,
            inner="quartile",
        )

    plt.savefig("allSubs_{}_violinplot".format(k[0]))  # variable
    # plt.savefig('allSubs_{}_violinplot.pdf'.format(k[2]))
    # plt.savefig('allSubs_{}_violinplot.svg'.format(k[2]))

# export data for LMM
if main_dir == dirImposed:
    dataSub.to_csv(f"{lmm_dir}/dataANEMO_allSubs_imposedArrow.csv", index=False)
else:
    dataSub.to_csv(f"{lmm_dir}/dataANEMO_allSubs_voluntaryArrow.csv", index=False)
# %%

# data2plot = dataSub  # select the data to plot
#
# keys2plot = [
#     ["SPlat", 75, "Latency", "ms", [50, 350]],
#     ["SPacc", 70, "Pursuit acceleration", "deg/s^2", [0, 250]],
#     ["SPss", 70, "Steady state", "deg/s", [0, 20]],
# ]
# for k in keys2plot:  # for each variable to plot
#
#     fig1 = plt.figure(figsize=(two_col, 6 * cm))  # width, height in inches
#     plt.suptitle(k[2])
#     plt.subplot(1, 2, 1)
#     plt.title("Leftwards trials")
#     sns.violinplot(
#         x="cond",
#         y=k[0],
#         # hue="chosen_arrow",
#         hue="arrow",
#         hue_order=["down", "up"],
#         data=data2plot[data2plot["target_dir"] == -1],
#         saturation=1,
#         # bw=0.3,
#         cut=0,
#         # size=3,
#         # aspect=1,
#         linewidth=1,
#         split=True,
#         inner="quartile",
#     )
#     # ax.set_xticklabels(labels=list(condList.values()))
#     plt.xlabel("P(Right|Up)")
#     plt.ylim(k[4])
#
#     plt.subplot(1, 2, 2)
#     plt.title("Rightwards trials")
#     sns.violinplot(
#         x="cond",
#         y=k[0],
#         # hue="chosen_arrow",
#         hue="arrow",
#         hue_order=["down", "up"],
#         data=data2plot[data2plot["target_dir"] == 1],
#         saturation=1,
#         # bw=0.3,
#         cut=0,
#         # size=3,
#         # aspect=1,
#         linewidth=1,
#         split=True,
#         inner="quartile",
#     )
#     plt.xlabel("P(Right|Up)")
#     plt.tight_layout()
#     plt.ylim(k[4])
#     plt.savefig(f"allSubs_mean_{k[0]}_leftVsRight_violinplot")  # variable
#
#     plt.show()
#
# # %%
# # Run R code - LMM analysis before running the next cells
#
#
# # %%
# # read LME results from csv gnerated on R
# lme_raneff = pd.read_csv("{}/lmm_randomEffects.csv".format(lmm_dir))
# lme_fixeffAntiVel = pd.read_csv("{}/lmm_fixedEffectsAnti.csv".format(lmm_dir))
#
# lme_fixeffAntiVel.at[0, "Unnamed: 0"] = "Intercept"
#
# lme_raneff.set_index("Unnamed: 0", inplace=True)
# lme_fixeffAntiVel.set_index("Unnamed: 0", inplace=True)
#
# lme_fixeffAnti = lme_fixeffAntiVel
#
# lme_fixeffAnti.fillna(0, inplace=True)
# lme_raneff.fillna(0, inplace=True)
#
# anticipParams = [
#     ["aSPv", "Anticipatory eye velocity", [-1.7, 1.7], "Horizontal aSPv (°/s)"],
# ]
# anticipData = dataSub.groupby(["sub", "cond", "trial_color"]).mean()
# anticipData.reset_index(inplace=True)
#
#
# def listReverse(l):
#     return np.array(list(l).reverse())
#
#
# xAxis = np.array([-0.25, 0, 0.25])
# xAxis = np.array([0.25, 0.5, 0.75])
#
# for p in anticipParams:
#     print("Plotting: {}".format(p[1]))
#
#     intercept = lme_fixeffAnti.loc["Intercept", p[0]]
#     pR_Col = lme_fixeffAnti.loc["prob", p[0]]
#     trialCol = lme_fixeffAnti.loc["trial_colorGreen", p[0]]
#     pR_Col_tCol = lme_fixeffAnti.loc["prob:trial_colorGreen", p[0]]
#
#     fig1 = plt.figure(figsize=(single_col, 7 * cm))  # width, height
#     plt.suptitle(p[1])
#     ax = plt.subplot(1, 1, 1)
#     ax2 = ax.twiny()  # applies twinx to ax2, which is the second y axis.
#     for sub in subjects:
#         raneff = lme_raneff.loc[
#             (lme_raneff["sub"] == sub) & (lme_raneff["var"] == p[0])
#         ]
#         s_pR_Col = raneff["prob"]
#         s_trialCol = raneff["trial_color"]
#         s_intercept = raneff["Intercept"]
#
#         # x = xAxis if p[0]=='velocity_model_x' else abs(xAxis)
#         x = xAxis
#
#         pRRed25_redTg = (
#             (s_pR_Col + pR_Col) * x[0]
#             + (s_trialCol + trialCol) * 0
#             + (pR_Col_tCol) * x[0] * 0
#             + s_intercept
#             + intercept
#         )
#         pRRed50_redTg = (
#             (s_pR_Col + pR_Col) * x[1]
#             + (s_trialCol + trialCol) * 0
#             + (pR_Col_tCol) * x[1] * 0
#             + s_intercept
#             + intercept
#         )
#         pRRed75_redTg = (
#             (s_pR_Col + pR_Col) * x[2]
#             + (s_trialCol + trialCol) * 0
#             + (pR_Col_tCol) * x[2] * 0
#             + s_intercept
#             + intercept
#         )
#
#         pRRed25_greenTg = (
#             (s_pR_Col + pR_Col) * x[0]
#             + (s_trialCol + trialCol) * 1
#             + (pR_Col_tCol) * x[0] * 1
#             + s_intercept
#             + intercept
#         )
#         pRRed50_greenTg = (
#             (s_pR_Col + pR_Col) * x[1]
#             + (s_trialCol + trialCol) * 1
#             + (pR_Col_tCol) * x[1] * 1
#             + s_intercept
#             + intercept
#         )
#         pRRed75_greenTg = (
#             (s_pR_Col + pR_Col) * x[2]
#             + (s_trialCol + trialCol) * 1
#             + (pR_Col_tCol) * x[2] * 1
#             + s_intercept
#             + intercept
#         )
#
#         ax.plot(
#             np.array([0.25, 0.5, 0.75]) + 0.05,
#             [pRRed25_redTg, pRRed50_redTg, pRRed75_redTg],
#             color=colormap["Red"],
#             alpha=0.1,
#         )
#         ax.plot(
#             np.array([0.25, 0.5, 0.75]) - 0.05,
#             [pRRed25_greenTg, pRRed50_greenTg, pRRed75_greenTg],
#             color=colormap["Green"],
#             alpha=0.2,
#         )
#
#     sns.swarmplot(
#         data=anticipData,
#         x="pR-Red",
#         y=p[0],
#         hue="trial_color",
#         palette=colormap,
#         dodge=True,
#         legend=False,
#         size=4,
#         ax=ax2,
#     )
#     sns.boxplot(
#         data=anticipData,
#         x="pR-Red",
#         y=p[0],
#         hue="trial_color",
#         palette=colormap,
#         dodge=True,
#         ax=ax2,
#         width=0.85,
#         showfliers=False,
#         showmeans=False,
#         meanprops=dict(
#             marker="o",
#             markerfacecolor="white",
#             markeredgecolor="none",
#             markersize=3,
#             zorder=3,
#         ),
#         boxprops=dict(facecolor=(0.1, 0.1, 0.1, 0), linewidth=1, zorder=3),
#         whiskerprops=dict(linewidth=1),
#         capprops=dict(linewidth=0),
#         medianprops=dict(linewidth=2, color="gold", zorder=4),
#     )
#     # plt.setp(ax.collections, alpha=.3)
#     # plt.setp(ax2.collections, alpha=.4)
#     plt.legend("", frameon=False)
#     plt.ylim(p[2])
#
#     ax2.set_xticklabels(["0.75", "0.50", "0.25"])
#     ax2.set_xlabel("P(R|Green)")
#
#     ax.set_ylabel(p[3])
#     ax.set_xlim([0.12, 0.88])
#     ax.set_xticks([0.25, 0.5, 0.75])
#     ax.set_xlabel("P(R|Red)")
#
#     plt.tight_layout()
#     plt.savefig("colorCondProba_{}.pdf".format(p[0]))
#     plt.savefig("colorCondProba_{}.png".format(p[0]))
#
#
# # %%
