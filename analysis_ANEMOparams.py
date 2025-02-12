#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita

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
import sys
import h5py
import time as timer
import numpy as np
import pandas as pd
from functions.utils import *
from ANEMO.ANEMO import ANEMO, read_edf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

main_dir = "/Users/mango/oueld.h/contextuaLearning/ColorCue/imposedColorData/"
main_dir = "/Users/mango/oueld.h/attentionalTask/data/"
os.chdir(main_dir)  # pc lab

# %matplotlib auto
# %matplotlib inline
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
    "sub-12",
    "sub-13",
    # "sub-14",
    # "sub-15",
    # "sub-16",
]

conditions = [
    "col50-dir25",
    "col50-dir50",
    "col50-dir75",
]


colormap = {
    "Red": np.array([255, 25, 7, 255]) / 255,
    "Green": np.array([25, 255, 120, 255]) / 255,
}

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
        h5_file = "{sub}/{sub}_smoothPursuitData.h5".format(sub=sub)
        data_tmp = pd.read_hdf(h5_file, "data")

        # if the start_anti = latency-1, them there's no anticipation... so changes the value to nan
        data_tmp["aSPon"][data_tmp["aSPoff"] == data_tmp["aSPon"] + 1] = np.nan
        data_tmp["trial_color"] = [x[:-1] for x in data_tmp["trialType"]]
        data_tmp["sub"] = sub

        dataSub = pd.concat([dataSub, data_tmp], ignore_index=True)

    except Exception as e:
        print("Error! \n Couldn't process {}".format(sub))
        # traceback.print_exc()
# %%

dataSub["pR-Red"] = [int(x.split("-")[1][-2:]) / 100 for x in dataSub["cond"]]

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
        x="pR-Red",
        y=k[0],
        hue="trial_color",
        data=data2plot,
        palette=colormap,
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
    plt.xlabel("P(Right|Red)")
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
            x="pR-Red",
            y=k[0],
            hue="trial_color",
            data=data2plot[data2plot["sub"] == sub],
            palette=colormap,
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
dataSub.to_csv(f"{lmm_dir}/dataANEMO_allSubs_attentionColorCP.csv")

# %%

data2plot = dataSub  # select the data to plot

keys2plot = [
    ["SPlat", 75, "Latency", "ms", [50, 350]],
    ["SPacc", 70, "Pursuit acceleration", "deg/s^2", [0, 250]],
    ["SPss", 70, "Steady state", "deg/s", [0, 20]],
]
for k in keys2plot:  # for each variable to plot

    fig1 = plt.figure(figsize=(two_col, 6 * cm))  # width, height in inches
    plt.suptitle(k[2])
    plt.subplot(1, 2, 1)
    plt.title("Leftwards trials")
    sns.violinplot(
        x="pR-Red",
        y=k[0],
        hue="trial_color",
        hue_order=["Green", "Red"],
        data=data2plot[data2plot["target_dir"] == -1],
        palette=colormap,
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
    plt.xlabel("P(Right|Red)")
    plt.ylim(k[4])

    plt.subplot(1, 2, 2)
    plt.title("Rightwards trials")
    sns.violinplot(
        x="pR-Red",
        y=k[0],
        hue="trial_color",
        hue_order=["Green", "Red"],
        data=data2plot[data2plot["target_dir"] == 1],
        palette=colormap,
        saturation=1,
        # bw=0.3,
        cut=0,
        # size=3,
        # aspect=1,
        linewidth=1,
        split=True,
        inner="quartile",
    )
    plt.xlabel("P(Right|Red)")
    plt.tight_layout()
    plt.ylim(k[4])
    plt.savefig(f"allSubs_mean_{k[0]}_leftVsRight_violinplot")  # variable

    plt.show()

# %%
# Run R code - LMM analysis before running the next cells


# %%
# read LME results from csv gnerated on R
lme_raneff = pd.read_csv("{}/lmm_randomEffects.csv".format(lmm_dir))
lme_fixeffAntiVel = pd.read_csv("{}/lmm_fixedEffectsAnti.csv".format(lmm_dir))

lme_fixeffAntiVel.at[0, "Unnamed: 0"] = "Intercept"

lme_raneff.set_index("Unnamed: 0", inplace=True)
lme_fixeffAntiVel.set_index("Unnamed: 0", inplace=True)

lme_fixeffAnti = lme_fixeffAntiVel

lme_fixeffAnti.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)

anticipParams = [
    ["aSPv", "Anticipatory eye velocity", [-1.7, 1.7], "Horizontal aSPv (°/s)"],
]
anticipData = dataSub.groupby(["sub", "cond", "trial_color"]).mean()
anticipData.reset_index(inplace=True)


def listReverse(l):
    return np.array(list(l).reverse())


xAxis = np.array([-0.25, 0, 0.25])
xAxis = np.array([0.25, 0.5, 0.75])

for p in anticipParams:
    print("Plotting: {}".format(p[1]))

    intercept = lme_fixeffAnti.loc["Intercept", p[0]]
    pR_Col = lme_fixeffAnti.loc["prob", p[0]]
    trialCol = lme_fixeffAnti.loc["trial_colorGreen", p[0]]
    pR_Col_tCol = lme_fixeffAnti.loc["prob:trial_colorGreen", p[0]]

    fig1 = plt.figure(figsize=(single_col, 7 * cm))  # width, height
    plt.suptitle(p[1])
    ax = plt.subplot(1, 1, 1)
    ax2 = ax.twiny()  # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff = lme_raneff.loc[
            (lme_raneff["sub"] == sub) & (lme_raneff["var"] == p[0])
        ]
        s_pR_Col = raneff["prob"]
        s_trialCol = raneff["trial_color"]
        s_intercept = raneff["Intercept"]

        # x = xAxis if p[0]=='velocity_model_x' else abs(xAxis)
        x = xAxis

        pRRed25_redTg = (
            (s_pR_Col + pR_Col) * x[0]
            + (s_trialCol + trialCol) * 0
            + (pR_Col_tCol) * x[0] * 0
            + s_intercept
            + intercept
        )
        pRRed50_redTg = (
            (s_pR_Col + pR_Col) * x[1]
            + (s_trialCol + trialCol) * 0
            + (pR_Col_tCol) * x[1] * 0
            + s_intercept
            + intercept
        )
        pRRed75_redTg = (
            (s_pR_Col + pR_Col) * x[2]
            + (s_trialCol + trialCol) * 0
            + (pR_Col_tCol) * x[2] * 0
            + s_intercept
            + intercept
        )

        pRRed25_greenTg = (
            (s_pR_Col + pR_Col) * x[0]
            + (s_trialCol + trialCol) * 1
            + (pR_Col_tCol) * x[0] * 1
            + s_intercept
            + intercept
        )
        pRRed50_greenTg = (
            (s_pR_Col + pR_Col) * x[1]
            + (s_trialCol + trialCol) * 1
            + (pR_Col_tCol) * x[1] * 1
            + s_intercept
            + intercept
        )
        pRRed75_greenTg = (
            (s_pR_Col + pR_Col) * x[2]
            + (s_trialCol + trialCol) * 1
            + (pR_Col_tCol) * x[2] * 1
            + s_intercept
            + intercept
        )

        ax.plot(
            np.array([0.25, 0.5, 0.75]) + 0.05,
            [pRRed25_redTg, pRRed50_redTg, pRRed75_redTg],
            color=colormap["Red"],
            alpha=0.1,
        )
        ax.plot(
            np.array([0.25, 0.5, 0.75]) - 0.05,
            [pRRed25_greenTg, pRRed50_greenTg, pRRed75_greenTg],
            color=colormap["Green"],
            alpha=0.2,
        )

    sns.swarmplot(
        data=anticipData,
        x="pR-Red",
        y=p[0],
        hue="trial_color",
        palette=colormap,
        dodge=True,
        legend=False,
        size=4,
        ax=ax2,
    )
    sns.boxplot(
        data=anticipData,
        x="pR-Red",
        y=p[0],
        hue="trial_color",
        palette=colormap,
        dodge=True,
        ax=ax2,
        width=0.85,
        showfliers=False,
        showmeans=False,
        meanprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="none",
            markersize=3,
            zorder=3,
        ),
        boxprops=dict(facecolor=(0.1, 0.1, 0.1, 0), linewidth=1, zorder=3),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=0),
        medianprops=dict(linewidth=2, color="gold", zorder=4),
    )
    # plt.setp(ax.collections, alpha=.3)
    # plt.setp(ax2.collections, alpha=.4)
    plt.legend("", frameon=False)
    plt.ylim(p[2])

    ax2.set_xticklabels(["0.75", "0.50", "0.25"])
    ax2.set_xlabel("P(R|Green)")

    ax.set_ylabel(p[3])
    ax.set_xlim([0.12, 0.88])
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_xlabel("P(R|Red)")

    plt.tight_layout()
    plt.savefig("colorCondProba_{}.pdf".format(p[0]))
    plt.savefig("colorCondProba_{}.png".format(p[0]))


# %%
