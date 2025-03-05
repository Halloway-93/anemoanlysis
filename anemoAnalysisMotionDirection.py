from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from matplotlib.patches import Patch
import os

# %%
main_dir = "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue"
pathFig = "/Users/mango/Contextual-Learning/motionDirection/figures/"
df = pd.read_csv(
    "/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_motionDirectionCP.csv"
)
print(df)
# %%
allEvents = pd.read_csv(main_dir + "/allEvents.csv")
allEvents["proba"].unique()
allEvents.dropna(subset=["proba"], inplace=True)
# %%
allEvents["proba"].unique()
# %%
allEvents["trial"] = allEvents["trial"].values - 1
# %%
allEvents["firstSegmentMotion"] = allEvents["firstSegmentMotion"].apply(
    lambda x: "up" if x == 1 else "down"
)
# %%
print(df["sub"].unique())
# %%
df["aSPoff"]
# %%
sns.histplot(data=df, x="aSPv")
plt.show()
# %%
sns.histplot(data=df, x="SPlat")
plt.show()
# %%
sns.histplot(data=df, x="aSPoff")
plt.show()
# %%
sns.histplot(data=df, x="aSPon")
plt.show()
# %%
for sub in df["sub"].unique():
    for p in df[df["sub"] == sub]["proba"].unique():
        for t in df[(df["sub"] == sub) & (df["proba"] == p)]["trial"].unique():

            # Check if the previous trial exists in allEvents
            prev_trial = allEvents.loc[
                (allEvents["sub"] == sub)
                & (allEvents["proba"] == p)
                & (allEvents["trial"] == t - 1)
            ]

            if not prev_trial.empty:  # Ensure the previous trial exists
                # Assign trial direction from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "TD_prev",
                ] = prev_trial["secondSegmentMotion"].values[0]

                # print(prev_trial["trial_color_chosen"].values[0])
                # Assign color from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "firstSeg_prev",
                ] = prev_trial["firstSegmentMotion"].values[0]
# %%
df[df["TD_prev"].isna()]
# %%
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
df = df[~(df["TD_prev"].isna())]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
df["interaction"] = list(zip(df["TD_prev"], df["firstSeg_prev"]))
# %%
df[df["aSPv"] == df["aSPv"].max()]["aSPv"]
# %%
sns.histplot(data=df, x="aSPv")
plt.show()
# %%
badTrials = df[df["aSPv"] > 8]["aSPv"]
print(badTrials)
# %%
balance = df.groupby(["firstSeg", "sub", "proba"])["trial"].count().reset_index()
print(balance)
# %%
df = df[df["sub"] != "sub-011"]
# %%
for sub in balance["sub"].unique():
    sns.barplot(
        x="proba", y="trial", hue="firstSeg", data=balance[balance["sub"] == sub]
    )
    plt.title(f"Subject {sub}")
    plt.show()
# %%
dd = df.groupby(["sub", "firstSeg", "proba"])[["aSPv"]].mean().reset_index()
# %%

np.abs(dd.aSPv.values).max()
# %%

aSPv = dd[dd["firstSeg"] == "up"]["aSPv"]
proba = dd[dd["firstSeg"] == "up"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (up): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.firstSeg == "down"]["aSPv"]
proba = dd[dd.firstSeg == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (down): {correlation}, p-value: {p_value}")

# %%
aSPv = dd[dd.proba == 0.75]["aSPv"]
firstSeg = dd[dd.proba == 0.75]["firstSeg"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, firstSeg)
print(f"Spearman's correlation(Proba 0.75): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd.proba == 0.25]["aSPv"]
firstSeg = dd[dd.proba == 0.25]["firstSeg"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, firstSeg)
print(f"Spearman's correlation(Proba 0.25): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd.proba == 0.5]["aSPv"]
firstSeg = dd[dd.proba == 0.5]["firstSeg"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, firstSeg)
print(f"Spearman's correlation(Proba 0.5): {correlation}, p-value: {p_value}")


# %%
# cehcking the normality of the data
print(pg.normality(dd[dd.proba == 0.5]["aSPv"]))
# %%
stat, p = stats.kstest(
    dd["aSPv"], "norm", args=(dd["aSPv"].mean(), dd["aSPv"].std(ddof=1))
)
print(f"Statistic: {stat}, p-value: {p}")
# %%
x = dd["aSPv"]
ax = pg.qqplot(x, dist="norm")
plt.show()


# Set up the FacetGrid
facet_grid = sns.FacetGrid(data=df, col="proba", col_wrap=3, height=8, aspect=1.5)

facet_grid.add_legend()
# Create pointplots for each sub
facet_grid.map_dataframe(
    sns.histplot,
    x="aSPv",
    hue="firstSeg",
    hue_order=["Down", "Up"],
    alpha=0.3,
)
# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, np.sort(df.proba.unique())):
    ax.set_title(f"ASEM: P(Right|up)=P(Left|down)={p}")
    ax.legend(["up", "down"])
# Adjust spacing between subplots
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df1,
    x="proba",
    y="aSPv",
    capsize=0.1,
    n_boot=10000,
    errorbar="ci",
    hue="firstSeg",
    hue_order=["Down", "Up"],
)
plt.legend(title="firstSeg", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/asemAcrossProbappFullProba.svg", transparent=True)
plt.show()
# %%
sns.catplot(
    data=dd,
    x="proba",
    y="aSPv",
    hue="firstSeg",
    hue_order=["Down", "Up"],
    kind="violin",
    split=True,
    fill=False,
    gap=0.1,
    # inner="stick",
    height=10,
    cut=0,
)
# _ = plt.title("ASEM Across Probabilities", fontsize=30)
plt.legend(title="firstSeg", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/asemAcrossprobaviolinFullProba.svg", transparent=True)
plt.show()

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.firstSeg == "Up"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: firstSeg UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.svg", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=dd[dd.firstSeg == "Up"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("ASEM Per Subject: firstSeg UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.svg", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=dd[dd.firstSeg == "Down"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("ASEM Per Subject: firstSeg UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/individualsUPFullProba.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.firstSeg == "Down"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: firstSeg DOWN", fontsize=30)
plt.legend(fontsize=10)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsDOWNFullProba.svg", transparent=True)
plt.show()
# %%
model = smf.mixedlm(
    "aSPv~C( firstSeg )*proba",
    data=df,
    re_formula="~proba",
    groups=df["sub"],
).fit()
model.summary()
# %%
# %%
model = smf.mixedlm(
    "aSPv~C( firstSeg )",
    data=df[df.proba == 0.75],
    re_formula="~firstSeg",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=["lbfgs"])
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C( firstSeg )",
    data=df[df.proba == 0.25],
    re_formula="~firstSeg",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~firstSeg",
    data=df[df.proba == 0.5],
    re_formula="~firstSeg",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "aSPv~ proba",
    data=df[df.firstSeg == "Up"],
    re_formula="~proba",
    groups=df[df.firstSeg == "Up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~ proba",
    data=df[df.firstSeg == "Down"],
    re_formula="~proba",
    groups=df[df.firstSeg == "Down"]["sub"],
).fit()
model.summary()


# %%
downfirstSegsPalette = ["#0F68A9", "#A2D9FF"]
upfirstSegsPalette = ["#FAAE7B", "#FFD699"]
dd = df.groupby(["sub", "firstSeg", "proba"])[["aSPv"]].mean().reset_index()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="firstSeg",
    hue_order=["Down", "Up"],
    data=df,
    errorbar="ci",
    fill=False,
    palette=[downfirstSegsPalette[0], upfirstSegsPalette[0]],
)
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="firstSeg",
    data=dd,
    dodge=True,
    palette=[downfirstSegsPalette[0], upfirstSegsPalette[0]],
    jitter=True,
    size=6,
    linewidth=1,
    # alpha=0.5,
    legend=False,
)
# plt.title("ASEM Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(title="firstSeg", title_fontsize=20, fontsize=20)
plt.savefig(pathFig + "/aSPvfirstSegsFullProba.svg", transparent=True)
plt.show()
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "firstSeg",
        "target_dir",
        "TD_prev",
        "aSPv",
    ]
]
# %%
df["TD_prev"].unique()
# %%
df["firstSeg_prev"].unique()
# %%
learningCurve = (
    df_prime.groupby(["sub", "proba", "firstSeg", "TD_prev"])
    .mean()[["aSPv"]]
    .reset_index()
)


print(learningCurve)
# %%
df_prime.groupby(["proba", "firstSeg", "TD_prev"]).count()[["aSPv"]]

# %%
dd = df.groupby(["sub", "proba", "firstSeg", "TD_prev"])["aSPv"].mean().reset_index()

# %%
# Create the plot using catplot
g = sns.catplot(
    data=df[df.firstSeg == "Down"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[downfirstSegsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(downfirstSegsPalette[0])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[downfirstSegsPalette[0]],
    dodge=True,
    jitter=True,
    size=6,
    linewidth=1,
    # alpha=0.7,
    data=dd[dd.firstSeg == "Down"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=downfirstSegsPalette[0], alpha=1, label="Left"),
    Patch(
        facecolor="none", hatch="///", label="Right", edgecolor=downfirstSegsPalette[0]
    ),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize=20
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: firstSeg Down", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=25)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvdownTDFullProba.svg", transparent=True)
plt.show()

# %%
# Create the plot using catplot
g = sns.catplot(
    data=df[df.firstSeg == "Up"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[upfirstSegsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(upfirstSegsPalette[0])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[upfirstSegsPalette[0]],
    dodge=True,
    jitter=True,
    linewidth=1,
    size=6,
    # alpha=0.7,
    data=dd[dd.firstSeg == "Up"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=upfirstSegsPalette[0], alpha=1, label="Left"),
    Patch(
        facecolor="none", hatch="///", label="Right", edgecolor=upfirstSegsPalette[0]
    ),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize=20
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: firstSeg Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=25)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTDFullProba.svg", transparent=True)
plt.show()
# %%
df["interaction"] = list(zip(df["TD_prev"], df["firstSeg_prev"]))
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "firstSeg",
        "interaction",
        "aSPv",
    ]
]
# %%
learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "firstSeg"])
    .mean()[["aSPv"]]
    .reset_index()
)

# %%
df_prime.groupby(["sub", "proba", "interaction", "firstSeg"]).count()[["aSPv"]]

# %%
df_prime.groupby(["proba", "interaction", "firstSeg"]).count()[["aSPv"]]

# %%
print(learningCurveInteraction)
# %%
learningCurveInteraction[
    learningCurveInteraction["aSPv"] == learningCurveInteraction["aSPv"].max()
]
# %%
df["interaction"].unique()
# %%
hue_order = [
    ("left", "down"),
    ("left", "up"),
    ("right", "down"),
    ("right", "up"),
]
colorsPalettes = ["#0F68A9", "#FAAE7B", "#0F68A9", "#FAAE7B"]
# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.firstSeg == "Up"],
    x="proba",
    y="aSPv",
    hue="interaction",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=colorsPalettes,
    height=10,
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=hue_order,
    legend=False,
)

# Determine the number of bars per x-value
n_categories = len(df_prime["interaction"].unique())
n_x_values = len(df_prime["proba"].unique())
total_bars = n_categories * n_x_values
# Add hatching to the bars for 'right' categories
for i, bar in enumerate(g.ax.patches):
    # Determine if this bar represents a 'right' category
    if i > 5:  # Second half of bars for each x-value

        bar.set_facecolor("none")  # Make bar empty
        bar.set_hatch("///")  # Add diagonal lines
        bar.set_edgecolor(colorsPalettes[i // n_x_values])

# Add stripplot
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    hue_order=hue_order,
    palette=colorsPalettes,
    dodge=True,
    jitter=True,
    marker="o",
    linewidth=1,
    size=6,
    data=learningCurveInteraction[learningCurveInteraction.firstSeg == "Up"],
    # legend=False,
)
#
# # Create custom legend with all four categories
legend_elements = [
    # Left categories (solid fill)
    Patch(facecolor=colorsPalettes[0], alpha=1, label="Left, Down"),
    Patch(facecolor=colorsPalettes[1], alpha=1, label="Left, Up"),
    # Right categories (hatched)
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalettes[0],
        label="Right, Down",
    ),
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalettes[1],
        label="Right, Up",
    ),
]

# Add the legend
g.ax.legend(
    handles=legend_elements, fontsize=15, title="Previous Trial", title_fontsize=15
)

# Customize the plot
g.ax.set_title("Up Trials:\n Previous TD and its firstSeg", fontsize=30)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvUpInteractionFullProba.svg", transparent=True)
plt.show()

# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.firstSeg == "Down"],
    x="proba",
    y="aSPv",
    hue="interaction",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=colorsPalettes,
    height=10,
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=hue_order,
    legend=False,
)

# Determine the number of bars per x-value
n_categories = len(df_prime["interaction"].unique())
n_x_values = len(df_prime["proba"].unique())
total_bars = n_categories * n_x_values
print(total_bars)
# Add hatching to the bars for 'right' categories
for i, bar in enumerate(g.ax.patches):
    # Determine if this bar represents a 'right' category
    if i > 5:  # Second half of bars for each x-value

        bar.set_facecolor("none")  # Make bar empty
        bar.set_hatch("///")  # Add diagonal lines
        bar.set_edgecolor(colorsPalettes[i // n_x_values])


# Add stripplot
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    hue_order=hue_order,
    palette=colorsPalettes,
    dodge=True,
    jitter=True,
    marker="o",
    linewidth=1,
    size=6,
    data=learningCurveInteraction[learningCurveInteraction.firstSeg == "Down"],
    # legend=False,
)
#
# # Create custom legend with all four categories
legend_elements = [
    # Left categories (solid fill)
    Patch(facecolor=colorsPalettes[0], alpha=1, label="Left, Down"),
    Patch(facecolor=colorsPalettes[1], alpha=1, label="Left, Up"),
    # Right categories (hatched)
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalettes[0],
        label="Right, Down",
    ),
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalettes[1],
        label="Right, Up",
    ),
]

# Add the legend
g.ax.legend(
    handles=legend_elements, fontsize=15, title="Previous Trial", title_fontsize=15
)

# Customize the plot
g.ax.set_title("Down Trials:\n Previous TD and its firstSeg", fontsize=30)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvDownInteractionFullProba.svg", transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "firstSeg", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(firstSeg)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(firstSeg)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(firstSeg)*C(TD_prev)",
    data=df[df.proba == 0.5],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.5]["sub"],
).fit(method="lbfgs")
model.summary()
# %%

# Define transition counts for previous state = down
down_transitions = (
    df[df["firstSeg_prev"] == "down"]
    .groupby(["sub", "proba", "firstSeg"])["aSPv"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"firstSeg": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["firstSeg_prev"] == "up"]
    .groupby(["sub", "proba", "firstSeg"])["aSPv"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"firstSeg": "current_state"})
up_transitions["previous_state"] = "up"
# %%
# Combine results
conditional_probabilities = pd.concat([down_transitions, up_transitions])
print(conditional_probabilities)
# %%
conditional_probabilities["transition_state"] = list(
    zip(
        conditional_probabilities["current_state"],
        conditional_probabilities["previous_state"],
    )
)

conditional_probabilities["transition_state"] = conditional_probabilities[
    "transition_state"
].astype(str)
print(conditional_probabilities)
# %%
for s in conditional_probabilities["sub"].unique():
    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=conditional_probabilities[conditional_probabilities["sub"] == s],
        col="proba",
        col_wrap=3,
        height=8,
        aspect=1.5,
    )

    # Create barplots for each sub
    facet_grid.map_dataframe(
        sns.barplot,
        x="transition_state",
        y="conditional_prob",
    )

    # Adjust the layout to prevent title overlap
    plt.subplots_adjust(top=0.85)  # Increases space above subplots

    # Add a main title for the entire figure
    facet_grid.figure.suptitle(f"Subject {s}", fontsize=16, y=0.98)

    # Set titles for each subplot
    for ax, p in zip(
        facet_grid.axes.flat, np.sort(conditional_probabilities.proba.unique())
    ):
        ax.set_title(f"Sampling Bias p={p} : P(C(t+1)|C(t))")
        ax.set_xlabel("Transition State")
        ax.set_ylabel("Conditional Probability")
        # ax.tick_params(axis='x', rotation=45)

    # Adjust spacing between subplots
    facet_grid.figure.subplots_adjust(
        wspace=0.2, hspace=0.3  # Slightly increased to provide more vertical space
    )

    # Show the plot
    plt.show()
# %%
# Define transition counts for previous state = down
down_transitions = (
    df[df["firstSeg_prev"] == "down"]
    .groupby(["sub", "proba", "firstSeg"])["aSPv"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"firstSeg": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["firstSeg_prev"] == "up"]
    .groupby(["sub", "proba", "firstSeg"])["aSPv"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"firstSeg": "current_state"})
up_transitions["previous_state"] = "up"
# %%
# Combine results
conditional_probabilities = pd.concat([down_transitions, up_transitions])
# %%
conditional_probabilities["transition_state"] = list(
    zip(
        conditional_probabilities["current_state"],
        conditional_probabilities["previous_state"],
    )
)

conditional_probabilities["transition_state"] = conditional_probabilities[
    "transition_state"
].astype(str)
conditional_probabilities["transition_state"].unique()


# %%
def classify_subject_behavior(conditional_probabilities):
    # Create a function to categorize behavior for a single probability condition
    def categorize_single_proba(group):
        # Transition probabilities for this probability condition
        down_to_down = group[group["transition_state"] == "('down', 'down')"][
            "conditional_prob"
        ].values[0]
        up_to_up = group[group["transition_state"] == "('up', 'up')"][
            "conditional_prob"
        ].values[0]
        down_to_up = group[group["transition_state"] == "('up', 'down')"][
            "conditional_prob"
        ].values[0]
        up_to_down = group[group["transition_state"] == "('down', 'up')"][
            "conditional_prob"
        ].values[0]

        # Persistent: high probability of staying in the same state
        if down_to_down > 0.6 and up_to_up > 0.6:
            return "Persistent"

        # Alternating: high probability of switching states
        if down_to_up > 0.6 and up_to_down > 0.6:
            return "Alternating"

        return "Random"

    # Classify behavior for each subject and probability
    subject_proba_behavior = (
        conditional_probabilities.groupby(["sub", "proba"])
        .apply(lambda x: categorize_single_proba(x))
        .reset_index(name="behavior")
    )
    print(subject_proba_behavior)

    # Count behaviors for each subject across probabilities
    behavior_counts = (
        subject_proba_behavior.groupby(["sub", "behavior"]).size().unstack(fill_value=0)
    )

    # Classify subject based on behavior consistency across at least two probabilities
    def final_classification(row):
        if row["Persistent"] >= 2:
            return "Persistent"
        elif row["Alternating"] >= 2:
            return "Alternating"
        else:
            return "Random"

    subject_classification = behavior_counts.apply(
        final_classification, axis=1
    ).reset_index()
    subject_classification.columns = ["sub", "behavior_class"]

    # Visualize classification
    plt.figure(figsize=(10, 6))
    behavior_counts = subject_classification["behavior_class"].value_counts()
    plt.pie(behavior_counts, labels=behavior_counts.index, autopct="%1.1f%%")
    plt.title("Subject Behavior Classification\n(Consistent Across Probabilities)")
    plt.show()

    # Print detailed results
    print(subject_classification)

    # Additional detailed view
    detailed_behavior = subject_proba_behavior.pivot_table(
        index="sub", columns="proba", values="behavior", aggfunc="first"
    )
    print("\nDetailed Behavior Across Probabilities:")
    print(detailed_behavior)

    return subject_classification


subject_classification = classify_subject_behavior(conditional_probabilities)
# %%
# Perform classification
# Optional: Create a more detailed summary
summary = subject_classification.groupby("behavior_class").size()
print("\nBehavior Classification Summary:")
print(summary)
# %%
