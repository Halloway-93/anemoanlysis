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
main_dir = "/Users/mango/oueld.h/contextuaLearning/directionCue/results_imposeDirection"
pathFig = "/Users/mango/Contextual-Learning/directionCue/figures/imposeDirection"
df = pd.read_csv("/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_imposedArrow.csv")
print(df)
# %%
allEvents = pd.read_csv(os.path.join(main_dir, "allEvents.csv"))
# To align anemo data that start at trial 0
allEvents["trial"] = allEvents["trial"].values - 1
allEvents.columns
# %%
df = df[df["sub"] != 10]
# %%
df.rename(columns={"cond": "proba"}, inplace=True)
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
            if prev_trial.empty:
                print(df[df["trial"] == t]["trial"])

            if not prev_trial.empty:  # Ensure the previous trial exists
                # Assign trial direction from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "TD_prev",
                ] = prev_trial["target_direction"].values[0]

                # print(prev_trial["trial_color_chosen"].values[0])
                # Assign color from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "arrow_prev",
                ] = prev_trial["arrow"].values[0]
# %%
df[df["TD_prev"].isna()]
# %%
df = df[~(df["TD_prev"].isna())]
df[df["TD_prev"].isna()]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
# %%
df[df["aSPv"] == df["aSPv"].max()]
# %%
sns.histplot(data=df, x="aSPv")
plt.show()
# %%
balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
print(balance)
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    plt.show()
# %%
dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
# %%
np.abs(dd.aSPv.values).max()
# %%

aSPv = dd[dd.arrow == "up"]["aSPv"]
proba = dd[dd.arrow == "up"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (up): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.arrow == "down"]["aSPv"]
proba = dd[dd.arrow == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (down): {correlation}, p-value: {p_value}")

# %%
aSPv = dd[dd.proba == 1]["aSPv"]

arrow = dd[dd.proba == 1]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 1): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.proba == 0]["aSPv"]

arrow = dd[dd.proba == 0]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 0): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.proba == 0.75]["aSPv"]
arrow = dd[dd.proba == 0.75]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 0.75): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd.proba == 0.25]["aSPv"]
arrow = dd[dd.proba == 0.25]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 0.25): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd.proba == 0.5]["aSPv"]
arrow = dd[dd.proba == 0.5]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
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
    hue="arrow",
    hue_order=["down", "up"],
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

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df,
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="arrow",
    hue_order=["down", "up"],
)
_ = plt.title("ASEM Across Probabilities", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
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
    hue="arrow",
    hue_order=["down", "up"],
    kind="violin",
    split=True,
    height=10,
)
_ = plt.title("ASEM Across Probabilities", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/asemAcrossprobaviolin.svg", transparent=True)
plt.show()

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "up"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow DOWN", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsDOWNFullProba.svg", transparent=True)
plt.show()

model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 1.0],
    re_formula="~arrow",
    groups=df[df.proba == 1.0]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 0.0],
    re_formula="~arrow",
    groups=df[df.proba == 0.0]["sub"],
).fit(method=["lbfgs"])
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=["lbfgs"])
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 0.25],
    re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# a %%
model = smf.mixedlm(
    "aSPv~C( arrow,Treatment('up') )",
    data=df[df.proba == 0.5],
    re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "aSPv~ C(proba,Treatment(0.5))",
    data=df[df.arrow == "up"],
    re_formula="~proba",
    groups=df[df.arrow == "up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~ C(proba,Treatment(0.5))",
    data=df[df.arrow == "down"],
    re_formula="~proba",
    groups=df[df.arrow == "down"]["sub"],
).fit()
model.summary()


# %%
downarrowsPalette = ["#0000FF", "#A2D9FF"]
uparrowsPalette = ["#FFA500", "#FFD699"]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    hue_order=["down", "up"],
    data=df,
    errorbar="ci",
    fill=False,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
)
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    data=dd,
    dodge=True,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    jitter=True,
    size=8,
    # alpha=0.5,
    legend=False,
)
# plt.title("ASEM Across 5 Probabilities", fontsize=30)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/aSPvarrowsFullProba.svg", transparent=True)
plt.show()
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "target_direction",
        "TD_prev",
        "aSPv",
    ]
]
# %%
df["TD_prev"].unique()
# %%
df["arrow_prev"].unique()
# %%
learningCurve = (
    df_prime.groupby(["sub", "proba", "arrow", "TD_prev"])
    .mean()[["aSPv"]]
    .reset_index()
)


print(learningCurve)
# %%
df_prime.groupby(["proba", "arrow", "TD_prev"]).count()[["aSPv"]]

# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()

# %%
# Create the plot using catplot
g = sns.catplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[downarrowsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.5,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(downarrowsPalette[0])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[downarrowsPalette[0]],
    dodge=True,
    jitter=True,
    size=8,
    # alpha=0.7,
    data=dd[dd.arrow == "down"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=downarrowsPalette[0]),
]
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Down", fontsize=30)
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
    data=df[df.arrow == "up"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[uparrowsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.5,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(uparrowsPalette[0])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[uparrowsPalette[0]],
    dodge=True,
    jitter=True,
    size=8,
    # alpha=0.7,
    data=dd[dd.arrow == "up"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=uparrowsPalette[0]),
]
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=25)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTDFullProba.svg", transparent=True)
plt.show()
# %%
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "interaction",
        "aSPv",
    ]
]
# %%
learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "arrow"])
    .mean()[["aSPv"]]
    .reset_index()
)

# %%
df_prime.groupby(["sub", "proba", "interaction", "arrow"]).count()[["aSPv"]]

# %%
df_prime.groupby(["proba", "interaction", "arrow"]).count()[["aSPv"]]

# %%
print(learningCurveInteraction)
# %%
df["interaction"].unique()
# %%
colorsPalettes = ["#0000FF", "#FFA500", "#A2D9FF", "#FFD699"]
hue_order = [("right", "down"), ("right", "up"), ("left", "down"), ("left", "up")]
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "down"],
)
plt.title(
    "ASEM:Arrow Down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.5, 1.5)
plt.xlabel("P(Left|Down)", fontsize=30)
plt.savefig(pathFig + "/aSPvDownInteractionFullProba.svg", transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()
# %%

# %%

# Doing the analysis without the the deterministic blocks

df = df[~((df["proba"] == 0) | (df["proba"] == 1))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# %%

balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="arrow", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    plt.show()
# %%
dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
# %%
np.abs(dd.aSPv.values).max()
# %%

aSPv = dd[dd.arrow == "up"]["aSPv"]
proba = dd[dd.arrow == "up"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (up): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.arrow == "down"]["aSPv"]
proba = dd[dd.arrow == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (down): {correlation}, p-value: {p_value}")

# %%
aSPv = dd[dd.proba == 0.75]["aSPv"]
arrow = dd[dd.proba == 0.75]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 0.75): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd.proba == 0.25]["aSPv"]
arrow = dd[dd.proba == 0.25]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 0.25): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd.proba == 0.5]["aSPv"]
arrow = dd[dd.proba == 0.5]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
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
# %%
sns.histplot(data=df, x="aSPv", alpha=0.5)
plt.show()
# %%


# Set up the FacetGrid
facet_grid = sns.FacetGrid(data=df, col="proba", col_wrap=3, height=8, aspect=1.5)

# Create pointplots for each sub
facet_grid.map_dataframe(
    sns.histplot,
    x="aSPv",
    hue="arrow",
    hue_order=["down", "up"],
    alpha=0.3,
    # palette=[downarrowsPalette[0], uparrowsPalette[0]],
)

# Add legends
facet_grid.add_legend()

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

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df,
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="arrow",
    hue_order=["down", "up"],
)
_ = plt.title("ASEM Across 3 Different Probabilities", fontsize="30")
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-0.5, 0.5)
plt.savefig(pathFig + "/asemAcrossProbapp.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "up"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUP.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    alpha=0.7,
    palette="tab20",
)
_ = plt.title("ASEM Per Subject: Arrow DOWN", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.75, 1.75)
plt.savefig(pathFig + "/individualsDOWN.svg", transparent=True)
plt.show()
# %%

model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 0.75],
    re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=["lbfgs"])
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 0.25],
    # re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~C( arrow )",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "aSPv~ C(proba, Treatment(0.5))",
    data=df[df.arrow == "up"],
    # re_formula="~proba",
    groups=df[df.arrow == "up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~ C(proba, Treatment(0.5))",
    data=df[df.arrow == "down"],
    # re_formula="~proba",
    groups=df[df.arrow == "down"]["sub"],
).fit()
model.summary()


# %%
downarrowsPalette = ["#0000FF", "#A2D9FF"]
uparrowsPalette = ["#FFA500", "#FFD699"]
# %%

dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    hue_order=dd["arrow"].unique(),
    data=df,
    errorbar="ci",
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    fill=False,
)
# plt.title("ASEM Across 3 Different Probabilities", fontsize=30)
plt.xlabel("P(Right|UP)=P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/aSPvarrows.svg", transparent=True)

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    data=dd,
    dodge=True,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    jitter=True,
    size=8,
    # alpha=0.5,
    legend=False,
)
plt.show()
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "target_direction",
        "TD_prev",
        "aSPv",
    ]
]
# %%
learningCurve = (
    df_prime.groupby(["sub", "proba", "arrow", "TD_prev"])
    .mean()[["aSPv"]]
    .reset_index()
)


# %%
df_prime.groupby(["proba", "arrow", "TD_prev"]).count()[["aSPv"]]

dd = df.groupby(["sub", "arrow", "proba", "TD_prev"])[["aSPv"]].mean().reset_index()
# %%
g = sns.catplot(
    data=df[df.arrow == "up"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[uparrowsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.5,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(uparrowsPalette[0])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[uparrowsPalette[0]],
    dodge=True,
    jitter=True,
    size=8,
    # alpha=0.7,
    data=dd[dd.arrow == "up"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=uparrowsPalette[0]),
]
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=25)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTD.svg", transparent=True)
plt.show()
# %%
# Create the plot using catplot
g = sns.catplot(
    data=df[df.arrow == "down"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[downarrowsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.5,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(downarrowsPalette[0])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[downarrowsPalette[0]],
    dodge=True,
    jitter=True,
    size=8,
    # alpha=0.7,
    data=dd[dd.arrow == "down"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=downarrowsPalette[0]),
]
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Down", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=25)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvdownTD.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    hue_order=["down", "up"],
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    data=df[df.TD_prev == "right"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD(Right) ", fontsize=30)
plt.xlabel("P(Left|Down)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/aSPvTDRight.svg", transparent=True)
plt.show()
# Adding the interacrion between  previous arrow and previous TD.
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    hue_order=["down", "up"],
    palette=[downarrowsPalette[1], uparrowsPalette[1]],
    data=df[df.TD_prev == "left"],
)
plt.legend(fontsize=20)
plt.title("Anticipatory Velocity Given Previous TD(Left) ", fontsize=30)
plt.xlabel("P(Left|Down)", fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=25)
plt.ylim(-0.75, 0.75)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig(pathFig + "/aSPvTDLeft.svg", transparent=True)
plt.show()
# Adding the interacrion between  previous arrow and previous TD.
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "arrow",
        "interaction",
        "aSPv",
    ]
]
# %%

learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "arrow"])
    .mean()["aSPv"]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["sub", "proba", "interaction", "arrow"]).count()[["aSPv"]]

# %%
df["interaction"].unique()
# %%
colorsPalettes = ["#0000FF", "#FFA500", "#A2D9FF", "#FFD699"]
hue_order = [("right", "down"), ("right", "up"), ("left", "down"), ("left", "up")]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "up"],
)
plt.title(
    "Anticipatory Smooth Eye Movement: Arrow Up\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.xlabel("P(Right|Up)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.ylim(-1.25, 1.25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.savefig(pathFig + "/aSPvUpInteraction.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    palette=colorsPalettes,
    hue="interaction",
    hue_order=hue_order,
    data=df[df.arrow == "down"],
)
plt.title(
    "ASEM:Arrow Down\n Interaction of Previous Target Direction & arrow Chosen",
    fontsize=30,
)
plt.ylim(-1.25, 1.25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.xlabel("P(Left|Down)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvDownInteraction.svg", transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow,Treatment('up'))*C(TD_prev)",
    data=df[df.proba == 0.25],
    # re_formula="~arrow",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow,Treatment('up'))*C(TD_prev)",
    data=df[df.proba == 0.75],
    # re_formula="~arrow",
    groups=df[df.proba == 0.75]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    # re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# Sampling Bias Analysis:

df["sampling"] = list(zip(df["arrow"], df["arrow_prev"]))
# %%
# Group by and count
grouped = (
    df.groupby(["sub", "proba", "sampling"])["aSPv"].count().reset_index(name="count")
)

grouped["sampling"] = grouped["sampling"].astype(str)
# Calculate percentages using transform to keep the original DataFrame structure
grouped["percentage"] = grouped.groupby(["sub", "proba"])["count"].transform(
    lambda x: x / x.sum() * 100
)
# %%
grouped["sampling"].count()
# %%
sns.histplot(data=grouped, x="sampling")
plt.show()
# %%
for s in grouped["sub"].unique():
    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=grouped[grouped["sub"] == s], col="proba", col_wrap=3, height=8, aspect=1.5
    )

    facet_grid.add_legend()
    # Create pointplots for each sub
    facet_grid.map_dataframe(
        sns.barplot,
        x="sampling",
        y="percentage",
    )
    # Set titles for each subplot
    for ax, p in zip(facet_grid.axes.flat, np.sort(df.proba.unique())):
        ax.set_title(f"Sampling Bias p={p} : P(C(t+1)|C(t))")
    # Adjust spacing between subplots
    facet_grid.figure.subplots_adjust(
        wspace=0.2, hspace=0.2
    )  # Adjust wspace and hspace as needed

    # Show the plot
    plt.show()

# %%


# %%
# df.groupby(["sub", "proba", "arrow", "up_arrow_position", "TD_prev"])[
#     "aSPv"
# ].mean().reset_index()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "arrow"])["aSPv"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="arrow", values="aSPv"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = (
    np.abs(pivot_table["down"]) + np.abs(pivot_table["up"])
) / 2

print(pivot_table)
# %%
sns.scatterplot(
    data=pivot_table, x="up", y="down", hue="proba", palette="viridis", s=50
)
plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="adaptation",
)
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="down",
)
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="up",
)
plt.show()
# %%
# Create the plot with connected dots for each participant
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pivot_table,
    x="up",
    y="down",
    hue="proba",
    palette="viridis",
    style="proba",
    markers=["o", "s", "D", "^", "v"],
    s=50,
)
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["up"], subset["down"], color="gray", alpha=0.5, linestyle="--")
# Add plot formatting
plt.axhline(0, color="black", linestyle="--")
plt.axvline(0, color="black", linestyle="--")
plt.title(f"Participants adaptaion across probabilites")
plt.xlabel("up")
plt.ylabel("down")
plt.ylim(-2.5, 2.5)
plt.xlim(-2.5, 2.5)
plt.legend(title="proba")
plt.tight_layout()
plt.show()
plt.show()
# %%
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["up"], subset["down"], color="gray", alpha=0.5, linestyle="--")
    sns.scatterplot(
        data=pivot_table[pivot_table["sub"] == sub],
        x="up",
        y="down",
        hue="proba",
        palette="viridis",
        style="proba",
        markers=["o", "s", "D"],
    )
    # Add plot formatting
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Participant:{sub}")
    plt.xlabel("up")
    plt.ylabel("down")
    plt.legend(title="proba")
    plt.tight_layout()
    plt.show()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "interaction"])["aSPv"].mean().reset_index()
print(mean_velo)
mean_velo["interaction"] = mean_velo["interaction"].astype("str")
# %%
# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="interaction", values="aSPv"
).reset_index()

# Calculate the adaptation
# pivot_table["adaptation"] = (
#     np.abs(pivot_table["green"]) + np.abs(pivot_table["red"])
# ) / 2

print(pivot_table)
pivot_table = pd.DataFrame(pivot_table)
pivot_table.columns
# %%
pivot_table.columns[2]
# %%
# pivot_table.rename(
#     columns={
#         ('left', 'green'): "left_green",
#         ('left', 'red'): "left_red",
#         ('right', 'green'): "right_green",
#         ('right', 'red'): "right_red",
#     },
#     inplace=True,
# )
#
# pivot_table.columns
# %%
sns.scatterplot(
    data=pivot_table, x="('right', 'up')", y="('right', 'down')", hue="proba"
)
# Or alternatively
plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()
# a %%
sns.pointplot(
    data=pivot_table,
    x="proba",
    y="adaptation",
)
plt.show()
