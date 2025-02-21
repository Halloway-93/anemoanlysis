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
main_dir = (
    "/Users/mango/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection"
)
pathFig = "/Users/mango/Contextual-Learning/directionCue/figures/voluntaryDirection"
df = pd.read_csv("/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_voluntaryArrow.csv")
print(df)
# %%
allEvents = pd.read_csv(os.path.join(main_dir, "allEvents.csv"))
# To align anemo data that start at trial 0
allEvents["trial"] = allEvents["trial"].values - 1
allEvents.columns
# %%
allEvents
# %%
df.rename(columns={"cond": "proba"}, inplace=True)
df.rename(columns={"chosen_arrow": "arrow"}, inplace=True)
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
                ] = prev_trial["target_direction"].values[0]

                # print(prev_trial["trial_color_chosen"].values[0])
                # Assign color from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "arrow_prev",
                ] = prev_trial["chosen_arrow"].values[0]
# %%
df[df["TD_prev"].isna()]
# %%
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
df = df[~(df["TD_prev"].isna())]
df[df["TD_prev"].isna()]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
# %%
df[df["aSPv"] == df["aSPv"].max()]["aSPv"]
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
dd = df.groupby(["sub", "arrow", "proba", "TD_prev"])[["aSPv"]].mean().reset_index()
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
df1 = df[(df["session"] == "session-04") & ((df["proba"] == 1))]
df1
# %%
df2 = df[(df["session"] == "session-04") & ((df["proba"] == 0))]
df2
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
# %%
model = smf.mixedlm(
    "aSPv~C( arrow,Treatment('up') )",
    data=df[df.proba == 0.5],
    re_formula="~arrow",
    groups=df[df.proba == 0.5]["sub"],
).fit()
model.summary()


# %%
model = smf.mixedlm(
    "aSPv~ proba",
    data=df[df.arrow == "up"],
    re_formula="~proba",
    groups=df[df.arrow == "up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~ proba",
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

# Define transition counts for previous state = down
down_transitions = (
    df[df["arrow_prev"] == "down"]
    .groupby(["sub", "proba", "arrow"])["aSPv"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"arrow": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["arrow_prev"] == "up"]
    .groupby(["sub", "proba", "arrow"])["aSPv"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"arrow": "current_state"})
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
    df[df["arrow_prev"] == "down"]
    .groupby(["sub", "proba", "arrow"])["aSPv"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"arrow": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["arrow_prev"] == "up"]
    .groupby(["sub", "proba", "arrow"])["aSPv"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"arrow": "current_state"})
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
    # re_formula="~arrow",
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
    re_formula="~arrow",
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
plt.savefig(pathFig + "/aSPvarrows.svg", transparent=True)
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


# Define transition counts for previous state = down
down_transitions = (
    df[df["arrow_prev"] == "down"]
    .groupby(["sub", "proba", "arrow"])["aSPv"]
    .count()
    .reset_index(name="count")
)
down_transitions["total"] = down_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")
down_transitions["conditional_prob"] = (
    down_transitions["count"] / down_transitions["total"]
)
down_transitions = down_transitions.rename(columns={"arrow": "current_state"})
down_transitions["previous_state"] = "down"

# Define transition counts for previous state = up
up_transitions = (
    df[df["arrow_prev"] == "up"]
    .groupby(["sub", "proba", "arrow"])["aSPv"]
    .count()
    .reset_index(name="count")
)
up_transitions["total"] = up_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
up_transitions["conditional_prob"] = up_transitions["count"] / up_transitions["total"]
up_transitions = up_transitions.rename(columns={"arrow": "current_state"})
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
def classify_subject_behavior(conditional_probabilities):
    # Create a function to categorize behavior for a single probability condition
    def categorize_single_proba(group):
        # Transition probabilities for this probability condition
        if (
            len(
                group[group["transition_state"] == "('down', 'down')"][
                    "conditional_prob"
                ]
            )
            > 0
        ):
            down_to_down = group[group["transition_state"] == "('down', 'down')"][
                "conditional_prob"
            ].values[0]
        else:
            down_to_down = 0

        if (
            len(group[group["transition_state"] == "('up', 'up')"]["conditional_prob"])
            > 0
        ):
            up_to_up = group[group["transition_state"] == "('up', 'up')"][
                "conditional_prob"
            ].values[0]
        else:
            up_to_up = 0

        if (
            len(
                group[group["transition_state"] == "('up', 'down')"]["conditional_prob"]
            )
            > 0
        ):
            down_to_up = group[group["transition_state"] == "('up', 'down')"][
                "conditional_prob"
            ].values[0]
        else:
            down_to_up = 0

        if len(
            group[group["transition_state"] == "('down', 'up')"]["conditional_prob"]
        ):

            up_to_down = group[group["transition_state"] == "('down', 'up')"][
                "conditional_prob"
            ].values[0]
        else:
            up_to_down = 0

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
dd = df.groupby(["sub", "proba", "arrow"])["aSPv"].mean().reset_index()
# %%
for s in dd["sub"].unique():
    behavior_value = subject_classification[subject_classification["sub"] == s][
        "behavior_class"
    ].values[0]
    dd.loc[dd["sub"] == s, "behavior"] = behavior_value

# %%
# %%
sns.lmplot(
    data=dd[(dd["arrow"] == "down")],
    x="proba",
    y="aSPv",
    hue="behavior",
)
plt.title("Down Arrow", fontsize=30)
plt.show()
# %%
sns.lmplot(
    data=dd[(dd["arrow"] == "up")],
    x="proba",
    y="aSPv",
    hue="behavior",
)
plt.title("Up Arrow", fontsize=30)
plt.show()

# %%

# Computing the peristance score based on the transition probabilites
# %%
conditional_probabilities.columns
# %%
dd = (
    conditional_probabilities.groupby(["sub", "transition_state"])["conditional_prob"]
    .mean()
    .reset_index()
)
# %%
dd["sub"].value_counts()
# %%
dd["transition_state"].unique()
# %%
ts = dd["transition_state"].unique()
new_rows = []
for s in dd["sub"].unique():
    existing_ts = dd[dd["sub"] == s]["transition_state"].unique()
    for t in ts:
        if t not in existing_ts:
            # Add a new row with sub = s and transition_state = t, setting transition_state to 0
            new_row = {"sub": s, "transition_state": t, "conditional_prob": 0}
            new_rows.append(new_row)

# Concatenate the new rows to the original DataFrame
dd = pd.concat([dd, pd.DataFrame(new_rows)], ignore_index=True)

print(dd)
# %%

dd = dd.groupby(["sub", "transition_state"])["conditional_prob"].mean().reset_index()
# %%
dd["transition_state"].unique()


# %%
# Function to classify transition_state as persistent or alternating
def classify_transition(state):
    return (
        "persistent"
        if state == "('up', 'up')" or state == "('down', 'down')"
        else "alternating"
    )


# Apply the classification function
dd["transition_type"] = dd["transition_state"].apply(classify_transition)
# %%

# Group by 'sub' and calculate the score
result = (
    dd.groupby("sub")
    .apply(
        lambda x: x[x["transition_type"] == "persistent"]["conditional_prob"].sum()
        - x[x["transition_type"] == "alternating"]["conditional_prob"].sum()
    )
    .reset_index(name="persistence_score")
)

print(result)

# %%
dd["transition_state"].unique()
# %%
result["persistence_score_up"] = np.nan
result["persistence_score_down"] = np.nan
print(result)
# %%
for s in dd["sub"].unique():
    up_up_prob = dd[(dd["sub"] == s) & (dd["transition_state"] == "('up', 'up')")][
        "conditional_prob"
    ].values[0]
    down_up_prob = dd[(dd["sub"] == s) & (dd["transition_state"] == "('down', 'up')")][
        "conditional_prob"
    ].values[0]
    result.loc[result["sub"] == s, "persistence_score_up"] = up_up_prob - down_up_prob
    down_down_prob = dd[
        (dd["sub"] == s) & (dd["transition_state"] == "('down', 'down')")
    ]["conditional_prob"].values[0]
    up_down_prob = dd[(dd["sub"] == s) & (dd["transition_state"] == "('up', 'down')")][
        "conditional_prob"
    ].values[0]
    result.loc[result["sub"] == s, "persistence_score_down"] = (
        down_down_prob - up_down_prob
    )
print(result)
# %%

# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "arrow"])["aSPv"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "arrow"], columns="proba", values="aSPv"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = np.abs(pivot_table[0.75] - pivot_table[0.25])
# %%
print(pivot_table)
# %%

# Select the relevant columns
adaptation = pivot_table[["sub", "arrow", "adaptation"]]

print(adaptation)
# %%
adaptation.groupby("sub")["adaptation"].mean().values
# %%

result["adaptation"] = adaptation.groupby("sub")["adaptation"].mean().values
result["adaptation_down"] = adaptation[adaptation["arrow"] == "down"][
    "adaptation"
].values
result["adaptation_up"] = adaptation[adaptation["arrow"] == "up"]["adaptation"].values
# %%
result = pd.DataFrame(result)
print(result)
# %%
sns.lmplot(data=result, x="persistence_score", y="adaptation")
plt.savefig(pathFig + "/samplingBiasArrows.svg", transparent=True)
plt.show()
# %%
sns.lmplot(data=result, x="persistence_score_down", y="adaptation_down")
plt.savefig(pathFig + "/samplingBiasDown.svg", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=result,
    x="persistence_score_up",
    y="adaptation_up",
)
plt.savefig(pathFig + "/samplingBiasUp.svg", transparent=True)

plt.show()
# %%
sns.lmplot(
    data=result,
    x="adaptation_down",
    y="adaptation_up",
)
plt.show()
# %%
sns.lmplot(
    data=result,
    x="persistence_score_down",
    y="persistence_score_up",
)
plt.show()
# %%


correlation, p_value = spearmanr(
    result["persistence_score"],
    result["adaptation"],
)
print(
    f"Spearman's correlation for the adaptation score): {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_down"],
    result["adaptation_down"],
)
print(
    f"Spearman's correlation for the adaptation score for down): {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_up"],
    result["adaptation_up"],
)
print(
    f"Spearman's correlation for the adaptation score for up): {correlation}, p-value: {p_value}"
)

# %%
model = sm.OLS.from_formula("adaptation_up~ persistence_score_up ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation_down~ persistence_score_down ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation~ persistence_score ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation_up~ adaptation_down ", result).fit()

print(model.summary())
# %%
df.columns
# %%
df.groupby(["sub", "proba", "arrow", "up_arrow_position", "TD_prev"])[
    "aSPv"
].mean().reset_index()
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
