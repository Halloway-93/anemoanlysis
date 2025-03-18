from scipy import stats
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
badTrials = df[df["aSPv"] > 8]["aSPv"]
print(badTrials)
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
for s in df["sub"].unique():
    sns.lmplot(
        data=df[(df["sub"] == s)],
        y="aSPv",
        hue="arrow",
        hue_order=['down','up'],
        x="proba",
    )
    plt.title(f"Subject{s}")
    plt.show()

# %%
for a in df["arrow"].unique():
    sns.lmplot(data=df[(df["arrow"] == a)], x="aSPv", hue="sub", y="proba", height=10)
    plt.title(f"Arrow {a}")
    plt.show()

# %% 
# Create a list to store results
slope_data = []

# Loop through each unique color
for c in df["arrow"].unique():
    # Loop through each subject within this color
    for s in df[df["arrow"] == c]["sub"].unique():
        # Get data for this specific color and subject
        subset = df[(df["arrow"] == c) & (df["sub"] == s)]

        # Only calculate slope if we have enough data points
        if len(subset) > 1:
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                subset["proba"], subset["aSPv"]
            )

            # Store results
            slope_data.append(
                {
                    "sub": s,
                    "arrow": c,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                }
            )

# Convert to DataFrame
slope_df = pd.DataFrame(slope_data)

# If you want to merge this with your original dataframe
# First create a unique key for merging
df["arrow_sub"] = df["arrow"] + "_" + df["sub"].astype(str)
slope_df["arrow_sub"] = slope_df["arrow"] + "_" + slope_df["sub"].astype(str)
print(slope_df)
# %%
slopeD = slope_df[slope_df["arrow"] == "down"]["slope"]
slopeU = slope_df[slope_df["arrow"] == "up"]["slope"]

# Spearman's rank correlation
correlation, p_value = spearmanr(slopeD, slopeU)
print(
    f"Spearman's correlation(Slope Down, Slope Up): {correlation}, p-value: {p_value}"
)
# %%
aSPvD = (
    dd[(dd["arrow"] == "down") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["arrow"] == "down") & (dd["proba"] == 0.25)]["aSPv"].values
)
aSPvU = (
    dd[(dd["arrow"] == "Up") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["arrow"] == "Up") & (dd["proba"] == 0.25)]["aSPv"].values
)
# %%

# Spearman's rank correlation
correlation, p_value = spearmanr(slopeD, aSPvD)
print(f"Spearman's correlation(Slope Green, aSPvG): {correlation}, p-value: {p_value}")
# %%
# Extract slope values for Green and Red colors
green_slopes = slope_df[slope_df.arrow == "down"]["slope"]
red_slopes = slope_df[slope_df.arrow == "up"]["slope"]

# Create scatter plot
plt.figure(figsize=(8, 8))  # Square figure for equal axes
plt.scatter(x=green_slopes, y=red_slopes, alpha=0.7)

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(green_slopes, red_slopes)

# Find the range for both axes to center around 0
all_values = np.concatenate([green_slopes, red_slopes])
max_abs_val = max(abs(all_values.min()), abs(all_values.max()))
axis_limit = max_abs_val * 1.1  # Add 10% margin

# Set equal limits centered on 0
plt.xlim(-axis_limit, axis_limit)
plt.ylim(-axis_limit, axis_limit)

# Create x values for the regression line
x_line = np.linspace(-axis_limit, axis_limit, 100)

# Calculate corresponding y values for regression line
y_line = slope * x_line + intercept

# Plot the regression line
plt.plot(
    x_line,
    y_line,
    color="red",
    linestyle="--",
    label=f"Regression: y = {slope:.3f}x + {intercept:.3f}",
)

# Add x=y line
plt.plot(
    [-axis_limit, axis_limit],
    [axis_limit, -axis_limit],
    "k-",
    alpha=0.5,
    label="x = -y",
)

# Add text with regression parameters
plt.annotate(
    f"y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.3f}",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)

# Add reference lines at x=0 and y=0
plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
plt.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

# Add labels and title
plt.xlabel("Down Slope")
plt.ylabel("Up Slope")
plt.title("Relationship Between Down and Up Condition Slopes")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")

# Make axes equal
plt.axis("equal")

# Show plot
plt.tight_layout()
plt.savefig(pathFig + "/linearRegressionSlopesFullProba.pdf", transparent=True)
plt.show()
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
    ax.set_title(f"Horizontal aSPv: P(Right|up)=P(Left|down)={p}")
    ax.legend(["up", "down"])
# Adjust spacing between subplots
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

# %%
df1 = df[((df["proba"] < 1))]
df1
# %%
df2 = df[((df["proba"] > 0))]
df2
# %%
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
    hue="arrow",
    hue_order=["down", "up"],
)
sns.pointplot(
    data=df2,
    x="proba",
    y="aSPv",
    n_boot=10000,
    capsize=0.1,
    errorbar="ci",
    hue="arrow",
    hue_order=["down", "up"],
    ls="--",
    legend=False,
)
# _ = plt.title("Horizontal aSPv Across Probabilities", fontsize=30)
plt.legend(title="Arrow", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/asemAcrossProbappFullProba.pdf", transparent=True)
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
    fill=False,
    gap=0.1,
    # inner="stick",
    height=10,
    cut=0,
)
# _ = plt.title("Horizontal aSPv Across Probabilities", fontsize=30)
plt.legend(title="Arrow", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/asemAcrossprobaviolinFullProba.pdf", transparent=True)
plt.show()

# %%
dd[(dd.proba == 1) & (dd.arrow == "up") & (dd.aSPv < 0)]
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
_ = plt.title("Horizontal aSPv Per Subject: Arrow UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.pdf", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=dd[dd.arrow == "up"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("Horizontal aSPv Per Subject: Arrow UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.pdf", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=dd[dd.arrow == "down"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("Horizontal aSPv Per Subject: Arrow Down", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/individualsDownFullProba.pdf", transparent=True)
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
_ = plt.title("Horizontal aSPv Per Subject: Arrow DOWN", fontsize=30)
plt.legend(fontsize=10)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsDOWNFullProba.pdf", transparent=True)
plt.show()
# %%
model = smf.mixedlm(
    "aSPv~C( arrow )*proba",
    data=df,
    re_formula="~proba",
    groups=df["sub"],
).fit()
model.summary()
# %%
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
    "aSPv~arrow",
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
downarrowsPalette = ["#0F68A9", "#A2D9FF"]
uparrowsPalette = ["#FAAE7B", "#FFD699"]
dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
# %%
g=sns.catplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=["down", "up"],
    data=df,
    fill=False,
    legend=False,
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
    size=6,
    linewidth=1,
    # alpha=0.5,
    # legend=False,
)
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Arrow", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvarrowsFullProba.pdf", transparent=True)
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
    alpha=0.8,
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
    size=6,
    linewidth=1,
    # alpha=0.7,
    data=dd[dd.arrow == "down"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=downarrowsPalette[0]),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize=20
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Down", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvdownTDFullProba.pdf", transparent=True)
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
    alpha=0.8,
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
    linewidth=1,
    size=6,
    # alpha=0.7,
    data=dd[dd.arrow == "up"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=uparrowsPalette[0]),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize=20
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTDFullProba.pdf", transparent=True)
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
colorsPalettes = ["#0F68A9", "#FAAE7B"]
# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.arrow == "up"],
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
    if i > 7:  # Second half of bars for each x-value

        bar.set_facecolor("none")  # Make bar empty
        bar.set_hatch("///")  # Add diagonal lines
        (
            bar.set_edgecolor(colorsPalettes[0])
            if i < 12
            else bar.set_edgecolor(colorsPalettes[1])
        )  # Maintain the category color

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
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
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
g.ax.set_title("Up Trials:\n Previous TD and its Arrow", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvUpInteractionFullProba.pdf", transparent=True)
plt.show()

# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.arrow == "down"],
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
    if i > 7:  # Second half of bars for each x-value

        bar.set_facecolor("none")  # Make bar empty
        bar.set_hatch("///")  # Add diagonal lines
        (
            bar.set_edgecolor(colorsPalettes[0])
            if i < 12
            else bar.set_edgecolor(colorsPalettes[1])
        )  # Maintain the category color

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
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
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
g.ax.set_title("Down Trials:\n Previous TD and its Arrow", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvDownInteractionFullProba.pdf", transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.5]["sub"],
).fit(method="lbfgs")
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

# Doing the analysis without the the deterministic blocks

df = df[~((df["proba"] == 0) | (df["proba"] == 1))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# %%

balance = df.groupby(["arrow", "sub", "proba"])["trial"].count().reset_index()
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


for s in df["sub"].unique():
    sns.lmplot(
        data=df[(df["sub"] == s)],
        y="aSPv",
        hue="arrow",
        hue_order=['down','up'],
        x="proba",
    )
    plt.title(f"Subject{s}")
    plt.show()

# %%
for a in df["arrow"].unique():
    sns.lmplot(data=df[(df["arrow"] == a)], x="aSPv", hue="sub", y="proba", height=10)
    plt.title(f"Arrow {a}")
    plt.show()

# %% 
# Create a list to store results
slope_data = []

# Loop through each unique color
for c in df["arrow"].unique():
    # Loop through each subject within this color
    for s in df[df["arrow"] == c]["sub"].unique():
        # Get data for this specific color and subject
        subset = df[(df["arrow"] == c) & (df["sub"] == s)]

        # Only calculate slope if we have enough data points
        if len(subset) > 1:
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                subset["proba"], subset["aSPv"]
            )

            # Store results
            slope_data.append(
                {
                    "sub": s,
                    "arrow": c,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                }
            )

# Convert to DataFrame
slope_df = pd.DataFrame(slope_data)

# If you want to merge this with your original dataframe
# First create a unique key for merging
df["arrow_sub"] = df["arrow"] + "_" + df["sub"].astype(str)
slope_df["arrow_sub"] = slope_df["arrow"] + "_" + slope_df["sub"].astype(str)
print(slope_df)
# %%
slopeD = slope_df[slope_df["arrow"] == "down"]["slope"]
slopeU = slope_df[slope_df["arrow"] == "up"]["slope"]

# Spearman's rank correlation
correlation, p_value = spearmanr(slopeD, slopeU)
print(
    f"Spearman's correlation(Slope Down, Slope Up): {correlation}, p-value: {p_value}"
)
# %%
aSPvD = (
    dd[(dd["arrow"] == "down") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["arrow"] == "down") & (dd["proba"] == 0.25)]["aSPv"].values
)
aSPvU = (
    dd[(dd["arrow"] == "Up") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["arrow"] == "Up") & (dd["proba"] == 0.25)]["aSPv"].values
)
# %%
# Spearman's rank correlation
correlation, p_value = spearmanr(slopeD, aSPvD)
print(f"Spearman's correlation(Slope Green, aSPvG): {correlation}, p-value: {p_value}")
# %%
# Extract slope values for Green and Red colors
green_slopes = slope_df[slope_df.arrow == "down"]["slope"]
red_slopes = slope_df[slope_df.arrow == "up"]["slope"]

# Create scatter plot
plt.figure(figsize=(8, 8))  # Square figure for equal axes
plt.scatter(x=green_slopes, y=red_slopes, alpha=0.7)

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(green_slopes, red_slopes)

# Find the range for both axes to center around 0
all_values = np.concatenate([green_slopes, red_slopes])
max_abs_val = max(abs(all_values.min()), abs(all_values.max()))
axis_limit = max_abs_val * 1.1  # Add 10% margin

# Set equal limits centered on 0
plt.xlim(-axis_limit, axis_limit)
plt.ylim(-axis_limit, axis_limit)

# Create x values for the regression line
x_line = np.linspace(-axis_limit, axis_limit, 100)

# Calculate corresponding y values for regression line
y_line = slope * x_line + intercept

# Plot the regression line
plt.plot(
    x_line,
    y_line,
    color="red",
    linestyle="--",
    label=f"Regression: y = {slope:.3f}x + {intercept:.3f}",
)

# Add x=y line
plt.plot(
    [-axis_limit, axis_limit],
    [axis_limit, -axis_limit],
    "k-",
    alpha=0.5,
    label="x = -y",
)

# Add text with regression parameters
plt.annotate(
    f"y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.3f}",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)

# Add reference lines at x=0 and y=0
plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
plt.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

# Add labels and title
plt.xlabel("Down Slope")
plt.ylabel("Up Slope")
plt.title("Relationship Between Down and Up Condition Slopes")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")

# Make axes equal
plt.axis("equal")

# Show plot
plt.tight_layout()
plt.savefig(pathFig + "/linearRegressionSlopes.pdf", transparent=True)
plt.show()
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
    ax.set_title(f"Horizontal aSPv: P(Right|up)=P(Left|down)={p}")
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
    n_boot=10000,
    capsize=0.1,
    errorbar="ci",
    hue="arrow",
    hue_order=["down", "up"],
)
# _ = plt.title("Horizontal aSPv Across Probabilities", fontsize=30)
plt.legend(title="Arrow", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/asemAcrossProbapp.pdf", transparent=True)
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
    fill=False,
    gap=0.1,
    # inner="stick",
    height=10,
    cut=0,
)
# _ = plt.title("Horizontal aSPv Across Probabilities", fontsize=30)
plt.legend(title="Arrow", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/asemAcrossprobaviolinnfp.pdf", transparent=True)
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
_ = plt.title("Horizontal aSPv Per Subject: Arrow UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUP.pdf", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=dd[dd.arrow == "up"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("Horizontal aSPv Per Subject: Arrow UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUP.pdf", transparent=True)
plt.show()
# %%
sns.lmplot(
    data=dd[dd.arrow == "down"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("Horizontal aSPv Per Subject: Arrow UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/individualsUP.pdf", transparent=True)
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
_ = plt.title("Horizontal aSPv Per Subject: Arrow DOWN", fontsize=30)
plt.legend(fontsize=10)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsDOWN.pdf", transparent=True)
plt.show()
# %%
model = smf.mixedlm(
    "aSPv~C( arrow,Treatment('up') )*C(proba,Treatment(0.5))",
    data=df,
    re_formula="~proba",
    groups=df["sub"],
).fit()
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
    "aSPv~arrow",
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
    "aSPv~ proba",
    data=df[df.arrow == "down"],
    re_formula="~proba",
    groups=df[df.arrow == "down"]["sub"],
).fit()
model.summary()


# %%
downarrowsPalette = ["#0F68A9", "#A2D9FF"]
uparrowsPalette = ["#FAAE7B", "#FFD699"]
dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
# %%
g=sns.catplot(
    x="proba",
    y="aSPv",
    hue="arrow",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=["down", "up"],
    data=df,
    fill=False,
    legend=False,
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
    size=6,
    linewidth=1,
    # alpha=0.5,
    # legend=False,
)
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Arrow", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvarrows.pdf", transparent=True)
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
    alpha=0.8,
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
    size=6,
    linewidth=1,
    # alpha=0.7,
    data=dd[dd.arrow == "down"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=downarrowsPalette[0]),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize=20
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Down", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvdownTD.pdf", transparent=True)
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
    alpha=0.8,
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
    linewidth=1,
    size=6,
    # alpha=0.7,
    data=dd[dd.arrow == "up"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=uparrowsPalette[0]),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize=20
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTD.pdf", transparent=True)
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
    data=df_prime[df_prime.arrow == "up"],
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
    data=learningCurveInteraction[learningCurveInteraction.arrow == "up"],
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
g.ax.set_title("Up Trials:\n Previous TD and its Arrow", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvUpInteraction.pdf", transparent=True)
plt.show()

# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.arrow == "down"],
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
    data=learningCurveInteraction[learningCurveInteraction.arrow == "down"],
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
g.ax.set_title("Down Trials:\n Previous TD and its Arrow", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvDownInteraction.pdf", transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.5]["sub"],
).fit(method="lbfgs")
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
mean_velo = df.groupby(["sub", "proba", "arrow"])["aSPv"].mean().reset_index()
print(mean_velo)
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="arrow", values="aSPv"
).reset_index()
pivot_table
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
plt.title("Participants adaptaion across probabilites")
plt.xlabel("up")
plt.ylabel("down")
plt.ylim(-4, 4)
plt.xlim(-4, 4)
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
# %%
