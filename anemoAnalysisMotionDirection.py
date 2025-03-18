from scipy import stats
from scipy.stats import spearmanr,pearsonr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

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
#plt.show()
# %%
sns.histplot(data=df, x="SPlat")
#plt.show()
# %%
sns.histplot(data=df, x="aSPoff")
#plt.show()
# %%
sns.histplot(data=df, x="aSPon")
#plt.show()
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
df = df[~(df["TD_prev"].isna())]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
df["interaction"] = list(zip(df["TD_prev"], df["firstSeg_prev"]))
# %%
df[df["aSPv"] == df["aSPv"].max()]["aSPv"]
# %%
sns.histplot(data=df, x="aSPv")
#plt.show()
# %%
balance = df.groupby(["firstSeg", "sub", "proba"])["trial"].count().reset_index()
print(balance)
# %%
for sub in balance["sub"].unique():
    sns.barplot(
        x="proba", y="trial", hue="firstSeg", data=balance[balance["sub"] == sub]
    )
    plt.title(f"Subject {sub}")
    #plt.show()
# %%
dd = df.groupby(["sub", "firstSeg", "proba"])[["aSPv"]].mean().reset_index()
# %%
dd
# %%

np.abs(dd.aSPv.values).max()
# %%

aSPv = dd[dd["firstSeg"] == "Up"]["aSPv"]
proba = dd[dd["firstSeg"] == "Up"]["proba"]

# Spearman's rank correlation
correlation, p_value = pearsonr(aSPv, proba)
print(f"Pearson's correlation (Up): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.firstSeg == "Down"]["aSPv"]
proba = dd[dd.firstSeg == "Down"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (Down): {correlation}, p-value: {p_value}")

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
for s in df["sub"].unique():
    sns.lmplot(
        data=df[(df["sub"] == s)],
        y="aSPv",
        hue="firstSeg",
        hue_order=['Down','Up'],
        x="proba",
    )
    plt.title(f"Subject{s}")
    #plt.show()

# %%
for a in df["firstSeg"].unique():
    sns.lmplot(data=df[(df["firstSeg"] == a)], x="aSPv", hue="sub", y="proba", height=10)
    plt.title(f" First Seg {a}")
    #plt.show()

# %% 
# Create a list to store results
slope_data = []

# Loop through each unique color
for c in df["firstSeg"].unique():
    # Loop through each subject within this color
    for s in df[df["firstSeg"] == c]["sub"].unique():
        # Get data for this specific color and subject
        subset = df[(df["firstSeg"] == c) & (df["sub"] == s)]

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
                    "firstSeg": c,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                }
            )

# Convert to DataFrame
slope_df = pd.DataFrame(slope_data)

# If you want to merge this with your original dataframe
# First create a unique key for merging
df["firstSeg_sub"] = df["firstSeg"] + "_" + df["sub"].astype(str)
slope_df["firstSeg_sub"] = slope_df["firstSeg"] + "_" + slope_df["sub"].astype(str)
print(slope_df)
# %%
slopeD = slope_df[slope_df["firstSeg"] == "Down"]["slope"]
slopeU = slope_df[slope_df["firstSeg"] == "Up"]["slope"]

# Spearman's rank correlation
correlation, p_value = pearsonr(slopeD, slopeU)
print(
    f"Pearson's correlation(Slope Down, Slope Up): {correlation}, p-value: {p_value}"
)
# %%
aSPvD = (
    dd[(dd["firstSeg"] == "Down") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["firstSeg"] == "Down") & (dd["proba"] == 0.25)]["aSPv"].values
)
aSPvU = (
    dd[(dd["firstSeg"] == "Up") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["firstSeg"] == "Up") & (dd["proba"] == 0.25)]["aSPv"].values
)
# %%

# Spearman's rank correlation
correlation, p_value = spearmanr(slopeU, aSPvU)
print(f"Spearman's correlation(Slope Up, aSPvU): {correlation}, p-value: {p_value}")
# %%
correlation, p_value = spearmanr(slopeD, aSPvD)
print(f"Spearman's correlation(Slope Down, aSPvD): {correlation}, p-value: {p_value}")
# Extract slope values for Green and Red colors
green_slopes = slope_df[slope_df.firstSeg == "Down"]["slope"]
red_slopes = slope_df[slope_df.firstSeg == "Up"]["slope"]

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
#plt.show()
# %%
# cehcking the normality of the data
print(pg.normality(dd[dd.proba == 0.25]["aSPv"]))
# %%
stat, p = stats.kstest(
    dd["aSPv"], "norm", args=(dd["aSPv"].mean(), dd["aSPv"].std(ddof=1))
)
print(f"Statistic: {stat}, p-value: {p}")
# %%
x = dd["aSPv"]
ax = pg.qqplot(x, dist="norm")
#plt.show()


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
    ax.set_title(f"Horizontal aSPv: P(Right|up)=P(Left|down)={p}")
    ax.legend(["up", "down"])
# Adjust spacing between subplots
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
#plt.show()

fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df,
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
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/asemAcrossProbappFullProba.pdf", transparent=True)
#plt.show()
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
# _ = plt.title("Horizontal aSPv Across Probabilities", fontsize=30)
plt.legend(title="firstSeg", fontsize=20, title_fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|UP)=$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/asemAcrossprobaviolinFullProba.pdf", transparent=True)
#plt.show()

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
_ = plt.title("Horizontal aSPv Per Subject: firstSeg UP", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.pdf", transparent=True)
#plt.show()
# %%
sns.lmplot(
    data=dd[dd.firstSeg == "Up"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("Horizontal aSPv Per Subject: firstSeg UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|UP)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsUPFullProba.pdf", transparent=True)
#plt.show()
# %%
sns.lmplot(
    data=dd[dd.firstSeg == "Down"],
    x="proba",
    y="aSPv",
    hue="sub",
    palette="tab20",
    height=10,
)
_ = plt.title("Horizontal aSPv Per Subject: firstSeg UP", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.tight_layout()
plt.savefig(pathFig + "/individualsUPFullProba.pdf", transparent=True)
#plt.show()
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
_ = plt.title("Horizontal aSPv Per Subject: firstSeg DOWN", fontsize=30)
plt.legend(fontsize=10)
plt.xlabel("P(Left|DOWN)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsDOWNFullProba.pdf", transparent=True)
#plt.show()
# %%
model = smf.mixedlm(
    "aSPv~C( firstSeg )*C( proba,Treatment(0.5) )",
    data=df,
    re_formula="~proba",
    groups=df["sub"],
).fit()
model.summary()
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
    "aSPv~ C(proba,Treatment(0.5))",
    data=df[df.firstSeg == "Up"],
    re_formula="~proba",
    groups=df[df.firstSeg == "Up"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~ C(proba,Treatment(0.5))",
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
# fig = plt.figure()
# # Toggle full screen mode
# figManager = plt.get_current_fig_manager()
# figManager.full_screen_toggle()
g=sns.catplot(
    x="proba",
    y="aSPv",
    hue="firstSeg",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=["Down", "Up"],
    data=df,
    fill=False,
    legend=False,
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
    # legend=False,
)
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
legend_elements = [
    Patch(facecolor=downfirstSegsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=upfirstSegsPalette[0], alpha=1, label="Up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="firstSeg", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvfirstSegsFullProba.pdf", transparent=True)
#plt.show()
# %%
# %%
downfirstSegsPalette = ["#0F68A9", "#A2D9FF"]
upfirstSegsPalette = ["#FAAE7B", "#FFD699"]
dd = df.groupby(["sub", "firstSeg", "proba"])[["aSPv"]].mean().reset_index()
# %%
# fig = plt.figure()
# # Toggle full screen mode
# figManager = plt.get_current_fig_manager()
# figManager.full_screen_toggle()
g=sns.catplot(
    x="proba",
    y="aSPv",
    hue="firstSeg",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=["Down", "Up"],
    data=df,
    fill=False,
    legend=False,
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
    # legend=False,
)
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
legend_elements = [
    Patch(facecolor=downfirstSegsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=upfirstSegsPalette[0], alpha=1, label="Up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="firstSeg", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvfirstSegsFullProba.pdf", transparent=True)
#plt.show()
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["firstSeg"] == "Up")]["aSPv"],
    y=dd[(dd["proba"] == 0.25) & (dd["firstSeg"] == "Down")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.75) & (dd["firstSeg"] == "Up")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["firstSeg"] == "Down")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.5) & (dd["firstSeg"] == "Up")]["aSPv"],
    y=dd[(dd["proba"] == 0.5) & (dd["firstSeg"] == "Down")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["firstSeg"] == "Down")]["aSPv"],
    y=dd[(dd["proba"] == 0.50) & (dd["firstSeg"] == "Down")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.5) & (dd["firstSeg"] == "Down")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["firstSeg"] == "Down")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["firstSeg"] == "Down")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["firstSeg"] == "Down")]["aSPv"],
    paired=True,
)
print(ttest_results)

# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["firstSeg"] == "Up")]["aSPv"],
    y=dd[(dd["proba"] == 0.50) & (dd["firstSeg"] == "Up")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.5) & (dd["firstSeg"] == "Up")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["firstSeg"] == "Up")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["firstSeg"] == "Up")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["firstSeg"] == "Up")]["aSPv"],
    paired=True,
)
print(ttest_results)

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
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvdownTDFullProba.pdf", transparent=True)
#plt.show()

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
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTDFullProba.pdf", transparent=True)
#plt.show()
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
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvUpInteractionFullProba.pdf", transparent=True)
#plt.show()

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
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Down)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvDownInteractionFullProba.pdf", transparent=True)
#plt.show()
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
    #plt.show()
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
mean_velo = df.groupby(["sub", "proba", "firstSeg"])["aSPv"].mean().reset_index()
print(mean_velo)
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="firstSeg", values="aSPv"
).reset_index()
pivot_table
# %%
# Create the plot with connected dots for each participant
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pivot_table,
    x="Up",
    y="Down",
    hue="proba",
    palette="viridis",
    style="proba",
    markers=["o", "s", "D", "^", "v"],
    s=50,
)
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["Up"], subset["Down"], color="gray", alpha=0.5, linestyle="--")
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
#plt.show()
#plt.show()
# %%
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["Up"], subset["Down"], color="gray", alpha=0.5, linestyle="--")
    sns.scatterplot(
        data=pivot_table[pivot_table["sub"] == sub],
        x="Up",
        y="Down",
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
    #plt.show()
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
#plt.show()
# %%
