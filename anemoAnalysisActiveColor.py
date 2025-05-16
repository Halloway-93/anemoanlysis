from scipy import stats
from sklearn.preprocessing import LabelEncoder
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import spearmanr,pearsonr
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
from statannotations.Annotator import Annotator
# %%
main_dir = "/Volumes/work/brainets/oueld.h/contextuaLearning/ColorCue/data/"
pathFig = "/Users/mango/Contextual-Learning/ColorCue/figures/voluntaryColor/"
df = pd.read_csv("/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_activeColorCP.csv")
RedcolorsPalette = ["#e83865", "#cc3131"]
GreencolorsPalette = ["#008000", "#285943"]

# %%

df.rename(columns={"pR-Red": "proba"}, inplace=True)
df.rename(columns={"trial_color": "color"}, inplace=True)
print(df["sub"].unique())

# %%
df["sub"] = [int(x.split("-")[1]) for x in df["sub"]]
df["sub"].unique()
# %%
df.columns
# %%
allEvents = pd.read_csv(os.path.join(main_dir, "allEvents.csv"))
# To align anemo data that start at trial 0
allEvents["trial"] = allEvents["trial"].values - 1
allEvents["proba"] = allEvents["proba"].values / 100
allEvents.columns
# %%
allEvents
# %%
df = df.merge(allEvents, on=["sub", "proba", "trial"], how="inner").reset_index()
df["trial_color_UP"] = df["trial_color_UP"].apply(
    lambda x: "red" if x == 1 else "green"
)
allEvents["trial_color_chosen"] = allEvents["trial_color_chosen"].apply(
    lambda x: "red" if x == 1 else "green"
)
df.rename(columns={"trial_color_UP": "trialTgUP"}, inplace=True)
# %%
for s in df["sub"].unique():
    l1 = len(
        df[
            (df["proba"] == 0.25)
            & (df["color"] == "Red")
            & (df["sub"] == s)
            & (df["aSPv"] > 0)
        ]
    )
    l2 = len(
        df[
            (df["proba"] == 0.25)
            & (df["color"] == "Red")
            & (df["sub"] == s)
            & (df["aSPv"] < 0)
        ]
    )
    print(
        f" Subject{s}, Ratio of probability matching color green P=0.25: {l1/(l1+l2)}"
    )
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
allEvents.columns
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
            # print(prev_trial)
            # if prev_trial.empty:
            # print(df[df["trial"] == t]["trial"])

            if not prev_trial.empty:  # Ensure the previous trial exists
                # Assign trial direction from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "TD_prev",
                ] = prev_trial["trial_direction"].values[0]

                # print(prev_trial["trial_color_chosen"].values[0])
                # Assign color from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "color_prev",
                ] = prev_trial["trial_color_chosen"].values[0]
# %%
balance = df.groupby(["color", "sub", "proba"])["trial"].count().reset_index()
print(balance)
# %%
for sub in balance["sub"].unique():
    sns.barplot(x="proba", y="trial", hue="color", data=balance[balance["sub"] == sub])
    plt.title(f"Subject {sub}")
    plt.show()
# %%
df
# %%
df[(df["TD_prev"].isna())]

# %%
df = df[~(df["TD_prev"].isna())]
# %%
df.TD_prev
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
# %%
df = df[(df["sub"] != 9)]
df = df[(df["sub"] != 15)]
# df = df[df["sub"].isin([1, 2, 5, 7, 8, 11, 13])]
# df = df[(df["aSPoff"] <= 120)]
# %%
df.columns
# %%
colors = ["Green", "Red"]
# %%
# dd = df.groupby(["sub", "color", "proba", "TD_prev"])[["aSPv"]].mean().reset_index()
dd = df.groupby(["sub", "color", "proba"])[["aSPv", "aSPoff"]].mean().reset_index()
dd[dd["aSPv"] == dd["aSPv"].min()]
# %%
df[
    (df["sub"] == 2)
    & (df["proba"] == 0.75)
    & (df["TD_prev"] == "left")
    & (df["color"] == "Green")
]["aSPv"].mean()


# %%
np.abs(dd.aSPv.values).max()
# %%
dd[np.abs(dd.aSPv.values) > 1]
# %%
aSPv = dd[dd.color == "Red"]["aSPv"]
proba = dd[dd.color == "Red"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation(Red): {correlation}, p-value: {p_value}")
# %%
correlation, p_value = pearsonr(aSPv, proba)
print(f"Pearson's correlation(Red): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.color == "Green"]["aSPv"]
proba = dd[dd.color == "Green"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (Green): {correlation}, p-value: {p_value}")

# %%
correlation, p_value = pearsonr(aSPv, proba)
print(f"Pearson's correlation(Green): {correlation}, p-value: {p_value}")
# %%
df.columns
# %%
for s in df["sub"].unique():
    sns.lmplot(
        data=df[(df["sub"] == s)],
        y="aSPv",
        hue="color",
        hue_order=colors,
        palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
        x="proba",
    )
    plt.title(f"Subject{s}")
    plt.show()

# %%
# Create a mapping of color names to numerical values
for s in df["sub"].unique():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df["sub"] == s], x="color", hue="proba", y="aSPv")
    plt.title(f"Subject {s}")
    plt.show()
# %%
color_map = {name: i for i, name in enumerate(df["color"].unique())}

# Add a new column with numerical values
df["color_numeric"] = df["color"].map(color_map)
for s in df["sub"].unique():
    sns.lmplot(
        data=df[(df["sub"] == s)],
        y="aSPv",
        hue="proba",
        x="color_numeric",
    )
    plt.title(f"Subject{s}")
    plt.show()

# %%
for c in df["color"].unique():
    sns.lmplot(data=df[(df["color"] == c)], x="aSPv", hue="sub", y="proba", height=10)
    plt.title(f"Color {c}")
    plt.show()

# %%

# Create a list to store results
slope_data = []

# Loop through each unique color
for c in df["color"].unique():
    # Loop through each subject within this color
    for s in df[df["color"] == c]["sub"].unique():
        # Get data for this specific color and subject
        subset = df[(df["color"] == c) & (df["sub"] == s)]

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
                    "color": c,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                }
            )

# Convert to DataFrame
slope_df = pd.DataFrame(slope_data)

# If you want to merge this with your original dataframe
# First create a unique key for merging
df["color_sub"] = df["color"] + "_" + df["sub"].astype(str)
slope_df["color_sub"] = slope_df["color"] + "_" + slope_df["sub"].astype(str)
print(slope_df)
# %%
slopeG = slope_df[slope_df["color"] == "Green"]["slope"]
slopeR = slope_df[slope_df["color"] == "Red"]["slope"]

# Spearman's rank correlation
correlation, p_value = pearsonr(slopeG, slopeR)
print(
    f"Pearson's correlation(Slope Green, Slope Red): {correlation}, p-value: {p_value}"
)
# %%
aSPvG = (
    dd[(dd["color"] == "Green") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["color"] == "Green") & (dd["proba"] == 0.25)]["aSPv"].values
)
aSPvR = (
    dd[(dd["color"] == "Red") & (dd["proba"] == 0.75)]["aSPv"].values
    - dd[(dd["color"] == "Red") & (dd["proba"] == 0.25)]["aSPv"].values
)
# %%

# Spearman's rank correlation
correlation, p_value = pearsonr(slopeG, aSPvG)
print(f"Pearson's correlation(Slope Green, aSPvG): {correlation}, p-value: {p_value}")
# %%
# Extract slope values for Green and Red colors
green_slopes = slope_df[slope_df.color == "Green"]["slope"]
red_slopes = slope_df[slope_df.color == "Red"]["slope"]

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
plt.xlabel("Green Slope")
plt.ylabel("Red Slope")
# plt.title("Relationship Between Green and Red Condition Slopes")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")

# Make axes equal
plt.axis("equal")

# Show plot
plt.tight_layout()
plt.savefig(pathFig + "/linearRegressionSlopesFullProba.png",dpi=300, transparent=True)
plt.show()
# %%

aSPv = dd[dd["proba"] == 0.75]["aSPv"].values
color = dd[dd["proba"] == 0.75]["color"].values


encoder = LabelEncoder()
color_numeric = encoder.fit_transform(color)
# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 75): {correlation}, p-value: {p_value}")
correlation, p_value = pearsonr(aSPv, color_numeric)
print(f"Pearson's correlation(proba 75): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd["proba"] == 0.25]["aSPv"]
color = dd[dd["proba"] == 0.25]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 25): {correlation}, p-value: {p_value}")

correlation, p_value = pearsonr(aSPv, color_numeric)
print(f"Pearson's correlation(proba 25): {correlation}, p-value: {p_value}")

# %%

aSPv = dd[dd["proba"] == 0.50]["aSPv"]
color = dd[dd["proba"] == 0.50]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 50): {correlation}, p-value: {p_value}")

correlation, p_value = pearsonr(aSPv, color_numeric)
print(f"Pearson's correlation(proba 50): {correlation}, p-value: {p_value}")

# %%

# Friedman test for Red color
friedmantest=pg.friedman(data=df[df.color == "Red"], dv="aSPv", within="proba", subject="sub",method='f')
print(friedmantest)
# %%
pg.friedman(data=df[df.color == "Green"], dv="aSPv", within="proba", subject="sub",method='f')
# %%
# Wilcoxon Test to see whether the color has an effect within each proba
# It is the equivalent of paiRed t-test
pg.wilcoxon(
    x=dd[(dd.color == "Green") & (dd["proba"] == 0.50)].aSPv,
    y=dd[(dd.color == "Red") & (dd["proba"] == 0.50)].aSPv,
)
# %%

pg.wilcoxon(
    x=dd[(dd.color == "Green") & (dd["proba"] == 0.25)].aSPv,
    y=dd[(dd.color == "Red") & (dd["proba"] == 0.25)].aSPv,
)
# %%
pg.wilcoxon(
    x=dd[(dd.color == "Green") & (dd["proba"] == 0.75)].aSPv,
    y=dd[(dd.color == "Red") & (dd["proba"] == 0.75)].aSPv,
)
# %%
# Assuming dd is your DataFrame and it contains paired samples
green_values = dd[(dd.color == "Green") & (dd["proba"] == 0.75)].aSPv
red_values = dd[(dd.color == "Red") & (dd["proba"] == 0.75)].aSPv

# Perform the Wilcoxon signed-rank test
result = pg.wilcoxon(x=green_values, y=red_values, alternative="two-sided")

print(result)
# %%
dd
# %%
sns.boxplot(data=dd, hue="proba", x="color", y="aSPoff")
plt.show()
# %%
# Pivot the data for proba
pivot_proba = dd[dd["color"] == "Green"].pivot(
    index="sub", columns="proba", values="aSPv"
)
pivot_proba
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[0.25], pivot_proba[0.50], pivot_proba[0.75]
)
print(
    f"Friedman Test for proba: Statistic(Green) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# pos-hoc analysis

# Perform the Nemenyi Test
posthoc = sp.posthoc_nemenyi_friedman(pivot_proba.values)
print(posthoc)
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(posthoc, **heatmap_args)
plt.show()
# %%
# Perform the Friedman Test for proba
statistic_proba, p_value_proba = friedmanchisquare(
    pivot_proba[0.25], pivot_proba[0.50], pivot_proba[0.75]
)
print(
    f"Friedman Test for proba: Statistic(Green) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_proba = dd[dd.color == "Red"].pivot(index="sub", columns="proba", values="aSPv")
pivot_proba
# %%
# Perform the Friedman Test for proba
statpivot_proba, p_pivot_proba = friedmanchisquare(
    pivot_proba[0.25], pivot_proba[0.50], pivot_proba[0.75]
)
print(
    f"Friedman Test for proba: Statistic(Red) = {statpivot_proba}, p-value = {p_pivot_proba}"
)


# %%
# pos-hoc analysis

# Perform the Nemenyi Test
posthoc = sp.posthoc_nemenyi_friedman(pivot_proba.values)
print(posthoc)
# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"] == 0.25].pivot(index="sub", columns="color", values="aSPv")
pivot_color
# %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["Green"], pivot_color["Red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=25) = {statistic_color}, p-value = {p_value_color}"
)
# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"] == 0.50].pivot(index="sub", columns="color", values="aSPv")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["Green"], pivot_color["Red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=50) = {statistic_color}, p-value = {p_value_color}"
)

# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"] == 0.75].pivot(index="sub", columns="color", values="aSPv")
pivot_color

# a %%
# Perform the wilcoxon Test for color
statistic_color, p_value_color = wilcoxon(pivot_color["Green"], pivot_color["Red"])
print(
    f"Wilcoxon Test for color Statistic(P(Right|Red)=75) = {statistic_color}, p-value = {p_value_color}"
)



# %%
model = sm.OLS.from_formula("aSPv~ (proba) ", data=dd[dd.color == "Red"])
result = model.fit()

print(result.summary())
# %%
model = sm.OLS.from_formula("aSPv~ (proba) ", data=dd[dd.color == "Green"])
result = model.fit()

print(result.summary())
# %%
colors = ["Green", "Red"]
for s in df["sub"].unique():
    print(s)
    sns.displot(
        data=df[(df.proba == 0.25) & (df["sub"] == s)],
        x="aSPv",
        hue="color",
        hue_order=colors,
        alpha=0.5,
        # element="step",
        kind="kde",
        fill=True,
        # multiple="dodge",
        palette=colors,
    )
    plt.show()
# s%%
sns.displot(
    data=df[df.proba == 0.75],
    x="aSPv",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # element="step",
    kind="kde",
    fill=True,
    # multiple="dodge",
    palette=colors,
)
plt.show()
# %%
# Early trials
earlyTrials = 40
p = 0.25
sns.displot(
    data=df[(df.proba == p) & (df.trial <= earlyTrials)],
    x="aSPv",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    element="step",
    # multiple="dodge",
    palette=colors,
)
plt.title(f"Early Trials: {earlyTrials}, P(Right|Red)={p}")
plt.show()

# %%
# Mid trials
midTrials = [60, 180]
sns.histplot(
    data=df[(df.proba == p) & (df.trial <= midTrials[1]) & (df.trial > midTrials[0])],
    x="aSPv",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.title(f"Mid Trials: {midTrials[0]},{midTrials[1]}: P(Right|Red)={p}")
plt.show()
# %%
# Late trials
lateTrials = 200
sns.histplot(
    data=df[(df.proba == p) & (df.trial > lateTrials)],
    x="aSPv",
    hue="color",
    hue_order=colors,
    alpha=0.5,
    # multiple="dodge",
    palette=colors,
)
plt.title(f"Late Trials >{lateTrials}: P(Right|Red)={p}")
plt.show()
# %%
# Repeated measures ANOVA
# Perform mixed ANOVA
model = ols("aSPv ~ C(color)*(proba) ", data=dd).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# %%
anova_table = sm.stats.AnovaRM(
    data=dd, depvar="aSPv", within=["proba", "color"], subject="sub"
).fit()
print(anova_table.summary())
# %%
# checking the normality of the data
print(pg.normality(dd[dd.color == "Green"]["aSPv"]))
# %%
print(pg.normality(dd[dd.color == "Red"]["aSPv"]))
# %%
print(pg.normality(dd["aSPv"]))
# %%

x = dd[dd.color == "Red"]["aSPv"]
ax = pg.qqplot(x, dist="norm")
plt.show()
# %%
x = dd[dd.color == "Green"]["aSPv"]
ax = pg.qqplot(x, dist="norm")
plt.show()

# %%

# Set up the FacetGrid
facet_grid = sns.FacetGrid(
    data=df,
    col="proba",
    col_wrap=3,
    height=8,
    aspect=1.5,
)

# Create pointplots for each sub
facet_grid.map_dataframe(
    sns.histplot, x="aSPv", hue="color", palette=colors, hue_order=colors
)

# Add legends
facet_grid.add_legend()

# Set titles for each subplot
for ax, p in zip(facet_grid.axes.flat, df.proba.unique()):
    ax.set_title(f"Horizontal aSPv: P(Right|Red)=P(Left|Green)={p}")
    ax.legend(["Red", "Green"])
# Adjust spacing between subplots
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()
# %%
# Perform mixed repeated measures ANOVA
anova_results = pg.rm_anova(
    dv="aSPv",
    within=["proba", "color"],
    subject="sub",
    data=dd,
)

print(anova_results)
# %%
# Perform mixed repeated measures ANOVA
anova_results = pg.rm_anova(
    dv="aSPv",
    within="proba",
    subject="sub",
    data=dd[dd["color"] == "Green"],
)

print(anova_results)
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within="proba",
    subject="sub",
    data=dd[dd.color == "Red"],
)

print(anova_results)
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within="color",
    subject="sub",
    data=dd[dd["proba"] == 0.75],
)

print(anova_results)

# %%
dd[(dd["proba"] == 0.75) & (dd["color"] == "Green")]
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.75) & (dd["color"] == "Red")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["color"] == "Green")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within="color",
    subject="sub",
    data=dd[dd["proba"] == 0.5],
)

print(anova_results)

# %%

ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.5) & (dd["color"] == "Red")]["aSPv"],
    y=dd[(dd["proba"] == 0.5) & (dd["color"] == "Green")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within="color",
    subject="sub",
    data=dd[dd["proba"] == 0.25],
)

print(anova_results)

# %%

ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["color"] == "Red")]["aSPv"],
    y=dd[(dd["proba"] == 0.25) & (dd["color"] == "Green")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["color"] == "Red")]["aSPv"],
    y=dd[(dd["proba"] == 0.50) & (dd["color"] == "Red")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.5) & (dd["color"] == "Red")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["color"] == "Red")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["color"] == "Red")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["color"] == "Red")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
#Green Color
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["color"] == "Green")]["aSPv"],
    y=dd[(dd["proba"] == 0.50) & (dd["color"] == "Green")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.5) & (dd["color"] == "Green")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["color"] == "Green")]["aSPv"],
    paired=True,
)
print(ttest_results)
# %%
ttest_results = pg.ttest(
    x=dd[(dd["proba"] == 0.25) & (dd["color"] == "Green")]["aSPv"],
    y=dd[(dd["proba"] == 0.75) & (dd["color"] == "Green")]["aSPv"],
    paired=True,
)

print(ttest_results)
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=dd,
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="se",
    n_boot=10000,
    hue="color",
    hue_order=colors,
    palette=colors,
)
# _ = plt.title("Horizontal aSPv Across probabilities", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|Red)=$\mathbb{P}$(Left|Green)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvAcrossprobapp.png",dpi=300, transparent=True)
plt.show()
# %%
sns.catplot(
    data=dd,
    x="proba",
    y="aSPv",
    hue="color",
    hue_order=colors,
    kind="violin",
    split=True,
    palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
    inner="quart",
    fill=False,
    cut=0,
)
plt.savefig(pathFig + "/aSPvAcrossprobaviolin.png",dpi=300, transparent=True)
plt.show()
# %%
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.color == "Red"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    palette="tab20",
    alpha=0.8,
)
# _ = plt.title("Horizontal aSPv Per Subject: color Red", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel(r"$\mathbb{P}$(Right|Red)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsRed.png",dpi=300, transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

# Get the points for probability 0.25
green_25 = dd[(dd["color"] == "Green") & (dd["proba"] == 0.25)]["aSPv"].values
red_25 = dd[(dd["color"] == "Red") & (dd["proba"] == 0.25)]["aSPv"].values

# Get the points for probability 0.75
green_75 = dd[(dd["color"] == "Green") & (dd["proba"] == 0.75)]["aSPv"].values
red_75 = dd[(dd["color"] == "Red") & (dd["proba"] == 0.75)]["aSPv"].values

# Plot the scatter points
plt.scatter(green_25, red_25, label="proba=0.25")
plt.scatter(green_75, red_75, label="proba=0.75")

# Add connecting lines between corresponding points
for i in range(len(green_25)):
    plt.plot(
        [green_25[i], green_75[i]],
        [red_25[i], red_75[i]],
        color="gray",
        alpha=0.3,
        linestyle="-",
    )

plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.xlabel(r"aSPv Color Green", fontsize=30)
plt.ylabel(r"aSPv Color Red", fontsize=30)
plt.legend()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

# Get the points for probability 0.25
green_25 = dd[(dd["color"] == "Green") & (dd["proba"] == 0.25)]["aSPv"].values
red_25 = dd[(dd["color"] == "Red") & (dd["proba"] == 0.25)]["aSPv"].values

# Get the points for probability 0.75
green_75 = dd[(dd["color"] == "Green") & (dd["proba"] == 0.75)]["aSPv"].values
red_75 = dd[(dd["color"] == "Red") & (dd["proba"] == 0.75)]["aSPv"].values

# Plot the scatter points
plt.scatter(green_25, red_25, label="proba=0.25")
plt.scatter(green_75, red_75, label="proba=0.75")

# Add connecting lines between corresponding points and index labels
for i in range(len(green_25)):
    # Draw the line
    plt.plot(
        [green_25[i], green_75[i]],
        [red_25[i], red_75[i]],
        color="gray",
        alpha=0.3,
        linestyle="-",
    )

    # Add index number at the middle of the line
    mid_x = (green_25[i] + green_75[i]) / 2
    mid_y = (red_25[i] + red_75[i]) / 2
    plt.annotate(str(i + 1), (mid_x, mid_y), fontsize=12)

plt.axhline(y=0, color="k", linestyle="--")  # Horizontal line at y=0
plt.axvline(x=0, color="k", linestyle="--")  # Vertical line at x=0
plt.xlabel(r"aSPv Color Green", fontsize=30)
plt.ylabel(r"aSPv Color Red", fontsize=30)
plt.legend()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
# %%
pg.normality(dd['aSPv'])
# %%
anova_results = pg.rm_anova(
    dv="aSPv",
    within=["proba", "color"],
    subject="sub",
    data=dd,
)

print(anova_results)
# %%
model = smf.mixedlm(
    "aSPv~C(proba,Treatment(0.5))*C(color,Treatment('Red'))",
    data=df,
    re_formula="~proba*color",
    groups=df["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~proba*color",
    data=df,
    re_formula="~proba*color",
    groups=df["sub"],
).fit()
model.summary()
# %%
# Fixed effects
fe = model.fe_params
intc = fe['Intercept']
slope = fe['proba']
color_effect = fe['color[T.Red]']
interaction = fe['proba:color[T.Red]']

# Random effects
re_df = pd.DataFrame(model.random_effects).T
re_df.columns = ['Intercept_re', 'proba_re', 'color[T.Red]_re', 'proba:color[T.Red]_re']
re_df['sub'] = re_df.index
# %%
# Proba range
proba_range = np.linspace(df['proba'].min(), df['proba'].max(), 50)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

for _, row in re_df.iterrows():
    # Subject-level random intercept and slope
    i_re = row['Intercept_re']
    s_re = row['proba_re']

    # Green (reference)
    y_green = (intc + i_re) + (slope + s_re) * proba_range
    ax.plot(proba_range, y_green, color='green', alpha=0.3)

    # Red (with fixed and random adjustments)
    y_red = (intc + color_effect + i_re) + (slope + interaction + s_re) * proba_range
    ax.plot(proba_range, y_red, color='red', alpha=0.3)

# Fixed effect mean lines (thick black)
mean_green = intc + slope * proba_range
mean_red = (intc + color_effect) + (slope + interaction) * proba_range
ax.plot(proba_range, mean_green, color='black', linewidth=3, label='Green mean')
ax.plot(proba_range, mean_red, color='black', linewidth=3, linestyle='--', label='Red mean')

# Labels and legend
ax.set_xlabel('proba')
ax.set_ylabel('aSPv')
ax.set_title('Random Slopes and Intercepts by Color')
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Extract fixed effects
fe = model.fe_params
base_slope = fe['proba']  # base slope (for green)
interaction = fe['proba:color[T.Red]']  # additional slope for red

# Extract random effects
re_dict = model.random_effects

# Create an empty dataframe for subject-specific slopes
participant_slopes = []

# Loop through each subject
for sub, re in re_dict.items():
    # The random effects order depends on the model specification
    # Typically for re_formula="~proba*color"
    # The order is [intercept, color, proba, proba:color]
    # You may need to verify this by inspecting re
    
    proba_re = re[2]  # Random effect for proba (green condition)
    interaction_re = re[3]  # Random effect for proba:color interaction
    
    green_slope = base_slope + proba_re
    red_slope = base_slope + interaction + proba_re + interaction_re
    
    participant_slopes.append({
        'sub': sub,
        'green_slope': green_slope,
        'red_slope': red_slope,
        'avg_abs_slope': (abs(green_slope) + abs(red_slope)) / 2
    })

# Convert to dataframe
participant_slopes_df = pd.DataFrame(participant_slopes)
print(participant_slopes_df)


# %%
# Add the regression plot her
# %%

# residuals = model.resid
#
# # Q-Q plot
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q plot of residuals")
# plt.show()
# # %%
# pg.qqplot(residuals, dist="norm")
# plt.show()
# # %%
# # Histogram
# plt.hist(residuals, bins=50)
# plt.title("Histogram of residuals")
# plt.show()
# %%
model = smf.mixedlm(
    "aSPv~proba",
    data=df[df.color == "Red"],
    re_formula="~proba",
    groups=df[df.color == "Red"]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~proba",
    data=df[df.color == "Green"],
    re_formula="~proba",
    groups=df[df.color == "Green"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C(proba,Treatment(0.5))",
    data=df[df.color == "Red"],
    re_formula="~proba",
    groups=df[df.color == "Red"]["sub"],
).fit()
model.summary()

# %%
model = smf.mixedlm(
    "aSPv~C(proba,Treatment(0.5))",
    data=df[df.color == "Green"],
    re_formula="~proba",
    groups=df[df.color == "Green"]["sub"],
).fit()
model.summary()

# %%
residuals = model.resid

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()

# Histogram
plt.hist(residuals, bins=50)
plt.title("Histogram of residuals")
plt.show()
# %%
stat, p = stats.kstest(residuals, "norm")

print(f"KS test statistic: {stat:.4f}")
print(f"KS test p-value: {p:.4f}")
normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
# %%
model = smf.mixedlm(
    "aSPv~ color",
    data=df[df.proba == 0.25],
    re_formula="~color",
    groups=df[df.proba == 0.25]["sub"],
).fit()
model.summary()
# %%
residuals = model.resid

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()

# Histogram
plt.hist(residuals, bins=50)
plt.title("Histogram of residuals")
plt.show()

normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
# %%
model = smf.mixedlm(
    "aSPv~ color",
    data=df[df.proba == 0.50],
    re_formula="~color",
    groups=df[df.proba == 0.50]["sub"],
).fit()
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~ C(color)",
    data=df[df.proba == 0.75],
    re_formula="~color",
    groups=df[df.proba == 0.75]["sub"],
).fit()
model.summary()

# %%

# %%
dd = df.groupby(["sub", "proba", "color"])[["aSPv"]].mean().reset_index()
# %%
# fig = plt.figure()
# # Toggle full screen mode
# figManager = plt.get_current_fig_manager()
# figManager.full_screen_toggle()
g=sns.catplot(
    x="proba",
    y="aSPv",
    hue="color",
    errorbar=("ci", 95),
    n_boot=1000,
    kind='bar',
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
    hue_order=colors,
    fill=False,
    legend=False,
    data=df,
    # alpha=0.5,
)
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="color",
    data=dd,
    dodge=True,
    palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
    jitter=True,
    size=6,
    # alpha=0.5,
    legend=False,
)

order=[0.25,0.5,0.75]
hue_order=["Green", "Red"]
hue='color'
pairs = [
    ((0.25, "Green"), (0.25, "Red")),
    ((0.5, "Green"), (0.5, "Red")),
    ((0.75, "Green"), (0.75, "Red")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator = Annotator(g.ax, pairs, data=dd, x='proba', y="aSPv", hue=hue,hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside')
annotator.apply_and_annotate()
legend_elements = [
    Patch(facecolor=GreencolorsPalette[1], alpha=1, label="Green"),
    Patch(facecolor=RedcolorsPalette[1], alpha=1, label="Red"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Color", title_fontsize=20
)
# plt.title("Horizontal aSPv across 3 different probabilites", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Red)=$\mathbb{P}$(Left|Green)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.ylim(-0.75, 0.75)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvcolors.png",dpi=300, transparent=True)
plt.show()

# %%

g=sns.catplot(
    data=df,
    x="color",
    y="aSPv",
    hue="proba",
    kind="bar",
    errorbar=("ci", 95),
    # errorbar='se',
    n_boot=10000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=[0.25,0.5,0.75],
    fill=False,
    legend=False,
    palette='viridis',
)

sns.stripplot(
    x="color",
    y="aSPv",
    hue="proba",
    data=dd,
    dodge=True,
    palette='viridis',
    jitter=True,
    size=6,
    linewidth=1,
    # alpha=0.5,
    # legend=False,
)
hue_order=[0.25,0.5,0.75]
order=["Red", "Green"]
pairs = [
    (("Green",0.25), ('Green',0.50)),
    (("Green",0.75), ('Green',0.50)),
    (("Green",0.25), ('Green',0.75)),

    (("Red",0.25), ('Red',0.50)),
    (("Red",0.75), ('Red',0.50)),
    (("Red",0.25), ('Red',0.75)),

]
annotator = Annotator(g.ax, pairs, data=dd, x='color', y="aSPv", hue="proba",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired',  loc='outside',fontsize=20,comparisons_correction='HB')
annotator.apply_and_annotate()
plt.xlabel(r"Cue", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

viridis_colors = plt.cm.viridis([0, 0.5, 1])  # Maps to your 0.25, 0.50, 0.75 proba values

legend_elements = [
    Patch(facecolor=viridis_colors[0], edgecolor='black', label='0.25'),
    Patch(facecolor=viridis_colors[1], edgecolor='black', label='0.50'),
    Patch(facecolor=viridis_colors[2], edgecolor='black', label='0.75')
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title=r"$\mathbb{P}$(Right|Red)=$\mathbb{P}$(Left|Green)", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvfirstSegsFullProbabis.png",dpi=300, transparent=True)
plt.show()
# %%

figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.boxplot(
    x="proba",
    y="aSPv",
    hue="color",
    # errorbar="ci",
    palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
    hue_order=colors,
    fill=False,
    data=dd,
    # alpha=0.5,
)
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="color",
    data=dd,
    dodge=True,
    palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
    jitter=True,
    size=6,
    # alpha=0.5,
    legend=False,
)

plt.legend(fontsize=20, title="Color", title_fontsize=20)
# plt.title("Horizontal aSPv across 3 different probabilites", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Red)=$\mathbb{P}$(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.ylim(-0.75, 0.75)
plt.legend(fontsize=20)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvcolorsbp.png",dpi=300, transparent=True)
plt.show()
# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "color",
        "TD_prev",
        "aSPv",
    ]
]
df_prime.groupby(["sub", "proba", "color", "TD_prev"]).count()[["aSPv"]]

# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    color=RedcolorsPalette[1],
    data=df[df["color"] == "Red"],
)
# plt.title("Anticipatory Smooth Eye Movement: color Red", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Red)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvRed.png",dpi=300, transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    color=GreencolorsPalette[1],
    errorbar="ci",
    data=df[df.color == "Green"],
)
# plt.title("Anticipatory Smooth Eye Movement: color Green", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.savefig(pathFig + "/aSPvGreen.png",dpi=300, transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "color", "proba", "TD_prev"])[["aSPv"]].mean().reset_index()
# %%
# Create the plot using catplot
g = sns.catplot(
    data=df[df.color == "Green"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[GreencolorsPalette[1]],  # Same color for both
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
    bar.set_edgecolor(GreencolorsPalette[1])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[GreencolorsPalette[1]],
    dodge=True,
    jitter=True,
    linewidth=1,
    size=6,
    # alpha=0.7,
    marker="o",
    data=dd[dd.color == "Green"],
    legend=False,
)

order=[0.25,0.5,0.75]
hue_order=["left", "right"]
hue='TD_prev'
pairs = [
    ((0.25, "left"), (0.25, "right")),
    ((0.5, "left"), (0.5, "right")),
    ((0.75, "left"), (0.75, "right")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator = Annotator(g.ax, pairs,    data=dd[dd.color == "Green"], x='proba', y="aSPv", hue=hue,hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()

# Create custom legend

legend_elements = [
    Patch(facecolor=GreencolorsPalette[1], alpha=1, label="Left"),
    Patch(
        facecolor="none", hatch="///", label="Right", edgecolor=GreencolorsPalette[1]
    ),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize="20"
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: color Green ", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvGreenTD.png",dpi=300, transparent=True)
plt.show()
# %%
g = sns.catplot(
    data=df[df.color == "Red"],
    x="proba",
    y="aSPv",
    hue="TD_prev",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=[RedcolorsPalette[1]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
    # zorder=2,
)

# Add hatching to the right bars
for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
    bar.set_facecolor("none")  # Make bar empty
    bar.set_hatch("///")  # Add diagonal lines
    bar.set_edgecolor(RedcolorsPalette[1])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[RedcolorsPalette[1]],  # Set the color to gray
    # palette=["white"],  # Set the color to gray
    dodge=True,
    jitter=True,
    size=6,
    linewidth=1,
    marker="o",
    # edgecolor="gray",
    # facecolor="none",
    # alpha=0.7,  # You can adjust the transparency if needed
    data=dd[dd.color == "Red"],
    legend=False,
    # zorder=1,
)

order=[0.25,0.5,0.75]
hue_order=["left", "right"]
hue='TD_prev'
pairs = [
    ((0.25, "left"), (0.25, "right")),
    ((0.5, "left"), (0.5, "right")),
    ((0.75, "left"), (0.75, "right")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator = Annotator(g.ax, pairs,    data=dd[dd.color == "Green"], x='proba', y="aSPv", hue=hue,hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()


# Create custom legend

legend_elements = [
    Patch(facecolor=RedcolorsPalette[1], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=RedcolorsPalette[1]),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous TD", title_fontsize="20"
)

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: Color Red ", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Red)", fontsize=30)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvRedTD.png",dpi=300, transparent=True)
plt.show()

# %%
# %%
df[(df['sub']==1)&(df['proba']==0.25)&(df['trial']==2)][['aSPon','aSPoff','aSPv']]
# %%
dd[
    (dd["color"] == "Red")
    & (dd["TD_prev"] == "left")
    & (dd["proba"] == 0.25)
    & (dd["aSPv"] > 0)
]
# %%
df[
    (df["color"] == "Red")
    & (df["TD_prev"] == "left")
    & (df["proba"] == 0.25)
    & (df["sub"] == 12)
]["aSPv"].mean()

# %%
df["interaction"] = list(zip(df["TD_prev"].values, df["color_prev"].values))
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "color",
        "interaction",
        "aSPv",
    ]
]
df_prime
# %%

learningCurveInteraction = (
    df_prime.groupby(["sub", "proba", "interaction", "color"])[["aSPv"]]
    .mean()
    .reset_index()
)
# %%
learningCurveInteraction[
    learningCurveInteraction["aSPv"] == learningCurveInteraction["aSPv"].min()
]
# %%

df_prime[
    (df_prime["sub"] == 15)
    & (df_prime["color"] == "Green")
    & (df_prime["proba"] == 0.5)
    & (df_prime["interaction"] == ("right", "green"))
]["aSPv"]
# %%
df_prime[
    (df_prime["sub"] == 15)
    & (df_prime["color"] == "Green")
    & (df_prime["proba"] == 0.5)
]["interaction"].value_counts()
# %%
df_prime.groupby(["sub", "proba", "interaction", "color"]).count()[["aSPv"]]
# %%
df_prime.groupby(["proba", "interaction", "color"]).count()[["aSPv"]]
# %%
# cc = df_prime.groupby(["sub", "proba", "interaction", "color"]).count()[["aSPv"]]
# sns.barplot(data=cc, x="sub", y="aSPv", hue="interaction")
# plt.show()
# %%
learningCurveInteraction["interaction"].unique()

# %%
# Cmap for Green and Red for the interaction plots

df_prime["interaction"].unique()
# %%
hue_order = [
    ("left", "green"),
    ("left", "red"),
    ("right", "green"),
    ("right", "red"),
]
# %%
RedcolorsPalette = ["#e83865", "#cc3131"]
GreencolorsPalette = ["#8cd790", "#285943"]
# colorsPalette = ["#285943", "#cc3131", "#e83865", "#8cd790"]
colorsPalette = [
    GreencolorsPalette[1],
    RedcolorsPalette[1],
    GreencolorsPalette[1],
    RedcolorsPalette[1],
]
# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.color == "Red"],
    x="proba",
    y="aSPv",
    hue="interaction",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=colorsPalette,
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
        bar.set_edgecolor(colorsPalette[i // n_x_values])  # Maintain the category color

# Add stripplot
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    hue_order=hue_order,
    palette=colorsPalette,
    dodge=True,
    jitter=True,
    marker="o",
    linewidth=1,
    size=6,
    data=learningCurveInteraction[learningCurveInteraction.color == "Red"],
    # legend=False,
)
order=[0.25,0.5,0.75]

hue_order = [
    ("left", "green"),
    ("left", "red"),
    ("right", "green"),
    ("right", "red"),
]
hue='interaction'
pairs = [
    # ((0.25, ("left", "green")), (0.25, ("left", "red"))),
    # ((0.5, ("left", "green")), (0.5, ("left", "red"))),
    # ((0.75, ("left", "green")), (0.75, ("left", "red"))),
    # ((0.25, ("right", "green")), (0.25, ("right", "red"))),
    # ((0.5, ("right", "green")), (0.5, ("right", "red"))),
    # ((0.75, ("right", "green")), (0.75, ("right", "red"))),
    ((0.25, ("right", "red")), (0.25, ("left", "red"))),
    ((0.5, ("right", "red")), (0.5, ("left", "red"))),
    ((0.75, ("right", "red")), (0.75, ("left", "red"))),
    # ((0.5, "left"), (0.5, "right")),
    # ((0.75, "left"), (0.75, "right")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator.configure(test='t-test_paired', text_format='star', loc='outside',comparisons_correction="HB")
annotator.configure(test='t-test_paired', text_format='star', loc='outside',comparisons_correction="HB")
annotator.apply_and_annotate()


# # Create custom legend with all four categories
legend_elements = [
    # Left categories (solid fill)
    Patch(facecolor=colorsPalette[0], alpha=1, label="Left, Green"),
    Patch(facecolor=colorsPalette[1], alpha=1, label="Left, Red"),
    # Right categories (hatched)
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalette[2],
        label="Right, Green",
    ),
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalette[3],
        label="Right, Red",
    ),
]

# Add the legend
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous Trial", title_fontsize=20
)

# Customize the plot
g.ax.set_title("Red Trials:\n Previous TD and Color", fontsize=30)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Red)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvRedInteraction.png",dpi=300, transparent=True)
plt.show()
# %%
# Create the base plot
g = sns.catplot(
    data=df_prime[df_prime.color == "Green"],
    x="proba",
    y="aSPv",
    hue="interaction",
    kind="bar",
    errorbar=("ci", 95),
    n_boot=1000,
    palette=colorsPalette,
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
        bar.set_edgecolor(colorsPalette[i // n_x_values])  # Maintain the category color

# Add stripplot
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    hue_order=hue_order,
    palette=colorsPalette,
    dodge=True,
    jitter=True,
    size=6,
    linewidth=1,
    data=learningCurveInteraction[learningCurveInteraction.color == "Green"],
    # legend=False,
)

# # Create custom legend with all four categories
legend_elements = [
    # Left categories (solid fill)
    Patch(facecolor=colorsPalette[0], alpha=1, label="Left, Green"),
    Patch(facecolor=colorsPalette[1], alpha=1, label="Left, Red"),
    # Right categories (hatched)
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalette[2],
        label="Right, Green",
    ),
    Patch(
        facecolor="none",
        hatch="///",
        edgecolor=colorsPalette[3],
        label="Right, Red",
    ),
]

# Add the legend
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Previous Trial", title_fontsize=20
)
plt.title(
    "Green Trials\n Previous Target Direction & Color",
    fontsize=30,
)
plt.xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
g.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvGreenInteraction.png",dpi=300, transparent=True)
plt.show()
# %%
learningCurveInteraction[
    learningCurveInteraction["aSPv"] == learningCurveInteraction["aSPv"].min()
]
# %%
plt.hist(
    df[
        (df["sub"] == 15)
        & (df["interaction"] == ("right", "green"))
        & (df["proba"] == 0.5)
    ]["aSPv"].values
)
np.mean(
    df[
        (df["sub"] == 15)
        & (df["interaction"] == ("right", "green"))
        & (df["proba"] == 0.5)
    ]["aSPv"].values
)
# %%
model = smf.mixedlm(
    "aSPv~  C(color)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.25]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
# model = smf.mixedlm(
#     "aSPv~ C( proba,Treatment(0.5) )* C(color,Treatment('Red'))*C(TD_prev)",
#     data=df,
#     re_formula="~color*proba",
#     groups=df["sub"],
# ).fit(method="lbfgs")
# model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(color,Treatment) + C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(color)*C(TD_prev)",
    data=df[df.proba == 0.50],
    re_formula="~TD_prev",
    groups=df[df.proba == 0.50]["sub"],
).fit(method=["lbfgs"])
model.summary()
# %%
df.color_prev
# %%
# Sampling Bias analysis

# Define transition counts for previous state = Green
Green_transitions = (
    df[df["color_prev"] == "green"]
    .groupby(["sub", "proba", "color"])["aSPv"]
    .count()
    .reset_index(name="count")
)
Green_transitions["total"] = Green_transitions.groupby(["sub", "proba"])[
    "count"
].transform("sum")

Green_transitions["conditional_prob"] = (
    Green_transitions["count"] / Green_transitions["total"]
)

Green_transitions = Green_transitions.rename(columns={"color": "current_state"})
Green_transitions["previous_state"] = "green"


# Define transition counts for previous state = Red
Red_transitions = (
    df[df["color_prev"] == "red"]
    .groupby(["sub", "proba", "color"])["aSPv"]
    .count()
    .reset_index(name="count")
)
Red_transitions["total"] = Red_transitions.groupby(["sub", "proba"])["count"].transform(
    "sum"
)
Red_transitions["conditional_prob"] = (
    Red_transitions["count"] / Red_transitions["total"]
)
Red_transitions = Red_transitions.rename(columns={"color": "current_state"})
Red_transitions["previous_state"] = "red"
# %%
# Combine results
conditional_probabilities = pd.concat([Green_transitions, Red_transitions])
conditional_probabilities
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
conditional_probabilities


# %%
for s in conditional_probabilities["sub"].unique():
    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=conditional_probabilities[conditional_probabilities["sub"] == s],
        col="proba",
        col_wrap=3,
        height=8,
        aspect=2,
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
        ax.set_ylabel("Conditional probability")
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
                group[group["transition_state"] == "('Green', 'green')"][
                    "conditional_prob"
                ]
            )
            > 0
        ):
            Green_to_Green = group[group["transition_state"] == "('Green', 'green')"][
                "conditional_prob"
            ].values[0]
        else:
            Green_to_Green = 0

        if (
            len(
                group[group["transition_state"] == "('Red', 'red')"]["conditional_prob"]
            )
            > 0
        ):
            Red_to_Red = group[group["transition_state"] == "('Red', 'red')"][
                "conditional_prob"
            ].values[0]
        else:
            Red_to_Red = 0

        if (
            len(
                group[group["transition_state"] == "('Red', 'green')"][
                    "conditional_prob"
                ]
            )
            > 0
        ):
            Green_to_Red = group[group["transition_state"] == "('Red', 'green')"][
                "conditional_prob"
            ].values[0]
        else:
            Green_to_Red = 0

        if len(
            group[group["transition_state"] == "('Green', 'red')"]["conditional_prob"]
        ):

            Red_to_Green = group[group["transition_state"] == "('Green', 'red')"][
                "conditional_prob"
            ].values[0]
        else:
            Red_to_Green = 0

        # Persistent: high probability of staying in the same state
        if Green_to_Green > 0.6 and Red_to_Red > 0.6:
            return "Persistent"

        # Alternating: high probability of switching states
        if Green_to_Red > 0.6 and Red_to_Green > 0.6:
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
    plt.title("Subject Behavior Classification\n(Consistent Across probabilities)")
    plt.show()

    # Print detailed results
    print(subject_classification)

    # Additional detailed view
    detailed_behavior = subject_proba_behavior.pivot_table(
        index="sub", columns="proba", values="behavior", aggfunc="first"
    )
    print("\nDetailed Behavior Across probabilities:")
    print(detailed_behavior)

    return subject_classification


subject_classification = classify_subject_behavior(conditional_probabilities)
# %%
subject_classification

# %%
# Perform classification
# Optional: Create a more detailed summary
summary = subject_classification.groupby("behavior_class").size()
print("\nBehavior Classification Summary:")
print(summary)
# %%
dd = df.groupby(["sub", "proba", "color"])["aSPv"].mean().reset_index()
# %%
for s in dd["sub"].unique():
    behavior_value = subject_classification[subject_classification["sub"] == s][
        "behavior_class"
    ].values[0]
    dd.loc[dd["sub"] == s, "behavior"] = behavior_value

# %%
dd
# %%
sns.lmplot(
    data=dd[(dd["color"] == "Green")],
    x="proba",
    y="aSPv",
    hue="behavior",
)
plt.show()
# %%
sns.lmplot(
    data=dd[(dd["color"] == "Red")],
    x="proba",
    y="aSPv",
    hue="behavior",
)
plt.show()

# %%
dd
# %%
# Computing the peristance score based on the transition probabilites
# One should expect a U shape fit here
for p in df["proba"].unique():
    ddp = dd[dd["proba"] == p].copy()
    sns.regplot(
        data=ddp,
        x=ddp[ddp["color"] == "Green"]["aSPv"].values,
        y=ddp[ddp["color"] == "Red"]["aSPv"].values,
    )
    plt.ylabel("adaptation_Red", fontsize=20)
    plt.xlabel("adaptation_Green", fontsize=20)
    plt.title(f"proba={p}")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
# %%
conditional_probabilities
# %%
dd = (
    conditional_probabilities.groupby(["sub", "transition_state", "proba"])[
        "conditional_prob"
    ]
    .mean()
    .reset_index()
)
dd
# %%
dd["sub"].value_counts()
# %%
ts = dd["transition_state"].unique()
new_rows = []
for s in dd["sub"].unique():
    for p in dd[dd["sub"] == s]["proba"].unique():
        existing_ts = dd[(dd["sub"] == s) & (dd["proba"] == p)][
            "transition_state"
        ].unique()
        for t in ts:
            if t not in existing_ts:
                # Add a new row with sub = s and transition_state = t, setting transition_state to 0
                new_row = {
                    "sub": s,
                    "proba": p,
                    "transition_state": t,
                    "conditional_prob": 0,
                }
                new_rows.append(new_row)

# Concatenate the new rows to the original DataFrame
dd = pd.concat([dd, pd.DataFrame(new_rows)], ignore_index=True)

print(dd)
# %%

dd = (
    dd.groupby(["sub", "proba", "transition_state"])["conditional_prob"]
    .mean()
    .reset_index()
)
dd
# %%
dd["transition_state"].unique()

# %%
dd["sub"].value_counts()


# %%
# Function to classify transition_state as persistent or alternating
def classify_transition(state):
    return (
        "persistent"
        if state == "('Red', 'red')" or state == "('Green', 'green')"
        else "alternating"
    )


# Apply the classification function
dd["transition_type"] = dd["transition_state"].apply(classify_transition)
dd
# %%
adaptation = (
    dd.groupby(["sub", "transition_type"])["conditional_prob"].mean().reset_index()
)
adaptation
# %%
# Group by 'sub' and calculate the score
result = pd.DataFrame()
result["sub"] = df["sub"].unique()
# %%
result["persistence_score"] = (
    adaptation[adaptation["transition_type"] == "persistent"]["conditional_prob"].values
    - adaptation[adaptation["transition_type"] == "alternating"][
        "conditional_prob"
    ].values
)
result
# %%

# Assuming you already have participant_slopes_df from the previous step
x = participant_slopes_df['green_slope']  # Green slopes on x-axis
y = participant_slopes_df['red_slope']    # Red slopes on y-axis

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, color='purple', alpha=0.7, s=100)

# Add labels for each point (subject ID)
for i, txt in enumerate(participant_slopes_df['sub']):
    plt.annotate(txt, (x.iloc[i], y.iloc[i]), fontsize=10, 
                 xytext=(5, 5), textcoords='offset points')

# Fit a linear regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create line of best fit
x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + intercept

# Plot the regression line
plt.plot(x_line, y_line, color='blue', linestyle='--')

# Add regression details as text at the bottom left
plt.text(0.05, 0.05, f'y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r_value**2:.3f}\np = {p_value:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

# Add labels and title
plt.xlabel('Green Slope', fontsize=14)
plt.ylabel('Red Slope', fontsize=14)
plt.title('Relationship Between Participant-Specific Slopes for Green and Red Conditions', fontsize=16)

# Add horizontal and vertical lines at 0
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Center the plot on (0,0)
x_min = -4
x_max = 4
y_min = -4
y_max = 4
# Ensure equal scale on both axes
max_range = max(x_max - x_min, y_max - y_min) / 2
plt.xlim(-max_range, max_range)
plt.ylim(-max_range, max_range)

# Add grid
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
participant_slopes_df
# %%
result['adaptation']=participant_slopes_df["avg_abs_slope"].values
result
# %%

# Create the visualization with lmplot
sns.lmplot(data=result, x="persistence_score", y="adaptation", height=10)
plt.ylabel("adaptation", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Calculate the slope and p-value using scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(
    result["persistence_score"], 
    result["adaptation"]
)

# Print the results
print(f"Slope: {slope:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"Standard Error: {std_err:.4f}")
print(f"Intercept: {intercept:.4f}")

# Optionally add annotation to the plot
equation = f"y = {slope:.4f}x + {intercept:.4f}\np = {p_value:.4f}, R² = {r_value**2:.4f}"
plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.savefig(pathFig + "/samplingBiasColours.png", dpi=300, transparent=True)
plt.show()
# %%
dd["transition_state"].unique()
# %%
result["persistence_score_Red"] = np.nan
result["persistence_score_Green"] = np.nan
result
# %%
dd
# %%

for s in dd["sub"].unique():
    Red_Red_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Red', 'Red')")][
            "conditional_prob"
        ]
    )
    Green_Red_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Green', 'Red')")][
            "conditional_prob"
        ]
    )

    result.loc[result["sub"] == s, "persistence_score_Red"] = (
        Red_Red_prob - Green_Red_prob
    )

    Green_Green_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Green', 'Green')")][
            "conditional_prob"
        ]
    )

    Red_Green_prob = np.mean(
        dd[(dd["sub"] == s) & (dd["transition_state"] == "('Red', 'Green')")][
            "conditional_prob"
        ]
    )
    result.loc[result["sub"] == s, "persistence_score_Green"] = (
        Green_Green_prob - Red_Green_prob
    )
result
# %%

# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "color"])["aSPv"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "color"], columns="proba", values="aSPv"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = (
    np.abs(pivot_table[0.75] + pivot_table[0.25] - 2 * pivot_table[0.50]) / 2
)
# %%
print(pivot_table)
# %%

# Select the relevant columns
adaptation = pivot_table[["sub", "color", "adaptation"]]

print(adaptation)
# %%

result["adaptation"] = adaptation.groupby("sub")["adaptation"].mean().values
result["adaptation_Green"] = adaptation[adaptation["color"] == "Green"][
    "adaptation"
].values
result["adaptation_Red"] = adaptation[adaptation["color"] == "Red"]["adaptation"].values
# %%
result = pd.DataFrame(result)
result
# %%
sns.lmplot(data=result, x="persistence_score", y="adaptation", height=10)
plt.ylabel("adaptation", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(pathFig + "/samplingBiasColours.png",dpi=300, transparent=True)
plt.show()
# %%
sns.lmplot(
    data=result,
    x="persistence_score_Green",
    y="adaptation_Green",
    height=10,
    scatter_kws={"color": "seaGreen"},
    line_kws={"color": "seaGreen"},
)
plt.ylabel("adaptation_Green", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(pathFig + "/samplingBiasGreen.png",dpi=300, transparent=True)
plt.show()
# %%

sns.lmplot(
    data=result,
    x="persistence_score_Red",
    y="adaptation_Red",
    height=10,
    scatter_kws={"color": "salmon"},
    line_kws={"color": "salmon"},
)
plt.ylabel("adaptation_Red", fontsize=20)
plt.xlabel("persistence_score", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(pathFig + "/samplingBiasRed.png",dpi=300, transparent=True)
plt.show()
# %%
sns.lmplot(data=result, x="adaptation_Green", y="adaptation_Red")
plt.show()
# %%
correlation, p_value = spearmanr(result["adaptation_Green"], result["adaptation_Red"])
print(
    f"Spearman's correlation adaptation red vs green: {correlation}, p-value: {p_value}"
)
# %%
sns.scatterplot(
    data=result,
    x="adaptation_Green",
    y="adaptation_Red",
    hue="sub",
    s=100,
    palette="tab20",
)
plt.show()
# %%

sns.lmplot(
    data=result,
    x="persistence_score_Green",
    y="persistence_score_Red",
)
plt.show()

# %%

correlation, p_value = spearmanr(
    result["persistence_score"],
    result["adaptation"],
)
print(
    f"Spearman's correlation for the adaptation score: {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_Green"],
    result["adaptation_Green"],
)
print(
    f"Spearman's correlation for the adaptation score for Green: {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["persistence_score_Red"],
    result["adaptation_Red"],
)
print(
    f"Spearman's correlation for the adaptation score for Red: {correlation}, p-value: {p_value}"
)

# %%
correlation, p_value = spearmanr(
    result["adaptation_Green"],
    result["adaptation_Red"],
)
print(
    f"Spearman's correlation for the adaptation score for Red): {correlation}, p-value: {p_value}"
)

# %%

correlation, p_value = spearmanr(
    result["persistence_score_Red"],
    result["persistence_score_Green"],
)
print(
    f"Spearman's correlation for the adaptation score for Red): {correlation}, p-value: {p_value}"
)

# %%
model = sm.OLS.from_formula("adaptation_Red~ persistence_score_Red ", result).fit()

print(model.summary())
# %%
model = sm.OLS.from_formula("adaptation_Green~ persistence_score_Green ", result).fit()

print(model.summary())
# %%

model = sm.OLS.from_formula("adaptation_Red~ adaptation_Green ", result).fit()

print(model.summary())
# %%
df.columns
# %%
df.trialTgUP
# %%
df.groupby(["sub", "proba", "color", "trialTgUP", "TD_prev"])[
    "aSPv"
].mean().reset_index()
# %%
# Group by 'sub', 'proba', and 'color' and calculate the mean of 'aSPv'
mean_velo = df.groupby(["sub", "proba", "color"])["aSPv"].mean().reset_index()

# Pivot the table to have 'proba' as columns
pivot_table = mean_velo.pivot_table(
    index=["sub", "proba"], columns="color", values="aSPv"
).reset_index()

# Calculate the adaptation
pivot_table["adaptation"] = (
    np.abs(pivot_table["Green"]) + np.abs(pivot_table["Red"])
) / 2

print(pivot_table)
# %%
sns.scatterplot(
    data=pivot_table, x="Red", y="Green", hue="proba", palette="viridis", s=50
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
    y="Green",
)
plt.show()
# %%
sns.boxplot(
    data=pivot_table,
    x="proba",
    y="Green",
)
plt.show()
# %%
# Create the plot with connected dots for each participant
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pivot_table,
    x="Red",
    y="Green",
    hue="proba",
    palette="viridis",
    style="proba",
    markers=["o", "s", "D"],
)
# Connect dots for each participant
for sub in pivot_table["sub"].unique():
    subset = pivot_table[pivot_table["sub"] == sub]
    plt.plot(subset["Red"], subset["Green"], color="gray", alpha=0.5, linestyle="--")
# Add plot formatting
plt.axhline(0, color="black", linestyle="--")
plt.axvline(0, color="black", linestyle="--")
# plt.title(f"Participants adaptaion across probabilites")
plt.xlabel("Red")
plt.ylabel("Green")
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
    plt.plot(subset["Red"], subset["Green"], color="gray", alpha=0.5, linestyle="--")
    sns.scatterplot(
        data=pivot_table[pivot_table["sub"] == sub],
        x="Red",
        y="Green",
        hue="proba",
        palette="viridis",
        style="proba",
        markers=["o", "s", "D"],
    )
    # Add plot formatting
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Participant:{sub}")
    plt.xlabel("Red")
    plt.ylabel("Green")
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
#     np.abs(pivot_table["Green"]) + np.abs(pivot_table["Red"])
# ) / 2

print(pivot_table)
pivot_table = pd.DataFrame(pivot_table)
pivot_table.columns
# %%
pivot_table.columns[2]
# %%
# pivot_table.rename(
#     columns={
#         ('left', 'Green'): "left_Green",
#         ('left', 'Red'): "left_Red",
#         ('right', 'Green'): "right_Green",
#         ('right', 'Red'): "right_Red",
#     },
#     inplace=True,
# )
#
# pivot_table.columns
# %%
sns.scatterplot(
    data=pivot_table, x="('right', 'Red')", y="('right', 'Green')", hue="proba"
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

# %%
sns.scatterplot(
    data=pivot_table, x="('left', 'Red')", y="('left', 'Green')", hue="proba"
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

# %%
# Looking at the interaction between the position and the choice of the color

plt.hist(
    df.groupby(["sub", "proba", "color", "trialTgUP"])["aSPv"]
    .count()
    .reset_index(name="count")["count"]
)
plt.show()
# %%
df_inter = (
    df.groupby(["sub", "proba", "color", "trialTgUP", "TD_prev"])["aSPv"]
    .mean()
    .reset_index()
)

# %%
df_inter["interaction"] = list(zip(df_inter["color"], df_inter["trialTgUP"]))
df_inter["interaction"] = df_inter["interaction"].astype("str")
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    palette=GreencolorsPalette,
    data=df_inter[df_inter.color == "Green"],
)
plt.legend(fontsize=20)
# plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.pdf")
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    palette=RedcolorsPalette,
    data=df_inter[df_inter.color == "Red"],
)
plt.legend(fontsize=20)
# plt.title("Anticipatory Velocity Given the color Position: Red", fontsize=30)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.pdf")
plt.show()

# %%

# %%
df["interColPos"] = list(zip(df["color"], df["trialTgUP"]))
df["interColPos"] = df["interColPos"].astype("str")

# %%

for s in df["sub"].unique():
    dfs = df[df["sub"] == s]
    sns.histplot(data=dfs, x="interColPos")
    plt.show()


# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="interColPos",
    palette=GreencolorsPalette,
    data=df[df.color == "Green"],
)
plt.legend(fontsize=20)
# plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.pdf")
plt.show()
# %%
for s in df["sub"].unique():
    dfs = df[df["sub"] == s]
    sns.histplot(data=dfs, x="interColPos")
    plt.show()


# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="interColPos",
    palette=RedcolorsPalette,
    data=df[df.color == "Red"],
)
plt.legend(fontsize=20)
# plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.pdf")
plt.show()
# %%
