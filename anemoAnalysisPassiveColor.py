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
pathFig = "/Users/mango/Contextual-Learning/ColorCue/figures/imposedColor/"
df = pd.read_csv("/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_passiveColorCP.csv")
main_dir = "/Users/mango/oueld.h/attentionalTask/data/"
print(df)
# %%
df.rename(columns={"pR-Red": "proba"}, inplace=True)
df.rename(columns={"trial_color": "color"}, inplace=True)
df["sub"] = [int(x.split("-")[1]) for x in df["sub"]]
# %%
allEvents = pd.read_csv(os.path.join(main_dir, "allEvents.csv"))
# To align anemo data that start at trial 0
allEvents["trial"] = allEvents["trial"].values - 1
allEvents["proba"] = allEvents["proba"].values / 100
allEvents
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
                ] = prev_trial["trial_direction"].values[0]

                # print(prev_trial["trial_color_chosen"].values[0])
                # Assign color from previous trial
                df.loc[
                    (df["sub"] == sub) & (df["proba"] == p) & (df["trial"] == t),
                    "color_prev",
                ] = prev_trial["trial_color"].values[0]
# %%
df.columns
df[(df["TD_prev"].isna())]
# %%
df = df[~(df["TD_prev"].isna())]
# %%
df.TD_prev
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
# %%
df.columns
# %%
colors = ["Green", "Red"]
# %%
# dd = df.groupby(["sub", "color", "proba", "TD_prev"])[["aSPv"]].mean().reset_index()
dd = df.groupby(["sub", "color", "proba"])[["aSPv"]].mean().reset_index()

dd
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
aSPv = dd[dd.color == "Green"]["aSPv"]
proba = dd[dd.color == "Green"]["proba"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, proba)
print(f"Spearman's correlation (Green): {correlation}, p-value: {p_value}")

# %%
aSPv = dd[dd["proba"] == 0.75]["aSPv"]
color = dd[dd["proba"] == 0.75]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 75): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd["proba"] == 0.25]["aSPv"]
color = dd[dd["proba"] == 0.25]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 25): {correlation}, p-value: {p_value}")


# %%

aSPv = dd[dd["proba"] == 0.50]["aSPv"]
color = dd[dd["proba"] == 0.50]["color"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, color)
print(f"Spearman's correlation(proba 50): {correlation}, p-value: {p_value}")


# %%

# Friedman test for Red color
pg.friedman(data=df[df.color == "Red"], dv="aSPv", within="proba", subject="sub")
# %%

pg.friedman(data=df[df.color == "Green"], dv="aSPv", within="proba", subject="sub")
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
dd
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
# Pivot the data for proba
pivot_proba = dd[dd.color == "Red"].pivot(index="sub", columns="proba", values="aSPv")
pivot_proba
# %%
# Perform the Friedman Test for proba
statpivot_proba, p_pivot_proba = friedmanchisquare(
    pivot_proba[0.25], pivot_proba[0.50], pivot_proba[0.75]
)
print(
    f"Friedman Test for proba: Statistic(Red) = {statistic_proba}, p-value = {p_value_proba}"
)


# %%
# Pivot the data for proba
pivot_color = dd[dd["proba"] == 0.25].pivot(index="sub", columns="color", values="aSPv")
pivot_color

# a %%
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
# pos-hoc analysis

# Perform the Nemenyi Test
posthoc = sp.posthoc_nemenyi_friedman(pivot_proba.values)
print(posthoc)

# %%

# Perform the Wilcoxon Test post-hoc analysis
posthoc = sp.posthoc_wilcoxon(pivot_proba.values.T)
print(posthoc)
# %%
# Apply the Holm-Bonferroni correction to the Wilcoxon Test p-values
corrected_p_values = multipletests(posthoc.values.flatten(), method="holm")[1]
corrected_p_values = corrected_p_values.reshape(posthoc.shape)

print("Holm-Bonferroni corrected Wilcoxon Test p-values:")
print(pd.DataFrame(corrected_p_values, index=posthoc.index, columns=posthoc.columns))
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
p = 0.75
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
plt.title(f"Mid Trials{midTrials[0]},{midTrials[1]}: P(Right|Red)={proba}")
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
plt.title(f"Early Trials>{lateTrials}: P(Right|Red)={proba}")
plt.show()
# %%
# Repeated measures ANOVA
# Perform mixed ANOVA
model = ols("aSPv ~ C(color)*(proba) ", data=dd).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# %%
# cehcking the normality of the data
print(pg.normality(dd["aSPv"]))
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
    ax.set_title(f"ASEM: P(Right|Red)=P(Left|Green)={p}")
    ax.legend(["Red", "Green"])
# Adjust spacing between subplots
facet_grid.figure.subplots_adjust(
    wspace=0.2, hspace=0.2
)  # Adjust wspace and hspace as needed

# Show the plot
plt.show()

# %%
for s in df["sub"].unique():
    df_s = df[df["sub"] == s]

    # Set up the FacetGrid
    facet_grid = sns.FacetGrid(
        data=df_s,
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
    for ax, p in zip(facet_grid.axes.flat, df_s.proba.unique()):
        ax.set_title(f"ASEM Subject {s}: P(Right|Red)=P(Left|Green)={p}")
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
    within="proba",
    subject="sub",
    data=df[df.color == "Green"],
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
dd["sub"].unique()
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
    hue="color",
    hue_order=colors,
    palette=colors,
)
_ = plt.title("ASEM Across probabilities", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|Red)=P(Left|Green)", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/asemAcrossprobapp.svg", transparent=True)
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
    palette=colors,
)
plt.show()
plt.savefig(pathFig + "/asemAcrossviolinplot.svg", transparent=True)
# %%
dd
# %%
fig = plt.figure()
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
_ = plt.title("ASEM Per Subject: color Red", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsRed.svg", transparent=True)
plt.show()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.pointplot(
    data=df[df.color == "Green"],
    x="proba",
    y="aSPv",
    capsize=0.1,
    errorbar="ci",
    hue="sub",
    palette="tab20",
    alpha=0.8,
)
_ = plt.title("ASEM Per Subject: color Green", fontsize=30)
plt.legend(fontsize=20)
plt.xlabel("P(Right|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/individualsGreen.svg", transparent=True)
plt.show()
# %%
# pg.normality(df[df.color == "Red"], group="proba", dv="aSPv")
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
    "aSPv~C( proba,Treatment(.50)) *color",
    data=df,
    re_formula="~proba",
    groups=df["sub"],
).fit()
model.summary()

# %%
residuals = model.resid

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()
# %%
pg.qqplot(residuals, dist="norm")
plt.show()
# %%
# Histogram
plt.hist(residuals, bins=50)
plt.title("Histogram of residuals")
plt.show()
# %%
# Shapiro-Wilk test for normality# Perform the KS test on the residuals
stat, p = stats.kstest(residuals, "norm")

print(f"KS test statistic: {stat:.4f}")
print(f"KS test p-value: {p:.4f}")
# a %%
normaltest_result = stats.normaltest(residuals)
print(f"D'Agostino's K^2 test p-value: {normaltest_result.pvalue:.4f}")
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
    "aSPv~ C(color)",
    data=df[df.proba == 0.25],
    # re_formula="~color",
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
    "aSPv~ C(color)",
    data=df[df.proba == 0.50],
    # re_formula="~color",
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
RedcolorsPalette = ["#e83865", "#cc3131"]
GreencolorsPalette = ["#008000", "#285943"]
dd = df.groupby(["sub", "proba", "color"])[["aSPv"]].mean().reset_index()
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    hue="color",
    errorbar="ci",
    palette=[GreencolorsPalette[1], RedcolorsPalette[1]],
    hue_order=colors,
    fill=False,
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
    size=8,
    # alpha=0.5,
    legend=False,
)

plt.legend(fontsize=20)
plt.title("ASEM across 3 different probabilites", fontsize=30)
plt.xlabel("P(Right|Red)=P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.ylim(-0.75, 0.75)
plt.legend(fontsize=20)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvcolors.svg", transparent=True)
plt.show()

# %%
df_prime = df[
    [
        "sub",
        "trial",
        "proba",
        "color",
        "target_dir",
        "TD_prev",
        "aSPv",
    ]
]
# sa%%
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
plt.title("Anticipatory Smooth Eye Movement: color Red", fontsize=30)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvRed.svg", transparent=True)
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
plt.title("Anticipatory Smooth Eye Movement: color Green", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-0.75, 0.75)
plt.savefig(pathFig + "/aSPvGreen.svg", transparent=True)
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
    palette=[GreencolorsPalette[1]],
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
    bar.set_edgecolor(GreencolorsPalette[1])  # Set edge color

sns.stripplot(
    x="proba",
    y="aSPv",
    hue="TD_prev",
    hue_order=["left", "right"],
    palette=[GreencolorsPalette[1]],
    dodge=True,
    jitter=True,
    size=8,
    # alpha=0.7,
    data=dd[dd.color == "Green"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=GreencolorsPalette[1], alpha=1, label="Left"),
    Patch(
        facecolor="none", hatch="///", label="Right", edgecolor=GreencolorsPalette[1]
    ),
]
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
g.ax.set_title("Anticipatory Velocity Given Previous TD: color Green ", fontsize=30)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Green)", fontsize=30)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvGreenTD.svg", transparent=True)
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
    palette=[RedcolorsPalette[1]],
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.7,
    capsize=0.1,
    hue_order=["left", "right"],
    legend=False,
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
    palette=[RedcolorsPalette[1]],
    dodge=True,
    jitter=True,
    size=8,
    # alpha=0.7,
    data=dd[dd.color == "Red"],
    legend=False,
)


# Create custom legend

legend_elements = [
    Patch(facecolor=RedcolorsPalette[1], alpha=1, label="Left"),
    Patch(facecolor="none", hatch="///", label="Right", edgecolor=RedcolorsPalette[1]),
]
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
g.ax.set_title("Anticipatory Velocity Given Previous TD: Color Red ", fontsize=30)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Red)", fontsize=30)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvRedTD.svg", transparent=True)
plt.show()

# %%
df["interaction"] = list(zip(df["TD_prev"], df["color_prev"]))
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
    df_prime.groupby(["sub", "proba", "interaction", "color"])
    .mean()[["aSPv"]]
    .reset_index()
)

# %%
df.columns
# %%
df_prime.groupby(["proba", "interaction", "color"]).count()[["aSPv"]]
# %%
learningCurveInteraction["interaction"].unique()
# %%
# Cmap for Green and Red for the interaction plots

df_prime["interaction"].unique()

hue_order = [
    ("left", "Green"),
    ("left", "Red"),
    ("right", "Green"),
    ("right", "Red"),
]
# %%
RedcolorsPalette = ["#e83865", "#cc3131"]
GreencolorsPalette = ["#8cd790", "#285943"]
# colorsPalette = ["#285943", "#cc3131", "#e83865", "#8cd790"]
colorsPalette = [
    GreencolorsPalette[0],
    RedcolorsPalette[0],
    GreencolorsPalette[1],
    RedcolorsPalette[1],
]
# %%
fig = plt.figure()
# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    palette=colorsPalette,
    hue="interaction",
    hue_order=hue_order,
    data=df_prime[df_prime.color == "Red"],
)
plt.title(
    "ASEM: color Red\n Interaction of Previous Target Direction & color Chosen",
    fontsize=30,
)
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1.25, 1.25)
plt.xlabel("P(Right|Red)", fontsize=30)
plt.ylabel("ASEM(deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvRedInteraction.svg", transparent=True)
plt.show()
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
    alpha=0.5,
    capsize=0.1,
    hue_order=[
        ("left", "Green"),
        ("left", "Red"),
        ("right", "Green"),
        ("right", "Red"),
    ],
    legend=False,
)

# Determine the number of bars per x-value
n_categories = len(df_prime["interaction"].unique())
n_x_values = len(df_prime["proba"].unique())
total_bars = n_categories * n_x_values

# Add hatching to the bars for 'right' categories
for i, bar in enumerate(g.ax.patches):
    # Determine if this bar represents a 'right' category
    if i % n_categories >= n_categories // 2:  # Second half of bars for each x-value
        bar.set_facecolor("none")  # Make bar empty
        bar.set_hatch("///")  # Add diagonal lines
        bar.set_edgecolor(colorsPalette[i // n_x_values])  # Maintain the category color

# Add stripplot
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    hue_order=[
        ("left", "Green"),
        ("left", "Red"),
        ("right", "Green"),
        ("right", "Red"),
    ],
    palette=colorsPalette,
    dodge=True,
    jitter=True,
    size=8,
    data=learningCurveInteraction[learningCurveInteraction.color == "Red"],
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
g.ax.legend(handles=legend_elements, fontsize=20)

# Customize the plot
g.ax.set_title("Anticipatory Velocity Given Previous TD: color Red", fontsize=30)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Red)", fontsize=30)
g.ax.tick_params(labelsize=25)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvRedInteraction.svg", transparent=True)
plt.show()
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
    alpha=0.5,
    capsize=0.1,
    hue_order=[
        ("left", "Green"),
        ("left", "Red"),
        ("right", "Green"),
        ("right", "Red"),
    ],
    legend=False,
)

# Get the number of categories and x-values
n_categories = len(df_prime["interaction"].unique())
n_x_values = len(df_prime["proba"].unique())

# Add hatching to the bars for 'right' categories
for prob_idx in range(n_x_values):  # For each probability value
    for cat_idx in range(n_categories):  # For each category within that probability
        bar_idx = prob_idx * n_categories + cat_idx  # Calculate the actual bar index
        bar = g.ax.patches[bar_idx]

        # Check if this is a "Right" category (3rd or 4th bar in each group)
        if cat_idx >= 2:  # Right categories are in positions 2 and 3 within each group
            bar.set_facecolor("none")
            bar.set_hatch("///")
            bar.set_edgecolor(colorsPalette[cat_idx])

# Add stripplot
sns.stripplot(
    x="proba",
    y="aSPv",
    hue="interaction",
    hue_order=[
        ("left", "Green"),
        ("left", "Red"),
        ("right", "Green"),
        ("right", "Red"),
    ],
    palette=colorsPalette,
    dodge=True,
    jitter=True,
    size=8,
    data=learningCurveInteraction[learningCurveInteraction.color == "Red"],
)

# Create custom legend
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

# Add the legend and customize the plot
g.ax.legend(handles=legend_elements, fontsize=20)
g.ax.set_title("Anticipatory Velocity Given Previous TD: color Red", fontsize=30)
g.ax.set_ylabel("ASEM (deg/s)", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Left|Red)", fontsize=30)
g.ax.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvRedInteraction.svg", transparent=True)
plt.show()
# %%

fig = plt.figure()

# Toggle full screen mode
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
sns.barplot(
    x="proba",
    y="aSPv",
    palette=colorsPalette,
    hue="interaction",
    hue_order=df_prime["interaction"].unique(),
    data=df_prime[df_prime.color == "Green"],
)
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1.25, 1.25)
plt.title(
    "ASEM:color Green\n Interaction of Previous Target Direction & color Chosen",
    fontsize=30,
)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.savefig(pathFig + "/aSPvGreenInteraction.svg", transparent=True)
plt.show()
# %%
df.columns
# %%
model = smf.mixedlm(
    "aSPv~  C(color,Treatment('Red'))*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~color",
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
    "aSPv~  C(color,Treatment('Red')) * C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~color",
    groups=df[df.proba == 0.75]["sub"],
).fit(method="lbfgs")
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(color,Treatment('Red'))*C(TD_prev)",
    data=df[df.proba == 0.50],
    re_formula="~color",
    groups=df[df.proba == 0.50]["sub"],
).fit(method=["lbfgs"])
model.summary()
# %%
df.color_prev
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
    y="Red",
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
plt.title(f"Participants adaptaion across probabilites")
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
plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
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
plt.title("Anticipatory Velocity Given the color Position: Red", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
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
plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
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
plt.title("Anticipatory Velocity Given the color Position: Green ", fontsize=30)
plt.ylabel("ASEM (deg/s)", fontsize=30)
plt.xlabel("P(Left|Green)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.ylim(-1, 1)
# plt.savefig(pathFig + "/aSPvGreenTD.svg")
plt.show()
# %%
