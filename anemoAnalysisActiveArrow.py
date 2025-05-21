from scipy import stats
from scipy.stats import spearmanr, permutation_test,pearsonr
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import os
from scipy.stats import anderson
from statannotations.Annotator import Annotator
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

result = anderson(df["aSPv"].values, dist='norm')
print(result)
# %%
sns.histplot(data=df, x="SPlat")
plt.show()
# %%
df[df.SPlat==df['SPlat'].values.max()]['SPlat']
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

df=df[~( df["TD_prev"].isna() )]
df=df[~( df["training"]=='yes' )]
# Getting Rid of the training trials for the first subject
df=df[~(( df['sub']==1 )&(df['session']=='session-01')&(df['trial']<10))]
# %%
# %%
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
# df = df[~((df["sub"] == 6) & (df["proba"] == 0.5))]
df = df[~((df["sub"] == 4)| (df["sub"] == 2)| (df["sub"] == 12))]
# df = df[~((df["sub"] == 12))]
# %%
df["TD_prev"] = df["TD_prev"].apply(lambda x: "right" if x == 1 else "left")
df["interaction"] = list(zip(df["TD_prev"], df["arrow_prev"]))
# %%
df[(df["TD_prev"].isna())]
df[df["aSPv"] == df["aSPv"].max()]["aSPv"]
# %%
sns.histplot(data=df, x="aSPv")
plt.show()
# %%
badTrials = df[( df["aSPv"] < - 5) | ( df["aSPv"] > 5 )]["aSPv"]
print(badTrials )
# %%
df = df[~(( df["aSPv"] < -5 ) | ( df["aSPv"] > 5 ))]
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

# checking the normality of the data
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

# s%%
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
plt.savefig(pathFig + "/asemAcrossProbappFullProba.png",dpi=300, transparent=True)
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
plt.savefig(pathFig + "/asemAcrossprobaviolinFullProba.png",dpi=300, transparent=True)
plt.show()

# %%
model = smf.mixedlm(
    "aSPv~proba*arrow",
    data=df,
    re_formula="~proba*arrow",
    groups=df["sub"],
).fit(method=['lbfgs'])
model.summary()
# %%
downarrowsPalette = ["#0F68A9", "#A2D9FF"]
uparrowsPalette = ["#FAAE7B", "#FFD699"]
# %%
# Extract fixed effects
fe = model.fe_params
base_slope = fe['proba']  # base slope (Down)
interaction = fe['proba:arrow[T.up]']  # additional slope for Up

# Extract random effects
re_dict = model.random_effects

# Create an empty dataframe for subject-specific slopes
participant_slopes = []

# Loop through each subject
for sub, re in re_dict.items():
    # The random effects order depends on the model specification
    # Typically for re_formula="~proba*color"
    # The order is [intercept, color, proba, proba:color]
    
    proba_re = re[2]  # Random effect for proba (green condition)
    interaction_re = re[3]  # Random effect for proba:color interaction
    
    down_slope = base_slope + proba_re
    up_slope = base_slope + interaction + proba_re + interaction_re
    
    participant_slopes.append({
        'sub': sub,
        'down_slope': down_slope,
        'up_slope': up_slope,
        'avg_abs_slope': (abs(down_slope) + abs(up_slope)) / 2
    })

# Convert to dataframe
participant_slopes_df = pd.DataFrame(participant_slopes)
print(participant_slopes_df)
# %%
# Extract slope values for up and down

green_slopes = participant_slopes_df['down_slope'].values
red_slopes =  participant_slopes_df['up_slope'].values

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
plt.xlabel("Adaptive Slope for Down")
plt.ylabel("Adaptive Slope for Up")
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
fe = model.fe_params
intc = fe['Intercept']
slope = fe['proba']
color_effect = fe['arrow[T.up]']
interaction = fe['proba:arrow[T.up]']

# Random effects
re_df = pd.DataFrame(model.random_effects).T
re_df.columns = ['Intercept_re', 'proba_re', 'arrow[T.Up]_re', 'proba:arrow[T.Up]_re']
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
    ax.plot(proba_range, y_green, color='blue', alpha=0.3)

    # Red (with fixed and random adjustments)
    y_red = (intc + color_effect + i_re) + (slope + interaction + s_re) * proba_range
    ax.plot(proba_range, y_red, color='orange', alpha=0.3)

# Fixed effect mean lines (thick black)
mean_green = intc + slope * proba_range
mean_red = (intc + color_effect) + (slope + interaction) * proba_range
ax.plot(proba_range, mean_green, color='black', linewidth=3, label='Down mean')
ax.plot(proba_range, mean_red, color='black', linewidth=3, linestyle='--', label='Up mean')

# Labels and legend
ax.set_xlabel('proba')
ax.set_ylabel('aSPv')
ax.set_title('Random Slopes and Intercepts by Color')
ax.legend()
plt.tight_layout()
plt.show()

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
).fit(method=['lbfgs'])
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
dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
# %%
# fig = plt.figure()
# # Toggle full screen mode
# figManager = plt.get_current_fig_manager()
# figManager.full_screen_toggle()
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
plt.savefig(pathFig + "/aSPvarrowsFullProba.png",dpi=300, transparent=True)
plt.show()
# %%
dd
# %%
def statistic(x, y,axis):

    return np.mean(x,axis=axis) - np.mean(y,axis=axis)
# %%
ddd=dd[~( dd['sub']==6 )]
x=ddd[( ddd['arrow']=='down' )& (ddd['proba']==0.75)]['aSPv'].values
y=ddd[( ddd['arrow']=='down' )& ( ddd['proba']==0.5 )]['aSPv'].values
statistic(x,y,0)
# %%
res = permutation_test(data=(x, y), statistic=statistic,permutation_type='samples', vectorized=True,n_resamples=2**(len(x)),)
print(res.statistic)
# %%
print(res.pvalue)

# %%
g=sns.catplot(
    data=dd,
    x="proba",
    y="aSPv",
    hue="arrow",
    kind="bar",
    errorbar='se',
    # errorbar='se',
    n_boot=1000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=["down", "up"],
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
order=[0,0.25,0.5,0.75,1]
hue_order=["down", "up"]
pairs = [
    ((0., "down"), (0., "up")),
    ((0.25, "down"), (0.25, "up")),
    ((0.5, "down"), (0.5, "up")),
    ((0.75, "down"), (0.75, "up")),
    ((1, "down"), (1, "up")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator = Annotator(g.ax, pairs, data=dd, x='proba', y="aSPv", hue="arrow",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Arrow", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvarrowsFullProba.png",dpi=300, transparent=True)
plt.show()
# %%
dd[dd.proba==0.25]
# %%
hue_order=[0,0.25,0.5,0.75,1]
order=["down", "up"]
g=sns.catplot(
    data=df,
    x="arrow",
    y="aSPv",
    hue="proba",
    kind="bar",
    errorbar=("ci", 95),
    # errorbar='se',
    n_boot=10000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=hue_order,
    fill=False,
    legend=False,
    palette='viridis',
)

sns.stripplot(
    x="arrow",
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
pairs = [
    (("down",0.25), ('down',0.50)),
    (("down",0.75), ('down',0.50)),
    (("down",0.25), ('down',0.75)),

    (("up",0.25), ('up',0.50)),
    (("up",0.75), ('up',0.50)),
    (("up",0.25), ('up',0.75)),
]
annotator = Annotator(g.ax, pairs, data=dd, x='arrow', y="aSPv", hue="proba",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired',  loc='outside',fontsize=20,comparisons_correction='HB')
annotator.apply_and_annotate()
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"Cue", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

viridis_colors = plt.cm.viridis([0, 0.25,0.5,   0.75, 1])  # Maps to your 0.25, 0.50, 0.75 proba values

legend_elements = [
    Patch(facecolor=viridis_colors[1], edgecolor='black', label='0.25'),
    Patch(facecolor=viridis_colors[2], edgecolor='black', label='0.50'),
    Patch(facecolor=viridis_colors[3], edgecolor='black', label='0.75')
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title=r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvarrowsFullProbabis.png",dpi=300, transparent=True)
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

order=[0,0.25,0.5,0.75,1]
hue_order=["left", "right"]
pairs = [
    ((0., "left"), (0., "right")),
    ((0.25, "left"), (0.25, "right")),
    ((0.5, "left"), (0.5, "right")),
    ((0.75, "left"), (0.75, "right")),
    ((1, "left"), (1, "right")),
]
annotator = Annotator(g.ax, pairs, data=dd[dd.arrow == "down"], x='proba', y="aSPv", hue="TD_prev",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()

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
plt.savefig(pathFig + "/aSPvdownTDFullProba.png",dpi=300, transparent=True)
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

order=[0,0.25,0.5,0.75,1]
hue_order=["left", "right"]
pairs = [
    ((0., "left"), (0., "right")),
    ((0.25, "left"), (0.25, "right")),
    ((0.5, "left"), (0.5, "right")),
    ((0.75, "left"), (0.75, "right")),
    ((1, "left"), (1, "right")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator = Annotator(g.ax, pairs, data=dd[dd.arrow == "up"], x='proba', y="aSPv", hue="TD_prev",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()


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
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTDFullProba.png",dpi=300, transparent=True)
plt.show()
# %%
pivot_df = dd.pivot_table(
    index=['sub', 'proba', 'arrow'],
    columns='TD_prev',
    values='aSPv'
)

# Step 2: Calculate the difference (right - left)
pivot_df['Diff'] = pivot_df['right'] - pivot_df['left']

# Step 3: Reset index to get back to a regular dataframe format
result_df = pivot_df.reset_index()

# Display the result
print(result_df)
# %%
# Create the plot using catplot
g = sns.catplot(
    data=result_df,
    x="proba",
    y="Diff",
    hue="arrow",
    kind="bar",
    errorbar='se',
    n_boot=1000,
    palette=[downarrowsPalette[0],uparrowsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=["down", "up"],
    legend=False,
    fill=False,
)

# # Add hatching to the right bars
# for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
#     bar.set_facecolor("none")  # Make bar empty
#     bar.set_hatch("///")  # Add diagonal lines
#     bar.set_edgecolor(uparrowsPalette[0])  # Set edge color

sns.stripplot(
    data=result_df,
    x="proba",
    y="Diff",
    hue="arrow",
    hue_order=["down", "up"],
    palette=[downarrowsPalette[0],uparrowsPalette[0]],  # Same color for both
    dodge=True,
    jitter=True,
    linewidth=1,
    size=6,
    legend=False,
)

# Create custom legend

legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="arrow", title_fontsize=20
)
plt.tight_layout()

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=25)
g.ax.set_ylabel(" aSPv Diff (Previous TD: Right - Left ) (deg/s)", fontsize=20)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvTDiff.png",dpi=300, transparent=True)
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
pivot_df = learningCurveInteraction.pivot_table(
    index=['sub', 'proba', 'arrow'],
    columns='interaction',
    values='aSPv'
)

# Step 2: Calculate the difference (right - left)
pivot_df['DiffDown'] = pivot_df[('right','down')] - pivot_df[('left','down')]
pivot_df['DiffUp'] = pivot_df[('right','up')] - pivot_df[('left','up')]

# Step 3: Reset index to get back to a regular dataframe format
result_df = pivot_df.reset_index()

# Display the result
print(result_df)

# %% 

result_df.dropna(inplace=True)
# %%


def perm_test(x, y, n_perms=1000, stat=np.nanmean):
    diff = np.abs(stat(x)-stat(y))
    pooled = np.concatenate([x, y])
    count = 0
    for n in range(n_perms):
        rng = np.random.default_rng(n)
        rng.shuffle(pooled)
        x_perm = pooled[:len(x)]
        y_perm = pooled[len(x):]
        diff_perm = np.abs(stat(x_perm)-stat(y_perm))
        if diff_perm >= diff:
            count += 1

    pv = count / n_perms
    return pv
# %%
perm_test(result_df[result_df['arrow']=='up']['DiffUp'].values,result_df[result_df['arrow']=='up']['DiffDown'].values)
# %%
perm_test(result_df[result_df['arrow']=='down']['DiffUp'].values,result_df[result_df['arrow']=='down']['DiffDown'].values)
# %%
perm_test(result_df[result_df['arrow']=='up']['DiffUp'].values,result_df[result_df['arrow']=='down']['DiffDown'].values)
# %%
perm_test(result_df[result_df['arrow']=='up']['DiffDown'].values,result_df[result_df['arrow']=='down']['DiffUp'].values)
# %%
melted_df = pd.melt(
    result_df,
    id_vars=["sub", "proba", "arrow"],
    value_vars=["DiffDown", "DiffUp"],
    var_name="DiffType",
    value_name="Diff"
)


print(melted_df)
# Set up the figure
# %%
# Create grouped bar plot
g = sns.catplot(
    data=melted_df,
    x="proba",
    y="Diff",
    hue="arrow",
    col="DiffType",  # Split by DiffType
    kind="bar",
    errorbar='ci',
    n_boot=1000,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    height=8,
    aspect=1.2,
    alpha=0.8,
    capsize=0.1,
    hue_order=["down", "up"],
    fill=False,
    legend=False,
)

# Add individual data points as strips
for ax in g.axes.flat:
    diff_type = ax.get_title().split(" ")[-1]  # Get the DiffType from title
    sns.stripplot(
        data=melted_df[melted_df["DiffType"] == diff_type],
        x="proba",
        y="Diff",
        hue="arrow",
        hue_order=["down", "up"],
        palette=[downarrowsPalette[0], uparrowsPalette[0]],
        dodge=True,
        jitter=True,
        linewidth=1,
        size=6,
        legend=False,
        ax=ax
    )

# Create custom legend
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
]

# Add legend to the figure
plt.legend(
    handles=legend_elements,
    fontsize=15,
    title="arrow",
    title_fontsize=15,
    # loc='upper center',
    # bbox_to_anchor=(0.5, 1.05),
    # ncol=2
)

# Customize the plot titles and labels
g.set_titles("{col_name}")
g.set_axis_labels(r"$\mathbb{P}$(Right|Up)", "aSPv Diff (Previous TD: Right - Left) (deg/s)")
g.set_xticklabels(fontsize=20)
g.set_yticklabels(fontsize=20)

# Customize labels and ticks
for ax in g.axes.flat:
    ax.set_xlabel(r"$\mathbb{P}_{Up}$(Right)", fontsize=25)
    ax.set_ylabel("aSPv Diff (Previous Trial: (Cue,Right) - (Cue,Left)) (deg/s)", fontsize=15)
    ax.tick_params(labelsize=20)
    
    # Add statistical annotations if needed
    order = [0.25, 0.5, 0.75]
    hue_order = ["down", "up"]
    pairs = [
        ((0.25, "down"), (0.25, "up")),
        ((0.5, "down"), (0.5, "up")),
        ((0.75, "down"), (0.75, "up")),
    ]
    
    # Add statistical annotations
    diff_type = ax.get_title().split(" ")[-1]
    subset_df = melted_df[melted_df["DiffType"] == diff_type]

g.set_titles("")
plt.tight_layout()
plt.savefig("aSPvTDiff_combined.png", dpi=300, transparent=True)
plt.show()
# %%

# Create a combined plot where one type uses hatching
g = sns.catplot(
    data=melted_df,
    x="proba",
    y="Diff",
    hue="arrow",
    col="DiffType",
    kind="bar",
    errorbar='ci',
    n_boot=1000,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    height=8,
    aspect=1.,
    alpha=0.8,
    capsize=0.1,
    hue_order=["down", "up"],
    legend=False,
    fill=False,
)

# Apply different patterns to distinguish between DiffDown and DiffUp
for i, ax in enumerate(g.axes.flat):
    bars = ax.patches
    diff_type = ax.get_title().split(" ")[-1]
    
    # Apply hatching to one of the diff types
    if diff_type == "DiffUp":
        for bar in bars:
            bar.set_hatch("///")
            
    # Add individual data points
    sns.stripplot(
        data=melted_df[melted_df["DiffType"] == diff_type],
        x="proba",
        y="Diff",
        hue="arrow",
        hue_order=["down", "up"],
        palette=[downarrowsPalette[0], uparrowsPalette[0]],
        dodge=True,
        jitter=True,
        linewidth=1,
        size=6,
        legend=False,
        ax=ax
    )

# Create custom legend with both arrow and DiffType
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
    Patch(facecolor='gray', hatch="///", alpha=0.6, label="DiffUp"),
    Patch(facecolor='gray', alpha=0.6, label="DiffDown"),
]

# Add legend to the figure
g.figure.legend(
    handles=legend_elements,
    fontsize=15,
    title="Types",
    title_fontsize=15,
    loc='upper center',
    bbox_to_anchor=(0.88, .97),
    ncol=2
)

# Customize the plot titles and labels
g.set_titles("{col_name}")
g.set_axis_labels(r"$\mathbb{P}_{Up}$(Right)", "aSPv Diff (Previous TD: Right - Left) (deg/s)")

# Customize labels and ticks
for ax in g.axes.flat:
    ax.set_xlabel(r"$\mathbb{P}_{Up}$(Right)", fontsize=25)
    ax.set_ylabel("aSPv Diff (Previous Trial: (Cue,Right) - (Cue,Left)) (deg/s)", fontsize=15)
    ax.tick_params(labelsize=20)
    
    # Add statistical annotations if needed
    order = [0.25, 0.5, 0.75]
    hue_order = ["Down", "Up"]
    pairs = [
        ((0.25, "Down"), (0.25, "Up")),
        ((0.5, "Down"), (0.5, "Up")),
        ((0.75, "Down"), (0.75, "Up")),
    ]
    

g.set_titles("")
plt.tight_layout()
plt.savefig("aSPvTDiff_patterned.png", dpi=300, transparent=True)
plt.show()
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
# plt.tight_layout()
plt.savefig(pathFig + "/aSPvUpInteractionFullProba.png",dpi=300, transparent=True,bbox_inches='tight')
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
plt.savefig(pathFig + "/aSPvDownInteractionFullProba.png",dpi=300, transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~arrow*TD_prev",
    groups=df[df.proba == 0.25]["sub"],
).fit(method=['lbfgs'])
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~arrow*TD_prev",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=[ "lbfgs" ])
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    re_formula="~arrow*TD_prev",
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
correlation, p_value = pearsonr(aSPv, proba)
print(f"Spearman's correlation (up): {correlation}, p-value: {p_value}")
# %%
aSPv = dd[dd.arrow == "down"]["aSPv"]
proba = dd[dd.arrow == "down"]["proba"]

# Spearman's rank correlation
correlation, p_value = pearsonr(aSPv, proba)
print(f"Spearman's correlation (down): {correlation}, p-value: {p_value}")

# %%
aSPv = dd[dd.proba == 0.75]["aSPv"]
arrow = dd[dd.proba == 0.75]["arrow"]

# Spearman's rank correlation
correlation, p_value = spearmanr(aSPv, arrow)
print(f"Spearman's correlation(Proba 0.75): {correlation}, p-value: {p_value}")


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

# checking the normality of the data
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

# s%%
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
    n_boot=10000,
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
plt.savefig(pathFig + "/asemAcrossProbapp.png",dpi=300, transparent=True)
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
plt.savefig(pathFig + "/asemAcrossprobaviolin.png",dpi=300, transparent=True)
plt.show()

# %%
model = smf.mixedlm(
    "aSPv~proba*arrow",
    data=df,
    re_formula="~proba*arrow",
    groups=df["sub"],
).fit()
model.summary()
# %%
downarrowsPalette = ["#0F68A9", "#A2D9FF"]
uparrowsPalette = ["#FAAE7B", "#FFD699"]
# %%
# Extract fixed effects
fe = model.fe_params
base_slope = fe['proba']  # base slope (Down)
interaction = fe['proba:arrow[T.up]']  # additional slope for Up

# Extract random effects
re_dict = model.random_effects

# Create an empty dataframe for subject-specific slopes
participant_slopes = []

# Loop through each subject
for sub, re in re_dict.items():
    # The random effects order depends on the model specification
    # Typically for re_formula="~proba*color"
    # The order is [intercept, color, proba, proba:color]
    
    proba_re = re[2]  # Random effect for proba (green condition)
    interaction_re = re[3]  # Random effect for proba:color interaction
    
    down_slope = base_slope + proba_re
    up_slope = base_slope + interaction + proba_re + interaction_re
    
    participant_slopes.append({
        'sub': sub,
        'down_slope': down_slope,
        'up_slope': up_slope,
        'avg_abs_slope': (abs(down_slope) + abs(up_slope)) / 2
    })

# Convert to dataframe
participant_slopes_df = pd.DataFrame(participant_slopes)
print(participant_slopes_df)
# %%
# Extract slope values for up and down

green_slopes = participant_slopes_df['down_slope'].values
red_slopes =  participant_slopes_df['up_slope'].values

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
plt.xlabel("Adaptive Slope for Down")
plt.ylabel("Adaptive Slope for Up")
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
fe = model.fe_params
intc = fe['Intercept']
slope = fe['proba']
color_effect = fe['arrow[T.up]']
interaction = fe['proba:arrow[T.up]']

# Random effects
re_df = pd.DataFrame(model.random_effects).T
re_df.columns = ['Intercept_re', 'proba_re', 'arrow[T.Up]_re', 'proba:arrow[T.Up]_re']
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
    ax.plot(proba_range, y_green, color='blue', alpha=0.3)

    # Red (with fixed and random adjustments)
    y_red = (intc + color_effect + i_re) + (slope + interaction + s_re) * proba_range
    ax.plot(proba_range, y_red, color='orange', alpha=0.3)

# Fixed effect mean lines (thick black)
mean_green = intc + slope * proba_range
mean_red = (intc + color_effect) + (slope + interaction) * proba_range
ax.plot(proba_range, mean_green, color='black', linewidth=3, label='Down mean')
ax.plot(proba_range, mean_red, color='black', linewidth=3, linestyle='--', label='Up mean')

# Labels and legend
ax.set_xlabel('proba')
ax.set_ylabel('aSPv')
ax.set_title('Random Slopes and Intercepts by Color')
ax.legend()
plt.tight_layout()
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
).fit(method=['lbfgs'])
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
dd = df.groupby(["sub", "arrow", "proba"])[["aSPv"]].mean().reset_index()
# %%
# fig = plt.figure()
# # Toggle full screen mode
# figManager = plt.get_current_fig_manager()
# figManager.full_screen_toggle()
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
plt.savefig(pathFig + "/aSPvarrowsFullProba.png",dpi=300, transparent=True)
plt.show()
# %%
# %%
def statistic(x, y,axis):

    return np.mean(x,axis=axis) - np.mean(y,axis=axis)
# %%
# ddd=dd[~( dd['sub']==6 )]
x=dd[( dd['arrow']=='up' )& (dd['proba']==0.25)]['aSPv'].values
y=dd[( dd['arrow']=='up' )& ( dd['proba']==0.75 )]['aSPv'].values
statistic(x,y,0)
# %%
res = permutation_test(data=(x, y), statistic=statistic,permutation_type='samples', vectorized=True,n_resamples=2**(len(x)),)
print(res.statistic)
# %%
print(res.pvalue)
# %%
g=sns.catplot(
    data=dd,
    x="proba",
    y="aSPv",
    hue="arrow",
    kind="bar",
    errorbar='se',
    # errorbar='se',
    n_boot=1000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=["down", "up"],
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
order=[0.25,0.5,0.75]
hue_order=["down", "up"]
pairs = [
    ((0.25, "down"), (0.25, "up")),
    ((0.5, "down"), (0.5, "up")),
    ((0.75, "down"), (0.75, "up")),
    # ((0.25, "Down"), (0.5, "Down")),
    # ((0.75, "Down"), (0.5, "Down")),
    # ((0.25, "Up"), (0.5, "Up")),
    # ((0.75, "Up"), (0.5, "Up"))


]
annotator = Annotator(g.ax, pairs, data=dd, x='proba', y="aSPv", hue="arrow",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
plt.xlabel(r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", fontsize=25)
plt.ylabel("Horizontal aSPv (deg/s)", fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="Arrow", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvarrows.png",dpi=300, transparent=True)
plt.show()
# %%
dd[dd.proba==0.25]
# %%
hue_order=[0.25,0.5,0.75]
order=["down", "up"]
g=sns.catplot(
    data=df,
    x="arrow",
    y="aSPv",
    hue="proba",
    kind="bar",
    errorbar=("ci", 95),
    # errorbar='se',
    n_boot=10000,
    height=10,  # Set the height of the figure
    aspect=1.5,
    capsize=0.1,
    hue_order=hue_order,
    fill=False,
    legend=False,
    palette='viridis',
)

sns.stripplot(
    x="arrow",
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
pairs = [
    (("down",0.25), ('down',0.50)),
    (("down",0.75), ('down',0.50)),
    (("down",0.25), ('down',0.75)),

    (("up",0.25), ('up',0.50)),
    (("up",0.75), ('up',0.50)),
    (("up",0.25), ('up',0.75)),
]
annotator = Annotator(g.ax, pairs, data=dd, x='arrow', y="aSPv", hue="proba",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired',  loc='outside',fontsize=20,comparisons_correction='HB')
annotator.apply_and_annotate()
# plt.title("Horizontal aSPv Across 5 Probabilities", fontsize=30)
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
    handles=legend_elements, fontsize=20, title=r"$\mathbb{P}$(Right|Up)=$\mathbb{P}$(Left|Down)", title_fontsize=20
)
plt.tight_layout()
plt.savefig(pathFig + "/aSPvarrowsbis.png",dpi=300, transparent=True)
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

order=[0.25,0.5,0.75]
hue_order=["left", "right"]
pairs = [
    ((0.25, "left"), (0.25, "right")),
    ((0.5, "left"), (0.5, "right")),
    ((0.75, "left"), (0.75, "right")),
]
annotator = Annotator(g.ax, pairs, data=dd[dd.arrow == "down"], x='proba', y="aSPv", hue="TD_prev",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()

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
plt.savefig(pathFig + "/aSPvdownTD.png",dpi=300, transparent=True)
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

order=[0.25,0.5,0.75]
hue_order=["left", "right"]
pairs = [
    ((0.25, "left"), (0.25, "right")),
    ((0.5, "left"), (0.5, "right")),
    ((0.75, "left"), (0.75, "right")),

]
annotator = Annotator(g.ax, pairs, data=dd[dd.arrow == "up"], x='proba', y="aSPv", hue="TD_prev",hue_order=hue_order, order=order)
annotator.configure(test='t-test_paired', text_format='star', loc='outside',fontsize=20)
annotator.apply_and_annotate()


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
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=25)
g.ax.set_ylabel("Horizontal aSPv (deg/s)", fontsize=25)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvupTD.png",dpi=300, transparent=True)
plt.show()
# %%
pivot_df = dd.pivot_table(
    index=['sub', 'proba', 'arrow'],
    columns='TD_prev',
    values='aSPv'
)

# Step 2: Calculate the difference (right - left)
pivot_df['Diff'] = pivot_df['right'] - pivot_df['left']

# Step 3: Reset index to get back to a regular dataframe format
result_df = pivot_df.reset_index()

# Display the result
print(result_df)
# %%
# Create the plot using catplot
g = sns.catplot(
    data=result_df,
    x="proba",
    y="Diff",
    hue="arrow",
    kind="bar",
    errorbar='se',
    n_boot=1000,
    palette=[downarrowsPalette[0],uparrowsPalette[0]],  # Same color for both
    height=10,  # Set the height of the figure
    aspect=1.5,
    alpha=0.8,
    capsize=0.1,
    hue_order=["down", "up"],
    legend=False,
    fill=False,
)

# # Add hatching to the right bars
# for i, bar in enumerate(g.ax.patches[len(g.ax.patches) // 2 :]):  # Second half of bars
#     bar.set_facecolor("none")  # Make bar empty
#     bar.set_hatch("///")  # Add diagonal lines
#     bar.set_edgecolor(uparrowsPalette[0])  # Set edge color

sns.stripplot(
    data=result_df,
    x="proba",
    y="Diff",
    hue="arrow",
    hue_order=["down", "up"],
    palette=[downarrowsPalette[0],uparrowsPalette[0]],  # Same color for both
    dodge=True,
    jitter=True,
    linewidth=1,
    size=6,
    legend=False,
)

# Create custom legend

legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
]
g.ax.legend(
    handles=legend_elements, fontsize=20, title="arrow", title_fontsize=20
)
plt.tight_layout()

# Customize the plot
# g.ax.set_title("Anticipatory Velocity Given Previous TD: arrow Up", fontsize=30)
g.ax.set_xlabel(r"$\mathbb{P}$(Right|Up)", fontsize=25)
g.ax.set_ylabel(" aSPv Diff (Previous TD: Right - Left ) (deg/s)", fontsize=20)
g.ax.tick_params(labelsize=25)
# g.ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(pathFig + "/aSPvTDiff.png",dpi=300, transparent=True)
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
pivot_df = learningCurveInteraction.pivot_table(
    index=['sub', 'proba', 'arrow'],
    columns='interaction',
    values='aSPv'
)

# Step 2: Calculate the difference (right - left)
pivot_df['DiffDown'] = pivot_df[('right','down')] - pivot_df[('left','down')]
pivot_df['DiffUp'] = pivot_df[('right','up')] - pivot_df[('left','up')]

# Step 3: Reset index to get back to a regular dataframe format
result_df = pivot_df.reset_index()

# Display the result
print(result_df)

# %% 

result_df.dropna(inplace=True)
# %%


def perm_test(x, y, n_perms=1000, stat=np.nanmean):
    diff = np.abs(stat(x)-stat(y))
    pooled = np.concatenate([x, y])
    count = 0
    for n in range(n_perms):
        rng = np.random.default_rng(n)
        rng.shuffle(pooled)
        x_perm = pooled[:len(x)]
        y_perm = pooled[len(x):]
        diff_perm = np.abs(stat(x_perm)-stat(y_perm))
        if diff_perm >= diff:
            count += 1

    pv = count / n_perms
    return pv
# %%
perm_test(result_df[result_df['arrow']=='up']['DiffUp'].values,result_df[result_df['arrow']=='up']['DiffDown'].values)
# %%
perm_test(result_df[result_df['arrow']=='down']['DiffUp'].values,result_df[result_df['arrow']=='down']['DiffDown'].values)
# %%
perm_test(result_df[result_df['arrow']=='up']['DiffUp'].values,result_df[result_df['arrow']=='down']['DiffDown'].values)
# %%
perm_test(result_df[result_df['arrow']=='up']['DiffDown'].values,result_df[result_df['arrow']=='down']['DiffUp'].values)
# %%
melted_df = pd.melt(
    result_df,
    id_vars=["sub", "proba", "arrow"],
    value_vars=["DiffDown", "DiffUp"],
    var_name="DiffType",
    value_name="Diff"
)


print(melted_df)
# Set up the figure
# %%
# Create grouped bar plot
g = sns.catplot(
    data=melted_df,
    x="proba",
    y="Diff",
    hue="arrow",
    col="DiffType",  # Split by DiffType
    kind="bar",
    errorbar='ci',
    n_boot=1000,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    height=8,
    aspect=1.2,
    alpha=0.8,
    capsize=0.1,
    hue_order=["down", "up"],
    fill=False,
    legend=False,
)

# Add individual data points as strips
for ax in g.axes.flat:
    diff_type = ax.get_title().split(" ")[-1]  # Get the DiffType from title
    sns.stripplot(
        data=melted_df[melted_df["DiffType"] == diff_type],
        x="proba",
        y="Diff",
        hue="arrow",
        hue_order=["down", "up"],
        palette=[downarrowsPalette[0], uparrowsPalette[0]],
        dodge=True,
        jitter=True,
        linewidth=1,
        size=6,
        legend=False,
        ax=ax
    )

# Create custom legend
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
]

# Add legend to the figure
plt.legend(
    handles=legend_elements,
    fontsize=15,
    title="arrow",
    title_fontsize=15,
    # loc='upper center',
    # bbox_to_anchor=(0.5, 1.05),
    # ncol=2
)

# Customize the plot titles and labels
g.set_titles("{col_name}")
g.set_axis_labels(r"$\mathbb{P}$(Right|Up)", "aSPv Diff (Previous TD: Right - Left) (deg/s)")
g.set_xticklabels(fontsize=20)
g.set_yticklabels(fontsize=20)

# Customize labels and ticks
for ax in g.axes.flat:
    ax.set_xlabel(r"$\mathbb{P}_{Up}$(Right)", fontsize=25)
    ax.set_ylabel("aSPv Diff (Previous Trial: (Cue,Right) - (Cue,Left)) (deg/s)", fontsize=15)
    ax.tick_params(labelsize=20)
    
    # Add statistical annotations if needed
    order = [0.25, 0.5, 0.75]
    hue_order = ["down", "up"]
    pairs = [
        ((0.25, "down"), (0.25, "up")),
        ((0.5, "down"), (0.5, "up")),
        ((0.75, "down"), (0.75, "up")),
    ]
    
    # Add statistical annotations
    diff_type = ax.get_title().split(" ")[-1]
    subset_df = melted_df[melted_df["DiffType"] == diff_type]

g.set_titles("")
plt.tight_layout()
plt.savefig("aSPvTDiff_combined.png", dpi=300, transparent=True)
plt.show()
# %%

# Create a combined plot where one type uses hatching
g = sns.catplot(
    data=melted_df,
    x="proba",
    y="Diff",
    hue="arrow",
    col="DiffType",
    kind="bar",
    errorbar='ci',
    n_boot=1000,
    palette=[downarrowsPalette[0], uparrowsPalette[0]],
    height=8,
    aspect=1.,
    alpha=0.8,
    capsize=0.1,
    hue_order=["down", "up"],
    legend=False,
    fill=False,
)

# Apply different patterns to distinguish between DiffDown and DiffUp
for i, ax in enumerate(g.axes.flat):
    bars = ax.patches
    diff_type = ax.get_title().split(" ")[-1]
    
    # Apply hatching to one of the diff types
    if diff_type == "DiffUp":
        for bar in bars:
            bar.set_hatch("///")
            
    # Add individual data points
    sns.stripplot(
        data=melted_df[melted_df["DiffType"] == diff_type],
        x="proba",
        y="Diff",
        hue="arrow",
        hue_order=["down", "up"],
        palette=[downarrowsPalette[0], uparrowsPalette[0]],
        dodge=True,
        jitter=True,
        linewidth=1,
        size=6,
        legend=False,
        ax=ax
    )

# Create custom legend with both arrow and DiffType
legend_elements = [
    Patch(facecolor=downarrowsPalette[0], alpha=1, label="Down"),
    Patch(facecolor=uparrowsPalette[0], alpha=1, label="Up"),
    Patch(facecolor='gray', hatch="///", alpha=0.6, label="DiffUp"),
    Patch(facecolor='gray', alpha=0.6, label="DiffDown"),
]

# Add legend to the figure
g.figure.legend(
    handles=legend_elements,
    fontsize=15,
    title="Types",
    title_fontsize=15,
    loc='upper center',
    bbox_to_anchor=(0.88, .97),
    ncol=2
)

# Customize the plot titles and labels
g.set_titles("{col_name}")
g.set_axis_labels(r"$\mathbb{P}_{Up}$(Right)", "aSPv Diff (Previous TD: Right - Left) (deg/s)")

# Customize labels and ticks
for ax in g.axes.flat:
    ax.set_xlabel(r"$\mathbb{P}_{Up}$(Right)", fontsize=25)
    ax.set_ylabel("aSPv Diff (Previous Trial: (Cue,Right) - (Cue,Left)) (deg/s)", fontsize=15)
    ax.tick_params(labelsize=20)
    
    # Add statistical annotations if needed
    order = [0.25, 0.5, 0.75]
    hue_order = ["Down", "Up"]
    pairs = [
        ((0.25, "Down"), (0.25, "Up")),
        ((0.5, "Down"), (0.5, "Up")),
        ((0.75, "Down"), (0.75, "Up")),
    ]
    

g.set_titles("")
plt.tight_layout()
plt.savefig("aSPvTDiff_patterned.png", dpi=300, transparent=True)
plt.show()
# %%
df["interaction"].unique()
# %%
hue_order = [
    ("left", "down"),
    ("left", "up"),
    ("right", "down"),
    ("right", "up"),
]
colorsPalettes = ["#0F68A9", "#FAAE7B","#0F68A9", "#FAAE7B"]
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
# plt.tight_layout()
plt.savefig(pathFig + "/aSPvUpInteraction.png",dpi=300, transparent=True,bbox_inches='tight')
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
plt.savefig(pathFig + "/aSPvDownInteraction.png",dpi=300, transparent=True)
plt.show()
# %%
dd = df.groupby(["sub", "proba", "arrow", "TD_prev"])["aSPv"].mean().reset_index()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.25],
    re_formula="~arrow*TD_prev",
    groups=df[df.proba == 0.25]["sub"],
).fit(method=['lbfgs'])
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.75],
    re_formula="~arrow*TD_prev",
    groups=df[df.proba == 0.75]["sub"],
).fit(method=[ "lbfgs" ])
model.summary()
# %%
model = smf.mixedlm(
    "aSPv~  C(arrow)*C(TD_prev)",
    data=df[df.proba == 0.5],
    re_formula="~arrow*TD_prev",
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

