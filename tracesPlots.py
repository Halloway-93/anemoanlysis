import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%

df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/ColorCue/data/sub-05/sub-05_col50-dir25_rawData.h5",
    "data",
)
data = pd.read_csv("/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_activeColorCP.csv")
# %%
dd = data[(data["pR-Red"] == 0.25) & (data["sub"] == "sub-05")]
dd
# %%
# Looking for only the  valid trials from Anemo.
df = df[((df["trial"] - 1).isin(dd["trial"].values))]
# %%
df.columns
# %%
for t, v in zip(df.time_x.values, df.velocity_x.values):
    plt.plot(t, v)
    plt.show()

# %%
greenL = df[df["trialType"] == "GreenL"][["time_x", "velocity_x"]]
greenR = df[df["trialType"] == "GreenR"][["time_x", "velocity_x"]]
green = pd.concat([greenL, greenR], axis=0)
# %%
redL = df[df["trialType"] == "RedL"][["time_x", "velocity_x"]]
redR = df[df["trialType"] == "RedR"][["time_x", "velocity_x"]]
red = pd.concat([redL, redR], axis=0)
# %%

for t, v in zip(greenL.time_x.values, greenL.velocity_x.values):
    plt.plot(t, v)

plt.show()
# %%
for t, v in zip(greenR.time_x.values, greenR.velocity_x.values):
    plt.plot(t, v)

plt.show()
# %%
for t, v in zip(green.time_x.values, green.velocity_x.values):
    plt.plot(t, v)

plt.show()
# %%
arrGreen = np.zeros((len(green), 801))
arrGreen
for i in range(len(arrGreen)):
    arrGreen[i] = green.iloc[i]["velocity_x"]
# %%
greenTrials = np.nanmean(arrGreen, axis=0)
# %%
arrRed = np.zeros((len(red), 801))
for i in range(len(arrRed)):
    arrRed[i] = red.iloc[i]["velocity_x"]
# %%
redTrials = np.nanmean(arrRed, axis=0)
# %%
plt.plot(
    df.time_x.iloc[0][:300], greenTrials[:300], label="Green Trials", color="green"
)
plt.plot(df.time_x.iloc[0][:300], redTrials[:300], label="Red Trials", color="red")
plt.show()
# %%
arrGreenR = np.zeros((len(greenR), 801))
for i in range(len(arrGreenR)):
    arrGreenR[i] = greenR.iloc[i]["velocity_x"]
greenTrialsR = np.nanmean(arrGreenR, axis=0)
# %%
arrGreenL = np.zeros((len(greenL), 801))
for i in range(len(arrGreenL)):
    arrGreenL[i] = greenL.iloc[i]["velocity_x"]
greenTrialsL = np.nanmean(arrGreenL, axis=0)
# %%
arrRedR = np.zeros((len(redR), 801))
for i in range(len(arrRedR)):
    arrRedR[i] = redR.iloc[i]["velocity_x"]
redTrialsR = np.nanmean(arrRedR, axis=0)
# %%
arrRedL = np.zeros((len(redL), 801))
for i in range(len(arrRedL)):
    arrRedL[i] = redL.iloc[i]["velocity_x"]
redTrialsL = np.nanmean(arrRedL, axis=0)
# %%
plt.figure(figsize=(12, 6))
plt.plot(df.time_x.iloc[0], redTrialsL, label="Red Trials Left", color="red", ls="--")
plt.plot(df.time_x.iloc[0], greenTrialsR, label="Green Trials Right", color="green")
plt.plot(
    df.time_x.iloc[0], greenTrialsL, label="Green Trials Left", color="green", ls="--"
)
plt.plot(df.time_x.iloc[0], redTrialsR, label="Red Trials Right", color="red")
plt.legend()
plt.xlabel("Time in ms", fontsize=20)
plt.ylabel("aSPV (deg/s)", fontsize=20)
plt.show()
# %%


# First, let's reshape the data into a format seaborn can use
def create_long_format_data(arr, time_values, cue, direction):
    # Create a long format dataframe for each trial type
    dfs = []
    for i in range(len(arr)):
        df = pd.DataFrame(
            {
                "Time": time_values,
                "Velocity": arr[i],
                "Cue": cue,
                "Direction": direction,
            }
        )
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Combine all conditions into one dataframe
all_data = pd.concat(
    [
        create_long_format_data(arrRedL, df.time_x.iloc[0], "Red", "Left"),
        create_long_format_data(arrRedR, df.time_x.iloc[0], "Red", "Right"),
        create_long_format_data(arrGreenL, df.time_x.iloc[0], "Green", "Left"),
        create_long_format_data(arrGreenR, df.time_x.iloc[0], "Green", "Right"),
    ],
    ignore_index=True,
)

# %%
all_data
# %%
# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=all_data,
    x="Time",
    y="Velocity",
    hue="Cue",
    style="Direction",
    errorbar="ci",
    palette=["red", "green"],
)

plt.xlabel("Time in ms", fontsize=20)
plt.ylabel("aSPV (deg/s)", fontsize=20)
plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")


# Update colors to match your original plot

plt.tight_layout()
plt.show()
# %%
# Doing it for another block
df = pd.read_hdf(
    "/Users/mango/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/sub-009/session-03/rawData.h5",
    "data",
)
data = pd.read_csv("/Users/mango/anemoanlysis/LMM/dataANEMO_allSubs_voluntaryArrow.csv")
# %%
dd = data[(data["session"] == "session-03") & (data["sub"] == 9)]
dd
# %%
df = df[(df["trial"].isin(dd["trial"].values))]
# %%
downL = df[df["trialType"] == "downL"][["time_x", "velocity_x"]]
downR = df[df["trialType"] == "downR"][["time_x", "velocity_x"]]
down = pd.concat([downL, downR], axis=0)
down
# %%
upL = df[df["trialType"] == "upL"][["time_x", "velocity_x"]]
upR = df[df["trialType"] == "upR"][["time_x", "velocity_x"]]
up = pd.concat([upL, upR], axis=0)
up
# %%

for t, v in zip(downL.time_x.values, downL.velocity_x.values):
    plt.plot(t, v)

plt.show()
# %%
for t, v in zip(downR.time_x.values, downR.velocity_x.values):
    plt.plot(t, v)

plt.show()
# %%
for t, v in zip(down.time_x.values, down.velocity_x.values):
    plt.plot(t, v)

plt.show()
# %%
arrDown = np.zeros((len(down), 800))
arrDown
for i in range(len(arrDown)):
    arrDown[i] = down.iloc[i]["velocity_x"]
# %%
downTrials = np.nanmean(arrDown, axis=0)
# %%
arrUp = np.zeros((len(up), 800))
for i in range(len(arrUp)):
    arrUp[i] = up.iloc[i]["velocity_x"]
# %%
upTrials = np.nanmean(arrUp, axis=0)
# %%
plt.plot(df.time_x.iloc[0][:300], upTrials[:300], label="Up Trials", color="orange")
plt.plot(df.time_x.iloc[0][:300], downTrials[:300], label="Down Trials", color="blue")
plt.show()
# %%
arrDownR = np.zeros((len(downR), 800))
for i in range(len(arrDownR)):
    arrDownR[i] = downR.iloc[i]["velocity_x"]
downTrialsR = np.nanmean(arrDownR, axis=0)
# %%
arrDownL = np.zeros((len(downL), 800))
for i in range(len(arrDownL)):
    arrDownL[i] = downL.iloc[i]["velocity_x"]
downTrialsL = np.nanmean(arrDownL, axis=0)
# %%
arrUpR = np.zeros((len(upR), 800))
for i in range(len(arrUpR)):
    arrUpR[i] = upR.iloc[i]["velocity_x"]
upTrialsR = np.nanmean(arrUpR, axis=0)
# %%
arrUpL = np.zeros((len(upL), 800))
for i in range(len(arrUpL)):
    arrUpL[i] = upL.iloc[i]["velocity_x"]
upTrialsL = np.nanmean(arrUpL, axis=0)
# %%
plt.figure(figsize=(12, 6))
plt.plot(df.time_x.iloc[0], upTrialsL, label="Up Trials Left", color="orange", ls="--")
plt.plot(df.time_x.iloc[0], downTrialsR, label="Down Trials Right", color="blue")
plt.plot(
    df.time_x.iloc[0], downTrialsL, label="Down Trials Left", color="blue", ls="--"
)
plt.plot(df.time_x.iloc[0], upTrialsR, label="Up Trials Right", color="orange")
plt.legend()
plt.xlabel("Time in ms", fontsize=20)
plt.ylabel("aSPV (deg/s)", fontsize=20)
plt.show()
# %%
all_data = pd.concat(
    [
        create_long_format_data(arrUpL, df.time_x.iloc[0], "Up", "Left"),
        # create_long_format_data(arrUpR, df.time_x.iloc[0], "Up", "Right"),
        # create_long_format_data(arrDownL, df.time_x.iloc[0], "Down", "Left"),
        create_long_format_data(arrDownR, df.time_x.iloc[0], "Down", "Right"),
    ],
    ignore_index=True,
)

# %%
all_data
# %%
# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=all_data,
    x="Time",
    y="Velocity",
    hue="Cue",
    style="Direction",
    errorbar="ci",
    palette=["orange", "blue"],
)

plt.xlabel("Time in ms", fontsize=20)
plt.ylabel("aSPV (deg/s)", fontsize=20)
plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()
