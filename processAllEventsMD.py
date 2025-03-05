import json
import os
from pathlib import Path
import pandas as pd
import re


def read_file(filepath):
    if filepath.suffix == ".csv":
        print(f"Read data from {filepath}")
        data = pd.read_csv(filepath)
        return data
    elif filepath.suffix == ".tsv":
        print(f"Read data from {filepath}")
        data = pd.read_csv(filepath, sep="\t")
        return data
    elif filepath.suffix == ".json":
        print(f"Read data from {filepath}")
        with open(filepath, "r") as f:
            metadata = json.load(f)
        return metadata
    return None


def process_all_events(data_dir, filename="allEvents.csv"):
    all_events = []
    data_dir_path = Path(data_dir)

    for filepath in sorted(data_dir_path.rglob("*")):
        # print(filepath)
        if filepath.is_file():
            if filepath.suffix == ".tsv" and filepath.stem.startswith("sub"):
                df = read_file(filepath)
                print(int(re.search(r"sub-(\d+)", str(filepath)).group(1)))
                df["sub"] = "sub-" + (re.search(r"sub-(\d+)", str(filepath)).group(1))
                df["cond"] = "c" + (re.search(r"_c(\d+)", str(filepath)).group(1))
                if int(re.search(r"_c(\d+)", str(filepath)).group(1)) == 1:
                    df["proba"] = 0.5
                elif int(re.search(r"_c(\d+)", str(filepath)).group(1)) == 2:
                    df["proba"] = 0.75
                if int(re.search(r"_c(\d+)", str(filepath)).group(1)) == 3:
                    df["proba"] = 0.25
                if "trial" not in df.columns:
                    df["trial"] = [i + 1 for i in range(len(df))]

                all_events.append(df)

    if all_events:
        big_df = pd.concat(all_events, axis=0, ignore_index=True)
        big_df.to_csv(os.path.join(data_dir, filename), index=False)
    else:
        print("No events found to process.")


# Running the code on the server
dirPath = "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue/"
process_all_events(dirPath)
