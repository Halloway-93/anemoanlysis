import os
import sys
import h5py
import time as timer
import numpy as np
import pandas as pd
from functions.utils import *
from ANEMO.ANEMO import ANEMO, read_edf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import multiprocessing as mp
from functools import partial
import warnings
import traceback

warnings.filterwarnings("ignore")

# ANEMO parameters
screen_width_px = 1920  # px
screen_height_px = 1080  # px
screen_width_cm = 70  # cm
viewingDistance = 57.0  # cm

tan = np.arctan((screen_width_cm / 2) / viewingDistance)
screen_width_deg = 2.0 * tan * 180 / np.pi
px_per_deg = screen_width_px / screen_width_deg

param_exp = {
    "N_trials": 1,
    "N_blocks": 1,
    "dir_target": [[], []],  # will be defined in the main loop
    "px_per_deg": px_per_deg,
    "screen_width": screen_width_px,
    "screen_height": screen_height_px,
    "list_events": ["StimulusOn\n", "StimulusOff\n", "TargetOn\n", "TargetOff\n"],
}

subjects = [
    "sub-001",
    "sub-002",
    "sub-003",
    "sub-004",
    "sub-005",
    "sub-006",
    "sub-007",
    "sub-008",
    "sub-009",
    "sub-010",
    "sub-011",
]

conditions = ["c1", "c2", "c3"]  # Balanced cues
time_sup = 10  # time window to cut at the end of the trial


# Define unified saccade parameters
def get_unified_sacc_params(subjects):
    """Create unified saccade parameters for all subjects."""
    base_params = {
        "mindur": 5,
        "maxdur": 100,
        "minsep": 30,
        "before_sacc": 20,
        "after_sacc": 20,
    }

    return {
        int(subject.split("-")[1]): base_params.copy()
        for subject in subjects
        if subject.startswith("sub-")
    }


sacc_params = get_unified_sacc_params(subjects)

# Global settings
showPlots = 0
manualCheck = 0
equation = "sigmoid"
allow_baseline, allow_horizontalShift, allow_acceleration = True, True, False


def process_trial(
    A,
    Fit,
    sacc_params,
    trial,
    data,
    motionFirstSegment,
    motionSecondSegment,
    param_exp,
    sub,
    cond,
    subNumber,
    paramsRaw,
    qualityCtrl,
    paramsSub,
    pdf,
    nanOnsetpdf,
    nanOverallpdf,
    nanSequencepdf,
    firstTrial,
    proba,
):
    """Process a single trial and return updated dataframes."""
    print(f"Trial {trial}, cond {cond}, sub {sub}")

    try:
        if len(data[trial]["x"]) and len(data[trial]["y"]):
            data[trial]["y"] = screen_height_px - data[trial]["y"]

            type_dir1 = "Up" if motionFirstSegment[trial] == 1 else "Down"
            type_dir2 = "R" if param_exp["dir_target"][trial] == 1 else "L"

            trialType_txt = f"{type_dir1}{type_dir2}"

            # Get trial data and transform into the arg
            arg = A.arg(data_trial=data[trial], trial=trial, block=0)

            # Index of TargetOnset
            TargetOnIndex = arg.TargetOn + 600 - arg.t_0
            print("Start of the second segment", TargetOnIndex)

            pos_deg_x = A.data_deg(
                data=arg.data_x,
                StimulusOf=arg.StimulusOf,
                t_0=arg.t_0,
                saccades=arg.saccades,
                before_sacc=sacc_params[1]["before_sacc"],
                after_sacc=sacc_params[1]["after_sacc"],
                filt=None,
                cutoff=30,
                sample_rate=1000,
            )

            pos_deg_y = A.data_deg(
                data=arg.data_y,
                StimulusOf=arg.StimulusOf,
                t_0=arg.t_0,
                saccades=arg.saccades,
                before_sacc=sacc_params[1]["before_sacc"],
                after_sacc=sacc_params[1]["after_sacc"],
                filt=None,
                cutoff=30,
                sample_rate=1000,
            )

            velocity_deg_x = A.velocity(
                data=pos_deg_x,
                filter_before=True,
                filter_after=False,
                cutoff=30,
                sample_rate=1000,
            )

            velocity_deg_y = A.velocity(
                data=pos_deg_y,
                filter_before=True,
                filter_after=False,
                cutoff=30,
                sample_rate=1000,
            )

            misac = A.detec_misac(
                velocity_x=velocity_deg_x,
                velocity_y=velocity_deg_y,
                t_0=arg.t_0,
                VFAC=5,
                mindur=sacc_params[1]["mindur"],
                maxdur=sacc_params[1]["maxdur"],
                minsep=sacc_params[1]["minsep"],
            )

            new_saccades = arg.saccades
            [
                sacc.extend([0, 0, 0, 0, 0]) for sacc in misac
            ]  # transform misac into the eyelink format
            new_saccades.extend(misac)

            sac = A.detec_sac(
                velocity_x=velocity_deg_x,
                velocity_y=velocity_deg_y,
                t_0=arg.t_0,
                VFAC=5,
                mindur=sacc_params[1]["mindur"],
                maxdur=sacc_params[1]["maxdur"],
                minsep=sacc_params[1]["minsep"],
            )

            [
                sacc.extend([0, 0, 0, 0, 0]) for sacc in sac
            ]  # transform misac into the eyelink format
            new_saccades.extend(sac)

            blinks = data[trial]["events"]["Eblk"].copy()
            if len(data[trial]["events"]["Sblk"]) > len(data[trial]["events"]["Eblk"]):
                blinks.append(
                    list(
                        [
                            data[trial]["events"]["Sblk"][-1],
                            data[trial]["trackertime"][-1],
                        ]
                    )
                )
            blinks = [x[:2] for x in blinks]

            velocity_x_NAN = A.data_NAN(
                data=velocity_deg_x,
                saccades=new_saccades,
                trackertime=arg.trackertime,
                before_sacc=sacc_params[1]["before_sacc"],
                after_sacc=sacc_params[1]["after_sacc"],
            )

            velocity_y_NAN = A.data_NAN(
                data=velocity_deg_y,
                saccades=new_saccades,
                trackertime=arg.trackertime,
                before_sacc=sacc_params[1]["before_sacc"],
                after_sacc=sacc_params[1]["after_sacc"],
            )

            time = arg.trackertime - (arg.TargetOn + 600)
            print("time:", time[TargetOnIndex])

            idx2keep_y = np.logical_and(time >= -600, time <= 600)
            pos_y = arg.data_y[idx2keep_y]
            vel_y = velocity_y_NAN[idx2keep_y]

            idx2keep_x = np.logical_and(time >= -600, time <= 600)
            pos_x = arg.data_x[idx2keep_x]
            vel_x = velocity_x_NAN[idx2keep_x]
            time = time[idx2keep_x]
            pos_deg_x = pos_deg_x[idx2keep_x]
            pos_deg_y = pos_deg_y[idx2keep_y]
            time = time[idx2keep_x]

            # Calc saccades relative to t_0
            for sacc in new_saccades:
                sacc[0] = sacc[0] - (arg.TargetOn + 600)
                sacc[1] = sacc[1] - (arg.TargetOn + 600)

            sDict = {
                "condition": cond,
                "proba": proba,
                "trial": trial,
                "trialType": trialType_txt,
                "direction": param_exp["dir_target"][trial],
                "time": time,
                "posPxl_x": pos_x,
                "posPxl_y": pos_y,
                "posDeg_x": pos_deg_x,
                "posDeg_y": pos_deg_y,
                "velocity_x": vel_x,
                "velocity_y": vel_y,
                "saccades": new_saccades,
            }

            # Save trial data to a dataframe
            if firstTrial:
                paramsRaw_local = pd.DataFrame([sDict], columns=sDict.keys())
            else:
                paramsRaw_local = pd.DataFrame([sDict], columns=sDict.keys())

            # Test: if bad trial
            # Getting the newTargetOnset index
            newTargetOnset = np.where(time == 0)[0][0]

            if (
                np.mean(np.isnan(vel_x[newTargetOnset - 100 : newTargetOnset + 100]))
                > 0.7
                or np.mean(np.isnan(vel_x[:-time_sup])) > 0.5
                or longestNanRun(vel_x[newTargetOnset - 150 : newTargetOnset + 600])
                > 200
                or abs(np.nanmean(vel_x[newTargetOnset + 300 : newTargetOnset + 600]))
                < 4
            ):
                print("Skipping bad trial...")

                plt.clf()
                fig = plt.figure(figsize=(12, 4))
                plt.suptitle(f"Trial {trial}")
                plt.subplot(1, 2, 1)
                plt.plot(time, vel_x)
                plt.axvline(x=time[0], linewidth=1, linestyle="--", color="k")
                plt.axvline(x=time[-1], linewidth=1, linestyle="--", color="k")
                plt.ylim(-15, 15)
                plt.xlabel("Time (ms)")
                plt.ylabel("Velocity - x axis")
                plt.subplot(1, 2, 2)
                plt.plot(time, vel_y)
                plt.axvline(x=time[0], linewidth=1, linestyle="--", color="k")
                plt.axvline(x=time[-1], linewidth=1, linestyle="--", color="k")
                plt.ylim(-35, 35)
                plt.xlabel("Time (ms)")
                plt.ylabel("Velocity - y axis")

                print("Time", newTargetOnset)
                reason = ""

                if (
                    np.mean(
                        np.isnan(vel_x[newTargetOnset - 100 : newTargetOnset + 100])
                    )
                    > 0.7
                ):
                    print("too many NaNs around the start of the pursuit")
                    reason = reason + " >.70 of NaNs around the start of the pursuit"
                    nanOnsetpdf.savefig(fig)
                if np.mean(np.isnan(vel_x[:-time_sup])) > 0.6:
                    print("too many NaNs overall")
                    reason = reason + f" >{0.6} of NaNs overall"
                    nanOverallpdf.savefig(fig)
                if (
                    longestNanRun(vel_x[newTargetOnset - 150 : newTargetOnset + 600])
                    > 200
                ):
                    print("at least one nan sequence with more than 200ms")
                    reason = reason + " At least one nan sequence with more than 200ms"
                    nanSequencepdf.savefig(fig)
                if (
                    abs(np.nanmean(vel_x[newTargetOnset + 300 : newTargetOnset + 600]))
                    < 4
                ):
                    print("No smooth pursuit")
                    reason = reason + " No smooth pursuit"
                    nanSequencepdf.savefig(fig)

                plt.close(fig)

                newResult = dict()
                newResult["condition"] = cond
                newResult["trial"] = trial
                newResult["trialType"] = trialType_txt
                newResult["target_dir"] = param_exp["dir_target"][trial]

                x = arg.trackertime - arg.TargetOn - 600
                newResult["time"] = x[:-time_sup]
                newResult["velocity_x"], newResult["velocity_y"] = (
                    velocity_x_NAN[:-time_sup],
                    velocity_y_NAN[:-time_sup],
                )
                newResult["saccades"] = np.array(new_saccades)

                qCtrl = dict()
                qCtrl["sub"] = sub
                qCtrl["condition"] = cond
                qCtrl["trial"] = trial
                qCtrl["keep_trial"] = 0
                qCtrl["good_fit"] = 0
                qCtrl["discard_reason"] = reason

            else:  # if not a bad trial, does the fit
                classic_lat_x, classic_max_x, classic_ant_x = A.classical_method.Full(
                    vel_x, 200
                )
                classic_ant = classic_ant_x if not np.isnan(classic_ant_x) else 0.5

                if equation == "sigmoid":
                    param_fit, inde_var = Fit.generation_param_fit(
                        equation="fct_velocity_sigmo",
                        dir_target=param_exp["dir_target"][trial],
                        trackertime=time,
                        TargetOn=0,
                        StimulusOf=time[0],
                        saccades=new_saccades,
                        value_latency=classic_lat_x - 200,
                        value_steady_state=classic_max_x,
                        value_anti=classic_ant * 5,
                    )

                    result_x = Fit.Fit_trial(
                        vel_x,
                        equation="fct_velocity_sigmo",
                        dir_target=int(param_exp["dir_target"][trial]),
                        trackertime=list(inde_var["x"]),
                        TargetOn=0,
                        StimulusOf=time[0],
                        saccades=new_saccades,
                        time_sup=None,
                        step_fit=2,
                        param_fit=param_fit,
                        inde_vars=inde_var,
                        value_latency=classic_lat_x - 200,
                        value_steady_state=classic_max_x,
                        value_anti=classic_ant * 5,
                        allow_baseline=True,
                        allow_horizontalShift=True,
                    )

                    eq_x_tmp = ANEMO.Equation.fct_velocity_sigmo(
                        x=inde_var["x"],
                        t_0=result_x.params["t_0"],
                        t_end=result_x.params["t_end"],
                        dir_target=result_x.params["dir_target"],
                        baseline=result_x.params["baseline"],
                        start_anti=result_x.params["start_anti"],
                        a_anti=result_x.params["a_anti"],
                        latency=result_x.params["latency"],
                        ramp_pursuit=result_x.params["ramp_pursuit"],
                        horizontal_shift=result_x.params["horizontal_shift"],
                        steady_state=result_x.params["steady_state"],
                        allow_baseline=result_x.params["allow_baseline"],
                        allow_horizontalShift=result_x.params["allow_horizontalShift"],
                    )

                    eq_x_tmp = np.array(eq_x_tmp)
                    eq_x = np.zeros(len(time))
                    eq_x[:] = np.nan
                    eq_x[: len(eq_x_tmp)] = eq_x_tmp

                newResult = dict()
                newResult["condition"] = cond
                newResult["proba"] = proba
                newResult["trial"] = trial
                newResult["trialType"] = trialType_txt
                newResult["target_dir"] = param_exp["dir_target"][trial]
                newResult["time"] = time
                newResult["velocity_x"] = vel_x
                newResult["saccades"] = np.array(new_saccades)

                newResult["t_0"] = result_x.params["t_0"].value
                newResult["t_end"] = result_x.params["t_end"].value
                newResult["dir_target"] = result_x.params["dir_target"].value
                newResult["baseline"] = result_x.params["baseline"].value
                newResult["aSPon"] = np.round(result_x.params["start_anti"].value)
                newResult["aSPv_slope"] = result_x.params["a_anti"].value
                newResult["SPacc"] = result_x.params["ramp_pursuit"].value
                newResult["SPss"] = result_x.params["steady_state"].value
                newResult["do_whitening_x"] = result_x.params["do_whitening"].value

                if equation == "sigmoid":
                    newResult["aSPoff"] = np.round(result_x.params["latency"].value)
                    newResult["horizontal_shift"] = result_x.params[
                        "horizontal_shift"
                    ].value

                    idx_aSPoff = np.where(time == newResult["aSPoff"])[0][0]
                    vel_at_latency = (
                        eq_x[idx_aSPoff]
                        + (newResult["SPss"] - eq_x[idx_aSPoff])
                        * 0.05
                        * newResult["dir_target"]
                    )
                    vel, idx = closest(eq_x[idx_aSPoff + 1 :], vel_at_latency)

                    newResult["SPlat"] = time[idx + idx_aSPoff + 1]
                    newResult["aSPv"] = eq_x[idx_aSPoff]
                    newResult["allow_baseline"] = allow_baseline
                    newResult["allow_horizontalShift"] = allow_horizontalShift

                newResult["aic_x"] = result_x.aic
                newResult["bic_x"] = result_x.bic
                newResult["chisqr_x"] = result_x.chisqr
                newResult["redchi_x"] = result_x.redchi
                newResult["residual_x"] = result_x.residual
                newResult["rmse_x"] = np.sqrt(
                    np.mean([x * x for x in result_x.residual])
                )
                newResult["classic_lat_x"] = classic_lat_x
                newResult["classic_max_x"] = classic_max_x
                newResult["classic_ant_x"] = classic_ant_x
                newResult["fit_x"] = eq_x

                f = plotFig(
                    trial,
                    newResult["target_dir"],
                    newResult["time"],
                    newResult["velocity_x"],
                    eq_x,
                    newResult["aSPon"],
                    newResult["aSPoff"],
                    newResult["SPlat"],
                    show=showPlots,
                )

                pdf.savefig(f)
                plt.close(f)

                qCtrl = dict()
                qCtrl["sub"] = sub
                qCtrl["condition"] = cond
                qCtrl["trial"] = trial

                if newResult["rmse_x"] > 10:
                    qCtrl["keep_trial"] = np.nan
                    qCtrl["good_fit"] = 0
                    qCtrl["discard_reason"] = "RMSE > 10"
                elif manualCheck:
                    qCtrl["keep_trial"] = int(
                        input("Keep trial? \npress (1) to keep or (0) to discard\n ")
                    )
                    while qCtrl["keep_trial"] != 0 and qCtrl["keep_trial"] != 1:
                        qCtrl["keep_trial"] = int(
                            input(
                                "Keep trial? \npress (1) to keep or (0) to discard\n "
                            )
                        )

                    qCtrl["good_fit"] = int(
                        input("Good fit? \npress (1) for yes or (0) for no\n ")
                    )
                    while qCtrl["good_fit"] != 0 and qCtrl["keep_trial"] != 1:
                        qCtrl["good_fit"] = int(
                            input("Good fit? \npress (1) for yes or (0) for no\n ")
                        )

                    qCtrl["discard_reason"] = np.nan
                else:
                    qCtrl["keep_trial"] = np.nan
                    qCtrl["good_fit"] = np.nan
                    qCtrl["discard_reason"] = np.nan

            # Save trial's fit data to a dataframe
            if firstTrial:
                paramsSub_local = pd.DataFrame([newResult], columns=newResult.keys())
                qualityCtrl_local = pd.DataFrame([qCtrl], columns=qCtrl.keys())
                firstTrial = False
            else:
                paramsSub_local = pd.DataFrame([newResult], columns=newResult.keys())
                qualityCtrl_local = pd.DataFrame([qCtrl], columns=qCtrl.keys())

            return {
                "paramsRaw": paramsRaw_local,
                "paramsSub": paramsSub_local,
                "qualityCtrl": qualityCtrl_local,
                "firstTrial": False,
            }

        return None  # Return None if there's no data for this trial

    except Exception as e:
        print(f"Error processing trial {trial} for {sub}, condition {cond}")
        traceback.print_exc()
        return None


def process_condition(sub, cond, main_dir):
    """Process a single subject and condition."""
    try:
        subNumber = int(sub.split("-")[1])
        print(f"Processing subject: {sub}, condition: {cond}")

        # Setup file paths
        h5_file = f"{sub}/CP_s{subNumber}{cond}_posFilter.h5"
        h5_rawfile = f"{sub}/CP_s{subNumber}{cond}_rawData.h5"
        h5_qcfile = f"{sub}/CP_s{subNumber}{cond}_qualityControl.h5"

        dataFile = f"{main_dir}/{sub}/CP_s{subNumber}{cond}.asc"
        tgDirFile = f"{main_dir}/{sub}/{sub}_{cond}.tsv"

        outputFolder_plots = f"{sub}/plots/"
        fitPDFFile = f"{outputFolder_plots}{sub}_{cond}_fit.pdf"

        # Create output directories
        os.makedirs(outputFolder_plots, exist_ok=True)
        os.makedirs(f"{outputFolder_plots}qc/", exist_ok=True)

        # Setup PDF files for quality control
        nanOverallFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanOverall.pdf"
        nanOnsetFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanOnset.pdf"
        nanSequenceFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanSequence.pdf"

        nanOverallpdf = PdfPages(nanOverallFile)
        nanOnsetpdf = PdfPages(nanOnsetFile)
        nanSequencepdf = PdfPages(nanSequenceFile)
        pdf = PdfPages(fitPDFFile)

        # Read data
        data = read_edf(dataFile, start="StimulusOff", stop="TargetOff")
        motionDirection = pd.read_csv(tgDirFile, sep="\t")
        motionFirstSegment = motionDirection["firstSegmentMotion"].values
        motionSecondSegment = motionDirection["secondSegmentMotion"].values

        # Change directions from 0/1 diagonals to -1/1
        param_exp["dir_target"] = [x if x == 1 else -1 for x in motionSecondSegment]
        param_exp["N_trials"] = len(data)

        # Create ANEMO instance
        A = ANEMO(param_exp)
        Fit = A.Fit(param_exp)

        # Set probability based on condition
        if cond == "c1":
            proba = 0.5
        elif cond == "c2":
            proba = 0.75
        else:
            proba = 0.25

        # Initialize dataframes
        paramsRaw = pd.DataFrame()
        paramsSub = pd.DataFrame()
        qualityCtrl = pd.DataFrame()
        firstTrial = True

        # Process each trial
        all_trial_results = []
        for trial in range(param_exp["N_trials"]):
            result = process_trial(
                A,
                Fit,
                sacc_params,
                trial,
                data,
                motionFirstSegment,
                motionSecondSegment,
                param_exp,
                sub,
                cond,
                subNumber,
                paramsRaw,
                qualityCtrl,
                paramsSub,
                pdf,
                nanOnsetpdf,
                nanOverallpdf,
                nanSequencepdf,
                firstTrial,
                proba,
            )
            if result:
                all_trial_results.append(result)
                firstTrial = result["firstTrial"]

        # Combine results from all trials
        if all_trial_results:
            for key in ["paramsRaw", "paramsSub", "qualityCtrl"]:
                frames = [r[key] for r in all_trial_results if r and key in r]
                if frames:
                    locals()[key] = pd.concat(frames, ignore_index=True)

        # Close PDF files
        nanOnsetpdf.close()
        nanOverallpdf.close()
        nanSequencepdf.close()
        pdf.close()
        plt.close("all")

        # Save results to HDF files
        if not paramsSub.empty:
            paramsSub.to_hdf(h5_file, "data")
        if not paramsRaw.empty:
            paramsRaw.to_hdf(h5_rawfile, "data")
        if not qualityCtrl.empty:
            qualityCtrl.to_hdf(h5_qcfile, "data")

        # Test if it can read the file
        abc = pd.read_hdf(h5_file, "data")
        print(f"Successfully processed {sub}, condition {cond}")

        return True

    except Exception as e:
        print(f"Error! Couldn't process {sub}, condition {cond}")
        traceback.print_exc()
        return False


def process_subject(sub, conditions, main_dir):
    """Process all conditions for a given subject."""
    results = []
    for cond in conditions:
        result = process_condition(sub, cond, main_dir)
        results.append((sub, cond, result))
    return results


def main():
    """Main function to parallelize processing across subjects."""
    print("Current working directory:", os.getcwd())
    print("Contents of current directory:", os.listdir())

    motionDirectionCue = "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue"
    main_dir = motionDirectionCue
    os.chdir(main_dir)

    # Number of processes to use (adjust based on your system's capabilities)
    num_processes = min(mp.cpu_count() - 1, len(subjects))
    print(f"Running with {num_processes} parallel processes")

    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Map the process_subject function to each subject
        process_func = partial(
            process_subject, conditions=conditions, main_dir=main_dir
        )
        results = pool.map(process_func, subjects)

    # Print summary of results
    print("\nProcessing Summary:")
    for subject_results in results:
        for sub, cond, success in subject_results:
            status = "Success" if success else "Failed"
            print(f"Subject: {sub}, Condition: {cond} - {status}")


if __name__ == "__main__":
    main()
