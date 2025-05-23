import os
import numpy as np
import pandas as pd
from functions.utils import *
from ANEMO.ANEMO import ANEMO, read_edf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import re
import warnings
import traceback
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Define global constants
ACTIVE_DIR = "/Users/mango/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection/"
PASSIVE_DIR = "/Users/mango/oueld.h/contextuaLearning/directionCue/results_imposeDirection/"

# Global parameters
screen_width_px = 1920  # px
screen_height_px = 1080  # px
screen_width_cm = 70  # cm
viewingDistance = 57.0  # cm

tan = np.arctan((screen_width_cm / 2) / viewingDistance)
screen_width_deg = 2.0 * tan * 180 / np.pi
px_per_deg = screen_width_px / screen_width_deg
# the actual value coming from eylink setup
px_per_deg = 27.4620

# Common ANEMO fitting parameters
equation = "sigmoid"
allow_baseline, allow_horizontalShift, allow_acceleration = True, True, False
time_sup = 10  # time window to cut at the end of the trial
showPlots = 0  # int(input("Do you want to see and save the plots for every trial? \npress (1) for yes or (0) for no\n "))
manualCheck = 0


def get_subjects_and_sessions(base_path):
    """
    Scans directory structure to get ordered subject and session information.

    Args:
        base_path (str): Base directory path containing subject folders

    Returns:
        dict: Dictionary with subjects as keys and list of sessions as values
    """
    base_dir = Path(base_path)
    subject_pattern = re.compile(r"sub-(\d+)")
    session_pattern = re.compile(r"session-(\d+)")

    # Get and sort subjects
    subjects = {}
    for item in base_dir.iterdir():
        if item.is_dir():
            subject_match = subject_pattern.match(item.name)
            if subject_match:
                subject_num = int(subject_match.group(1))
                # Get sessions for this subject
                sessions = []
                for session_dir in item.iterdir():
                    if session_dir.is_dir():
                        session_match = session_pattern.match(session_dir.name)
                        if session_match:
                            sessions.append(int(session_match.group(1)))
                if sessions:
                    subjects[subject_num] = sorted(sessions)

    return dict(sorted(subjects.items()))


def get_unified_sacc_params(subjects):
    """Create unified saccade parameters for all subjects."""
    base_params = {
        "mindur": 5,
        "maxdur": 100,
        "minsep": 30,
        "before_sacc": 25,
        "after_sacc": 25,
    }

    return {
        int(subject.split("-")[1]): base_params.copy()
        for subject in subjects
        if subject.startswith("sub-")
    }


def get_experiment_params(main_dir):
    """Get experiment parameters based on directory type"""
    if main_dir == PASSIVE_DIR:
        return {
            # - number of trials per block :
            "N_trials": 1,
            # - number of blocks :
            "N_blocks": 1,
            # - direction of the target :
            # list of lists for each block containing the direction of
            # the target for each trial is to -1 for left 1 for right
            "dir_target": [[], []],  # will be defined in the main loop
            # - number of px per degree for the experiment :
            "px_per_deg": px_per_deg,
            "screen_width": screen_width_px,
            "screen_height": screen_height_px,
            # OPTIONAL :
            # - list of the names of the events of the trial :
            "list_events": ["StimOn\n", "StimOff\n", "TargetOnSet\n", "TargetOffSet\n"],
        }
    else:
        return {
            # - number of trials per block :
            "N_trials": 1,
            # - number of blocks :
            "N_blocks": 1,
            # - direction of the target :
            # list of lists for each block containing the direction of
            # the target for each trial is to -1 for left 1 for right
            "dir_target": [[], []],  # will be defined in the main loop
            # - number of px per degree for the experiment :
            "px_per_deg": px_per_deg,
            "screen_width": screen_width_px,
            "screen_height": screen_height_px,
            # OPTIONAL :
            # - list of the names of the events of the trial :
            "list_events": ["FixOn\n", "FixOff\n", "TargetOnSet\n", "TargetOffSet\n"],
        }


def prepare_directories(main_dir, sub, session):
    """Create output directories for plots and results"""
    outputFolder_plots = os.path.join(main_dir, sub, "plots")
    try:
        os.makedirs(outputFolder_plots, exist_ok=True)
        os.makedirs(os.path.join(outputFolder_plots, "qc"), exist_ok=True)  # QC folder
    except OSError as e:
        print(f"Error creating directories for {sub}, session {session}: {e}")
        return None

    return outputFolder_plots


def get_file_paths(main_dir, sub, session, session_dir):
    """Get all required file paths for processing"""
    # Find the .asc file
    asc_files = [
        f for f in os.listdir(session_dir)
        if f.endswith(".asc") and f.startswith(f"{sub}_ses")
    ]
    if not asc_files:
        print(f"No .asc file found for {sub}, session {session}")
        return None, None, None, None, None, None, None, None, None

    # Find the .csv file
    csv_files = [
        f for f in os.listdir(session_dir)
        if f.endswith(".csv") and f.startswith(f"{sub}_ses")
    ]
    if not csv_files:
        print(f"No .csv file found for {sub}, session {session}")
        return None, None, None, None, None, None, None, None, None

    # Set up all file paths
    asc_file = os.path.join(session_dir, asc_files[0])
    tsv_file = os.path.join(session_dir, csv_files[0])
    
    # Output paths
    outputFolder_plots = os.path.join(main_dir, sub, "plots")
    fitPDFFile = os.path.join(outputFolder_plots, f"{sub}_{session}_fit.pdf")
    nanOverallFile = os.path.join(outputFolder_plots, "qc", f"{sub}_{session}_nanOverall.pdf")
    nanOnsetFile = os.path.join(outputFolder_plots, "qc", f"{sub}_{session}_nanOnset.pdf")
    nanSequenceFile = os.path.join(outputFolder_plots, "qc", f"{sub}_{session}_nanSequence.pdf")
    h5_file = os.path.join(session_dir, "posFilter.h5")
    h5_rawfile = os.path.join(session_dir, "rawData.h5")
    h5_qcfile = os.path.join(session_dir, "qualityControl.h5")

    return (asc_file, tsv_file, fitPDFFile, nanOverallFile, nanOnsetFile, 
            nanSequenceFile, h5_file, h5_rawfile, h5_qcfile)


def process_subject_session(main_dir, sub, session):
    """Process a single subject and session"""
    print(f"Processing {sub}, session {session} in {main_dir}")
    try:
        # Setup paths
        session_dir = os.path.join(main_dir, sub, session)
        outputFolder_plots = prepare_directories(main_dir, sub, session)
        
        if not outputFolder_plots:
            return f"Failed to create directories for {sub}, session {session}"
        
        # Get file paths
        (asc_file, tsv_file, fitPDFFile, nanOverallFile, nanOnsetFile, 
         nanSequenceFile, h5_file, h5_rawfile, h5_qcfile) = get_file_paths(main_dir, sub, session, session_dir)
        
        if not asc_file:
            return f"Missing files for {sub}, session {session}"
        
        # Initialize PDF files
        nanOverallpdf = PdfPages(nanOverallFile)
        nanOnsetpdf = PdfPages(nanOnsetFile)
        nanSequencepdf = PdfPages(nanSequenceFile)
        pdf = PdfPages(fitPDFFile)
        
        # Get experiment parameters
        param_exp = get_experiment_params(main_dir)
        
        # Get subjects list for saccade params
        subjects = [sub]  # Just need the current subject
        sacc_params = get_unified_sacc_params(subjects)
        
        # Read data
        if main_dir == PASSIVE_DIR:
            data = read_edf(asc_file, start="StimOn", stop="blank_screen")
        else:
            data = read_edf(asc_file, start="FixOn", stop="blank_screen")
        
        # Read the .tsv file
        tg_dir = pd.read_csv(tsv_file)["target_direction"].values
        
        if main_dir == PASSIVE_DIR:
            arrow = pd.read_csv(tsv_file)["arrow"].values
        else:
            arrow = pd.read_csv(tsv_file)["chosen_arrow"].values
            
        # Getting the probability from the csv file
        proba = pd.read_csv(tsv_file)["proba"].values[0]
        
        # Change directions from 0/1 diagonals to -1/1
        param_exp["dir_target"] = [x if x == 1 else -1 for x in tg_dir]
        param_exp["N_trials"] = len(data)
        
        # Create an ANEMO instance
        A = ANEMO(param_exp)
        Fit = A.Fit(param_exp)
        
        # Process each trial
        firstTrial = True
        paramsSub = []
        paramsRaw = []
        qualityCtrl = []
        sub_num = int(sub.split("-")[1])       
        for trial in range(param_exp["N_trials"]):
            print(f"Trial {trial}, session {session}, sub {sub}")
            
            if len(data[trial]["x"]) and len(data[trial]["y"]):
                data[trial]["y"] = screen_height_px - data[trial]["y"]

                type_dir = "R" if param_exp["dir_target"][trial] == 1 else "L"
                type_arrow = arrow[trial]
                trialType_txt = "{a}{d}".format(a=type_arrow, d=type_dir)
                # Get trial data and transform into the arg
                arg = A.arg(data_trial=data[trial], trial=trial, block=0)
                # print(arg)

                TargetOnIndex = arg.TargetOn - arg.t_0

                pos_deg_x = A.data_deg(
                    data=arg.data_x,
                    StimulusOf=arg.StimulusOf,
                    t_0=arg.t_0,
                    saccades=arg.saccades,
                    before_sacc=sacc_params[sub_num]["before_sacc"],
                    after_sacc=sacc_params[sub_num]["after_sacc"],
                    filt=False,
                    cutoff=30,
                    sample_rate=1000,
                )

                pos_deg_y = A.data_deg(
                    data=arg.data_y,
                    StimulusOf=arg.StimulusOf,
                    t_0=arg.t_0,
                    saccades=arg.saccades,
                    before_sacc=sacc_params[sub_num]["before_sacc"],
                    after_sacc=sacc_params[sub_num]["after_sacc"],
                    filt=False,
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

                new_saccades = arg.saccades
                # misac = A.detec_misac(
                #     velocity_x=velocity_deg_x,
                #     velocity_y=velocity_deg_y,
                #     t_0=arg.t_0,
                #     VFAC=5,
                #     mindur=sacc_params[sub_num]["mindur"],
                #     maxdur=sacc_params[sub_num]["maxdur"],
                #     minsep=sacc_params[sub_num]["minsep"],
                # )
                #
                # [
                #     sacc.extend([0, 0, 0, 0, 0]) for sacc in misac
                # ]  # transform misac into the eyelink format
                # new_saccades.extend(misac)
                # new_saccades = [x[:2] for x in new_saccades]

                sac = A.detec_sac(
                    velocity_x=velocity_deg_x,
                    velocity_y=velocity_deg_y,
                    t_0=arg.t_0,
                    VFAC=5,
                    mindur=sacc_params[sub_num]["mindur"],
                    maxdur=sacc_params[sub_num]["maxdur"],
                    minsep=sacc_params[sub_num]["minsep"],
                )

                [
                    sacc.extend([0, 0, 0, 0, 0]) for sacc in sac
                ]  # transform misac into the eyelink format
                new_saccades.extend(sac)

                blinks = data[trial]["events"]["Eblk"].copy()
                if len(data[trial]["events"]["Sblk"]) > len(
                    data[trial]["events"]["Eblk"]
                ):
                    blinks.append(
                        list(
                            [
                                data[trial]["events"]["Sblk"][-1],
                                data[trial]["trackertime"][-1],
                            ]
                        )
                    )
                # blinks = [x[:2] for x in blinks]

                velocity_x_NAN = A.data_NAN(
                    data=velocity_deg_x,
                    saccades=new_saccades,
                    trackertime=arg.trackertime,
                    before_sacc=sacc_params[sub_num]["before_sacc"],
                    after_sacc=sacc_params[sub_num]["after_sacc"],
                )

                velocity_y_NAN = A.data_NAN(
                    data=velocity_deg_y,
                    saccades=new_saccades,
                    trackertime=arg.trackertime,
                    before_sacc=sacc_params[sub_num]["before_sacc"],
                    after_sacc=sacc_params[sub_num]["after_sacc"],
                )

                time = arg.trackertime - arg.TargetOn

                idx2keep_y = np.logical_and(time >= -200, time < 600)
                time_y = time[idx2keep_y]
                pos_y = arg.data_y[idx2keep_y]
                vel_y = velocity_y_NAN[idx2keep_y]

                idx2keep_x = np.logical_and(time >= -200, time < 600)
                time_x = time[idx2keep_x]
                pos_x = arg.data_x[idx2keep_x]
                vel_x = velocity_x_NAN[idx2keep_x]

                # vel_x[:25] = np.nan
                # vel_y[:25] = np.nan
                # vel_x[-25:] = np.nan
                # vel_y[-25:] = np.nan

                pos_deg_x = pos_deg_x[idx2keep_x]
                pos_deg_y = pos_deg_y[idx2keep_y]

                # calc saccades relative to t_0
                # because I am passing the time relative to t_0 to ANEMO
                for sacc in new_saccades:
                    sacc[0] = sacc[0] - arg.TargetOn
                    sacc[1] = sacc[1] - arg.TargetOn

                sDict = {
                    "condition": proba,
                    "session": session,
                    "trial": trial,
                    "trialType": trialType_txt,
                    "direction": param_exp["dir_target"][trial],
                    "time_x": time_x,
                    "time_y": time_y,
                    "posPxl_x": pos_x,
                    "posPxl_y": pos_y,
                    "posDeg_x": pos_deg_x,
                    "posDeg_y": pos_deg_y,
                    "velocity_x": vel_x,
                    "velocity_y": vel_y,
                    "saccades": new_saccades,
                }

                # save trial data to a dataframe
                if firstTrial:
                    paramsRaw = pd.DataFrame([sDict], columns=sDict.keys())
                #                         firstTrial = False # DELETE THIS LINE WHEN RUNNING THE FIT
                else:
                    paramsRaw = pd.concat(
                        [paramsRaw, pd.DataFrame([sDict], columns=sDict.keys())],
                        ignore_index=True,
                    )

                # test: if bad trial

                # Getting the newTargetOnset index
                newTargetOnset = np.where(time_x == 0)[0][0]

                if (
                    np.mean(
                        np.isnan(vel_x[newTargetOnset - 200 : newTargetOnset + 100])
                    )
                    >1/3
                    or np.mean(np.isnan(vel_x)) > 0.7
                    or longestNanRun(
                        vel_x[newTargetOnset - 100 : newTargetOnset + 100]
                    )
                    >100
                    # or abs(
                    #     np.nanmean(
                    #         vel_x[newTargetOnset + 250 : newTargetOnset + 500]
                    #     )
                    # )
                    # < 3
                    # or abs(np.nanmean(vel_x[TargetOnIndex : TargetOnIndex + 100])) > 8
                ):

                    print("Skipping bad trial...")

                    plt.clf()
                    fig = plt.figure(figsize=(10, 4))
                    plt.suptitle("Trial %d" % trial)
                    plt.subplot(1, 2, 1)
                    plt.plot(time_x, vel_x)
                    plt.axvline(x=time_x[0], linewidth=1, linestyle="--", color="k")
                    plt.axvline(
                        x=time_x[-1], linewidth=1, linestyle="--", color="k"
                    )
                    # plt.xlim(-100, 1200)
                    plt.ylim(-15, 15)
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Velocity - x axis")
                    plt.subplot(1, 2, 2)
                    plt.plot(time_y, vel_y)
                    plt.axvline(x=time_y[0], linewidth=1, linestyle="--", color="k")
                    plt.axvline(
                        x=time_y[-1], linewidth=1, linestyle="--", color="k"
                    )
                    # plt.xlim(-100, 1200)
                    plt.ylim(-35, 35)
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Velocity - y axis")
                    # plt.show()
                    # plt.pause(0.050)
                    # plt.clf()
                    # print("Time", newTargetOnset)
                    reason = ""
                    # print(vel_x[TargetOnIndex - 100 : TargetOnIndex + 100])
                    if (
                        np.mean(
                            np.isnan(
                                vel_x[newTargetOnset - 200 : newTargetOnset + 100]
                            )
                        )
                        > 1/3                    ):
                        print("too many NaNs around the start of the pursuit")
                        reason = (
                            reason + " >.1/3 of NaNs around the start of the pursuit"
                        )
                        nanOnsetpdf.savefig(fig)
                    elif np.mean(np.isnan(vel_x)) > 0.7:
                        print("too many NaNs overall")
                        reason = reason + " >{0} of NaNs overall".format(0.6)
                        nanOverallpdf.savefig(fig)
                    elif (
                        longestNanRun(
                            vel_x[newTargetOnset - 100 : newTargetOnset +100]
                        )
                        > 100
                    ):
                        print("at least one nan sequence with more than 100ms")
                        reason = (
                            reason
                            + " At least one nan sequence with more than 100ms"
                        )
                        nanSequencepdf.savefig(fig)
                    # elif (
                    #     abs(
                    #         np.nanmean(
                    #             vel_x[newTargetOnset + 250 : newTargetOnset + 500]
                    #         )
                    #     )
                    #     < 3
                    # ):
                    #     print("No smooth pursuit")
                    #     reason = reason + " No smooth pursuit"
                    #     nanSequencepdf.savefig(fig)
                    # if abs(np.nanmean(vel_x[TargetOnIndex : TargetOnIndex + 100])) > 8:
                    #     print("Noisy around target onset")
                    #     reason = reason + " Noisy around target onset"
                    #     nanSequencepdf.savefig(fig)

                    plt.close(fig)

                    newResult = dict()
                    newResult["condition"] = proba
                    newResult["session"] = session
                    newResult["trial"] = trial
                    newResult["trialType"] = trialType_txt
                    newResult["target_dir"] = param_exp["dir_target"][trial]

                    newResult["time"] = time_x
                    newResult["velocity_x"], newResult["velocity_y"] = (
                        velocity_x_NAN[:-time_sup],
                        velocity_y_NAN[:-time_sup],
                    )
                    newResult["saccades"] = np.array(new_saccades)

                    qCtrl = dict()
                    qCtrl["sub"] = sub
                    qCtrl["condition"] = proba
                    qCtrl["session"] = session
                    qCtrl["trial"] = trial
                    qCtrl["keep_trial"] = 0
                    qCtrl["good_fit"] = 0
                    qCtrl["discard_reason"] = reason

                else:  # if not a bad trial, does the fit

                    classic_lat_x, classic_max_x, classic_ant_x = (
                        A.classical_method.Full(vel_x, 200)
                    )
                    # print('classical latency: {:+.2f}, max: {:+.2f}, anti: {:+.2f}'.format(classic_lat_x, classic_max_x, classic_ant_x))
                    classic_ant = (
                        classic_ant_x if not np.isnan(classic_ant_x) else 0.5
                    )

                    if equation == "sigmoid":

                        param_fit, inde_var = Fit.generation_param_fit(
                            equation="fct_velocity_sigmo",
                            dir_target=param_exp["dir_target"][trial],
                            trackertime=time_x,
                            TargetOn=0,
                            StimulusOf=time_x[0],
                            saccades=new_saccades,
                            value_latency=classic_lat_x - 200,
                            value_steady_state=classic_max_x,
                            value_anti=classic_ant * 5,
                        )

                        result_x = Fit.Fit_trial(
                            vel_x,
                            equation="fct_velocity_sigmo",
                            dir_target=int(param_exp["dir_target"][trial]),
                            trackertime=time_x,
                            TargetOn=0,
                            StimulusOf=time_x[0],
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
                            allow_baseline=allow_baseline,
                            allow_horizontalShift=allow_horizontalShift,
                        )

                        eq_x_tmp = np.array(eq_x_tmp)
                        eq_x = np.zeros(len(time_x))
                        eq_x[:] = np.nan
                        eq_x[: len(eq_x_tmp)] = eq_x_tmp

                    newResult = dict()
                    newResult["condition"] = proba
                    newResult["session"] = session
                    newResult["trial"] = trial
                    newResult["trialType"] = trialType_txt
                    newResult["target_dir"] = param_exp["dir_target"][trial]
                    newResult["time_x"] = time_x
                    newResult["velocity_x"] = vel_x
                    newResult["saccades"] = np.array(new_saccades)

                    newResult["t_0"] = result_x.params["t_0"].value
                    newResult["t_end"] = result_x.params["t_end"].value
                    newResult["dir_target"] = result_x.params["dir_target"].value
                    newResult["baseline"] = result_x.params["baseline"].value
                    newResult["aSPon"] = np.round(
                        result_x.params["start_anti"].value
                    )
                    newResult["aSPv_slope"] = result_x.params["a_anti"].value
                    newResult["SPacc"] = result_x.params["ramp_pursuit"].value
                    newResult["SPss"] = result_x.params["steady_state"].value

                    newResult["do_whitening_x"] = result_x.params[
                        "do_whitening"
                    ].value

                    if equation == "sigmoid":
                        newResult["aSPoff"] = np.round(
                            result_x.params["latency"].value
                        )
                        newResult["horizontal_shift"] = result_x.params[
                            "horizontal_shift"
                        ].value

                        idx_aSPoff = np.where(time_x == newResult["aSPoff"])[0][0]
                        vel_at_latency = (
                            eq_x[idx_aSPoff]
                            + (newResult["SPss"] - eq_x[idx_aSPoff])
                            * 0.05
                            * newResult["dir_target"]
                        )
                        vel, idx = closest(eq_x[idx_aSPoff + 1 :], vel_at_latency)

                        newResult["SPlat"] = time_x[idx + idx_aSPoff + 1]
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
                        newResult["time_x"],
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
                    qCtrl["condition"] = proba
                    qCtrl["session"] = session
                    qCtrl["trial"] = trial

                    if newResult["rmse_x"] > 10:
                        qCtrl["keep_trial"] = np.nan
                        qCtrl["good_fit"] = 0
                        qCtrl["discard_reason"] = "RMSE > 10"
                    elif manualCheck:
                        qCtrl["keep_trial"] = int(
                            input(
                                "Keep trial? \npress (1) to keep or (0) to discard\n "
                            )
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
                                input(
                                    "Good fit? \npress (1) for yes or (0) for no\n "
                                )
                            )

                        qCtrl["discard_reason"] = np.nan
                    else:
                        qCtrl["keep_trial"] = np.nan
                        qCtrl["good_fit"] = np.nan
                        qCtrl["discard_reason"] = np.nan

                # save trial's fit data to a dataframe
                if firstTrial:
                    paramsSub = pd.DataFrame([newResult], columns=newResult.keys())
                    qualityCtrl = pd.DataFrame([qCtrl], columns=qCtrl.keys())
                    firstTrial = False
                else:
                    paramsSub = pd.concat(
                        [
                            paramsSub,
                            pd.DataFrame([newResult], columns=newResult.keys()),
                        ],
                        ignore_index=True,
                    )
                    qualityCtrl = pd.concat(
                        [qualityCtrl, pd.DataFrame([qCtrl], columns=qCtrl.keys())],
                        ignore_index=True,
                    )
            # Here you would normally have the trial processing code
            # Since you mentioned skipping this part, I'll include a placeholder
            # This is where all the processing for each trial would go
            
            # ... [Trial processing code goes here - you'll copy your existing code] ...
            
                
        # Close all PDF files
        nanOnsetpdf.close()
        nanOverallpdf.close()
        nanSequencepdf.close()
        pdf.close()
        plt.close("all")
        
        # Save the dataframes to HDF files
        paramsSub.to_hdf(h5_file, "data")
        paramsRaw.to_hdf(h5_rawfile, "data")
        qualityCtrl.to_hdf(h5_qcfile, "data")

        # Test if files can be read
        abc = pd.read_hdf(h5_file, "data")
        del paramsRaw, abc, paramsSub, qualityCtrl, newResult
            


        print(f"Successfully processed {sub}, session {session}")
        return f"Success: {sub}, session {session}"
            
    except Exception as e:
        print(f"Error processing {sub}, session {session}: {str(e)}")
        traceback.print_exc()
        return f"Error: {sub}, session {session} - {str(e)}"


def get_subjects_by_directory(main_dir):
    """Get list of subjects from a directory"""
    subject_sessions = get_subjects_and_sessions(main_dir)
    return [f"sub-{str(num).zfill(3)}" for num in subject_sessions.keys()]


def get_all_subject_session_pairs(main_dir):
    """Get all subject-session pairs for a directory"""
    subject_sessions = get_subjects_and_sessions(main_dir)
    
    # Convert to the format expected by the processing function
    pairs = []
    for sub_num, sessions in subject_sessions.items():
        sub = f"sub-{str(sub_num).zfill(3)}"
        for sess in sessions:
            sess_str = f"session-{str(sess).zfill(2)}"
            pairs.append((sub, sess_str))
    
    return pairs


def main():
    """Main function to process all directories in parallel."""
    print("Starting parallel processing...")
    
    # Process each directory separately
    for main_dir in [ACTIVE_DIR, PASSIVE_DIR]:
        print(f"Processing directory: {main_dir}")
        
        # Get all subject-session pairs for this directory
        subject_session_pairs = get_all_subject_session_pairs(main_dir)
        
        # Process all pairs in parallel
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_subject_session)(main_dir, sub, session) 
            for sub, session in subject_session_pairs
        )
        
        # Print results summary
        success_count = sum(1 for result in results if result.startswith("Success"))
        print(f"Directory {main_dir} complete: {success_count}/{len(results)} successful")
    
    print("All processing complete!")


if __name__ == "__main__":
    main()

