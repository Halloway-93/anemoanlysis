from joblib import Parallel, delayed
import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import traceback
from ANEMO.ANEMO import ANEMO, read_edf
from functions.utils import *

def process_subject_condition(sub, cond, main_dir, param_exp, sacc_params, showPlots=0, manualCheck=0, equation="sigmoid"):
    """
    Process a single subject-condition pair.
    """
    allow_baseline, allow_horizontalShift, allow_acceleration = True, True, False
    try:
        print(f"Processing Subject: {sub}, Condition: {cond}")
        
        h5_file = f"{sub}/{sub}_{cond}_posFilter.h5"
        h5_rawfile = f"{sub}/{sub}_{cond}_rawData.h5"
        h5_qcfile = f"{sub}/{sub}_{cond}_qualityControl.h5"
        
        paramsRaw = []
        qualityCtrl = []
        paramsSub = []
        
        dataFile = f"{main_dir}/{sub}/{sub}_{cond}_eyeData.asc"
        tgDirFile = f"{main_dir}/{sub}/{sub}_{cond}_events.tsv"
        
        outputFolder_plots = f"{sub}/plots/"
        fitPDFFile = f"{outputFolder_plots}{sub}_{cond}_fit.pdf"
        
        # Create output directories
        os.makedirs(outputFolder_plots, exist_ok=True)
        os.makedirs(f"{outputFolder_plots}qc/", exist_ok=True)
        
        # Setup PDF files
        nanOverallFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanOverall.pdf"
        nanOnsetFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanOnset.pdf"
        nanSequenceFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanSequence.pdf"
        
        with PdfPages(nanOverallFile) as nanOverallpdf, \
             PdfPages(nanOnsetFile) as nanOnsetpdf, \
             PdfPages(nanSequenceFile) as nanSequencepdf, \
             PdfPages(fitPDFFile) as pdf:
            
            # Read data
            data = read_edf(dataFile, start="FixOff", stop="TargetOff")
            tg_up = pd.read_csv(tgDirFile, sep="\t")["trial_color_UP"]
            tg_color = pd.read_csv(tgDirFile, sep="\t")["trial_color_chosen"]
            tg_dir = pd.read_csv(tgDirFile, sep="\t")["trial_direction"]

            # change directions from 0/1 diagonals to -1/1
            param_exp["dir_target"] = [x if x == 1 else -1 for x in tg_dir]

            param_exp["N_trials"] = len(data)
            # param_exp['N_trials'] = 10

            # creates an ANEMO instance
            A = ANEMO(param_exp)
            Fit = A.Fit(param_exp)

            firstTrial = True
            
            # Update param_exp
            param_exp = param_exp.copy()  # Create a local copy
            param_exp["dir_target"] = [1 if x == 1 else -1 for x in tg_dir.values]
            param_exp["N_trials"] = len(data)
            
            # Create ANEMO instance
            A = ANEMO(param_exp)
            Fit = A.Fit(param_exp)
            
            # Process each trial
            firstTrial = True
            for trial in list(range(param_exp["N_trials"])):
                print("Trial {0}, cond {1}, sub {2}".format(trial, cond, sub))

                if len(data[trial]["x"]) and len(data[trial]["y"]):

                    data[trial]["y"] = screen_height_px - data[trial]["y"]

                    type_col = "Red" if tg_color[trial] == 1 else "Green"
                    type_dir = "R" if param_exp["dir_target"][trial] == 1 else "L"

                    trialType_txt = "{c}{d}".format(c=type_col, d=type_dir)
                    trialTgUP_txt = "Red" if tg_up[trial] == 1 else "Green"

                    # get trial data and transform into the arg
                    arg = A.arg(data_trial=data[trial], trial=trial, block=0)

                    TargetOn_0 = arg.TargetOn - arg.t_0

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
                    # new_saccades = [x[:2] for x in new_saccades]

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

                    time = arg.trackertime - arg.TargetOn

                    idx2keep_y = np.logical_and(time >= -200, time < 600)
                    time_y = time[idx2keep_y]
                    pos_y = arg.data_y[idx2keep_y]
                    vel_y = velocity_y_NAN[idx2keep_y]

                    idx2keep_x = np.logical_and(time >= -200, time < 600)
                    time_x = time[idx2keep_x]
                    pos_x = arg.data_x[idx2keep_x]
                    vel_x = velocity_x_NAN[idx2keep_x]

                    vel_x[:25] = np.nan
                    vel_y[:25] = np.nan
                    vel_x[-25:] = np.nan
                    vel_y[-25:] = np.nan

                    pos_deg_x = pos_deg_x[idx2keep_x]
                    pos_deg_y = pos_deg_y[idx2keep_y]

                    # calc saccades relative to t_0
                    # because I am passing the time relative to t_0 to ANEMO
                    for sacc in new_saccades:
                        sacc[0] = sacc[0] - arg.TargetOn
                        sacc[1] = sacc[1] - arg.TargetOn

                    sDict = {
                        "condition": cond,
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
                    if (
                        np.mean(np.isnan(vel_x[TargetOn_0 - 100 : TargetOn_0 + 100]))
                        > 0.7
                        or np.mean(np.isnan(vel_x[:-time_sup])) > 0.6
                        or longestNanRun(vel_x[TargetOn_0 - 150 : TargetOn_0 + 600])
                        > 200
                        or abs(np.nanmean(vel_x[TargetOn_0 + 300 : TargetOn_0 + 600]))
                        < 4
                        or abs(np.nanmean(vel_x[TargetOn_0 : TargetOn_0 + 100])) > 8
                    ):

                        print("Skipping bad trial...")

                        plt.clf()
                        fig = plt.figure(figsize=(10, 4))
                        plt.suptitle("Trial %d" % trial)
                        plt.subplot(1, 2, 1)
                        plt.plot(vel_x[:-time_sup])
                        plt.axvline(x=200, linewidth=1, linestyle="--", color="k")
                        plt.axvline(x=800, linewidth=1, linestyle="--", color="k")
                        plt.xlim(-100, 1200)
                        plt.ylim(-15, 15)
                        plt.xlabel("Time (ms)")
                        plt.ylabel("Velocity - x axis")
                        plt.subplot(1, 2, 2)
                        plt.plot(vel_x[:-time_sup])
                        plt.axvline(x=200, linewidth=1, linestyle="--", color="k")
                        plt.axvline(x=800, linewidth=1, linestyle="--", color="k")
                        plt.xlim(-100, 1200)
                        plt.ylim(-35, 35)
                        plt.xlabel("Time (ms)")
                        plt.ylabel("Velocity - y axis")
                        # plt.show()
                        # plt.pause(0.050)
                        # plt.clf()

                        reason = ""
                        if (
                            np.mean(
                                np.isnan(vel_x[TargetOn_0 - 100 : TargetOn_0 + 100])
                            )
                            > 0.7
                        ):
                            print("too many NaNs around the start of the pursuit")
                            reason = (
                                reason + " >.70 of NaNs around the start of the pursuit"
                            )
                            nanOnsetpdf.savefig(fig)
                        if np.mean(np.isnan(vel_x[:-time_sup])) > 0.6:
                            print("too many NaNs overall")
                            reason = reason + " >{0} of NaNs overall".format(0.6)
                            nanOverallpdf.savefig(fig)
                        if (
                            longestNanRun(vel_x[TargetOn_0 - 150 : TargetOn_0 + 600])
                            > 200
                        ):
                            print("at least one nan sequence with more than 200ms")
                            reason = (
                                reason
                                + " At least one nan sequence with more than 200ms"
                            )
                            nanSequencepdf.savefig(fig)
                        if (
                            abs(np.nanmean(vel_x[TargetOn_0 + 300 : TargetOn_0 + 600]))
                            < 4
                        ):
                            print("No smooth pursuit")
                            reason = reason + " No smooth pursuit"
                            nanSequencepdf.savefig(fig)
                        if abs(np.nanmean(vel_x[TargetOn_0 : TargetOn_0 + 100])) > 8:
                            print("Noisy around target onset")
                            reason = reason + " Noisy around target onset"
                            nanSequencepdf.savefig(fig)

                        plt.close(fig)

                        newResult = dict()
                        newResult["condition"] = cond
                        newResult["trial"] = trial
                        newResult["trialType"] = trialType_txt
                        newResult["target_dir"] = param_exp["dir_target"][trial]

                        x = arg.trackertime - arg.TargetOn
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
                                trackertime=list(inde_var["x"]),
                                TargetOn=0,
                                StimulusOf=-200,
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
                                allow_horizontalShift=result_x.params[
                                    "allow_horizontalShift"
                                ],
                            )

                            eq_x_tmp = np.array(eq_x_tmp)
                            eq_x = np.zeros(len(time_x))
                            eq_x[:] = np.nan
                            eq_x[: len(eq_x_tmp)] = eq_x_tmp

                        newResult = dict()
                        newResult["condition"] = cond
                        newResult["trial"] = trial
                        newResult["trialType"] = trialType_txt
                        newResult["trialTgUP"] = trialTgUP_txt
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
                        qCtrl["condition"] = cond
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
            
            # Save results
            paramsSub.to_hdf(h5_file, "data/")
            paramsRaw.to_hdf(h5_rawfile, "data/")
            qualityCtrl.to_hdf(h5_qcfile, "data/")
            
            return True
            
    except Exception as e:
        print(f"Error processing {sub}, condition {cond}")
        traceback.print_exc()
        return False

def parallel_process_subjects(subjects, conditions, main_dir, param_exp, sacc_params, n_jobs=-1,time_sup=None):
    """
    Process all subjects and conditions in parallel.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    conditions : list
        List of conditions
    main_dir : str
        Main directory path
    param_exp : dict
        Experiment parameters
    sacc_params : dict
        Saccade parameters
    n_jobs : int
        Number of parallel jobs. -1 means using all available cores.
    """
    # Create all combinations of subjects and conditions
    tasks = [(sub, cond) for sub in subjects for cond in conditions]
    
    # Run processing in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_subject_condition)(
            sub, 
            cond, 
            main_dir, 
            param_exp, 
            sacc_params
        ) for sub, cond in tasks
    )
    
    # Report results
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    print(f"\nProcessing complete:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Setup your parameters
    subjects = [
        "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
        "sub-06", "sub-07", "sub-08", "sub-10", "sub-11",
        "sub-12", "sub-13", "sub-14", "sub-15", "sub-16"
    ]
    
    conditions = [
        "col50-dir25",
        "col50-dir50",
        "col50-dir75",
    ]
    
    main_dir="/envau/work/brainets/oueld.h/attentionalTask/data"
    # Run parallel processing

    time_sup = 10  # time window to cut at the end of the trial
    sacc_params = {
    1: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 15, "after_sacc": 25},
    2: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 10, "after_sacc": 20},
    3: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 15, "after_sacc": 25},
    4: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 15, "after_sacc": 25},
    5: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 15, "after_sacc": 25},
    6: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 15, "after_sacc": 25},
    7: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 20, "after_sacc": 30},
    8: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 20, "after_sacc": 30},
    10: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 5, "after_sacc": 15},
    11: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 10, "after_sacc": 20},
    12: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 5, "after_sacc": 20},
    13: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 10, "after_sacc": 20},
    14: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 5, "after_sacc": 15},
    15: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 10, "after_sacc": 15},
    16: {"mindur": 5, "maxdur": 100, "minsep": 30, "before_sacc": 15, "after_sacc": 25},
    }


    #  ANEMO parameters
    screen_width_px = 1920  # px
    screen_height_px = 1080  # px
    screen_width_cm = 70  # cm
    viewingDistance = 57.0  # cm

    tan = np.arctan((screen_width_cm / 2) / viewingDistance)
    screen_width_deg = 2.0 * tan * 180 / np.pi
    px_per_deg = screen_width_px / screen_width_deg

    param_exp = {  # Mandatory :
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
    # - subject name :
    #'observer' : 's',
    # - list of the names of the events of the trial :
    "list_events": ["FixOn\n", "FixOff\n", "TargetOn\n", "TargetOff\n"],
    # - target velocity in deg/s :
    # 'V_X_deg' : 15,
    # - presentation time of the target :
    #'stim_tau' : 0.75,
    # - the time the target has to arrive at the center of the screen in ms,
    # to move the target back to t=0 of its RashBass = velocity*latency
    #'RashBass' : 100,
    }
    results = parallel_process_subjects(
        subjects=subjects,
        conditions=conditions,
        main_dir=main_dir,
        param_exp=param_exp,
        sacc_params=sacc_params,
        n_jobs=-1,  # Use all available cores
        time_sup=time_sup
    )
