import os
import numpy as np
import pandas as pd
from functions.utils import *
from ANEMO.ANEMO import ANEMO, read_edf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from joblib import Parallel, delayed
import traceback

warnings.filterwarnings("ignore")

# Define the main directories
ACTIVE_COLOR_DIR = "/Users/mango/oueld.h/contextuaLearning/ColorCue/data"
PASSIVE_COLOR_DIR = "/Users/mango/oueld.h/contextuaLearning/ColorCue/imposedColorData"
ATTENTION_COLOR_DIR = "/Users/mango/oueld.h/attentionalTask/data"

# ANEMO parameters
screen_width_px = 1920  # px
screen_height_px = 1080  # px
screen_width_cm = 70  # cm
viewingDistance = 57.0  # cm

tan = np.arctan((screen_width_cm / 2) / viewingDistance)
screen_width_deg = 2.0 * tan * 180 / np.pi
px_per_deg = screen_width_px / screen_width_deg

px_per_deg = 27.46

param_exp = {
    "N_trials": 1,
    "N_blocks": 1,
    "dir_target": [[], []],
    "px_per_deg": px_per_deg,
    "screen_width": screen_width_px,
    "screen_height": screen_height_px,
    "list_events": ["FixOn\n", "FixOff\n", "TargetOn\n", "TargetOff\n"],
}

# Define subjects for each directory
def get_subjects_by_directory(main_dir):
    if main_dir == ACTIVE_COLOR_DIR:
        return [
            "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
            "sub-09", "sub-10", "sub-11", "sub-12", "sub-13", "sub-14", "sub-15", "sub-16",
        ]
    elif main_dir == PASSIVE_COLOR_DIR:
        return [
            "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", 
            "sub-08", "sub-09", "sub-10", "sub-11",
        ]
    else:  # ATTENTION_COLOR_DIR
        return [
            "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07",
            "sub-08", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13",
        ]

conditions = ["col50-dir25", "col50-dir50", "col50-dir75"]

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

time_sup = 10  # time window to cut at the end of the trial
showPlots = 0
manualCheck = 0
equation = "sigmoid"
allow_baseline, allow_horizontalShift, allow_acceleration = True, True, False

def process_subject_condition(main_dir, sub, cond):
    """Process a single subject and condition combination."""
    print(f"Processing {main_dir} - {sub} - {cond}")
    
    try:
        # Change to the main directory
        os.chdir(main_dir)
        
        # Generate sacc_params for this subject
        subjects = get_subjects_by_directory(main_dir)
        sacc_params = get_unified_sacc_params(subjects)
        
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

        # make figure folders
        os.makedirs(outputFolder_plots, exist_ok=True)
        os.makedirs(f"{outputFolder_plots}qc/", exist_ok=True)

        # Create PDF files for quality control
        nanOverallFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanOverall.pdf"
        nanOnsetFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanOnset.pdf"
        nanSequenceFile = f"{outputFolder_plots}qc/{sub}_{cond}_nanSequence.pdf"

        nanOverallpdf = PdfPages(nanOverallFile)
        nanOnsetpdf = PdfPages(nanOnsetFile)
        nanSequencepdf = PdfPages(nanSequenceFile)
        pdf = PdfPages(fitPDFFile)

        data = read_edf(dataFile, start="FixOff", stop="TargetOff")

        # Read target color based on the directory
        if main_dir == ACTIVE_COLOR_DIR:
            tg_color = pd.read_csv(tgDirFile, sep="\t")["trial_color_chosen"]
        elif main_dir == ATTENTION_COLOR_DIR:
            tg_color = pd.read_csv(tgDirFile, sep="\t")["trial_color_imposed"]
        else:
            tg_color = pd.read_csv(tgDirFile, sep="\t")["trial_color"]

        tg_dir = pd.read_csv(tgDirFile, sep="\t")["trial_direction"]

        # change directions from 0/1 diagonals to -1/1
        param_exp["dir_target"] = [x if x == 1 else -1 for x in tg_dir]
        param_exp["N_trials"] = len(data)

        # creates an ANEMO instance
        A = ANEMO(param_exp)
        Fit = A.Fit(param_exp)

        firstTrial = True

        for trial in list(range(param_exp["N_trials"])):
            print(f"Trial {trial}, cond {cond}, sub {sub}")

            if len(data[trial]["x"]) and len(data[trial]["y"]):
                # Process the trial data
                data[trial]["y"] = screen_height_px - data[trial]["y"]

                type_col = "Red" if tg_color[trial] == 1 else "Green"
                type_dir = "R" if param_exp["dir_target"][trial] == 1 else "L"

                trialType_txt = f"{type_col}{type_dir}"

                # get trial data and transform into the arg
                arg = A.arg(data_trial=data[trial], trial=trial, block=0)

                # Index of TargetOnset
                TargetOnIndex = arg.TargetOn - arg.t_0

                # Process position and velocity data
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
                    filter_after=True,
                    cutoff=30,
                    sample_rate=1000,
                )

                velocity_deg_y = A.velocity(
                    data=pos_deg_y,
                    filter_before=True,
                    filter_after=True,
                    cutoff=30,
                    sample_rate=1000,
                )

                new_saccades = arg.saccades
                # Detect saccades
                # misac = A.detec_misac(
                #     velocity_x=velocity_deg_x,
                #     velocity_y=velocity_deg_y,
                #     t_0=arg.t_0,
                #     VFAC=5,
                #     mindur=sacc_params[1]["mindur"],
                #     maxdur=sacc_params[1]["maxdur"],
                #     minsep=sacc_params[1]["minsep"],
                # )

                # [sacc.extend([0, 0, 0, 0, 0]) for sacc in misac]  # transform misac into the eyelink format
                # new_saccades.extend(misac)

                sac = A.detec_sac(
                    velocity_x=velocity_deg_x,
                    velocity_y=velocity_deg_y,
                    t_0=arg.t_0,
                    VFAC=5,
                    mindur=sacc_params[1]["mindur"],
                    maxdur=sacc_params[1]["maxdur"],
                    minsep=sacc_params[1]["minsep"],
                )

                [sacc.extend([0, 0, 0, 0, 0]) for sacc in sac]  # transform misac into the eyelink format
                new_saccades.extend(sac)

                blinks = data[trial]["events"]["Eblk"].copy()
                if len(data[trial]["events"]["Sblk"]) > len(data[trial]["events"]["Eblk"]):
                    blinks.append(
                        list([data[trial]["events"]["Sblk"][-1], data[trial]["trackertime"][-1]])
                    )

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
                
                # Keep only data in the desired time window
                idx2keep = np.logical_and(time >= -200, time < 600)
                time = time[idx2keep]
                pos_y = arg.data_y[idx2keep]
                vel_y = velocity_y_NAN[idx2keep]
                pos_x = arg.data_x[idx2keep]
                vel_x = velocity_x_NAN[idx2keep]
                pos_deg_x = pos_deg_x[idx2keep]
                pos_deg_y = pos_deg_y[idx2keep]

                # calc saccades relative to t_0
                for sacc in new_saccades:
                    sacc[0] = sacc[0] - arg.TargetOn
                    sacc[1] = sacc[1] - arg.TargetOn

                sDict = {
                    "condition": cond,
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

                # save trial data to a dataframe
                if firstTrial:
                    paramsRaw = pd.DataFrame([sDict], columns=sDict.keys())

                else:
                    paramsRaw = pd.concat(
                        [paramsRaw, pd.DataFrame([sDict], columns=sDict.keys())],
                        ignore_index=True,
                    )

                # test: if bad trial
                # Getting the newTargetOnset index
                newTargetOnset = np.where(time == 0)[0][0]

                if (
                    np.mean(np.isnan(vel_x[newTargetOnset - 200 : newTargetOnset + 100])) > (1/3)
                    or np.mean(np.isnan(vel_x)) > 0.7
                    or longestNanRun(vel_x[newTargetOnset - 100 : newTargetOnset + 100]) > 100
                    # or np.mean(np.isnan(vel_x[newTargetOnset + 300 :])) > 0.8

                ):
                    print("Skipping bad trial...")

                    plt.clf()
                    fig = plt.figure(figsize=(10, 4))
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
                    plt.axvline(x=time[newTargetOnset], linewidth=1, linestyle="--", color="r")
                    plt.axvline(x=time[-1], linewidth=1, linestyle="--", color="k")
                    plt.ylim(-35, 35)
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Velocity - y axis")

                    reason = ""
                    if np.mean(np.isnan(vel_x[:-time_sup])) > 0.7:
                        print("too many NaNs overall")
                        reason = reason + f" >{0.6} of NaNs overall"
                        nanOverallpdf.savefig(fig)
                    elif longestNanRun(vel_x[newTargetOnset - 100 : newTargetOnset + 100]) > 100:
                        print("at least one nan sequence with more than 100ms")
                        reason = reason + " At least one nan sequence with more than 100ms"
                        nanSequencepdf.savefig(fig)
                    elif np.mean(np.isnan(vel_x[newTargetOnset - 200 : newTargetOnset + 100])) > (1/3):
                        print("too many NaNs around the start of the pursuit")
                        reason = reason + " >1/3 of NaNs around the start of the pursuit"
                        nanOnsetpdf.savefig(fig)

                    plt.close(fig)

                    newResult = dict()
                    newResult["condition"] = cond
                    newResult["trial"] = trial
                    newResult["trialType"] = trialType_txt
                    newResult["target_dir"] = param_exp["dir_target"][trial]
                    newResult["time"] = time
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
                    classic_lat_x, classic_max_x, classic_ant_x = A.classical_method.Full(vel_x, 200)
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
                            trackertime=time,
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
                            x=time,
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
                        newResult["horizontal_shift"] = result_x.params["horizontal_shift"].value

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
                    newResult["rmse_x"] = np.sqrt(np.mean([x * x for x in result_x.residual]))
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
                        qCtrl["keep_trial"] = int(input("Keep trial? \npress (1) to keep or (0) to discard\n "))
                        while qCtrl["keep_trial"] != 0 and qCtrl["keep_trial"] != 1:
                            qCtrl["keep_trial"] = int(input("Keep trial? \npress (1) to keep or (0) to discard\n "))

                        qCtrl["good_fit"] = int(input("Good fit? \npress (1) for yes or (0) for no\n "))
                        while qCtrl["good_fit"] != 0 and qCtrl["keep_trial"] != 1:
                            qCtrl["good_fit"] = int(input("Good fit? \npress (1) for yes or (0) for no\n "))

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
                        [paramsSub, pd.DataFrame([newResult], columns=newResult.keys())],
                        ignore_index=True,
                    )
                    qualityCtrl = pd.concat(
                        [qualityCtrl, pd.DataFrame([qCtrl], columns=qCtrl.keys())],
                        ignore_index=True,
                    )

        # Close all PDFs
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
        
        return f"Successfully processed {sub}, condition {cond}"


    except Exception as e:
        error_msg = f"Error! \nCouldn't process {sub}, condition {cond}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg


def main():
    """Main function to process all directories in parallel."""
    print("Starting parallel processing...")
    
    # Process each directory separately
    for main_dir in [ACTIVE_COLOR_DIR, PASSIVE_COLOR_DIR, ATTENTION_COLOR_DIR]:
        print(f"Processing directory: {main_dir}")
        
        # Get subjects for this directory
        subjects = get_subjects_by_directory(main_dir)
        
        # Create a list of all subject-condition pairs
        tasks = [(main_dir, sub, cond) for sub in subjects for cond in conditions]
        
        # Process all tasks in parallel
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_subject_condition)(main_dir, sub, cond) 
            for main_dir, sub, cond in tasks
        )
        
        # Print results
        for result in results:
            print(result)
            
    print("All processing complete!")


if __name__ == "__main__":
    main()
