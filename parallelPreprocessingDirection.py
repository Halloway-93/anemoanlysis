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
        
        for trial in range(param_exp["N_trials"]):
            print(f"Trial {trial}, session {session}, sub {sub}")
            
            # Here you would normally have the trial processing code
            # Since you mentioned skipping this part, I'll include a placeholder
            # This is where all the processing for each trial would go
            
            # ... [Trial processing code goes here - you'll copy your existing code] ...
            
            # For now, just a placeholder to show structure
            if firstTrial:
                firstTrial = False
                
        # Close all PDF files
        nanOnsetpdf.close()
        nanOverallpdf.close()
        nanSequencepdf.close()
        pdf.close()
        plt.close("all")
        
        # Save results to HDF files
        if paramsSub:  # Only save if we have data
            pd.DataFrame(paramsSub).to_hdf(h5_file, "data")
            pd.DataFrame(paramsRaw).to_hdf(h5_rawfile, "data")
            pd.DataFrame(qualityCtrl).to_hdf(h5_qcfile, "data")
            
            # Verify we can read the file
            abc = pd.read_hdf(h5_file, "data")
            print(f"Successfully processed {sub}, session {session}")
            return f"Success: {sub}, session {session}"
        else:
            return f"No data processed for {sub}, session {session}"
            
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

