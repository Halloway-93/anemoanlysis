import os
import scipy.io
import pandas as pd
import h5py
import numpy as np

# Main directory containing subject subdirectories
main_dir = "/Users/mango/oueld.h/contextuaLearning/motionDirectionCue"

# Loop through all subdirectories that match the pattern "sub-*"
for subdir in [d for d in os.listdir(main_dir) if d.startswith("sub-")]:
    subject_path = os.path.join(main_dir, subdir)

    # Extract subject number from directory name (e.g., "sub-001" -> "001")
    sub_num = subdir.split("-")[1]

    # Check if it's a directory
    if os.path.isdir(subject_path):
        # Find all .mat files in the subject directory
        for file in os.listdir(subject_path):
            if file.endswith(".mat") and file.startswith("list_CP_"):
                mat_file_path = os.path.join(subject_path, file)

                # Extract condition from filename (list_CP_s1c9.mat)
                filename_without_ext = file.split(".mat")[0]
                cond = filename_without_ext.split("c")[-1]

                try:
                    # Try to load with scipy.io first (older MATLAB formats)
                    try:
                        mat_data = scipy.io.loadmat(mat_file_path)
                        df = pd.DataFrame(
                            mat_data["trialType"],
                            columns=["firstSegmentMotion", "secondSegmentMotion"],
                        )
                    except NotImplementedError:
                        # For v7.3 MATLAB files, use h5py
                        with h5py.File(mat_file_path, "r") as f:
                            # H5py loads MATLAB variables differently
                            # Convert to numpy array and transpose (MATLAB stores arrays column-wise)
                            firstSeg = np.array(f.get("listFirstSeg"))[0]
                            secondSeg = np.array(f.get("listSecondSeg"))[0]

                            # Convert to numpy array and transpose (MATLAB stores arrays column-wise)
                            # We need to get and transpose the dataset
                            data = np.array([firstSeg, secondSeg]).T
                            df = pd.DataFrame(
                                data,
                                columns=["firstSegmentMotion", "secondSegmentMotion"],
                            )

                    # Create output filename: sub-XXX_c9.tsv
                    output_file = f"{subdir}_c{cond}.tsv"
                    output_path = os.path.join(subject_path, output_file)

                    # Save as TSV file
                    df.to_csv(output_path, sep="\t", index=False)

                    print(f"Processed {file} -> {output_file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
