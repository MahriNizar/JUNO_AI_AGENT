import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob

def list_missed_rmu(folder_path):

    # 1) Read data & filter out BECID == -1, and ensure BECID and RMU_2nd are integer types
    df = pd.read_csv("test_files/pmt_bec_rmu_map.csv")
    df = df[df["BECID"] != -1].copy()
    df["BECID"] = df["BECID"].astype(int)
    # df["RMU_2nd"] = df["RMU_2nd"].astype(int)

    # 2) Define a discrete colormap
    unique_RMU_2nd = sorted(df["RMU_2nd"].unique())
    cmap = plt.colormaps["jet"].resampled(len(unique_RMU_2nd))
    norm = mcolors.BoundaryNorm(boundaries=np.arange(0.5, len(unique_RMU_2nd) + 1.5), ncolors=len(unique_RMU_2nd))

    # 3) Specify the folder path
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 4) Initialize data containers
    missing_becids_all = []
    missing_RMU_2nds = []
    incomplete_rmu_files = []  # store filenames where some RMU_2nd are missing
    present_RMU_2nds = []      # store RMU_2nd that are present (not missing)
    count = 0

    # 5) Iterate over CSV files
    for file in csv_files:
        count += 1
        # if count > 3:
        #     break
        # print(count)
        df_output = pd.read_csv(file, header=None, names=["PMT_ID"])
        df_output = df_output.drop_duplicates(subset=["PMT_ID"])
        
        merged_df = pd.merge(df_output, df, on="PMT_ID", how="inner")
        
        # RMU_2nd appearing in this file
        file_RMU_2nds = set(merged_df["RMU_2nd"].unique())
        present_RMU_2nds.extend(file_RMU_2nds)
        
        # Compute missing BECID and RMU_2nd
        all_RMU_2nds = set(df["RMU_2nd"].unique())
        missing_RMU_2nds_set = all_RMU_2nds - file_RMU_2nds
        
        # Record files where some RMU_2nd are missing
        if missing_RMU_2nds_set:
            incomplete_rmu_files.append((file, missing_RMU_2nds_set))
        
        missing_RMU_2nds.extend(missing_RMU_2nds_set)

    # 6) Output filenames with partially missing RMU_2nd and list those missing RMU_2nd
    f_list = []
    print("Files where some RMU_2nds are missing:")
    for f, missing_rmus in incomplete_rmu_files:
        print(f"{f}: Missing RMU_2nds {sorted(missing_rmus)}")
        f_list.append(f)

    file_names = [os.path.basename(f) for f in f_list]
    print(f'count is {count}, missed files={len(file_names)}')
    print(f'f_list={f_list}')

    # 7) Plot histogram of missing RMU_2nd
    plt.figure(figsize=(10, 6))
    plt.hist(missing_RMU_2nds, bins=15, color="black", alpha=0.7)
    plt.title("Histogram of Missing RMU_2nds[whole RMU lost]")
    plt.xlabel("RMU_2nd")
    plt.ylabel("Count")
    plt.savefig("plt_missingRMU.png")
    plt.close()
    return "Plot saved at plt_missingRMU.png"
