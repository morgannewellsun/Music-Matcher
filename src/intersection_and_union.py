
import glob
import os
import shutil

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

# directory paths
dir_a_directory = "FILL\\ME\\IN\\"
dir_b_directory = "FILL\\ME\\IN\\"
output_directory = "FILL\\ME\\IN\\"

# hyperparameters and options
copy_files = True  # if False, moves the files; if True, copies them

# =============================================================================
# INTERSECTION AND UNION
# =============================================================================

# load matches (after they've been checked by the user)
try:
    matches_df = pd.read_csv(output_directory + "matches_modified.csv")
    matches_df = matches_df[~matches_df.isnull().any(axis=1)]
except FileNotFoundError as error:
    print("[ERROR] matches_modified.csv was not found. "
          "Please copy matches.csv to matches_modified.csv, "
          "manually delete rows with bad matches,"
          "and then rerun this script.")
    raise error

# create output directories
try:
    os.mkdir(output_directory + "dir_a_unmatched")
    os.mkdir(output_directory + "dir_b_unmatched")
    os.mkdir(output_directory + "dir_a_matched")
    os.mkdir(output_directory + "dir_b_matched")
except FileExistsError as error:
    print("[ERROR] Output directories already exist. "
          "Please manually clean them up and rerun this script.")
    raise error

# get filepath lists
dir_a_filepaths = list(glob.glob(dir_a_directory + "*"))
dir_b_filepaths = list(glob.glob(dir_b_directory + "*"))

# copy/move matched files
for dir_a_filepath in matches_df["matched_dir_a_filepaths"]:
    if copy_files:
        shutil.copy2(dir_a_filepath, output_directory + "dir_a_matched")
    else:
        shutil.move(dir_a_filepath, output_directory + "dir_a_matched")
    dir_a_filepaths.remove(dir_a_filepath)
for dir_b_filepath in matches_df["matched_dir_b_filepaths"]:
    if copy_files:
        shutil.copy2(dir_b_filepath, output_directory + "dir_b_matched")
    else:
        shutil.move(dir_b_filepath, output_directory + "dir_b_matched")
    dir_b_filepaths.remove(dir_b_filepath)

# copy/move unmatched files
for dir_a_filepath in dir_a_filepaths:
    if copy_files:
        shutil.copy2(dir_a_filepath, output_directory + "dir_a_unmatched")
    else:
        shutil.move(dir_a_filepath, output_directory + "dir_a_unmatched")
for dir_b_filepath in dir_b_filepaths:
    if copy_files:
        shutil.copy2(dir_b_filepath, output_directory + "dir_b_unmatched")
    else:
        shutil.move(dir_b_filepath, output_directory + "dir_b_unmatched")



