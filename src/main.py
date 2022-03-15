
import glob
import os
import pickle
import time

import librosa
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

# directory paths
dir_a_directory = "D:\\Large Files\\Youtube-DLG\\"
dir_b_directory = "D:\\Music\\Assorted Weeaboo Music\\"
output_directory = "D:\\"

# hyperparameters and options
duration_tolerance_seconds = 0.5
max_chroma_len = 2**12
output_all = False  # if False, performs greedy matching; if True, outputs all similarity scores

# =============================================================================
# PART 0: GET FILE PATHS AND NAMES
# =============================================================================

# get filepath lists
dir_a_filepaths = glob.glob(dir_a_directory + "*")
dir_b_filepaths = glob.glob(dir_b_directory + "*")

# get filename lists
dir_a_filenames = [os.path.splitext(os.path.split(filepath)[1])[0] for filepath in dir_a_filepaths]
dir_b_filenames = [os.path.splitext(os.path.split(filepath)[1])[0] for filepath in dir_b_filepaths]

print(f"[INFO] Matching {len(dir_a_filepaths)} files in directory A with {len(dir_b_filepaths)} files in directory B")

# =============================================================================
# PART 1: GET DURATIONS AND CHROMAGRAMS
# =============================================================================

print(f"[INFO] Loading and processing audio files - this will take some time")
# get chromagrams and durations for files in dir a
dir_a_chromas = []
dir_a_durations = []
for filepath in dir_a_filepaths:
    y, sr = librosa.load(filepath, mono=True, sr=22050)
    dir_a_chromas.append(librosa.feature.chroma_cens(y=y, sr=sr))
    dir_a_durations.append(librosa.get_duration(y=y, sr=sr))
dir_a_durations_np = np.array(dir_a_durations)
# get chromagrams and durations for files in dir b
dir_b_chromas = []
dir_b_durations = []
for filepath in dir_b_filepaths:
    y, sr = librosa.load(filepath, mono=True, sr=22050)
    dir_b_chromas.append(librosa.feature.chroma_cens(y=y, sr=sr))
    dir_b_durations.append(librosa.get_duration(y=y, sr=sr))
dir_b_durations_np = np.array(dir_b_durations)
# save to file
with open(output_directory + "dir_a_chromas.pickle", "wb") as writefile:
    pickle.dump(dir_a_chromas, writefile)
with open(output_directory + "dir_b_chromas.pickle", "wb") as writefile:
    pickle.dump(dir_b_chromas, writefile)
np.save(output_directory + "dir_a_durations_np.npy", dir_a_durations_np)
np.save(output_directory + "dir_b_durations_np.npy", dir_b_durations_np)

# print("[INFO] Loading previously-computed chromagrams and durations")
# # load chromagrams and durations from file
# with open(output_directory + "dir_a_chromas.pickle", "rb") as readfile:
#     dir_a_chromas = pickle.load(readfile)
# with open(output_directory + "dir_b_chromas.pickle", "rb") as readfile:
#     dir_b_chromas = pickle.load(readfile)
# dir_a_durations_np = np.load(output_directory + "dir_a_durations_np.npy")
# dir_b_durations_np = np.load(output_directory + "dir_b_durations_np.npy")

# =============================================================================
# PART 2: MATCH FILES BASED ON DURATION
# =============================================================================

print("[INFO] Matching based on duration")
duration_diffs_np = np.abs(dir_a_durations_np.reshape((-1, 1)) - dir_b_durations_np.reshape((1, -1)))
duration_matched_indices = list(zip(*np.where(duration_diffs_np <= duration_tolerance_seconds)))

# =============================================================================
# PART 3: COMPUTE SIMILARITY USING DYNAMIC TIME WARPING
# =============================================================================

print("[INFO] Computing similarity using dynamic time warping - this will take some time")
# do dynamic time warp for each pair of duration-matched files
dtw_costs_np = np.full(shape=(len(dir_a_filepaths), len(dir_b_filepaths)), fill_value=np.inf, dtype=float)
start_time = time.time()
for idx_i, (idx_a, idx_b) in enumerate(duration_matched_indices):
    chroma_a = dir_a_chromas[idx_a]
    chroma_b = dir_b_chromas[idx_b]
    chroma_a_subseq_lbound = max(0, int((chroma_a.shape[1] - max_chroma_len) / 2))
    chroma_a_subseq_ubound = chroma_a_subseq_lbound + max_chroma_len
    chroma_b_subseq_lbound = max(0, int((chroma_b.shape[1] - max_chroma_len) / 2))
    chroma_b_subseq_ubound = chroma_b_subseq_lbound + max_chroma_len
    cost_matrix, warping_path = librosa.sequence.dtw(
        chroma_a[:, chroma_a_subseq_lbound:chroma_a_subseq_ubound],
        chroma_b[:, chroma_b_subseq_lbound:chroma_b_subseq_ubound],
        subseq=True,
        backtrack=True)
    dtw_costs_np[idx_a, idx_b] = np.min(cost_matrix[-1, :] / warping_path.shape[0])
    minutes_remaining = ((time.time() - start_time) * (len(duration_matched_indices) - idx_i - 1) / (idx_i + 1)) / 60
    # if (idx_i + 1) % 40 == 0:
    #     print(f"[INFO] Estimated remaining time: {minutes_remaining}")
np.save(output_directory + "dtw_costs_np.npy", dtw_costs_np)

# # load dynamic time warp similarities from file
# dtw_costs_np = np.load(output_directory + "dtw_costs_np.npy")

# =============================================================================
# PART 4: GREEDILY MATCH FILES
# =============================================================================

print("[INFO] Performing greedy matching of files")
duration_matched_indices_np = np.array(duration_matched_indices)
duration_matched_costs_np = dtw_costs_np[duration_matched_indices_np[:, 0], duration_matched_indices_np[:, 1]]
duration_matched_indices_sorted_np = duration_matched_indices_np[np.argsort(duration_matched_costs_np)]
matched_dir_a_indices = []
matched_dir_b_indices = []
for idx_a, idx_b in duration_matched_indices_sorted_np:
    if (idx_a not in matched_dir_a_indices) and (idx_b not in matched_dir_b_indices):
        matched_dir_a_indices.append(idx_a)
        matched_dir_b_indices.append(idx_b)

# =============================================================================
# PART 4: SAVE TO CSV
# =============================================================================

print("[INFO] Saving results to CSV")

# output matches to csv
matched_dir_a_filepaths = []
matched_dir_b_filepaths = []
matched_dir_a_filenames = []
matched_dir_b_filenames = []
matched_diffs = []
for idx_a, idx_b in zip(matched_dir_a_indices, matched_dir_b_indices):
    matched_dir_a_filepaths.append(dir_a_filepaths[idx_a])
    matched_dir_b_filepaths.append(dir_b_filepaths[idx_b])
    matched_dir_a_filenames.append(dir_a_filenames[idx_a])
    matched_dir_b_filenames.append(dir_b_filenames[idx_b])
    matched_diffs.append(dtw_costs_np[idx_a, idx_b])
output_df = pd.DataFrame({
    "matched_dir_a_filepaths": matched_dir_a_filepaths,
    "matched_dir_b_filepaths": matched_dir_b_filepaths,
    "matched_dir_a_filenames": matched_dir_a_filenames,
    "matched_dir_b_filenames": matched_dir_b_filenames,
    "matched_diffs": matched_diffs})
output_df = output_df.sort_values(["matched_diffs"])
output_df.to_csv(output_directory + "matches.csv", index=False)
