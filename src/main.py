
import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# =============================================================================
# SETUP
# =============================================================================

# directory paths
dir_a_directory = "D:\\Large Files\\Youtube-DLG\\"
dir_b_directory = "D:\\Music\\Assorted Weeaboo Music\\"
output_directory = "D:\\"

# get filepath lists
dir_a_filepaths = glob.glob(dir_a_directory + "*")
dir_b_filepaths = glob.glob(dir_b_directory + "*")

# dir_a_filepaths = ["D:\\Large Files\\Youtube-DLG\\Thoughts of a Slayer - KILLING THE KING - by Michal Nitecki _ Epic Hybrid Battle Music.opus"]
# dir_b_filepaths = ["D:\\Music\\Assorted Weeaboo Music\\Michal Nitecki - KILLING THE KING.mp3", "D:\\Music\\Assorted Weeaboo Music\\MEMODEMO - Digital Comet.mp3"]

dir_a_filepaths = ["D:\\Large Files\\Youtube-DLG\\Raudi - XI - Blue Zenith.opus"]
dir_b_filepaths = ["D:\\Music\\Assorted Weeaboo Music\\XI - Blue Zenith.mp3", "D:\\Music\\Assorted Weeaboo Music\\Vonikk - Typhoon.mp3"]

# get filename lists
dir_a_filenames = [os.path.splitext(os.path.split(filepath)[1])[0] for filepath in dir_a_filepaths]
dir_b_filenames = [os.path.splitext(os.path.split(filepath)[1])[0] for filepath in dir_b_filepaths]

# =============================================================================
# PART 1:
# =============================================================================

# construct features and compute durations for first part of all files - this will take time
dir_a_features = []
for filepath in dir_a_filepaths:
    y, sr = librosa.load(filepath, mono=True, sr=22050, duration=30)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    dir_a_features.append(chroma_cens)
dir_b_features = []
for filepath in dir_b_filepaths:
    y, sr = librosa.load(filepath, mono=True, sr=22050, duration=30)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    dir_b_features.append(chroma_cens)

# construct numpy arrays for features
# dir_a_features_np = np.array(dir_a_features)
# dir_b_features_np = np.array(dir_b_features)

D, wp = librosa.sequence.dtw(dir_a_features[0], dir_b_features[0], subseq=True)
plt.subplot(2, 1, 1)
plt.title('Database excerpt')
plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(D[-1, :] / wp.shape[0])
plt.xlim([0, dir_a_features[0].shape[1]])
plt.ylim([0, 2])
plt.title('Matching cost function')
plt.tight_layout()
plt.show()

D, wp = librosa.sequence.dtw(dir_a_features[0], dir_b_features[1], subseq=True)
plt.subplot(2, 1, 1)
plt.title('Database excerpt')
plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(D[-1, :] / wp.shape[0])
plt.xlim([0, dir_a_features[0].shape[1]])
plt.ylim([0, 2])
plt.title('Matching cost function')
plt.tight_layout()
plt.show()

raise Exception

# store feature numpy arrays so we don't have to wait for all that to run again...
np.save(output_directory + "dir_a_features_np", dir_a_features_np)
np.save(output_directory + "dir_b_features_np", dir_b_features_np)

# =============================================================================
# PART 2
# =============================================================================

# # do the same for durations
# dir_a_durations = []
# for filepath in dir_a_filepaths:
#     y, sr = librosa.load(filepath, mono=True, sr=22050)
#     dir_a_durations.append(librosa.get_duration(y=y, sr=sr))
# dir_b_durations = []
# for filepath in dir_b_filepaths:
#     y, sr = librosa.load(filepath, mono=True, sr=22050)
#     dir_b_durations.append(librosa.get_duration(y=y, sr=sr))
# dir_a_durations_np = np.array(dir_a_durations)
# dir_b_durations_np = np.array(dir_b_durations)
# np.save(output_directory + "dir_a_durations_np", dir_a_durations_np)
# np.save(output_directory + "dir_b_durations_np", dir_b_durations_np)

# =============================================================================
# PART 2
# =============================================================================

# load features from file
dir_a_features_np = np.load(output_directory + "dir_a_features_np.npy")
dir_b_features_np = np.load(output_directory + "dir_b_features_np.npy")
dir_a_durations_np = np.load(output_directory + "dir_a_durations_np.npy")
dir_b_durations_np = np.load(output_directory + "dir_b_durations_np.npy")

# compute squared mean differences in chromagrams
try:  # vectorized, high memory consumption
    feature_shape = dir_a_features_np.shape[1:3]
    mean_squared_diffs_np = (
            dir_a_features_np.reshape((-1, 1, *feature_shape))
            - dir_b_features_np.reshape((1, -1, *feature_shape)))
    mean_squared_diffs_np = np.square(mean_squared_diffs_np)
    mean_squared_diffs_np = np.mean(mean_squared_diffs_np, axis=(2, 3))
except np.core._exceptions._ArrayMemoryError:  # not vectorized, less memory
    mean_squared_diffs_np = np.zeros(shape=(len(dir_a_features_np), len(dir_b_features_np)))
    for idx_a, feature_a_np in enumerate(dir_a_features_np):
        for idx_b, feature_b_np in enumerate(dir_b_features_np):
            mean_squared_diffs_np[idx_a, idx_b] = np.mean(np.square(feature_a_np - feature_b_np))

# group by duration (to the nearest second) and find matches
matched_dir_a_indices = []
matched_dir_b_indices = []
is_direct_match = []
dir_a_durations_df = pd.DataFrame({"duration": dir_a_durations_np.astype(int)})
dir_b_durations_df = pd.DataFrame({"duration": dir_b_durations_np.astype(int)})
for duration, dir_a_duration_df in dir_a_durations_df.groupby("duration", sort=False):
    duration_indices_a = dir_a_duration_df.index
    duration_indices_b = dir_b_durations_df[dir_b_durations_df["duration"] == duration].index
    duration_mean_squared_diffs_np = mean_squared_diffs_np[duration_indices_a, :][:, duration_indices_b]
    matched_dir_a_inner_indices, matched_dir_b_inner_indices = linear_sum_assignment(duration_mean_squared_diffs_np)
    matched_dir_a_indices.extend(list(duration_indices_a[matched_dir_a_inner_indices]))
    matched_dir_b_indices.extend(list(duration_indices_b[matched_dir_b_inner_indices]))
    is_direct_match.extend([len(duration_mean_squared_diffs_np.flatten()) == 1] * len(matched_dir_a_inner_indices))

# norm = plt.Normalize()
# # norm.autoscale(mean_squared_diffs_np)
# for a, b in zip(matched_dir_a_indices, matched_dir_b_indices):
#     print(dir_a_filenames[a])
#     print(dir_b_filenames[b])
#     print(mean_squared_diffs_np[a, b])
#     if mean_squared_diffs_np[a, b] > 10000:
#         plt.imshow(dir_a_features_np[a] - dir_b_features_np[b], norm=norm, interpolation=None)
#         plt.show()
#     print("")

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
    matched_diffs.append(mean_squared_diffs_np[idx_a, idx_b])
output_df = pd.DataFrame({
    "matched_dir_a_filepaths": matched_dir_a_filepaths,
    "matched_dir_b_filepaths": matched_dir_b_filepaths,
    "matched_dir_a_filenames": matched_dir_a_filenames,
    "matched_dir_b_filenames": matched_dir_b_filenames,
    "matched_diffs": matched_diffs,
    "is_direct_match": is_direct_match})
output_df = output_df.sort_values(["is_direct_match", "matched_diffs"])
output_df.to_csv(output_directory + "matches.csv", index=False)
























# # construct chromagrams and compute durations for first part of all files - this will take time
# dir_a_features = []
# dir_a_durations = []
# for filepath in dir_a_filepaths:
#     y, sr = librosa.load(filepath, sr=22050, duration=30)
#     spectro = librosa.feature.melspectrogram(y=y, sr=sr)
#     spectro = np.max(spectro.reshape((-1, 19)), axis=1).reshape((128, -1))  # max pool time axis with kernel size 19
#     dir_a_features.append(spectro)
#     y, sr = librosa.load(filepath, sr=22050)
#     dir_a_durations.append(librosa.get_duration(y=y, sr=sr))
# dir_b_features = []
# dir_b_durations = []
# for filepath in dir_b_filepaths:
#     y, sr = librosa.load(filepath, sr=22050, duration=30)
#     spectro = librosa.feature.melspectrogram(y=y, sr=sr)
#     spectro = np.max(spectro.reshape((-1, 19)), axis=1).reshape((128, -1))  # max pool time axis with kernel size 19
#     dir_b_features.append(spectro)
#     y, sr = librosa.load(filepath, sr=22050)
#     dir_b_durations.append(librosa.get_duration(y=y, sr=sr))
#
# # construct numpy arrays for chromagrams
# dir_a_features_np = np.array(dir_a_features)
# dir_b_features_np = np.array(dir_b_features)
# dir_a_durations_np = np.array(dir_a_durations)
# dir_b_durations_np = np.array(dir_b_durations)
#
# # store chromagram numpy arrays so we don't have to wait for all that to run again...
# np.save(output_directory + "dir_a_features_np", dir_a_features_np)
# np.save(output_directory + "dir_b_features_np", dir_b_features_np)
# np.save(output_directory + "dir_a_durations_np", dir_a_durations_np)
# np.save(output_directory + "dir_b_durations_np", dir_b_durations_np)