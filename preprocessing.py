import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import h5py




# file directory
data_folder = "C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/EEG_Data"




# find left or right hand trials
edf_files = [f for f in os.listdir(data_folder) if f.endswith('.edf')]




# seperating by left or right hand
left_hand_files = sorted([f for f in edf_files if "R05" in f])
right_hand_files = sorted([f for f in edf_files if "R06" in f])




# Initialize lists for storing data and labels
X = []  # X is going to be for the EEG signals
y = []  # labells for which of the hands it is, with 0 = left hand, 1 = right hand
max_time_points = 0  # longest trial in case we need to concat




# loaing each of tose edf files
for file in left_hand_files + right_hand_files:
    file_path = os.path.join(data_folder, file)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data, _ = raw[:, :]  # the shape is given in (channels, time)
   
    # Update maximum time points for uniformity
    max_time_points = max(max_time_points, data.shape[1])
   
    X.append(data)
    # appplying all of the labells; 0 if "R05" (left-hand), 1 if "R06" (right-hand) <- How it is labelled in the data.
    y.append(0 if "R05" in file else 1)




# each trial needs to be the same length in order to operate on it.
X_padded = [np.pad(trial, ((0, 0), (0, max_time_points - trial.shape[1])), mode='constant') for trial in X]
X = np.array(X_padded)  # the shape is given in (n_samples, 64, max_time_points)
y = np.array(y)
print(f"Loaded {len(X)} EEG trials, each with {X.shape[1]} channels and {X.shape[2]} time points (padded to max length).")




#############################################################################
# REMOVING ALL OF THE NOISE IN THE DATA




X_filtered = []
for trial in X:
    #     RawArray qmade using the the trial data
    info = mne.create_info(ch_names=64, sfreq=160, ch_types="eeg")
    raw_trial = mne.io.RawArray(trial, info)
    raw_trial.filter(1, 40, fir_design='firwin', verbose=False)
    X_filtered.append(raw_trial.get_data())




X_filtered = np.array(X_filtered)
print("Applied bandpass filtering (1-40 Hz) to all trials.")




###################################################################################
# NORMALIZING AND SEGMENTING OUR DATA




# Z-score per trial
scaler = StandardScaler()
X_normalized = np.array([scaler.fit_transform(trial) for trial in X_filtered])




# Segment the data into some overlapping windows, preserves a lot of the temporal informaton,  should help with generalization
window_size = 320  # 2 seconds * 160 Hz
stride = 160       # 50% overlap
X_segmented = []
y_segmented = []


# making the window span across the time dimension


for i in range(len(X_normalized)):
    for j in range(0, X_normalized.shape[2] - window_size + 1, stride):
        X_segmented.append(X_normalized[i, :, j:j + window_size])
        y_segmented.append(y[i])




X_segmented = np.array(X_segmented)
y_segmented = np.array(y_segmented)
print(f"Segmented data into {X_segmented.shape[0]} trials of {window_size} time points each.")
print(f"Final dataset shape: {X_segmented.shape}")
print(f"Labels shape: {y_segmented.shape}")




############################################################
#saving data to a npz file for later py scripts
np.savez_compressed(
    "C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/X_segmented.npz",
    X_segmented=X_segmented,
    y_segmented=y_segmented
)
print("Saved preprocessed data in compressed NPZ format.")
