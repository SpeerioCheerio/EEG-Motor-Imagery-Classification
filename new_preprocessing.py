import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd


#prev storage stuff
print("=== EEG Preprocessing Script Start ===\n")
print("Loading .edf files from directory...")
data_folder = "C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/EEG_Data"
edf_files = [f for f in os.listdir(data_folder) if f.endswith('.edf')]
print(f"Total .edf files found: {len(edf_files)}")
left_hand_files = sorted([f for f in edf_files if "R05" in f])
right_hand_files = sorted([f for f in edf_files if "R06" in f])
print(f"Left-hand files (R05): {len(left_hand_files)}")
print(f"Right-hand files (R06): {len(right_hand_files)}\n")
X = []
y = []
max_time_points = 0




print("Loading each .edf file and extracting EEG data:")
for file in left_hand_files + right_hand_files:
    file_path = os.path.join(data_folder, file)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data, _ = raw[:, :]
    max_time_points = max(max_time_points, data.shape[1])
    X.append(data)
    y.append(0 if "R05" in file else 1)
    print(f"  Loaded {file}: shape = {data.shape}")


X_padded = [np.pad(trial, ((0, 0), (0, max_time_points - trial.shape[1])), mode='constant')
            for trial in X]
X = np.array(X_padded)
y = np.array(y)






# This is where the original Downsampling occurs, and also the old filtering -
X_filtered = []
new_sfreq = 80      # the target sampling frequency.
original_sfreq = 160  #the og frequency




for idx, trial in enumerate(X):
    info = mne.create_info(ch_names=64, sfreq=original_sfreq, ch_types="eeg")
    raw_trial = mne.io.RawArray(trial, info)
    raw_trial.filter(1, 40, fir_design='firwin', verbose=False)
    raw_trial.resample(new_sfreq, npad="auto")
    filtered_data = raw_trial.get_data()
    X_filtered.append(filtered_data)


X_filtered = np.array(X_filtered)


scaler = StandardScaler()
X_normalized = np.array([scaler.fit_transform(trial) for trial in X_filtered])




# same window dimensions
window_size = 160
stride = 80
X_segmented = []
y_segmented = []




for i in range(X_normalized.shape[0]):
    trial_segments = 0
    for j in range(0, X_normalized.shape[2] - window_size + 1, stride):
        X_segmented.append(X_normalized[i, :, j:j + window_size])
        y_segmented.append(y[i])
        trial_segments += 1


X_segmented = np.array(X_segmented)
y_segmented = np.array(y_segmented)


# the print statements to see what is going on
unique, counts = np.unique(y_segmented, return_counts=True)
print("\nLabel distribution after segmentation:")
for label, count in zip(unique, counts):
    label_str = "Left (R05)" if label == 0 else "Right (R06)"
    print(f"  {label_str}: {count} segments")




original_trials = len(X)
segmented_trials = X_segmented.shape[0]
print(f"\nOriginal number of trials: {original_trials}")
print(f"Total segments after segmentation: {segmented_trials}")
print(f"Average segments per trial: {segmented_trials / original_trials:.2f}\n")
#comparisons to check




#save everything
np.save("C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/X_segmented.npy", X_segmented)
np.save("C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/y_segmented.npy", y_segmented)
print("Saved preprocessed data as NumPy files.")


print("\nPreprocessing Complete")
