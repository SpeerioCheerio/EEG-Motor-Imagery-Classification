import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




#getting the data from the project folder
npz_path = "C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/X_segmented.npz"
data = np.load(npz_path)
X_segmented = data["X_segmented"]  #(n_trials, 64, 320) should be at least
y_segmented = data["y_segmented"]




print("Data loaded.")
print(f"Original X_segmented shape: {X_segmented.shape}")




#Step 2: We are going to have to downsample the Data from 160 Hz to 80 Hz
# also take every second time point instead.
print("Downsampling data from 160 Hz to 80 Hz...")
X_down = X_segmented[:, :, ::2]  # after this the new shape should be (n_trials, 64, 160)
print(f"Downsampled X_segmented shape: {X_down.shape}")




# its so bad now I have to convert down to float 32
X_down = X_down.astype(np.float32)




# flattening the data for PCA
# Compressing into a 2D array -> should be -> (n_samples, n_features)
n_trials, n_channels, n_timepoints = X_down.shape
X_flat = X_down.reshape(n_trials, n_channels * n_timepoints)
print(f"Flattened data shape for PCA: {X_flat.shape}")




# just apply PCA now
print("Fitting PCA on the flattened downsampled data...")
pca = PCA(n_components=None)
pca.fit(X_flat)




# Calculate the "Cumulative Explained Variance" hahahahahaha
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)




# using chat gpt to plot everything
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(cumulative_explained_variance) + 1),
    cumulative_explained_variance,
    marker='o',
    label="Cumulative Explained Variance"
)
threshold = 0.90
plt.axhline(y=threshold, color='r', linestyle='--', label=f"{int(threshold*100)}% Variance Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components (Downsampled to 80 Hz)")
plt.legend()
plt.grid(True)
plt.show()




# -Finding out the number of Components that are going to be requried to explain 90% of variance
num_components_90 = np.searchsorted(cumulative_explained_variance, threshold) + 1
print(f"\nNumber of components needed to reach {int(threshold*100)}% variance: {num_components_90}")
