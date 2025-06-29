import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data_path = "C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/"
data_file = data_path + "X_segmented_80Hz.npz"  # Use the correct file
data = np.load(data_file)
X_segmented = data["X_segmented"]
y_segmented = data["y_segmented"]  
print("Loaded preprocessed segmented data:")
print(f"  X_segmented shape: {X_segmented.shape} (segments, channels, timepoints)")
print(f"  y_segmented shape: {y_segmented.shape}\n")


#var(x) per electrode
print("Computing variance per electrode across all segments and timepoints...")
#variance for each channel over time.
variance_per_channel = np.var(X_segmented, axis=(0, 2))
print("Variance per channel computed.")


# creating the default channel names so that we have some sort of understanding what channel is where
n_channels = X_segmented.shape[1]
channel_names = [f"Ch{i}" for i in range(1, n_channels + 1)]
channel_variance = dict(zip(channel_names, variance_per_channel))
sorted_channels = sorted(channel_variance.items(), key=lambda x: x[1], reverse=True)


#variance chatGPT plot
plt.figure(figsize=(10, 5))
plt.bar(channel_names, variance_per_channel)
plt.xlabel("Channel")
plt.ylabel("Variance")
plt.title("Variance per Channel")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# flattening teh data for PCA
print("\nFlattening segmented data for PCA...")
n_segments, n_channels, n_timepoints = X_segmented.shape
X_flat = X_segmented.reshape(n_segments, n_channels * n_timepoints)
print(f"Flattened data shape: {X_flat.shape}")


# PCA with no limit
print("\nApplying PCA (all components will be computed)...")
pca = PCA(n_components=None)  # all components will  be computed
X_pca = pca.fit_transform(X_flat)
print("PCA transformation complete.")


explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)


print("\nExplained variance ratios (first 10 components):")
for i, ratio in enumerate(explained_variance_ratio[:10]):
    print(f"  PC {i+1}: {ratio:.4f}")


# finding out the number of component sthat we will need to hti 90 % variance
num_components_90 = np.searchsorted(cumulative_explained_variance, 0.90) + 1
print(f"\nNumber of components needed to reach 90% variance: {num_components_90}")


#using chatgpt to make these plots.
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(cumulative_explained_variance) + 1),
    cumulative_explained_variance,
    marker='o',
    linestyle='--',
    label="Cumulative Explained Variance"
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance for All Principal Components")
plt.axhline(y=0.90, color='r', linestyle='--', label="90% Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Summary statstics
print("\n=== PCA Feature Extraction & Variance Analysis Summary ===")
print(f"1. Loaded segmented data with {n_segments} segments, each with {n_channels} channels and {n_timepoints} timepoints.")
print("2. Computed variance per electrode; top 10 channels by variance:")
for ch, var in sorted_channels[:10]:
    print(f"   {ch}: {var:.4f}")
print(f"3. Data was flattened to shape {X_flat.shape} for PCA.")
print(f"4. PCA computed all {X_flat.shape[1]} components. (The full decomposition is available if needed.)")
print("5. Explained variance ratios for the first 10 components are printed above.")
print(f"6. {num_components_90} components are needed to reach 90% cumulative variance (as shown in the plot).")
print("7. This analysis identifies which channels contribute most to the overall variance,")
print("   which may correspond to relevant motor activity (e.g., C3, Cz, C4) for your EEG motor imagery task.")
print("\n=== PCA Analysis Complete ===")
