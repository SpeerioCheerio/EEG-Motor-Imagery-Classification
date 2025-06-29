import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score




data_path = "C:/Users/Alexander Speer/Desktop/Columbia Spring 2025/AML/Project1/"
X_segmented = np.load(data_path + "X_segmented.npy")
y_segmented = np.load(data_path + "y_segmented.npy")
n_segments, n_channels, n_timepoints = X_segmented.shape
print("Loaded segmented data:")
print(f"  X_segmented shape: {X_segmented.shape} (segments, channels, timepoints)")
print(f"  y_segmented shape: {y_segmented.shape}\n")




#Set up a CSP classifier pipeline
csp = CSP(n_components=8, reg='ledoit_wolf', log=True, norm_trace=False)
# 'ledoit_wolf' makes covariance estimation more stable since I believe it helps with noisy EEG data
clf = SVC(kernel='rbf', C=1.0)
# and now we're using an SVM with an some sort of special kernel kernel.


pipeline = Pipeline([('CSP', csp),
                     ('scaler', StandardScaler()),
                     ('SVM', clf)])
print("Pipeline configured.\n")




# cross validation with the CSP pipeline
print("Performing 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_segmented, y_segmented, cv=cv, scoring='accuracy')
print("Cross-validation accuracies for each fold:")
for i, score in enumerate(scores, start=1):
    print(f"  Fold {i}: {score:.4f}")
print(f"Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")


# Step 4: Fiting CSP on the entire dataset for patten analysis
print("Fitting CSP on the entire dataset for pattern analysis...")
csp.fit(X_segmented, y_segmented)
patterns = csp.patterns_  #shape that we should expect outof this -> (n_components, n_channels)
print("CSP fitting complete.\n")




#Visualizing the CSP Patterns and then all of the top channels that we want


print("Visualizing CSP patterns and identifying top contributing channels...")
channel_names = [f"Ch{i}" for i in range(1, n_channels + 1)]
num_components = patterns.shape[0]




for comp in range(num_components):
    plt.figure(figsize=(10, 4))
    # plotting the abs weight?? GPT used
    plt.bar(channel_names, np.abs(patterns[comp, :]))
    plt.xlabel("Channel")
    plt.ylabel("Absolute Pattern Weight")
    plt.title(f"CSP Component {comp+1} Pattern Weights")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
   
    # find the top 3 channels per componetn and display them
    sorted_idx = np.argsort(np.abs(patterns[comp, :]))[::-1]
    top_channels = [channel_names[i] for i in sorted_idx[:3]]
    print(f"  Top channels for CSP component {comp+1}: {', '.join(top_channels)}\n")




#the final summary
print("=== CSP Analysis Summary ===")
print(f"1. Loaded segmented data with {n_segments} segments, each having {n_channels} channels and {n_timepoints} timepoints.")
print("2. Configured CSP with 8 components using Ledoit-Wolf regularization and log-variance features.")
print("3. Performed 5-fold cross-validation with the CSP -> StandardScaler -> SVM pipeline:")
for i, score in enumerate(scores, start=1):
    print(f"   Fold {i}: {score:.4f}")
print(f"   Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print("4. Fitted CSP on the entire dataset and visualized each component's spatial pattern.")
print("   For each CSP component, the top 3 channels (by absolute weight) are listed above.")
print("   These channels likely indicate the most relevant regions for motor imagery (e.g., C3, Cz, C4 if correctly positioned).")
print("\n=== CSP Analysis Complete ===")
