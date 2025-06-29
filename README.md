# EEG Motor Imagery Classification: ML Project 1

## Overview

This experiment systematically explored **EEG motor imagery classification pipelines** to prepare for a future seizure detection project. Using the **EEG Motor Movement/Imagery Dataset (109 subjects, 64 channels, 160 Hz, 1-2 min trials)**, the goal was to evaluate **dimensionality reduction (PCA, CSP)** and **classifiers (SVM, CNN)** for efficient, accurate EEG classification.

## Dataset

- **EEG Motor Movement/Imagery Dataset** (PhysioNet)
- **64-channel scalp EEG**, sampled at 160 Hz
- Labels: resting, left/right hand movement or imagery, bilateral hand/foot
- Each trial ~125 seconds
- Files stored in EDF format, separated by handedness (`R05`, `R06`)

## Preprocessing Pipeline

1. **Loading and inspection** using `mne` to ensure data integrity.
2. **Filtering:** Bandpass 1-40 Hz (mu, beta rhythms relevant to motor imagery).
3. **Downsampling:** 160 Hz -> 80 Hz to reduce computation while retaining relevant signals.
4. **Z-score normalization** per channel.
5. **Segmentation:** 2-second windows (160 timepoints) with 50% overlap for data augmentation.
6. Saved preprocessed data in `.npz` for efficient reuse.

Result:  
- Reduced raw long trials into **27,032 structured segments (64 channels x 160 timepoints)** for ML pipelines.

## Dimensionality Reduction

### PCA
- Flattened data to `(n_samples, 10240)` for PCA.
- ~2639 components required for 90% variance, indicating **extreme high-dimensionality** in EEG data.
- PCA deemed inefficient due to computational burden and lack of clear structure alignment with EEG.

### CSP
- Applied **Common Spatial Patterns (CSP, 8 components)** with Ledoit-Wolf regularization.
- Paired with `StandardScaler` and an SVM classifier (`rbf` kernel).
- 5-fold cross-validation:
  - Accuracy: **~58.9%**, slightly above chance.
- CSP identified motor-relevant channels (C3, Cz, C4 equivalents) but **lacked temporal modeling**.

## CNN Approach

### Initial Attempts
- A 2D CNN performed at **~50% (chance)** due to misaligned architecture ignoring temporal structure.

### Refined 1D CNN
- Built a **1D CNN** to align temporal structure:
  - Stacked `Conv1D` layers (16 -> 32 -> 64 filters, kernel size 5).
  - Batch Normalization after convolutions.
  - `GlobalAveragePooling1D` to reduce overfitting while retaining signal structure.
  - Dropout (0.5) in dense layers.
  - `Adam` optimizer, `sparse_categorical_crossentropy` loss.

### Performance
- Achieved **~75.3% test accuracy**.
- Validation accuracy stabilized around 74-75%.
- Training accuracy climbed up to 94%, indicating learning of discriminative patterns.
- Loss remained moderately high, indicating room for further tuning.

## Key Learnings

- Raw EEG data is **computationally heavy**; aggressive but mindful preprocessing is essential.
- PCA is impractical for EEG unless paired with further feature selection.
- CSP is limited for EEG classification when paired with SVMs due to lack of temporal modeling.
- **CNNs, particularly 1D CNNs**, effectively capture **spatial-temporal dependencies**, significantly outperforming traditional methods.
- EEG data remains **high-dimensional post-cleaning**, necessitating advanced methods (attention, domain adaptation, residual architectures) for scalability.

## Next Steps

- Integrate **attention mechanisms and LSTM/GRU layers** for capturing temporal dependencies.
- Explore **domain adaptation** for cross-dataset generalizability.
- Experiment with **ResNets and attention-based CNNs** for feature refinement.
- Apply these methods to **seizure detection pipelines** with real clinical EEG datasets.

## Dependencies

- Python (`numpy`, `scikit-learn`, `tensorflow`, `keras`, `mne`, `matplotlib`, `pandas`)

## Repository Structure

```
/EEG_Motor_Imagery_Classification
    ├── data/
    │   └── EEG_Data/         # Raw EDF files
    │   └── X_segmented.npz   # Preprocessed data
    ├── preprocessing.py      # Preprocessing pipeline
    ├── pca_analysis.py       # PCA dimensionality reduction
    ├── csp_svm_pipeline.py   # CSP + SVM pipeline
    ├── cnn_1d_training.py    # Refined 1D CNN pipeline
    ├── README.md             # This file
```