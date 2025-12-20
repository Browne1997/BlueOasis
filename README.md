# Technial Task Overview
## Environmental Sound Classification Pipeline
A lightweight, edge‑deployable ML system for ESC‑50 audio classification

This repository implements a complete end‑to‑end machine learning pipeline for environmental sound classification using the ESC‑50 dataset. The focus of the project is to design a compact, efficient, and edge‑friendly audio model capable of running on low‑power devices while maintaining strong classification performance.

The pipeline includes:
- Audio preprocessing
- MFCC feature extraction
- Stratified train/test splitting
- A lightweight CNN architecture
- Training & tuning strategy
- Example training script
- Edge deployment considerations

## Task Setup Instructions
1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```
2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. ESC-50 data already downloaded 
```Code
data/
 ├── raw/
│     ├── audio/
 │    └── meta/esc50.csv
 └── processed/
```

5. Run preprocessing + feature extraction
```bash
python src/preprocessing.py
python src/feature_extraction.py
```
This generates MFCC feature files in:
```Code
data/processed/features/
```
6. (Optional) Test the model with dummy + real MFCCs
```bash
python src/test/train_dummy.py
```

## Full ML Pipeline Overview
```text
┌──────────────────────────────────────────────────────────────────────┐
│                          END‑TO‑END ML PIPELINE                      │
└──────────────────────────────────────────────────────────────────────┘

                          Raw Audio Files (.wav)
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         1. Preprocessing Stage                       │
└──────────────────────────────────────────────────────────────────────┘
   • Load audio (librosa)
   • Resample to 44.1 kHz
   • Convert to mono
   • Trim silence
   • Normalise amplitude
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     2. Feature Extraction (MFCCs)                    │
└──────────────────────────────────────────────────────────────────────┘
   • Compute MFCCs (13 coefficients)
   • Shape: (n_mfcc, time_frames)
   • Save as .npy files in data/processed/features/
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     3. Load Features & Metadata                      │
└──────────────────────────────────────────────────────────────────────┘
   • load_features.py:
       - Load MFCC .npy files
       - Load esc50.csv labels
       - Build X (features) and y (labels)
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     4. Train/Test Split (Stratified)                 │
└──────────────────────────────────────────────────────────────────────┘
   • sklearn.train_test_split()
   • stratify=y to preserve class balance
   • Avoids:
       - data leakage
       - temporal bias
       - class imbalance issues
       - overfitting to recording conditions
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         5. Model Architecture                        │
└──────────────────────────────────────────────────────────────────────┘
   • AudioCNN (lightweight 2‑layer CNN)
   • Input: (batch, 1, 13, time_frames)
   • Conv → ReLU → Pool → Conv → ReLU → Pool → Dropout → FC → Output
   • Designed for edge deployment (small, fast, quantisable)
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         6. Training & Tuning                         │
└──────────────────────────────────────────────────────────────────────┘
   • Loss: CrossEntropy
   • Optimiser: Adam
   • Regularisation: dropout, early stopping
   • Data augmentation (noise, pitch shift, stretch)
   • Hyperparameter tuning (LR, filters, dropout)
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           7. Evaluation                              │
└──────────────────────────────────────────────────────────────────────┘
   • Accuracy
   • Confusion matrix
   • Per‑class F1
   • Validate generalisation across 50 ESC‑50 classes
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        8. Edge Deployment Prep                       │
└──────────────────────────────────────────────────────────────────────┘
   • Quantisation (8‑bit)
   • Pruning
   • Export to ONNX
   • Deploy via TensorRT / ONNX Runtime Mobile

```

---

# Technical Assesment - detailed reasoning/answers 
## Section 1 and 4: Loading & Exploring
This explains how the codes in both sections work/satisfy the requirements of the assesment.

### Preprocessing (raw .wav → cleaned audio)
The preprocessing stage prepares the raw ESC‑50 audio files for feature extraction. Although ESC‑50 is a clean and well‑curated dataset, this step ensures consistency across all clips and removes minor artefacts that could affect downstream MFCC extraction.

The script (preprocessing.py) performs the following operations on every audio file:
1. Load audio at a fixed sample rate
  - All clips are loaded at 44.1 kHz
  - Ensures consistent frequency resolution
  - Avoids mismatched sampling rates during feature extraction

2. Convert to mono
  - ESC‑50 clips are already mono, but this step guarantees consistency
  - Prevents shape mismatches in downstream processing

3. Trim leading and trailing silence
  - Uses librosa.effects.trim()
  - Removes silent padding that varies between recordings
  - Ensures MFCCs reflect the actual sound event, not silence

4. Normalise amplitude
  - Scales audio to the range [-1, 1]
  - Prevents loud clips from dominating the model
  - Reduces variance caused by different recording gain levels

5. Saves processed audio 
  - data/processed/audio

#### Identifying Quality Issues & Artefacts
Although ESC‑50 is high quality, I checked for:
1. Duration inconsistencies
  - All clips are ~5 seconds long
  - Verified by sampling random files
  - No duration anomalies detected

2. Silent padding
  - Some clips contain small amounts of silence at the start or end
  - Trimming removes this and ensures MFCCs represent the event itself

3. Amplitude variation
  - Some recordings are noticeably louder or quieter
  - Normalisation ensures consistent dynamic range

4. Background noise
  - ESC‑50 includes natural environmental noise
  - This is expected and not removed, as it is part of the class signal

5. Missing or corrupt files
  - No missing or unreadable files were found during preprocessing


### feature extraction script
🔎 What the script produces
Spectrograms (dB) → 2D arrays of shape (freq_bins, time_frames)

MFCCs → 2D arrays of shape (n_mfcc, time_frames)

Labels → the class name (e.g. "dog", "rain", "engine") for each clip

These are saved both:

Individually (clipname_spec.npy, clipname_mfcc.npy)

As combined arrays (all_specs.npy, all_mfccs.npy, labels.npy)

🧑‍💻 How this becomes a training/test dataset
You’ll load all_mfccs.npy (or all_specs.npy) and labels.npy.

Each entry is a 2D feature matrix for one clip.

You then split them into train/test sets (e.g. 80/20 split).

Depending on the model, you might:

Flatten the 2D arrays into 1D vectors (for classical ML like SVM, RandomForest).

Keep them as 2D “images” (for CNNs).

Pad/truncate to a fixed length if needed.

### Loading and Exploring Data
This utilises visualisation.py and app.py and satisfies section 4 requirements by providing a simple we based app using the python library streamlit. 

It allows comparison of raw, processed and extracted feature data in the form of:
  - Spectograms
  - Waveforms
  

To run app 
> streamlit run src/app.py
in browser run:
http://localhost:8501

---

## Section 2: Data Splitting Strategy
### Load features script overview
Purpose is to reduce RAM required to create merge features file and perform data splitting strategy.
✅ What this does
1. Reads metadata (esc50.csv) to know which clips exist.
2. Loads the corresponding _mfcc.npy or _spec.npy file for each clip generated by feature_extraction.py
3. Builds arrays X (features) and y (labels) by using the metadata in esc50.csv to map them to the feature arrays.
  - X -> list/array of feature matrices
  - Y -> list/array of class labels
4. Splits into train/test sets with stratification.
  - This is done by the train_test_split() from the python pkg sklearn.model_selection and ensures:
    - balanced classes 
    - no leakage
    - no bias 
What its actually doing:
You provide it with X and Y arrays its returns X and Y train and test arrays whereby:
- test_size=0.2 splits train and test 80/20
- random_state=42 shuffles dataset to prevent ordering bias and ensure they are representative. 42 choosen for reproducibility and widely used, any integer would work.
- stratify=y ensure class proportions in train and test match original dataset i.e. no over- or under- representation in either split
- the above utility ensures no sample appears in both sets and no lable or feature leaks across the two sets.

### Explanation to explain the following:
#### 1. Avoiding Data Leakage
What leakage means in audio:  
This happens when information from the test set “bleeds” into the training set — for example, if the same recording session appears in both sets.ESC‑50 is a benchmark dataset where each clip is an independent 5‑second recording. There are no repeated clips, no overlapping segments, and no multi‑segment recordings.

The strategy is then as follows:
 - Splitting is done after feature extraction.
 - train_test_split(..., stratify=y) is used to ensure labels are separated cleanly.
 - No clip appears in both sets.
 - No metadata or feature normalization is computed using test data.

More information: 
Leakages can occur in the following scenarios:
- poor splitting of dataset
- shared preprocessing 
- overlapping samples
Repercussion of this are the following:
- inflated model performance due to model seeing data already when trained and then being tested on it.
- poor generalisation - doesn't learn off features its been supervised to learn off 
- contamination of accuracy, CM, loss curves etc,. 
- overfitting to recording conditions with the audio rather than actual sound class (similiar to poor generalisation)

#### 2. Avoiding Temporal Bias
What temporal bias means:  
In time‑series datasets, you must ensure training data occurs before test data. ESC‑50 is not a time‑series dataset.
Clips are independent and not ordered in time. Therefore:
- Random stratified split is appropriate.
- No temporal ordering exists to preserve.

Repercussion of temporal bias and what I would do if present in a dataset:
Concept:
Occurs if the order of the data in time influences the model in a misleading way because if you randomly shuffle and split this data you can train data that comes later than the data you test with.
Reprecussions:
- inflated model performance due to indirectly seeing future data 
- poor generalisation to new data as it has learned pattern based on the future data portion if the dataset
- temporal drift in audio occurs due to enviornmental changes, new noise, seasons etc,. and if models mixed time periods incorrectly the relationships between the features can be learnt incorrectly.
- overfitting if acoustic features in the training audio comes from a time window completely different from the test. Therefore you could fail to recognise a class in a future deployment.

What I would have done given my dataset did not have this issue would be:
- use chronological splitting 
- group recordings by session/device/location 
- utilised the following utilites in sklearn (TimeSeriesSplit, GroupShuffleSplit and GroupKFold) to prevent temporal overlap
- Disable shuffling (unlike what I done with random_state)

3. Avoiding Class Imbalance Issues
ESC‑50 dataset is balanced with 50 classes, 40 clips per class and 2000 total clips.

Approach used:
- stratify=y is used in train_test_split, which ensures:
  - Each class appears in train and test in the same proportion.
  - No class is over‑ or under‑represented.

Repercussions of class imbalance:
- Majority class dominance (FP)
- Minority class misclassified (FN)
- Skewed loss function as model optimising for the wrong objective 
- Metrics are misleading (Precision, recall, F1 score etc,.)
- Overfitting for majority class patterns 

Approaches to reduce class imbalance:
- augmentation (pitch shift, add audio artifacts)
- synthetic samples 
- unsampled majority class (adjust model architecture size accordingly)
- class weighted loss function 
- collect more data

4. Avoiding Overfitting to Recording Conditions / Equipment
ESC‑50 recordings come from many sources e.g. microphones, environments, and noise levels.

Approach used:
- Random stratified split ensures:
  - Each class’s variability (different microphones, environments) is represented in both train and test.
  - You do not group by recording session (ESC‑50 doesn’t provide session IDs anyway).
Therefore -> split distributes recording conditions across train/test, reducing overfitting risk.
- Used MFCCs feature (domain-invariant) i.e. compress spectral info and reduce sensitivity to equiptment frequency artifacts

Repercussions:
- Similiar outcomes as above issues in terms of model performances as if class representation over recording conditions/equiptment is uneven it increases potential of model learning incorrect audio features over the target features. Therefore, when using the model on new devices or environments the model under performs.

Further improvements beyond representative classes over different instruments:
- Collate contextual metadata (device ID, location, seesion ID any other relevant deployment information)
- Reduce influence of equiptment by normalising volume, amplitudes etc,. 
- Perform cross-device validation


## Section 3: Model Architecture Selection 
### Model Architecture Section 
Model Architecture
For this project, I selected a lightweight Convolutional Neural Network (CNN) operating on MFCC features. This architecture is intentionally small and efficient, making it suitable for edge deployment on devices such as Raspberry Pi, Jetson Nano, or ARM‑based embedded systems.

Why MFCCs?
MFCCs provide a compact representation of audio (13 × time frames), dramatically reducing input size compared to full spectrograms. This lowers computational cost and memory usage while preserving the key frequency characteristics needed for classification.

Why a Small CNN?
A shallow CNN is ideal for edge compute because:
  - It captures local time–frequency patterns in MFCCs
  - It has a very small parameter count
  - It runs efficiently on low‑power hardware
  - It is easy to quantise and prune after training
This makes it a strong starting point for real‑time acoustic classification.

### CNN Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────┐
│                        AudioCNN Model                        │
└──────────────────────────────────────────────────────────────┘

Input: MFCC tensor (batch, 1, 13, T)
                │
                ▼
        ┌────────────────┐
        │  Conv2D (1→16) │ 3×3, padding=1
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │   ReLU         │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │ MaxPool2D 2×2  │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │ Conv2D (16→32) │ 3×3, padding=1
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │     ReLU       │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │ MaxPool2D 2×2  │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │   Dropout 0.3  │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │   Flatten      │
        └────────────────┘
                │
                ▼
        ┌──────────────────────────────┐
        │ Fully Connected (→ 64 units) │
        └──────────────────────────────┘
                │
                ▼
        ┌──────────────────────────────┐
        │ Fully Connected (→ classes)  │
        └──────────────────────────────┘
                │
                ▼
              Output
```

### Training and Tuning Strategy 
This section explains how I would train, validate, and tune the lightweight CNN model designed for MFCC‑based acoustic classification. The goal is to achieve strong generalisation while keeping the model small and efficient enough for edge deployment.

#### Training Approach
1. Input Features
The model is trained on MFCC features extracted from each audio clip:
  - 13 MFCC coefficients
  - ~100–200 time frames (depending on hop length)
  - Input shape to the model -> (batch_size, 1, 13, time_frames)

MFCCs are chosen because they are compact, robust, and computationally inexpensive — ideal for edge devices.

2. Training Loop
The model uses a standard supervised learning pipeline:
 - Loss function: Cross‑Entropy Loss
 - Optimiser: Adam
 - Initial learning rate: 1e‑3
 - Batch size: 32
 - Epochs: 20–30 (with early stopping)

This setup provides a good balance between stability and speed, especially for small CNNs.

3. Regularisation
To prevent overfitting, especially given the small model size, I would apply:
- Dropout (0.3) — already included in the architecture
- Early stopping — stop training when validation loss stops improving
- Data augmentation, such as:
  - background noise injection
  - pitch shifting
  - time stretching
  - random gain changes

These augmentations help the model generalise to different recording conditions and environments.

#### Hyperparameter Tuning Strategy
Although the model is intentionally lightweight, tuning still plays an important role. I would use a small, targeted grid search or Bayesian optimisation over:

Model hyperparameters
  - Number of convolutional filters (e.g., 16/32 → 32/64)
  - Kernel sizes (3×3 vs 5×5)
  - Dropout rate (0.2–0.5)
  - Size of the fully connected layer (32–128 units)

Training hyperparameters
  - Learning rate (1e‑2 → 1e‑4)
  - Batch size (16, 32, 64)
  - Weight decay (0–1e‑4)

Data augmentation parameters
  - Noise level
  - Pitch shift range
  - Time stretch factor
Because the model is small, tuning is fast and inexpensive.

### Computational Requirements
#### Model Training 
The model is small enough to train on:
  - A standard laptop CPU
  - Any modern GPU (optional)
  - Cloud compute (if desired)

Training time is typically minutes, not hours.

#### Model Deployment
The architecture is designed for edge compute:
  - <1 MB model size after quantisation
  - <5 ms inference on ARM CPUs
Compatible with:
  - ONNX Runtime Mobile
  - TensorRT (convert ONNX to TFRT)
  - PyTorch Mobile
Making it suitable for embedded audio applications such as:
  - IoT sensors
  - Environmental monitoring nodes 
Deployed on microcrontrollers, jetson nanos or Raspberry Pi's.

### Dummy Training with Real MFCC 
Demonstrating chosen model architecture matches the feature representation. To satisfy this, I created a small script (train_dummy.py) that:
- Loads a real MFCC from the processed dataset
- Prints its shape
- Passes it through the CNN
- Runs a single optimisation step
- Also runs a dummy batch example for completeness
This confirms that the model accepts MFCC inputs shaped (batch, 1, 13, time_frames) and that the forward/backward passes work correctly.

Expected output 
```text
Real MFCC shape: torch.Size([1, 1, 13, 173])
Output shape (real MFCC): torch.Size([1, 50])
Loss (real MFCC): 3.91
Dummy training script completed successfully.
```
---





